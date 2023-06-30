from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
import chex

from smolGPT.utils import canonicalize_dtype, safe_int32_increment, hvp


class ScaleBySketchySGDState(NamedTuple):
    """State for the SketchySGD algorithm."""
    count: chex.Array
    s: optax.Updates
    u: optax.Updates


def scale_by_sketchy_sgd(
    rank: int = 1,
    rho: float = 0.01,
    update_freq: int = 1,
    precond_dtype: Optional[chex.ArrayDType] = None,
) -> optax.GradientTransformationExtraArgs:
    """Rescale updates according to the SketchySDG algorithm.

    References:
        [Frangella et al, 2023](SketchySGD: Reliable Stochastic Optimization via Randomized Curvature Estimates)

    Args:
        rank: sketch rank.
        rho: regularization.
        update_freq: how frequently we should update the Hessian approximation.
        precond_dtype: Optional `dtype` to be used for the preconditioner; if
            `None` then the dtype is inferred from `params` and `updates`.

    Returns:
        A `GradientTransformation` object.
    """

    precond_dtype = canonicalize_dtype(precond_dtype)

    def init_fn(params):
        s = jax.tree_util.tree_map(  # Eigenvalues
            lambda t: jnp.empty_like(t, dtype=precond_dtype), params)
        u = jax.tree_util.tree_map(  # Eigenvectors
            lambda t: jnp.empty_like(t, dtype=precond_dtype), params)
        return ScaleBySketchySGDState(count=jnp.zeros([], jnp.int32), s=s, u=u)

    def update_fn(updates, state, params = None, inputs = None, targets = None,
                  loss_fn = None, loss_scale = None, policy = None):
        if params is None:
            NO_PARAMS_MSG = (
                'You are using a transformation that requires the current value of '
                'parameters, but you are not passing `params` when calling `update`.')
            raise ValueError(NO_PARAMS_MSG)

        if inputs is None or targets is None or loss_fn is None:
            NO_INPUTS_TARGETS_LOSS_MSG = (
                'You are using a transformation that requires the current loss function, '
                'inputs, and targets, but you are not passing `loss_fn`, `inputs`, and '
                '`targets` when calling `update`.')
            raise ValueError(NO_INPUTS_TARGETS_LOSS_MSG)

        def sum_hvps(grads, x):
            input_, target, rangerfinder, mini_step = x
            curr_grads, loss = hvp(loss_fn, has_aux=True)(params, input_, target)
            welford_update = lambda acc, new: acc + (new - acc) / (mini_step + 1)
            new_grads = jax.tree_map(welford_update, grads, curr_grads)
            return new_grads, loss

        init = jax.tree_map(jnp.zeros_like, params)
        xs = (inputs, targets, state.rangefinder, jnp.arange(inputs.shape[0]))
        grads, losses = jax.lax.scan(sum_hvps, init, xs)
        losses = policy.cast_to_output(losses)
        loss = jnp.mean(losses)

        grads = policy.cast_to_param(grads)
        grads = jax.lax.pmean(grads, axis_name="batch")
        grads = loss_scale.unscale(grads)
        grads_norm = optax.global_norm(grads)

        scale = loss_fn(params)
        if loss_scale is not None:

        count_inc = safe_int32_increment(state.count)
        updates = jax.tree_map(lambda x: x * 0, updates)
        return updates, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
