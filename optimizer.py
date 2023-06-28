from typing import NamedTuple, Optional

import optax
import chex

from utils import canonicalize_dtype


class ScaleBySketchySGDState(NamedTuple):
    """State for the SketchySGD algorithm."""
    eigenvalues: optax.Updates
    eigenvectors: optax.Updates


def scale_by_sketchy_sgd(
    hessian_dtype: Optional[chex.ArrayDType] = None,
) -> optax.GradientTransformation:
    """Rescale updates according to the SketchySDG algorithm.

    References:
        [Frangella et al, 2023](SketchySGD: Reliable Stochastic Optimization via Randomized Curvature Estimates)

    Returns:
        A `GradientTransformation` object.
    """

    hessian_dtype = canonicalize_dtype(hessian_dtype)

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            NO_PARAMS_MSG = (
                'You are using a transformation that requires the current value of '
                'parameters, but you are not passing `params` when calling `update`.')
            raise ValueError(NO_PARAMS_MSG)
        updates = ...
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)
