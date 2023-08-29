from typing import NamedTuple
from pathlib import Path
from functools import partial
from itertools import islice
from hashlib import sha1
from sys import maxsize

import jax
import jax.numpy as jnp
import jax.flatten_util     # suppress warning
import jmp
import optax
import numpy as np
from tqdm import tqdm
import fire
import wandb

from smolGPT.model import gpt2
from smolGPT.utils import (
    load_encoder_hparams_and_params,
    replicate,
    unreplicate,
    is_penultimate,
    canonicalize_dtype,
)


class DataLoader:
    def __init__(self, filename, context_length, gradient_accumulation, batch_size, seed = 42) -> None:
        self.data = np.memmap(filename, dtype=np.uint16, mode="r")
        self.context_length = context_length
        device_count = jax.local_device_count()
        if gradient_accumulation % device_count != 0:
            raise ValueError(f"{gradient_accumulation % device_count = }")
        self.index_shape = (device_count, gradient_accumulation // device_count, batch_size)
        self.shape = (device_count, gradient_accumulation // device_count, batch_size, self.context_length)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        ix = self.rng.integers(len(self.data) - self.context_length, size=self.index_shape)
        x = np.empty(self.shape, dtype=np.uint16)
        y = np.empty(self.shape, dtype=np.uint16)
        for ij, index in np.ndenumerate(ix):
            x[ij] = self.data[index:index+self.context_length]
            y[ij] = self.data[index+1:index+1+self.context_length]
        return x, y

    def reset(self):
        self.rng = np.random.default_rng(self.seed)


def path_to_key(path, data = None):
    path_bytes = "".join(str(p) for p in path).encode()
    path_hash = int(sha1(path_bytes).hexdigest(), 16) % maxsize
    key = jax.random.PRNGKey(path_hash)

    if data is not None:
        key = jax.random.fold_in(key, data)

    return key


def randomize_params(params, n_layer):
    # see https://github.com/karpathy/nanoGPT/blob/4eb7a96b077998f28b57938c2f1e511b0d8cab7c/model.py#L140-L145
    def randomize(path, leaf):
        key_leaf = path_to_key(path)

        if path[-1].key == "b":                 # bias
            return jnp.zeros_like(leaf)
        elif path[-1].key == "g":               # layer norm
            return jnp.ones_like(leaf)
        elif path[-1].key == "w":               # mlp/attention
            noise = jax.random.normal(key_leaf, leaf.shape, leaf.dtype)
            if path[-2].key == "c_proj":
                std = 0.02/jnp.sqrt(2*n_layer)
            else:
                std = 0.02
            return std * noise
        elif path[-1].key in {"wpe", "wte"}:    # embedding
            noise = jax.random.normal(key_leaf, leaf.shape, leaf.dtype)
            return 0.02 * noise
        else:
            raise ValueError(f"Unknown path {path}")

    # not doing gradient tying because that doesn't seem to help finetuning
    params = jax.tree_util.tree_map_with_path(randomize, params)
    return params


def inject_uv(params):
    def inject(path, leaf):
        # FIXME: we want to train the biases as well (actually no?)
        # FIXME: split c_attn into w_q, w_k and w_v (but ignore the multi-head thing)
        if path[-1].key in {"c_attn", "c_proj"} and path[-2].key == "attn":
            return {**leaf, "u": None, "v": None}
        else:
            return leaf

    params = jax.tree_util.tree_map_with_path(inject, params, is_leaf=is_penultimate)
    return params


def init_lora(params, lora_rank):
    def init(path, leaf):
        if isinstance(leaf, np.ndarray) or isinstance(leaf, jax.Array):
            return None

        key_leaf = path_to_key(path)

        null = {k: None for k in leaf}
        if path[-1].key in {"c_attn", "c_proj"} and path[-2].key == "attn":
            w = leaf["w"]
            u = 1./lora_rank * jax.random.normal(key_leaf, (w.shape[0], lora_rank), w.dtype)
            v = jnp.zeros((lora_rank, w.shape[1]), w.dtype)
            return {**null, "u": u, "v": v}
        else:
            return null

    params = jax.tree_util.tree_map_with_path(init, params, is_leaf=is_penultimate)
    return params


class SketchySGDState(NamedTuple):
    """State for the SketchySGD algorithm."""
    rho: jax.Array
    s: jax.Array
    u: jax.Array


def create_sketchy_state(key, params, rank, rho, precond_dtype):
    precond_dtype = canonicalize_dtype(precond_dtype)

    # eigenvalues
    s = jax.tree_util.tree_map(lambda x: jnp.zeros(rank, dtype=precond_dtype or x.dtype), params)

    def random_orthogonal(path, x):
        # NOTE: use jax.flatten_util.ravel_pytree if we want to precondition jointly
        path_bytes = "".join(str(p) for p in path).encode()
        path_hash = int(sha1(path_bytes).hexdigest(), 16) % maxsize
        key_leaf = jax.random.fold_in(key, path_hash)

        p = np.prod(x.shape, dtype=int)
        rand = jax.random.normal(key_leaf, (p, rank), dtype=jnp.float32)
        q, _ = jnp.linalg.qr(rand)  # jnp.linalg.qr does not support bfloat16
        q = q.astype(precond_dtype or x.dtype)
        q = jnp.reshape(q.T, (rank, *x.shape))
        return q

    # eigenvectors
    u = jax.tree_util.tree_map_with_path(random_orthogonal, params)

    rho = jnp.array(rho, precond_dtype)
    sketchy_state = SketchySGDState(rho, s, u)
    return sketchy_state


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(7, 8), donate_argnums=(0, 1))
def estimate_step(sketchy_state, loss_scale, params, frozen, es_inputs, es_targets, seed, n_head, policy):

    def loss_fn(p, input_, target):
        merged = jax.tree_map(lambda a, b: a if a is not None else b,
                              frozen, p, is_leaf=lambda x: x is None)
        logits = gpt2(input_, **merged, n_head=n_head)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, target)
        loss = jnp.mean(losses)
        return loss_scale.scale(loss), loss

    def avg_grads_hvps(grads_hvps, x):
        grads, hvps = grads_hvps
        input_, target, mini_step = x
        grad_fn = lambda p: jax.grad(loss_fn, has_aux=True)(p, input_, target)
        hvp_fn = lambda o: jax.jvp(grad_fn, (params_compute,), (o,), has_aux=True)
        curr_grads, curr_hvps, loss = jax.lax.map(hvp_fn, omega)
        curr_grads, loss = unreplicate((curr_grads, loss))
        welford_update = lambda acc, new: acc + (new - acc) / (mini_step + 1)
        new_grads = jax.tree_map(welford_update, grads, curr_grads)
        new_hvps = jax.tree_map(welford_update, hvps, curr_hvps)
        return (new_grads, new_hvps), loss

    def random_normal(path, x):
        # NOTE: use jax.flatten_util.ravel_pytree if we want to precondition jointly
        key_leaf = path_to_key(path, data=seed)
        rand = jax.random.normal(key_leaf, x.shape, x.dtype)
        return rand

    def random_orthogonal(path, x):
        rand = random_normal(path, x)

        r, *shape = x.shape
        p = np.prod(shape, dtype=int)
        rand = jnp.reshape(rand, (p, r)).astype(jnp.float32)    # jnp.linalg.qr does not support bfloat16
        q, _ = jnp.linalg.qr(rand)
        q = q.astype(x.dtype)
        q = jnp.reshape(q.T, (r, *shape))   # represent rank with the leading axis to make vectorization easier
        return q

    omega = jax.tree_util.tree_map_with_path(random_orthogonal, sketchy_state.u)

    # alternatively, do power iteration:
    # omega = sketchy_state.u

    params_compute, omega, inputs, targets = policy.cast_to_compute((params, omega, es_inputs, es_targets))

    init = (jax.tree_map(jnp.zeros_like, params_compute),
            jax.tree_map(jnp.zeros_like, omega))
    xs = (inputs, targets, jnp.arange(inputs.shape[0]))

    # using map/scan-over-grad instead of grad-over-map/scan to reduce memory consumption
    (grads, hvps), losses = jax.lax.scan(avg_grads_hvps, init, xs)
    losses = policy.cast_to_output(losses)
    loss = jnp.mean(losses)

    grads = policy.cast_to_param(grads)
    grads = jax.lax.pmean(grads, axis_name="batch")
    grads = loss_scale.unscale(grads)
    grads_norm = optax.global_norm(grads)

    hvps = policy.cast_to_param(hvps)
    hvps = jax.lax.pmean(hvps, axis_name="batch")
    hvps = loss_scale.unscale(hvps)
    hvps_norm = optax.global_norm(hvps)

    # Nystrom approximation
    def estimate_hessian(hvp, omg):
        # TODO: instead of reshaping weight matrices into vectors, can we precondition each row/column separately?
        rank, *shape = hvp.shape
        p = np.prod(shape, dtype=int)
        hvp = jnp.reshape(hvp, (rank, p))
        omg = jnp.reshape(omg, (rank, p))

        def svd(b):
            # use host callback when CUDA runs out of memory for large p
            #
            u, sigma, _ = jnp.linalg.svd(b, full_matrices=False)

            # result_shape = (jax.ShapeDtypeStruct((p, rank), b.dtype),
            #                 jax.ShapeDtypeStruct((rank,), b.dtype),
            #                 jax.ShapeDtypeStruct((rank, rank), b.dtype))
            # u, sigma, _ = jax.pure_callback(partial(np.linalg.svd, full_matrices=False), result_shape, b)

            return sigma, u

        regularizer = jnp.linalg.norm(hvp) * jnp.finfo(hvp.dtype).eps
        hvp = hvp + regularizer * omg

        quad = (omg @ hvp.T).astype(jnp.float32)    # jnp.linalg.{svd,eigh} does not support bfloat16
        quad = (quad + quad.T) / 2                  # symmetrization
        l = jnp.linalg.cholesky(quad)
        is_convex = jnp.logical_not(jnp.any(jnp.isnan(l)))

        def convex_case():
            b = jax.scipy.linalg.solve_triangular(l, hvp, lower=True).T
            sigma, u = svd(b)
            s = jnp.maximum(0, sigma**2 - regularizer).T
            return s, u

        def nonconvex_case():
            gamma, w = jnp.linalg.eigh(quad)
            shift = -jnp.min(gamma)
            gamma_sft = gamma + shift
            gamma_neg_sqrt = jnp.where(gamma_sft == 0,
                                       gamma_sft,   # Moore-Penrose pseudoinverse
                                       gamma_sft**(-0.5))
            r = w * gamma_neg_sqrt @ w.T
            b = hvp.T @ r
            sigma, u = svd(b)
            s = jnp.abs(sigma**2 - regularizer - shift).T
            return s, u

        s, u = jax.lax.cond(is_convex, convex_case, nonconvex_case)

        s = s.astype(hvp.dtype)
        u = u.astype(hvp.dtype)
        u = jnp.reshape(u.T, (rank, *shape))

        return s, u

    su = jax.tree_map(estimate_hessian, policy.cast_to_compute(hvps), omega)
    s = jax.tree_map(lambda x: x[0], su, is_leaf=lambda x: isinstance(x, tuple))
    u = jax.tree_map(lambda x: x[1], su, is_leaf=lambda x: isinstance(x, tuple))

    s_max_norm = optax.global_norm(jax.tree_map(lambda x: x[0], s))
    s_min_norm = optax.global_norm(jax.tree_map(lambda x: x[-1], s))
    u_norm = optax.global_norm(u)

    new_sketchy_state = sketchy_state._replace(s=s, u=u)

    # spectral norm of the difference between consecutive Hessian estimations
    def project(v, new_u, new_s, old_u, old_s):
        r, *shape = new_u.shape
        p = np.prod(shape, dtype=int)
        v = jnp.reshape(v, (p,))
        new_u = jnp.reshape(new_u, (r, p))
        old_u = jnp.reshape(old_u, (r, p))

        projection = new_u.T * new_s @ (new_u @ v) - old_u.T * old_s @ (old_u @ v)
        projection = jnp.reshape(projection, shape)
        return projection

    def should_continue(val):
        # https://scicomp.stackexchange.com/questions/592
        _, _, r_norm, iter = val

        cond = jnp.logical_and(
            jnp.logical_not(jnp.isnan(r_norm)),
            jnp.logical_and(r_norm > 1e-6, iter < 64)
        )

        return cond

    def power_iteration(val):
        _, eigenvector, _, iter = val

        projection = jax.tree_map(project,
                                  eigenvector,
                                  new_sketchy_state.u,
                                  new_sketchy_state.s,
                                  sketchy_state.u,
                                  sketchy_state.s)

        def eigenvalue_fn(p, d):
            p, _ = jax.flatten_util.ravel_pytree(p)
            d, _ = jax.flatten_util.ravel_pytree(d)
            eigenvalue = jnp.sign(jnp.dot(p, d)) * jnp.linalg.norm(p)
            return eigenvalue

        eigenvalue = jax.tree_map(eigenvalue_fn, projection, eigenvector)
        eigenvector = jax.tree_map(lambda x, n: x/n, projection, eigenvalue)

        residual = jax.tree_map(lambda p, e, v: jnp.linalg.norm(p - e * v),
                                projection,
                                eigenvalue,
                                eigenvector)
        r_norm = optax.global_norm(residual)

        return eigenvalue, eigenvector, r_norm, iter + 1

    eigenvalue = jax.tree_map(lambda _: 0., grads)
    eigenvector = jax.tree_util.tree_map_with_path(random_normal, grads)
    init_val = eigenvalue, eigenvector, jnp.inf, 0
    eigenvalue, _, _, _ = jax.lax.while_loop(should_continue, power_iteration, init_val)
    dist_spectral = optax.global_norm(eigenvalue)

    # loss_scale will always be the same across devices thanks to pmean
    grads_finite = jmp.all_finite((grads, hvps))
    new_loss_scale = loss_scale.adjust(grads_finite)
    new_sketchy_state = jmp.select_tree(grads_finite, new_sketchy_state, sketchy_state)

    return new_sketchy_state, new_loss_scale, loss, grads_norm, hvps_norm, s_max_norm, s_min_norm, u_norm, dist_spectral


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(7, 8, 9), donate_argnums=(0, 1, 2))
def train_step(params, opt_state, loss_scale, sketchy_state, frozen, inputs, targets, n_head, gradient_transform, policy):

    def loss_fn(p, input_, target):
        merged = jax.tree_map(lambda a, b: a if a is not None else b,
                              frozen, p, is_leaf=lambda x: x is None)
        logits = gpt2(input_, **merged, n_head=n_head)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, target)
        loss = jnp.mean(losses)
        return loss_scale.scale(loss), loss

    def avg_grads(grads, x):
        input_, target, mini_step = x
        curr_grads, loss = jax.grad(loss_fn, has_aux=True)(params_compute, input_, target)
        welford_update = lambda acc, new: acc + (new - acc) / (mini_step + 1)
        new_grads = jax.tree_map(welford_update, grads, curr_grads)
        return new_grads, loss

    params_compute, inputs, targets = policy.cast_to_compute((params, inputs, targets))

    init = jax.tree_map(jnp.zeros_like, params_compute)
    xs = (inputs, targets, jnp.arange(inputs.shape[0]))

    # using map/scan-over-grad instead of grad-over-map/scan to reduce memory consumption
    grads, losses = jax.lax.scan(avg_grads, init, xs)
    losses = policy.cast_to_output(losses)
    loss = jnp.mean(losses)

    grads = policy.cast_to_param(grads)
    grads = jax.lax.pmean(grads, axis_name="batch")
    grads = loss_scale.unscale(grads)
    grads_norm = optax.global_norm(grads)

    def precondition(grad, s, u):
        r, *shape = u.shape
        p = np.prod(shape, dtype=int)
        grad = jnp.reshape(grad, (p,))
        u = jnp.reshape(u, (r, p))

        # Precondition with the Woodbury formula, equivalent to
        #   precond = jnp.linalg.solve(u.T @ jnp.diag(s) @ u + rho * jnp.eye(grad.shape[0]), grad)
        rho = sketchy_state.rho
        u_grad = u @ grad
        precond = u.T / (s + rho) @ u_grad + (grad - u.T @ u_grad) / rho

        precond = jnp.reshape(precond, shape)
        return precond

    if sketchy_state is not None:
        preconditioned = jax.tree_util.tree_map(precondition, grads, sketchy_state.s, sketchy_state.u)
    else:
        preconditioned = grads

    precond_norm = optax.global_norm(preconditioned)

    x, _ = jax.flatten_util.ravel_pytree(preconditioned)
    y, _ = jax.flatten_util.ravel_pytree(grads)
    dist_cos = jnp.dot(x, y) / jnp.linalg.norm(x) / jnp.linalg.norm(y)
    dist_euc = jnp.linalg.norm(x - y)

    # TODO: implement automatic learning rate
    updates, new_opt_state = gradient_transform.update(preconditioned, opt_state, params_compute,
                                                       inputs=inputs, targets=targets, loss_fn=loss_fn,
                                                       loss_scale=loss_scale, policy=policy)
    new_params = optax.apply_updates(params, updates)

    # loss_scale will always be the same across devices thanks to pmean
    grads_finite = jmp.all_finite(grads)
    new_loss_scale = loss_scale.adjust(grads_finite)
    new_params, new_opt_state = jmp.select_tree(
        grads_finite,
        (new_params, new_opt_state),
        (params, opt_state))

    return new_params, new_opt_state, new_loss_scale, loss, grads_norm, precond_norm, dist_cos, dist_euc


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(4,))
def valid_step(params, frozen, va_inputs, va_targets, n_head):
    def loss_fn(args):
        va_input, va_target = args
        merged = jax.tree_map(lambda a, b: a if a is not None else b,
                              frozen, params, is_leaf=lambda x: x is None)
        logits = gpt2(va_input, **merged, n_head=n_head)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, va_target)
        loss = jnp.mean(losses)
        return loss

    losses = jax.lax.map(loss_fn, (va_inputs, va_targets))
    loss = jnp.mean(losses)
    return loss


def train(params,
          frozen,
          tr,
          es,
          va,
          n_head,
          sketchy_rank,
          sketchy_rho,
          sketchy_freq,
          learning_rate,
          max_iter,
          weight_decay,
          beta1,
          beta2,
          grad_clip,
          warmup_iters,
          lr_decay_iters,
          min_lr,
          policy,
          eval_freq):
    scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=learning_rate,
            warmup_steps=warmup_iters,
            decay_steps=lr_decay_iters,
            end_value=min_lr)

    # FIXME: may gradient clipping render SketchySGD ineffective?
    mask = jax.tree_map(lambda x: x.ndim >= 2, params)
    gradient_transform = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            # optax.scale_by_adam(beta1, beta2),
            optax.add_decayed_weights(weight_decay, mask),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0),
    )
    opt_state = gradient_transform.init(params)

    key = jax.random.PRNGKey(42)
    # The computation of (eigenvalues, eigenvectors) = (s, u) does not
    # include accumulation (e.g. moving average), so they cannot benefit
    # from being stored in high precision. As a result, we store them
    # as policy.compute_dtype.
    if sketchy_rank > 0:
        sketchy_state = create_sketchy_state(key, params, rank=sketchy_rank, rho=sketchy_rho, precond_dtype=policy.compute_dtype)
    else:
        sketchy_state = None

    if policy.compute_dtype is not jnp.float32:
        loss_scale = jmp.DynamicLossScale(jnp.array(2**16, dtype=policy.param_dtype))
    else:
        loss_scale = jmp.NoOpLossScale()

    params, frozen, opt_state, sketchy_state, loss_scale = replicate((params, frozen, opt_state, sketchy_state, loss_scale))
    device_count = jax.local_device_count()
    device_ids = jnp.arange(device_count)
    va_loss = hvps_norm = s_max_norm = s_min_norm = u_norm = dist_spectral = jnp.nan

    tr_loader = islice(tr, max_iter)
    for step, (inputs, targets) in enumerate(pbar := tqdm(tr_loader, "Training")):

        if sketchy_state is not None and step % sketchy_freq == 0:
            es_inputs, es_targets = next(es)
            step_pmap = replicate(device_count * step) + device_ids
            sketchy_state, loss_scale, loss, grads_norm, hvps_norm, s_max_norm, s_min_norm, u_norm, dist_spectral \
                    = estimate_step(sketchy_state, loss_scale, params, frozen, es_inputs, es_targets, step_pmap, n_head, policy)
            hvps_norm = float(unreplicate(hvps_norm))
            s_max_norm = float(unreplicate(s_max_norm))
            s_min_norm = float(unreplicate(s_min_norm))
            u_norm = float(unreplicate(u_norm))
            dist_spectral = float(unreplicate(dist_spectral))

        params, opt_state, loss_scale, loss, grads_norm, precond_norm, dist_cos, dist_euc \
                = train_step(params, opt_state, loss_scale, sketchy_state, frozen, inputs, targets, n_head, gradient_transform, policy)
        lr = float(scheduler(step))
        scale = int(unreplicate(loss_scale).loss_scale).bit_length() - 1
        loss = float(jnp.mean(loss))
        grads_norm = float(unreplicate(grads_norm))
        precond_norm = float(unreplicate(precond_norm))
        dist_cos = float(unreplicate(dist_cos))
        dist_euc = float(unreplicate(dist_euc))

        if step % eval_freq == 0:
            va.reset()
            va_inputs, va_targets = next(va)
            va_loss = valid_step(params, frozen, va_inputs, va_targets, n_head)
            va_loss = float(jnp.mean(va_loss))

        pbar.set_description(f"{loss = :.3}, {va_loss = :.3}, {s_max_norm = :.3}, {dist_cos = :.3}, {dist_spectral = :.3}")
        wandb.log({
            "step": step,
            "lr": lr,
            "scale": scale,
            "loss": loss,
            "va_loss": va_loss,
            "grads_norm": grads_norm,
            "hvps_norm": hvps_norm,
            "s_max_norm": s_max_norm,
            "s_min_norm": s_min_norm,
            "u_norm": u_norm,
            "precond_norm": precond_norm,
            "dist_cos": dist_cos,
            "dist_euc": dist_euc,
            "dist_spectral": dist_spectral,
        }, step)

    return unreplicate(params)


def main(model_size: str = "124M",
         models_dir: str = "models",
         data_dir: str = "data/openwebtext",
         finetune: bool = True,
         lora_rank: int = 1,
         learning_rate: float = 6e-4,
         min_lr: float = 6e-5,
         warmup_iters: int = 2000,
         lr_decay_iters: int = 600000,
         max_iter: int = 600000,
         gradient_accumulation: int = 256,
         batch_size: int = 4,
         sketchy_rank: int = 4,
         sketchy_rho: float = 0.1,
         sketchy_freq: int = 1024,
         sketchy_accumulation: int = 512,
         sketchy_batch_size: int = 2,
         grad_clip: float = 1.0,
         weight_decay: float = 1e-1,
         beta1: float = 0.9,
         beta2: float = 0.95,
         jmp_policy: str = "params=float32,compute=bfloat16,output=float32",
         eval_freq: int = 64,
         eval_accumulation: int = 64,
         eval_batch_size: int = 16):

    if not finetune and lora_rank > 0:
        raise ValueError("You cannot train a LoRA model from scratch")

    config = locals()
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    context_length = hparams["n_ctx"]
    n_head = hparams["n_head"]
    n_layer = hparams["n_layer"]

    data_path = Path(data_dir)
    tr = DataLoader(data_path/"train.bin", context_length, gradient_accumulation, batch_size)
    es = DataLoader(data_path/"train.bin", context_length, sketchy_accumulation, sketchy_batch_size)
    va = DataLoader(data_path/"val.bin", context_length, eval_accumulation, eval_batch_size)

    policy = jmp.get_policy(jmp_policy)
    params = policy.cast_to_param(params)

    if not finetune:
        params = randomize_params(params, n_layer)

    params = inject_uv(params)
    if lora_rank > 0:
        frozen, params = params, init_lora(params, lora_rank)
    else:
        frozen, params = jax.tree_map(lambda _: None, params), params

    wandb.init(project="smolGPT", config=config)
    params = train(params,
                   frozen,
                   tr,
                   es,
                   va,
                   n_head,
                   sketchy_rank,
                   sketchy_rho,
                   sketchy_freq,
                   learning_rate,
                   max_iter,
                   weight_decay,
                   beta1,
                   beta2,
                   grad_clip,
                   warmup_iters,
                   lr_decay_iters,
                   min_lr,
                   policy,
                   eval_freq)
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
