from typing import NamedTuple, Optional
from pathlib import Path
from functools import partial
from itertools import islice
from hashlib import sha1
from sys import maxsize

import jax
import jax.numpy as jnp
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
    safe_int32_increment
)


class DataLoader:
    def __init__(self, filename, context_length, gradient_accumulation, batch_size) -> None:
        self.data = np.memmap(filename, dtype=np.uint16, mode="r")
        self.context_length = context_length
        device_count = jax.local_device_count()
        if gradient_accumulation % device_count != 0:
            raise ValueError(f"{gradient_accumulation % device_count = }")
        self.index_shape = (device_count, gradient_accumulation // device_count, batch_size)
        self.shape = (device_count, gradient_accumulation // device_count, batch_size, self.context_length)
        self.rng = np.random.default_rng(42)

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


def randomize_params(params, n_layer):
    # See https://github.com/karpathy/nanoGPT/blob/4eb7a96b077998f28b57938c2f1e511b0d8cab7c/model.py#L140-L145
    def randomize(path, leaf):
        path_bytes = "".join(str(p) for p in path).encode()
        path_hash = int(sha1(path_bytes).hexdigest(), 16) % maxsize
        key_leaf = jax.random.PRNGKey(path_hash)

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
        if path[-1].key in {"c_fc", "c_proj", "c_attn"}:               # mlp/attention
            return {**leaf, "uv": None}
        else:
            return leaf

    params = jax.tree_util.tree_map_with_path(inject, params, is_leaf=is_penultimate)
    return params


def init_lora(params, lora_rank):
    def init(path, leaf):
        if isinstance(leaf, np.ndarray) or isinstance(leaf, jax.Array):
            return None

        path_bytes = "".join(str(p) for p in path).encode()
        path_hash = int(sha1(path_bytes).hexdigest(), 16) % maxsize
        key_leaf = jax.random.PRNGKey(path_hash)

        null = {k: None for k in leaf}
        if path[-1].key in {"c_attn", "c_fc", "c_proj"}:               # mlp/attention
            w = leaf["w"]
            key_u, key_v = jax.random.split(key_leaf)
            u = 0.02 * jax.random.normal(key_u, (w.shape[0], lora_rank), w.dtype)
            v = 0.02 * jax.random.normal(key_v, (lora_rank, w.shape[1]), w.dtype)
            return {**null, "uv": (u, v)}
        else:
            return null

    params = jax.tree_util.tree_map_with_path(init, params, is_leaf=is_penultimate)
    return params


class SketchySGDState(NamedTuple):
    """State for the SketchySGD algorithm."""
    rho: jax.Array
    update_freq: jax.Array
    count: jax.Array
    s: jax.Array
    u: jax.Array


def create_sketchy_state(key, params, rank = 1, rho = 0.01, update_freq = 1, precond_dtype = None):
    precond_dtype = canonicalize_dtype(precond_dtype)
    s = jax.tree_util.tree_map(  # Eigenvalues
        lambda x: jnp.empty_like(x, dtype=precond_dtype), params)
    u = jax.tree_util.tree_map(  # Eigenvectors
        lambda x: jax.random.normal(key, (rank, *x.shape), dtype=precond_dtype or x.dtype), params)
    # TODO: orthogonalize u with QR (and jax.flatten_util.ravel_pytree if we want to precondition jointly)
    rho = jnp.array(rho, precond_dtype)
    update_freq = jnp.array(update_freq, jnp.int32)
    count = jnp.zeros([], jnp.int32)
    sketchy_state = SketchySGDState(rho, update_freq, count, s, u)
    return sketchy_state


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(7, 8, 9), donate_argnums=(0, 1, 2, 3))
def train_step(params, opt_state, sketchy_state, loss_scale, frozen, inputs, targets, n_head, gradient_transform, policy):

    def loss_fn(p, input_, target):
        merged = jax.tree_map(lambda a, b: a if a is not None else b,
                              frozen, p, is_leaf=lambda x: x is None)
        logits = gpt2(input_, **merged, n_head=n_head)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, target)
        loss = jnp.mean(losses)
        return loss_scale.scale(loss), loss

    def avg_grads(grads_hvps, x):
        grads, hvps = grads_hvps
        input_, target, mini_step = x
        curr_grads, loss = jax.grad(loss_fn, has_aux=True)(params_compute, input_, target)
        welford_update = lambda acc, new: acc + (new - acc) / (mini_step + 1)
        new_grads = jax.tree_map(welford_update, grads, curr_grads)
        return (new_grads, hvps), loss

    def avg_grads_hvps(grads_hvps, x):
        grads, hvps = grads_hvps
        input_, target, mini_step = x
        grad_fn = lambda p: jax.grad(loss_fn, has_aux=True)(p, input_, target)
        hvp_fn = lambda u: jax.jvp(grad_fn, (params_compute,), (u,), has_aux=True)
        curr_grads, curr_hvps, loss = jax.lax.map(hvp_fn, sketchy_state.u)
        curr_grads, loss = unreplicate((curr_grads, loss))
        welford_update = lambda acc, new: acc + (new - acc) / (mini_step + 1)
        new_grads = jax.tree_map(welford_update, grads, curr_grads)
        new_hvps = jax.tree_map(welford_update, hvps, curr_hvps)
        return (new_grads, new_hvps), loss

    params_compute, inputs, targets = policy.cast_to_compute((params, inputs, targets))

    # use map/scan-over-grad instead of grad-over-map/scan to reduce memory consumption
    init = (jax.tree_map(jnp.zeros_like, params_compute),
            jax.tree_map(jnp.zeros_like, sketchy_state.u))
    xs = (inputs, targets, jnp.arange(inputs.shape[0]))
    (grads, hvps), losses = jax.lax.cond(
        sketchy_state.count % sketchy_state.update_freq == 0,
        lambda i, x: jax.lax.scan(avg_grads_hvps, i, x),
        lambda i, x: jax.lax.scan(avg_grads, i, x),
        init,
        xs)
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

    count_inc = safe_int32_increment(sketchy_state.count)
    new_sketchy_state = sketchy_state._replace(
        count=count_inc,
        s=policy.cast_to_compute(grads),    # placeholder
        u=policy.cast_to_compute(jax.tree_map(lambda x: x/hvps_norm, hvps)),
    )

    updates, new_opt_state = gradient_transform.update(grads, opt_state, params_compute,
                                                       inputs=inputs, targets=targets, loss_fn=loss_fn,
                                                       loss_scale=loss_scale, policy=policy)
    new_params = optax.apply_updates(params, updates)

    # loss_scale will always be the same across devices thanks to pmean
    grads_finite = jmp.all_finite(grads)
    new_loss_scale = loss_scale.adjust(grads_finite)
    new_params, new_opt_state, new_sketchy_state = jmp.select_tree(
        grads_finite,
        (new_params, new_opt_state, new_sketchy_state),
        (params, opt_state, sketchy_state))

    return new_params, new_opt_state, new_sketchy_state, new_loss_scale, loss, grads_norm, hvps_norm


def train(params,
          frozen,
          tr,
          va,
          n_head,
          sketchy_rank,
          learning_rate,
          max_iter,
          weight_decay,
          beta1,
          beta2,
          grad_clip,
          warmup_iters,
          lr_decay_iters,
          min_lr,
          policy):
    scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=learning_rate,
            warmup_steps=warmup_iters,
            decay_steps=lr_decay_iters,
            end_value=min_lr)
    mask = jax.tree_map(lambda x: x.ndim >= 2, params)
    gradient_transform = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.scale_by_adam(beta1, beta2),
            optax.add_decayed_weights(weight_decay, mask),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0),
    )
    opt_state = gradient_transform.init(params)

    key = jax.random.PRNGKey(42)
    sketchy_state = create_sketchy_state(key, params, rank=sketchy_rank, rho=0.01, update_freq=1, precond_dtype=policy.compute_dtype)

    if policy.compute_dtype is jnp.float32:
        loss_scale = jmp.NoOpLossScale()
    else:
        loss_scale = jmp.DynamicLossScale(jnp.array(2**16, dtype=policy.param_dtype))

    params, frozen, opt_state, sketchy_state, loss_scale = replicate((params, frozen, opt_state, sketchy_state, loss_scale))
    dataloader = islice(zip(tr, va), max_iter)

    for (inputs, targets), _ in (pbar := tqdm(dataloader, "Training")):
        params, opt_state, sketchy_state, loss_scale, loss, grads_norm, hvps_norm \
                = train_step(params, opt_state, sketchy_state, loss_scale, frozen, inputs, targets, n_head, gradient_transform, policy)
        _, _, _, schedule_state, _ = opt_state
        step = int(unreplicate(schedule_state.count))
        lr = float(scheduler(step))
        scale = int(unreplicate(loss_scale).loss_scale).bit_length() - 1
        loss = float(jnp.mean(loss))
        grads_norm = float(unreplicate(grads_norm))
        hvps_norm = float(unreplicate(hvps_norm))

        pbar.set_description(f"{scale = }, {lr = :.3}, {loss = :.3}, {grads_norm = :.3}, {hvps_norm = :.3}")
        wandb.log({
            "step": step,
            "lr": lr,
            "scale": scale,
            "loss": loss,
            "grads_norm": grads_norm,
            "hvps_norm": hvps_norm,
        })

    return unreplicate(params)


def main(model_size: str = "124M",
         models_dir: str = "models",
         data_dir: str = "data/openwebtext",
         finetune: bool = True,
         lora_rank: Optional[int] = None,
         sketchy_rank: int = 1,
         gradient_accumulation: int = 5 * 8,
         batch_size: int = 6,
         learning_rate: float = 6e-4,
         max_iter: int = 600000,
         weight_decay: float = 1e-1,
         beta1: float = 0.9,
         beta2: float = 0.95,
         grad_clip: float = 1.0,
         warmup_iters: int = 2000,
         lr_decay_iters: int = 600000,
         min_lr: float = 6e-5,
         jmp_policy: str = "params=float32,compute=bfloat16,output=float32"):

    if not finetune and lora_rank is not None:
        raise ValueError("You cannot train a LoRA model from scratch")

    config = locals()
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    context_length = hparams["n_ctx"]
    n_head = hparams["n_head"]
    n_layer = hparams["n_layer"]

    data_path = Path(data_dir)
    tr = DataLoader(data_path/"train.bin", context_length, gradient_accumulation, batch_size)
    va = DataLoader(data_path/"val.bin", context_length, gradient_accumulation, batch_size)

    policy = jmp.get_policy(jmp_policy)
    params = policy.cast_to_param(params)

    if not finetune:
        params = randomize_params(params, n_layer)

    params = inject_uv(params)
    if lora_rank is not None:
        frozen, params = params, init_lora(params, lora_rank)
    else:
        frozen, params = jax.tree_map(lambda _: None, params), params

    wandb.init(project="smolGPT", config=config)
    params = train(params,
                   frozen,
                   tr,
                   va,
                   n_head,
                   sketchy_rank,
                   learning_rate,
                   max_iter,
                   weight_decay,
                   beta1,
                   beta2,
                   grad_clip,
                   warmup_iters,
                   lr_decay_iters,
                   min_lr,
                   policy)
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
