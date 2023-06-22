from pathlib import Path
from functools import partial
from itertools import islice
from operator import add
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

from gpt2 import gpt2
from utils import load_encoder_hparams_and_params, replicate, unreplicate


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


def randomize(params, n_layer, policy):
    # See https://github.com/karpathy/nanoGPT/blob/4eb7a96b077998f28b57938c2f1e511b0d8cab7c/model.py#L140-L145
    def init_param(path, leaf):
        path_bytes = ".".join(str(p) for p in path).encode("utf-8")
        path_hash = int(sha1(path_bytes).hexdigest(), 16) % maxsize
        leaf_key = jax.random.PRNGKey(path_hash)

        noise = jax.random.normal(leaf_key, leaf.shape, policy.param_dtype)
        if path[-1].key in {"wpe", "wte"}:
            return noise * 0.02
        elif path[-1].key == "b":
            return jnp.zeros_like(noise)
        elif path[-1].key in {"w", "g"}:
            if path[-2].key == "c_proj":
                return noise * 0.02/jnp.sqrt(2*n_layer)
            else:
                return noise * 0.02
        else:
            raise ValueError(f"Unknown path {path}")

    params = jax.tree_util.tree_map_with_path(init_param, params)
    return params


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(5, 6, 7), donate_argnums=(0, 1, 2))
def train_step(params, opt_state, loss_scale, inputs, targets, n_head, gradient_transform, policy):

    def loss_fn(p, input_, target):
        p, input_, target = policy.cast_to_compute((p, input_, target))
        logits = gpt2(input_, **p, n_head=n_head)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, target)
        losses = policy.cast_to_output(losses)
        loss = jnp.mean(losses)
        return loss_scale.scale(loss), loss

    def sum_grads(grads, x):
        input_, target = x
        curr_grads, loss = jax.grad(loss_fn, has_aux=True)(params, input_, target)
        new_grads = jax.tree_map(add, grads, curr_grads)
        return new_grads, loss

    # use map/scan-over-grad instead of grad-over-map/scan to reduce memory consumption
    init = jax.tree_map(jnp.zeros_like, params)
    xs = (inputs, targets)
    grads, losses = jax.lax.scan(sum_grads, init, xs)
    loss = jnp.mean(losses)

    grads = loss_scale.unscale(grads)
    grads = policy.cast_to_compute(grads)
    grads = jax.lax.pmean(grads, axis_name="batch")
    grads = policy.cast_to_param(grads)

    updates, new_opt_state = gradient_transform.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # loss_scale will always be the same across devices thanks to pmean
    grads_finite = jmp.all_finite(grads)
    new_loss_scale = loss_scale.adjust(grads_finite)
    new_params, new_opt_state = jmp.select_tree(
        grads_finite,
        (new_params, new_opt_state),
        (params, opt_state))

    def agg(x, y):
        return jnp.sum(x * y)

    params, updates, grads = policy.cast_to_compute((params, updates, grads))
    pred1 = jax.tree_util.tree_reduce(add, jax.tree_map(agg, updates, grads))

    def sum_hvp(hvp, x):
        input_, target = x
        def grad_fn(p):
            grads, loss = jax.grad(loss_fn, has_aux=True)(p, input_, target)
            return loss_scale.unscale(grads), loss
        curr_grad, curr_hvp, loss = jax.jvp(grad_fn, (params,), (updates,), has_aux=True)
        new_hvp = jax.tree_map(add, hvp, curr_hvp)
        return new_hvp, loss

    hvp, _ = jax.lax.scan(sum_hvp, init, xs)
    hvp = jax.tree_map(lambda x: x/inputs.shape[0], hvp)
    hvp = loss_scale.unscale(hvp)
    hvp = policy.cast_to_compute(hvp)
    hvp = jax.lax.pmean(hvp, axis_name="batch")
    pred2 = pred1 + 0.5 * jax.tree_util.tree_reduce(add, jax.tree_map(agg, updates, hvp))

    return new_params, new_opt_state, new_loss_scale, loss, grads_finite, pred1, pred2


def train(params,
          tr,
          va,
          n_head,
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
    gradient_transform = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.scale_by_adam(beta1, beta2),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0),
    )
    opt_state = gradient_transform.init(params)
    loss_scale = jmp.DynamicLossScale(jnp.array(2. ** 15, dtype=policy.param_dtype))

    params, opt_state, loss_scale = replicate((params, opt_state, loss_scale))
    dataloader = islice(zip(tr, va), max_iter)
    last_loss = float("inf")
    last_pred1 = 0
    last_pred2 = 0

    for step, ((inputs, targets), _) in enumerate(pbar := tqdm(dataloader, "Training")):
        params, opt_state, loss_scale, loss, grads_finite, pred1, pred2 \
                = train_step(params, opt_state, loss_scale, inputs, targets, n_head, gradient_transform, policy)
        scale = float(unreplicate(loss_scale).loss_scale)
        loss = float(jnp.mean(loss))
        grads_finite = float(unreplicate(grads_finite))
        pred1 = float(unreplicate(pred1))
        pred2 = float(unreplicate(pred2))
        pbar.set_description(f"{scale = }, {loss = }, {grads_finite = }")

        delta = last_loss - loss
        wandb.log({
            "step": step,
            "scale": scale,
            "loss": loss,
            "grads_finite": grads_finite,
            "delta": delta,
            "pred1": last_pred1,
            "pred2": last_pred2,
            "err1": delta - last_pred1,
            "err2": delta - last_pred2,
        })
        last_loss, last_pred1, last_pred2 = loss, pred1, pred2

    return unreplicate(params)


def main(model_size: str = "124M",
         models_dir: str = "models",
         data_dir: str = "data/openwebtext",
         gradient_accumulation: int = 5 * 8,
         batch_size: int = 12,
         learning_rate: float = 6e-4,
         max_iter: int = 600000,
         weight_decay: float = 1e-1,
         beta1: float = 0.9,
         beta2: float = 0.95,
         grad_clip: float = 1.0,
         warmup_iters: int = 2000,
         lr_decay_iters: int = 600000,
         min_lr: float = 6e-5,
         jmp_policy: str = "params=float32,compute=float16,output=float32"):
    config = locals()
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    context_length = hparams["n_ctx"]
    n_head = hparams["n_head"]
    n_layer = hparams["n_layer"]

    data_path = Path(data_dir)
    tr = DataLoader(data_path/"train.bin", context_length, gradient_accumulation, batch_size)
    va = DataLoader(data_path/"val.bin", context_length, gradient_accumulation, batch_size)

    policy = jmp.get_policy(jmp_policy)

    wandb.init(project="smolGPT", config=config)
    params = randomize(params, n_layer, policy)
    params = train(params,
                   tr,
                   va,
                   n_head,
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
