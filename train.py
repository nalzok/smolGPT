from functools import partial
from itertools import islice
from operator import add
from hashlib import sha1
from sys import maxsize

import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
import fire
import wandb

from gpt2 import gpt2
from utils import load_encoder_hparams_and_params, replicate, unreplicate, hvp


class DataLoader:
    def __init__(self, filename, context_length, batch_size) -> None:
        self.data = np.memmap(filename, dtype=np.uint16, mode="r")
        self.context_length = context_length
        self.batch_size = batch_size
        device_count = jax.local_device_count()
        if batch_size % device_count != 0:
            raise ValueError(f"{batch_size % device_count = }")
        self.index_shape = (device_count, batch_size // device_count)
        self.shape = (device_count, batch_size // device_count, self.context_length)
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


def randomize(params, n_layer):
    # See https://github.com/karpathy/nanoGPT/blob/4eb7a96b077998f28b57938c2f1e511b0d8cab7c/model.py#L140-L145
    def init_param(path, leaf):
        path_bytes = ".".join(str(p) for p in path).encode("utf-8")
        path_hash = int(sha1(path_bytes).hexdigest(), 16) % maxsize
        leaf_key = jax.random.PRNGKey(path_hash)

        noise = jax.random.normal(leaf_key, leaf.shape, leaf.dtype)
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


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(4, 5), donate_argnums=(0, 1))
def train_step(params, opt_state, inputs, targets, n_head, optimizer):
    def loss_fn(p):
        logits = gpt2(inputs, **p, n_head=n_head)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return loss.mean()

    loss, grads = jax.value_and_grad(loss_fn)(params)
    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    def agg(x, y):
        return jnp.sum(x * y)

    pred1 = jax.tree_util.tree_reduce(add, jax.tree_map(agg, updates, grads))
    hv = hvp(loss_fn, (params,), (updates,))
    hv = jax.lax.pmean(hv, axis_name="batch")
    pred2 = pred1 + 0.5 * jax.tree_util.tree_reduce(add, jax.tree_map(agg, updates, hv))

    return params, opt_state, loss, pred1, pred2


def train(tr, va, params, n_head, learning_rate, num_steps):
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)

    params, opt_state = replicate((params, opt_state))
    dataloader = islice(zip(tr, va), num_steps)
    last_loss = float('inf')
    last_pred1 = 0
    last_pred2 = 0
    for step, ((inputs, targets), _) in enumerate(pbar := tqdm(dataloader, "Training")):
        params, opt_state, loss, pred1, pred2 = train_step(params, opt_state, inputs, targets, n_head, optimizer)
        loss = float(unreplicate(loss))
        pred1 = float(unreplicate(pred1))
        pred2 = float(unreplicate(pred2))
        pbar.set_description(f"{loss = }")

        delta = last_loss - loss
        wandb.log({
            "step": step,
            "loss": loss,
            "delta": delta,
            "pred1": last_pred1,
            "pred2": last_pred2,
            "err1": delta - last_pred1,
            "err2": delta - last_pred2,
        })
        last_loss, last_pred1, last_pred2 = loss, pred1, pred2

    return unreplicate(params)


def main(learning_rate: float = 1e-4,
         num_steps: int = 2**20,
         batch_size: int = 72,
         model_size: str = "124M",
         models_dir: str = "models"):
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    context_length = hparams["n_ctx"]
    n_head = hparams["n_head"]
    n_layer = hparams["n_layer"]

    tr = DataLoader("data/openwebtext/train.bin", context_length, batch_size)
    va = DataLoader("data/openwebtext/val.bin", context_length, batch_size)

    wandb.init(project="smolGPT")
    params = randomize(params, n_layer)
    params = train(tr, va, params, n_head, learning_rate, num_steps)
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
