from functools import partial
from itertools import islice

import jax
import optax
import numpy as np
from tqdm import tqdm
import fire

from gpt2 import gpt2
from utils import load_encoder_hparams_and_params, replicate, unreplicate


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



@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(4, 5), donate_argnums=(0, 1))
def train_step(params, opt_state, inputs, targets, n_head, optimizer):
    @partial(jax.value_and_grad)
    def loss_fn(p):
        logits = gpt2(inputs, **p, n_head=n_head)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        # TODO: ignore label == -1
        return loss.mean()

    loss, grads = loss_fn(params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train(tr, va, params, n_head, learning_rate, num_steps):
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)

    params, opt_state = replicate((params, opt_state))
    dataloader = islice(zip(tr, va), num_steps)
    for (inputs, targets), _ in (pbar := tqdm(dataloader, "Training")):
        params, opt_state, losses = train_step(params, opt_state, inputs, targets, n_head, optimizer)
        pbar.set_description(f"{losses.mean() = }")

    return unreplicate(params)


def main(learning_rate: float = 1e-4,
         num_steps: int = 1024,
         batch_size: int = 64,
         model_size: str = "124M",
         models_dir: str = "models"):
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    context_length = hparams["n_ctx"]
    n_head = hparams["n_head"]

    tr = DataLoader("data/openwebtext/train.bin", context_length, batch_size)
    va = DataLoader("data/openwebtext/val.bin", context_length, batch_size)

    params = train(tr, va, params, n_head, learning_rate, num_steps)


if __name__ == "__main__":
    fire.Fire(main)
