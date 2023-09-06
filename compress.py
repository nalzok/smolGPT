from pathlib import Path
from functools import partial
from itertools import islice

import jax
import jax.numpy as jnp
import numpy as np
import fire
import wandb

from smolGPT.model import gpt2
from smolGPT.utils import (
    load_encoder_hparams_and_params,
    DataLoader,
    replicate,
    unreplicate,
    is_penultimate,
    canonicalize_dtype,
)



def svdvals_fn(params, first_n_svd):
    def svd(path, leaf):
        if len(leaf.shape) < 2:
            return None

        print(path, leaf.shape)
        r = min(leaf.shape)
        result_shape = jax.ShapeDtypeStruct((r,), leaf.dtype)
        S = jax.pure_callback(partial(np.linalg.svd, compute_uv=False, hermitian=False), result_shape, leaf)

        # S = jnp.linalg.svd(leaf, compute_uv=False, hermitian=False)

        return S[first_n_svd:]

    Smax = jax.tree_util.tree_map_with_path(svd, params)
    return Smax


def compress(params, va, iterations, first_n_svd, n_head):
    # svdvals = svdvals_fn(params, first_n_svd)

    loader = islice(va, iterations)
    inputs, _ = next(loader)

    logits, activations = gpt2(inputs, **params, n_head=n_head)

    def show(path, leaf):
        print(path, leaf.shape)
    jax.tree_util.tree_map_with_path(show, activations)


def main(model_size: str = "124M",
         models_dir: str = "models",
         data_dir: str = "data/openwebtext",
         first_n_svd: int = 16,
         iterations: int = 64,
         gradient_accumulation: int = 2,
         batch_size: int = 8):

    config = locals()
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    context_length = hparams["n_ctx"]
    n_head = hparams["n_head"]

    data_path = Path(data_dir)
    va = DataLoader(data_path/"val.bin", context_length, gradient_accumulation, batch_size)

    # wandb.init(project="smolGPT", config=config)
    params = compress(params,
                      va,
                      iterations,
                      first_n_svd,
                      n_head)
    # wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
