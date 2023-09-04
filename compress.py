from pathlib import Path

import jax
import jax.numpy as jnp
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



def svdvals(params, first_n_svd):
    def svd(path, leaf):
        S = jnp.linalg.svd(leaf, compute_uv=False, hermitian=False)
        Ssorted = jnp.sort(jnp.max(jnp.abs(S)))
        return Ssorted[first_n_svd:]

    Smax = jax.tree_util.tree_map_with_path(svd, params)
    return Smax


def compress(params, first_n_svd):
    pass


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

    data_path = Path(data_dir)
    va = DataLoader(data_path/"val.bin", context_length, gradient_accumulation, batch_size)

    wandb.init(project="smolGPT", config=config)
    params = compress(params,
                      va,
                      iterations,
                      first_n_svd)
    wandb.finish()


if __name__ == "__main__":
    pass
