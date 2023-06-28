import jax.numpy as jnp
from tqdm import tqdm
import fire

from smolGPT.model import gpt2
from smolGPT.utils import load_encoder_hparams_and_params


def generate(inputs, params, n_head, n_tokens_to_generate):
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        input_ids = jnp.atleast_2d(jnp.array(inputs))
        logits = gpt2(input_ids, **params, n_head=n_head)
        next_id = jnp.argmax(logits[:, -1, :], axis=-1)
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate :]


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    fire.Fire(main)
