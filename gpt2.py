import jax
import jax.numpy as jnp
from tqdm import tqdm
import fire

from utils import load_encoder_hparams_and_params


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / jnp.sqrt(variance + eps) + b


def linear(x, w, b):
    return x @ w + b


def ffn(x, c_fc, c_proj):
    return linear(jax.nn.gelu(linear(x, **c_fc)), **c_proj)


def attention(q, k, v, mask):
    affinity = q @ k.swapaxes(-2, -1) / jnp.sqrt(q.shape[-1]) + mask
    return jax.nn.softmax(affinity) @ v


def mha(x, c_attn, c_proj, n_head):
    B, T, C = x.shape
    x = linear(x, **c_attn)
    q, k, v = jnp.split(x, 3, axis=-1)
    q = q.reshape(B, T, n_head, C // n_head).swapaxes(-3, -2)
    k = k.reshape(B, T, n_head, C // n_head).swapaxes(-3, -2)
    v = v.reshape(B, T, n_head, C // n_head).swapaxes(-3, -2)
    causal_mask = (1 - jnp.tri(T, dtype=x.dtype)) * jnp.finfo(x.dtype).min
    out_heads = attention(q, k, v, causal_mask)
    x = out_heads.swapaxes(-3, -2).reshape(B, T, C)
    x = linear(x, **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


# @partial(jax.jit, static_argnums=(5,))
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[:inputs.shape[-1]]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T


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
