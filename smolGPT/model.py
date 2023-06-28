import jax
import jax.numpy as jnp


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


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[:inputs.shape[-1]]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T
