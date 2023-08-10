import jax
import jax.numpy as jnp


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / jnp.sqrt(variance + eps) + b


def linear(x, w, b, u = None, v = None):
    if u is None and v is None:
        return x @ w + b

    return x @ w + (x @ u) @ v + b


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
    # TODO: switch to FlashAttention? (adding LoRA might be tricker)
    #       https://github.com/lhao499/blockwise-parallel-transformer/issues/2
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    def access(block, path):
        for p in path:
            block = block[p.key]
        return block

    def collector(path, _):
        return jnp.stack([access(block, path) for block in blocks])

    blocks_transposed = jax.tree_util.tree_map_with_path(collector, blocks[0])

    x = wte[inputs] + wpe[:inputs.shape[-1]]
    def f(x, block):
        x = transformer_block(x, **block, n_head=n_head)
        return x, None
    x, _ = jax.lax.scan(f, x, blocks_transposed)
    return layer_norm(x, **ln_f) @ wte.T
