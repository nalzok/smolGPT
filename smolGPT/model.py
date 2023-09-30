from functools import partial

import jax
import jax.numpy as jnp


def layer_norm(act, g, b, eps: float = 1e-5):
    mean = jnp.mean(act, axis=-1, keepdims=True)
    variance = jnp.var(act, axis=-1, keepdims=True)
    act_g = (act - mean) / jnp.sqrt(variance + eps)
    out = act_g * g + b

    activation = {"g": act_g}
    return out, activation


def linear(act, w, b, u = None, v = None):
    out = act @ w + b
    activation = {"w": act}

    if u is not None or v is not None:
        act_v = out @ u
        out = out + act_v @ v
        activation.update({"u": act, "v": act_v})

    return out, activation


def ffn(act, c_fc, c_proj):
    out_fc, act_fc = linear(act, **c_fc)
    out, act_proj = linear(jax.nn.gelu(out_fc), **c_proj)

    activation = {"c_fc": act_fc, "c_proj": act_proj}
    return out, activation


def attention(q, k, v, mask):
    affinity = q @ k.swapaxes(-2, -1) / jnp.sqrt(q.shape[-1]) + mask
    return jax.nn.softmax(affinity) @ v


def mha(act, c_attn, c_proj, n_head):
    *B, T, C = act.shape
    out_attn, act_attn = linear(act, **c_attn)

    q, k, v = jnp.split(out_attn, 3, axis=-1)
    q = q.reshape(*B, T, n_head, C // n_head).swapaxes(-3, -2)
    k = k.reshape(*B, T, n_head, C // n_head).swapaxes(-3, -2)
    v = v.reshape(*B, T, n_head, C // n_head).swapaxes(-3, -2)
    causal_mask = (1 - jnp.tri(T, dtype=act.dtype)) * jnp.finfo(act.dtype).min
    out_heads = attention(q, k, v, causal_mask)
    out_heads = out_heads.swapaxes(-3, -2).reshape(*B, T, C)
    out, act_proj = linear(out_heads, **c_proj)

    activation = {"c_attn": act_attn, "c_proj": act_proj}
    return out, activation


def transformer_block(act, ln_1, attn, ln_2, mlp, n_head):
    # TODO: switch to FlashAttention? (adding LoRA might be tricker)
    #       https://github.com/lhao499/blockwise-parallel-transformer/issues/2

    out_ln_1, act_ln_1 = layer_norm(act, **ln_1)
    out_attn, act_attn = mha(out_ln_1, **attn, n_head=n_head)
    out_res_1 = act + out_attn

    out_ln_2, act_ln_2 = layer_norm(out_res_1, **ln_2)
    out_mlp, act_mlp = ffn(out_ln_2, **mlp)
    out = out_res_1 + out_mlp

    activation = {"ln_1": act_ln_1, "attn": act_attn, "ln_2": act_ln_2, "mlp": act_mlp}
    return out, activation


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_remat.html#practical-notes
    @partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def f(act_block, block):
        act_block, attn_block = transformer_block(act, **block, n_head=n_head)
        return act_block, attn_block

    act = wte[inputs] + wpe[:inputs.shape[-1]]
    out_blocks, attn_blocks = jax.lax.scan(f, act, blocks)

    act_wte, attn_ln = layer_norm(out_blocks, **ln_f)
    out = act_wte @ wte.T

    activations = {"wte": act_wte, "wpe": None, "blocks": attn_blocks, "ln_f": attn_ln}
    return out, activations
