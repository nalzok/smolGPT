from typing import Optional
import json
import os
import re

import numpy as np
import jax
import chex
import requests
import tensorflow as tf
from tqdm import tqdm

from smolGPT.encoder import get_encoder


def download_gpt2_files(model_size, model_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()

        with open(os.path.join(model_dir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True,
                unit="b",
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name[len("model/") :]
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            if m is None:
                raise ValueError(f"Invalid {name = }")
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def load_encoder_hparams_and_params(model_size, models_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]

    model_dir = os.path.join(models_dir, model_size)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not tf_ckpt_path:  # download files if necessary
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2_files(model_size, model_dir)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    encoder = get_encoder(model_size, models_dir)
    hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

    return encoder, hparams, params


def make_scan_friendly(blocks):
    def access(block, path):
        for p in path:
            block = block[p.key]
        return block

    def collector(path, _):
        return np.stack([access(block, path) for block in blocks])

    blocks_transposed = jax.tree_util.tree_map_with_path(collector, blocks[0])
    return blocks_transposed


class DataLoader:
    def __init__(self, filename, context_length, gradient_accumulation, batch_size, seed = 42) -> None:
        self.data = np.memmap(filename, dtype=np.uint16, mode="r")
        self.context_length = context_length
        device_count = jax.local_device_count()
        if gradient_accumulation % device_count != 0:
            raise ValueError(f"{gradient_accumulation % device_count = }")
        self.index_shape = (device_count, gradient_accumulation // device_count, batch_size)
        self.shape = (device_count, gradient_accumulation // device_count, batch_size, self.context_length)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

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

    def reset(self):
        self.rng = np.random.default_rng(self.seed)


def replicate(tree):
    return jax.device_put_replicated(tree, jax.local_devices())


def unreplicate(tree):
    return jax.tree_map(lambda x: x[0], tree)


def is_penultimate(tree):
    children = jax.tree_util.tree_structure(tree).children()
    return all(jax.tree_util.treedef_is_leaf(treedef) for treedef in children)


# forward-over-reverse
def hvp(f, primals, tangents):
    primals_out, tangents_out = jax.jvp(jax.grad(f), primals, tangents)
    return tangents_out


def canonicalize_dtype(dtype: Optional[chex.ArrayDType]) -> Optional[chex.ArrayDType]:
    if dtype is not None:
        return jax.dtypes.canonicalize_dtype(dtype)
    return dtype
