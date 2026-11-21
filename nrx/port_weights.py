"""Convert the Sionna PyTorch NRX state_dict to Flax params."""

import jax.numpy as jnp
import numpy as np
import torch

from nrx.config import WEIGHTS_PATH


def torch_to_flax(state_dict: dict[str, torch.Tensor]) -> dict:
    params: dict = {}
    for key, value in state_dict.items():
        *path, leaf = key.split(".")
        path = [p.lstrip("_") for p in path]
        v = value.detach().cpu().numpy()
        if path[-1].startswith("layer_norm"):
            name = {"weight": "scale", "bias": "bias"}[leaf]
            v = np.transpose(v, (1, 2, 0))  # [C, T, F] -> [T, F, C]
        elif leaf == "weight":
            name = "kernel"
            v = np.transpose(v, (2, 3, 1, 0))  # OIHW -> HWIO
        else:
            name = "bias"
        node = params
        for p in path:
            node = node.setdefault(p, {})
        node[name] = jnp.asarray(v)
    return {"params": params}


def load_reference(path=WEIGHTS_PATH) -> dict:
    return torch_to_flax(torch.load(path, weights_only=True, map_location="cpu"))
