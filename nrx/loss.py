"""BMD-rate objective, as in the tutorial: rate = 1 - mean BCE(llr, c) / ln 2 over data REs."""

import math

import jax.numpy as jnp
import optax

from nrx.data import gather_data_llrs


def bmd_rate(llr_grid: jnp.ndarray, c: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    """llr_grid: [B, T, F, bits] on all REs, c: [B, n] coded bits -> scalar rate in bit."""
    llr = gather_data_llrs(llr_grid, indices)
    return 1.0 - optax.sigmoid_binary_cross_entropy(llr, c).mean() / math.log(2.0)
