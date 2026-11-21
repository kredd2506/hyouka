"""Flax neural receiver. NHWC layout: [batch, ofdm_symbols, subcarriers, channels]."""

import flax.linen as nn
import jax.numpy as jnp

from nrx.config import CONFIG


def make_inputs(y: jnp.ndarray, no: jnp.ndarray) -> jnp.ndarray:
    """y: [B, 2, T, F] complex, no: [B] -> [B, T, F, 5] as [Re0, Re1, Im0, Im1, log10(no)]."""
    y = jnp.moveaxis(y, 1, -1)
    log_no = jnp.broadcast_to(jnp.log10(no)[:, None, None, None], (*y.shape[:3], 1))
    return jnp.concatenate([y.real, y.imag, log_no], axis=-1).astype(jnp.float32)


class GridLayerNorm(nn.Module):
    """LayerNorm over the last three axes with per-element scale and offset,
    equivalent to torch nn.LayerNorm([C, T, F]) on NCHW input."""

    eps: float = CONFIG.layer_norm_eps

    @nn.compact
    def __call__(self, x):
        shape = x.shape[-3:]
        scale = self.param("scale", nn.initializers.ones, shape)
        bias = self.param("bias", nn.initializers.zeros, shape)
        mean = jnp.mean(x, axis=(-3, -2, -1), keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=(-3, -2, -1), keepdims=True)
        return (x - mean) * jnp.reciprocal(jnp.sqrt(var + self.eps)) * scale + bias


class ResidualBlock(nn.Module):
    num_conv_channels: int

    @nn.compact
    def __call__(self, inputs):
        z = GridLayerNorm(name="layer_norm_1")(inputs)
        z = nn.relu(z)
        z = nn.Conv(self.num_conv_channels, (3, 3), padding="SAME", name="conv_1")(z)
        z = GridLayerNorm(name="layer_norm_2")(z)
        z = nn.relu(z)
        z = nn.Conv(self.num_conv_channels, (3, 3), padding="SAME", name="conv_2")(z)
        return z + inputs


class NeuralReceiver(nn.Module):
    """[B, T, F, 5] -> LLRs [B, T, F, num_bits_per_symbol] on every resource element."""

    num_conv_channels: int = CONFIG.num_conv_channels
    num_bits_per_symbol: int = CONFIG.num_bits_per_symbol
    num_res_blocks: int = CONFIG.num_res_blocks

    @nn.compact
    def __call__(self, z):
        z = nn.Conv(self.num_conv_channels, (3, 3), padding="SAME", name="input_conv")(z)
        for i in range(1, self.num_res_blocks + 1):
            z = ResidualBlock(self.num_conv_channels, name=f"res_block_{i}")(z)
        return nn.Conv(self.num_bits_per_symbol, (3, 3), padding="SAME", name="output_conv")(z)
