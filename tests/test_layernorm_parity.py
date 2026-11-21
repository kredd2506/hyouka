import jax
import jax.numpy as jnp
import numpy as np
import torch

from nrx.model import GridLayerNorm

C, T, F = 128, 14, 128


def test_layernorm_matches_torch():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 3.0, (4, C, T, F)).astype(np.float32)
    scale = rng.normal(1.0, 0.1, (C, T, F)).astype(np.float32)
    bias = rng.normal(0.0, 0.1, (C, T, F)).astype(np.float32)

    ln = torch.nn.LayerNorm([C, T, F])
    with torch.no_grad():
        ln.weight.copy_(torch.from_numpy(scale))
        ln.bias.copy_(torch.from_numpy(bias))
        ref = ln(torch.from_numpy(x)).numpy()

    params = {"params": {"scale": jnp.asarray(scale.transpose(1, 2, 0)),
                         "bias": jnp.asarray(bias.transpose(1, 2, 0))}}
    out = GridLayerNorm().apply(params, jnp.asarray(x.transpose(0, 2, 3, 1)))
    np.testing.assert_allclose(np.asarray(out).transpose(0, 3, 1, 2), ref, atol=1e-5)


def test_layernorm_param_shapes():
    params = GridLayerNorm().init(jax.random.key(0), jnp.zeros((1, T, F, C)))
    assert params["params"]["scale"].shape == (T, F, C)
