import jax
import jax.numpy as jnp
import numpy as np
import torch
from sionna.phy.ofdm import ResourceGridDemapper

from nrx import torch_ref
from nrx.config import CONFIG
from nrx.data import data_re_indices, gather_data_llrs, make_resource_grid, make_stream_management
from nrx.model import NeuralReceiver, make_inputs

B, T, F = 8, CONFIG.num_ofdm_symbols, CONFIG.fft_size


def test_link_shapes(link):
    out = link.sample(B, 3.0)
    assert out["y"].shape == (B, CONFIG.num_rx_ant, T, F) and out["y"].dtype == jnp.complex64
    assert out["no"].shape == (B,) and out["no"].dtype == jnp.float32
    assert out["c"].shape == (B, CONFIG.n)
    assert set(np.unique(np.asarray(out["c"]))) <= {0.0, 1.0}
    assert jnp.all(jnp.isfinite(out["y"]))


def test_noise_range(link):
    lo = link.sample(2, CONFIG.ebno_db_max)["no"]
    hi = link.sample(2, CONFIG.ebno_db_min)["no"]
    np.testing.assert_allclose(hi / lo, 10 ** ((CONFIG.ebno_db_max - CONFIG.ebno_db_min) / 10), rtol=1e-5)


def test_make_inputs_matches_torch_layout(batch):
    z = make_inputs(jnp.asarray(batch["y"].numpy()), jnp.asarray(batch["no"].numpy()))
    assert z.shape == (B, T, F, CONFIG.num_input_channels)
    y = batch["y"].numpy()
    np.testing.assert_array_equal(z[..., 0], y[:, 0].real)
    np.testing.assert_array_equal(z[..., 1], y[:, 1].real)
    np.testing.assert_array_equal(z[..., 2], y[:, 0].imag)
    np.testing.assert_array_equal(z[..., 3], y[:, 1].imag)
    # XLA:TPU's log is approximate: up to 3e-5 absolute error here (v5e), vs exact on CPU/GPU
    np.testing.assert_allclose(z[:, 0, 0, 4], np.log10(batch["no"].numpy()), rtol=1e-5, atol=1e-4)


def test_model_shapes_and_param_count():
    model = NeuralReceiver()
    params = model.init(jax.random.key(0), jnp.zeros((B, T, F, CONFIG.num_input_channels)))
    llr = model.apply(params, jnp.zeros((B, T, F, CONFIG.num_input_channels)))
    assert llr.shape == (B, T, F, CONFIG.num_bits_per_symbol)
    n_jax = sum(x.size for x in jax.tree.leaves(params))
    n_torch = sum(p.numel() for p in torch_ref.NeuralReceiver().parameters())
    assert n_jax == n_torch == 4_858_882


def test_gather_matches_resource_grid_demapper():
    llr = torch.randn(B, T, F, CONFIG.num_bits_per_symbol)
    ref = ResourceGridDemapper(make_resource_grid(), make_stream_management())(llr[:, None, None])
    ref = ref.reshape(B, CONFIG.n).numpy()
    out = gather_data_llrs(jnp.asarray(llr.numpy()), jnp.asarray(data_re_indices()))
    assert out.shape == (B, CONFIG.n)
    np.testing.assert_array_equal(out, ref)
