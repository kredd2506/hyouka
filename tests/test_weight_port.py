import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from nrx import torch_ref
from nrx.config import WEIGHTS_PATH
from nrx.model import NeuralReceiver, make_inputs
from nrx.port_weights import load_reference

WEIGHTS = WEIGHTS_PATH
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="pre-trained weights not downloaded")


def test_ported_llrs_match_torch(batch):
    ref_model = torch_ref.NeuralReceiver()
    ref_model.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu"))
    ref_model.eval()
    with torch.no_grad():
        ref = ref_model(batch["y"], batch["no"]).numpy()

    params = load_reference(str(WEIGHTS))
    z = make_inputs(jnp.asarray(batch["y"].numpy()), jnp.asarray(batch["no"].numpy()))
    with jax.default_matmul_precision("highest"):  # GPUs default to TF32 convs (~1e-2 LLR error)
        out = np.asarray(NeuralReceiver().apply(params, z))

    err = np.max(np.abs(out - ref))
    print(f"max|LLR_jax - LLR_torch| = {err:.3e}, max|LLR| = {np.max(np.abs(ref)):.1f}")
    assert err < 1e-4
