import numpy as np
import pytest
import torch
from sionna.phy.ofdm import ResourceGridMapper

from nrx.data import make_resource_grid
from nrx.config import PILOTS_PATH
from nrx.pilots import QPSK, load_pilots, pilot_positions


def test_pilot_positions_reconstruct_mapper_output():
    rg = make_resource_grid(apply_saved_pilots=False)
    pos = pilot_positions(rg)
    x = ResourceGridMapper(rg)(torch.zeros(1, 1, 1, 1392, dtype=torch.complex64)).reshape(-1)
    recon = torch.zeros_like(x)
    recon[torch.as_tensor(pos, dtype=torch.long)] = rg.pilot_pattern.pilots.reshape(-1)
    assert len(pos) == 232 and set((pos // 128).tolist()) == {2, 11}
    assert torch.equal(recon, x)


@pytest.mark.skipif(not PILOTS_PATH.exists(), reason="recovered pilots not present")
def test_saved_pilots_are_used_and_qpsk():
    p = load_pilots()
    assert np.all(np.min(np.abs(p.reshape(-1, 1) - QPSK[None]), axis=1) < 1e-6)
    a, b = make_resource_grid(), make_resource_grid()
    np.testing.assert_array_equal(a.pilot_pattern.pilots.cpu().numpy(), p)
    np.testing.assert_array_equal(b.pilot_pattern.pilots.cpu().numpy(), p)
