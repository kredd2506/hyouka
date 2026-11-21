import pytest
import sionna.phy
import torch

from nrx.data import SionnaLink

# Tests run on CPU even on a GPU host: they compare against numpy and a CPU torch model.
sionna.phy.config.device = "cpu"


@pytest.fixture(scope="session")
def link():
    sionna.phy.config.seed = 42
    return SionnaLink()


@pytest.fixture(scope="session")
def batch(link):
    return link.sample_torch(8, torch.linspace(-5.0, 10.0, 8))
