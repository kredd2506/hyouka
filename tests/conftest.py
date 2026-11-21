import pytest
import sionna.phy
import torch

from nrx.data import SionnaLink


@pytest.fixture(scope="session")
def link():
    sionna.phy.config.seed = 42
    return SionnaLink()


@pytest.fixture(scope="session")
def batch(link):
    return link.sample_torch(8, torch.linspace(-5.0, 10.0, 8))
