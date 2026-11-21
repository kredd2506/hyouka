"""Sionna (PyTorch) link: transmitter + CDL channel -> JAX arrays for the receiver."""

import jax.numpy as jnp
import numpy as np
import sionna.phy
import torch
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import CDL, Antenna, AntennaArray
from sionna.phy.fec.ldpc import LDPC5GEncoder
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridDemapper, ResourceGridMapper
from sionna.phy.utils import ebnodb2no, expand_to_rank

from nrx.config import CONFIG, Config
from nrx.pilots import load_pilots


def make_resource_grid(cfg: Config = CONFIG, apply_saved_pilots: bool = True) -> ResourceGrid:
    """Resource grid of the tutorial. Pilots are random per construction in Sionna 2.x, so the
    pilots the reference weights were trained with are loaded from disk when available."""
    rg = ResourceGrid(num_ofdm_symbols=cfg.num_ofdm_symbols,
                        fft_size=cfg.fft_size,
                        subcarrier_spacing=cfg.subcarrier_spacing,
                        num_tx=1,
                        num_streams_per_tx=1,
                        cyclic_prefix_length=cfg.cyclic_prefix_length,
                        dc_null=cfg.dc_null,
                        pilot_pattern=cfg.pilot_pattern,
                        pilot_ofdm_symbol_indices=list(cfg.pilot_ofdm_symbol_indices),
                        num_guard_carriers=list(cfg.num_guard_carriers))
    pilots = load_pilots() if apply_saved_pilots else None
    if pilots is not None:
        rg.pilot_pattern.pilots = torch.as_tensor(pilots, device=rg.pilot_pattern.pilots.device)
    return rg


def make_stream_management() -> StreamManagement:
    return StreamManagement(np.array([[1]]), 1)


def data_re_indices(cfg: Config = CONFIG) -> np.ndarray:
    """Flat (symbol * fft_size + subcarrier) indices of data REs, in the order
    Sionna's ResourceGridDemapper emits them (and the mapper consumes coded bits)."""
    demapper = ResourceGridDemapper(make_resource_grid(cfg), make_stream_management())
    grid = torch.arange(cfg.num_ofdm_symbols * cfg.fft_size, dtype=torch.float32, device=sionna.phy.config.device)
    grid = grid.reshape(1, 1, 1, cfg.num_ofdm_symbols, cfg.fft_size, 1)
    return demapper(grid).reshape(-1).cpu().numpy().astype(np.int32)


def gather_data_llrs(llr: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    """[B, T, F, bits] LLRs on all REs -> [B, n] LLRs on data REs, bit order matching c."""
    flat = llr.reshape(llr.shape[0], -1, llr.shape[-1])
    return flat[:, indices, :].reshape(llr.shape[0], -1)


class SionnaLink:
    """Transmitter and channel of the tutorial's E2ESystem.

    sample() returns JAX arrays:
      y:  [B, num_rx_ant, T, F] complex64   post-DFT received grid
      no: [B] float32                       noise variance
      c:  [B, n] float32                    coded bits
      b:  [B, k] float32                    info bits (only when coded=True)
    """

    def __init__(self, cfg: Config = CONFIG, coded: bool = False, apply_saved_pilots: bool = True):
        self.cfg = cfg
        self.coded = coded
        self.resource_grid = make_resource_grid(cfg, apply_saved_pilots)
        assert self.resource_grid.num_data_symbols == cfg.num_data_symbols
        self._binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(cfg.k, cfg.n) if coded else None
        self._mapper = Mapper("qam", cfg.num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self.resource_grid)
        ut_antenna = Antenna(polarization="single", polarization_type="V",
                             antenna_pattern="38.901", carrier_frequency=cfg.carrier_frequency)
        bs_array = AntennaArray(num_rows=1, num_cols=1, polarization="dual", polarization_type="VH",
                                antenna_pattern="38.901", carrier_frequency=cfg.carrier_frequency)
        cdl = CDL(cfg.cdl_model, cfg.delay_spread, cfg.carrier_frequency,
                  ut_antenna, bs_array, "uplink", min_speed=cfg.speed)
        self._channel = OFDMChannel(cdl, self.resource_grid, normalize_channel=True, return_channel=True)

    @torch.no_grad()
    def sample_torch(self, batch_size: int, ebno_db: float | torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        ebno_db = torch.as_tensor(ebno_db, dtype=torch.float32, device=sionna.phy.config.device)
        ebno_db = ebno_db.expand(batch_size)
        no = ebnodb2no(ebno_db, cfg.num_bits_per_symbol, cfg.coderate)
        out = {}
        if self.coded:
            b = self._binary_source([batch_size, 1, 1, cfg.k])
            c = self.encoder(b)
            out["b"] = b.reshape(batch_size, cfg.k)
        else:
            c = self._binary_source([batch_size, 1, 1, cfg.n])
        x_rg = self._rg_mapper(self._mapper(c))
        y, h = self._channel(x_rg, expand_to_rank(no, x_rg.ndim))
        out.update(y=y.squeeze(1), no=no, c=c.reshape(batch_size, cfg.n), h=h)
        return out

    def sample(self, batch_size: int, ebno_db: float | torch.Tensor) -> dict[str, jnp.ndarray]:
        out = self.sample_torch(batch_size, ebno_db)
        out.pop("h")
        return {k: jnp.asarray(v.cpu().numpy()) for k, v in out.items()}
