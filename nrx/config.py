"""System parameters, matching Sionna's Neural_Receiver tutorial (Sionna 2.2.0), and file paths."""

import os
import pathlib
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "weights/reference/neural_receiver_weights"
WEIGHTS_SHA256 = "efa5c7adfec68ab31921adcbc5bb3e2ccd295ae219c2f98841e325ca4bb7bc9a"
WEIGHTS_GDRIVE_ID = "1wjBB3U8Cp4a6VMZSrLBYJl6aG5sPoG_I"
# Pilots the reference weights were trained with: regenerated from the CUDA RNG (seed 42) on a Colab L4.
# assets/pilots_recovered.npy, recovered from the weights alone, is identical. NRX_PILOTS overrides.
PILOTS_PATH = pathlib.Path(os.environ.get("NRX_PILOTS", ROOT / "assets/pilots_cuda_seed42.npy"))


@dataclass(frozen=True)
class Config:
    # Channel
    carrier_frequency: float = 3.5e9
    delay_spread: float = 100e-9
    cdl_model: str = "C"
    speed: float = 10.0
    ebno_db_min: float = -5.0
    ebno_db_max: float = 10.0

    # OFDM
    subcarrier_spacing: float = 30e3
    fft_size: int = 128
    num_ofdm_symbols: int = 14
    dc_null: bool = True
    num_guard_carriers: tuple[int, int] = (5, 6)
    pilot_pattern: str = "kronecker"
    pilot_ofdm_symbol_indices: tuple[int, ...] = (2, 11)
    cyclic_prefix_length: int = 0
    num_rx_ant: int = 2

    # Modulation and coding
    num_bits_per_symbol: int = 2
    coderate: float = 0.5
    num_data_symbols: int = 1392

    # Neural receiver
    num_conv_channels: int = 128
    num_res_blocks: int = 4
    layer_norm_eps: float = 1e-5

    # Training
    num_training_iterations: int = 30000
    training_batch_size: int = 128
    learning_rate: float = 1e-3

    # Seeds
    seed: int = 42

    @property
    def n(self) -> int:
        return self.num_data_symbols * self.num_bits_per_symbol

    @property
    def k(self) -> int:
        return int(self.n * self.coderate)

    @property
    def num_input_channels(self) -> int:
        return 2 * self.num_rx_ant + 1


CONFIG = Config()
