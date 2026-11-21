"""PyTorch neural receiver from Sionna's Neural_Receiver tutorial, kept verbatim
apart from taking the grid size as arguments. Used as the parity reference."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nrx.config import CONFIG


class ResidualBlock(nn.Module):
    def __init__(self, num_conv_channels: int, num_ofdm_symbols: int, fft_size: int):
        super().__init__()
        self._layer_norm_1 = nn.LayerNorm([num_conv_channels, num_ofdm_symbols, fft_size])
        self._conv_1 = nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=3, padding=1)
        self._layer_norm_2 = nn.LayerNorm([num_conv_channels, num_ofdm_symbols, fft_size])
        self._conv_2 = nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = self._layer_norm_1(inputs)
        z = F.relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = F.relu(z)
        z = self._conv_2(z)
        return z + inputs


class NeuralReceiver(nn.Module):
    """y: [B, num_rx_ant, num_ofdm_symbols, fft_size] complex, no: [B] -> llr: [B, T, F, bits]"""

    def __init__(self, num_conv_channels: int = CONFIG.num_conv_channels,
                 num_bits_per_symbol: int = CONFIG.num_bits_per_symbol,
                 num_ofdm_symbols: int = CONFIG.num_ofdm_symbols,
                 fft_size: int = CONFIG.fft_size):
        super().__init__()
        self._input_conv = nn.Conv2d(2 * 2 + 1, num_conv_channels, kernel_size=3, padding=1)
        self._res_block_1 = ResidualBlock(num_conv_channels, num_ofdm_symbols, fft_size)
        self._res_block_2 = ResidualBlock(num_conv_channels, num_ofdm_symbols, fft_size)
        self._res_block_3 = ResidualBlock(num_conv_channels, num_ofdm_symbols, fft_size)
        self._res_block_4 = ResidualBlock(num_conv_channels, num_ofdm_symbols, fft_size)
        self._output_conv = nn.Conv2d(num_conv_channels, num_bits_per_symbol, kernel_size=3, padding=1)

    def forward(self, y: torch.Tensor, no: torch.Tensor) -> torch.Tensor:
        no = torch.log10(no)
        y_real, y_imag = y.real, y.imag
        batch_size = y.shape[0]
        no = no.view(-1, 1, 1, 1).expand(batch_size, 1, y.shape[2], y.shape[3])
        z = torch.cat([y_real[:, 0:1], y_real[:, 1:2], y_imag[:, 0:1], y_imag[:, 1:2], no], dim=1)
        z = self._input_conv(z)
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        z = self._output_conv(z)
        return z.permute(0, 2, 3, 1)
