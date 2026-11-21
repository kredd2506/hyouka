"""Export the torch reference NRX to ONNX for TensorRT, and check it with onnxruntime.

ONNX has no complex dtype, so the exported graph takes Re/Im split on the host:
    y_re, y_im: [B, 2, 14, 128] float32, no: [B] float32 -> llr: [B, 14, 128, 2]

    uv run python bench/export_onnx.py            # writes results/onnx/nrx_b{B}.onnx
On the GPU host:
    trtexec --onnx=results/onnx/nrx_b1.onnx --saveEngine=nrx_b1.plan          # fp32
    trtexec --onnx=results/onnx/nrx_b1.onnx --fp16 --saveEngine=nrx_b1_fp16.plan
"""

import pathlib
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nrx import torch_ref  # noqa: E402

from nrx.config import WEIGHTS_PATH as WEIGHTS  # noqa: E402
FIXTURES = pathlib.Path("data/fixtures")
OUT = pathlib.Path("results/onnx")
BATCH_SIZES = (1, 8, 128)  # static shapes, one engine per batch size


class RealInputNRX(torch.nn.Module):
    def __init__(self, nrx):
        super().__init__()
        self.nrx = nrx

    def forward(self, y_re, y_im, no):
        return self.nrx(torch.complex(y_re, y_im), no)


def main():
    nrx = torch_ref.NeuralReceiver()
    nrx.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu"))
    model = RealInputNRX(nrx).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    for b in BATCH_SIZES:
        fx = np.load(FIXTURES / f"batch_{b}.npz")
        args = (torch.from_numpy(fx["y"].real.copy()), torch.from_numpy(fx["y"].imag.copy()),
                torch.from_numpy(fx["no"]))
        path = OUT / f"nrx_b{b}.onnx"
        try:
            torch.onnx.export(model, args, path, input_names=["y_re", "y_im", "no"],
                              output_names=["llr"], opset_version=18, dynamo=True, external_data=False)
        except TypeError:  # older torch without external_data
            torch.onnx.export(model, args, path, input_names=["y_re", "y_im", "no"],
                              output_names=["llr"], opset_version=18, dynamo=True)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        llr = sess.run(None, {"y_re": args[0].numpy(), "y_im": args[1].numpy(), "no": args[2].numpy()})[0]
        err = float(np.max(np.abs(llr - fx["llr"])))
        rms = float(np.sqrt(np.mean((llr - fx["llr"]) ** 2)))
        print(f"B={b:<4} {path}  {path.stat().st_size / 1e6:.1f} MB  "
              f"LLR_ort - LLR_torch: max {err:.2e}, rms {rms:.2e}")
        # ORT's fp32 LayerNormalization over 229k elements is ~100x less accurate than a two-pass
        # reduction (torch, XLA), so ORT is gated on rms; its max grows with B (more outliers).
        assert rms < 2e-3


if __name__ == "__main__":
    main()
