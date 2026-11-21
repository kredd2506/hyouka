"""BER/BLER sweep through Sionna's LDPC decoder for the JAX NRX, the torch reference, and baselines.

    uv run python -m nrx.evaluate --systems jax-ported torch-ref --max-mc-iter 50
"""

import argparse
import csv
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import sionna.phy
import torch
from sionna.phy.fec.ldpc import LDPC5GDecoder
from sionna.phy.mapping import Demapper
from sionna.phy.ofdm import LMMSEEqualizer, LSChannelEstimator, RemoveNulledSubcarriers
from sionna.phy.utils import expand_to_rank, sim_ber

from nrx import torch_ref
from nrx.config import CONFIG, ROOT, WEIGHTS_PATH
from nrx.data import SionnaLink, data_re_indices, gather_data_llrs, make_stream_management
from nrx.model import NeuralReceiver, make_inputs
from nrx.port_weights import load_reference

WEIGHTS = WEIGHTS_PATH
SYSTEMS = ("baseline-perfect-csi", "baseline-ls-estimation", "torch-ref", "jax-ported", "jax-trained")


def jax_llr_fn(params):
    indices = jnp.asarray(data_re_indices())

    @jax.jit
    def fn(y, no):
        return gather_data_llrs(NeuralReceiver().apply(params, make_inputs(y, no)), indices)

    def run(y: torch.Tensor, no: torch.Tensor) -> torch.Tensor:
        llr = fn(jnp.asarray(y.cpu().numpy()), jnp.asarray(no.cpu().numpy()))
        return torch.from_numpy(np.array(llr)).to(y.device)

    return run


def torch_llr_fn():
    model = torch_ref.NeuralReceiver()
    model.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu"))
    model.eval().to(sionna.phy.config.device)
    indices = torch.as_tensor(data_re_indices(), dtype=torch.long)

    def run(y, no):
        llr = model(y, no)
        return llr.reshape(llr.shape[0], -1, llr.shape[-1])[:, indices].reshape(llr.shape[0], -1)

    return run


class E2E:
    """mc_fun for sim_ber: (batch_size, ebno_db) -> (b, b_hat)."""

    def __init__(self, system: str, params=None):
        self.system = system
        self.link = SionnaLink(coded=True)
        self.decoder = LDPC5GDecoder(self.link.encoder, hard_out=True)
        rg = self.link.resource_grid
        if system == "jax-ported":
            self.llr_fn = jax_llr_fn(load_reference(WEIGHTS))
        elif system == "jax-trained":
            self.llr_fn = jax_llr_fn(params)
        elif system == "torch-ref":
            self.llr_fn = torch_llr_fn()
        else:
            self.remove_nulled = RemoveNulledSubcarriers(rg)
            self.ls_est = LSChannelEstimator(rg, interpolation_type="nn")
            self.lmmse = LMMSEEqualizer(rg, make_stream_management())
            self.demapper = Demapper("app", "qam", CONFIG.num_bits_per_symbol)

    @torch.no_grad()
    def __call__(self, batch_size, ebno_db):
        out = self.link.sample_torch(batch_size, ebno_db)
        y, no = out["y"], out["no"]
        if self.system.startswith("baseline"):
            y5 = y.unsqueeze(1)
            if self.system == "baseline-perfect-csi":
                h_hat, err_var = self.remove_nulled(out["h"]), 0.0
            else:
                h_hat, err_var = self.ls_est(y5, no)
            x_hat, no_eff = self.lmmse(y5, h_hat, err_var, no)
            llr = self.demapper(x_hat, expand_to_rank(no_eff, x_hat.ndim)).reshape(batch_size, CONFIG.n)
        else:
            llr = self.llr_fn(y, no)
        b_hat = self.decoder(llr.reshape(batch_size, 1, 1, CONFIG.n))
        return out["b"].reshape(batch_size, 1, 1, CONFIG.k), b_hat


def sweep(system, ebno_dbs, batch_size, max_mc_iter, params=None, seed=CONFIG.seed):
    sionna.phy.config.seed = seed  # same seed per system -> same channel draws
    ber, bler = sim_ber(E2E(system, params), torch.as_tensor(ebno_dbs, dtype=torch.float32),
                        batch_size=batch_size, num_target_block_errors=100, max_mc_iter=max_mc_iter,
                        target_bler=1e-4, verbose=True)
    return ber.cpu().numpy(), bler.cpu().numpy()


def ebno_at_bler(ebno_dbs, bler, target):
    """Eb/N0 where BLER first crosses target, log-linear interpolation; nan if never."""
    ebno_dbs, bler = np.asarray(ebno_dbs), np.asarray(bler)
    for i in range(1, len(bler)):
        if bler[i - 1] >= target > bler[i] and bler[i] > 0:
            t = (np.log10(bler[i - 1]) - np.log10(target)) / (np.log10(bler[i - 1]) - np.log10(bler[i]))
            return float(ebno_dbs[i - 1] + t * (ebno_dbs[i] - ebno_dbs[i - 1]))
    return float("nan")


def read_curves(path):
    curves: dict = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            curves.setdefault(r["system"], ([], []))
            curves[r["system"]][0].append(float(r["ebno_db"]))
            curves[r["system"]][1].append(float(r["bler"]))
    return curves


def plot(curves: dict, reference: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    for i, (name, (x, y)) in enumerate(reference.items()):
        plt.semilogy(x, y, "--", c=f"C{i}", alpha=0.5, label=f"{name} (notebook)")
    for i, (name, (x, y)) in enumerate(curves.items()):
        plt.semilogy(x, np.where(np.asarray(y) > 0, y, np.nan), "o-", ms=3, c=f"C{i + 3}", label=name)
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BLER")
    plt.ylim(1e-4, 1.0)
    plt.grid(which="both", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["jax-ported"], choices=SYSTEMS)
    ap.add_argument("--checkpoint", help="orbax checkpoint dir for jax-trained")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-mc-iter", type=int, default=1000, help="notebook uses 1000")
    ap.add_argument("--ebno-min", type=float, default=CONFIG.ebno_db_min)
    ap.add_argument("--ebno-max", type=float, default=CONFIG.ebno_db_max)
    ap.add_argument("--out", default="results/ber_bler.csv")
    args = ap.parse_args()

    params = None
    if "jax-trained" in args.systems:
        from nrx.train import load_checkpoint
        params = load_checkpoint(args.checkpoint)

    ebno_dbs = np.arange(args.ebno_min, args.ebno_max, 0.5)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    curves = {}
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ebno_db", "system", "ber", "bler", "max_mc_iter", "batch_size"])
        for system in args.systems:
            ber, bler = sweep(system, ebno_dbs, args.batch_size, args.max_mc_iter, params)
            curves[system] = (ebno_dbs, bler)
            for e, be, bl in zip(ebno_dbs, ber, bler):
                w.writerow([f"{e:.1f}", system, f"{be:.6e}", f"{bl:.6e}", args.max_mc_iter, args.batch_size])
            f.flush()

    reference = read_curves(ROOT / "results/reference.csv")
    plot(curves, reference, out.with_suffix(".png"))
    print(f"\n{'system':24} " + " ".join(f"{'@' + t:>10}" for t in ("1e-1", "1e-2", "1e-3", "1e-4")))
    for name, (x, y) in {**{f"{k} (nb)": v for k, v in reference.items()}, **curves}.items():
        print(f"{name:24} " + " ".join(f"{ebno_at_bler(x, y, t):>10.2f}" for t in (1e-1, 1e-2, 1e-3, 1e-4)))


if __name__ == "__main__":
    main()
