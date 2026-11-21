"""Pilot sequence handling.

Sionna 2.x draws Kronecker pilots from the torch RNG of the active device when the
ResourceGrid is built. The tutorial weights were trained on CUDA with seed 42, so their
pilots can't be regenerated on a CPU-only host, and they're not stored in the state_dict.

`recover` estimates them from the weights: y = h * x + n is linear in the pilots, so we
maximize the NRX's BMD rate over the pilot values with gradients, then snap to QPSK.

    uv run python -m nrx.pilots recover                 # -> assets/pilots_recovered.npy (~2 min CPU)
    uv run python -m nrx.pilots cuda                    # exact pilots, CUDA host only
    uv run python -m nrx.pilots compare A.npy B.npy
"""

import argparse
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from nrx.config import CONFIG, PILOTS_PATH, ROOT

QPSK = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], np.complex64) / np.sqrt(2)


def load_pilots(path=PILOTS_PATH) -> np.ndarray | None:
    path = pathlib.Path(path)
    return np.load(path) if path.exists() else None


def pilot_positions(resource_grid) -> np.ndarray:
    """Flat (symbol * fft_size + subcarrier) index of each entry of pilot_pattern.pilots."""
    from sionna.phy.ofdm import ResourceGridMapper

    pp = resource_grid.pilot_pattern
    old = pp.pilots.clone()
    n = old.shape[-1]
    tags = torch.exp(1j * 2 * math.pi * torch.arange(n) / (4 * n)).to(old.dtype).reshape(old.shape)
    pp.pilots = tags
    x = ResourceGridMapper(resource_grid)(torch.zeros(1, 1, 1, CONFIG.num_data_symbols, dtype=old.dtype))
    pp.pilots = old
    flat = x.reshape(-1)
    pos = torch.nonzero(flat.abs() > 0.5).flatten()
    order = torch.argsort(torch.angle(flat[pos]) % (2 * math.pi))
    return pos[order].numpy().astype(np.int32)


def recover(steps=300, batch_size=32, ebno_db=4.0, seed=0) -> np.ndarray:
    import sionna.phy
    from sionna.phy.ofdm import ResourceGridMapper

    from nrx.data import SionnaLink, data_re_indices, gather_data_llrs
    from nrx.model import NeuralReceiver, make_inputs
    from nrx.port_weights import load_reference

    sionna.phy.config.seed = seed
    link = SionnaLink(apply_saved_pilots=False)
    rg = link.resource_grid
    positions = jnp.asarray(pilot_positions(rg))
    n_p = positions.shape[0]
    T, F = CONFIG.num_ofdm_symbols, CONFIG.fft_size

    def draw():
        out = link.sample_torch(batch_size, ebno_db)
        c = out["c"]
        x_data = ResourceGridMapper(rg)(link._mapper(c.reshape(batch_size, 1, 1, -1))).reshape(batch_size, T * F)
        x_data[:, torch.as_tensor(np.array(positions), dtype=torch.long)] = 0
        h = out["h"][:, 0, :, 0, 0].reshape(batch_size, CONFIG.num_rx_ant, T * F)
        no = out["no"]
        std = torch.sqrt(no / 2)[:, None, None]
        noise = std * (torch.randn_like(h.real) + 1j * torch.randn_like(h.real))
        return [jnp.asarray(v.numpy()) for v in (h, x_data, noise, no, c)]

    batches = [draw() for _ in range(4)]
    params = load_reference()
    indices = jnp.asarray(data_re_indices())

    def rate(p, h, x_data, noise, no, c):
        x = x_data.at[:, positions].add(p[None])
        y = (h * x[:, None] + noise).reshape(batch_size, CONFIG.num_rx_ant, T, F)
        llr = gather_data_llrs(NeuralReceiver().apply(params, make_inputs(y, no)), indices)
        bce = optax.sigmoid_binary_cross_entropy(llr, c).mean()
        return 1.0 - bce / math.log(2.0)

    rate_c = jax.jit(lambda v, *b: rate((v[0] + 1j * v[1]) / jnp.sqrt(2.0), *b))
    grad = jax.jit(jax.grad(lambda v, *b: -rate_c(v, *b)))
    v = jax.random.normal(jax.random.key(seed), (2, n_p)) * 0.1
    opt = optax.adam(0.05)
    state = opt.init(v)
    for i in range(steps):
        b = batches[i % len(batches)]
        g = grad(v, *b)
        upd, state = opt.update(g, state)
        v = optax.apply_updates(v, upd)
        if i % 50 == 0 or i == steps - 1:
            print(f"step {i:4d}  rate {float(rate_c(v, *b)):.4f}")

    snapped = jnp.sign(v)
    p = np.asarray((snapped[0] + 1j * snapped[1]) / np.sqrt(2.0)).astype(np.complex64)
    eval_b = draw()
    score = jax.jit(lambda p: rate(p, *eval_b))
    print(f"snapped to QPSK: rate {float(score(jnp.asarray(p))):.4f} on a fresh batch")

    # Greedy polish: try every QPSK value at every pilot.
    for sweep in range(2):
        changed = 0
        best = float(score(jnp.asarray(p)))
        for k in range(n_p):
            for q in QPSK:
                if q == p[k]:
                    continue
                trial = p.copy()
                trial[k] = q
                r = float(score(jnp.asarray(trial)))
                if r > best + 1e-5:
                    best, p, changed = r, trial, changed + 1
        print(f"polish sweep {sweep}: {changed} pilots changed, rate {best:.4f}")
        if changed == 0:
            break
    return p.reshape(tuple(rg.pilot_pattern.pilots.shape))


def cuda_pilots(seed=42) -> np.ndarray:
    """The pilots the tutorial notebook builds on a CUDA device right after setting the seed."""
    import sionna.phy

    from nrx.data import make_resource_grid

    assert torch.cuda.is_available(), "exact pilots need the CUDA RNG"
    assert sionna.phy.config.device.startswith("cuda"), sionna.phy.config.device  # Sionna defaults to cuda:0
    sionna.phy.config.seed = seed
    return make_resource_grid(apply_saved_pilots=False).pilot_pattern.pilots.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recover")
    r.add_argument("--out", default=str(ROOT / "assets/pilots_recovered.npy"))
    c = sub.add_parser("cuda")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--out", default=str(ROOT / "assets/pilots_cuda_seed42.npy"))
    m = sub.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    args = ap.parse_args()

    if args.cmd == "compare":
        a, b = np.load(args.a).reshape(-1), np.load(args.b).reshape(-1)
        diff = np.flatnonzero(np.abs(a - b) > 1e-6)
        print(f"{len(diff)}/{a.size} pilots differ" + (f" at {diff.tolist()}" if len(diff) else ""))
        return
    p = recover() if args.cmd == "recover" else cuda_pilots(args.seed)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, p)
    print(f"saved {args.out}  shape {p.shape}")


if __name__ == "__main__":
    main()
