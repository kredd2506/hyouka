"""Compile and time the Flax NRX on whatever XLA backend JAX picks up.

Needs only jax/flax/numpy at run time, so it runs on CPU, GPU, TPU or a PJRT plugin.
Parity is checked against torch LLRs stored in a fixture made once with --make-fixture.

    uv run python bench/backends.py --make-fixture   # needs torch + sionna + weights
    uv run python bench/backends.py                  # any backend
"""

import argparse
import json
import pathlib
import platform
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nrx.model import NeuralReceiver, make_inputs  # noqa: E402

FIXTURES = pathlib.Path("data/fixtures")
from nrx.config import WEIGHTS_PATH as WEIGHTS  # noqa: E402
PARAMS_NPZ = FIXTURES / "params.npz"
RESULTS = pathlib.Path("results")
SLOT_BUDGET_MS = 0.5
BATCH_SIZES = (1, 8, 128)


def make_fixture():
    import sionna.phy
    import torch

    from nrx import torch_ref
    from nrx.data import SionnaLink
    from nrx.port_weights import load_reference

    FIXTURES.mkdir(parents=True, exist_ok=True)
    params = load_reference(str(WEIGHTS))
    flat = {"/".join(k.key for k in path): np.asarray(v)
            for path, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    np.savez(PARAMS_NPZ, **flat)

    ref = torch_ref.NeuralReceiver()
    ref.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu"))
    ref.eval()
    sionna.phy.config.seed = 7
    link = SionnaLink()
    for b in BATCH_SIZES:
        batch = {k: v.cpu() for k, v in link.sample_torch(b, torch.empty(b).uniform_(-5.0, 10.0)).items()}
        with torch.no_grad():
            llr = ref(batch["y"], batch["no"]).numpy()  # reference LLRs always from torch on CPU
        np.savez(FIXTURES / f"batch_{b}.npz", y=batch["y"].numpy(), no=batch["no"].numpy(), llr=llr)
    print(f"fixtures written to {FIXTURES}/")


def load_params():
    flat = np.load(PARAMS_NPZ)
    params: dict = {}
    for key in flat.files:
        *path, leaf = key.split("/")
        node = params
        for p in path:
            node = node.setdefault(p, {})
        node[leaf] = jnp.asarray(flat[key])
    return params


def forward(params, y, no):
    return NeuralReceiver().apply(params, make_inputs(y, no))


def forward_real(params, y_re, y_im, no):
    """Same as forward, but complex y is split on the host, for backends without complex64."""
    y_re, y_im = jnp.moveaxis(y_re, 1, -1), jnp.moveaxis(y_im, 1, -1)
    log_no = jnp.broadcast_to(jnp.log10(no)[:, None, None, None], (*y_re.shape[:3], 1))
    return NeuralReceiver().apply(params, jnp.concatenate([y_re, y_im, log_no], axis=-1))


def time_calls(fn, args, iters):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return statistics.median(times), times[int(0.99 * (len(times) - 1))]


def run(warmup, dump_hlo, real_input):
    backend = jax.default_backend()
    device = jax.devices()[0]
    params = jax.device_put(load_params(), device)
    rows = []
    for b in BATCH_SIZES:
        fx = np.load(FIXTURES / f"batch_{b}.npz")
        if real_input:
            fn = forward_real
            inputs = (fx["y"].real.copy(), fx["y"].imag.copy(), fx["no"])
        else:
            fn = forward
            inputs = (fx["y"], fx["no"])
        inputs = jax.device_put(tuple(jnp.asarray(x) for x in inputs), device)
        args = (params, *inputs)

        t0 = time.perf_counter()
        lowered = jax.jit(fn).lower(*args)
        t_lower = time.perf_counter() - t0
        t0 = time.perf_counter()
        compiled = lowered.compile()
        t_compile = time.perf_counter() - t0

        t0 = time.perf_counter()
        llr = jax.block_until_ready(compiled(*args))
        t_first = (time.perf_counter() - t0) * 1e3
        err = float(np.max(np.abs(np.asarray(llr) - fx["llr"])))

        for _ in range(warmup):
            compiled(*args).block_until_ready()
        t_warm, _ = time_calls(compiled, args, 1)
        iters = max(20, min(500, int(2000 / max(t_warm, 1e-3))))  # ~2 s per batch size
        p50, p99 = time_calls(compiled, args, iters)

        try:
            cost = compiled.cost_analysis()
            cost = cost[0] if isinstance(cost, (list, tuple)) else cost
            flops = float(cost.get("flops", float("nan"))) if cost else float("nan")
        except Exception:  # not every PJRT plugin implements cost analysis
            flops = float("nan")
        hlo = compiled.as_text() or ""  # some PJRT plugins expose no HLO text
        n_fusions = hlo.count(" fusion(") + hlo.count("= fusion")
        if dump_hlo:
            out = RESULTS / "hlo" / backend
            out.mkdir(parents=True, exist_ok=True)
            (out / f"b{b}_stablehlo.mlir").write_text(lowered.as_text())
            (out / f"b{b}_optimized_hlo.txt").write_text(hlo)

        row = dict(backend=backend, device=str(device.device_kind), jax=jax.__version__,
                   host=platform.machine(), input="real" if real_input else "complex", batch=b, lower_s=round(t_lower, 3),
                   compile_s=round(t_compile, 3), first_call_ms=round(t_first, 3),
                   p50_ms=round(p50, 3), p99_ms=round(p99, 3), iters=iters,
                   per_user_ms=round(p50 / b, 4), gflops=round(flops / 1e9, 2),
                   gflop_per_s=round(flops / 1e9 / (p50 / 1e3), 1),
                   fusions=n_fusions, max_abs_llr_err=float(f"{err:.3e}"))
        rows.append(row)
        print(json.dumps(row))

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "backend_latency.jsonl", "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    # Batch = users decoded in the same slot, so the whole batch must finish within one slot.
    print(f"\nslot budget {SLOT_BUDGET_MS} ms @ 30 kHz SCS, judged on p99 batch latency")
    for r in rows:
        ok = "meets" if r["p99_ms"] <= SLOT_BUDGET_MS else "misses"
        print(f"  {r['backend']:>6} B={r['batch']:<4} p99 {r['p99_ms']:>9.3f} ms  ({ok})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-fixture", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-dump-hlo", action="store_true")
    ap.add_argument("--real-input", action="store_true", help="split complex y on the host")
    args = ap.parse_args()
    make_fixture() if args.make_fixture else run(args.warmup, not args.no_dump_hlo, args.real_input)
