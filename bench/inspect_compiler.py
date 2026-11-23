"""Walk the compiler stack for the NRX on the current backend and estimate where time goes.

jaxpr -> StableHLO -> optimized HLO (kernels, fusion kinds) -> memory -> time breakdown by ablation.

    uv run python bench/inspect_compiler.py [--batch 1] [--dump-passes] [--precision highest]
"""

import argparse
import collections
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nrx.model import GridLayerNorm, NeuralReceiver, make_inputs  # noqa: E402
from nrx.port_weights import load_reference  # noqa: E402

T, F, C = 14, 128, 128


def bench(fn, *args, iters=200):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def ablated_forward(params, y, no, layer_norm=True, res_convs=True):
    """The NRX written out functionally, with LayerNorms and/or the 8 residual convs replaceable by identity.
    Timing variants of the *whole* compiled graph avoids the per-call dispatch cost that dominates
    isolated-op timings on GPU."""
    p = params["params"]

    def conv(x, c):
        return jax.lax.conv_general_dilated(x, c["kernel"], (1, 1), "SAME",
                                            dimension_numbers=("NHWC", "HWIO", "NHWC")) + c["bias"]

    def norm(x, ln):
        return GridLayerNorm().apply({"params": ln}, x) if layer_norm else x

    z = conv(make_inputs(y, no), p["input_conv"])
    for i in range(1, 5):
        r = p[f"res_block_{i}"]
        h = jax.nn.relu(norm(z, r["layer_norm_1"]))
        h = conv(h, r["conv_1"]) if res_convs else h
        h = jax.nn.relu(norm(h, r["layer_norm_2"]))
        h = conv(h, r["conv_2"]) if res_convs else h
        z = h + z
    return conv(z, p["output_conv"])


def entry_kernels(hlo: str):
    entry = hlo[hlo.index("\nENTRY"):]
    kernels = []
    for line in entry.splitlines()[1:]:
        # the shape may be a tuple with spaces; it ends at the first closing bracket followed by " opcode("
        m = re.match(r"\s*(?:ROOT )?%?([\w.\-]+) = (.*?[\]})]) ([a-z][\w-]*)\(", line)
        if not m or m.group(3) in ("parameter", "constant", "tuple", "get-tuple-element", "bitcast", "copy-start", "copy-done"):
            continue
        kind = re.search(r"kind=(k\w+)", line)
        kernels.append((m.group(1), m.group(2), m.group(3) + (f":{kind.group(1)}" if kind else "")))
    return kernels


def dump_passes(B, precision):
    """Compile the forward in a child process with XLA's per-pass dump on; summarize the passes.
    Must run before this process claims the device, since a TPU can only be opened by one process."""
    tmp = pathlib.Path("data/xla_dump/_run")
    shutil.rmtree(tmp, ignore_errors=True)
    env = dict(os.environ, XLA_FLAGS=f"--xla_dump_to={tmp} --xla_dump_hlo_pass_re=.* --xla_dump_hlo_as_text")
    code = ("import sys; sys.path.insert(0,'.'); import jax, jax.numpy as jnp; "
            f"jax.config.update('jax_default_matmul_precision', {None if precision == 'default' else repr(precision)}); "
            "from nrx.model import NeuralReceiver, make_inputs; from nrx.port_weights import load_reference; "
            f"p=load_reference(); y=jnp.zeros(({B},2,14,128),jnp.complex64); no=jnp.ones(({B},)); "
            "jax.jit(lambda p,y,no: NeuralReceiver().apply(p, make_inputs(y,no))).lower(p,y,no).compile(); "
            "print(jax.default_backend())")
    out = subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
    dump = pathlib.Path("data/xla_dump") / out.stdout.split()[-1]
    shutil.rmtree(dump, ignore_errors=True)
    tmp.rename(dump)
    files = sorted(dump.glob("*.txt"))
    # files: <module>.<seq>.<pipeline>.after_<prev>.before_<next>.txt -> <next> is the pass that runs
    steps = [m.groups() for f in files if (m := re.search(r"\.\d{4}\.([\w\-]+)\.after_[\w\-]+\.before_([\w\-]+)", f.name))]
    names = [p for _, p in steps]
    pipelines = list(dict.fromkeys(pl for pl, _ in steps))
    return [f"\n## 6. Pass dump: {len(names)} pass runs, {len(set(names))} distinct passes, "
            f"{len(pipelines)} pipelines, in {dump}/",
            "  pipelines: " + " → ".join(pipelines),
            "  most frequent: " + ", ".join(f"{n}×{c}" for n, c in collections.Counter(names).most_common(8)),
            "  first 8: " + " → ".join(names[:8]),
            "  last 8:  " + " → ".join(names[-8:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dump-passes", action="store_true")
    ap.add_argument("--precision", default="default", choices=["default", "high", "highest"])
    args = ap.parse_args()
    jax.config.update("jax_default_matmul_precision", None if args.precision == "default" else args.precision)
    B = args.batch
    pass_report = dump_passes(B, args.precision) if args.dump_passes else None
    backend = jax.default_backend()
    out_lines = []

    def say(s=""):
        print(s)
        out_lines.append(s)

    params = load_reference()
    rng = np.random.default_rng(0)
    y = jnp.asarray((rng.normal(size=(B, 2, T, F)) + 1j * rng.normal(size=(B, 2, T, F))).astype(np.complex64))
    no = jnp.full((B,), 0.5, jnp.float32)
    forward = lambda p, y, no: NeuralReceiver().apply(p, make_inputs(y, no))  # noqa: E731

    say(f"# Compiler walk — backend={backend} ({jax.devices()[0].device_kind}), jax {jax.__version__}, "
        f"B={B}, precision={args.precision}\n")

    jaxpr = jax.make_jaxpr(forward)(params, y, no)
    prims = collections.Counter(e.primitive.name for e in jaxpr.jaxpr.eqns)
    say(f"## 1. jaxpr: {len(jaxpr.jaxpr.eqns)} equations")
    say("  " + ", ".join(f"{k}×{v}" for k, v in prims.most_common()))

    lowered = jax.jit(forward).lower(params, y, no)
    shlo = lowered.as_text()
    ops = collections.Counter(re.findall(r"(stablehlo\.\w+)", shlo))
    say(f"\n## 2. StableHLO: {sum(ops.values())} ops")
    say("  " + ", ".join(f"{k.split('.')[1]}×{v}" for k, v in ops.most_common()))

    compiled = lowered.compile()
    hlo = compiled.as_text() or ""
    kernels = entry_kernels(hlo) if hlo else []
    kinds = collections.Counter(k[2] for k in kernels)
    say(f"\n## 3. Optimized HLO: {len(kernels)} top-level ops per call (each one a kernel launch on GPU)")
    say("  " + ", ".join(f"{k}×{v}" for k, v in kinds.most_common()))
    say("  One residual block, in launch order:")
    start = next((i for i, k in enumerate(kernels) if "layer_norm" in k[0] or "maximum" in k[0]), 0)
    for name, shape, kind in kernels[max(start - 3, 0):start + 13]:
        say(f"    {kind:18} {shape:34} {name}")

    mem = compiled.memory_analysis()
    if mem is not None:
        say(f"\n## 4. Memory: args {mem.argument_size_in_bytes / 1e6:.1f} MB, "
            f"output {mem.output_size_in_bytes / 1e6:.2f} MB, temp {mem.temp_size_in_bytes / 1e6:.1f} MB")
        say(f"  one activation [B,14,128,128] f32 = {B * T * F * C * 4 / 1e6:.2f} MB")

    say("\n## 5. Where the time goes (ablation inside the full compiled graph, median ms)")
    variants = {
        "full NRX": dict(),
        "without LayerNorm": dict(layer_norm=False),
        "without the 8 residual convs": dict(res_convs=False),
        "without both (in/out conv + glue)": dict(layer_norm=False, res_convs=False),
    }
    ref = np.asarray(jax.jit(forward)(params, y, no))
    chk = np.asarray(jax.jit(ablated_forward)(params, y, no))
    assert np.allclose(chk, ref, atol=1e-2 * max(1.0, np.abs(ref).max())), "ablated_forward drifted from the model"
    t = {name: bench(jax.jit(lambda p, y, no, kw=kw: ablated_forward(p, y, no, **kw)), params, y, no)
         for name, kw in variants.items()}
    floor = bench(jax.jit(lambda x: x + 1), jnp.zeros((1,), jnp.float32), iters=500)
    full = t["full NRX"]
    for name, ms in t.items():
        say(f"  {name:36} {ms:8.3f}")
    say(f"  {'dispatch floor (jit x+1)':36} {floor:8.3f}")
    say(f"  ⇒ 8 residual convs ≈ {full - t['without the 8 residual convs']:.3f} ms "
        f"({(full - t['without the 8 residual convs']) / full:.0%}), "
        f"8 LayerNorms ≈ {full - t['without LayerNorm']:.3f} ms ({(full - t['without LayerNorm']) / full:.0%}), "
        f"dispatch ≈ {floor / full:.0%} of a call")

    for line in pass_report or []:
        say(line)

    suffix = "" if args.precision == "default" else f"_{args.precision}"
    out = pathlib.Path("results") / f"compiler_{backend}_b{B}{suffix}.md"
    out.write_text("\n".join(out_lines) + "\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
