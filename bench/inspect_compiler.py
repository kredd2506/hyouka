"""Walk the compiler stack for the NRX on the current backend and estimate where time goes.

jaxpr -> StableHLO -> optimized HLO (kernels, fusion kinds) -> memory -> per-op microbenchmarks.

    uv run python bench/inspect_compiler.py [--batch 1] [--dump-passes]
"""

import argparse
import collections
import os
import pathlib
import re
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


def entry_kernels(hlo: str):
    entry = hlo[hlo.index("\nENTRY"):]
    kernels = []
    for line in entry.splitlines()[1:]:
        m = re.match(r"\s*(?:ROOT )?%?([\w.\-]+) = (\S+) (\w[\w-]*)\(", line)
        if not m or m.group(3) in ("parameter", "constant", "tuple", "get-tuple-element", "bitcast", "copy-start", "copy-done"):
            continue
        kind = re.search(r"kind=(k\w+)", line)
        kernels.append((m.group(1), m.group(2), m.group(3) + (f":{kind.group(1)}" if kind else "")))
    return kernels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dump-passes", action="store_true")
    args = ap.parse_args()
    B = args.batch
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

    say(f"# Compiler walk — backend={backend}, jax {jax.__version__}, B={B}\n")

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
    say(f"\n## 3. Optimized HLO: {len(kernels)} kernels launched per call")
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

    say("\n## 5. Where the time goes (isolated microbenchmarks, median ms)")
    x = jnp.asarray(rng.normal(size=(B, T, F, C)).astype(np.float32))
    k = params["params"]["res_block_1"]["conv_1"]
    ln = params["params"]["res_block_1"]["layer_norm_1"]
    conv = jax.jit(lambda x, w: jax.lax.conv_general_dilated(x, w, (1, 1), "SAME",
                                                            dimension_numbers=("NHWC", "HWIO", "NHWC")))
    conv_bias = jax.jit(lambda x, w, b: conv(x, w) + b)
    lnorm = jax.jit(lambda x: GridLayerNorm().apply({"params": ln}, x))
    ln_relu = jax.jit(lambda x: jax.nn.relu(GridLayerNorm().apply({"params": ln}, x)))
    full = jax.jit(forward)
    t_conv = bench(conv, x, k["kernel"])
    t_conv_b = bench(conv_bias, x, k["kernel"], k["bias"])
    t_ln = bench(lnorm, x)
    t_ln_r = bench(ln_relu, x)
    t_full = bench(full, params, y, no)
    est = 8 * t_conv_b + 8 * t_ln_r
    say(f"  conv 128→128 3×3         {t_conv:8.3f}")
    say(f"  conv + bias              {t_conv_b:8.3f}   (bias add costs {t_conv_b - t_conv:+.3f})")
    say(f"  LayerNorm                {t_ln:8.3f}")
    say(f"  LayerNorm + ReLU         {t_ln_r:8.3f}")
    say(f"  full NRX                 {t_full:8.3f}")
    say(f"  8×(conv+bias) = {8 * t_conv_b:.3f} ({8 * t_conv_b / t_full:.0%}),  "
        f"8×(LN+ReLU) = {8 * t_ln_r:.3f} ({8 * t_ln_r / t_full:.0%}),  sum {est:.3f} vs full {t_full:.3f}")

    if args.dump_passes:
        dump = pathlib.Path("data/xla_dump") / backend
        env = dict(os.environ, XLA_FLAGS=f"--xla_dump_to={dump} --xla_dump_hlo_pass_re=.* --xla_dump_hlo_as_text")
        code = ("import sys; sys.path.insert(0,'.'); import jax, jax.numpy as jnp; "
                "from nrx.model import NeuralReceiver, make_inputs; from nrx.port_weights import load_reference; "
                f"p=load_reference(); y=jnp.zeros(({B},2,14,128),jnp.complex64); no=jnp.ones(({B},)); "
                "jax.jit(lambda p,y,no: NeuralReceiver().apply(p, make_inputs(y,no))).lower(p,y,no).compile()")
        subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)
        files = sorted(dump.glob("*.txt"))
        # files: <module>.<seq>.<pipeline>.after_<prev>.before_<next>.txt -> <next> is the pass that runs
        steps = [m.groups() for f in files if (m := re.search(r"\.\d{4}\.([\w\-]+)\.after_[\w\-]+\.before_([\w\-]+)", f.name))]
        names = [p for _, p in steps]
        pipelines = list(dict.fromkeys(pl for pl, _ in steps))
        say(f"\n## 6. Pass dump: {len(names)} pass runs, {len(set(names))} distinct passes, "
            f"{len(pipelines)} pipelines, in {dump}/")
        say("  pipelines: " + " → ".join(pipelines))
        say("  most frequent: " + ", ".join(f"{n}×{c}" for n, c in collections.Counter(names).most_common(8)))
        say("  first 8: " + " → ".join(names[:8]))
        say("  last 8:  " + " → ".join(names[-8:]))

    out = pathlib.Path("results") / f"compiler_{backend}_b{B}.md"
    out.write_text("\n".join(out_lines) + "\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
