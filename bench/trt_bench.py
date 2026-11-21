"""Build TensorRT engines from the ONNX export and time them like bench/backends.py does XLA.

Same fixtures, same batch sizes, same wall-clock method (launch + synchronize), so the two are comparable.
pip wheels of TensorRT don't ship trtexec, hence the Python API.

    uv pip install tensorrt-cu13   # or tensorrt-cu12, matching torch's CUDA
    uv run --no-sync python bench/trt_bench.py
"""

import json
import pathlib
import statistics
import time

import numpy as np
import tensorrt as trt
import torch

ONNX_DIR = pathlib.Path("results/onnx")
FIXTURES = pathlib.Path("data/fixtures")
RESULTS = pathlib.Path("results")
BATCH_SIZES = (1, 8, 128)
SLOT_BUDGET_MS = 0.5
LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path, fp16):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if fp16:
        if not hasattr(trt.BuilderFlag, "FP16"):
            raise NotImplementedError("this TensorRT has no FP16 builder flag (strongly-typed only)")
        config.set_flag(trt.BuilderFlag.FP16)
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("engine build failed")
    build_s = time.perf_counter() - t0
    engine = trt.Runtime(LOGGER).deserialize_cuda_engine(plan)
    return engine, build_s


def run(engine, feeds):
    ctx = engine.create_execution_context()
    bufs = {k: torch.from_numpy(v).cuda() for k, v in feeds.items()}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            bufs[name] = torch.empty(tuple(engine.get_tensor_shape(name)), dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(name, bufs[name].data_ptr())
    stream = torch.cuda.Stream()

    def call():
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    return call, bufs


def time_calls(fn, iters):
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return statistics.median(ts), ts[int(0.99 * (len(ts) - 1))]


def main():
    device = torch.cuda.get_device_name()
    rows = []
    for precision in ("fp32", "fp16"):
        for b in BATCH_SIZES:
            fx = np.load(FIXTURES / f"batch_{b}.npz")
            feeds = {"y_re": fx["y"].real.astype(np.float32), "y_im": fx["y"].imag.astype(np.float32),
                     "no": fx["no"].astype(np.float32)}
            try:
                engine, build_s = build_engine(ONNX_DIR / f"nrx_b{b}.onnx", precision == "fp16")
            except NotImplementedError as e:
                print(f"{precision} B={b}: skipped ({e})")
                continue
            call, bufs = run(engine, feeds)
            call()
            llr = bufs["llr"].cpu().numpy()
            err = float(np.max(np.abs(llr - fx["llr"])))
            rms = float(np.sqrt(np.mean((llr - fx["llr"]) ** 2)))
            for _ in range(10):
                call()
            t_one, _ = time_calls(call, 1)
            iters = max(20, min(1000, int(2000 / max(t_one, 1e-3))))
            p50, p99 = time_calls(call, iters)
            row = dict(backend="tensorrt", precision=precision, device=device, trt=trt.__version__,
                       torch=torch.__version__, batch=b, build_s=round(build_s, 2), p50_ms=round(p50, 4),
                       p99_ms=round(p99, 4), iters=iters, per_user_ms=round(p50 / b, 4),
                       max_abs_llr_err=float(f"{err:.3e}"), rms_llr_err=float(f"{rms:.3e}"))
            rows.append(row)
            print(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "trt_latency.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nslot budget {SLOT_BUDGET_MS} ms, judged on p99 batch latency")
    for r in rows:
        ok = "meets" if r["p99_ms"] <= SLOT_BUDGET_MS else "misses"
        print(f"  TRT {r['precision']} B={r['batch']:<4} p99 {r['p99_ms']:>9.3f} ms  ({ok})")


if __name__ == "__main__":
    main()
