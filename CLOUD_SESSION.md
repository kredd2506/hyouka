# Cloud session log

Log of the cloud runs (Colab L4 sessions 1–3, TPU v5e sessions 4–5). Done on the `cloud-session` branch and merged
into `main` on 2026-11-23; new runs can go straight to `main`.

## Colab
Upload [`notebooks/nrx_colab.ipynb`](notebooks/nrx_colab.ipynb) to Colab (*File → Upload notebook*), pick a GPU runtime
(A100 or L4 preferred), set the options in the first cell, and *Run all*. It needs the code too, in one of two ways:
- **GitHub:** push `main` and `cloud-session`. For a private repo, add a Colab secret `GH_TOKEN`.
- **No push:** upload `dist/hyouka.bundle` (`git bundle create dist/hyouka.bundle main cloud-session`) to `/content`.

The notebook handles the details: it picks CUDA 13 or 12 builds from the driver version, keeps checkpoints on Drive
(resumes after disconnects), runs TensorRT through its Python API, and ends with a summary and a results zip.

## On a plain GPU host
```bash
git clone -b cloud-session https://github.com/kredd2506/hyouka.git && cd hyouka
make setup EXTRA=cuda        # or EXTRA=tpu
make host-info               # -> results/host_<hostname>.txt
make all-cloud               # or the targets one by one, below
git add results checkpoints/run1/train_log.csv assets && git commit -m "Cloud results: <host>" && git push
```
Long runs: use `tmux` / `nohup`; `make train` resumes from the latest checkpoint if interrupted.

## Runs
Fill in as you go. Times are wall clock.

| # | Target | Host | Started | Took | Result / notes |
|---|---|---|---|---|---|
| 0 | `make host-info` | Colab L4 | 2026-11-21 12:21 UTC | | NVIDIA L4 24 GB, driver 580.82, torch 2.15.0+cu132, jax 0.12.1 CUDA |
| 1 | `make test` | L4 | | | 11 passed |
| 2 | `make pilots-cuda` | L4 | | | 20/232 differ from the polished recovery; **0/232 from gradient-only recovery** |
| 3 | `make backends` | L4 | | | B=1 p50 0.78 / p99 0.90 ms; B=128 44.7 ms. LLR err 6e-3–2e-2 → **TF32** convs |
| 4 | `make compiler` | L4 | | | 60 kernels (49 kInput, 9 kCustom, 2 kLoop); 10 cuDNN convBiasActivation (bias fused). Isolated-op timing invalid on GPU (dispatch overhead) |
| 5 | `make trt` | L4 | | | **not run**: `import tensorrt` failed after `uv pip install -q tensorrt-cu13` (output hidden by -q) |
| 6 | `make sweep` | L4 | | | NRX @1e-4 = 2.33 dB (notebook 2.21; CUDA pilots 2.41); jax ≡ torch |
| 7 | `make train` | L4 | | | **5k of 30k steps**: rate 0.76, 159 ms/step, Sionna wait 0.02 ms, traces = 1 |
| 8 | `make eval-trained` | L4 | | | 5k-step model @1e-2 = 1.51 dB (ref −1.03): undertrained |

### Session 2 (Colab L4, 2026-11-21 17:32 UTC, commit f0504d2)

| Target | Result / notes |
|---|---|
| `make test` | 10 passed, **weight-port parity failed: 1.0e-2** (JAX ran on GPU with TF32). Fixed: test forces `highest` precision |
| `make pilots-cuda` | **0/232** differ: recovery is exact |
| `make backends` default (TF32) | p99 B=1 **0.875 ms** (dLLR 1.8e-5), B=8 2.01 ms (9.1e-3), B=128 44.8 ms (1.6e-2) |
| `make backends` highest (fp32) | p99 B=1 **1.013 ms** (5.1e-5), B=8 3.05 ms (2.5e-5), B=128 71.8 ms (5.7e-5) |
| `make compiler` B=1 | 60 kernels both precisions (default 25 kCustom, highest 16). Ablation (median, TF32): full 0.797, −LN 0.689, −res convs 0.340, in/out conv + glue 0.234, dispatch floor 0.088 ms → res convs 57%, LN 13%, dispatch 11%. fp32: 0.802 ms, convs 59%, LN 11% |
| `make compiler` B=128 | 85 kernels. TF32: 43.6 ms, convs 71%, LN 23%. fp32: 67.8 ms, convs 81%, LN 11% |
| `make onnx` | ORT (x86) vs torch: max 1.2e-4–2.9e-4, rms 3–4e-5 |
| `make trt` | **not run**: `tensorrt_bindings` not importable after install, so uv installed into a different env. Fixed: `uv pip install --python .venv/bin/python` |
| `make train` | resumed at **28000** → 30000, rate ≈ 0.79, 158 ms/step, traces 1. `train_log.csv` on Drive has only 7000 rows (steps 5k–28k missing) |
| `make eval-trained` | **not run**: the kernel had restarted (`NameError: ACCEL`); summary/zip cells not run either |

### Session 3 (Colab L4, 2026-11-22 01:24 UTC, commit cf17d50)

| Target | Result / notes |
|---|---|
| `make test` | 11 passed on GPU |
| `make backends` | p99 B=1/8/128: TF32 0.869 / 2.07 / 44.2 ms, fp32 0.867 / 3.01 / 67.9 ms |
| `make trt` | TensorRT 11.5.0.63. p99 fp32 0.547 / 2.99 / 62.7 ms, TF32 0.553 / 1.85 / 53.3 ms. fp16 skipped: no FP16 builder flag in TRT 11 |
| `make train` | already at 30000, skipped |
| `make eval-trained` | jax-trained @1e-2/1e-3/1e-4 = −1.01 / 0.23 / 2.43 dB (notebook −1.03 / 0.14 / 2.21) → **acceptance #3 met** |

### Session 4 (Colab TPU v5e-1, 2026-11-23 02:10 UTC, commit 8a29fa9)
An earlier attempt (b88506f) ran on the host CPU: the notebook looked for `/dev/accel0`, which v5e doesn't have.

| Target | Result / notes |
|---|---|
| `make test` | 9 passed, 2 failed: log10 channel off in the last bits; weight-port 1.1e-3 at `highest` (fp32 emulated with bf16 passes). Fixed: TPU tolerances |
| `make backends` | p99 B=1/8/128: bf16 0.379 / 0.590 / 9.15 ms (LLR err 0.21–0.25); bf16×3 0.514 / 1.02 / 14.1 ms; fp32 0.638 / 1.68 / 24.6 ms |
| `make compiler` | B=1 ablation, bf16: full 0.342, dispatch floor 0.203 ms (59%). fp32: full 0.511, floor 0.195. The report and B=128 failed: the pass-dump child process can't open the TPU while the parent holds it. Fixed: dump first |
| HLO | 113 top-level ops; 10 conv+LayerNorm-reduction `kOutput` fusions. The op counter skipped tuple-shaped ops (fixed); with the fix the L4 graph has 70 ops, not 60 (10 cuDNN conv calls were missed) |
| `make train` 300 steps | 24.2 ms/step, Sionna wait 12.1 ms/step (data-bound). Process aborted (exit 134) after the last step: prefetch thread still in torch at exit. Fixed: thread joined |

### Session 5 (Colab TPU v5e-1, 2026-11-23 02:33 UTC, commit 1d712b4)

| Target | Result / notes |
|---|---|
| `make test` | 10 passed; log10 channel off by up to 2.9e-5 absolute on TPU (fixed in 97844c1, after this run started) |
| `make backends` | p99 B=1/8/128: bf16 0.396 / 0.584 / 9.04 ms; bf16×3 0.541 / 1.11 / 14.1 ms; fp32 0.559 / 1.79 / 24.6 ms |
| `make compiler` | All six reports written. B=1: 113 ops; B=128: 39–40 ops. Pass dump 604 runs / 75 passes / 42 pipelines. B=128 ablation non-additive (removing convs → slower) |
| `make train` | resumed at 300, no steps, exited cleanly (no exit 134) |
| TPU sweep (40 MC iters) | bf16 −1.04 / −0.01 dB, fp32 −1.03 / −0.01 dB at 1e-2 / 1e-3: **bf16 costs no BLER** |

### Next session (notebook defaults are set for this)
- [ ] TensorRT: `make trt-install` (libs + bindings wheels from pypi.nvidia.com; the PyPI `tensorrt` sdist failed), `make trt`
- [ ] Precision: `make backends PRECISION=highest` (true fp32) next to default (TF32); TRT fp32 / tf32 / fp16
- [ ] GPU time breakdown: `make compiler` now ablates inside the full graph (isolated ops were dominated by dispatch)
- [ ] Training: resumes the 5k checkpoint on Drive to 30k (~70 min on L4); `make eval-trained`

## Bring back
- [ ] `results/` (latency jsonl, compiler md, sweeps csv/png, host info)
- [ ] `checkpoints/run1/train_log.csv` (+ final checkpoint somewhere outside git, ~60 MB)
- [ ] `assets/pilots_cuda_seed42.npy`
- [ ] `results/trt_latency.jsonl`
