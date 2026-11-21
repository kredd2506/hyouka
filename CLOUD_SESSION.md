# Cloud session log

Branch `cloud-session`: run the checklist in [STEPS.md](STEPS.md#cloud-session-checklist-one-gpu-box-ideally--one-tpu-vm),
commit results here, merge back to `main` when done.

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
