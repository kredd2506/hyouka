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
| 0 | `make host-info` | | | | |
| 1 | `make test` | | | | |
| 2 | `make pilots-cuda` | | | | __/232 pilots differ |
| 3 | `make backends` | | | | B=1 p99 = __ ms |
| 4 | `make compiler` | | | | kernels = __, conv share = __% |
| 5 | `make trt` (after `uv pip install tensorrt-cu13`) | | | | TRT B=1 = __ ms |
| 6 | `make sweep` | | | | NRX @1e-4 = __ dB (notebook 2.21) |
| 7 | `make train` | | | | final rate = __, __ ms/step, traces = 1 |
| 8 | `make eval-trained` | | | | trained @1e-4 = __ dB |

## Bring back
- [ ] `results/` (latency jsonl, compiler md, sweeps csv/png, host info)
- [ ] `checkpoints/run1/train_log.csv` (+ final checkpoint somewhere outside git, ~60 MB)
- [ ] `assets/pilots_cuda_seed42.npy`
- [ ] `results/trt_latency.jsonl`
