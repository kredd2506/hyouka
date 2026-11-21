# Cloud session log

Branch `cloud-session`: run the checklist in [STEPS.md](STEPS.md#cloud-session-checklist-one-gpu-box-ideally--one-tpu-vm),
commit results here, merge back to `main` when done.

## On the host
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
| 5 | `make onnx` + `trtexec` fp32/fp16 | | | | TRT B=1 = __ ms |
| 6 | `make sweep` | | | | NRX @1e-4 = __ dB (notebook 2.21) |
| 7 | `make train` | | | | final rate = __, __ ms/step, traces = 1 |
| 8 | `make eval-trained` | | | | trained @1e-4 = __ dB |

## Bring back
- [ ] `results/` (latency jsonl, compiler md, sweeps csv/png, host info)
- [ ] `checkpoints/run1/train_log.csv` (+ final checkpoint somewhere outside git, ~60 MB)
- [ ] `assets/pilots_cuda_seed42.npy`
- [ ] TensorRT `trtexec` outputs → `results/trt/`
