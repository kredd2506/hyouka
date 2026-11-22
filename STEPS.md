# nrx-jax — Step 1 execution checklist

Companion to [PLAN.md](PLAN.md). PLAN says *what*; this says *in what order, with what, and how you know each step is done*.

---

## 0. Prerequisites

### 0.1 Machine / accounts
| Item | Status on this machine | Action |
|---|---|---|
| Local box | Apple M4 Pro, 24 GB, **no CUDA** | Fine for steps 1–6 (CPU JAX). Training (step 7) goes elsewhere. |
| Python 3.12 | Installed via uv (JAX 0.12 needs ≥ 3.12) | pinned in `.python-version` |
| `uv` | Installed | Use for venv + lockfile |
| GPU for training | — | Pick one: Colab (T4/L4/A100), a cloud GPU VM, or TPU VM (TRC). Decide before step 7. |
| Disk | — | ~5–10 GB for a pre-generated `.npz` training set (see 0.4) |

### 0.2 Facts confirmed from the notebook (Sionna 2.2.0, PyTorch) ✅
| Question | Answer |
|---|---|
| Framework | PyTorch (`sionna.phy`). Conv weights are `OIHW` → transpose to `HWIO`. |
| Pre-trained weights | Google Drive `1wjBB3U8Cp4a6VMZSrLBYJl6aG5sPoG_I`, a `torch.save` state_dict (19.4 MB, sha256 `efa5c7ad…7bc9a`), at `weights/reference/neural_receiver_weights`. Fetch: `make weights` |
| LayerNorm | `nn.LayerNorm([128, 14, 128])`, eps 1e-5, scale/offset `[C,14,128]` → transpose to `[14,128,C]` |
| Receiver I/O | Full 128-subcarrier grid in; LLRs on all REs; `ResourceGridDemapper` keeps 1392 data REs (n = 2784) |
| Input channel order | `[Re0, Re1, Im0, Im1, log10(N0)]`, not interleaved |
| Training | Adam torch defaults (lr 1e-3), batch 128, 30k iters, **one Eb/N0 per batch** |
| Eb/N0 → N0 | `ebnodb2no(ebno, 2, 0.5)`, no resource grid passed |
| Eval | `sim_ber(batch_size=128, num_target_block_errors=100, max_mc_iter=1000, target_bler=1e-4)` over `np.arange(-5, 10, 0.5)` |
| Reference NRX BLER | 1.4e-4 @ 2.0 dB, 6.3e-5 @ 2.5 dB (stored notebook output); note a ~1e-3 floor from −0.5 to 0.5 dB |

### 0.3 Repo decisions
- Code lives at the repo root (package `nrx/`).
- `.gitignore` added (`.venv/`, `data/`, `checkpoints/`, `weights/`, `results/*.npz`).
- Seeds: one global seed in `config.py`; separate seeds for data-gen, init, and eval.

### 0.4 Dataset strategy (decide before step 3)
Sionna's CDL-C generation is the likely bottleneck, not the NRX. Plan for:
- **Online** generation for correctness tests and eval (small).
- **Offline** `.npz` shards for training: e.g. 30k × 128 examples is too large to store raw (`3.84M × 14×128×2 complex64` ≈ 110 GB, before bits). Options: store ~200–500k examples and reuse across epochs, or generate on the training host in a background process. Pick and note it.

---

## 1. Environment
- [x] `pyproject.toml`: sionna 2.2.0, torch 2.15.0, jax 0.12.1, flax 0.12.12, optax 0.2.9, orbax-checkpoint 0.12.12
- [x] Extras: default CPU; `--extra cuda` → `jax[cuda13]` (matches torch's CUDA 13 on Linux); `--extra tpu`
- [x] `uv.lock` generated
- [x] Smoke: imports OK, `jax.devices() == [CpuDevice]`, torch MPS available
- [x] Pre-trained weights downloaded and inspected (4,858,882 params)
- Note: `jitc_llvm_init(): LLVM API initialization failed` on import is from sionna-rt's Dr.Jit, harmless for `sionna.phy`

**Done when:** fresh clone → `uv sync` → smoke passes.

## 2. Reference run ✅ (from the notebook's stored outputs)
- [x] `results/reference.csv`: BER/BLER of perfect-CSI, LS, and NRX parsed from the notebook's executed cells
- [x] Stopping criteria: `batch_size=128, num_target_block_errors=100, max_mc_iter=1000, target_bler=1e-4`
- Notebook Eb/N0 at BLER 1e-4: perfect CSI 0.55 dB, NRX 2.21 dB, LS 5.05 dB

## 3. Config + data bridge ✅
- [x] `nrx/config.py`: frozen `Config` with every notebook parameter
- [x] `nrx/data.py`: `SionnaLink.sample(B, ebno_db)` → `y [B,2,14,128] complex64`, `no [B]`, `c [B,2784]` (+ `b` when `coded=True`)
- [x] `data_re_indices()` + `gather_data_llrs()`: static JAX gather equal to Sionna's `ResourceGridDemapper` (bit-exact test)
- [x] `nrx/model.py::make_inputs` builds `[B,14,128,5]` in `[Re0, Re1, Im0, Im1, log10 N0]` order (checked against torch layout)
- Data moves via `.numpy()`, not DLPack (complex DLPack is not reliable across torch/JAX)

## 4. Flax model ✅
- [x] `nrx/model.py::NeuralReceiver` (flax.linen), param names mirror torch (`input_conv`, `res_block_i/{layer_norm,conv}_{1,2}`, `output_conv`)
- [x] `GridLayerNorm`: scale/offset `[14,128,C]`, eps 1e-5, biased variance; parity vs `torch.nn.LayerNorm` < 1e-5
- [x] Param count 4,858,882 = torch

## 5. Weight port ✅
- [x] `nrx/port_weights.py`: OIHW→HWIO for conv, `[C,T,F]`→`[T,F,C]` for LayerNorm
- [x] `tests/test_weight_port.py`: real CDL-C batch over Eb/N0 −5…10 dB, **max|ΔLLR| = 4.96e-5** (max |LLR| 42.8) on CPU — acceptance #1 met
- `nrx/torch_ref.py` holds the notebook's torch model for the parity test

## 6. Evaluate ported model — reduced run ✅, full run on cloud
- [x] `nrx/evaluate.py`: `sim_ber` sweep for `jax-ported`, `torch-ref`, both baselines, `jax-trained`; same seed per system → same channel draws
- [x] **Pilots fix**: Sionna 2.x draws Kronecker pilots from the device RNG; the weights were trained on CUDA (seed 42), so CPU pilots don't match and the NRX fails (BER ≈ 0.44). `nrx/pilots.py` recovers them from the weights by gradient ascent on the BMD rate (y is linear in pilots), snaps to QPSK, greedy polish → `assets/pilots_recovered.npy` (tracked). `make_resource_grid()` applies it; `NRX_PILOTS=path` overrides.
- [x] Reduced sweep (`max_mc_iter=40`): `results/ber_bler_ported_reduced.{csv,png}`. jax-ported ≡ torch-ref. vs notebook NRX at BLER 1e-1/1e-2/1e-3: −1.49/−1.05/−0.01 dB vs −1.53/−1.03/+0.14 dB
- [x] Full sweep at `max_mc_iter=1000` (Colab L4): jax-ported ≡ torch-ref, 2.33 dB @1e-4 vs notebook 2.21 → acceptance #2 met

## 7. Train from scratch — code ✅, run on cloud
- [x] `nrx/loss.py`: BMD rate over data REs (static gather)
- [x] `nrx/train.py`: jitted step, Adam (torch defaults), one Eb/N0 per batch, Sionna in a prefetch thread, Orbax every 1k (resumable), `train_log.csv` with step/wait ms, asserts **one trace**
- [x] CPU smoke: 40 steps at B=16, rate −3.0 → −0.03, 1 trace, resume + `load_checkpoint` OK; `tests/test_train.py`
- [x] 30k steps at B=128 on L4: final rate ≈ 0.79, 158 ms/step, data wait ≈ 0, one trace (log for steps 5k–28k lost on Drive)

## 8. Evaluate trained model
- [x] Sweep with the trained checkpoint → `results/ber_bler_trained.{csv,png}`
- [x] BLER 1e-2 / 1e-3 / 1e-4 at −1.01 / 0.23 / 2.43 dB vs notebook −1.03 / 0.14 / 2.21 → acceptance #3 met

**Done when:** BLER 1e-4 within 0.5 dB of reference NRX (acceptance #3).

## 9. Write-up
- [ ] README: setup, how to reproduce each step, results plot, parity numbers, ms/train-step and hardware, any deviations (iters, LR, dataset size)
- [ ] Tick acceptance boxes in PLAN.md

---

## Dependency order
```
0 → 1 → 2 ─┐
       └→ 3 → 4 → 5 → 6 ─→ 7 → 8 → 9
```
Steps 2 and 3–4 can run in parallel. Step 7 needs a GPU host and the dataset decision from 0.4.

---

## Cloud session checklist (one GPU box, ideally + one TPU VM)
Everything below is scripted; nothing needs writing on the host.

**Setup**
- [ ] `git clone` + `uv sync --extra cuda` (or `--extra tpu`)
- [ ] `make weights` (downloads + verifies sha256); recovered pilots are tracked in `assets/`
- [ ] `uv run pytest -q` passes on the host

**1. Exact pilots (CUDA only)** — the tutorial weights were trained with pilots from the CUDA RNG at seed 42
- [ ] `make pilots-cuda` (writes `assets/pilots_cuda_seed42.npy` and prints how many differ from the recovered ones)
- [ ] If they differ: `NRX_PILOTS=assets/pilots_cuda_seed42.npy make sweep` and compare

**2. Backends (goal 1)**
- [ ] `python bench/backends.py --make-fixture && python bench/backends.py` on GPU and TPU
- [x] `python bench/inspect_compiler.py --batch 1 --dump-passes` on GPU and TPU (GPU done on L4)
- [ ] TPU: `for p in default high highest; do make backends compiler PRECISION=$p; done` and `make train RUN=checkpoints/run_tpu STEPS=300` (Colab TPU runtime or TPU VM; see README)

**3. Full BLER sweep (step 6, acceptance #2)** — notebook settings (`max_mc_iter=1000`)
- [ ] `python -m nrx.evaluate --systems jax-ported torch-ref baseline-perfect-csi baseline-ls-estimation --out results/ber_bler_ported.csv`

**4. Training (step 7, acceptance #3–4)**
- [ ] `python -m nrx.train --steps 30000 --ckpt-dir checkpoints/run1` (resumable; logs `train_log.csv`; asserts one trace)
- [ ] `python -m nrx.evaluate --systems jax-trained --checkpoint checkpoints/run1 --out results/ber_bler_trained.csv`

**5. TensorRT (goal 2)** — GPU only
- [ ] `python bench/export_onnx.py` (or copy `results/onnx/`)
- [ ] `trtexec --onnx=results/onnx/nrx_b1.onnx` fp32 and `--fp16`; same for B=8, 128. Record p50/p99 GPU compute time
- [ ] Same GPU: `bench/backends.py` p50/p99 for XLA → matched table

**6. Bring back** `results/`, `checkpoints/run1/train_log.csv`, final checkpoint
