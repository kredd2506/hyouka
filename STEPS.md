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
| Pre-trained weights | Google Drive `1wjBB3U8Cp4a6VMZSrLBYJl6aG5sPoG_I`, a `torch.save` state_dict (19.4 MB, sha256 `efa5c7ad…7bc9a`), at `weights/reference/neural_receiver_weights`. Fetch: `uv run gdown 1wjBB3U8Cp4a6VMZSrLBYJl6aG5sPoG_I -O weights/reference/neural_receiver_weights` |
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

## 2. Reference run
- [ ] Run the Sionna notebook unchanged (locally, CPU is fine for eval with fewer batches; or on GPU)
- [ ] Export its BER/BLER for: perfect-CSI LMMSE, LS LMMSE, NRX (pre-trained) → `results/reference.csv` (cols: `ebno_db, system, ber, bler, n_bits, n_blocks`)
- [ ] Record the stopping criteria used (max iters / target block errors) — reuse them in step 6/8 for a fair overlay

**Done when:** `reference.csv` exists and the NRX curve hits BLER 1e-4 near ~2.5 dB.

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

## 6. Evaluate ported model
- [ ] `evaluate.py`: sweep −5…10 dB in 0.5 dB steps; JAX LLRs → reference framework → Sionna resource-grid demapper → LDPC decoder → BER/BLER
- [ ] Same stopping criteria as step 2
- [ ] Overlay with `reference.csv` → `results/ber_bler_ported.png`

**Done when:** BLER gap ≤ 0.2 dB (acceptance #2). **Local work ends here.**

## 7. Train from scratch (GPU/TPU host)
- [ ] `loss.py`: `1 − BCE_log2(LLR, bits)` over data REs only (mask pilots/guards/DC — confirm vs 0.2.4)
- [ ] `train.py`: `jax.jit` step, Optax Adam, batch 128, 30k iters, Orbax checkpoint every 1k, log BMD rate + step wall-clock
- [ ] Assert no retracing (count compiles via `jax.log_compiles` or a trace counter) — acceptance #4
- [ ] Short LR sweep {1e-3, 5e-4} for ~2k iters; continue the better one
- [ ] If compute-limited: cut to ~5k iters and report it

**Done when:** BMD-rate curve plateaus; final checkpoint saved.

## 8. Evaluate trained model
- [ ] Run step 6 sweep with the trained checkpoint
- [ ] Plot vs the 3 baselines + ported model → `results/ber_bler.png`, `results/ber_bler.csv`

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
