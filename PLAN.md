# nrx-jax — Step 1: Port the Sionna Neural Receiver to JAX/Flax

## Goal
Reimplement Sionna's OFDM SIMO neural receiver (NRX) in JAX/Flax, train it, and match the PyTorch reference BER/BLER. This is the verified baseline for Step 2 (XLA latency and precision benchmarking on GPU/TPU).

## Reference
Sionna tutorial: `phy/tutorials/notebooks/Neural_Receiver.ipynb` (PyTorch).

### System setup (match exactly)
| Parameter | Value |
|---|---|
| Channel | 3GPP 38.901 CDL-C, 100 ns delay spread, 3.5 GHz, UT speed 10 m/s |
| Link | Uplink SIMO, 1 stream; UT: 1 V-pol antenna; BS: 1×1 dual-pol (V/H) → 2 Rx ports |
| OFDM | 30 kHz SCS, FFT 128, 14 symbols, DC null, guard [5, 6], no CP (freq-domain) |
| Pilots | Kronecker, OFDM symbols 2 and 11 |
| Modulation / FEC | QPSK; 5G NR LDPC, rate 0.5 (n = 2784, k = 1392) |
| Eb/N0 | Train: uniform over [-5, 10] dB. Eval: `np.arange(-5, 10, 0.5)` (10 dB excluded) |

### Architecture
- Input: full post-DFT grid (all 128 subcarriers incl. guards/DC), channel order `[Re0, Re1, Im0, Im1, log10(N0)]` → shape `[B, 14, 128, 5]`
- Conv 3×3 → 128
- 4 × ResBlock: LayerNorm → ReLU → Conv 3×3 → LayerNorm → ReLU → Conv 3×3, + skip
- Conv 3×3 → 2 (bits/symbol) → LLRs `[B, 14, 128, 2]`
- LayerNorm: torch `nn.LayerNorm([C, 14, 128])`, eps 1e-5, per-element scale/offset of shape `[C, 14, 128]` (→ `[14, 128, C]` in NHWC)
- LLRs predicted on every RE; Sionna `ResourceGridDemapper` keeps the 1392 data REs → n = 2784
- 4,858,882 parameters
- Same padding throughout

### Training
- Loss: −BMD rate, rate = 1 − mean(BCE(LLR, c)) / ln 2, over data REs only; no outer code during training
- Adam with torch defaults (lr 1e-3, β 0.9/0.999, eps 1e-8), batch 128, 30k iterations
- Eb/N0: **one** value per batch, uniform in [-5, 10] dB (reference does this; per-example is a possible later variant)
- N0 = `ebnodb2no(ebno, 2, 0.5)` with no resource-grid argument (pilot overhead not accounted for)

### Baselines (from Sionna, not re-implemented)
1. Perfect CSI + LMMSE + APP demapper
2. LS estimation (nearest-neighbor interpolation) + LMMSE + APP demapper
3. Sionna PyTorch NRX (pre-trained weights)

## Design decisions
- **Data generation stays in Sionna.** CDL-C, resource grid, mapper, and LDPC encoder/decoder come from Sionna (PyTorch). Only the receiver moves to JAX. Isolates the port; avoids re-implementing 38.901.
- **Bridge via DLPack/NumPy.** Sionna produces `y`, `no`, coded bits as tensors → `jnp` arrays. JAX LLRs → torch → Sionna LDPC decoder for BLER.
- **Static shapes.** Fixed `[B, 14, 128, ·]` everywhere; prerequisite for Step 2 AOT/HLO work.
- **Weight-transfer test.** Load the Sionna pre-trained PyTorch weights into the Flax model; outputs must match to float tolerance before training from scratch. Proves architecture parity independently of training.

## Repo layout
Package lives at the repo root.
```
./
├── PLAN.md
├── README.md
├── pyproject.toml
├── nrx/
│   ├── config.py        # system parameters (table above)
│   ├── data.py          # Sionna-based batch generator → jnp arrays
│   ├── model.py         # Flax NRX
│   ├── loss.py          # BMD-rate loss
│   ├── train.py         # jitted train step, checkpointing (orbax)
│   ├── port_weights.py  # torch state_dict → Flax params
│   ├── torch_ref.py     # notebook's torch NRX, verbatim, for parity tests
│   └── evaluate.py      # BER/BLER sweep incl. baselines
├── tests/
│   ├── test_shapes.py
│   ├── test_layernorm_parity.py
│   └── test_weight_port.py  # Flax(ported) ≈ Torch outputs
└── results/
    ├── ber_bler.csv
    └── ber_bler.png
```

## Tasks
1. **Env** — Python 3.12 (JAX 0.12 requires it), `sionna`, `torch`, `jax[cuda13]`/`jax[tpu]` extras (CPU by default), `flax`, `optax`, `orbax-checkpoint`. Pin versions in `pyproject.toml`.
2. **Reference run** — Execute the Sionna notebook as-is with pre-trained weights. Save its BER/BLER curves to `results/reference.csv`.
3. **Data bridge** — `data.py` yields `(y_grid, log10_no, coded_bits)` as `jnp` arrays with fixed shapes. Test shapes and dtypes.
4. **Model** — `model.py` in Flax (`nn.Conv`, custom LayerNorm over last 3 axes). Test shapes.
5. **Weight port** — Map torch Conv (OIHW) → Flax (HWIO) and LayerNorm params. Assert `max|LLR_jax − LLR_torch| < 1e-4` on a fixed batch.
6. **Evaluate ported model** — Full BER/BLER sweep through Sionna LDPC decoder. Must overlay the reference curve.
7. **Train from scratch in JAX** — `jax.jit` train step, Optax Adam. Log BMD rate. Checkpoint every 1k steps.
8. **Evaluate trained model** — Sweep; plot vs. 3 baselines.
9. **Write-up** — README: setup, results plot, parity numbers, wall-clock per train step.

## Acceptance criteria
- [x] Ported-weights Flax model matches torch LLRs within 1e-4 (4.96e-5, CPU)
- [x] Ported model's BLER curve overlays Sionna reference (≤ 0.2 dB gap): 0.00/0.07/0.12 dB at 1e-2/1e-3/1e-4 (L4, full sweep; ±~0.15 dB MC noise at 1e-4)
- [x] JAX-trained model reaches BLER 1e-4 within 0.5 dB of the reference NRX: 2.43 dB vs 2.21 dB (0.22 dB), 30k steps on L4
- [x] All shapes static; train step compiles once (no retracing) — asserted in `train.py` and `tests/test_train.py`

## Risks
| Risk | Mitigation |
|---|---|
| No GPU → 30k iters slow | Weight-port path (task 5–6) validates without training; train on Colab/TPU VM or cut to ~5k iters and report it |
| LayerNorm axis/layout mismatch (CHW vs HWC scale/offset) | Dedicated parity test |
| Torch↔JAX transfer overhead dominates | Pre-generate an on-disk dataset (`.npz`) for training |
| Training recipe drift | Match reference exactly first (per-batch Eb/N0, Adam defaults); variants only after |

## Finding: pilots are part of the model
Sionna 2.x draws Kronecker pilots from the active device's torch RNG. The tutorial weights were trained on CUDA with seed 42 and the pilots aren't in the state_dict, so on CPU the reference NRX fails until the pilots are recovered (`nrx/pilots.py`). Exact pilots: regenerate on any CUDA host (STEPS cloud checklist).

## Out of scope (Step 2+)
HLO dumps, latency vs. 0.5 ms slot budget, bf16/int8, TPU runs, StableHLO export, XLA issues/PRs.
