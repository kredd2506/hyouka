# nrx-jax

Sionna's 5G neural receiver (NRX), ported from PyTorch to JAX/Flax, verified against the original, and
benchmarked across XLA backends.

**Why.** Nvidia runs neural receivers with TensorRT, which only targets Nvidia GPUs. A JAX version compiles through
XLA, so the same receiver can run on GPU, TPU, or any accelerator with a PJRT plugin. This repo is the verified
starting point for measuring whether neural PHY models can meet 5G slot deadlines (0.5 ms at 30 kHz SCS) outside
Nvidia's stack. Holding the model, weights and channel fixed means the compiler stack is the only thing that changes.

| | |
|---|---|
| Model | Sionna tutorial NRX: SIMO uplink, 4 residual conv blocks, per-element LayerNorm, 4.86 M params |
| Channel | 3GPP 38.901 CDL-C, 3.5 GHz, 30 kHz SCS, 128 subcarriers × 14 symbols, QPSK, LDPC rate 1/2 |
| Reference | [`tutorials/phy/Neural_Receiver.ipynb`](https://github.com/NVlabs/sionna/blob/main/tutorials/phy/Neural_Receiver.ipynb), Sionna 2.2.0 |

Details: [PLAN.md](PLAN.md) (spec) · [STEPS.md](STEPS.md) (checklist and status).

---

## Results so far

**1. The port matches PyTorch.** With Sionna's pre-trained weights loaded, JAX and torch LLRs agree to
**4.96e-5** (max |LLR| ≈ 43). Both are ~3e-5 from an fp64 reference.

**2. The port matches the published error rate.** A reduced Monte-Carlo sweep (5,120 blocks per point) gives
the same BLER for the JAX and torch receivers, and both sit within 0.15 dB of the notebook:

| BLER | notebook NRX | JAX port |
|---|---|---|
| 1e-1 | −1.53 dB | −1.49 dB |
| 1e-2 | −1.03 dB | −1.05 dB |
| 1e-3 | +0.14 dB | −0.01 dB |

![BLER](results/ber_bler_ported_reduced.png)

**Full sweep (Colab L4, notebook settings).** JAX and torch are again identical. At BLER 1e-2 / 1e-3 / 1e-4 the
port sits at −1.03 / 0.21 / 2.33 dB against the notebook's −1.03 / 0.14 / 2.21 dB, a gap of 0.00 / 0.07 / 0.12 dB.
At 1e-4 each point rests on ~13 block errors, so that last number carries roughly ±0.15 dB of Monte-Carlo noise.
See `results/ber_bler_ported.{csv,png}`.

**3. The same code runs on more than one backend.** Apple M4 Pro, B = 1 (one user per slot):

| Backend | p50 | p99 | max \|ΔLLR\| vs torch |
|---|---|---|---|
| XLA:CPU (jax 0.12.1) | 3.04 ms | 3.17 ms | 4.96e-5 |
| Metal GPU via `jax-metal` PJRT plugin (jax 0.4.34) | 1.25 ms | 1.34 ms | 3.72e-5 |
| XLA:GPU on NVIDIA L4 (jax 0.12.1, CUDA 13) | 0.81 ms | 0.87 ms | 2.1e-5 |
| XLA:TPU v5e, fp32 (`highest`) (jax 0.12.1, Colab v5e-1) | 0.51 ms | 0.64 ms | 3.4e-5 |

The model costs 4.05 GFLOP per user per slot, so meeting 0.5 ms at B = 1 needs ≥ 8.1 TFLOP/s sustained.

**4. XLA vs TensorRT on the same GPU (NVIDIA L4).** Same ONNX-exported weights, same fixtures, p99 batch latency:

| B | TensorRT 11.5 fp32 | XLA fp32 (`highest`) | TensorRT TF32 | XLA TF32 (default) |
|---|---|---|---|---|
| 1 | **0.547 ms** | 0.867 ms | 0.553 ms | 0.869 ms |
| 8 | 2.99 ms | 3.01 ms | **1.85 ms** | 2.07 ms |
| 128 | 62.7 ms | 67.9 ms | 53.3 ms | **44.2 ms** |

At one user per slot TensorRT is 1.6× faster and only just misses the 0.5 ms deadline; XLA is 0.32 ms behind. The
gap closes by B = 8, and at B = 128 XLA in TF32 is 17% faster than TensorRT. Neither stack meets 0.5 ms on an L4.
At B = 1 both pick fp32-accurate conv kernels even when TF32 is allowed (LLR error ~2e-5 either way); TF32 only
pays from B = 8 up, at ~1e-2 LLR error. TensorRT 11 dropped the FP16 builder flag (strongly-typed networks only),
so fp16 needs an fp16 ONNX graph and isn't measured yet. Data: `results/{backend,trt}_latency.jsonl`.

Where XLA's B = 1 time goes, by ablation inside the compiled graph: the 8 residual convs ≈ 58%, the 8 LayerNorms
≈ 11%, per-call dispatch ≈ 11% (0.085 ms), and the input/output convs plus glue alone ≈ 0.22 ms
([results/compiler_gpu_b1.md](results/compiler_gpu_b1.md)).

**5. TPU v5e: the only configuration that meets 0.5 ms, at bf16.** Same code and fixtures on a Colab v5e-1, p99:

| B | TPU bf16 (default) | TPU bf16×3 (`high`) | TPU fp32 (`highest`) | L4 TensorRT TF32 | L4 XLA TF32 |
|---|---|---|---|---|---|
| 1 | **0.379 ms** | 0.514 ms | 0.638 ms | 0.553 ms | 0.869 ms |
| 8 | 0.590 ms | 1.02 ms | 1.68 ms | 1.85 ms | 2.07 ms |
| 128 | **9.15 ms** | 14.1 ms | 24.6 ms | 53.3 ms | 44.2 ms |
| max \|ΔLLR\| | 0.21–0.25 | 8e-4–1.5e-3 | 3e-5–1.3e-3 | 5e-5–1.6e-2 | 2e-5–1.6e-2 |

- XLA:TPU's default precision runs the convs in a single bf16 pass. That meets the deadline at B = 1, but the LLRs
  move by ~0.2 (max |LLR| ≈ 33). Whether that costs BLER is the next measurement (notebook section 8, TPU).
- At fp32-level accuracy the TPU loses to TensorRT at B = 1 (0.64 vs 0.55 ms) and wins 2.5× at B = 128 (24.6 vs
  62.7 ms TensorRT fp32).
- B = 1 on the TPU is overhead-bound. A trivial jitted call already takes ~0.2 ms (vs 0.085 ms on the L4), out of a
  0.34 ms median, so the network itself takes ~0.14 ms, and the 8 residual convs only ~0.05 ms of that in bf16.
- XLA:TPU compiles the graph differently from XLA:GPU. It fuses each conv with the following LayerNorm's mean and
  variance reduction (10 `kOutput` fusions), prefetches the weights into on-chip memory asynchronously
  (`copy-start`/`copy-done`), keeps activations in bf16 and LayerNorm statistics in fp32, and splits the complex input
  with `X64SplitHigh/Low` custom calls ([results/hlo/tpu/](results/hlo/tpu/)).
- Training runs at 24 ms/step on the TPU (bf16) vs 158 ms on the L4, but Sionna generates batches on the host CPU at
  ~36 ms each, so a TPU step waits ~12 ms for data. Feeding the TPU needs the channel model on the accelerator too.

**6. Training from scratch in JAX reproduces the receiver.** 30k Adam steps at B = 128 on the L4 (158 ms/step, Sionna
data generation fully hidden behind the step). The JAX-trained model reaches BLER 1e-2 / 1e-3 / 1e-4 at
−1.01 / 0.23 / 2.43 dB against the notebook's −1.03 / 0.14 / 2.21 dB: within 0.22 dB
([results/ber_bler_trained.png](results/ber_bler_trained.png)).

**7. A look inside the compiler (XLA:CPU, B = 1).** 186 jaxpr equations become 305 StableHLO ops and then
62 kernels. Convs take ~75% of the time and LayerNorm ~20%. Each LayerNorm becomes 5 kernels, and the conv bias add
isn't fused into the conv. See [results/compiler_cpu_b1.md](results/compiler_cpu_b1.md).

### Finding: the pilots are part of the model
Sionna 2.x draws the Kronecker pilot symbols from the active device's torch RNG when the resource grid is built.
The tutorial weights were trained on CUDA with seed 42, and the pilots are **not** stored in the checkpoint.
On any other device the pilots differ, and the pre-trained receiver fails completely (uncoded BER ≈ 0.44). The
network-level parity test can't catch this, because both models get the same wrong pilots.

[`nrx/pilots.py`](nrx/pilots.py) recovers them from the weights. The received grid is linear in the pilots, so it
maximizes the receiver's BMD rate over the 232 pilot values with JAX gradients and snaps the result to QPSK. The
result matches all 232 pilots regenerated from the CUDA RNG on a Colab L4. Both sets are checked in
([`assets/`](assets/)), and the CUDA set is applied automatically.

---

## Reproduce

### Requirements
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.12 itself)
- macOS (arm64) or Linux x86-64; ~3 GB disk for the venv
- Optional: an NVIDIA GPU (CUDA 13 driver) or a Cloud TPU VM, for training, full sweeps, and real latency numbers

### Laptop (CPU), about 2 minutes
```bash
git clone https://github.com/kredd2506/hyouka.git && cd hyouka
make setup          # uv sync: exact pinned versions from uv.lock
make test           # downloads + checksums the weights, runs 11 tests (~5 s)
make all-local      # backends + compiler walk + ONNX export (~80 s)
```
Run `make help` for every target. Each one is a single `uv run python …` command, so you can also run them by hand.

| Target | What it does | Time (M4 Pro CPU) |
|---|---|---|
| `make weights` | Downloads Sionna's pre-trained weights (Google Drive, 19 MB) and verifies sha256 | seconds |
| `make test` | Shapes, LayerNorm parity, data-RE gather vs Sionna, weight-port parity, pilots, one-trace train step | ~5 s |
| `make backends` | Compile + p50/p99 latency + LLR parity for B ∈ {1, 8, 128} on the current backend | ~1 min |
| `make compiler` | jaxpr → StableHLO → HLO walk, per-op timing, XLA pass dump to `data/xla_dump/` | ~2 min |
| `make onnx` | Exports the torch NRX to ONNX (real-valued inputs) and checks it with onnxruntime | ~1 min |
| `make sweep-reduced` | BLER sweep, 40 MC iterations, 4 systems | ~30 min |
| `make pilots-recover` | Re-derives the training pilots from the weights | ~2 min |

### Colab (GPU or TPU), no setup
Open [`notebooks/nrx_colab.ipynb`](notebooks/nrx_colab.ipynb) in Colab
([direct link](https://colab.research.google.com/github/kredd2506/hyouka/blob/cloud-session/notebooks/nrx_colab.ipynb)),
pick an **L4/A100 GPU** or a **v5e-1/v6e-1 TPU** runtime, and use *Runtime → Run all*. The notebook detects the
accelerator, installs the matching build, runs the sections that apply, keeps checkpoints on Google Drive, and ends
with a summary table and a results zip.

### GPU host (CUDA)
```bash
make setup EXTRA=cuda
make all-cloud       # test, exact pilots, backends, compiler, ONNX, full sweep, 30k-step training, trained sweep
```
Or step by step. [STEPS.md → Cloud session checklist](STEPS.md#cloud-session-checklist-one-gpu-box-ideally--one-tpu-vm)
lists the order and what to bring back.

TensorRT comparison (same GPU):
```bash
make trt-install                                       # TRT_CUDA=cu12 on drivers < 580
make trt                                               # ONNX export + TensorRT fp32 / tf32
make backends PRECISION=highest && make backends       # XLA fp32 and TF32 on the same GPU
```

Notes:
- The Makefile sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Otherwise JAX reserves 75% of GPU memory and Sionna's
  torch (which generates the channel on the same GPU) runs out.
- Training resumes from the latest checkpoint in `RUN` (default `checkpoints/run1`). Change length with
  `make train STEPS=5000`.

### TPU
TensorRT, the exact-pilot step and the sweeps need a GPU. On a TPU the useful runs are XLA:TPU latency and the
compiler walk, in three precisions, plus a short training run:

| `PRECISION` | What XLA:TPU does for f32 convs | Compare with |
|---|---|---|
| `default` | one bf16 pass (fastest, least accurate; ~0.2 LLR error here) | GPU TF32 |
| `high` | bf16×3, close to fp32 | |
| `highest` | full fp32 emulation (~6 bf16 passes) | GPU fp32 |

On Colab, pick a TPU runtime and *Run all* (see above). On a Cloud TPU VM:
```bash
gcloud compute tpus tpu-vm create nrx --zone=us-central1-a \
    --accelerator-type=v5litepod-1 --version=v2-alpha-tpuv5-lite
gcloud compute tpus tpu-vm ssh nrx --zone=us-central1-a
# on the VM:
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone -b cloud-session https://github.com/kredd2506/hyouka.git && cd hyouka
make setup EXTRA=tpu
make host-info test
for p in default high highest; do make backends compiler PRECISION=$p; done
make train RUN=checkpoints/run_tpu STEPS=300     # Sionna runs on the host CPU: compare wait_ms with step_ms
```
Delete the VM afterwards (`gcloud compute tpus tpu-vm delete nrx --zone=us-central1-a`), since it bills while it
exists. Pilots come from `assets/`, so the pre-trained weights work on a TPU host even though its torch RNG differs.


### Apple GPU (Metal), optional
`jax-metal` was last released for jax 0.4.34 and can't load into this project's env, so it gets a separate venv:
```bash
make fixtures                                    # in the main env
uv venv -p 3.12 .venv-metal
VIRTUAL_ENV=.venv-metal uv pip install "jax==0.4.34" "jaxlib==0.4.34" "jax-metal==0.1.1" "flax==0.10.0" numpy
.venv-metal/bin/python bench/backends.py --real-input --no-dump-hlo
```
Metal rejects `complex64` inputs, so `--real-input` splits Re/Im on the host.

### Determinism
- Versions: `uv.lock` pins everything. `make setup` installs exactly those versions.
- Weights: sha256-checked (`nrx/config.py`).
- Pilots: checked in. Set `NRX_PILOTS=path.npy` to use another set.
- Monte-Carlo: `sionna.phy.config.seed` is reset per system, so every receiver sees the same channel draws.
  Results still vary with device (CPU vs CUDA RNG) and with batch size.

---

## Layout
```
nrx/
  config.py         system parameters (from the notebook) and paths
  data.py           Sionna transmitter + CDL channel -> JAX arrays; static data-RE gather
  model.py          Flax NRX (NHWC) and GridLayerNorm
  torch_ref.py      the notebook's torch NRX, verbatim, for parity
  port_weights.py   torch state_dict -> Flax params (OIHW->HWIO, LN [C,T,F]->[T,F,C])
  pilots.py         recover / regenerate / compare training pilots
  loss.py           BMD rate
  train.py          jitted train step, Optax Adam, Orbax checkpoints, single-trace assert
  evaluate.py       BER/BLER sweeps through Sionna's LDPC decoder; plots vs the notebook
  fetch_weights.py  download + checksum
bench/
  backends.py       latency + parity on any XLA backend
  inspect_compiler.py  compiler-stack walk and ablation time breakdown
  export_onnx.py    ONNX export for TensorRT
  trt_bench.py      TensorRT engines (fp32 / TF32) timed like backends.py
notebooks/
  nrx_colab.ipynb   one-click Colab run on GPU or TPU
tests/              pytest suite (make test)
assets/             recovered training pilots
results/            reference.csv (from the notebook), sweeps, latency, compiler reports
```
Generated and ignored: `weights/`, `data/` (fixtures, XLA dumps), `checkpoints/`, `results/onnx/`.

## Status
- [x] Port + weight parity (< 1e-4)
- [x] Reduced BLER overlay (≤ 0.15 dB down to 1e-3)
- [x] Training loop compiles once (verified on CPU)
- [x] Full BLER sweep to 1e-4 (L4): 0.12 dB from the notebook
- [x] JAX-trained model within 0.5 dB of reference at BLER 1e-4 (L4: 0.22 dB)
- [x] XLA vs TensorRT on the same GPU (L4, fp32 and TF32)
- [x] XLA:TPU latency in bf16 / bf16×3 / fp32 (Colab v5e-1)
- [ ] BLER cost of TPU bf16 (reduced sweep, notebook section 8); TPU compiler report (pass-dump fix pending a run)
- [ ] TensorRT fp16 (needs an fp16 ONNX graph)

## Credits
Model, system setup and pre-trained weights come from NVIDIA's [Sionna](https://github.com/NVlabs/sionna)
neural receiver tutorial. This repo reimplements the receiver in JAX for compiler and latency research.
