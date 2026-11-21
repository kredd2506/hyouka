# Reproduce nrx-jax. Run `make help`.
# On a CUDA host: `make setup EXTRA=cuda`; on a TPU VM: `make setup EXTRA=tpu`.

EXTRA ?=
UV_RUN := uv run
# JAX preallocates 75% of GPU memory by default, which starves Sionna's torch on the same GPU.
export XLA_PYTHON_CLIENT_PREALLOCATE ?= false

MC_ITER_REDUCED ?= 40
# jax_default_matmul_precision: default (TF32 on Ampere+ GPUs) | highest (true fp32)
PRECISION ?= default
# TensorRT wheels: cu13 (driver >= 580) or cu12
TRT_CUDA ?= cu13
TRT_VERSION ?= 11.5.0.63
STEPS ?= 30000
RUN ?= checkpoints/run1

.PHONY: help host-info setup weights test fixtures backends compiler onnx sweep-reduced sweep train eval-trained \
        pilots-recover pilots-cuda trt-install trt all-local all-cloud clean

help:  ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

host-info:  ## Record hardware + versions -> results/host_<hostname>.txt
	@mkdir -p results
	@{ date -u; uname -a; nvidia-smi 2>/dev/null || true; \
	  $(UV_RUN) python -c "import jax, torch, sionna, flax; print('jax', jax.__version__, jax.devices()); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available()); print('flax', flax.__version__)"; \
	} 2>&1 | grep -v jitc | tee results/host_$$(hostname -s).txt

setup:  ## Create the venv (EXTRA=cuda|tpu for accelerators)
	uv sync $(if $(EXTRA),--extra $(EXTRA),)

weights:  ## Download + checksum the pre-trained Sionna weights
	$(UV_RUN) python -m nrx.fetch_weights

test: weights  ## Unit + parity tests (~5 s)
	$(UV_RUN) pytest -q

fixtures: weights  ## Fixed input batches + torch LLRs for the benchmarks
	$(UV_RUN) python bench/backends.py --make-fixture

backends: fixtures  ## Latency + parity on the current XLA backend (PRECISION=highest for true fp32)
	$(UV_RUN) python bench/backends.py --precision $(PRECISION)

compiler: weights  ## jaxpr/StableHLO/HLO walk + per-op timing -> results/compiler_<backend>_b*.md
	$(UV_RUN) python bench/inspect_compiler.py --batch 1 --dump-passes --precision $(PRECISION)
	$(UV_RUN) python bench/inspect_compiler.py --batch 128 --precision $(PRECISION)

onnx: fixtures  ## Export torch NRX to ONNX for TensorRT -> results/onnx/
	$(UV_RUN) python bench/export_onnx.py

sweep-reduced: weights  ## BLER sweep, 40 MC iters (~30 min CPU) -> results/ber_bler_ported_reduced.*
	$(UV_RUN) python -m nrx.evaluate --systems jax-ported torch-ref baseline-perfect-csi baseline-ls-estimation \
		--max-mc-iter $(MC_ITER_REDUCED) --out results/ber_bler_ported_reduced.csv

sweep: weights  ## Full BLER sweep with notebook settings (GPU) -> results/ber_bler_ported.*
	$(UV_RUN) python -m nrx.evaluate --systems jax-ported torch-ref baseline-perfect-csi baseline-ls-estimation \
		--out results/ber_bler_ported.csv

train:  ## Train from scratch (resumable; GPU/TPU) -> $(RUN)
	$(UV_RUN) python -m nrx.train --steps $(STEPS) --ckpt-dir $(RUN)

eval-trained: weights  ## BLER sweep of the trained model -> results/ber_bler_trained.*
	$(UV_RUN) python -m nrx.evaluate --systems jax-trained --checkpoint $(RUN) --out results/ber_bler_trained.csv

pilots-recover: weights  ## Re-derive training pilots from the weights (~2 min CPU)
	$(UV_RUN) python -m nrx.pilots recover --out assets/pilots_recovered.npy

pilots-cuda:  ## Exact seed-42 pilots from the CUDA RNG, compared to the recovered ones (CUDA only)
	$(UV_RUN) python -m nrx.pilots cuda --out assets/pilots_cuda_seed42.npy
	$(UV_RUN) python -m nrx.pilots compare assets/pilots_recovered.npy assets/pilots_cuda_seed42.npy

all-local: test backends compiler onnx  ## Everything that runs on a laptop in minutes

all-cloud: test pilots-cuda backends compiler onnx sweep train eval-trained  ## Full cloud session

clean:  ## Remove generated data (keeps weights, results, checkpoints)
	rm -rf data/ .pytest_cache

trt-install:  ## TensorRT libs + bindings wheels from pypi.nvidia.com (TRT_CUDA=cu12 on old drivers)
	uv pip install --extra-index-url https://pypi.nvidia.com \
		tensorrt-$(TRT_CUDA)-libs==$(TRT_VERSION) tensorrt-$(TRT_CUDA)-bindings==$(TRT_VERSION)
	$(UV_RUN) python -c "import tensorrt_bindings as t; print('TensorRT', t.__version__)"

trt: onnx  ## TensorRT fp32 (TF32 off) / tf32 / fp16, timed like `backends` (run trt-install first)
	$(UV_RUN) python bench/trt_bench.py
