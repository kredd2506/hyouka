"""Train the Flax NRX from scratch with the tutorial's recipe.

    uv run python -m nrx.train --steps 30000 --ckpt-dir checkpoints/run1
"""

import argparse
import csv
import pathlib
import queue
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import sionna.phy
import torch

from nrx.config import CONFIG
from nrx.data import SionnaLink, data_re_indices
from nrx.loss import bmd_rate
from nrx.model import NeuralReceiver, make_inputs

TRACE_COUNT = {"train_step": 0}


def make_train_step(optimizer, indices):
    model = NeuralReceiver()

    def loss_fn(params, y, no, c):
        return -bmd_rate(model.apply(params, make_inputs(y, no)), c, indices)

    @jax.jit
    def train_step(params, opt_state, y, no, c):
        TRACE_COUNT["train_step"] += 1  # runs only while tracing
        loss, grads = jax.value_and_grad(loss_fn)(params, y, no, c)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, -loss

    return train_step


def batches(link, batch_size, seed, prefetch=4):
    """Background thread running Sionna, so data generation overlaps the JAX step.
    One Eb/N0 per batch, uniform in [min, max], as in the tutorial."""
    rng = np.random.default_rng(seed)
    q: queue.Queue = queue.Queue(maxsize=prefetch)

    def worker():
        while True:
            ebno = float(rng.uniform(CONFIG.ebno_db_min, CONFIG.ebno_db_max))
            out = link.sample_torch(batch_size, torch.tensor(ebno))
            q.put((ebno, out["y"].cpu().numpy(), out["no"].cpu().numpy(), out["c"].cpu().numpy()))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        yield q.get()


def checkpoint_manager(ckpt_dir, every):
    return ocp.CheckpointManager(pathlib.Path(ckpt_dir).absolute(),
                                 options=ocp.CheckpointManagerOptions(save_interval_steps=every, max_to_keep=3))


def load_checkpoint(ckpt_dir):
    """Params of the latest checkpoint in ckpt_dir."""
    mngr = checkpoint_manager(ckpt_dir, 1)
    state = mngr.restore(mngr.latest_step(), args=ocp.args.StandardRestore())
    return jax.tree.map(jnp.asarray, state["params"])


def train(steps, batch_size, lr, ckpt_dir, ckpt_every, log_every, seed):
    sionna.phy.config.seed = seed
    link = SionnaLink()
    indices = jnp.asarray(data_re_indices())
    T, F = CONFIG.num_ofdm_symbols, CONFIG.fft_size

    params = NeuralReceiver().init(jax.random.key(seed), jnp.zeros((batch_size, T, F, CONFIG.num_input_channels)))
    optimizer = optax.adam(lr)  # torch defaults: b1 0.9, b2 0.999, eps 1e-8
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer, indices)

    mngr = checkpoint_manager(ckpt_dir, ckpt_every)
    start = 0
    if mngr.latest_step() is not None:
        start = mngr.latest_step()
        state = mngr.restore(start, args=ocp.args.StandardRestore({"params": params, "opt_state": opt_state}))
        params, opt_state = state["params"], state["opt_state"]
        print(f"resumed from step {start}")

    log_path = pathlib.Path(ckpt_dir) / "train_log.csv"
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if start == 0:
        writer.writerow(["step", "ebno_db", "rate", "step_ms", "wait_ms"])

    data = batches(link, batch_size, seed + start)
    rates, t_steps = [], []
    for step in range(start, steps):
        t0 = time.perf_counter()
        ebno, y, no, c = next(data)
        t1 = time.perf_counter()
        params, opt_state, rate = train_step(params, opt_state, y, no, c)
        rate = float(rate)  # sync
        t2 = time.perf_counter()
        rates.append(rate)
        t_steps.append((t2 - t1) * 1e3)
        writer.writerow([step, f"{ebno:.2f}", f"{rate:.5f}", f"{(t2 - t1) * 1e3:.2f}", f"{(t1 - t0) * 1e3:.2f}"])
        if (step + 1) % log_every == 0 or step == steps - 1:
            log.flush()
            print(f"step {step + 1:6d}/{steps}  rate {np.mean(rates):.4f}  "
                  f"step {np.median(t_steps[1:] or t_steps):.1f} ms  traces {TRACE_COUNT['train_step']}")
            rates.clear()
        mngr.save(step + 1, args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}))
    if mngr.latest_step() != steps:
        mngr.save(steps, args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}), force=True)
    mngr.wait_until_finished()
    log.close()
    assert TRACE_COUNT["train_step"] == (1 if start < steps else 0), f"train_step retraced {TRACE_COUNT['train_step']} times"
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=CONFIG.num_training_iterations)
    ap.add_argument("--batch-size", type=int, default=CONFIG.training_batch_size)
    ap.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    ap.add_argument("--ckpt-dir", default="checkpoints/run1")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=CONFIG.seed)
    args = ap.parse_args()
    pathlib.Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    train(args.steps, args.batch_size, args.lr, args.ckpt_dir, args.ckpt_every, args.log_every, args.seed)


if __name__ == "__main__":
    main()
