import jax.numpy as jnp
import numpy as np
import optax

from nrx import train
from nrx.data import data_re_indices
from nrx.model import NeuralReceiver
from nrx.config import CONFIG


def test_train_step_compiles_once_and_improves(link):
    train.TRACE_COUNT["train_step"] = 0
    B = 4
    params = NeuralReceiver().init(__import__("jax").random.key(0),
                                   jnp.zeros((B, CONFIG.num_ofdm_symbols, CONFIG.fft_size, 5)))
    opt = optax.adam(1e-3)
    state = opt.init(params)
    step = train.make_train_step(opt, jnp.asarray(data_re_indices()))
    rates = []
    for ebno in (0.0, 5.0, 10.0, 0.0, 5.0, 10.0):
        out = link.sample(B, ebno)
        params, state, rate = step(params, state, out["y"], out["no"], out["c"])
        rates.append(float(rate))
    assert train.TRACE_COUNT["train_step"] == 1
    assert np.all(np.isfinite(rates))
