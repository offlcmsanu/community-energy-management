"""Gradient checks: analytic back-propagation (incl. BPTT) vs. finite differences."""
import numpy as np

from comergy.nn import LSTM, MLP, Adam


def _numeric_check(loss_fn, params, grads, n_probe=25, eps=1e-5, tol=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for p, g in zip(params, grads):
        flat = p.reshape(-1)
        for k in rng.choice(flat.size, size=min(n_probe, flat.size), replace=False):
            old = flat[k]
            flat[k] = old + eps
            lp = loss_fn()
            flat[k] = old - eps
            lm = loss_fn()
            flat[k] = old
            num = (lp - lm) / (2 * eps)
            ana = g.reshape(-1)[k]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst


def test_mlp_gradients():
    rng = np.random.default_rng(1)
    net = MLP([5, 7, 6, 3], rng)
    x, y = rng.normal(size=(8, 5)), rng.normal(size=(8, 3))

    def loss():
        return 0.5 * np.mean(np.sum((net.forward(x) - y) ** 2, axis=1))

    out = net.forward(x)
    net.backward((out - y) / len(x))
    assert _numeric_check(loss, net.params(), net.grads()) < 1e-5


def test_lstm_bptt_gradients():
    rng = np.random.default_rng(2)
    lstm = LSTM(4, 6, rng)
    head = MLP([6, 3], rng)
    X, y = rng.normal(size=(5, 7, 4)), rng.normal(size=(5, 3))

    def loss():
        return 0.5 * np.mean(np.sum((head.forward(lstm.forward(X)) - y) ** 2, axis=1))

    out = head.forward(lstm.forward(X))
    dh = head.backward((out - y) / len(X))
    lstm.backward(dh)
    assert _numeric_check(loss, lstm.params() + head.params(), lstm.grads() + head.grads()) < 1e-5


def test_adam_learns_regression():
    rng = np.random.default_rng(3)
    net = MLP([2, 16, 1], rng)
    X = rng.uniform(-1, 1, (256, 2))
    y = (X[:, :1] * X[:, 1:2]) * 2.0                 # non-linear target
    opt = Adam(net.params(), lr=1e-2)
    first = None
    for _ in range(400):
        out = net.forward(X)
        loss = np.mean((out - y) ** 2)
        first = loss if first is None else first
        net.backward(2 * (out - y) / len(X))
        opt.step(net.grads())
    assert loss < 0.2 * first


def test_lstm_forecaster_pipeline_shapes():
    """Sequence builder + tiny LSTM training run end-to-end (few epochs, one fold)."""
    from comergy.data_loader import load_dataset
    from comergy.dl_forecaster import SeqModel, build_sequences, fit, predict
    d = load_dataset()
    X, ctx, Y, base, mu, sd = build_sequences(d)
    assert X.shape == (8760, 24, 13) and ctx.shape[0] == 8760 and Y.shape == (8760, 10)
    # leakage guard: the history window must end 24 h before the target hour
    t = 500
    assert np.allclose(X[t, -1, :10], ((np.hstack([d.demand.values, d.ghi[:, None]]) - mu) / sd)[t - 24])
    m = SeqModel(X.shape[2], ctx.shape[1], 10, hidden=8)
    hist = fit(m, X, ctx, Y, np.arange(2000), epochs=2)
    assert hist[-1][0] < hist[0][0] + 1e-9
    assert np.isfinite(predict(m, X[:50], ctx[:50])).all()


def test_dqn_smoke():
    """DQN trains for a handful of episodes and produces a physically consistent rollout."""
    from comergy.config import BatteryParams
    from comergy.data_loader import load_dataset
    from comergy.rl_agent import run_rl
    d = load_dataset()
    net = d.net_load().sum(axis=1).values
    r = run_rl(d, BatteryParams(), net, 0.0, 0.0, episodes=100)
    cap = d.members.batt_kwh.sum()
    assert r.soc.min() >= 0.1 * cap - 1e-6 and r.soc.max() <= cap + 1e-6
    assert np.allclose(r.imp - r.exp - r.ch + r.dis, net, atol=1e-6)


def _encoder_check(make_encoder, seed):
    rng = np.random.default_rng(seed)
    enc = make_encoder(rng)
    head = MLP([enc.out_dim, 3], rng)
    X, y = rng.normal(size=(4, 8, 5)), rng.normal(size=(4, 3))

    def loss():
        return 0.5 * np.mean(np.sum((head.forward(enc.forward(X)) - y) ** 2, axis=1))

    out = head.forward(enc.forward(X))
    enc.backward(head.backward((out - y) / len(X)))
    return _numeric_check(loss, enc.params() + head.params(), enc.grads() + head.grads(), n_probe=20)


def test_gru_gradients():
    from comergy.nn import GRU
    assert _encoder_check(lambda r: GRU(5, 6, r), 4) < 1e-5


def test_tcn_gradients():
    from comergy.nn import TCN
    assert _encoder_check(lambda r: TCN(5, 6, r), 5) < 1e-5


def test_transformer_gradients():
    from comergy.nn import TransformerEncoder
    assert _encoder_check(lambda r: TransformerEncoder(5, 8, r, heads=2, T=8), 6) < 1e-5


def test_all_architectures_train_and_forecast():
    from comergy.data_loader import load_dataset
    from comergy.dl_forecaster import ARCHS, SeqModel, build_sequences, fit, predict
    d = load_dataset()
    X, ctx, Y, base, mu, sd = build_sequences(d)
    for arch in ARCHS:
        m = SeqModel(X.shape[2], ctx.shape[1], 10, hidden=8, arch=arch)
        h = fit(m, X, ctx, Y, np.arange(1500), epochs=2)
        assert np.isfinite(h[-1]).all() and np.isfinite(predict(m, X[:20], ctx[:20])).all()
