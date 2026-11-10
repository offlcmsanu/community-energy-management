"""Multivariate deep sequence forecasters (LSTM, GRU, temporal CNN, Transformer) in pure NumPy, see nn.py.

Task: predict hour t of all 10 series (9 household demands + solar GHI) *at day-ahead gate closure*.

Architecture
    history window  = 24 h ending 24 h before t   (t-47 ... t-24), 13 features per hour
                      [10 normalised series, temperature, sin/cos hour]
        -> LSTM(32)  -> last hidden state h
    context at t    = calendar (hour, day-of-year, day-of-week), temperature forecast, clear-sky
                      irradiance, tariff tier                                  (11 values)
        -> concat [h, context] -> Dense(64)+ReLU -> Dense(10)
    target          = normalised value at t minus the mean of the history window (residual learning,
                      copes with level shifts)
Training: Adam, MSE, L2 weight decay, gradient clipping, mini-batches, early stopping on a validation
split, 4 contiguous out-of-fold blocks (every hour is predicted by a network that never saw it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import KFold

from .data_loader import CommunityData
import time

from .nn import GRU, LSTM, MLP, TCN, Adam, TransformerEncoder, count_params

WINDOW = 24
LAG = 24            # history ends 24 h before the target hour (day-ahead information only)


ARCHS = ("lstm", "gru", "tcn", "transformer")


def make_encoder(arch: str, n_in: int, hidden: int, rng):
    if arch == "lstm":
        return LSTM(n_in, hidden, rng)
    if arch == "gru":
        return GRU(n_in, hidden, rng)
    if arch == "tcn":
        return TCN(n_in, hidden, rng)
    if arch == "transformer":
        return TransformerEncoder(n_in, hidden, rng, heads=2, T=WINDOW)
    raise ValueError(arch)


class SeqModel:
    """encoder(history window) -> concat[context] -> Dense(64)+ReLU -> Dense(10)."""

    def __init__(self, n_in: int, n_ctx: int, n_out: int, hidden: int = 32, head: int = 64, seed: int = 0,
                 arch: str = "lstm"):
        rng = np.random.default_rng(seed)
        self.enc = make_encoder(arch, n_in, hidden, rng)
        self.head = MLP([self.enc.out_dim + n_ctx, head, n_out], rng)

    def forward(self, X, ctx):
        return self.head.forward(np.hstack([self.enc.forward(X), ctx]))

    def backward(self, dout):
        dz = self.head.backward(dout)
        self.enc.backward(dz[:, :self.enc.out_dim])

    def params(self):
        return self.enc.params() + self.head.params()

    def grads(self):
        return self.enc.grads() + self.head.grads()


# --------------------------------------------------------------------------
def build_sequences(data: CommunityData):
    """Returns X (N,24,13), ctx (N,11), Y residual targets (N,10), base (N,10), mu, sd."""
    from .forecasting import clear_sky_ghi              # local import avoids a circular dependency
    S = np.hstack([data.demand.values, data.ghi[:, None]])          # (T,10)
    mu, sd = S.mean(axis=0), S.std(axis=0) + 1e-9
    Sn = (S - mu) / sd
    T = len(S)
    idx = data.index
    temp_n = ((data.temp - data.temp.mean()) / data.temp.std())[:, None]
    hs = np.sin(2 * np.pi * idx.hour.values / 24)[:, None]
    hc = np.cos(2 * np.pi * idx.hour.values / 24)[:, None]
    F = np.hstack([Sn, temp_n, hs, hc])                             # (T,13)
    pad = 2 * 24                                                    # first 2 days borrow day 1 as history
    Fp = np.vstack([F[:pad], F])
    win = sliding_window_view(Fp, (WINDOW, F.shape[1]))[:, 0]       # (T+pad-23, 24, 13)
    X = win[1:T + 1]                                                # sample t -> rows t-47..t-24
    base = X[:, :, :10].mean(axis=1)                                # window mean (N,10)
    Y = Sn - base
    doy = idx.dayofyear.values
    dow = idx.dayofweek.values
    prev_temp = np.vstack([temp_n[:LAG], temp_n[:-LAG]])
    def lagged(k):                                       # normalised series k hours ago (day 1 borrows itself)
        return np.vstack([Sn[:k], Sn[:-k]])

    ctx = np.hstack([lagged(48), lagged(168), hs, hc,
                     np.sin(2 * np.pi * doy / 365)[:, None], np.cos(2 * np.pi * doy / 365)[:, None],
                     np.sin(2 * np.pi * dow / 7)[:, None], np.cos(2 * np.pi * dow / 7)[:, None],
                     (dow >= 5).astype(float)[:, None], temp_n, temp_n - prev_temp,
                     (clear_sky_ghi(idx) / 1000.0)[:, None], (data.price / data.price.max())[:, None]])
    return X.copy(), ctx, Y, base, mu, sd


def _batches(idx, bs, rng):
    idx = rng.permutation(idx)
    for k in range(0, len(idx), bs):
        yield idx[k:k + bs]


def fit(model: SeqModel, X, ctx, Y, train_idx, epochs=20, batch=128, lr=3e-3, wd=5e-3, patience=4,
        val_frac=0.1, seed=0, log=None):
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(train_idx)
    n_val = int(len(train_idx) * val_frac)
    va, tr = train_idx[:n_val], train_idx[n_val:]
    opt = Adam(model.params(), lr=lr, weight_decay=wd, clip=1.0)
    best, best_params, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        tot = 0.0
        for b in _batches(tr, batch, rng):
            out = model.forward(X[b], ctx[b])
            err = out - Y[b]
            tot += float((err ** 2).mean()) * len(b)
            model.backward(2 * err / err.size)              # d(mean sq. error)/d out
            opt.step(model.grads())
        vloss = float(((model.forward(X[va], ctx[va]) - Y[va]) ** 2).mean())
        hist.append((tot / len(tr), vloss))
        if log:
            log(ep, hist[-1])
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_params = [p.copy() for p in model.params()]
        else:
            bad += 1
            if bad >= patience:
                break
    if best_params is not None:                             # restore best weights (early stopping)
        for p, q in zip(model.params(), best_params):
            p[...] = q
    return hist


def predict(model: SeqModel, X, ctx, batch=1024):
    return np.vstack([model.forward(X[k:k + batch], ctx[k:k + batch]) for k in range(0, len(X), batch)])


def seq_oof_forecasts(data: CommunityData, arch: str = "lstm", n_splits=4, epochs=25, hidden=32, seed=0,
                      progress=None):
    """Out-of-fold forecasts of one architecture.
    Returns (demand_hat DataFrame, ghi_hat array, training histories, info dict with params / seconds)."""
    from .forecasting import clear_sky_ghi
    X, ctx, Y, base, mu, sd = build_sequences(data)
    N = len(X)
    pred = np.zeros_like(Y)
    histories, n_params = [], 0
    t0 = time.time()
    for k, (tr, te) in enumerate(KFold(n_splits=n_splits, shuffle=False).split(np.arange(N))):
        if progress:
            progress(k / n_splits, f"{arch} fold {k + 1}/{n_splits}")
        model = SeqModel(X.shape[2], ctx.shape[1], Y.shape[1], hidden=hidden, seed=seed + k, arch=arch)
        n_params = count_params(model.params())
        histories.append(fit(model, X, ctx, Y, tr, epochs=epochs, seed=seed + k))
        pred[te] = predict(model, X[te], ctx[te])
    S_hat = np.clip((pred + base) * sd + mu, 0.0, None)
    ghi_hat = S_hat[:, 9].copy()
    ghi_hat[clear_sky_ghi(data.index) <= 0] = 0.0
    dhat = pd.DataFrame(S_hat[:, :9], index=data.index, columns=data.ids)
    return dhat, ghi_hat, histories, dict(params=n_params, seconds=time.time() - t0,
                                          epochs=[len(h) for h in histories])


def lstm_oof_forecasts(data: CommunityData, n_splits=4, epochs=25, hidden=32, seed=0, progress=None):
    """Backwards-compatible wrapper (LSTM)."""
    d, g, h, _ = seq_oof_forecasts(data, "lstm", n_splits, epochs, hidden, seed, progress)
    return d, g, h
