"""A tiny deep-learning library written from scratch in NumPy (no PyTorch / TensorFlow).

Contents
    Adam      optimiser with bias correction, L2 weight decay and global-norm gradient clipping
    MLP       fully-connected network (He init, ReLU) with manual back-propagation
    LSTM      long short-term memory layer with manual back-propagation through time (BPTT)

Every layer exposes   forward(...)   backward(...)   params()   grads()   so that models can be
composed by hand.  Correctness of the analytic gradients is verified against finite differences in
`tests/test_nn.py` (gradient check), which is the standard way to validate a hand-written backward pass.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


# ==========================================================================
class Adam:
    """Adam (Kingma & Ba 2015).  `params` are NumPy arrays that are updated *in place*."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, clip=None):
        self.p = params
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.wd, self.clip = weight_decay, clip
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        if self.clip:                                    # global-norm clipping (stabilises BPTT)
            norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            scale = min(1.0, self.clip / (norm + 1e-12))
            grads = [g * scale for g in grads]
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            if self.wd:
                g = g + self.wd * p
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ==========================================================================
class MLP:
    """Fully connected network  x -> [Dense+ReLU]* -> Dense (linear output)."""

    def __init__(self, sizes, rng: np.random.Generator):
        self.W = [rng.normal(0.0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]
        self._x: list = []
        self._z: list = []

    def forward(self, x):
        self._x, self._z = [], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            self._x.append(x)
            z = x @ W + b
            self._z.append(z)
            x = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
        return x

    def backward(self, dout):
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                dout = dout * (self._z[i] > 0)
            self.dW[i] = self._x[i].T @ dout
            self.db[i] = dout.sum(axis=0)
            dout = dout @ self.W[i].T
        return dout                                      # gradient w.r.t. the input

    def params(self):
        return [a for pair in zip(self.W, self.b) for a in pair]

    def grads(self):
        return [a for pair in zip(self.dW, self.db) for a in pair]

    def copy_from(self, other: "MLP"):
        for a, b in zip(self.params(), other.params()):
            a[...] = b


# ==========================================================================
class LSTM:
    """Single-layer LSTM, returns the last hidden state.

        i,f,o = sigmoid(.)   g = tanh(.)
        c_t = f*c_{t-1} + i*g          h_t = o*tanh(c_t)

    Input X has shape (batch, time, features).  Forget-gate bias starts at 1 (Jozefowicz et al. 2015).
    """

    def __init__(self, n_in: int, n_hidden: int, rng: np.random.Generator):
        H = n_hidden
        self.H = self.out_dim = H
        self.Wx = rng.normal(0.0, np.sqrt(1.0 / n_in), (n_in, 4 * H))
        self.Wh = rng.normal(0.0, np.sqrt(1.0 / H), (H, 4 * H))
        self.b = np.zeros(4 * H)
        self.b[H:2 * H] = 1.0
        self.dWx, self.dWh, self.db = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)
        self._cache = None

    def forward(self, X):
        B, T, _ = X.shape
        H = self.H
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache = []
        xw = X @ self.Wx + self.b                       # (B,T,4H) - one big matmul for all time steps
        for t in range(T):
            z = xw[:, t] + h @ self.Wh
            i, f = sigmoid(z[:, :H]), sigmoid(z[:, H:2 * H])
            g, o = np.tanh(z[:, 2 * H:3 * H]), sigmoid(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            cache.append((h, c, i, f, g, o, tc))
            h, c = h_new, c_new
        self._cache = (X, cache)
        return h

    def backward(self, dh_last):
        """Back-propagation through time.  Returns nothing (input gradient is not needed)."""
        X, cache = self._cache
        B, T, _ = X.shape
        H = self.H
        self.dWx[...] = 0.0
        self.dWh[...] = 0.0
        self.db[...] = 0.0
        dh, dc = dh_last, np.zeros((B, H))
        for t in reversed(range(T)):
            h_prev, c_prev, i, f, g, o, tc = cache[t]
            do = dh * tc
            dc = dc + dh * o * (1.0 - tc ** 2)
            di, df, dg = dc * g, dc * c_prev, dc * i
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
            self.dWx += X[:, t].T @ dz
            self.dWh += h_prev.T @ dz
            self.db += dz.sum(axis=0)
            dh = dz @ self.Wh.T
            dc = dc * f

    def params(self):
        return [self.Wx, self.Wh, self.b]

    def grads(self):
        return [self.dWx, self.dWh, self.db]


# ==========================================================================
# Sequence encoders used by the forecaster comparison.  Common interface:
#     out_dim, forward(X: (B,T,C)) -> (B,out_dim), backward(d_out), params(), grads()
# ==========================================================================
class GRU:
    """Gated recurrent unit (Cho et al. 2014).  Returns the last hidden state.

        r = s(x Wr + h Ur)   z = s(x Wz + h Uz)   n = tanh(x Wn + r*(h Un))   h' = (1-z)*n + z*h
    """

    def __init__(self, n_in: int, n_hidden: int, rng: np.random.Generator):
        H = n_hidden
        self.H = self.out_dim = H
        self.Wx = rng.normal(0.0, np.sqrt(1.0 / n_in), (n_in, 3 * H))
        self.Wh = rng.normal(0.0, np.sqrt(1.0 / H), (H, 3 * H))
        self.b = np.zeros(3 * H)
        self.dWx, self.dWh, self.db = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)
        self._cache = None

    def forward(self, X):
        B, T, _ = X.shape
        H = self.H
        h = np.zeros((B, H))
        xw = X @ self.Wx + self.b
        cache = []
        for t in range(T):
            hh = h @ self.Wh
            r = sigmoid(xw[:, t, :H] + hh[:, :H])
            z = sigmoid(xw[:, t, H:2 * H] + hh[:, H:2 * H])
            n = np.tanh(xw[:, t, 2 * H:] + r * hh[:, 2 * H:])
            cache.append((h, r, z, n, hh[:, 2 * H:]))
            h = (1.0 - z) * n + z * h
        self._cache = (X, cache)
        return h

    def backward(self, dh):
        X, cache = self._cache
        H = self.H
        self.dWx[...] = self.dWh[...] = self.db[...] = 0.0
        for t in reversed(range(X.shape[1])):
            h_prev, r, z, n, hhn = cache[t]
            dn_pre = dh * (1.0 - z) * (1.0 - n ** 2)
            dz_pre = dh * (h_prev - n) * z * (1.0 - z)
            dr_pre = dn_pre * hhn * r * (1.0 - r)
            dxw = np.concatenate([dr_pre, dz_pre, dn_pre], axis=1)
            dhh = np.concatenate([dr_pre, dz_pre, dn_pre * r], axis=1)
            self.dWx += X[:, t].T @ dxw
            self.dWh += h_prev.T @ dhh
            self.db += dxw.sum(axis=0)
            dh = dh * z + dhh @ self.Wh.T

    def params(self):
        return [self.Wx, self.Wh, self.b]

    def grads(self):
        return [self.dWx, self.dWh, self.db]


class Conv1D:
    """'Valid' 1-D convolution over time:  (B,T,C) -> (B,T-k+1,F)   via im2col."""

    def __init__(self, c_in: int, c_out: int, k: int, rng: np.random.Generator):
        self.k, self.c_in = k, c_in
        self.W = rng.normal(0.0, np.sqrt(2.0 / (k * c_in)), (c_in * k, c_out))
        self.b = np.zeros(c_out)
        self.dW, self.db = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, X):
        B, T, C = X.shape
        win = np.lib.stride_tricks.sliding_window_view(X, self.k, axis=1)     # (B,L,C,k)
        self._cols = win.reshape(B, T - self.k + 1, C * self.k)
        self._shape = X.shape
        return self._cols @ self.W + self.b

    def backward(self, dout):
        B, T, C = self._shape
        L = T - self.k + 1
        self.dW = self._cols.reshape(-1, C * self.k).T @ dout.reshape(-1, dout.shape[-1])
        self.db = dout.sum(axis=(0, 1))
        dcols = (dout @ self.W.T).reshape(B, L, C, self.k)
        dX = np.zeros(self._shape)
        for j in range(self.k):
            dX[:, j:j + L, :] += dcols[..., j]
        return dX

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class TCN:
    """Temporal CNN encoder: Conv(k)-ReLU-Conv(k)-ReLU, then [mean over time , last step] features."""

    def __init__(self, n_in: int, n_filters: int, rng: np.random.Generator, k: int = 3):
        self.c1, self.c2 = Conv1D(n_in, n_filters, k, rng), Conv1D(n_filters, n_filters, k, rng)
        self.out_dim = 2 * n_filters
        self.F = n_filters

    def forward(self, X):
        self._a1 = self.c1.forward(X)
        h1 = np.maximum(self._a1, 0.0)
        self._a2 = self.c2.forward(h1)
        h2 = np.maximum(self._a2, 0.0)
        self._L = h2.shape[1]
        return np.concatenate([h2.mean(axis=1), h2[:, -1]], axis=1)

    def backward(self, dout):
        F, L = self.F, self._L
        dh2 = np.repeat(dout[:, None, :F] / L, L, axis=1)
        dh2[:, -1] += dout[:, F:]
        dh1 = self.c2.backward(dh2 * (self._a2 > 0))
        self.c1.backward(dh1 * (self._a1 > 0))

    def params(self):
        return self.c1.params() + self.c2.params()

    def grads(self):
        return self.c1.grads() + self.c2.grads()


class _Linear:
    """Affine map applied to the last axis of an array with any number of leading dims."""

    def __init__(self, n_in, n_out, rng, bias=True):
        self.W = rng.normal(0.0, np.sqrt(1.0 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out) if bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros(n_out) if bias else None

    def forward(self, x):
        self._x = x
        return x @ self.W + (self.b if self.b is not None else 0.0)

    def backward(self, dy):
        x2, dy2 = self._x.reshape(-1, self.W.shape[0]), dy.reshape(-1, self.W.shape[1])
        self.dW = x2.T @ dy2
        if self.b is not None:
            self.db = dy2.sum(axis=0)
        return dy @ self.W.T

    def params(self):
        return [self.W] + ([self.b] if self.b is not None else [])

    def grads(self):
        return [self.dW] + ([self.db] if self.b is not None else [])


class _LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.g, self.b, self.eps = np.ones(d), np.zeros(d), eps
        self.dg, self.db = np.zeros(d), np.zeros(d)

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self._inv = 1.0 / np.sqrt(var + self.eps)
        self._xh = (x - mu) * self._inv
        return self.g * self._xh + self.b

    def backward(self, dy):
        self.dg = (dy * self._xh).sum(axis=tuple(range(dy.ndim - 1)))
        self.db = dy.sum(axis=tuple(range(dy.ndim - 1)))
        dxh = dy * self.g
        return self._inv * (dxh - dxh.mean(axis=-1, keepdims=True)
                            - self._xh * (dxh * self._xh).mean(axis=-1, keepdims=True))

    def params(self):
        return [self.g, self.b]

    def grads(self):
        return [self.dg, self.db]


class _MultiHeadSelfAttention:
    def __init__(self, d, heads, rng):
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (_Linear(d, d, rng, bias=False) for _ in range(4))

    def _split(self, x):                                   # (B,T,d) -> (B,h,T,dh)
        B, T, _ = x.shape
        return x.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)

    def _merge(self, x):                                   # (B,h,T,dh) -> (B,T,d)
        B, _, T, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.h * self.dh)

    def forward(self, x):
        self._q, self._k, self._v = self._split(self.q.forward(x)), self._split(self.k.forward(x)), self._split(self.v.forward(x))
        s = self._q @ self._k.transpose(0, 1, 3, 2) / np.sqrt(self.dh)
        s -= s.max(axis=-1, keepdims=True)
        e = np.exp(s)
        self._a = e / e.sum(axis=-1, keepdims=True)         # attention weights (B,h,T,T)
        return self.o.forward(self._merge(self._a @ self._v))

    def backward(self, dout):
        dctx = self._split(self.o.backward(dout))
        da = dctx @ self._v.transpose(0, 1, 3, 2)
        dv = self._a.transpose(0, 1, 3, 2) @ dctx
        ds = self._a * (da - (da * self._a).sum(axis=-1, keepdims=True)) / np.sqrt(self.dh)
        dq, dk = ds @ self._k, ds.transpose(0, 1, 3, 2) @ self._q
        return (self.q.backward(self._merge(dq)) + self.k.backward(self._merge(dk))
                + self.v.backward(self._merge(dv)))

    def params(self):
        return [p for m in (self.q, self.k, self.v, self.o) for p in m.params()]

    def grads(self):
        return [g for m in (self.q, self.k, self.v, self.o) for g in m.grads()]


class TransformerEncoder:
    """One pre-LayerNorm Transformer block (multi-head self-attention + feed-forward, residuals)
    with learned input projection and fixed sinusoidal positional encoding.
    Output = [mean over time , last time step]."""

    def __init__(self, n_in: int, d_model: int, rng: np.random.Generator, heads: int = 2, T: int = 24):
        d = d_model
        self.emb = _Linear(n_in, d, rng)
        self.ln1, self.ln2 = _LayerNorm(d), _LayerNorm(d)
        self.att = _MultiHeadSelfAttention(d, heads, rng)
        self.ff1, self.ff2 = _Linear(d, 2 * d, rng), _Linear(2 * d, d, rng)
        pos = np.arange(T)[:, None]
        i = np.arange(d)[None, :]
        ang = pos / np.power(10000.0, (2 * (i // 2)) / d)
        self.pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang))
        self.out_dim = 2 * d
        self.d = d

    def forward(self, X):
        e = self.emb.forward(X) + self.pe[None, :X.shape[1]]
        e1 = e + self.att.forward(self.ln1.forward(e))
        f = self.ff1.forward(self.ln2.forward(e1))
        self._ffmask = f > 0
        e2 = e1 + self.ff2.forward(np.maximum(f, 0.0))
        self._T = e2.shape[1]
        return np.concatenate([e2.mean(axis=1), e2[:, -1]], axis=1)

    def backward(self, dout):
        d, T = self.d, self._T
        de2 = np.repeat(dout[:, None, :d] / T, T, axis=1)
        de2[:, -1] += dout[:, d:]
        dff = self.ff2.backward(de2) * self._ffmask
        de1 = de2 + self.ln2.backward(self.ff1.backward(dff))
        de = de1 + self.ln1.backward(self.att.backward(de1))
        self.emb.backward(de)

    def _mods(self):
        return (self.emb, self.ln1, self.att, self.ln2, self.ff1, self.ff2)

    def params(self):
        return [p for m in self._mods() for p in m.params()]

    def grads(self):
        return [g for m in self._mods() for g in m.grads()]


def count_params(params) -> int:
    return int(sum(p.size for p in params))
