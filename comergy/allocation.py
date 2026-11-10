"""Fair sharing of the benefits of cooperation.

The community problem is split in two layers
    1. PHYSICAL layer  : one optimiser dispatches the pooled batteries (optimizer.py)
    2. FINANCIAL layer : this module decides who pays / earns what.

Mechanisms
    mid_market   price-based local market, price = (retail + FiT)/2   (benchmark-report LEM)
    sdr          price-based local market, price follows supply/demand ratio
    shapley      cooperative-game allocation of the saving (marginal contribution)
    equity       Shapley blended with a need-based (energy-burden) objective, solved as a QP
    + optional   "no-worse-off" top-up that guarantees nobody pays more than standalone.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from .config import BatteryParams
from .data_loader import CommunityData
from .optimizer import Dispatch, solve_battery


# ==========================================================================
# 1. Representative days (KMeans) + coalition values + Shapley
# ==========================================================================
def representative_days(data: CommunityData, k: int = 24, seed: int = 0):
    """Cluster the 365 days (demand shape, solar shape, tariff shape) with KMeans and return
    one real *medoid* day per cluster plus the cluster size as weight.  This shrinks the
    Shapley game from 365 to k days per coalition (>15x faster) with ~1-2% error."""
    D = data.days
    ghi = data.ghi.reshape(D, 24)
    dem = data.demand.sum(axis=1).values.reshape(D, 24)
    prc = data.price.reshape(D, 24)
    X = np.hstack([ghi / ghi.max(), dem / dem.max(), prc / prc.max()])
    k = min(k, D)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    days, weights = [], []
    for c in range(k):
        mem = np.where(km.labels_ == c)[0]
        days.append(int(mem[np.argmin(((X[mem] - km.cluster_centers_[c]) ** 2).sum(axis=1))]))
        weights.append(len(mem))
    return np.array(days), np.array(weights, dtype=float)


def coalition_costs(data: CommunityData, bp: BatteryParams, days, weights, progress=None) -> np.ndarray:
    """C[mask] = optimal annual energy cost of the coalition `mask` (bit i = member i)
    when its members pool PV demand and batteries (energy cost only, no connection fee)."""
    N = data.n
    D = data.days
    net = data.net_load().values.reshape(D, 24, N)
    price, fit = data.price.reshape(D, 24), data.fit.reshape(D, 24)
    cap_i = data.members["batt_kwh"].values
    C = np.zeros(1 << N)
    for mask in range(1, 1 << N):
        idx = [i for i in range(N) if (mask >> i) & 1]
        cap = cap_i[idx].sum()
        tot = 0.0
        for d, w in zip(days, weights):
            tot += w * solve_battery(net[d][:, idx].sum(axis=1), price[d], fit[d], cap, bp,
                                     cyclic=True).cost
        C[mask] = tot
        if progress and mask % 32 == 0:
            progress(mask / ((1 << N) - 1))
    return C


def shapley_savings(C: np.ndarray, N: int):
    """Exact Shapley value of the cost-saving game  v(S) = sum_{i in S} C({i}) - C(S).
    Returns (phi, v, diagnostics)."""
    single = np.array([C[1 << i] for i in range(N)])
    v = np.zeros(1 << N)
    for mask in range(1, 1 << N):
        v[mask] = sum(single[i] for i in range(N) if (mask >> i) & 1) - C[mask]
    phi = np.zeros(N)
    for i in range(N):
        for mask in range(1 << N):
            if (mask >> i) & 1:
                continue
            s = bin(mask).count("1")
            w = factorial(s) * factorial(N - s - 1) / factorial(N)
            phi[i] += w * (v[mask | (1 << i)] - v[mask])
    # superadditivity check on random disjoint coalition pairs (a union can always imitate its parts)
    rng, viol, full = np.random.default_rng(0), 0, (1 << N) - 1
    for _ in range(3000):
        a = int(rng.integers(1, full))
        rest = full ^ a
        if rest == 0:
            continue
        b = int(rng.integers(1, rest + 1)) & rest
        if b and v[a | b] + 1e-6 < v[a] + v[b]:
            viol += 1
    return phi, v, dict(efficiency_error=float(phi.sum() - v[-1]), superadditivity_violations=viol)


# ==========================================================================
# 2. Price-based local energy market (settlement of a given physical dispatch)
# ==========================================================================
def settle_market(data: CommunityData, pooled: Dispatch, rule: str = "mid_market",
                  battery_revenue: str = "owners") -> tuple[pd.Series, dict]:
    """Hourly internal market.

    Sellers = members with PV surplus (+ battery discharge); buyers = members with a deficit
    (+ battery charging).  Matched volume V = min(supply, demand) trades at the internal price p;
    the remainder trades with the grid at retail (buyers) / FiT (sellers).  Pro-rata matching.
    The virtual battery earns / pays on its own account; the net result is paid to
    `owners` (in proportion to installed kWh) or `equal` (shared by every member).
    Returns annual bills (incl. connection fee) and a details dict.
    """
    net = data.net_load().values
    buy, sell = np.clip(net, 0, None), np.clip(-net, 0, None)
    ch, dis = pooled.ch, pooled.dis
    B, S = buy.sum(axis=1) + ch, sell.sum(axis=1) + dis
    V = np.minimum(B, S)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(B > 1e-12, V / B, 0.0)          # share of buying that is served locally
        rs = np.where(S > 1e-12, V / S, 0.0)          # share of selling that is sold locally
        sdr = np.where(B > 1e-12, S / B, 1.0)
    retail, fit = data.price, data.fit
    if rule == "mid_market":
        p = (retail + fit) / 2.0
    elif rule == "sdr":
        p = fit + (retail - fit) * np.clip(1.0 - sdr, 0.0, 1.0)
    else:
        raise ValueError(rule)
    unit_buy = rb * p + (1 - rb) * retail             # average price paid per kWh bought
    unit_sell = rs * p + (1 - rs) * fit               # average price received per kWh sold
    cost_i = (buy * unit_buy[:, None]).sum(axis=0) - (sell * unit_sell[:, None]).sum(axis=0)
    batt_rev = float((dis * unit_sell).sum() - (ch * unit_buy).sum())   # AUD earned by the battery
    cap = data.members["batt_kwh"].values
    if battery_revenue == "owners" and cap.sum() > 0:
        share = cap / cap.sum()
    else:
        share = np.full(data.n, 1.0 / data.n)
    bills = pd.Series(cost_i - share * batt_rev + data.fee_year, index=data.ids)
    return bills, dict(internal_price=p, local_volume=float(V.sum()), battery_revenue=batt_rev)


# ==========================================================================
# 3. Cooperative-game and equity allocations
# ==========================================================================
def shapley_bills(ref: pd.Series, total_saving: float, phi: np.ndarray) -> pd.Series:
    """Distribute the *realised* total saving in proportion to the Shapley values."""
    share = phi / phi.sum() if phi.sum() > 1e-9 else np.full(len(phi), 1.0 / len(phi))
    return ref - total_saving * share


def equity_bills(ref: pd.Series, total_saving: float, phi: np.ndarray, income: pd.Series,
                 lam: float = 0.5, alpha: float = 0.6) -> pd.Series:
    """Blend merit (Shapley) with need (energy burden = bill / income).

        min_s   (1-lam) * mean(((s_i - phi_i)/(S/N))^2) + lam * mean(((burden_i - tau)/tau)^2)
        s.t.    sum_i s_i = S ,   s_i >= alpha * phi_i  (>= 0 -> no-worse-off by construction)

    s_i = saving of member i, S = total realised saving, burden_i = (ref_i - s_i)/income_i,
    tau = community-average burden.  lam=0 -> pure Shapley ; lam=1 -> pure need.
    """
    N, S = len(ref), float(total_saving)
    share = phi / phi.sum() if phi.sum() > 1e-9 else np.full(N, 1.0 / N)
    base = S * share
    if S <= 1e-6 or lam <= 0:
        return ref - base
    inc = income.values.astype(float)
    r = ref.values.astype(float)
    tau = (r.sum() - S) / inc.sum()

    def obj(s):
        burden = (r - s) / inc
        return (1 - lam) * np.mean(((s - base) / (S / N)) ** 2) + lam * np.mean(((burden - tau) / tau) ** 2)

    res = minimize(obj, base, method="SLSQP", bounds=[(alpha * b, None) for b in base],
                   constraints=[{"type": "eq", "fun": lambda s: s.sum() - S}],
                   options=dict(maxiter=300, ftol=1e-12))
    s = res.x if res.success else base
    s = s * (S / s.sum())                                # exact budget balance
    return pd.Series(r - s, index=ref.index)


def enforce_no_worse_off(bills: pd.Series, ref: pd.Series) -> pd.Series:
    """Side-payments so that nobody pays more than in the standalone case.
    Members who would lose are topped up to their standalone bill; the cost is borne by the
    winners in proportion to their savings (budget-balanced as long as total saving >= 0)."""
    s = ref - bills
    deficit = float(-s[s < 0].sum())
    if deficit <= 1e-9:
        return bills.copy()
    pos = s[s > 0]
    if pos.sum() < deficit:                              # cooperation does not pay -> cannot fix
        return bills.copy()
    out = bills.copy()
    out[s < 0] = ref[s < 0]
    out[s > 0] = bills[s > 0] + deficit * pos / pos.sum()
    return out
