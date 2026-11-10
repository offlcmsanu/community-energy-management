"""Deep Q-Network (DQN) controller for the community battery - pure NumPy, see nn.py.

Environment  : the pooled 65 kWh community battery, hourly steps, real tariff, real net load.
State  (16)  : state of charge, sin/cos hour, weekend flag, retail price now + next 5 h,
               actual net load now, ML-forecast net load for the next 5 h.
Actions (7)  : hold | absorb PV surplus only | cover deficit only | follow load (both) |
               grid-charge 50 % | grid-charge 100 % | discharge 100 %   (relative to the current net load,
               so the agent starts from sensible behaviour and learns *when to deviate*).
Reward       : - (energy cost of the hour) / 5   (+ a salvage value for energy left in the battery
               at the end of a day-episode, the same device used by the MPC).
Algorithm    : Double DQN, experience replay, target network, epsilon-greedy exploration,
               Huber loss, Adam.  Episodes are random days with random initial SOC.

Fair evaluation: days are split by week into two folds (even / odd weeks).  Agent k is trained only
on the other fold and then controls the battery on fold k.  The full year is therefore always
driven by an agent that never saw those days; SOC is carried across days like in the MPC run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BatteryParams
from .data_loader import CommunityData
from .nn import MLP, Adam

ACTION_NAMES = ["hold", "absorb surplus", "cover deficit", "follow load", "grid-charge 50%",
                "grid-charge 100%", "discharge 100%"]
N_ACTIONS = len(ACTION_NAMES)
LOOKAHEAD = 5
R_SCALE = 5.0


# --------------------------------------------------------------------------
def _features(data: CommunityData, net_actual: np.ndarray, net_fc: np.ndarray) -> np.ndarray:
    """Static part of the state for every hour (everything except the SOC)."""
    T = len(net_actual)
    idx = data.index
    pmax_price = data.price.max()

    def ahead(a, k):                                    # value k hours ahead (edge padded)
        return np.concatenate([a[k:], np.full(k, a[-1])])

    cols = [np.sin(2 * np.pi * idx.hour.values / 24), np.cos(2 * np.pi * idx.hour.values / 24),
            (idx.dayofweek.values >= 5).astype(float), data.price / pmax_price]
    cols += [ahead(data.price, k) / pmax_price for k in range(1, LOOKAHEAD + 1)]
    cols += [net_actual / 25.0]
    cols += [ahead(net_fc, k) / 25.0 for k in range(1, LOOKAHEAD + 1)]
    return np.column_stack(cols)


def action_to_power(a: int, net: float, p_max: float) -> float:
    """Map a discrete action to a battery power fraction of P_max (+ = charge, - = discharge)."""
    follow = float(np.clip(-net / p_max, -1.0, 1.0))
    return [0.0, max(follow, 0.0), min(follow, 0.0), follow, 0.5, 1.0, -1.0][a]


def _step(soc, a_frac, net, price, fit, cap, bp: BatteryParams):
    """Physical battery step.  Returns (new_soc, charge, discharge, imp, exp, hourly cost)."""
    p_max = bp.c_rate * cap
    smin, smax = bp.soc_min_frac * cap, bp.soc_max_frac * cap
    c = d = 0.0
    if a_frac > 0:
        c = min(a_frac * p_max, (smax - soc) / bp.eta_charge)
        soc += bp.eta_charge * c
    elif a_frac < 0:
        d = min(-a_frac * p_max, (soc - smin) * bp.eta_discharge)
        soc -= d / bp.eta_discharge
    flow = net + c - d
    imp, exp = max(flow, 0.0), max(-flow, 0.0)
    return soc, c, d, imp, exp, price * imp - fit * exp


class DQNAgent:
    def __init__(self, n_state: int, seed: int = 0, lr: float = 1e-3, gamma: float = 0.95):
        self.rng = np.random.default_rng(seed)
        self.q = MLP([n_state, 64, 64, N_ACTIONS], self.rng)
        self.target = MLP([n_state, 64, 64, N_ACTIONS], self.rng)
        self.target.copy_from(self.q)
        self.opt = Adam(self.q.params(), lr=lr, clip=5.0)
        self.gamma = gamma

    def act(self, s, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(N_ACTIONS))
        return int(np.argmax(self.q.forward(s[None, :])[0]))

    def learn(self, batch):
        s, a, r, s2, done = batch
        a_next = np.argmax(self.q.forward(s2), axis=1)                      # Double DQN: online net chooses
        q_next = self.target.forward(s2)[np.arange(len(a)), a_next]         # target net evaluates
        y = r + self.gamma * (1.0 - done) * q_next
        out = self.q.forward(s)
        err = out[np.arange(len(a)), a] - y
        grad = np.zeros_like(out)
        grad[np.arange(len(a)), a] = np.clip(err, -1.0, 1.0) / len(a)       # Huber-loss gradient
        self.q.backward(grad)
        self.opt.step(self.q.grads())
        return float((np.where(np.abs(err) < 1, 0.5 * err ** 2, np.abs(err) - 0.5)).mean())


def train_dqn(data, bp, feats, net_actual, cap, train_days, episodes=2500, seed=0, progress=None):
    """Train one agent on random days drawn from `train_days`.  Returns (agent, learning curve)."""
    rng = np.random.default_rng(seed)
    agent = DQNAgent(feats.shape[1] + 1, seed)
    price, fit = data.price, data.fit
    smin, smax = bp.soc_min_frac * cap, bp.soc_max_frac * cap
    tv = price.min() * bp.eta_discharge                                      # salvage value [AUD/kWh]
    N = 40000
    S, A, R, S2, D = (np.zeros((N, feats.shape[1] + 1)), np.zeros(N, int), np.zeros(N),
                      np.zeros((N, feats.shape[1] + 1)), np.zeros(N))
    n_buf, ptr, steps = 0, 0, 0
    total_steps = episodes * 24
    curve, ep_costs = [], []
    for ep in range(episodes):
        d = int(rng.choice(train_days))
        soc = rng.uniform(smin, smax)
        ep_cost = 0.0
        for h in range(24):
            t = d * 24 + h
            s = np.concatenate([[soc / cap], feats[t]])
            eps = max(0.05, 1.0 - steps / (0.6 * total_steps))
            a = agent.act(s, eps)
            soc, c, dd, imp, exp, cost = _step(soc, action_to_power(a, net_actual[t], bp.c_rate * cap), net_actual[t], price[t], fit[t], cap, bp)
            last = h == 23
            r = -cost / R_SCALE + (tv * (soc - smin) / R_SCALE if last else 0.0)
            tn = min(t + 1, len(net_actual) - 1)
            s2 = np.concatenate([[soc / cap], feats[tn]])
            S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = s, a, r, s2, float(last)
            ptr, n_buf, steps = (ptr + 1) % N, min(n_buf + 1, N), steps + 1
            ep_cost += cost
            if n_buf >= 1000:
                b = rng.integers(0, n_buf, 64)
                agent.learn((S[b], A[b], R[b], S2[b], D[b]))
                if steps % 400 == 0:
                    agent.target.copy_from(agent.q)
        ep_costs.append(ep_cost)
        if (ep + 1) % 100 == 0:
            curve.append(float(np.mean(ep_costs[-100:])))
            if progress:
                progress((ep + 1) / episodes)
    return agent, curve


# --------------------------------------------------------------------------
@dataclass
class RLResult:
    imp: np.ndarray
    exp: np.ndarray
    ch: np.ndarray
    dis: np.ndarray
    soc: np.ndarray
    energy_cost: float
    curves: list                      # learning curve per fold (mean daily cost of training episodes)
    comparison: pd.DataFrame          # annual community bill per controller


def greedy_pooled(net, price, fit, cap, bp: BatteryParams) -> float:
    """Simple rule: always absorb surplus / cover deficit with the battery (no price awareness)."""
    soc, cost = bp.soc_init_frac * cap, 0.0
    p_max = bp.c_rate * cap
    for t in range(len(net)):
        a = float(np.clip(-net[t] / p_max, -1, 1)) if cap > 0 else 0.0
        soc, _, _, _, _, c = _step(soc, a, net[t], price[t], fit[t], cap, bp)
        cost += c
    return cost


def run_rl(data: CommunityData, bp: BatteryParams, net_fc: np.ndarray, mpc_cost: float, oracle_cost: float,
           episodes: int = 2500, seed: int = 0, progress=None) -> RLResult:
    """Train two DQN agents (even/odd weeks), roll them out over the year, compare with other controllers."""
    net_actual = data.net_load().sum(axis=1).values
    cap = float(data.members["batt_kwh"].sum())
    feats = _features(data, net_actual, np.asarray(net_fc))
    days = np.arange(data.days)
    fold_of_day = (days // 7) % 2
    agents, curves = [], []
    for k in range(2):
        a, c = train_dqn(data, bp, feats, net_actual, cap, days[fold_of_day != k], episodes, seed + k,
                         progress=(lambda f, k=k: progress((k + f) / 2, f"DQN agent {k + 1}/2") if progress else None))
        agents.append(a)
        curves.append(c)

    T = data.T
    imp, exp, ch, dis, soc_arr = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T + 1)
    soc = bp.soc_init_frac * cap
    soc_arr[0] = soc
    for t in range(T):
        ag = agents[int(fold_of_day[min(t // 24, data.days - 1)])]   # agent that did NOT train on this day
        s = np.concatenate([[soc / cap], feats[t]])
        a = int(np.argmax(ag.q.forward(s[None, :])[0]))
        soc, ch[t], dis[t], imp[t], exp[t], _ = _step(soc, action_to_power(a, net_actual[t], bp.c_rate * cap), net_actual[t], data.price[t], data.fit[t], cap, bp)
        soc_arr[t + 1] = soc
    energy = float(data.price @ imp - data.fit @ exp)
    fee = data.fee_year * data.n
    comp = pd.Series({
        "No battery": float(data.price @ np.clip(net_actual, 0, None) - data.fit @ np.clip(-net_actual, 0, None)) + fee,
        "Greedy rule (no price awareness)": greedy_pooled(net_actual, data.price, data.fit, cap, bp) + fee,
        "DQN agent (deep RL, out-of-sample)": energy + fee,
        "MPC with forecasts (optimisation)": mpc_cost,
        "MPC perfect foresight (upper bound)": oracle_cost,
    }, name="community_bill_AUD").to_frame()
    return RLResult(imp, exp, ch, dis, soc_arr, energy, curves, comp)
