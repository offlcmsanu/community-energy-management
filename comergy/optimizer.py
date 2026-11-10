"""Linear-programming dispatch of (pooled) batteries + rolling-horizon MPC.

Problem solved for one horizon of T hours (all arrays in kW / kWh per hour):

    min   sum_t  price_t * imp_t - fit_t * exp_t  - tv * soc_T  + w_peak * peak + eps*(ch+dis)
    s.t.  imp_t - exp_t - ch_t + dis_t = net_t                 (power balance, net = demand - PV)
          soc_{t+1} = soc_t + eta_c * ch_t - dis_t / eta_d      (battery energy balance)
          0 <= ch,dis <= P ;  soc_min <= soc <= soc_max ;  imp_t <= peak
          soc_0 = given   (or soc_T = soc_0 for "cyclic" representative days)

Because price > feed-in tariff at every hour, the LP never imports and exports
in the same hour, and the small eps term prevents pointless simultaneous
charge / discharge.  The problem is a pure LP -> solved to global optimality
by HiGHS (SciPy) in a few milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .config import BatteryParams

EPS_THROUGHPUT = 1e-5   # AUD/kWh tiny penalty on battery throughput


@dataclass
class Dispatch:
    imp: np.ndarray
    exp: np.ndarray
    ch: np.ndarray
    dis: np.ndarray
    soc: np.ndarray            # length T+1 [kWh]
    cost: float                # energy cost  sum(price*imp - fit*exp)   [AUD]


# --------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _structure(T: int, cyclic: bool, use_peak: bool):
    """Constraint matrices depend only on (T, cyclic, use_peak) -> cache them."""
    # variable layout: imp[T] exp[T] ch[T] dis[T] soc[T+1] peak[1]
    iI, iE, iC, iD = 0, T, 2 * T, 3 * T
    iS = 4 * T
    iP = iS + T + 1
    n = iP + 1
    I = sp.identity(T, format="csr")
    # power balance rows (T)
    Aeq1 = sp.hstack([I, -I, -I, I, sp.csr_matrix((T, T + 1)), sp.csr_matrix((T, 1))], format="csr")
    # soc dynamics rows (T):  soc_{t+1} - soc_t - eta_c*ch + dis/eta_d = 0   (eta applied per call)
    S_next = sp.hstack([sp.csr_matrix((T, 1)), I], format="csr")
    S_curr = sp.hstack([I, sp.csr_matrix((T, 1))], format="csr")
    Sdiff = (S_next - S_curr).tocsr()
    Z = sp.csr_matrix((T, T))
    extra = None
    if cyclic:
        e = sp.lil_matrix((1, n))
        e[0, iS + T] = 1.0
        e[0, iS] = -1.0
        extra = e.tocsr()
    Aub = None
    if use_peak:
        Aub = sp.hstack([I, sp.csr_matrix((T, 3 * T)), sp.csr_matrix((T, T + 1)),
                         -sp.csr_matrix(np.ones((T, 1)))], format="csr")
    return dict(n=n, iI=iI, iE=iE, iC=iC, iD=iD, iS=iS, iP=iP, Aeq1=Aeq1, Sdiff=Sdiff, Z=Z,
                I=I, extra=extra, Aub=Aub)


def solve_battery(net, price, fit, cap_kwh: float, bp: BatteryParams, soc0: float | None = None,
                  cyclic: bool = False, terminal_value: float | None = None,
                  peak_weight: float = 0.0) -> Dispatch:
    """Optimal dispatch of one (pooled) battery. cap_kwh == 0 -> no battery."""
    net = np.asarray(net, dtype=float)
    price = np.asarray(price, dtype=float)
    fit = np.asarray(fit, dtype=float)
    T = len(net)

    if cap_kwh <= 1e-9:  # no storage: closed form
        imp, exp = np.clip(net, 0, None), np.clip(-net, 0, None)
        z = np.zeros(T)
        return Dispatch(imp, exp, z, z.copy(), np.zeros(T + 1), float(price @ imp - fit @ exp))

    p_max = bp.c_rate * cap_kwh
    smin, smax = bp.soc_min_frac * cap_kwh, bp.soc_max_frac * cap_kwh
    if soc0 is None:
        soc0 = bp.soc_init_frac * cap_kwh
    soc0 = float(np.clip(soc0, smin, smax))
    use_peak = peak_weight > 0
    st = _structure(T, cyclic, use_peak)
    n = st["n"]

    # equality constraints ------------------------------------------------
    Z, I = st["Z"], st["I"]
    soc_rows = sp.hstack([Z, Z, -bp.eta_charge * I, (1.0 / bp.eta_discharge) * I, st["Sdiff"],
                          sp.csr_matrix((T, 1))], format="csr")
    blocks = [st["Aeq1"], soc_rows]
    beq = [net, np.zeros(T)]
    if cyclic:
        blocks.append(st["extra"])
        beq.append(np.zeros(1))
    Aeq = sp.vstack(blocks, format="csr")
    beq = np.concatenate(beq)

    # objective -----------------------------------------------------------
    c = np.zeros(n)
    c[st["iI"]:st["iI"] + T] = price
    c[st["iE"]:st["iE"] + T] = -fit
    c[st["iC"]:st["iC"] + T] = EPS_THROUGHPUT
    c[st["iD"]:st["iD"] + T] = EPS_THROUGHPUT
    if not cyclic:
        tv = (price.min() * bp.eta_discharge) if terminal_value is None else terminal_value
        c[st["iS"] + T] = -tv
    c[st["iP"]] = peak_weight

    # bounds --------------------------------------------------------------
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    ub[st["iC"]:st["iC"] + T] = p_max
    ub[st["iD"]:st["iD"] + T] = p_max
    lb[st["iS"]:st["iS"] + T + 1] = smin
    ub[st["iS"]:st["iS"] + T + 1] = smax
    if not cyclic:
        lb[st["iS"]] = ub[st["iS"]] = soc0
    bounds = np.column_stack([lb, ub])

    kw = {}
    if use_peak:
        kw = dict(A_ub=st["Aub"], b_ub=np.zeros(T))
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs", **kw)
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x
    imp = x[st["iI"]:st["iI"] + T]
    exp = x[st["iE"]:st["iE"] + T]
    ch = x[st["iC"]:st["iC"] + T]
    dis = x[st["iD"]:st["iD"] + T]
    soc = x[st["iS"]:st["iS"] + T + 1]
    return Dispatch(imp, exp, ch, dis, soc, float(price @ imp - fit @ exp))


# --------------------------------------------------------------------------
def rolling_dispatch(net_actual, net_forecast, price, fit, cap_kwh: float, bp: BatteryParams,
                     horizon: int = 48, step: int = 24, peak_weight: float = 0.0,
                     reactive: bool | None = None) -> Dispatch:
    """Receding-horizon (MPC) control.

    Every `step` hours the LP is solved over `horizon` hours using the *forecast* net load and the
    current state of charge.  The first `step` hours are then executed against the *actual* net load.

    reactive=True adds a real-time layer on top of the day-ahead plan (used automatically when the
    forecast differs from reality):
        * PV surplus that exceeds the planned charge is absorbed while there is headroom
        * in hours where the plan discharges - or in the highest-price tier - the battery covers
          the *actual* household deficit instead of just the planned amount
        * energy held above the plan's own SOC trajectory is spent on actual deficits
    This is what a real home-energy-management controller does and recovers most of the value
    that a rigid open-loop schedule loses to forecast errors.
    """
    net_actual = np.asarray(net_actual, float)
    net_forecast = np.asarray(net_forecast, float)
    price = np.asarray(price, float)
    fit = np.asarray(fit, float)
    T = len(net_actual)
    if reactive is None:
        reactive = not np.allclose(net_actual, net_forecast)
    if cap_kwh <= 1e-9:
        return solve_battery(net_actual, price, fit, 0.0, bp)

    imp, exp = np.zeros(T), np.zeros(T)
    ch, dis = np.zeros(T), np.zeros(T)
    soc = np.zeros(T + 1)
    p_max = bp.c_rate * cap_kwh
    smin, smax = bp.soc_min_frac * cap_kwh, bp.soc_max_frac * cap_kwh
    s = bp.soc_init_frac * cap_kwh
    soc[0] = s
    top_price = price.max()
    for t0 in range(0, T, step):
        H = min(horizon, T - t0)
        sol = solve_battery(net_forecast[t0:t0 + H], price[t0:t0 + H], fit[t0:t0 + H], cap_kwh, bp,
                            soc0=s, peak_weight=peak_weight)
        for k in range(min(step, T - t0)):
            t = t0 + k
            n = net_actual[t]
            c_p, d_p = sol.ch[k], sol.dis[k]
            head = (smax - s) / bp.eta_charge
            avail = (s - smin) * bp.eta_discharge
            if reactive:
                if n < 0:                                   # surplus -> absorb as much as possible
                    c_a, d_a = min(max(c_p, -n), p_max, head), 0.0
                elif d_p > 1e-9 or (price[t] >= top_price - 1e-9 and price[t] > fit[t]):
                    c_a, d_a = 0.0, min(n, p_max, avail)    # cover the actual deficit
                elif n > 0 and s > sol.soc[k] + 1e-9:       # energy above the plan's own reserve is free to use
                    c_a, d_a = 0.0, min(n, p_max, avail, (s - sol.soc[k]) * bp.eta_discharge)
                else:                                       # follow the plan (e.g. grid pre-charge)
                    c_a, d_a = min(c_p, p_max, head), 0.0
            else:                                           # open loop: execute the plan as is
                d_a = min(d_p, max(n, 0.0), avail, p_max)
                c_a = min(c_p, head, p_max)
                if c_a > 0 and d_a > 0:
                    d_a = 0.0
            s = s + bp.eta_charge * c_a - d_a / bp.eta_discharge
            flow = n + c_a - d_a
            imp[t], exp[t] = max(flow, 0.0), max(-flow, 0.0)
            ch[t], dis[t], soc[t + 1] = c_a, d_a, s
    return Dispatch(imp, exp, ch, dis, soc, float(price @ imp - fit @ exp))
