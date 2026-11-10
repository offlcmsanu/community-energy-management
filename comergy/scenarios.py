"""Operating scenarios: baseline, standalone (greedy / optimised) and community dispatch."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BatteryParams
from .data_loader import CommunityData
from .optimizer import Dispatch, rolling_dispatch


@dataclass
class HouseholdSet:
    """Per-household physical flows (kW) and energy cost (AUD, without connection fee)."""
    imp: pd.DataFrame
    exp: pd.DataFrame
    ch: pd.DataFrame
    dis: pd.DataFrame
    energy_cost: pd.Series
    fee: float

    @property
    def cost(self) -> pd.Series:           # incl. connection fee
        return self.energy_cost + self.fee


# --------------------------------------------------------------------------
def baseline_cost(data: CommunityData) -> pd.Series:
    """Scenario 1: everything bought from the grid, DER ignored."""
    return data.demand.mul(data.price, axis=0).sum() + data.fee_year


def _pack(data, imp, exp, ch, dis) -> HouseholdSet:
    mk = lambda a: pd.DataFrame(a, index=data.index, columns=data.ids)  # noqa: E731
    imp, exp, ch, dis = mk(imp), mk(exp), mk(ch), mk(dis)
    cost = imp.mul(data.price, axis=0).sum() - exp.mul(data.fit, axis=0).sum()
    return HouseholdSet(imp, exp, ch, dis, cost, data.fee_year)


def standalone_greedy(data: CommunityData, bp: BatteryParams) -> HouseholdSet:
    """Scenario 2a: 'naive' self-management - PV -> load -> battery -> grid, battery
    discharges whenever there is a deficit (no price awareness)."""
    T, N = data.T, data.n
    net = data.net_load().values
    imp, exp, ch, dis = (np.zeros((T, N)) for _ in range(4))
    for j, m in enumerate(data.ids):
        cap = data.members.loc[m, "batt_kwh"]
        p, smin, smax = bp.c_rate * cap, bp.soc_min_frac * cap, bp.soc_max_frac * cap
        s = bp.soc_init_frac * cap
        for t in range(T):
            n = net[t, j]
            if cap > 0 and n < 0:               # surplus -> charge
                c = min(-n, p, (smax - s) / bp.eta_charge)
                s += bp.eta_charge * c
                ch[t, j] = c
                n += c
            elif cap > 0 and n > 0:             # deficit -> discharge
                d = min(n, p, (s - smin) * bp.eta_discharge)
                s -= d / bp.eta_discharge
                dis[t, j] = d
                n -= d
            imp[t, j], exp[t, j] = max(n, 0.0), max(-n, 0.0)
    return _pack(data, imp, exp, ch, dis)


def standalone_optimal(data: CommunityData, bp: BatteryParams,
                       net_forecast: pd.DataFrame | None = None) -> HouseholdSet:
    """Scenario 2b: every household optimises its *own* battery (price-aware MPC).
    This is the strongest individual alternative -> the 'no-worse-off' reference.
    net_forecast=None -> perfect foresight (oracle); otherwise ML forecasts are used."""
    net = data.net_load()
    T, N = data.T, data.n
    imp, exp, ch, dis = (np.zeros((T, N)) for _ in range(4))
    for j, m in enumerate(data.ids):
        nf = net[m].values if net_forecast is None else net_forecast[m].values
        d = rolling_dispatch(net[m].values, nf, data.price, data.fit, data.members.loc[m, "batt_kwh"], bp)
        imp[:, j], exp[:, j], ch[:, j], dis[:, j] = d.imp, d.exp, d.ch, d.dis
    return _pack(data, imp, exp, ch, dis)


def community_dispatch(data: CommunityData, bp: BatteryParams, net_forecast_total=None,
                       peak_weight: float = 0.0) -> Dispatch:
    """Scenario 3 (physical layer): all batteries pooled into one *virtual* community
    battery, dispatched by MPC to minimise the community's grid bill."""
    net_total = data.net_load().sum(axis=1).values
    nf = net_total if net_forecast_total is None else np.asarray(net_forecast_total)
    cap = float(data.members["batt_kwh"].sum())
    return rolling_dispatch(net_total, nf, data.price, data.fit, cap, bp, peak_weight=peak_weight)
