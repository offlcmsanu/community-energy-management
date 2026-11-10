"""KPIs: economic, grid / environmental and fairness (energy-justice) metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data_loader import CommunityData


def gini(x) -> float:
    """Gini coefficient (0 = perfect equality). Negative values are shifted to be >= 0."""
    x = np.asarray(x, dtype=float)
    x = x - min(x.min(), 0.0)
    if x.sum() <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def jain(x) -> float:
    """Jain's fairness index (1 = perfectly equal, 1/n = maximally unfair)."""
    x = np.asarray(x, dtype=float)
    return float(x.sum() ** 2 / (len(x) * (x ** 2).sum())) if (x ** 2).sum() > 0 else 1.0


def community_kpis(data: CommunityData, imp, exp, ch=None, dis=None, cost_total=None) -> dict:
    """Grid / environmental KPIs from hourly community import & export [kW].
    imp / exp may be 1-D (community) or 2-D (per household, summed here)."""
    imp = np.asarray(imp, float)
    exp = np.asarray(exp, float)
    imp1 = imp.sum(axis=1) if imp.ndim == 2 else imp
    exp1 = exp.sum(axis=1) if exp.ndim == 2 else exp
    demand, pv = float(data.demand.values.sum()), float(data.pv.values.sum())
    peak_hours = data.price >= data.price.max() - 1e-9
    cap = float(data.members["batt_kwh"].sum())
    throughput = float(np.sum(dis)) if dis is not None else 0.0
    # peak import measured on the community net exchange (netted across households if 2-D)
    return dict(
        grid_import_kWh=float(imp1.sum()),
        grid_export_kWh=float(exp1.sum()),
        self_sufficiency_pct=100 * (1 - imp1.sum() / demand),
        pv_utilisation_pct=100 * (1 - exp1.sum() / pv) if pv > 0 else 0.0,
        peak_import_kW=float(imp1.max()),
        peak_tier_import_kWh=float(imp1[peak_hours].sum()),
        co2_t=float(imp1.sum() * C.GRID_EMISSION_FACTOR / 1000.0),
        battery_cycles=(throughput / cap) if cap > 0 else 0.0,
        total_cost_AUD=cost_total,
    )


def fairness_table(bills: pd.Series, ref: pd.Series, baseline: pd.Series, income: pd.Series,
                   phi_savings: np.ndarray | None = None) -> pd.DataFrame:
    """Per-household outcome table for one mechanism."""
    t = pd.DataFrame(index=bills.index)
    t["bill_AUD"] = bills
    t["standalone_AUD"] = ref
    t["baseline_AUD"] = baseline
    t["saving_vs_standalone_AUD"] = ref - bills
    t["saving_vs_standalone_pct"] = 100 * (ref - bills) / ref.abs().clip(lower=1.0)
    t["saving_vs_baseline_pct"] = 100 * (baseline - bills) / baseline
    t["income_AUD"] = income
    t["energy_burden_pct"] = 100 * bills / income
    if phi_savings is not None:
        t["shapley_saving_AUD"] = phi_savings
    return t


def fairness_summary(table: pd.DataFrame) -> dict:
    s = table["saving_vs_standalone_AUD"]
    return dict(
        total_saving_AUD=float(s.sum()),
        min_saving_AUD=float(s.min()),
        no_worse_off_violations=int((s < -1e-6).sum()),
        worst_loss_AUD=float(min(s.min(), 0.0)),
        burden_gini=gini(table["energy_burden_pct"]),
        burden_spread_pp=float(table["energy_burden_pct"].max() - table["energy_burden_pct"].min()),
        jain_saving_pct=jain(table["saving_vs_baseline_pct"].clip(lower=0)),
        lowest_income_saving_pct=float(table.loc[table["income_AUD"].idxmin(), "saving_vs_baseline_pct"]),
    )
