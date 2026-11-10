"""End-to-end pipeline used by both the CLI (main.py) and the Streamlit app (app.py)."""
from __future__ import annotations

import hashlib
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import allocation as A
from . import metrics as M
from . import scenarios as S
from .config import BatteryParams
from .data_loader import CommunityData, load_dataset
from . import config as C
from .forecasting import ALL_MODELS, MODEL_FAMILY, Forecasts, build_all_forecasts
from .optimizer import Dispatch
from .rl_agent import RLResult, run_rl

# Numbers printed in the benchmark analysis PDF (for reproduction checks)
REPORT_BASELINE = dict(zip([f"M{i}" for i in range(1, 10)],
                           [1720, 1509, 2881, 2212, 2237, 2613, 3399, 2944, 3097]))
REPORT_STANDALONE = dict(zip([f"M{i}" for i in range(1, 10)],
                             [603, 1508, -75, 582, 2237, 196, 3398, 718, 1978]))
REPORT_LEM = dict(zip([f"M{i}" for i in range(1, 10)],
                      [147, 976, -118, 65, 1362, 226, 2005, 771, 1186]))


@dataclass
class PipelineConfig:
    data_path: str | None = None
    pv_mode: str = "report"           # "report" | "realistic"
    use_ml: bool = True               # drive the MPC with ML forecasts (else perfect foresight)
    forecaster: str = "auto"          # "auto" (lowest community bill) or any trained model / "ensemble"
    models: tuple = ALL_MODELS        # forecasters to train and compare (gbm, mlp, lstm, gru, tcn, transformer)
    dl_epochs: int = 20               # max epochs of the NumPy sequence models (early stopping applies)
    use_cache: bool = True            # re-use trained forecasts / Shapley table from outputs/cache
    use_rl: bool = False              # train the DQN battery agent (adds ~1 min)
    rl_episodes: int = 2000
    rep_days: int = 24                # representative days for the Shapley game (365 = exact)
    equity_lambda: float = 0.5        # 0 = pure Shapley (merit)  ->  1 = pure need
    nwo_alpha: float = 0.6            # equity mechanism guarantees >= alpha * Shapley saving
    peak_weight: float = 0.0          # AUD/kW penalty on daily peak grid import (grid support)
    battery: BatteryParams = field(default_factory=BatteryParams)


@dataclass
class Results:
    cfg: PipelineConfig
    data: CommunityData
    baseline: pd.Series
    greedy: S.HouseholdSet
    standalone: S.HouseholdSet            # primary information level (ML or oracle)
    standalone_oracle: S.HouseholdSet
    community: Dispatch
    community_oracle: Dispatch
    forecasts: Forecasts | None
    forecast_sets: dict                    # every trained forecaster {name: Forecasts}
    forecast_value: pd.DataFrame           # community bill obtained with each forecaster
    phi: np.ndarray
    shapley_diag: dict
    rep_info: dict
    tables: dict                           # mechanism -> per-household DataFrame
    summary: pd.DataFrame                  # mechanism comparison
    kpis: pd.DataFrame                     # physical KPIs per scenario
    report_check: pd.DataFrame
    timings: dict
    rl: RLResult | None = None
    model_ranking: pd.DataFrame | None = None


def _cached(cfg: PipelineConfig, name: str, key: dict, fn):
    """Disk cache for the slow, deterministic steps (forecast training, Shapley table)."""
    if not cfg.use_cache:
        return fn()
    src = __import__("pathlib").Path(cfg.data_path) if cfg.data_path else C.DEFAULT_DATA
    st = src.stat() if src.exists() else None
    raw = repr((name, sorted(key.items()), st.st_size if st else 0, int(st.st_mtime) if st else 0, cfg.battery))
    path = C.DEFAULT_OUTPUT / "cache" / f"{name}_{hashlib.md5(raw.encode()).hexdigest()[:12]}.pkl"
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            pass
    val = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(val))
    return val


def run_pipeline(cfg: PipelineConfig | None = None, progress=None, shapley_cache: np.ndarray | None = None
                 ) -> Results:
    cfg = cfg or PipelineConfig()
    bp = cfg.battery
    tm: dict = {}
    say = progress or (lambda f, msg: None)

    def tick(name, t0):
        tm[name] = time.time() - t0

    t0 = time.time()
    say(0.02, "Loading dataset")
    data = load_dataset(cfg.data_path, cfg.pv_mode)
    tick("load", t0)

    t0 = time.time()
    say(0.08, "Baseline & standalone scenarios")
    baseline = S.baseline_cost(data)
    greedy = S.standalone_greedy(data, bp)
    standalone_oracle = S.standalone_optimal(data, bp)
    community_oracle = S.community_dispatch(data, bp, peak_weight=cfg.peak_weight)
    tick("scenarios_oracle", t0)

    forecasts, forecast_sets = None, {}
    forecast_value = pd.DataFrame({"community_bill_AUD": {"Perfect foresight (oracle)": community_oracle.cost + data.fee_year * data.n}})
    standalone, community = standalone_oracle, community_oracle
    if cfg.use_ml:
        t0 = time.time()
        kinds = tuple(cfg.models)
        say(0.20, "Training forecasters (out-of-fold): " + ", ".join(kinds))
        forecast_sets = _cached(cfg, "forecasts", dict(kinds=kinds, ep=cfg.dl_epochs, pv=cfg.pv_mode),
                                lambda: build_all_forecasts(data, kinds, dl_epochs=cfg.dl_epochs,
                                                            progress=lambda f, m: say(0.20 + 0.25 * f, f"Training forecaster: {m}")))
        tick("forecasting", t0)
        t0 = time.time()
        say(0.46, "Evaluating forecasters (community MPC bill)")
        fee_all = data.fee_year * data.n
        fv = {"Perfect foresight (oracle)": community_oracle.cost + fee_all}
        for name, fc in forecast_sets.items():
            fv[name] = S.community_dispatch(data, bp, fc.net_hat().sum(axis=1).values,
                                            peak_weight=cfg.peak_weight).cost + fee_all
        naive = data.net_load().sum(axis=1).shift(24).fillna(0.0).values
        fv["Naive persistence (yesterday)"] = S.community_dispatch(data, bp, naive, peak_weight=cfg.peak_weight).cost + fee_all
        forecast_value = pd.Series(fv, name="community_bill_AUD").to_frame()
        if cfg.forecaster == "auto":
            best = min(forecast_sets, key=lambda k: fv[k])
        else:
            best = cfg.forecaster if cfg.forecaster in forecast_sets else min(forecast_sets, key=lambda k: fv[k])
        forecasts = forecast_sets[best]
        say(0.50, f"Forecast-driven MPC dispatch (forecaster: {best})")
        nf = forecasts.net_hat()
        standalone = S.standalone_optimal(data, bp, nf)
        community = S.community_dispatch(data, bp, nf.sum(axis=1).values, peak_weight=cfg.peak_weight)
        tick("scenarios_ml", t0)

    # ---- cooperative game (Shapley) ---------------------------------------
    t0 = time.time()
    say(0.55, f"Shapley game: clustering days (k={cfg.rep_days}) and solving 511 coalitions")
    days, weights = A.representative_days(data, cfg.rep_days)
    if shapley_cache is not None:
        Cc = shapley_cache
    else:
        Cc = _cached(cfg, "shapley", dict(k=cfg.rep_days, pv=cfg.pv_mode),
                     lambda: A.coalition_costs(data, bp, days, weights, progress=lambda f: say(0.55 + 0.3 * f, "Shapley game")))
    phi, v, diag = A.shapley_savings(Cc, data.n)
    tick("shapley", t0)
    grand_year_oracle = community_oracle.cost
    diag["rep_day_cost_error_pct"] = 100 * (Cc[-1] - grand_year_oracle) / abs(grand_year_oracle)
    rep_info = dict(days=days, weights=weights, coalition_costs=Cc, v=v)

    # ---- settlement mechanisms -------------------------------------------------
    t0 = time.time()
    say(0.9, "Allocating benefits (market prices, Shapley, equity)")
    ref = standalone.cost
    income = data.members["income"]
    community_total = community.cost + data.fee_year * data.n
    total_saving = float(ref.sum() - community_total)

    mech: dict[str, pd.Series] = {}
    bills, det_mid = A.settle_market(data, community, "mid_market")
    mech["LEM mid-market price (report style)"] = bills
    mech["LEM mid-market + no-worse-off"] = A.enforce_no_worse_off(bills, ref)
    bills_sdr, det_sdr = A.settle_market(data, community, "sdr")
    mech["LEM supply/demand price"] = bills_sdr
    mech["LEM supply/demand + no-worse-off"] = A.enforce_no_worse_off(bills_sdr, ref)
    mech["Shapley (merit)"] = A.shapley_bills(ref, total_saving, phi)
    mech["Equity-adjusted Shapley (recommended)"] = A.equity_bills(
        ref, total_saving, phi, income, cfg.equity_lambda, cfg.nwo_alpha)

    phi_year = total_saving * (phi / phi.sum()) if phi.sum() > 0 else np.zeros(data.n)
    tables = {k: M.fairness_table(b, ref, baseline, income, phi_year) for k, b in mech.items()}
    rows = {}
    for k, t in tables.items():
        s = M.fairness_summary(t)
        s["community_cost_AUD"] = float(t["bill_AUD"].sum())
        rows[k] = s
    summary = pd.DataFrame(rows).T
    tick("allocation", t0)

    # ---- physical KPIs ----------------------------------------------------------
    zeros = np.zeros(data.T)
    kp = {
        "Baseline (grid only)": M.community_kpis(data, data.demand.values.sum(axis=1), zeros,
                                                  cost_total=float(baseline.sum())),
        "Standalone - greedy": M.community_kpis(data, greedy.imp.values, greedy.exp.values, greedy.ch.values,
                                                 greedy.dis.values, float(greedy.cost.sum())),
        "Standalone - optimised": M.community_kpis(data, standalone.imp.values, standalone.exp.values,
                                                    standalone.ch.values, standalone.dis.values,
                                                    float(standalone.cost.sum())),
        "Community (pooled MPC)": M.community_kpis(data, community.imp, community.exp, community.ch,
                                                    community.dis, community_total),
    }
    if cfg.use_ml:
        kp["Community (perfect foresight)"] = M.community_kpis(
            data, community_oracle.imp, community_oracle.exp, community_oracle.ch, community_oracle.dis,
            community_oracle.cost + data.fee_year * data.n)
    kp["Baseline (grid only)"]["pv_utilisation_pct"] = 0.0     # DER ignored in this scenario
    kp["Baseline (grid only)"]["pv_utilisation_pct"] = 0.0     # baseline ignores PV
    kpis = pd.DataFrame(kp).T

    rl = None
    if cfg.use_rl:
        t0 = time.time()
        say(0.95, "Training deep Q-network battery agent")
        net_fc = (forecasts.net_hat().sum(axis=1).values if forecasts is not None
                  else data.net_load().sum(axis=1).values)
        rl = run_rl(data, bp, net_fc, community_total, community_oracle.cost + data.fee_year * data.n,
                    episodes=cfg.rl_episodes, progress=lambda f, m: say(0.95 + 0.04 * f, m))
        tick("rl", t0)

    report_check = pd.DataFrame({
        "baseline_ours": baseline, "baseline_report": pd.Series(REPORT_BASELINE),
        "standalone_greedy_ours": greedy.cost, "standalone_report": pd.Series(REPORT_STANDALONE),
        "standalone_optimised_ours": standalone_oracle.cost,
        "LEM_report": pd.Series(REPORT_LEM),
        "recommended_ours": mech["Equity-adjusted Shapley (recommended)"],
    }).round(0)

    ranking = None
    if forecast_sets:
        rows = {}
        for name, fc in forecast_sets.items():
            m_ = fc.metrics
            info = fc.info or {}
            rows[name] = {"family": MODEL_FAMILY.get(name, name),
                          "parameters": info.get("params", float("nan")),
                          "train_seconds": info.get("seconds", float("nan")),
                          "community_RMSE_kW": m_.loc["demand COMMUNITY", "RMSE"],
                          "GHI_RMSE_Wm2": m_.loc["GHI", "RMSE"],
                          "mean_skill_vs_naive_pct": m_["RMSE_skill_vs_naive_pct"].mean(),
                          "community_bill_AUD": float(forecast_value.loc[name, "community_bill_AUD"])}
        ranking = pd.DataFrame(rows).T
        ranking["rank_by_RMSE"] = ranking["community_RMSE_kW"].astype(float).rank().astype(int)
        ranking = ranking.sort_values("rank_by_RMSE")
    say(1.0, "Done")
    return Results(cfg, data, baseline, greedy, standalone, standalone_oracle, community, community_oracle,
                   forecasts, forecast_sets, forecast_value, phi, diag, rep_info, tables, summary, kpis, report_check, tm, rl, ranking)
