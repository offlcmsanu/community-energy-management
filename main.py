"""Command-line entry point:  python main.py  [options]"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from comergy.config import DEFAULT_OUTPUT
from comergy.pipeline import PipelineConfig, run_pipeline
from comergy.reporting import export_excel, make_figures


def main(argv=None):
    ap = argparse.ArgumentParser(description="ComErgy 2026 - fair community energy market")
    ap.add_argument("--data", help="path to ComErgy_Benchmark_Dataset_2026.xlsx (default: data/ folder)")
    ap.add_argument("--out", default=str(DEFAULT_OUTPUT), help="output folder")
    ap.add_argument("--pv-mode", choices=["report", "realistic"], default="report")
    ap.add_argument("--no-ml", action="store_true", help="use perfect foresight instead of ML forecasts")
    ap.add_argument("--forecaster", default="auto",
                    help="model that drives the MPC: auto (lowest bill) | gbm | mlp | lstm | gru | tcn | transformer | ensemble")
    ap.add_argument("--models", default="gbm,mlp,lstm,gru,tcn,transformer",
                    help="comma-separated forecasters to train and compare")
    ap.add_argument("--fast", action="store_true", help="only gbm + mlp (skips the NumPy sequence models)")
    ap.add_argument("--dl-epochs", type=int, default=20, help="max epochs of the sequence models")
    ap.add_argument("--no-cache", action="store_true", help="ignore outputs/cache (retrain everything)")
    ap.add_argument("--rl", action="store_true", help="also train the deep Q-network battery agent (~1-2 min)")
    ap.add_argument("--rl-episodes", type=int, default=2000)
    ap.add_argument("--rep-days", type=int, default=24, help="representative days for Shapley (365=exact, slow)")
    ap.add_argument("--equity-lambda", type=float, default=0.5, help="0=merit (Shapley) ... 1=need")
    ap.add_argument("--nwo-alpha", type=float, default=0.6, help="guaranteed share of own Shapley saving")
    ap.add_argument("--peak-weight", type=float, default=0.0, help="AUD/kW penalty on peak grid import")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)

    models = ("gbm", "mlp") if a.fast else tuple(m.strip() for m in a.models.split(",") if m.strip())
    cfg = PipelineConfig(a.data, a.pv_mode, not a.no_ml, a.forecaster, models, a.dl_epochs, not a.no_cache, a.rl,
                         a.rl_episodes, a.rep_days, a.equity_lambda, a.nwo_alpha, a.peak_weight)
    last = [-1]

    def prog(f, msg):
        if int(f * 20) != last[0] or "Shapley" not in msg:
            last[0] = int(f * 20)
            print(f"[{f * 100:5.1f}%] {msg}")

    res = run_pipeline(cfg, prog)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== Physical KPIs ===\n", res.kpis.round(1))
    print("\n=== Mechanism comparison ===\n", res.summary.round(2).T)
    print("\n=== Recommended mechanism: per household ===\n",
          res.tables["Equity-adjusted Shapley (recommended)"][
              ["bill_AUD", "standalone_AUD", "saving_vs_standalone_AUD", "saving_vs_baseline_pct",
               "energy_burden_pct"]].round(1))
    if res.forecasts is not None:
        acc = pd.DataFrame({k: v.metrics["RMSE_skill_vs_naive_pct"] for k, v in res.forecast_sets.items()})
        rmse = pd.DataFrame({k: v.metrics["RMSE"] for k, v in res.forecast_sets.items()})
        print("\n=== Forecast RMSE [kW or W/m2] (out-of-fold; gbm = trees, mlp = neural network, lstm = recurrent network) ===\n", rmse.round(3))
        print("\n=== RMSE skill vs yesterday-persistence [%] ===\n", acc.round(1))
        print("\n=== Value of forecasting: community annual bill when each model drives the MPC ===\n",
              res.forecast_value.round(0))
    if res.model_ranking is not None:
        print("\n=== MODEL COMPARISON (lower RMSE / bill is better) ===\n", res.model_ranking.round(3).to_string())
    if res.forecasts is not None:
        print(f"\nForecaster driving the MPC: {res.forecasts.model}")
    if res.rl is not None:
        print("\n=== Deep RL battery agent vs optimisation (community annual bill, AUD) ===\n", res.rl.comparison.round(0))
    out = Path(a.out)
    print("\nExcel :", export_excel(res, out / "comergy_results.xlsx"))
    if not a.no_plots:
        for f in make_figures(res, out):
            print("Figure:", f)
    print("Timings [s]:", {k: round(v, 1) for k, v in res.timings.items()})


if __name__ == "__main__":
    sys.exit(main())
