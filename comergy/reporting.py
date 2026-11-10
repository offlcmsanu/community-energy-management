"""Excel export and static (matplotlib) figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .pipeline import Results  # noqa: E402

RECOMMENDED = "Equity-adjusted Shapley (recommended)"


def export_excel(res: Results, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        res.summary.round(3).to_excel(xw, sheet_name="Mechanism comparison")
        res.kpis.round(3).to_excel(xw, sheet_name="Physical KPIs")
        res.report_check.to_excel(xw, sheet_name="Check vs report")
        for name, t in res.tables.items():
            t.round(2).to_excel(xw, sheet_name=name.replace("/", "-")[:31])
        for name, fc in res.forecast_sets.items():
            fc.metrics.round(4).to_excel(xw, sheet_name=f"Forecast {name}"[:31])
        res.forecast_value.round(1).to_excel(xw, sheet_name="Value of forecasting")
        if res.model_ranking is not None:
            res.model_ranking.round(3).to_excel(xw, sheet_name="Model ranking")
        if res.rl is not None:
            res.rl.comparison.round(1).to_excel(xw, sheet_name="Deep RL vs MPC")
        hourly = pd.DataFrame({"grid_import_kW": res.community.imp, "grid_export_kW": res.community.exp,
                               "battery_charge_kW": res.community.ch, "battery_discharge_kW": res.community.dis,
                               "battery_soc_kWh": res.community.soc[1:]}, index=res.data.index)
        hourly.round(3).to_excel(xw, sheet_name="Community hourly")
    return path


def make_figures(res: Results, outdir) -> list:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    ids = res.data.ids

    # 1 annual bills per scenario ------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.2
    x = np.arange(len(ids))
    ax.bar(x - 1.5 * w, res.baseline, w, label="Baseline")
    ax.bar(x - 0.5 * w, res.standalone.cost, w, label="Standalone (optimised)")
    ax.bar(x + 0.5 * w, res.tables["LEM mid-market price (report style)"]["bill_AUD"], w, label="LEM mid-market")
    ax.bar(x + 1.5 * w, res.tables[RECOMMENDED]["bill_AUD"], w, label="Equity-adjusted Shapley")
    ax.set_xticks(x, ids)
    ax.set_ylabel("Annual bill [AUD]")
    ax.legend()
    ax.set_title("Annual electricity bill per household")
    p = out / "bills.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)

    # 2 savings vs standalone, mechanisms -----------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    names = list(res.tables)
    w = 0.8 / len(names)
    for k, n in enumerate(names):
        ax.bar(x + (k - len(names) / 2 + 0.5) * w, res.tables[n]["saving_vs_standalone_AUD"], w, label=n)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, ids)
    ax.set_ylabel("Saving vs standalone [AUD/yr]")
    ax.legend(fontsize=7)
    ax.set_title("Who gains what?  (bars below zero = household worse off)")
    p = out / "savings_by_mechanism.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)

    # 3 typical summer & winter week dispatch ----------------------------------
    d, c = res.data, res.community
    for label, start in (("summer", 10 * 24), ("winter", 190 * 24)):
        sl = slice(start, start + 7 * 24)
        t = np.arange(7 * 24)
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(t, d.demand.sum(axis=1).values[sl], label="Demand", c="k")
        ax[0].plot(t, d.pv.sum(axis=1).values[sl], label="PV", c="orange")
        ax[0].plot(t, c.imp[sl], label="Grid import", c="red", alpha=.7)
        ax[0].plot(t, -c.exp[sl], label="Grid export", c="green", alpha=.7)
        ax[0].set_ylabel("kW")
        ax[0].legend(ncol=4, fontsize=8)
        ax[0].set_title(f"Community dispatch - {label} week")
        ax[1].fill_between(t, c.soc[sl.start + 1:sl.stop + 1], color="tab:blue", alpha=.6)
        ax[1].set_ylabel("Community battery SOC [kWh]")
        fig.tight_layout()
        p = out / f"dispatch_{label}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p)

    # 4 deep-learning figures ---------------------------------------------------
    deep = {k: v for k, v in res.forecast_sets.items() if v.history}
    if deep:
        fig, ax = plt.subplots(1, len(deep), figsize=(4.2 * len(deep), 3.8), sharey=True, squeeze=False)
        for a_, (name, fc) in zip(ax[0], deep.items()):
            for k, h in enumerate(fc.history):
                h = np.array(h)
                a_.plot(h[:, 0], color=f"C{k}", label=f"fold {k + 1} train")
                a_.plot(h[:, 1], "--", color=f"C{k}")
            a_.set_title(f"{name}  (solid=train, dashed=val)", fontsize=9)
            a_.set_xlabel("epoch")
        ax[0][0].set_ylabel("MSE (normalised)")
        fig.tight_layout()
        p = out / "dl_training_curves.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p)
    if res.model_ranking is not None:
        r = res.model_ranking.astype({"community_RMSE_kW": float, "community_bill_AUD": float})
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        colors = ["tab:gray" if n == "gbm" else "tab:green" if n == "ensemble" else "tab:blue" for n in r.index]
        ax[0].bar(r.index, r["community_RMSE_kW"], color=colors)
        ax[0].set_ylabel("community demand RMSE [kW]")
        ax[0].set_title("Forecast error by model (lower is better)")
        ax[1].bar(r.index, r["community_bill_AUD"], color=colors)
        ax[1].set_ylim(r["community_bill_AUD"].min() * 0.99, r["community_bill_AUD"].max() * 1.005)
        ax[1].set_ylabel("community annual bill [AUD]")
        ax[1].set_title("Bill when the model drives the battery MPC")
        for a_ in ax:
            a_.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        p = out / "dl_model_comparison.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p)
    if res.rl is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, c in enumerate(res.rl.curves):
            ax[0].plot(np.arange(1, len(c) + 1) * 100, c, label=f"agent {k + 1}")
        ax[0].set_xlabel("training episode (days)")
        ax[0].set_ylabel("mean daily community cost [AUD]")
        ax[0].set_title("DQN learning curve")
        ax[0].legend()
        comp = res.rl.comparison["community_bill_AUD"].drop("No battery")
        ax[1].barh(comp.index, comp.values, color="tab:blue")
        ax[1].set_xlim(comp.min() * 0.9, comp.max() * 1.03)
        ax[1].set_xlabel("community annual bill [AUD]")
        ax[1].set_title("Controllers compared")
        fig.tight_layout()
        p = out / "dl_rl_agent.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p)
    return files
