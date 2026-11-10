"""Interactive dashboard:   streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from comergy import allocation as A
from comergy import metrics as M
from comergy.pipeline import PipelineConfig, run_pipeline
from comergy.reporting import export_excel

st.set_page_config(page_title="Community Energy Forcasting", page_icon="⚡", layout="wide")
REC = "Equity-adjusted Shapley (recommended)"

st.title("⚡Fair & intelligent community energy market")
st.caption("9 households · rooftop PV · 14 battery modules · Sydney ToU tariff · MPC + DL forecasting + Shapley/equity allocation")

# ---------------- sidebar -------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    pv_mode = st.selectbox("PV model", ["report", "realistic"], help="'report' = GHI/1000 x kWp as in the benchmark PDF")
    use_ml = st.checkbox("Use ML forecasts in MPC", True, help="Off = perfect foresight (upper bound)")
    models = st.multiselect("Forecasters to compare", ["gbm", "mlp", "lstm", "gru", "tcn", "transformer"],
                            default=["gbm", "mlp", "lstm", "gru", "tcn", "transformer"],
                            help="lstm / gru / tcn / transformer are deep networks written from scratch in NumPy (~1 min each, cached after the first run)")
    forecaster = st.selectbox("Forecaster driving the MPC", ["auto", "ensemble"] + models,
                              help="auto = whichever model gives the lowest community bill")
    use_rl = st.checkbox("Train deep-RL (DQN) battery agent (+~1.5 min)", False)
    rep_days = st.slider("Representative days (Shapley game)", 6, 60, 24, help="more = more exact but slower")
    peak_w = st.slider("Peak-import penalty [AUD/kW]", 0.0, 2.0, 0.0, 0.1, help="grid-support objective")
    st.divider()
    st.subheader("Fairness (live)")
    lam = st.slider("Equity weight λ  (0 = merit, 1 = need)", 0.0, 1.0, 0.5, 0.05)
    alpha = st.slider("Guaranteed share α of own Shapley saving", 0.0, 1.0, 0.6, 0.05)
    run = st.button("▶ Run / refresh model", type="primary")


@st.cache_resource(show_spinner=False)
def _shapley_store():
    return {}


@st.cache_resource(show_spinner=False)
def _run(pv_mode, use_ml, forecaster, models, use_rl, rep_days, peak_w):
    store = _shapley_store()
    key = (pv_mode, rep_days)
    bar = st.progress(0.0, text="Starting …")
    cfg = PipelineConfig(pv_mode=pv_mode, use_ml=use_ml, forecaster=forecaster, models=tuple(models), use_rl=use_rl,
                         rep_days=rep_days, peak_weight=peak_w)
    res = run_pipeline(cfg, lambda f, m: bar.progress(min(f, 1.0), text=m), store.get(key))
    store[key] = res.rep_info["coalition_costs"]
    bar.empty()
    return res


if "ready" not in st.session_state:
    st.session_state.ready = False
if run:
    st.session_state.ready = True
if not st.session_state.ready:
    st.info("Press **▶ Run / refresh model** in the sidebar (first run takes ~1.5 min: ML training + 511 coalition LPs).")
    st.stop()

res = _run(pv_mode, use_ml, forecaster, tuple(models), use_rl, rep_days, peak_w)
d = res.data
ref = res.standalone.cost
total_saving = float(ref.sum() - (res.community.cost + d.fee_year * d.n))
live_bills = A.equity_bills(ref, total_saving, res.phi, d.members["income"], lam, alpha)
tables = dict(res.tables)
tables[REC] = M.fairness_table(live_bills, ref, res.baseline, d.members["income"],
                               total_saving * res.phi / res.phi.sum())
summary = pd.DataFrame({k: M.fairness_summary(t) for k, t in tables.items()}).T

t1, t2, t3, t4, t5, t7, t6 = st.tabs(["Overview", "Households", "Fairness", "Dispatch explorer", "Forecasting (DL)", "Deep RL agent", "Method & checks"])

# ---------------- overview -------------------------------------------------------
with t1:
    k = res.kpis
    base, sa, com = k.loc["Baseline (grid only)"], k.loc["Standalone - optimised"], k.loc["Community (pooled MPC)"]
    c = st.columns(5)
    c[0].metric("Community bill", f"AUD {com.total_cost_AUD:,.0f}", f"{100 * (com.total_cost_AUD / base.total_cost_AUD - 1):.0f}% vs baseline")
    c[1].metric("Saving vs standalone", f"AUD {total_saving:,.0f}", f"{100 * total_saving / sa.total_cost_AUD:.0f}%")
    c[2].metric("Self-sufficiency", f"{com.self_sufficiency_pct:.1f}%", f"{com.self_sufficiency_pct - sa.self_sufficiency_pct:+.1f} pp vs standalone")
    c[3].metric("Peak-tier imports", f"{com.peak_tier_import_kWh:,.0f} kWh", f"{100 * (com.peak_tier_import_kWh / base.peak_tier_import_kWh - 1):.0f}% vs baseline")
    c[4].metric("CO₂", f"{com.co2_t:.1f} t", f"{com.co2_t - base.co2_t:+.1f} t vs baseline")
    st.dataframe(k.round(1), width="stretch")
    fig = px.bar(k.reset_index().rename(columns={"index": "scenario"}), x="scenario", y="total_cost_AUD", title="Community annual cost by scenario [AUD]")
    st.plotly_chart(fig, width="stretch")

# ---------------- households -----------------------------------------------------
with t2:
    mech = st.selectbox("Mechanism", list(tables), index=list(tables).index(REC))
    t = tables[mech]
    st.dataframe(t.round(1), width="stretch")
    fig = go.Figure()
    fig.add_bar(x=t.index, y=t["baseline_AUD"], name="Baseline")
    fig.add_bar(x=t.index, y=t["standalone_AUD"], name="Standalone")
    fig.add_bar(x=t.index, y=t["bill_AUD"], name=mech)
    fig.update_layout(barmode="group", yaxis_title="Annual bill [AUD]")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(d.members[["pv_kw", "batt_kwh", "income", "occupants", "description"]], width="stretch")

# ---------------- fairness -------------------------------------------------------
with t3:
    st.subheader("Mechanism comparison")
    st.dataframe(summary.round(2), width="stretch")
    long = pd.concat({k: v["saving_vs_standalone_AUD"] for k, v in tables.items()}, axis=1).reset_index().melt(
        id_vars="index", var_name="mechanism", value_name="saving")
    fig = px.bar(long, x="index", y="saving", color="mechanism", barmode="group",
                 labels={"index": "household", "saving": "saving vs standalone [AUD/yr]"})
    fig.add_hline(y=0, line_color="black")
    st.plotly_chart(fig, width="stretch")
    burden = pd.concat({k: v["energy_burden_pct"] for k, v in tables.items()}, axis=1).reset_index().melt(
        id_vars="index", var_name="mechanism", value_name="burden")
    st.plotly_chart(px.bar(burden, x="index", y="burden", color="mechanism", barmode="group",
                           title="Energy burden = bill / income [%]"), width="stretch")
    st.caption("Shapley = merit (marginal contribution to community value, never negative → no-worse-off). "
               "Equity-adjusted blends in need via a small QP; α guarantees everyone ≥ α × their Shapley saving.")

# ---------------- dispatch -------------------------------------------------------
with t4:
    day = st.slider("Start day of year", 1, 358, 15)
    sl = slice((day - 1) * 24, (day - 1) * 24 + 7 * 24)
    idx = d.index[sl]
    c = res.community
    fig = go.Figure()
    fig.add_scatter(x=idx, y=d.demand.sum(axis=1).values[sl], name="Demand", line_color="black")
    fig.add_scatter(x=idx, y=d.pv.sum(axis=1).values[sl], name="PV", line_color="orange")
    fig.add_scatter(x=idx, y=c.imp[sl], name="Grid import", line_color="red")
    fig.add_scatter(x=idx, y=-c.exp[sl], name="Grid export (−)", line_color="green")
    fig.add_scatter(x=idx, y=c.dis[sl] - c.ch[sl], name="Battery (+dis / −ch)", line_color="blue")
    fig.update_layout(yaxis_title="kW", height=420)
    st.plotly_chart(fig, width="stretch")
    soc = go.Figure(go.Scatter(x=idx, y=c.soc[sl.start + 1:sl.stop + 1], fill="tozeroy", name="SOC"))
    soc.add_scatter(x=idx, y=d.price[sl] * 100, name="Retail price [c/kWh]", yaxis="y2", line_color="gray")
    soc.update_layout(yaxis_title="Community battery SOC [kWh]", yaxis2=dict(overlaying="y", side="right"), height=300)
    st.plotly_chart(soc, width="stretch")

# ---------------- forecasting ----------------------------------------------------
with t5:
    if not res.forecast_sets:
        st.info("ML forecasts are switched off (perfect foresight).")
    else:
        st.subheader("Deep learning forecasters vs gradient-boosted trees")
        st.caption(f"Forecaster driving the battery controller: **{res.forecasts.model}**")
        rmse = pd.DataFrame({k: v.metrics["RMSE"] for k, v in res.forecast_sets.items()})
        skill = pd.DataFrame({k: v.metrics["RMSE_skill_vs_naive_pct"] for k, v in res.forecast_sets.items()})
        c1, c2 = st.columns(2)
        c1.write("RMSE (lower is better)")
        c1.dataframe(rmse.round(3), width="stretch")
        c2.write("Skill vs yesterday-persistence [%] (higher is better)")
        c2.dataframe(skill.round(1), width="stretch")
        st.write("**Value of forecasting** - community annual bill when each model drives the battery MPC [AUD]")
        st.dataframe(res.forecast_value.round(0), width="stretch")
        rk = res.model_ranking.copy()
        st.write("**Model ranking** (out-of-fold; bill = community bill when the model drives the battery controller)")
        st.dataframe(rk.round(3), width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(rk.reset_index().rename(columns={"index": "model"}), x="model", y="community_RMSE_kW",
                               color="family", title="Community demand RMSE [kW]"), width="stretch")
        rb = rk.reset_index().rename(columns={"index": "model"})
        c2.plotly_chart(px.bar(rb, x="model", y="community_bill_AUD", color="family", title="Community annual bill [AUD]")
                        .update_yaxes(range=[float(rb.community_bill_AUD.min()) * 0.99, float(rb.community_bill_AUD.max()) * 1.005]),
                        width="stretch")
        deep = {k: v for k, v in res.forecast_sets.items() if v.history}
        if deep:
            pick = st.selectbox("Training curves of", list(deep))
            fig_l = go.Figure()
            for k, h in enumerate(deep[pick].history):
                h = np.array(h)
                fig_l.add_scatter(y=h[:, 0], name=f"fold {k + 1} train", line_color=f"hsl({k * 80},60%,45%)")
                fig_l.add_scatter(y=h[:, 1], name=f"fold {k + 1} val", line_dash="dash", line_color=f"hsl({k * 80},60%,45%)")
            fig_l.update_layout(xaxis_title="epoch", yaxis_title="MSE (normalised)", height=320,
                                title=f"{pick}: written from scratch in NumPy, early stopping on validation loss")
            st.plotly_chart(fig_l, width="stretch")
        st.divider()
        which = st.selectbox("Model to plot", list(res.forecast_sets), index=list(res.forecast_sets).index(res.forecasts.model))
        f = res.forecast_sets[which]
        who = st.selectbox("Series", ["COMMUNITY"] + d.ids)
        a_ = (d.demand.sum(axis=1) if who == "COMMUNITY" else d.demand[who])
        b_ = (f.demand_hat.sum(axis=1) if who == "COMMUNITY" else f.demand_hat[who])
        s0 = st.slider("Start day", 1, 358, 100, key="fd")
        sl = slice((s0 - 1) * 24, (s0 - 1) * 24 + 7 * 24)
        fig = go.Figure()
        fig.add_scatter(x=d.index[sl], y=a_.values[sl], name="Actual")
        fig.add_scatter(x=d.index[sl], y=b_.values[sl], name=f"{which} forecast")
        st.plotly_chart(fig, width="stretch")
        st.caption("Out-of-fold forecasts (no day is predicted by a model that saw it).")

# ---------------- deep RL -------------------------------------------------------------
with t7:
    if res.rl is None:
        st.info("Tick **Train deep-RL (DQN) battery agent** in the sidebar and press Run to train it.")
    else:
        st.markdown("A **Double DQN** (written from scratch in NumPy) learns when to charge / discharge the shared 65 kWh battery. "
                    "Each agent is evaluated only on weeks it never trained on.")
        comp = res.rl.comparison
        st.dataframe(comp.round(0), width="stretch")
        st.plotly_chart(px.bar(comp.drop("No battery").reset_index().rename(columns={"index": "controller"}),
                               x="community_bill_AUD", y="controller", orientation="h",
                               title="Community annual bill by controller [AUD]"), width="stretch")
        fig = go.Figure()
        for k, c in enumerate(res.rl.curves):
            fig.add_scatter(x=np.arange(1, len(c) + 1) * 100, y=c, name=f"agent {k + 1}")
        fig.update_layout(xaxis_title="training episode (days)", yaxis_title="mean daily cost [AUD]", title="DQN learning curve")
        st.plotly_chart(fig, width="stretch")

# ---------------- method ----------------------------------------------------------
with t6:
    st.markdown("""
**Pipeline** – (1) data + PV model → (2) gradient-boosting day-ahead forecasts → (3) pooled-battery **LP/MPC** (HiGHS) with a real-time reactive layer
→ (4) **KMeans** representative days → 511 coalition LPs → **exact Shapley** values → (5) price-based LEM settlement, Shapley and equity-QP allocation, optional **no-worse-off** top-up.
""")
    st.subheader("Reproduction check vs the benchmark PDF (AUD/yr)")
    st.dataframe(res.report_check, width="stretch")
    st.json({k: (float(v) if not isinstance(v, (int,)) else v) for k, v in res.shapley_diag.items()})
    st.write("Timings [s]:", {k: round(v, 1) for k, v in res.timings.items()})
    if st.button("Export Excel"):
        p = export_excel(res, "outputs/comergy_results.xlsx")
        st.success(f"Saved {p}")
