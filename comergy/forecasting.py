"""Day-ahead forecasting with gradient-boosted trees (scikit-learn).

Why gradient boosting instead of a deep network?  The benchmark contains a
single year (8,760 hours) of nine smart-meter series.  On a data set of this
size tree ensembles are as accurate as LSTMs/Transformers, train in seconds,
need no GPU and are far more explainable - important for community trust.

Forecast targets
    * hourly demand of every household (and hence of the community)
    * hourly GHI (-> PV generation through the physical PV model)

Only information that is available at day-ahead gate closure is used:
calendar features, clear-sky irradiance, the temperature forecast (the
measured temperature stands in for a weather-service forecast) and
observations that are >= 24 h old.  Predictions are *out-of-fold*: the year
is cut into contiguous blocks and every block is predicted by a model that
did not see it, so reported errors are honest.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data_loader import CommunityData, pv_generation


# --------------------------------------------------------------------------
def clear_sky_ghi(index: pd.DatetimeIndex) -> np.ndarray:
    """Haurwitz clear-sky GHI [W/m2] for Sydney - used only as a model feature."""
    doy = index.dayofyear.values
    hour = index.hour.values + 0.5
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    solar_time = hour - C.UTC_OFFSET + C.LONGITUDE / 15.0
    ha = np.deg2rad(15.0 * (solar_time - 12.0))
    lat = np.deg2rad(C.LATITUDE)
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(cosz > 0.01, 1098.0 * cosz * np.exp(-0.057 / cosz), 0.0)
    return g


def _calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=index)
    f["hour"] = index.hour
    f["dow"] = index.dayofweek
    f["weekend"] = (index.dayofweek >= 5).astype(int)
    f["month"] = index.month
    f["h_sin"] = np.sin(2 * np.pi * index.hour / 24)
    f["h_cos"] = np.cos(2 * np.pi * index.hour / 24)
    f["d_sin"] = np.sin(2 * np.pi * index.dayofyear / 365)
    f["d_cos"] = np.cos(2 * np.pi * index.dayofyear / 365)
    return f


def _lags(s: pd.Series, name: str) -> pd.DataFrame:
    f = pd.DataFrame(index=s.index)
    for L in (24, 48, 168):
        f[f"{name}_lag{L}"] = s.shift(L)
    f[f"{name}_mean_prev_day"] = s.shift(24).rolling(24, min_periods=6).mean()
    return f


def demand_features(data: CommunityData, series: pd.Series, name: str) -> pd.DataFrame:
    f = pd.concat([_calendar(data.index), _lags(series, name)], axis=1)
    f["temp"] = data.temp
    f["tier"] = data.price                       # tariff tier proxies the household routine
    f["clear_sky"] = clear_sky_ghi(data.index)
    return f


def ghi_features(data: CommunityData) -> pd.DataFrame:
    g = pd.Series(data.ghi, index=data.index)
    f = pd.concat([_calendar(data.index), _lags(g, "ghi")], axis=1)
    f["temp"] = data.temp
    f["temp_lag24"] = pd.Series(data.temp, index=data.index).shift(24)
    f["clear_sky"] = clear_sky_ghi(data.index)
    return f


# --------------------------------------------------------------------------
def _make_model(kind: str, seed: int):
    if kind == "gbm":                      # gradient-boosted decision trees
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=20,
                                             min_samples_leaf=25, l2_regularization=1.0,
                                             random_state=seed)
    if kind == "mlp":                      # deep feed-forward neural network (2 hidden layers, ReLU, Adam)
        # strong L2 weight decay (alpha=1) is essential: one year of data overfits easily
        net = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                           alpha=1.0, learning_rate_init=2e-3, batch_size=128, max_iter=150,
                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=8,
                           random_state=seed)
        return TransformedTargetRegressor(
            regressor=make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), net),
            transformer=StandardScaler())
    raise ValueError(kind)


def _oof_predict(X: pd.DataFrame, y: np.ndarray, n_splits: int, seed: int, base=None,
                 kind: str = "gbm") -> np.ndarray:
    """Out-of-fold prediction.  If `base` is given the model learns the residual y - base
    (a persistence-style level) which lets it cope with regime shifts / unseen load levels."""
    target = y if base is None else y - base
    if kind == "mlp":                      # raw integer calendar codes mislead a network; sin/cos remain
        X = X.drop(columns=[c for c in ("hour", "dow", "month") if c in X.columns])
    pred = np.zeros(len(y))
    for tr, te in KFold(n_splits=n_splits, shuffle=False).split(X):
        model = _make_model(kind, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # sklearn ConvergenceWarning is harmless here
            model.fit(X.iloc[tr], target[tr])
        pred[te] = model.predict(X.iloc[te])
    if base is not None:
        pred = pred + base
    return np.clip(pred, 0, None)


def _metrics(name: str, y: np.ndarray, p: np.ndarray, naive: np.ndarray) -> dict:
    mask = ~np.isnan(naive)
    mae, rmse = np.mean(np.abs(y - p)), np.sqrt(np.mean((y - p) ** 2))
    rmse_n = np.sqrt(np.mean((y[mask] - naive[mask]) ** 2))
    rmse_m = np.sqrt(np.mean((y[mask] - p[mask]) ** 2))
    ss_res, ss_tot = np.sum((y - p) ** 2), np.sum((y - y.mean()) ** 2)
    return dict(target=name, MAE=mae, RMSE=rmse, R2=1 - ss_res / ss_tot,
                naive_RMSE=rmse_n, RMSE_skill_vs_naive_pct=100 * (1 - rmse_m / rmse_n))


@dataclass
class Forecasts:
    model: str
    demand_hat: pd.DataFrame      # per household, kW
    ghi_hat: np.ndarray
    pv_hat: pd.DataFrame          # per household, kW (physical model on forecast GHI)
    metrics: pd.DataFrame
    history: list | None = None   # training curves per fold [(train_loss, val_loss), ...] (deep sequence models)
    info: dict | None = None      # params / seconds / epochs

    def net_hat(self) -> pd.DataFrame:
        return self.demand_hat - self.pv_hat


def _score(data: CommunityData, model: str, dhat: pd.DataFrame, ghat: np.ndarray) -> Forecasts:
    rows = []
    for m in data.ids:
        naive = data.demand[m].shift(24).values
        rows.append(_metrics(f"demand {m}", data.demand[m].values, dhat[m].values, naive))
    tot = data.demand.sum(axis=1)
    rows.append(_metrics("demand COMMUNITY", tot.values, dhat.sum(axis=1).values, tot.shift(24).values))
    rows.append(_metrics("GHI", data.ghi, ghat, pd.Series(data.ghi).shift(24).values))
    pvhat = pd.DataFrame({m: pv_generation(ghat, data.temp, data.members.loc[m, "pv_kw"], data.pv_mode)
                          for m in data.ids}, index=data.index)
    return Forecasts(model, dhat, ghat, pvhat, pd.DataFrame(rows).set_index("target"))


MODEL_FAMILY = {"gbm": "decision trees", "mlp": "feed-forward NN", "lstm": "recurrent (LSTM)",
                "gru": "recurrent (GRU)", "tcn": "convolutional (TCN)", "transformer": "attention (Transformer)",
                "ensemble": "mean of all models"}
SEQ_MODELS = ("lstm", "gru", "tcn", "transformer")
ALL_MODELS = ("gbm", "mlp") + SEQ_MODELS


def build_forecasts(data: CommunityData, kind: str = "gbm", n_splits: int = 6, seed: int = 0,
                    dl_epochs: int = 20, progress=None) -> Forecasts:
    """Out-of-fold day-ahead forecasts of every household's demand and of GHI with one model type.
    kind: gbm | mlp | lstm | gru | tcn | transformer (the last four are written from scratch in NumPy)."""
    import time
    if kind in SEQ_MODELS:
        from .dl_forecaster import seq_oof_forecasts
        dh, gh, hist, info = seq_oof_forecasts(data, kind, epochs=dl_epochs, seed=seed, progress=progress)
        f = _score(data, kind, dh, gh)
        f.history, f.info = hist, info
        return f
    t0 = time.time()
    dhat = {}
    for m in data.ids:
        y = data.demand[m].values
        X = demand_features(data, data.demand[m], m)
        base = X[f"{m}_mean_prev_day"].fillna(float(np.mean(y))).values
        dhat[m] = _oof_predict(X, y, n_splits, seed, base=base, kind=kind)
    ghat = _oof_predict(ghi_features(data), data.ghi, n_splits, seed, kind=kind)
    f = _score(data, kind, pd.DataFrame(dhat, index=data.index), ghat)
    f.info = dict(params=float("nan"), seconds=time.time() - t0, epochs=[])
    return f


def build_all_forecasts(data: CommunityData, kinds=ALL_MODELS, progress=None, dl_epochs: int = 20) -> dict:
    """Train every requested model type and add an equal-weight ensemble.  Returns {name: Forecasts}."""
    out = {}
    for i, k in enumerate(kinds):
        sub = (lambda f, m, i=i: progress((i + f) / len(kinds), m)) if progress else None
        if progress:
            progress(i / len(kinds), k)
        out[k] = build_forecasts(data, k, dl_epochs=dl_epochs, progress=sub)
    if len(kinds) > 1:
        dh = sum(out[k].demand_hat for k in kinds) / len(kinds)
        gh = sum(out[k].ghi_hat for k in kinds) / len(kinds)
        out["ensemble"] = _score(data, "ensemble", dh, gh)
        out["ensemble"].info = dict(params=float("nan"), seconds=sum(out[k].info["seconds"] for k in kinds), epochs=[])
    return out
