"""Load the ComErgy 2026 Excel benchmark and build physical PV profiles."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# PV model
# --------------------------------------------------------------------------
def pv_generation(ghi, temp, pv_kw: float, mode: str = "report",
                  system_pr: float = 0.88, gamma: float = -0.0034, noct: float = 45.0) -> np.ndarray:
    """Hourly PV output [kW].

    mode="report"    : P = GHI/1000 * kWp  (identical to the benchmark analysis PDF,
                       e.g. 6.6 kWp -> 12,487 kWh/yr).
    mode="realistic" : adds cell-temperature derating and a system performance ratio
                       (inverter, soiling, wiring) - more conservative, more realistic.
    """
    ghi = np.asarray(ghi, dtype=float)
    temp = np.asarray(temp, dtype=float)
    if mode == "report":
        return ghi / 1000.0 * pv_kw
    if mode == "realistic":
        t_cell = temp + (noct - 20.0) / 800.0 * ghi
        return ghi / 1000.0 * pv_kw * system_pr * (1.0 + gamma * (t_cell - 25.0))
    raise ValueError(f"unknown pv mode: {mode}")


# --------------------------------------------------------------------------
@dataclass
class CommunityData:
    index: pd.DatetimeIndex
    demand: pd.DataFrame          # kW (= kWh per hour), columns M1..Mn
    pv: pd.DataFrame              # kW
    price: np.ndarray             # AUD/kWh retail (time-of-use)
    fit: np.ndarray               # AUD/kWh feed-in tariff
    temp: np.ndarray
    ghi: np.ndarray
    members: pd.DataFrame         # static household table
    fee_per_day: float = C.CONNECTION_FEE_PER_DAY
    pv_mode: str = "report"

    @property
    def ids(self) -> list:
        return list(self.demand.columns)

    @property
    def n(self) -> int:
        return self.demand.shape[1]

    @property
    def T(self) -> int:
        return len(self.index)

    @property
    def days(self) -> int:
        return self.T // 24

    @property
    def fee_year(self) -> float:
        return self.fee_per_day * self.days

    def net_load(self) -> pd.DataFrame:
        """demand - PV (kW, positive = needs energy)."""
        return self.demand - self.pv


# --------------------------------------------------------------------------
def _parse_members(raw: pd.DataFrame, n_members: int) -> pd.DataFrame:
    """Parse the household table located to the right of the hourly data."""
    try:
        block = raw.iloc[:, 14:20]
        hdr = block[block.iloc[:, 1].astype(str).str.contains("DGS", na=False)].index[0]
        rows = []
        for k in range(n_members):
            r = raw.iloc[hdr + 1 + k, 14:20].tolist()
            der = str(r[2]).lower()
            pv = re.search(r"(\d+)\s*pv", der)
            bt = re.search(r"(\d+)\s*batter", der)
            occ = re.search(r"(\d+)", str(r[4]))
            inc = re.sub(r"[^\d]", "", str(r[5]))
            rows.append(dict(id=f"M{k + 1}",
                             pv_modules=int(pv.group(1)) if pv else 0,
                             batt_modules=int(bt.group(1)) if bt else 0,
                             income=float(inc), occupants=int(occ.group(1)),
                             description=str(r[3]).strip()))
        df = pd.DataFrame(rows).set_index("id")
    except Exception:  # fall back to the table printed in the analysis PDF
        df = pd.DataFrame(C.FALLBACK_MEMBERS,
                          columns=["id", "pv_modules", "batt_modules", "income", "occupants",
                                   "description"]).set_index("id")
    df["pv_kw"] = df["pv_modules"] * C.PV_MODULE_KW
    df["batt_kwh"] = df["batt_modules"] * C.BATTERY_MODULE_KWH
    return df


def load_dataset(path=None, pv_mode: str = "report") -> CommunityData:
    path = Path(path) if path else C.DEFAULT_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPut 'ComErgy_Benchmark_Dataset_2026.xlsx' in the 'data' "
            f"folder or pass --data <path>.")
    raw = pd.read_excel(path, sheet_name=C.SHEET_NAME, header=None)
    body = raw.iloc[2:2 + C.HOURS_PER_YEAR, :13].astype(float).reset_index(drop=True)
    n = 9
    idx = pd.date_range("2026-01-01", periods=len(body), freq="h")
    ids = [f"M{i + 1}" for i in range(n)]
    demand = pd.DataFrame(body.iloc[:, :n].values, index=idx, columns=ids)
    price, fit = body.iloc[:, 9].values, body.iloc[:, 10].values
    temp, ghi = body.iloc[:, 11].values, body.iloc[:, 12].values
    members = _parse_members(raw, n)
    try:  # "Daily connection fee to the grid" cell
        row = raw[raw.iloc[:, 15].astype(str).str.contains("connection fee", case=False, na=False)]
        fee = float(raw.loc[row.index[0], 16])
    except Exception:
        fee = C.CONNECTION_FEE_PER_DAY
    pv = pd.DataFrame({m: pv_generation(ghi, temp, members.loc[m, "pv_kw"], pv_mode) for m in ids},
                      index=idx)
    return CommunityData(idx, demand, pv, price, fit, temp, ghi, members, fee, pv_mode)
