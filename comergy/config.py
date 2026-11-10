"""Central configuration: physical constants, battery model, defaults.

Every number that is an *assumption* (not given in the benchmark dataset)
is flagged ASSUMPTION so that it is easy to find and change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "ComErgy_Benchmark_Dataset_2026.xlsx"
DEFAULT_OUTPUT = ROOT / "outputs"
SHEET_NAME = "ComErgy Benchmark Dataset 2026"

# ---- Given by the benchmark sheet --------------------------------------
PV_MODULE_KW = 0.44            # LG NeON 440 W module
BATTERY_MODULE_KWH = 4.66      # Fox ESS EQ4800 module
CONNECTION_FEE_PER_DAY = 1.0   # AUD / household / day
HOURS_PER_YEAR = 8760

# ---- ASSUMPTIONS (not in the dataset) -----------------------------------
GRID_EMISSION_FACTOR = 0.66    # kgCO2e / kWh, approx. NSW grid (ASSUMPTION)
LATITUDE, LONGITUDE, UTC_OFFSET = -33.87, 151.21, 10.0   # Sydney (clear-sky feature)


@dataclass(frozen=True)
class BatteryParams:
    """Simple linear battery model (ASSUMPTIONS - edit to taste)."""
    eta_charge: float = 0.95        # one-way charge efficiency  (round trip ~0.90)
    eta_discharge: float = 0.95     # one-way discharge efficiency
    soc_min_frac: float = 0.10      # reserve kept in the battery
    soc_max_frac: float = 1.00
    c_rate: float = 0.5             # max charge/discharge power = c_rate * capacity (kW per kWh)
    soc_init_frac: float = 0.50     # state of charge on 1 Jan


# Household fall-back table (only used if the Excel metadata block cannot be parsed)
FALLBACK_MEMBERS = [
    # id, pv_modules, batt_modules, income, occupants, description
    ("M1", 15, 0, 200_000, 2, "Couple (FT/FT), no kids"),
    ("M2", 0, 2, 80_000, 2, "Couple, retired"),
    ("M3", 30, 3, 160_000, 5, "Couple + 3 kids (FT / stay-home)"),
    ("M4", 22, 0, 240_000, 4, "Couple + 2 kids (FT/FT)"),
    ("M5", 0, 0, 70_000, 3, "Single parent + 2 kids (PT)"),
    ("M6", 20, 2, 120_000, 5, "Couple + 3 kids (FT/PT)"),
    ("M7", 0, 4, 285_000, 7, "Multi-gen (4 adults, 3 kids)"),
    ("M8", 15, 3, 145_000, 6, "Couple + 4 kids (FT/PT)"),
    ("M9", 10, 0, 190_000, 7, "Share house (7 students)"),
]
