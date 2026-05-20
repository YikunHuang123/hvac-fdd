"""
LBNL-specific feature engineering.

Expects a DataFrame with normalized column names (output of normalize_column_names).
All features are derived from real LBNL signal names and directly map to ASHRAE
Guideline 36 fault mechanisms used by LBNLRulesDetector in Phase 3.
"""
from __future__ import annotations

import pandas as pd

from hvac_fdd.config import Settings
from hvac_fdd.exceptions import FeatureEngineeringError

# ── Constants ─────────────────────────────────────────────────────────────────

# Columns for which 15-min and 60-min rolling mean + std are computed.
# Each maps directly to a fault detection rule in detection/rules.py.
_ROLLING_COLS: list[str] = [
    "sa_temp_error_c",
    "valve_tracking_err",
    "damper_tracking_err",
]

# AHU-level power signal (replicated across zones after wide_zones_to_long).
# Rolling stats used by Rule 5 (FAN_POWER_ANOMALY).
_POWER_COL = "sf_power_w"

# Lag-1 features (1 step = 1 minute at the LBNL 1-min sampling rate).
_LAG_COLS: list[str] = [
    "temp_supply_celsius",
    "chwc_valve_pct",
    "oa_damper_pct",
]

# All engineered numeric features used for detection.
# Centralizing this here decouples detection/base.py from hardcoded strings.
ENG_FEATURE_COLUMNS: list[str] = [
    "valve_tracking_err",
    "damper_tracking_err",
    "sf_tracking_err",
    "sa_temp_error_c",
    "ma_oa_delta_c",
    "ra_sa_delta_c",
    "valve_tracking_err_15_mean",
    "valve_tracking_err_60_mean",
    "damper_tracking_err_15_mean",
    "damper_tracking_err_60_mean",
    "sa_temp_error_c_15_mean",
    "sa_temp_error_c_60_mean",
    "sf_power_w_60_mean",
    "temp_supply_celsius_lag1",
    "chwc_valve_pct_lag1",
    "oa_damper_pct_lag1",
]

# Minimum columns required before feature computation begins.
_REQUIRED_INPUT_COLS: list[str] = [
    "event_time", "zone_id",
    "chwc_valve_pct",       "chwc_valve_demand_pct",
    "oa_damper_pct",        "oa_damper_demand_pct",
    "sf_speed_pct",         "sf_speed_demand_pct",
    "temp_supply_celsius",  "temp_supply_setpoint_c",
    "temp_mixed_celsius",   "temp_outside_celsius",
    "temp_return_celsius",  _POWER_COL,
]


# ── Public function ───────────────────────────────────────────────────────────


def build_feature_set(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    Compute all LBNL-specific features for fault detection.

    Returns a new DataFrame (input is not mutated) with all original columns
    plus the following derived feature columns:

    A. Control tracking errors  — directly tied to LBNL fault mechanisms
         valve_tracking_err     = chwc_valve_pct - chwc_valve_demand_pct
         damper_tracking_err    = oa_damper_pct  - oa_damper_demand_pct
         sf_tracking_err        = sf_speed_pct   - sf_speed_demand_pct

    B. Temperature error signals
         sa_temp_error_c        = temp_supply_celsius - temp_supply_setpoint_c
         ma_oa_delta_c          = temp_mixed_celsius  - temp_outside_celsius
         ra_sa_delta_c          = temp_return_celsius - temp_supply_celsius

    C. Rolling statistics (15-min and 60-min windows, per zone_id)
         {col}_{15|60}_mean / std  for each col in _ROLLING_COLS
         sf_power_w_60_mean / std

    D. Lag features (1-step lag = 1 minute)
         {col}_lag1  for each col in _LAG_COLS

    Raises FeatureEngineeringError if any required input column is missing.
    """
    missing = [c for c in _REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise FeatureEngineeringError(f"Missing columns for feature engineering: {missing}")

    df = df.copy()

    # Sort so rolling windows run in chronological order within each zone.
    df = df.sort_values(["zone_id", "event_time"]).reset_index(drop=True)

    # ── A. Control tracking errors ────────────────────────────────────────────
    df["valve_tracking_err"]  = df["chwc_valve_pct"]  - df["chwc_valve_demand_pct"]
    df["damper_tracking_err"] = df["oa_damper_pct"]   - df["oa_damper_demand_pct"]
    df["sf_tracking_err"]     = df["sf_speed_pct"]    - df["sf_speed_demand_pct"]

    # ── B. Temperature error signals ──────────────────────────────────────────
    df["sa_temp_error_c"] = df["temp_supply_celsius"] - df["temp_supply_setpoint_c"]
    df["ma_oa_delta_c"]   = df["temp_mixed_celsius"]  - df["temp_outside_celsius"]
    df["ra_sa_delta_c"]   = df["temp_return_celsius"] - df["temp_supply_celsius"]

    # ── C. Rolling statistics (grouped by zone to avoid cross-zone leakage) ──
    # At 1-min sampling: window=15 → 15-min, window=60 → 60-min.
    # min_periods=1 prevents NaN at the start of each zone's history.
    # fillna(0.0) on std avoids NaN when a window has only one point.
    for col in _ROLLING_COLS:
        for window, suffix in ((15, "15"), (60, "60")):
            grp = df.groupby("zone_id", sort=False)[col]
            df[f"{col}_{suffix}_mean"] = grp.transform(
                lambda x, w=window: x.rolling(w, min_periods=1).mean()
            )
            df[f"{col}_{suffix}_std"] = grp.transform(
                lambda x, w=window: x.rolling(w, min_periods=1).std().fillna(0.0)
            )

    # sf_power_w is AHU-level (identical across zones for same timestamp),
    # but grouping by zone keeps the computation consistent with _ROLLING_COLS.
    pwr_grp = df.groupby("zone_id", sort=False)[_POWER_COL]
    df["sf_power_w_60_mean"] = pwr_grp.transform(
        lambda x: x.rolling(60, min_periods=1).mean()
    )
    df["sf_power_w_60_std"] = pwr_grp.transform(
        lambda x: x.rolling(60, min_periods=1).std().fillna(0.0)
    )

    # ── D. Lag features (1-step = 1 minute) ──────────────────────────────────
    for col in _LAG_COLS:
        df[f"{col}_lag1"] = df.groupby("zone_id", sort=False)[col].transform(
            lambda x: x.shift(1)
        )

    return df
