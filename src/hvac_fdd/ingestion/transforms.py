"""
Pure transformation functions for the LBNL ingestion pipeline.

Expected call order:
  1. wide_zones_to_long()     — ZONE_TEMP_1..5 → (zone_id, ZONE_TEMP) rows
  2. fahrenheit_to_celsius()  — convert all raw temperature columns to °C
  3. normalize_column_names() — rename LBNL column names to internal schema
"""
from __future__ import annotations

import pandas as pd

from hvac_fdd.exceptions import SchemaValidationError

# ── Constants ─────────────────────────────────────────────────────────────────

# Raw LBNL zone-temperature columns consumed by wide_zones_to_long.
_ZONE_COLS: list[str] = [
    "ZONE_TEMP_1", "ZONE_TEMP_2", "ZONE_TEMP_3", "ZONE_TEMP_4", "ZONE_TEMP_5",
]

# Temperature columns that exist between wide_zones_to_long and normalize_column_names.
# "ZONE_TEMP" is the interim name produced by wide_zones_to_long.
# SA_TEMPSPT is a setpoint but is stored in °F in the raw data.
TEMP_COLS_RAW: list[str] = [
    "SA_TEMP", "SA_TEMPSPT", "RA_TEMP", "MA_TEMP", "OA_TEMP", "ZONE_TEMP",
]

# LBNL column name → internal schema name.
# ZONE_TEMP_1..5 are absent here; they are consumed by wide_zones_to_long
# and replaced by the interim "ZONE_TEMP" column, which maps to "temp_zone_c".
COLUMN_MAP: dict[str, str] = {
    # Time
    "Datetime":    "event_time",
    # Temperatures (already °C after fahrenheit_to_celsius)
    "SA_TEMP":     "temp_supply_celsius",
    "SA_TEMPSPT":  "temp_supply_setpoint_c",
    "RA_TEMP":     "temp_return_celsius",
    "MA_TEMP":     "temp_mixed_celsius",
    "OA_TEMP":     "temp_outside_celsius",
    "ZONE_TEMP":   "temp_zone_c",             # created by wide_zones_to_long
    # Chilled-water coil valve
    "CHWC_VLV":    "chwc_valve_pct",
    "CHWC_VLV_DM": "chwc_valve_demand_pct",
    # Outside-air and return-air dampers
    "OA_DMPR":     "oa_damper_pct",
    "OA_DMPR_DM":  "oa_damper_demand_pct",
    "RA_DMPR":     "ra_damper_pct",
    "RA_DMPR_DM":  "ra_damper_demand_pct",
    # Supply fan
    "SF_SPD":      "sf_speed_pct",
    "SF_SPD_DM":   "sf_speed_demand_pct",
    "SF_WAT":      "sf_power_w",
    "SF_CS":       "sf_fan_status",           # 0 = Off, 1 = On
    # Return fan
    "RF_SPD":      "rf_speed_pct",
    "RF_SPD_DM":   "rf_speed_demand_pct",
    "RF_WAT":      "rf_power_w",
    "RF_CS":       "rf_fan_status",           # 0 = Off, 1 = On
    # Airflow
    "SA_CFM":      "supply_air_cfm",
    "OA_CFM":      "outside_air_cfm",
    "RA_CFM":      "return_air_cfm",
    # Static pressure
    "SA_SP":       "supply_air_static_pressure_pa",
    "SA_SPSPT":    "supply_air_static_pressure_setpoint_pa",
    # Control mode (raw integer encoding preserved)
    "SYS_CTL":     "sys_control_mode",
}

# ── Transform functions ───────────────────────────────────────────────────────


def wide_zones_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot ZONE_TEMP_1..5 columns into long form.

    Input:  N rows × 31 LBNL columns
    Output: N×5 rows; ZONE_TEMP_1..5 replaced by two new columns:
              zone_id   — "zone_1" … "zone_5"
              ZONE_TEMP — the temperature value (still in °F at this stage)

    Raises SchemaValidationError if any ZONE_TEMP_1..5 column is absent.
    """
    missing = [c for c in _ZONE_COLS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"wide_zones_to_long: missing zone columns: {missing}")

    id_cols = [c for c in df.columns if c not in _ZONE_COLS]
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=_ZONE_COLS,
        var_name="zone_id",
        value_name="ZONE_TEMP",
    )
    # "ZONE_TEMP_1" → "zone_1", …, "ZONE_TEMP_5" → "zone_5"
    long_df["zone_id"] = long_df["zone_id"].str.replace("ZONE_TEMP_", "zone_", regex=False)
    return long_df.reset_index(drop=True)


def fahrenheit_to_celsius(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert the specified columns from °F to °C on a copy of the DataFrame.
    Formula: C = (F − 32) × 5 / 9

    Raises SchemaValidationError if any column in `columns` is not present.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"fahrenheit_to_celsius: columns not found: {missing}")

    df = df.copy()
    df[columns] = (df[columns] - 32) * 5 / 9
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename LBNL raw column names to the internal schema defined in COLUMN_MAP.

    Must be called after wide_zones_to_long (ZONE_TEMP_1..5 are absent)
    and after fahrenheit_to_celsius (temperature values are already in °C).
    Columns not present in COLUMN_MAP are kept as-is (e.g., zone_id, fault_type).
    """
    return df.rename(columns=COLUMN_MAP)
