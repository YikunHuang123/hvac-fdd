"""
Phase 2 ingestion tests.

`pytest -v`

Coverage required by REFACTORING_PLAN.md:
  - Column name mapping completeness (normalize_column_names)
  - wide_zones_to_long row-count x5
  - SchemaValidationError trigger conditions
Plus: fahrenheit_to_celsius math, feature column creation, and loader helpers.

All tests use small in-memory DataFrames — no actual 140 MB CSVs are read.
The test_settings fixture (conftest.py) supplies Settings with safe defaults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hvac_fdd.domain import FaultType
from hvac_fdd.exceptions import DataLoadError, FeatureEngineeringError, SchemaValidationError
from hvac_fdd.ingestion.features import build_feature_set
from hvac_fdd.ingestion.lbnl_loader import LBNLDataLoader
from hvac_fdd.ingestion.transforms import (
    COLUMN_MAP,
    TEMP_COLS_RAW,
    fahrenheit_to_celsius,
    normalize_column_names,
    wide_zones_to_long,
)

# ── Fixture helpers ───────────────────────────────────────────────────────────


def _make_raw_frame(n_rows: int = 10) -> pd.DataFrame:
    deterministic__ = """
    Build a minimal valid LBNL-schema DataFrame (all temps in Fahrenheit).
    Uses a fixed RNG seed so tests are deterministic.
    """
    rng = np.random.default_rng(42)
    data: dict = {col: rng.uniform(0.0, 100.0, n_rows) for col in LBNLDataLoader.REQUIRED_COLUMNS}
    # Realistic zone-temp values in F so F->C sanity checks are meaningful.
    for zone_col in ["ZONE_TEMP_1", "ZONE_TEMP_2", "ZONE_TEMP_3", "ZONE_TEMP_4", "ZONE_TEMP_5"]:
        data[zone_col] = rng.uniform(68.0, 78.0, n_rows)
    data["Datetime"] = pd.date_range("2018-01-01", periods=n_rows, freq="min")
    return pd.DataFrame(data)


def _make_long_frame(n_rows: int = 10) -> pd.DataFrame:
    """Return a wide_zones_to_long-processed DataFrame."""
    return wide_zones_to_long(_make_raw_frame(n_rows))


def _make_normalized_frame(n_rows: int = 20) -> pd.DataFrame:
    """Return a fully transformed DataFrame: long + F->C + renamed."""
    df = _make_raw_frame(n_rows)
    df = wide_zones_to_long(df)
    df = fahrenheit_to_celsius(df, columns=TEMP_COLS_RAW)
    return normalize_column_names(df)


# ── transforms.fahrenheit_to_celsius ─────────────────────────────────────────


class TestFahrenheitToCelsius:
    def test_freezing_point(self):
        df = pd.DataFrame({"temp": [32.0]})
        result = fahrenheit_to_celsius(df, ["temp"])
        assert abs(result["temp"].iloc[0]) < 1e-9

    def test_boiling_point(self):
        df = pd.DataFrame({"temp": [212.0]})
        result = fahrenheit_to_celsius(df, ["temp"])
        assert abs(result["temp"].iloc[0] - 100.0) < 1e-9

    def test_body_temperature(self):
        df = pd.DataFrame({"temp": [98.6]})
        result = fahrenheit_to_celsius(df, ["temp"])
        assert abs(result["temp"].iloc[0] - 37.0) < 1e-6

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"temp": [32.0]})
        fahrenheit_to_celsius(df, ["temp"])
        assert df["temp"].iloc[0] == 32.0

    def test_multiple_columns_converted(self):
        df = pd.DataFrame({"a": [32.0, 212.0], "b": [32.0, 212.0]})
        result = fahrenheit_to_celsius(df, ["a", "b"])
        assert abs(result["a"].iloc[0]) < 1e-9
        assert abs(result["b"].iloc[1] - 100.0) < 1e-9

    def test_missing_column_raises_schema_error(self):
        df = pd.DataFrame({"temp": [32.0]})
        with pytest.raises(SchemaValidationError, match="columns not found"):
            fahrenheit_to_celsius(df, ["nonexistent"])

    def test_all_temp_cols_raw_present_after_wide_zones(self):
        """TEMP_COLS_RAW must all exist in the DataFrame produced by wide_zones_to_long."""
        long_df = _make_long_frame(4)
        for col in TEMP_COLS_RAW:
            assert col in long_df.columns, (
                f"TEMP_COLS_RAW entry {col!r} missing after wide_zones_to_long"
            )


# ── transforms.wide_zones_to_long ────────────────────────────────────────────


class TestWideZonesToLong:
    def test_row_count_multiplied_by_five(self):
        n = 10
        long_df = wide_zones_to_long(_make_raw_frame(n))
        assert len(long_df) == n * 5

    def test_zone_id_values(self):
        long_df = _make_long_frame(4)
        assert set(long_df["zone_id"].unique()) == {
            "zone_1", "zone_2", "zone_3", "zone_4", "zone_5",
        }

    def test_interim_zone_temp_column_created(self):
        long_df = _make_long_frame(4)
        assert "ZONE_TEMP" in long_df.columns

    def test_original_zone_columns_removed(self):
        long_df = _make_long_frame(4)
        for col in ["ZONE_TEMP_1", "ZONE_TEMP_2", "ZONE_TEMP_3", "ZONE_TEMP_4", "ZONE_TEMP_5"]:
            assert col not in long_df.columns

    def test_non_zone_columns_preserved(self):
        long_df = _make_long_frame(4)
        for col in ["Datetime", "SA_TEMP", "CHWC_VLV", "SYS_CTL"]:
            assert col in long_df.columns

    def test_missing_zone_columns_raises(self):
        df = pd.DataFrame({"Datetime": pd.date_range("2018-01-01", periods=3, freq="min")})
        with pytest.raises(SchemaValidationError, match="missing zone columns"):
            wide_zones_to_long(df)

    def test_zone_temp_values_match_source(self):
        """Values from ZONE_TEMP_3 must appear exactly in the zone_3 rows."""
        raw = _make_raw_frame(5)
        expected = set(raw["ZONE_TEMP_3"].round(6).tolist())
        long_df = wide_zones_to_long(raw)
        actual = set(long_df.loc[long_df["zone_id"] == "zone_3", "ZONE_TEMP"].round(6).tolist())
        assert expected == actual


# ── transforms.normalize_column_names ────────────────────────────────────────


class TestNormalizeColumnNames:
    def test_all_mapped_columns_renamed(self):
        """Every COLUMN_MAP entry present in the input must appear renamed in the output."""
        long_df = _make_long_frame(4)
        renamed = normalize_column_names(long_df)
        for old, new in COLUMN_MAP.items():
            if old in long_df.columns:
                assert new in renamed.columns, f"Expected {old!r} -> {new!r} but {new!r} missing"
                assert old not in renamed.columns, f"Old column {old!r} still present after rename"

    def test_zone_id_column_unchanged(self):
        renamed = normalize_column_names(_make_long_frame(4))
        assert "zone_id" in renamed.columns

    def test_fault_type_column_preserved(self):
        long_df = _make_long_frame(4)
        long_df["fault_type"] = FaultType.NORMAL.value
        renamed = normalize_column_names(long_df)
        assert "fault_type" in renamed.columns

    def test_key_internal_columns_present_after_full_transform(self):
        """End-to-end: all expected internal schema columns exist after the full transform."""
        df = _make_normalized_frame(6)
        expected = [
            "event_time", "zone_id",
            "temp_supply_celsius", "temp_supply_setpoint_c",
            "temp_return_celsius", "temp_mixed_celsius",
            "temp_outside_celsius", "temp_zone_c",
            "chwc_valve_pct", "chwc_valve_demand_pct",
            "oa_damper_pct",  "oa_damper_demand_pct",
            "sf_speed_pct",   "sf_speed_demand_pct",
            "sf_power_w",     "sf_fan_status",
            "rf_speed_pct",   "rf_power_w",
            "supply_air_cfm", "outside_air_cfm", "return_air_cfm",
            "supply_air_static_pressure_pa",
            "sys_control_mode",
        ]
        for col in expected:
            assert col in df.columns, f"Internal schema column missing: {col!r}"

    def test_temp_zone_is_celsius_after_full_transform(self):
        """Zone temps originally 68-78 F must convert to approx 20-26 C."""
        df = _make_normalized_frame(20)
        assert df["temp_zone_c"].min() > 18.0
        assert df["temp_zone_c"].max() < 28.0


# ── LBNLDataLoader._infer_fault_type ─────────────────────────────────────────


class TestInferFaultType:
    @pytest.fixture
    def loader(self, tmp_path) -> LBNLDataLoader:
        return LBNLDataLoader(tmp_path)

    @pytest.mark.parametrize("stem,expected", [
        ("AHU_annual",                    FaultType.NORMAL),
        ("coi_bias_-2_annual",            FaultType.COI_BIAS),
        ("coi_bias_4_annual",             FaultType.COI_BIAS),
        ("coi_leakage_010_annual",        FaultType.COI_LEAKAGE),
        ("coi_leakage_050_annual",        FaultType.COI_LEAKAGE),
        ("coi_stuck_025_annual",          FaultType.COI_STUCK),
        ("coi_stuck_075_annual",          FaultType.COI_STUCK),
        ("damper_stuck_010_annual",       FaultType.DAMPER_STUCK),
        ("damper_stuck_100_annual_short", FaultType.DAMPER_STUCK),
        ("oa_bias_-4_annual",             FaultType.OA_BIAS),
        ("oa_bias_2_annual",              FaultType.OA_BIAS),
    ])
    def test_known_filenames(self, loader, stem: str, expected: FaultType):
        assert loader._infer_fault_type(stem) == expected

    def test_unknown_stem_raises(self, loader):
        with pytest.raises(DataLoadError, match="Cannot infer fault type"):
            loader._infer_fault_type("unknown_sensor_data")


# ── LBNLDataLoader.validate_schema ───────────────────────────────────────────


class TestValidateSchema:
    @pytest.fixture
    def loader(self, tmp_path) -> LBNLDataLoader:
        return LBNLDataLoader(tmp_path)

    def test_valid_schema_passes(self, loader):
        loader.validate_schema(_make_raw_frame(5))  # must not raise

    def test_one_missing_column_raises(self, loader):
        df = _make_raw_frame(5).drop(columns=["SA_TEMP"])
        with pytest.raises(SchemaValidationError, match="SA_TEMP"):
            loader.validate_schema(df)

    def test_multiple_missing_columns_raises(self, loader):
        df = _make_raw_frame(5).drop(columns=["SA_TEMP", "OA_TEMP", "ZONE_TEMP_1"])
        with pytest.raises(SchemaValidationError):
            loader.validate_schema(df)

    def test_extra_columns_do_not_cause_error(self, loader):
        df = _make_raw_frame(5)
        df["extra_col"] = 0
        loader.validate_schema(df)  # must not raise

    def test_empty_dataframe_with_correct_columns_passes(self, loader):
        df = pd.DataFrame(columns=LBNLDataLoader.REQUIRED_COLUMNS)
        loader.validate_schema(df)  # must not raise


# ── LBNLDataLoader.load_all ───────────────────────────────────────────────────


class TestLoadAll:
    def test_no_csv_raises(self, tmp_path):
        with pytest.raises(DataLoadError, match="No CSV files found"):
            LBNLDataLoader(tmp_path).load_all()

    def test_loads_and_attaches_fault_type(self, tmp_path):
        """Write a minimal valid CSV; verify fault_type column is attached correctly."""
        raw = _make_raw_frame(5)
        raw["Datetime"] = raw["Datetime"].astype(str)
        (tmp_path / "AHU_annual.csv").write_text(raw.to_csv(index=False))

        df = LBNLDataLoader(tmp_path).load_all(chunksize=None)

        assert "fault_type" in df.columns
        assert set(df["fault_type"].unique()) == {FaultType.NORMAL.value}
        assert len(df) == 5

    def test_unknown_filename_raises(self, tmp_path):
        """A CSV whose name doesn't match any fault type must raise DataLoadError."""
        raw = _make_raw_frame(3)
        raw["Datetime"] = raw["Datetime"].astype(str)
        (tmp_path / "unknown_sensor_log.csv").write_text(raw.to_csv(index=False))

        with pytest.raises(DataLoadError, match="Cannot infer fault type"):
            LBNLDataLoader(tmp_path).load_all(chunksize=None)


# ── features.build_feature_set ───────────────────────────────────────────────


class TestBuildFeatureSet:
    @pytest.fixture
    def feature_df(self, test_settings) -> pd.DataFrame:
        df = _make_normalized_frame(n_rows=20)
        df["fault_type"] = FaultType.NORMAL.value
        return build_feature_set(df, test_settings)

    # ── Column existence ──────────────────────────────────────────────────────

    def test_tracking_error_columns_created(self, feature_df):
        assert "valve_tracking_err"  in feature_df.columns
        assert "damper_tracking_err" in feature_df.columns
        assert "sf_tracking_err"     in feature_df.columns

    def test_temp_error_columns_created(self, feature_df):
        assert "sa_temp_error_c" in feature_df.columns
        assert "ma_oa_delta_c"   in feature_df.columns
        assert "ra_sa_delta_c"   in feature_df.columns

    def test_rolling_mean_and_std_columns_created(self, feature_df):
        for col in ["sa_temp_error_c", "valve_tracking_err", "damper_tracking_err"]:
            for suffix in ("15", "60"):
                assert f"{col}_{suffix}_mean" in feature_df.columns
                assert f"{col}_{suffix}_std"  in feature_df.columns

    def test_power_rolling_columns_created(self, feature_df):
        assert "sf_power_w_60_mean" in feature_df.columns
        assert "sf_power_w_60_std"  in feature_df.columns

    def test_lag_columns_created(self, feature_df):
        for col in ["temp_supply_celsius", "chwc_valve_pct", "oa_damper_pct"]:
            assert f"{col}_lag1" in feature_df.columns

    # ── Arithmetic correctness ────────────────────────────────────────────────

    def test_valve_tracking_error_math(self, test_settings):
        df = _make_normalized_frame(10)
        df["chwc_valve_pct"]        = 60.0
        df["chwc_valve_demand_pct"] = 50.0
        result = build_feature_set(df, test_settings)
        assert (result["valve_tracking_err"] == 10.0).all()

    def test_sa_temp_error_math(self, test_settings):
        df = _make_normalized_frame(10)
        df["temp_supply_celsius"]    = 15.0
        df["temp_supply_setpoint_c"] = 13.0
        result = build_feature_set(df, test_settings)
        assert (result["sa_temp_error_c"] - 2.0).abs().max() < 1e-9

    def test_rolling_std_no_nan(self, feature_df):
        """std columns must not contain NaN (fillna(0.0) handles single-point windows)."""
        std_cols = [c for c in feature_df.columns if c.endswith("_std")]
        for col in std_cols:
            assert feature_df[col].isna().sum() == 0, f"NaN found in {col!r}"

    # ── Safety ────────────────────────────────────────────────────────────────

    def test_does_not_mutate_input(self, test_settings):
        df = _make_normalized_frame(10)
        original_cols = set(df.columns)
        build_feature_set(df, test_settings)
        assert set(df.columns) == original_cols

    def test_missing_required_column_raises(self, test_settings):
        df = _make_normalized_frame(10).drop(columns=["chwc_valve_pct"])
        with pytest.raises(FeatureEngineeringError, match="Missing columns"):
            build_feature_set(df, test_settings)

    def test_pyarrow_backend_compatibility(self, test_settings):
        """
        Verify that even if input columns use pyarrow dtypes, the rolling
        columns are cast to float64 (numpy) to avoid memory leaks.
        """
        try:
            import pyarrow as pa
        except ImportError:
            pytest.skip("pyarrow not installed")

        df = _make_normalized_frame(10)
        # Convert some source columns to pyarrow dtypes
        df["chwc_valve_pct"] = pd.Series(df["chwc_valve_pct"], dtype="float64[pyarrow]")
        df["temp_supply_celsius"] = pd.Series(df["temp_supply_celsius"], dtype="float64[pyarrow]")
        
        result = build_feature_set(df, test_settings)
        
        # Verify target columns are cast to float64 (numpy backend)
        # "valve_tracking_err" is derived from "chwc_valve_pct"
        assert result["valve_tracking_err"].dtype == "float64"
        assert not isinstance(result["valve_tracking_err"].dtype, pd.ArrowDtype)
        
        # Verify rolling columns are present and numeric
        assert "valve_tracking_err_15_mean" in result.columns
        assert result["valve_tracking_err_15_mean"].dtype == "float64"
