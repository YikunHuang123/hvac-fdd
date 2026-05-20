"""
Phase 3 detection tests.

Coverage per REFACTORING_PLAN.md Phase 3:
  - Each rule's boundary conditions (rules.py)
  - IsolationForestDetector._require_fitted() protection
  - FaultClassifier F1 > 0.80 on synthetic separable data

All tests use small in-memory DataFrames — no actual CSVs are read.
The test_settings fixture (conftest.py) provides Settings with safe defaults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from hvac_fdd.detection.base import NUMERIC_FEATURES, PREDICT_OUTPUT_COLS
from hvac_fdd.detection.classifier import CLASSIFIER_OUTPUT_COLS, FaultClassifier
from hvac_fdd.detection.ensemble import EnsembleDetector
from hvac_fdd.detection.isolation_forest import IsolationForestDetector
from hvac_fdd.detection.rules import LBNLRulesDetector
from hvac_fdd.domain import AlertLevel, DetectionEvent, FaultType
from hvac_fdd.exceptions import DetectorNotFittedError

# ── Fixture helpers ───────────────────────────────────────────────────────────


def _make_feature_df(
    n_rows: int = 80,
    fault_type: str = FaultType.NORMAL.value,
    *,
    zone_id: str = "zone_1",
) -> pd.DataFrame:
    """
    Build a feature-engineered DataFrame with all columns expected by detectors.

    All numeric signals default to steady-state values that do not trigger
    any detection rule.  Callers override specific columns to inject faults.
    """
    times = pd.date_range("2018-01-01", periods=n_rows, freq="min")
    return pd.DataFrame(
        {
            # Identifiers
            "event_time":                           times,
            "zone_id":                              zone_id,
            "fault_type":                           fault_type,
            # Raw signals (needed by rules)
            "chwc_valve_pct":                       50.0,
            "chwc_valve_demand_pct":                50.0,
            "oa_damper_pct":                        50.0,
            "oa_damper_demand_pct":                 50.0,
            "sf_speed_pct":                         50.0,
            "sf_speed_demand_pct":                  50.0,
            "temp_supply_celsius":                  13.0,
            "temp_supply_setpoint_c":               13.0,
            "temp_mixed_celsius":                   15.0,
            "temp_outside_celsius":                 10.0,
            "temp_return_celsius":                  20.0,
            "sf_power_w":                           1_000.0,
            # Control tracking errors
            "valve_tracking_err":                   0.0,
            "damper_tracking_err":                  0.0,
            "sf_tracking_err":                      0.0,
            # Temp error signals
            "sa_temp_error_c":                      0.0,
            "ma_oa_delta_c":                        5.0,
            "ra_sa_delta_c":                        7.0,
            # 15-min rolling stats
            "valve_tracking_err_15_mean":           0.0,
            "valve_tracking_err_15_std":            0.0,
            "damper_tracking_err_15_mean":          0.0,
            "damper_tracking_err_15_std":           0.0,
            "sa_temp_error_c_15_mean":              0.0,
            "sa_temp_error_c_15_std":               0.0,
            # 60-min rolling stats
            "valve_tracking_err_60_mean":           0.0,
            "valve_tracking_err_60_std":            0.0,
            "damper_tracking_err_60_mean":          0.0,
            "damper_tracking_err_60_std":           0.0,
            "sa_temp_error_c_60_mean":              0.0,
            "sa_temp_error_c_60_std":               0.3,   # non-zero so rule 3 std check is meaningful
            "sf_power_w_60_mean":                   1_000.0,
            "sf_power_w_60_std":                    100.0,
            # Lag features
            "temp_supply_celsius_lag1":             13.0,
            "chwc_valve_pct_lag1":                  50.0,
            "oa_damper_pct_lag1":                   50.0,
        }
    )


def _make_multi_class_df(n_per_class: int = 100) -> pd.DataFrame:
    """
    Build a clearly separable six-class DataFrame for FaultClassifier F1 tests.

    Each fault type is assigned a distinct combination of feature values so
    RandomForest should reach near-perfect F1 on the held-out slice.
    """
    rng = np.random.default_rng(0)

    def _noise(n: int) -> np.ndarray:
        return rng.normal(0, 0.05, n)

    def _class_df(ft: str, n: int, **overrides) -> pd.DataFrame:
        df = _make_feature_df(n_rows=n, fault_type=ft)
        for col, val in overrides.items():
            df[col] = val + _noise(n)
        return df

    frames = [
        _class_df(FaultType.NORMAL.value,       n_per_class),
        _class_df(FaultType.COI_STUCK.value,    n_per_class,
                  valve_tracking_err=30.0,
                  valve_tracking_err_15_mean=30.0,
                  valve_tracking_err_60_mean=30.0),
        _class_df(FaultType.DAMPER_STUCK.value, n_per_class,
                  damper_tracking_err=30.0,
                  damper_tracking_err_15_mean=30.0,
                  damper_tracking_err_60_mean=30.0),
        _class_df(FaultType.COI_BIAS.value,     n_per_class,
                  sa_temp_error_c=5.0,
                  sa_temp_error_c_15_mean=5.0,
                  sa_temp_error_c_60_mean=5.0),
        _class_df(FaultType.COI_LEAKAGE.value,  n_per_class,
                  sa_temp_error_c=-5.0,
                  sa_temp_error_c_15_mean=-5.0,
                  sa_temp_error_c_60_mean=-5.0,
                  chwc_valve_pct=2.0),
        _class_df(FaultType.OA_BIAS.value,      n_per_class,
                  ma_oa_delta_c=15.0,
                  temp_outside_celsius=-10.0),
    ]
    # Non-overlapping timestamps so a time-based split is unambiguous.
    total = n_per_class * 6
    df = pd.concat(frames, ignore_index=True)
    df["event_time"] = pd.date_range("2018-01-01", periods=total, freq="min")
    return df


# ── LBNLRulesDetector: general setup ─────────────────────────────────────────


class TestLBNLRulesDetectorSetup:
    def test_is_fitted_after_construction(self, test_settings):
        det = LBNLRulesDetector(test_settings)
        assert det._is_fitted is True

    def test_predict_returns_dataframe(self, test_settings):
        det = LBNLRulesDetector(test_settings)
        result = det.predict(_make_feature_df())
        assert isinstance(result, pd.DataFrame)

    def test_predict_output_columns_present(self, test_settings):
        det = LBNLRulesDetector(test_settings)
        result = det.predict(_make_feature_df())
        for col in PREDICT_OUTPUT_COLS:
            assert col in result.columns

    def test_no_fault_yields_no_events(self, test_settings):
        det = LBNLRulesDetector(test_settings)
        assert len(det.predict(_make_feature_df())) == 0

    def test_fit_predict_matches_predict(self, test_settings):
        det = LBNLRulesDetector(test_settings)
        df = _make_feature_df()
        assert det.fit_predict(df).equals(det.predict(df))


# ── Rule 1: VALVE_STUCK ───────────────────────────────────────────────────────


class TestRuleValveStuck:
    def test_fires_above_threshold(self, test_settings):
        df = _make_feature_df()
        df["valve_tracking_err_15_mean"] = 15.0  # threshold=10
        result = LBNLRulesDetector(test_settings).predict(df)
        assert (result["violated_policy"] == "VALVE_STUCK").any()

    def test_fires_for_negative_error(self, test_settings):
        df = _make_feature_df()
        df["valve_tracking_err_15_mean"] = -15.0  # abs(−15) > 10
        result = LBNLRulesDetector(test_settings).predict(df)
        assert (result["violated_policy"] == "VALVE_STUCK").any()

    def test_does_not_fire_at_exact_threshold(self, test_settings):
        df = _make_feature_df()
        df["valve_tracking_err_15_mean"] = 10.0  # equal → no fire
        result = LBNLRulesDetector(test_settings).predict(df)
        assert not (result["violated_policy"] == "VALVE_STUCK").any()

    def test_alert_level_is_critical(self, test_settings):
        df = _make_feature_df()
        df["valve_tracking_err_15_mean"] = 20.0
        result = LBNLRulesDetector(test_settings).predict(df)
        rows = result[result["violated_policy"] == "VALVE_STUCK"]
        assert (rows["alert_level"] == AlertLevel.CRITICAL.value).all()

    def test_anomaly_index_is_ratio(self, test_settings):
        df = _make_feature_df()
        df["valve_tracking_err_15_mean"] = 20.0
        result = LBNLRulesDetector(test_settings).predict(df)
        rows = result[result["violated_policy"] == "VALVE_STUCK"]
        assert (rows["anomaly_index"] - 2.0).abs().max() < 1e-9

    def test_trigger_signal(self, test_settings):
        df = _make_feature_df()
        df["valve_tracking_err_15_mean"] = 20.0
        result = LBNLRulesDetector(test_settings).predict(df)
        rows = result[result["violated_policy"] == "VALVE_STUCK"]
        assert (rows["trigger_signal"] == "chwc_valve_pct").all()


# ── Rule 2: DAMPER_STUCK ──────────────────────────────────────────────────────


class TestRuleDamperStuck:
    def test_fires_above_threshold(self, test_settings):
        df = _make_feature_df()
        df["damper_tracking_err_15_mean"] = 15.0
        result = LBNLRulesDetector(test_settings).predict(df)
        assert (result["violated_policy"] == "DAMPER_STUCK").any()

    def test_does_not_fire_below_threshold(self, test_settings):
        df = _make_feature_df()
        df["damper_tracking_err_15_mean"] = 5.0
        result = LBNLRulesDetector(test_settings).predict(df)
        assert not (result["violated_policy"] == "DAMPER_STUCK").any()

    def test_alert_level_is_critical(self, test_settings):
        df = _make_feature_df()
        df["damper_tracking_err_15_mean"] = 20.0
        result = LBNLRulesDetector(test_settings).predict(df)
        rows = result[result["violated_policy"] == "DAMPER_STUCK"]
        assert (rows["alert_level"] == AlertLevel.CRITICAL.value).all()


# ── Rule 3: SUPPLY_AIR_TEMP_BIAS ─────────────────────────────────────────────


class TestRuleSupplyAirTempBias:
    def test_fires_with_bias_and_low_std(self, test_settings):
        df = _make_feature_df()
        df["sa_temp_error_c_60_mean"] = 2.0   # > 1.5
        df["sa_temp_error_c_60_std"]  = 0.3   # < 0.5 → systematic bias
        result = LBNLRulesDetector(test_settings).predict(df)
        assert (result["violated_policy"] == "SUPPLY_AIR_TEMP_BIAS").any()

    def test_does_not_fire_when_std_too_high(self, test_settings):
        df = _make_feature_df()
        df["sa_temp_error_c_60_mean"] = 2.0
        df["sa_temp_error_c_60_std"]  = 0.6   # oscillation → no bias
        result = LBNLRulesDetector(test_settings).predict(df)
        assert not (result["violated_policy"] == "SUPPLY_AIR_TEMP_BIAS").any()

    def test_does_not_fire_when_mean_below_threshold(self, test_settings):
        df = _make_feature_df()
        df["sa_temp_error_c_60_mean"] = 1.0   # < 1.5
        df["sa_temp_error_c_60_std"]  = 0.3
        result = LBNLRulesDetector(test_settings).predict(df)
        assert not (result["violated_policy"] == "SUPPLY_AIR_TEMP_BIAS").any()

    def test_fires_for_negative_bias(self, test_settings):
        df = _make_feature_df()
        df["sa_temp_error_c_60_mean"] = -2.0
        df["sa_temp_error_c_60_std"]  = 0.3
        result = LBNLRulesDetector(test_settings).predict(df)
        assert (result["violated_policy"] == "SUPPLY_AIR_TEMP_BIAS").any()

    def test_alert_level_is_warning(self, test_settings):
        df = _make_feature_df()
        df["sa_temp_error_c_60_mean"] = 2.0
        df["sa_temp_error_c_60_std"]  = 0.3
        result = LBNLRulesDetector(test_settings).predict(df)
        rows = result[result["violated_policy"] == "SUPPLY_AIR_TEMP_BIAS"]
        assert (rows["alert_level"] == AlertLevel.WARNING.value).all()


# ── Rule 4: COIL_LEAKAGE ─────────────────────────────────────────────────────


class TestRuleCoilLeakage:
    def test_fires_when_valve_closed_and_supply_cold(self, test_settings):
        df = _make_feature_df()
        df["chwc_valve_pct"] = 2.0    # < 5
        df["sa_temp_error_c"] = -3.0  # < −1.5
        result = LBNLRulesDetector(test_settings).predict(df)
        assert (result["violated_policy"] == "COIL_LEAKAGE").any()

    def test_does_not_fire_when_valve_open(self, test_settings):
        df = _make_feature_df()
        df["chwc_valve_pct"] = 30.0   # not closed
        df["sa_temp_error_c"] = -3.0
        result = LBNLRulesDetector(test_settings).predict(df)
        assert not (result["violated_policy"] == "COIL_LEAKAGE").any()

    def test_does_not_fire_when_temp_error_positive(self, test_settings):
        df = _make_feature_df()
        df["chwc_valve_pct"] = 2.0
        df["sa_temp_error_c"] = 3.0   # supply warm → no leakage
        result = LBNLRulesDetector(test_settings).predict(df)
        assert not (result["violated_policy"] == "COIL_LEAKAGE").any()

    def test_alert_level_is_warning(self, test_settings):
        df = _make_feature_df()
        df["chwc_valve_pct"] = 2.0
        df["sa_temp_error_c"] = -3.0
        result = LBNLRulesDetector(test_settings).predict(df)
        rows = result[result["violated_policy"] == "COIL_LEAKAGE"]
        assert (rows["alert_level"] == AlertLevel.WARNING.value).all()


# ── Rule 5: FAN_POWER_ANOMALY ────────────────────────────────────────────────


class TestRuleFanPowerAnomaly:
    def test_fires_above_three_sigma(self, test_settings):
        df = _make_feature_df()
        df["sf_power_w"]         = 2_000.0  # (2000−1000)/100 = 10σ
        df["sf_power_w_60_mean"] = 1_000.0
        df["sf_power_w_60_std"]  = 100.0
        result = LBNLRulesDetector(test_settings).predict(df)
        assert (result["violated_policy"] == "FAN_POWER_ANOMALY").any()

    def test_does_not_fire_within_three_sigma(self, test_settings):
        df = _make_feature_df()
        df["sf_power_w"]         = 1_200.0  # only 2σ
        df["sf_power_w_60_mean"] = 1_000.0
        df["sf_power_w_60_std"]  = 100.0
        result = LBNLRulesDetector(test_settings).predict(df)
        assert not (result["violated_policy"] == "FAN_POWER_ANOMALY").any()

    def test_does_not_fire_when_std_is_zero(self, test_settings):
        # std=0 means rolling window not yet full; suppress to avoid false positives
        df = _make_feature_df()
        df["sf_power_w"]         = 9_999.0
        df["sf_power_w_60_mean"] = 1_000.0
        df["sf_power_w_60_std"]  = 0.0
        result = LBNLRulesDetector(test_settings).predict(df)
        assert not (result["violated_policy"] == "FAN_POWER_ANOMALY").any()

    def test_alert_level_is_info(self, test_settings):
        df = _make_feature_df()
        df["sf_power_w"]         = 2_000.0
        df["sf_power_w_60_mean"] = 1_000.0
        df["sf_power_w_60_std"]  = 100.0
        result = LBNLRulesDetector(test_settings).predict(df)
        rows = result[result["violated_policy"] == "FAN_POWER_ANOMALY"]
        assert (rows["alert_level"] == AlertLevel.INFO.value).all()


# ── IsolationForestDetector ───────────────────────────────────────────────────


class TestIsolationForestDetector:
    @pytest.fixture
    def fitted_if(self, test_settings) -> IsolationForestDetector:
        normal_df = _make_feature_df(n_rows=200, fault_type=FaultType.NORMAL.value)
        det = IsolationForestDetector(test_settings)
        det.fit(normal_df)
        return det

    def test_require_fitted_raises_before_fit(self, test_settings):
        det = IsolationForestDetector(test_settings)
        with pytest.raises(DetectorNotFittedError):
            det.predict(_make_feature_df())

    def test_fit_marks_as_fitted(self, test_settings):
        det = IsolationForestDetector(test_settings)
        det.fit(_make_feature_df(n_rows=100))
        assert det._is_fitted is True

    def test_predict_returns_dataframe(self, fitted_if):
        result = fitted_if.predict(_make_feature_df())
        assert isinstance(result, pd.DataFrame)

    def test_predict_output_columns_present(self, fitted_if):
        result = fitted_if.predict(_make_feature_df(n_rows=50))
        for col in PREDICT_OUTPUT_COLS:
            assert col in result.columns

    def test_anomaly_index_non_negative(self, fitted_if):
        result = fitted_if.predict(_make_feature_df(n_rows=100))
        if len(result) > 0:
            assert (result["anomaly_index"] >= 0).all()

    def test_alert_level_is_valid(self, fitted_if):
        valid = {AlertLevel.INFO.value, AlertLevel.WARNING.value, AlertLevel.CRITICAL.value}
        result = fitted_if.predict(_make_feature_df(n_rows=100))
        if len(result) > 0:
            assert set(result["alert_level"].unique()).issubset(valid)

    def test_save_and_load_roundtrip(self, fitted_if, test_settings, tmp_path):
        model_path = tmp_path / "if_model.joblib"
        fitted_if.save(model_path)
        loaded = IsolationForestDetector.load(model_path, settings=test_settings)
        assert loaded._is_fitted is True
        df = _make_feature_df(n_rows=50)
        original = fitted_if.predict(df)["anomaly_index"].tolist()
        restored = loaded.predict(df)["anomaly_index"].tolist()
        assert original == restored

    def test_save_unfitted_raises(self, test_settings, tmp_path):
        det = IsolationForestDetector(test_settings)
        with pytest.raises(DetectorNotFittedError):
            det.save(tmp_path / "model.joblib")

    def test_nan_rows_excluded(self, fitted_if):
        df = _make_feature_df(n_rows=50)
        df.loc[0, "valve_tracking_err"] = float("nan")
        result = fitted_if.predict(df)  # must not raise
        assert isinstance(result, pd.DataFrame)


# ── FaultClassifier ───────────────────────────────────────────────────────────


class TestFaultClassifier:
    @pytest.fixture
    def multi_class_df(self) -> pd.DataFrame:
        return _make_multi_class_df(n_per_class=100)

    @pytest.fixture
    def fitted_clf(self, test_settings, multi_class_df) -> FaultClassifier:
        train = multi_class_df.iloc[: int(len(multi_class_df) * 0.75)]
        clf = FaultClassifier(test_settings)
        clf.fit(train)
        return clf

    def test_require_fitted_raises_before_fit(self, test_settings):
        clf = FaultClassifier(test_settings)
        with pytest.raises(DetectorNotFittedError):
            clf.predict(_make_feature_df())

    def test_fit_marks_as_fitted(self, test_settings):
        clf = FaultClassifier(test_settings)
        clf.fit(_make_feature_df(n_rows=60))
        assert clf._is_fitted is True

    def test_predict_returns_dataframe(self, fitted_clf, multi_class_df):
        result = fitted_clf.predict(multi_class_df)
        assert isinstance(result, pd.DataFrame)

    def test_predict_output_columns_present(self, fitted_clf, multi_class_df):
        result = fitted_clf.predict(multi_class_df)
        for col in CLASSIFIER_OUTPUT_COLS:
            assert col in result.columns

    def test_predict_excludes_normal_rows(self, fitted_clf, multi_class_df):
        result = fitted_clf.predict(multi_class_df)
        if len(result) > 0:
            assert (result["predicted_fault"] != FaultType.NORMAL.value).all()

    def test_confidence_in_unit_interval(self, fitted_clf, multi_class_df):
        result = fitted_clf.predict(multi_class_df)
        if len(result) > 0:
            assert result["confidence"].between(0.0, 1.0).all()

    def test_f1_exceeds_0_80(self, test_settings, multi_class_df):
        """
        RandomForest on clearly separable synthetic data must achieve F1 > 0.80.
        Shuffle with a fixed seed before splitting so all six classes appear in
        both the training and test sets (the sequential concat order would put
        only the last two classes in the test slice).
        """
        shuffled = multi_class_df.sample(frac=1, random_state=42).reset_index(drop=True)
        split    = int(len(shuffled) * 0.75)
        train_df = shuffled.iloc[:split]
        test_df  = shuffled.iloc[split:]

        clf = FaultClassifier(test_settings)
        clf.fit(train_df)
        result = clf.predict(test_df)

        # Rows not returned by predict() are implicitly classified as NORMAL.
        merged = test_df[["event_time", "zone_id", "fault_type"]].merge(
            result[["event_time", "zone_id", "predicted_fault"]],
            on=["event_time", "zone_id"],
            how="left",
        )
        merged["predicted_fault"] = merged["predicted_fault"].fillna(FaultType.NORMAL.value)

        score = f1_score(merged["fault_type"], merged["predicted_fault"], average="macro")
        assert score > 0.80, f"F1 macro = {score:.4f}, expected > 0.80"

    def test_save_and_load_roundtrip(self, fitted_clf, test_settings, multi_class_df, tmp_path):
        model_path = tmp_path / "clf_model.joblib"
        fitted_clf.save(model_path)
        loaded  = FaultClassifier.load(model_path, settings=test_settings)
        assert loaded._is_fitted is True
        original = fitted_clf.predict(multi_class_df)["predicted_fault"].tolist()
        restored = loaded.predict(multi_class_df)["predicted_fault"].tolist()
        assert original == restored

    def test_save_unfitted_raises(self, test_settings, tmp_path):
        clf = FaultClassifier(test_settings)
        with pytest.raises(DetectorNotFittedError):
            clf.save(tmp_path / "clf.joblib")


# ── EnsembleDetector ──────────────────────────────────────────────────────────


class TestEnsembleDetector:
    @pytest.fixture
    def fitted_ensemble(self, test_settings) -> EnsembleDetector:
        normal_df = _make_feature_df(n_rows=200)
        iso   = IsolationForestDetector(test_settings)
        iso.fit(normal_df)
        rules = LBNLRulesDetector(test_settings)
        return EnsembleDetector(rules, iso, equipment_id="AHU-TEST")

    def test_detect_returns_list(self, fitted_ensemble):
        assert isinstance(fitted_ensemble.detect(_make_feature_df()), list)

    def test_events_are_detection_event_instances(self, fitted_ensemble):
        df = _make_feature_df()
        df["valve_tracking_err_15_mean"] = 20.0
        result = fitted_ensemble.detect(df)
        assert all(isinstance(e, DetectionEvent) for e in result)

    def test_equipment_id_propagated(self, fitted_ensemble):
        df = _make_feature_df()
        df["valve_tracking_err_15_mean"] = 20.0
        result = fitted_ensemble.detect(df)
        assert all(e.equipment_id == "AHU-TEST" for e in result)

    def test_dedup_one_event_per_time_zone(self, fitted_ensemble):
        df = _make_feature_df(n_rows=200)
        df["valve_tracking_err_15_mean"] = 20.0  # rules and possibly IF both fire
        result = fitted_ensemble.detect(df)
        keys = [(e.event_time, e.zone_id) for e in result]
        assert len(keys) == len(set(keys)), "Duplicate (event_time, zone_id) after dedup"

    def test_detect_result_is_list_for_normal_input(self, test_settings):
        normal_df = _make_feature_df(n_rows=200)
        iso   = IsolationForestDetector(test_settings)
        iso.fit(normal_df)
        rules = LBNLRulesDetector(test_settings)
        ensemble = EnsembleDetector(rules, iso)
        result = ensemble.detect(_make_feature_df(n_rows=50))
        assert isinstance(result, list)

    def test_with_classifier_annotates_predicted_fault(self, test_settings):
        multi_df = _make_multi_class_df(n_per_class=80)
        normal_rows = multi_df[multi_df["fault_type"] == FaultType.NORMAL.value]

        iso   = IsolationForestDetector(test_settings)
        iso.fit(normal_rows)
        rules = LBNLRulesDetector(test_settings)
        clf   = FaultClassifier(test_settings)
        clf.fit(multi_df)

        ensemble = EnsembleDetector(rules, iso, clf, equipment_id="AHU-1")

        fault_df = _make_feature_df(n_rows=50, fault_type=FaultType.COI_STUCK.value)
        fault_df["valve_tracking_err_15_mean"] = 25.0
        fault_df["valve_tracking_err"]         = 25.0
        fault_df["valve_tracking_err_60_mean"] = 25.0

        result = ensemble.detect(fault_df)
        assert len(result) > 0
        predicted = [e.predicted_fault for e in result if e.predicted_fault is not None]
        assert len(predicted) > 0

    def test_ground_truth_propagated(self, fitted_ensemble):
        df = _make_feature_df()
        df["fault_type"]                 = FaultType.COI_STUCK.value
        df["ground_truth"]               = FaultType.COI_STUCK.value
        df["valve_tracking_err_15_mean"] = 20.0
        result = fitted_ensemble.detect(df)
        assert any(e.ground_truth is not None for e in result)
