"""
Rule-based fault detector for the LBNL SDAHU dataset.

Five rules grounded in ASHRAE Guideline 36 and the LBNL fault mechanisms.
The detector is stateless: fit() (inherited) marks it as ready, and predict()
operates directly on feature columns produced by build_feature_set().
"""
from __future__ import annotations

import logging

import pandas as pd

from hvac_fdd.config import Settings, get_settings
from hvac_fdd.detection.base import DetectorBase, PREDICT_OUTPUT_COLS
from hvac_fdd.domain import AlertLevel

logger = logging.getLogger(__name__)

_DETECTOR_SOURCE = "rules"

# todo: 看笔记，目前的规则不符合专家定义。
class LBNLRulesDetector(DetectorBase):
    """
    Rule-based detector mapped to the five LBNL fault scenarios.

    Rules:
      VALVE_STUCK          → coi_stuck
      DAMPER_STUCK         → damper_stuck
      SUPPLY_AIR_TEMP_BIAS → coi_bias / oa_bias
      COIL_LEAKAGE         → coi_leakage
      FAN_POWER_ANOMALY    → generic AHU overload

    No training is required. The detector is ready immediately after construction.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self._settings = settings or get_settings()
        self._is_fitted = True  # stateless; no training step

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all five rules and return a DataFrame of triggered events.

        Args:
            df: Feature-engineered DataFrame (output of build_feature_set).

        Returns:
            DataFrame with PREDICT_OUTPUT_COLS columns.
            Only rows where at least one rule fires are included.
        """
        self._require_fitted()
        s = self._settings

        hits = [
            self._rule_valve_stuck(df, s),
            self._rule_damper_stuck(df, s),
            self._rule_supply_air_temp_bias(df, s),
            self._rule_coil_leakage(df, s),
            self._rule_fan_power_anomaly(df),
        ]

        non_empty = [h for h in hits if len(h) > 0]
        if not non_empty:
            return pd.DataFrame(columns=PREDICT_OUTPUT_COLS)

        result = pd.concat(non_empty, ignore_index=True)
        logger.debug("rules.predict: %d events fired", len(result))
        return result

    # ── Rule implementations ──────────────────────────────────────────────────

    @staticmethod
    def _rule_valve_stuck(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
        """Rule 1: chilled-water valve not tracking its demand signal (→ coi_stuck)."""
        mask = df["valve_tracking_err_15_mean"].abs() > s.valve_tracking_threshold_pct
        sub = df[mask].copy()
        sub["violated_policy"] = "VALVE_STUCK"
        sub["alert_level"]     = AlertLevel.CRITICAL.value
        sub["anomaly_index"]   = (
            sub["valve_tracking_err_15_mean"].abs() / s.valve_tracking_threshold_pct
        )
        sub["trigger_signal"]  = "chwc_valve_pct"
        sub["detector_source"] = _DETECTOR_SOURCE
        return _select_output(sub)

    @staticmethod
    def _rule_damper_stuck(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
        """Rule 2: outside-air damper not tracking its demand signal (→ damper_stuck)."""
        mask = df["damper_tracking_err_15_mean"].abs() > s.damper_tracking_threshold_pct
        sub = df[mask].copy()
        sub["violated_policy"] = "DAMPER_STUCK"
        sub["alert_level"]     = AlertLevel.CRITICAL.value
        sub["anomaly_index"]   = (
            sub["damper_tracking_err_15_mean"].abs() / s.damper_tracking_threshold_pct
        )
        sub["trigger_signal"]  = "oa_damper_pct"
        sub["detector_source"] = _DETECTOR_SOURCE
        return _select_output(sub)

    @staticmethod
    def _rule_supply_air_temp_bias(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
        """Rule 3: systematic supply-air temp offset — bias not oscillation (→ coi_bias / oa_bias)."""
        mask = (
            (df["sa_temp_error_c_60_mean"].abs() > s.sa_temp_error_threshold_c)
            & (df["sa_temp_error_c_60_std"] < 0.5)
        )
        sub = df[mask].copy()
        sub["violated_policy"] = "SUPPLY_AIR_TEMP_BIAS"
        sub["alert_level"]     = AlertLevel.WARNING.value
        sub["anomaly_index"]   = (
            sub["sa_temp_error_c_60_mean"].abs() / s.sa_temp_error_threshold_c
        )
        sub["trigger_signal"]  = "temp_supply_celsius"
        sub["detector_source"] = _DETECTOR_SOURCE
        return _select_output(sub)

    @staticmethod
    def _rule_coil_leakage(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
        """Rule 4: valve nearly closed yet supply air is unexpectedly cold (→ coi_leakage)."""
        mask = (
            (df["chwc_valve_pct"] < 5.0)
            & (df["sa_temp_error_c"] < -s.sa_temp_error_threshold_c)
        )
        sub = df[mask].copy()
        sub["violated_policy"] = "COIL_LEAKAGE"
        sub["alert_level"]     = AlertLevel.WARNING.value
        sub["anomaly_index"]   = (
            sub["sa_temp_error_c"].abs() / s.sa_temp_error_threshold_c
        )
        sub["trigger_signal"]  = "temp_supply_celsius"
        sub["detector_source"] = _DETECTOR_SOURCE
        return _select_output(sub)

    @staticmethod
    def _rule_fan_power_anomaly(df: pd.DataFrame) -> pd.DataFrame:
        """Rule 5: supply-fan power exceeds 3 sigma above the 60-min rolling mean."""
        # Skip rows where std == 0 (insufficient history); 3 * 0 would flag everything.
        mask = (
            (df["sf_power_w_60_std"] > 0.0)
            & (df["sf_power_w"] > df["sf_power_w_60_mean"] + 3.0 * df["sf_power_w_60_std"])
        )
        sub = df[mask].copy()
        sub["violated_policy"] = "FAN_POWER_ANOMALY"
        sub["alert_level"]     = AlertLevel.INFO.value
        sub["anomaly_index"]   = (
            (sub["sf_power_w"] - sub["sf_power_w_60_mean"]) / sub["sf_power_w_60_std"]
        )
        sub["trigger_signal"]  = "sf_power_w"
        sub["detector_source"] = _DETECTOR_SOURCE
        return _select_output(sub)


# ── Module helper ─────────────────────────────────────────────────────────────


def _select_output(df: pd.DataFrame) -> pd.DataFrame:
    """Return only PREDICT_OUTPUT_COLS; add ground_truth column when absent."""
    if "ground_truth" not in df.columns:
        df = df.copy()
        df["ground_truth"] = None
    return df[PREDICT_OUTPUT_COLS].reset_index(drop=True)
