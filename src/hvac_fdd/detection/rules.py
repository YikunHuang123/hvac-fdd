"""
Rule-based fault detector for the LBNL SDAHU dataset.

Hybrid Expert System: BMS hardware tracking errors + APAR thermodynamic constraints.
NOTE: LBNL raw data uses [0,1] fractional scale for valve/damper, NOT [0,100].
"""
from __future__ import annotations

import logging

import pandas as pd

from hvac_fdd.config import Settings, get_settings
from hvac_fdd.detection.base import DetectorBase, PREDICT_OUTPUT_COLS
from hvac_fdd.domain import AlertLevel

logger = logging.getLogger(__name__)

_DETECTOR_SOURCE = "rules"

# LBNL stores CHWC_VLV and OA_DMPR in [0,1] fractional range.
# Thresholds below are in the same fractional scale.
_VALVE_STUCK_THRESHOLD  = 0.08   # 8% fractional error sustained over 15 min
_DAMPER_STUCK_THRESHOLD = 0.08   # 8% fractional error sustained over 15 min
_VALVE_OPEN_MIN         = 0.10   # Valve considered "significantly open" (10%)
_VALVE_CLOSED_MAX       = 0.02   # Valve considered "commanded closed" (2%)
_DAMPER_FULLY_OPEN_MIN  = 0.95   # Damper at >= 95% is "full-open"
_LEAKAGE_SA_DELTA_C     = 1.5    # SA must be this much cooler than MA to flag leakage (°C)
_MA_CONSERVATION_TOL_C  = 1.5    # MA temperature conservation tolerance (°C)
_SA_BIAS_DELTA_C        = 1.2    # SA vs setpoint sustained bias threshold (°C)


class LBNLRulesDetector(DetectorBase):
    """
    Hybrid expert detector for LBNL SDAHU.

    Rules:
      1. VALVE_STUCK      — tracking error sustained >8% (BMS hardware signal)
      2. DAMPER_STUCK     — tracking error sustained >8% (BMS hardware signal)
      3. COIL_LEAKAGE     — valve closed but SA cooling detected (APAR)
      4. SA_SETPOINT_BIAS — SA systematically deviates from setpoint (targets oa_bias/coi_bias)
      5. FAN_POWER_ANOMALY— 3-sigma fan power exceedance
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self._settings = settings or get_settings()
        self._is_fitted = True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all rules and return triggered events."""
        self._require_fitted()
        s = self._settings

        hits = [
            self._rule_valve_stuck(df),
            self._rule_damper_stuck(df),
            self._rule_coil_leakage(df),
            self._rule_sa_setpoint_bias(df),
            self._rule_fan_power_anomaly(df),
        ]

        non_empty = [h for h in hits if len(h) > 0]
        if not non_empty:
            return pd.DataFrame(columns=PREDICT_OUTPUT_COLS)

        result = pd.concat(non_empty, ignore_index=True)
        result = result.drop_duplicates(subset=["event_time", "zone_id"], keep="first")
        logger.debug("rules.predict: %d events fired", len(result))
        return result

    # ── Rule implementations ──────────────────────────────────────────────────

    @staticmethod
    def _rule_valve_stuck(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 1: Chilled-water valve not tracking its demand signal.
        Uses 15-min rolling mean of tracking error in [0,1] fractional scale.
        """
        mask = df["valve_tracking_err_15_mean"].abs() > _VALVE_STUCK_THRESHOLD
        sub = df[mask].copy()
        sub["violated_policy"] = "VALVE_STUCK"
        sub["alert_level"]     = AlertLevel.CRITICAL.value
        sub["anomaly_index"]   = sub["valve_tracking_err_15_mean"].abs() / _VALVE_STUCK_THRESHOLD
        sub["trigger_signal"]  = "chwc_valve_pct"
        sub["detector_source"] = _DETECTOR_SOURCE
        return _select_output(sub)

    @staticmethod
    def _rule_damper_stuck(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 2: OA damper not tracking its demand signal.
        Uses 15-min rolling mean of tracking error in [0,1] fractional scale.
        """
        mask = df["damper_tracking_err_15_mean"].abs() > _DAMPER_STUCK_THRESHOLD
        sub = df[mask].copy()
        sub["violated_policy"] = "DAMPER_STUCK"
        sub["alert_level"]     = AlertLevel.CRITICAL.value
        sub["anomaly_index"]   = sub["damper_tracking_err_15_mean"].abs() / _DAMPER_STUCK_THRESHOLD
        sub["trigger_signal"]  = "oa_damper_pct"
        sub["detector_source"] = _DETECTOR_SOURCE
        return _select_output(sub)

    @staticmethod
    def _rule_coil_leakage(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 3: APAR-style coil leakage.
        Valve commanded closed (<2% fractional) but supply air is significantly
        cooler than mixed air → cold water is bleeding through the coil.
        """
        mask = (
            (df["chwc_valve_pct"] < _VALVE_CLOSED_MAX)
            & (df["temp_supply_celsius"] < df["temp_mixed_celsius"] - _LEAKAGE_SA_DELTA_C)
        )
        sub = df[mask].copy()
        sub["violated_policy"] = "COIL_LEAKAGE"
        sub["alert_level"]     = AlertLevel.WARNING.value
        sub["anomaly_index"]   = (
            (df["temp_mixed_celsius"] - df["temp_supply_celsius"]) / _LEAKAGE_SA_DELTA_C
        )
        sub["trigger_signal"]  = "temp_supply_celsius"
        sub["detector_source"] = _DETECTOR_SOURCE
        return _select_output(sub)

    @staticmethod
    def _rule_sa_setpoint_bias(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 4: Sustained SA temperature bias vs setpoint.
        Targets oa_bias and coi_bias: when OA or coil sensor drifts, the controller
        mis-calculates mixing ratio / cooling demand, causing a persistent SA
        error that a 60-min rolling mean can expose.
        Only flags when the error is stable (low std), separating bias from oscillation.
        """
        mask = (
            (df["sa_temp_error_c_60_mean"].abs() > _SA_BIAS_DELTA_C)
            & (df["sa_temp_error_c_60_std"] < 0.8)   # stable / sustained, not oscillation
        )
        sub = df[mask].copy()
        sub["violated_policy"] = "SA_SETPOINT_BIAS"
        sub["alert_level"]     = AlertLevel.WARNING.value
        sub["anomaly_index"]   = sub["sa_temp_error_c_60_mean"].abs() / _SA_BIAS_DELTA_C
        sub["trigger_signal"]  = "temp_supply_celsius"
        sub["detector_source"] = _DETECTOR_SOURCE
        return _select_output(sub)

    @staticmethod
    def _rule_fan_power_anomaly(df: pd.DataFrame) -> pd.DataFrame:
        """Rule 5: Supply-fan power exceeds 3-sigma above 60-min rolling mean."""
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
    df = df.copy()
    if "fault_type" in df.columns:
        df["ground_truth"] = df["fault_type"]
    elif "ground_truth" not in df.columns:
        df["ground_truth"] = None
    return df[PREDICT_OUTPUT_COLS].reset_index(drop=True)
