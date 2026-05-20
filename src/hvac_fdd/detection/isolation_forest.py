"""
Isolation Forest anomaly detector for the LBNL SDAHU dataset.

Training strategy: fit only on FaultType.NORMAL rows so the model learns
the boundary of normal operation. Any deviation from that boundary is scored
as anomalous.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from hvac_fdd.config import Settings, get_settings
from hvac_fdd.detection.base import DetectorBase, NUMERIC_FEATURES, PREDICT_OUTPUT_COLS
from hvac_fdd.domain import AlertLevel, FaultType
from hvac_fdd.exceptions import DetectorNotFittedError, ModelPersistenceError

logger = logging.getLogger(__name__)

_DETECTOR_SOURCE = "isolation_forest"

# Percentile offset above the 75th-percentile training threshold that triggers CRITICAL.
_CRITICAL_OFFSET_PCT = 10.0


class IsolationForestDetector(DetectorBase):
    """
    Unsupervised anomaly detector trained exclusively on normal-operating data.

    Fit: StandardScaler + sklearn IsolationForest on FaultType.NORMAL rows only.
    Predict: return rows flagged as anomalies (-1) with anomaly_index = -score_samples
             (higher → more anomalous). Alert level assigned via thresholds stored
             during fit().
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self._settings = settings or get_settings()
        self._scaler:             StandardScaler | None = None
        self._model:              IsolationForest | None = None
        self._threshold_warning:  float = 0.0
        self._threshold_critical: float = 0.0

    # ── DetectorBase interface ────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        """
        Fit the scaler and IsolationForest on FaultType.NORMAL rows only.

        Args:
            df: Feature-engineered DataFrame with a 'fault_type' column.

        Returns:
            self (for chaining).
        """
        s = self._settings
        normal_df = df[df["fault_type"] == FaultType.NORMAL.value]
        X = _clean_features(normal_df)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = IsolationForest(
            contamination=s.if_contamination,
            n_estimators=s.if_n_estimators,
            random_state=s.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)

        # Thresholds from the NORMAL training score distribution.
        train_scores = -self._model.score_samples(X_scaled)  # higher = more anomalous
        self._threshold_warning  = float(np.percentile(train_scores, 75.0))
        self._threshold_critical = float(np.percentile(train_scores, 75.0 + _CRITICAL_OFFSET_PCT))

        self._is_fitted = True
        logger.info(
            "IsolationForest fitted on %d normal rows (warning≥%.4f, critical≥%.4f)",
            len(X), self._threshold_warning, self._threshold_critical,
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all rows; return only rows flagged as anomalies.

        Rows with NaN in any NUMERIC_FEATURES column (lag/rolling window start)
        are excluded from scoring and never returned as detections.

        Args:
            df: Feature-engineered DataFrame.

        Returns:
            DataFrame with PREDICT_OUTPUT_COLS columns.
        """
        self._require_fitted()

        # Identify rows that have complete feature vectors.
        valid_mask = df[NUMERIC_FEATURES].notna().all(axis=1)
        valid_df   = df[valid_mask]
        if valid_df.empty:
            return pd.DataFrame(columns=PREDICT_OUTPUT_COLS)

        X_scaled = self._scaler.transform(valid_df[NUMERIC_FEATURES].values)  # type: ignore[union-attr]
        flags    = self._model.predict(X_scaled)                               # type: ignore[union-attr]
        scores   = -self._model.score_samples(X_scaled)                        # type: ignore[union-attr]

        anomaly_mask = flags == -1
        if not anomaly_mask.any():
            return pd.DataFrame(columns=PREDICT_OUTPUT_COLS)

        sub = valid_df.iloc[np.where(anomaly_mask)[0]].copy()
        sub["anomaly_index"]   = scores[anomaly_mask]
        sub["violated_policy"] = "ANOMALY_SCORE"
        sub["trigger_signal"]  = "isolation_forest_score"
        sub["detector_source"] = _DETECTOR_SOURCE
        sub["alert_level"]     = sub["anomaly_index"].map(self._map_alert_level)

        logger.debug("IsolationForest.predict: %d anomalies / %d valid rows", len(sub), len(valid_df))
        return _select_output(sub)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Persist the fitted scaler and model to a single joblib file."""
        if not self._is_fitted:
            raise DetectorNotFittedError("Cannot save an unfitted IsolationForestDetector")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(
                {
                    "scaler":             self._scaler,
                    "model":              self._model,
                    "threshold_warning":  self._threshold_warning,
                    "threshold_critical": self._threshold_critical,
                },
                out,
            )
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to save model to {out}: {exc}") from exc
        logger.info("IsolationForestDetector saved to %s", out)

    @classmethod
    def load(
        cls,
        path: Path | str,
        settings: Settings | None = None,
    ) -> "IsolationForestDetector":
        """Load a previously saved detector from a joblib file."""
        src = Path(path)
        try:
            data = joblib.load(src)
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to load model from {src}: {exc}") from exc
        obj = cls(settings=settings)
        obj._scaler             = data["scaler"]
        obj._model              = data["model"]
        obj._threshold_warning  = data["threshold_warning"]
        obj._threshold_critical = data["threshold_critical"]
        obj._is_fitted          = True
        logger.info("IsolationForestDetector loaded from %s", src)
        return obj

    # ── Private helpers ───────────────────────────────────────────────────────

    def _map_alert_level(self, score: float) -> str:
        if score >= self._threshold_critical:
            return AlertLevel.CRITICAL.value
        if score >= self._threshold_warning:
            return AlertLevel.WARNING.value
        return AlertLevel.INFO.value


# ── Module helpers ────────────────────────────────────────────────────────────


def _clean_features(df: pd.DataFrame) -> np.ndarray:
    """Extract NUMERIC_FEATURES and drop rows that contain NaN (lag/rolling start)."""
    return df[NUMERIC_FEATURES].dropna().values


def _select_output(df: pd.DataFrame) -> pd.DataFrame:
    if "ground_truth" not in df.columns:
        df = df.copy()
        df["ground_truth"] = None
    return df[PREDICT_OUTPUT_COLS].reset_index(drop=True)
