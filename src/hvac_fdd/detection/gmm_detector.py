"""
Gaussian Mixture Model anomaly detector for the LBNL SDAHU dataset.

Training strategy: fit only on FaultType.NORMAL rows so the model learns
the boundaries of normal operation (multiple clusters/modes). Any deviation 
from that boundary is scored as anomalous using Negative Log-Likelihood (NLL).
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from hvac_fdd.config import Settings, get_settings
from hvac_fdd.detection.base import DetectorBase, NUMERIC_FEATURES, PREDICT_OUTPUT_COLS
from hvac_fdd.domain import AlertLevel, FaultType
from hvac_fdd.exceptions import DetectorNotFittedError, ModelPersistenceError

logger = logging.getLogger(__name__)

_DETECTOR_SOURCE = "gmm_detector"

# Percentile thresholds computed from the normal-only training NLL distribution.
_THRESHOLD_WARNING_PCT  = 90.0
_THRESHOLD_CRITICAL_PCT = 99.0
_TRAIN_SUBSAMPLE_RATIO  = 0.20  # Use 20% of data to speed up GMM fitting

class GMMDetector(DetectorBase):
    """
    Unsupervised anomaly detector trained exclusively on normal-operating data,
    using Gaussian Mixture Models to handle multi-modal distributions (e.g. summer vs winter).

    Fit: StandardScaler + sklearn GaussianMixture on a sample of FaultType.NORMAL rows.
    Predict: return rows flagged as anomalies with anomaly_index = Negative Log-Likelihood (NLL).
             Alert level assigned via thresholds stored during fit().
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self._settings = settings or get_settings()
        self._scaler:             StandardScaler | None = None
        self._model:              GaussianMixture | None = None
        self._threshold_warning:  float = 0.0
        self._threshold_critical: float = 0.0

    # ── DetectorBase interface ────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "GMMDetector":
        """
        Fit the scaler and GMM on FaultType.NORMAL rows only.
        
        Args:
            df: Feature-engineered DataFrame with a 'fault_type' column.

        Returns:
            self (for chaining).
        """
        s = self._settings
        normal_df = df[df["fault_type"] == FaultType.NORMAL.value]
        
        # Subsample for faster GMM fitting
        if len(normal_df) > 100_000:
            sample_size = int(len(normal_df) * _TRAIN_SUBSAMPLE_RATIO)
            logger.info("Subsampling %d normal rows to %d for GMM fitting", len(normal_df), sample_size)
            normal_df = normal_df.sample(n=sample_size, random_state=s.random_state)
            
        X = _clean_features(normal_df)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        logger.info("Fitting GMM with n_components=%d, covariance_type=%s...", 
                    s.gmm_n_components, s.gmm_covariance_type)
        
        self._model = GaussianMixture(
            n_components=s.gmm_n_components,
            covariance_type=s.gmm_covariance_type,
            random_state=s.random_state,
            max_iter=100,
            reg_covar=1e-3,  # Added to prevent positive definite errors from highly correlated features
        )
        self._model.fit(X_scaled)

        # Thresholds from the NORMAL training score distribution (NLL)
        # NLL = -log-likelihood (higher = more anomalous)
        train_nll = -self._model.score_samples(X_scaled)
        self._threshold_warning  = float(np.percentile(train_nll, _THRESHOLD_WARNING_PCT))
        self._threshold_critical = float(np.percentile(train_nll, _THRESHOLD_CRITICAL_PCT))

        self._is_fitted = True
        logger.info(
            "GMM fitted on %d normal rows (warning≥%.4f, critical≥%.4f)",
            len(X), self._threshold_warning, self._threshold_critical,
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all rows; return only rows flagged as anomalies.
        """
        self._require_fitted()

        # Identify rows that have complete feature vectors.
        valid_mask = df[NUMERIC_FEATURES].notna().all(axis=1)
        valid_df   = df[valid_mask]
        if valid_df.empty:
            return pd.DataFrame(columns=PREDICT_OUTPUT_COLS)

        X_scaled = self._scaler.transform(valid_df[NUMERIC_FEATURES].values)  # type: ignore[union-attr]
        nll      = -self._model.score_samples(X_scaled)                       # type: ignore[union-attr]

        anomaly_mask = nll >= self._threshold_warning
        if not anomaly_mask.any():
            return pd.DataFrame(columns=PREDICT_OUTPUT_COLS)

        sub = valid_df.iloc[np.where(anomaly_mask)[0]].copy()
        sub["anomaly_index"]   = nll[anomaly_mask]
        sub["violated_policy"] = "ANOMALY_SCORE"
        sub["trigger_signal"]  = "gmm_nll_score"
        sub["detector_source"] = _DETECTOR_SOURCE
        sub["alert_level"]     = sub["anomaly_index"].map(self._map_alert_level)

        logger.debug("GMMDetector.predict: %d anomalies / %d valid rows", len(sub), len(valid_df))
        return _select_output(sub)

    def calibrate_warning_threshold(
        self,
        df: pd.DataFrame,
        target_fpr: float,
    ) -> "GMMDetector":
        """Calibrate the warning threshold on validation normal-operation data.

        The supplied frame must belong to a validation split; it is never used
        during GMM fitting.  ``target_fpr`` is the maximum desired row-level
        false-positive rate on that validation normal set.
        """
        self._require_fitted()
        if not 0.0 < target_fpr < 1.0:
            raise ValueError("target_fpr must be between 0 and 1")

        normal_df = df[df["fault_type"] == FaultType.NORMAL.value]
        X = _clean_features(normal_df)
        if len(X) == 0:
            raise ValueError("Validation data contains no complete normal rows")

        X_scaled = self._scaler.transform(X)  # type: ignore[union-attr]
        scores = -self._model.score_samples(X_scaled)  # type: ignore[union-attr]
        self._threshold_warning = float(np.quantile(scores, 1.0 - target_fpr))
        if self._threshold_critical < self._threshold_warning:
            self._threshold_critical = self._threshold_warning
        logger.info(
            "GMM warning threshold calibrated on %d validation normal rows: "
            "target_fpr=%.4f, threshold=%.4f",
            len(X), target_fpr, self._threshold_warning,
        )
        return self

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Persist the fitted scaler and model to a single joblib file."""
        if not self._is_fitted:
            raise DetectorNotFittedError("Cannot save an unfitted GMMDetector")
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
        logger.info("GMMDetector saved to %s", out)

    @classmethod
    def load(
        cls,
        path: Path | str,
        settings: Settings | None = None,
    ) -> "GMMDetector":
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
        logger.info("GMMDetector loaded from %s", src)
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
