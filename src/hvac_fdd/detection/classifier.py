"""
Supervised multi-class fault classifier for the LBNL SDAHU dataset.

Leverages the ground-truth labels encoded in LBNL filenames (six FaultType classes).
Intended train/test split: first ~9 months as training, last ~3 months as test,
computed from the global timestamp range across all 21 scenarios.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from hvac_fdd.config import Settings, get_settings
from hvac_fdd.detection.base import DetectorBase, NUMERIC_FEATURES, PREDICT_OUTPUT_COLS
from hvac_fdd.domain import AlertLevel, FaultType
from hvac_fdd.exceptions import DetectorNotFittedError, ModelPersistenceError

logger = logging.getLogger(__name__)

_DETECTOR_SOURCE = "classifier"

# Classifier output adds predicted_fault and confidence on top of PREDICT_OUTPUT_COLS.
CLASSIFIER_OUTPUT_COLS: list[str] = PREDICT_OUTPUT_COLS + ["predicted_fault", "confidence"]


class FaultClassifier(DetectorBase):
    """
    Supervised RandomForest classifier that labels each row with a FaultType.

    Fit: train on all provided rows using 'fault_type' as the target label.
         Caller is responsible for supplying only training-split data.
    Predict: return rows where the predicted class is not FaultType.NORMAL,
             annotated with predicted_fault and confidence (max class probability).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self._settings      = settings or get_settings()
        self._model:   RandomForestClassifier | XGBClassifier | None = None
        self._encoder: LabelEncoder | None = None

    # ── DetectorBase interface ────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FaultClassifier":
        """
        Train the classifier on all rows in df.

        Args:
            df: Feature-engineered DataFrame with a 'fault_type' column.
                NaN rows (lag/rolling window start) are silently dropped.

        Returns:
            self (for chaining).
        """
        clean = df[NUMERIC_FEATURES + ["fault_type"]].dropna()
        X = clean[NUMERIC_FEATURES].values
        y_raw = clean["fault_type"].values

        self._encoder = LabelEncoder()
        y = self._encoder.fit_transform(y_raw)

        model_type = self._settings.supervised_model.lower()
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self._model = RandomForestClassifier(
                n_estimators=self._settings.clf_n_estimators,
                random_state=self._settings.random_state,
                n_jobs=-1,
                verbose=1
            )
        elif model_type == "xgboost":
            from xgboost import XGBClassifier
            self._model = XGBClassifier(
                n_estimators=self._settings.clf_n_estimators,
                random_state=self._settings.random_state,
                n_jobs=-1,
                tree_method="hist",
                verbosity=1
            )
        elif model_type == "gnn":
            from hvac_fdd.detection.gnn_classifier import GNNClassifierWrapper
            self._model = GNNClassifierWrapper(self._settings)
        elif model_type == "tcn":
            from hvac_fdd.detection.tcn_classifier import TCNClassifierWrapper
            self._model = TCNClassifierWrapper(self._settings)
        elif model_type == "hierarchical_xgb":
            from hvac_fdd.detection.hierarchical_classifier import HierarchicalXGBWrapper
            self._model = HierarchicalXGBWrapper(self._settings)
        else:
            raise ValueError(f"Unknown supervised model type: {model_type}")
            
        if model_type == "hierarchical_xgb":
            self._model.fit(X, y, classes=self._encoder.classes_)
        else:
            self._model.fit(X, y)
            
        self._is_fitted = True
        unique, counts = np.unique(y_raw, return_counts=True)
        logger.info(
            "FaultClassifier fitted on %d rows (%d classes: %s) | class distribution: %s",
            len(X), len(self._encoder.classes_), list(self._encoder.classes_),
            dict(zip(unique, counts.tolist())),
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all rows; return only rows predicted as a non-NORMAL fault.

        Args:
            df: Feature-engineered DataFrame.

        Returns:
            DataFrame with CLASSIFIER_OUTPUT_COLS columns.
        """
        self._require_fitted()

        valid_mask = df[NUMERIC_FEATURES].notna().all(axis=1)
        valid_df   = df[valid_mask]
        if valid_df.empty:
            return pd.DataFrame(columns=CLASSIFIER_OUTPUT_COLS)

        X          = valid_df[NUMERIC_FEATURES].values
        proba      = self._model.predict_proba(X)           # type: ignore[union-attr]
        pred_idx   = proba.argmax(axis=1)
        confidence = proba.max(axis=1)
        pred_labels = self._encoder.inverse_transform(pred_idx)  # type: ignore[union-attr]

        fault_mask = pred_labels != FaultType.NORMAL.value
        if not fault_mask.any():
            return pd.DataFrame(columns=CLASSIFIER_OUTPUT_COLS)

        sub                    = valid_df.iloc[np.where(fault_mask)[0]].copy()
        sub["predicted_fault"] = pred_labels[fault_mask]
        sub["confidence"]      = confidence[fault_mask]
        sub["violated_policy"] = sub["predicted_fault"]  # predicted fault IS the policy
        sub["alert_level"]     = sub["confidence"].apply(
            _map_alert_level,
            critical=self._settings.classifier_conf_critical,
            warning=self._settings.classifier_conf_warning,
        )
        sub["anomaly_index"]   = sub["confidence"]       # higher confidence = stronger signal
        sub["trigger_signal"]  = "fault_classifier"
        sub["detector_source"] = _DETECTOR_SOURCE

        logger.debug("FaultClassifier.predict: %d non-normal rows / %d valid", len(sub), len(valid_df))
        return _select_output(sub)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Persist the fitted model and label encoder to a single joblib file."""
        if not self._is_fitted:
            raise DetectorNotFittedError("Cannot save an unfitted FaultClassifier")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump({"model": self._model, "encoder": self._encoder}, out)
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to save classifier to {out}: {exc}") from exc
        logger.info("FaultClassifier saved to %s", out)

    @classmethod
    def load(
        cls,
        path: Path | str,
        settings: Settings | None = None,
    ) -> "FaultClassifier":
        """Load a previously saved classifier from a joblib file."""
        src = Path(path)
        try:
            data = joblib.load(src)
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to load classifier from {src}: {exc}") from exc
        obj = cls(settings=settings)
        obj._model   = data["model"]
        obj._encoder = data["encoder"]
        obj._is_fitted = True
        logger.info("FaultClassifier loaded from %s", src)
        return obj


# ── Module helpers ────────────────────────────────────────────────────────────


def _map_alert_level(confidence: float, critical: float, warning: float) -> str:
    if confidence >= critical:
        return AlertLevel.CRITICAL.value
    if confidence >= warning:
        return AlertLevel.WARNING.value
    return AlertLevel.INFO.value


def _select_output(df: pd.DataFrame) -> pd.DataFrame:
    if "ground_truth" not in df.columns:
        df = df.copy()
        df["ground_truth"] = None
    return df[CLASSIFIER_OUTPUT_COLS].reset_index(drop=True)
