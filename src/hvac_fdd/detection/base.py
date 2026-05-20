from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from hvac_fdd.exceptions import DetectorNotFittedError
from hvac_fdd.ingestion.features import ENG_FEATURE_COLUMNS

# Numeric feature columns shared by IsolationForestDetector and FaultClassifier.
# All columns are produced by ingestion/features.py::build_feature_set().
NUMERIC_FEATURES: list[str] = ENG_FEATURE_COLUMNS

# Standard output columns produced by every detector's predict() method.
PREDICT_OUTPUT_COLS: list[str] = [
    "event_time",
    "zone_id",
    "violated_policy",
    "alert_level",
    "anomaly_index",
    "trigger_signal",
    "detector_source",
    "ground_truth",
]

class DetectorBase(ABC):
    def __init__(self) -> None:
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "DetectorBase":
        """Default for stateless detectors (e.g. rules). Marks detector as fitted."""
        self._is_fitted = True
        return self

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with [violated_policy, alert_level, anomaly_index, trigger_signal] columns."""

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).predict(df)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise DetectorNotFittedError(f"{self.__class__.__name__} is not fitted")
