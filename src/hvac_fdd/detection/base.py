from abc import ABC, abstractmethod
import pandas as pd
from src.hvac_fdd.exceptions import DetectorNotFittedError

class DetectorBase(ABC):
    def __init__(self) -> None:
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "DetectorBase":
        """Default implementation for detectors that don't need training"""
        return self

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with [violated_policy, alert_level, anomaly_index, trigger_signal] columns"""

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).predict(df)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise DetectorNotFittedError(f"{self.__class__.__name__} is not fitted")
