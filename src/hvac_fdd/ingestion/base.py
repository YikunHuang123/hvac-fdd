from abc import ABC, abstractmethod

import pandas as pd
from hvac_fdd.domain import FaultType


class DataLoaderBase(ABC):
    @abstractmethod
    def load_all(self, chunksize: int | None = None) -> pd.DataFrame:
        """Load all available scenarios and return a combined DataFrame."""

    @abstractmethod
    def load_scenario(self, fault_type: FaultType) -> pd.DataFrame:
        """Load data for a single fault scenario."""

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validate column names and dtypes; raise SchemaValidationError on failure."""
