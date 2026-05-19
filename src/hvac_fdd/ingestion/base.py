from abc import ABC, abstractmethod
import pandas as pd

class DataLoaderBase(ABC):
    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """Load and return raw DataFrame"""

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validate column names and types, raise SchemaValidationError on failure"""
