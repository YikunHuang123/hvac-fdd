class HVACFDDError(Exception):
    """Base class for all project exceptions."""

class ConfigurationError(HVACFDDError):
    """Invalid or missing configuration value."""

class DataLoadError(HVACFDDError):
    """Failed to read or parse a data file."""

class SchemaValidationError(DataLoadError):
    """Missing columns or type mismatch in loaded data."""

class FeatureEngineeringError(HVACFDDError):
    """Error during feature calculation."""

class DetectorNotFittedError(HVACFDDError):
    """predict() called before fit()."""

class DetectionError(HVACFDDError):
    """Runtime error during detection execution (distinct from fit errors)."""

class ModelPersistenceError(HVACFDDError):
    """Failed to save or load a model artifact."""

class DatabaseError(HVACFDDError):
    """Database operation failed."""

class PipelineError(HVACFDDError):
    """Failure during pipeline orchestration; includes the failing stage name."""

    def __init__(self, message: str, stage: str) -> None:
        super().__init__(f"[{stage}] {message}")  # stage visible in str(e) and logs
        self.stage = stage
