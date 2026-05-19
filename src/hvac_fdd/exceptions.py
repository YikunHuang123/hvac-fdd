class HVACFDDError(Exception):
    """Base class for all project exceptions"""

class DataLoadError(HVACFDDError):
    """Failed to read or parse data file"""

class SchemaValidationError(DataLoadError):
    """Missing data columns or type mismatch"""

class FeatureEngineeringError(HVACFDDError):
    """Error during feature calculation"""

class DetectorNotFittedError(HVACFDDError):
    """Called predict() before fit()"""

class ModelPersistenceError(HVACFDDError):
    """Failed to save or load model"""

class DatabaseError(HVACFDDError):
    """Database operation failed"""

class PipelineError(HVACFDDError):
    """Failure during pipeline orchestration"""
    def __init__(self, message: str, stage: str):
        super().__init__(message)
        self.stage = stage
