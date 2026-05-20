from enum import Enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FaultType(str, Enum):
    NORMAL       = "normal"
    COI_BIAS     = "coi_bias"      # Coil bias (supply-air sensor offset)
    COI_LEAKAGE  = "coi_leakage"   # Chilled-water coil leakage
    COI_STUCK    = "coi_stuck"     # Chilled-water valve stuck
    DAMPER_STUCK = "damper_stuck"  # Outside-air damper stuck
    OA_BIAS      = "oa_bias"       # Outside-air temperature sensor bias


class AlertLevel(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"

    # Severity order: INFO < WARNING < CRITICAL
    _severity: dict = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}

    def __lt__(self, other: "AlertLevel") -> bool:
        return self._severity[self.value] < self._severity[other.value]

    def __le__(self, other: "AlertLevel") -> bool:
        return self._severity[self.value] <= self._severity[other.value]

    def __gt__(self, other: "AlertLevel") -> bool:
        return self._severity[self.value] > self._severity[other.value]

    def __ge__(self, other: "AlertLevel") -> bool:
        return self._severity[self.value] >= self._severity[other.value]


class DetectionEvent(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,   # allows model_validate(orm_obj)
        use_enum_values=True,   # serializes enums to their .value string
    )

    event_time:      datetime
    zone_id:         str   = Field(min_length=1, max_length=32)
    equipment_id:    str   = Field(min_length=1, max_length=32)  # Equipment tag, e.g. "AHU-1"
    detector_source: str   = Field(max_length=32)                # "rules" | "isolation_forest" | "classifier"
    violated_policy: str   = Field(max_length=64)
    trigger_signal:  str   = Field(max_length=64)                # Signal name that triggered the alert
    anomaly_index:   float = Field(ge=0.0)                       # Higher = more anomalous
    alert_level:     AlertLevel
    ground_truth:    Optional[FaultType] = None  # From filename label; None in production
    predicted_fault: Optional[FaultType] = None  # Supervised classifier output
    confidence:      Optional[float]     = Field(default=None, ge=0.0, le=1.0)
