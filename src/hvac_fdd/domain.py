from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class FaultType(str, Enum):
    NORMAL       = "normal"
    COI_BIAS     = "coi_bias"      # Coil bias (sensor bias)
    COI_LEAKAGE  = "coi_leakage"   # Coil leakage
    COI_STUCK    = "coi_stuck"     # Coil valve stuck
    DAMPER_STUCK = "damper_stuck"  # Damper stuck
    OA_BIAS      = "oa_bias"       # Outside air sensor bias

class AlertLevel(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"

class DetectionEvent(BaseModel):
    event_time:       datetime
    zone_id:          str
    equipment_tag:    str          # Trigger source: rules | isolation_forest | classifier
    violated_policy:  str
    trigger_signal:   str          # Trigger signal name
    anomaly_index:    float
    alert_level:      AlertLevel
    ground_truth:     Optional[FaultType] = None   # From filename; None in production
    predicted_fault:  Optional[FaultType] = None   # Classifier prediction
    confidence:       Optional[float] = None        # Classifier confidence
