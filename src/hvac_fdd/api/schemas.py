from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_time: datetime
    zone_id: str
    equipment_id: str
    detector_source: str
    violated_policy: str
    trigger_signal: str
    anomaly_index: float
    alert_level: str
    ground_truth: Optional[str] = None
    predicted_fault: Optional[str] = None
    confidence: Optional[float] = None


class DetectionsQueryResponse(BaseModel):
    total: int
    skip: int
    page_size: int
    items: list[DetectionOut]


class PipelineTriggerIn(BaseModel):
    scenario: str = "all"


class PipelineJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    scenario: str
    status: str
    error_msg: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None
    records_processed: Optional[int] = None
    anomalies_found: Optional[int] = None


class StatsResponse(BaseModel):
    total: int
    by_alert_level: dict[str, int]
    by_fault_type: dict[str, int]
    by_equipment: dict[str, int]


class FaultTypeStatsResponse(BaseModel):
    by_fault_type: dict[str, int]
