from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query

from hvac_fdd.api.deps import get_detection_repo
from hvac_fdd.api.schemas import DetectionOut, DetectionsQueryResponse
from hvac_fdd.db.detections import DetectionFilter, DetectionRepository
from hvac_fdd.domain import AlertLevel

router = APIRouter(prefix="/api/v1/detections", tags=["Detections"])


@router.get("/", response_model=DetectionsQueryResponse)
def get_detections(
    start: Optional[datetime] = Query(None, description="ISO 8601 UTC start time"),
    end: Optional[datetime] = Query(None, description="ISO 8601 UTC end time"),
    zone_id: Optional[str] = Query(None),
    alert_level: Optional[AlertLevel] = Query(None),
    fault_type: Optional[str] = Query(None),
    page_size: int = Query(500, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    repo: DetectionRepository = Depends(get_detection_repo),
) -> DetectionsQueryResponse:
    filters = DetectionFilter(
        start=start,
        end=end,
        zone_id=zone_id,
        alert_level=alert_level.value if alert_level else None,
        fault_type=fault_type,
    )
    total = repo.count(filters)
    items = repo.query(filters, page_size=page_size, skip=skip)
    return DetectionsQueryResponse(
        total=total,
        skip=skip,
        page_size=page_size,
        items=[DetectionOut.model_validate(row) for row in items],
    )
