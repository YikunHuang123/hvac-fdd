from __future__ import annotations

from fastapi import APIRouter, Depends

from hvac_fdd.api.deps import get_detection_repo
from hvac_fdd.api.schemas import FaultTypeStatsResponse, StatsResponse
from hvac_fdd.db.detections import DetectionRepository

router = APIRouter(prefix="/stats", tags=["Stats"])


@router.get("/", response_model=StatsResponse)
def get_stats(
    repo: DetectionRepository = Depends(get_detection_repo),
) -> StatsResponse:
    return StatsResponse(**repo.statistics())


@router.get("/by-fault-type", response_model=FaultTypeStatsResponse)
def get_stats_by_fault_type(
    repo: DetectionRepository = Depends(get_detection_repo),
) -> FaultTypeStatsResponse:
    stats = repo.statistics()
    return FaultTypeStatsResponse(by_fault_type=stats["by_fault_type"])
