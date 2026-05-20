from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hvac_fdd.db.orm import DetectionORM
from hvac_fdd.domain import DetectionEvent
from hvac_fdd.exceptions import DatabaseError

logger = logging.getLogger(__name__)


@dataclass
class DetectionFilter:
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    zone_id: Optional[str] = None
    alert_level: Optional[str] = None   # AlertLevel.value string, e.g. "WARNING"
    fault_type: Optional[str] = None    # FaultType.value string, matched against predicted_fault


class DetectionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def insert_bulk(self, events: list[DetectionEvent]) -> int:
        """Bulk-insert domain events. Returns number of rows inserted."""
        if not events:
            return 0
        try:
            # DetectionEvent uses use_enum_values=True so enum fields are already strings.
            rows = [
                DetectionORM(
                    event_time=e.event_time,
                    zone_id=e.zone_id,
                    equipment_id=e.equipment_id,
                    detector_source=e.detector_source,
                    violated_policy=e.violated_policy,
                    trigger_signal=e.trigger_signal,
                    anomaly_index=e.anomaly_index,
                    alert_level=e.alert_level,
                    ground_truth=e.ground_truth,
                    predicted_fault=e.predicted_fault,
                    confidence=e.confidence,
                )
                for e in events
            ]
            self._session.add_all(rows)
            self._session.flush()
            logger.info("DetectionRepository: inserted %d rows", len(rows))
            return len(rows)
        except Exception as exc:
            raise DatabaseError(f"insert_bulk failed: {exc}") from exc

    def query(
        self,
        filters: DetectionFilter,
        page_size: int = 500,
        skip: int = 0,
    ) -> list[DetectionORM]:
        """Return detections matching filters, newest first, with pagination."""
        stmt = select(DetectionORM).order_by(DetectionORM.event_time.desc())
        stmt = self._apply_filters(stmt, filters)
        stmt = stmt.offset(skip).limit(page_size)
        return list(self._session.scalars(stmt).all())

    def count(self, filters: DetectionFilter) -> int:
        """Count detections matching filters."""
        stmt = select(func.count()).select_from(DetectionORM)
        stmt = self._apply_filters(stmt, filters)
        return self._session.scalar(stmt) or 0

    def statistics(self) -> dict:
        """Aggregate detection counts grouped by alert_level, predicted_fault, and equipment_id."""
        total = self._session.scalar(
            select(func.count()).select_from(DetectionORM)
        ) or 0

        by_level = {
            row[0]: row[1]
            for row in self._session.execute(
                select(DetectionORM.alert_level, func.count()).group_by(DetectionORM.alert_level)
            )
        }
        by_fault = {
            (row[0] or "unknown"): row[1]
            for row in self._session.execute(
                select(DetectionORM.predicted_fault, func.count()).group_by(DetectionORM.predicted_fault)
            )
        }
        by_equipment = {
            row[0]: row[1]
            for row in self._session.execute(
                select(DetectionORM.equipment_id, func.count()).group_by(DetectionORM.equipment_id)
            )
        }

        return {
            "total": total,
            "by_alert_level": by_level,
            "by_fault_type": by_fault,
            "by_equipment": by_equipment,
        }

    def _apply_filters(self, stmt, filters: DetectionFilter):
        if filters.start is not None:
            stmt = stmt.where(DetectionORM.event_time >= filters.start)
        if filters.end is not None:
            stmt = stmt.where(DetectionORM.event_time <= filters.end)
        if filters.zone_id is not None:
            stmt = stmt.where(DetectionORM.zone_id == filters.zone_id)
        if filters.alert_level is not None:
            stmt = stmt.where(DetectionORM.alert_level == filters.alert_level)
        if filters.fault_type is not None:
            stmt = stmt.where(DetectionORM.predicted_fault == filters.fault_type)
        return stmt
