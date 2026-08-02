from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from hvac_fdd.db.orm import PipelineJobORM
from hvac_fdd.exceptions import DatabaseError

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"done", "failed"})


class JobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, scenario: str) -> PipelineJobORM:
        """Create a new pipeline job in 'pending' status."""
        try:
            job = PipelineJobORM(
                scenario=scenario,
                status="pending",
                started_at=datetime.now(tz=timezone.utc),
            )
            self._session.add(job)
            self._session.flush()
            logger.info("JobRepository: created job id=%d scenario=%r", job.id, scenario)
            return job
        except Exception as exc:
            raise DatabaseError(f"create job failed: {exc}") from exc

    def commit(self) -> None:
        """Commit the current transaction before handing work to a background task."""
        try:
            self._session.commit()
        except Exception as exc:
            self._session.rollback()
            raise DatabaseError(f"job transaction commit failed: {exc}") from exc

    def get_by_id(self, job_id: int) -> Optional[PipelineJobORM]:
        """Return the job with the given id, or None if not found."""
        return self._session.scalar(
            select(PipelineJobORM).where(PipelineJobORM.id == job_id)
        )

    def update_status(
        self,
        job_id: int,
        status: str,
        *,
        error_msg: Optional[str] = None,
        records_processed: Optional[int] = None,
        anomalies_found: Optional[int] = None,
    ) -> PipelineJobORM:
        """Update job status and optional result fields.

        Sets finished_at automatically when status is 'done' or 'failed'.
        """
        job = self.get_by_id(job_id)
        if job is None:
            raise DatabaseError(f"Job {job_id} not found")
        try:
            job.status = status
            if error_msg is not None:
                job.error_msg = error_msg
            if records_processed is not None:
                job.records_processed = records_processed
            if anomalies_found is not None:
                job.anomalies_found = anomalies_found
            if status in _TERMINAL_STATUSES:
                job.finished_at = datetime.now(tz=timezone.utc)
            self._session.flush()
            return job
        except Exception as exc:
            raise DatabaseError(f"update_status failed for job {job_id}: {exc}") from exc

    def list_recent(self, limit: int = 20) -> list[PipelineJobORM]:
        """Return the most recent jobs, newest first."""
        return list(
            self._session.scalars(
                select(PipelineJobORM)
                .order_by(PipelineJobORM.started_at.desc())
                .limit(limit)
            ).all()
        )
