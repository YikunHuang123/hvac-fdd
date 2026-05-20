from __future__ import annotations

from typing import Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from hvac_fdd.db.detections import DetectionRepository
from hvac_fdd.db.jobs import JobRepository


def get_db_session(request: Request) -> Generator[Session, None, None]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_detection_repo(
    session: Session = Depends(get_db_session),
) -> DetectionRepository:
    return DetectionRepository(session)


def get_job_repo(
    session: Session = Depends(get_db_session),
) -> JobRepository:
    return JobRepository(session)
