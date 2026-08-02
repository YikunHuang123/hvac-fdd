from __future__ import annotations

import logging
import multiprocessing

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from hvac_fdd.api.deps import get_job_repo
from hvac_fdd.api.schemas import PipelineJobOut, PipelineTriggerIn
from hvac_fdd.config import get_settings
from hvac_fdd.db.base import make_engine, make_session_factory
from hvac_fdd.db.jobs import JobRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


def _run_pipeline_task(job_id: int, scenario: str, database_url: str) -> None:
    """Run ingestion in a worker process and persist the job status."""
    settings = get_settings()
    engine = make_engine(
        database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    session_factory = make_session_factory(engine)
    session = session_factory()
    job_repo = JobRepository(session)
    try:
        from hvac_fdd.ingestion.pipeline import iter_ingestion_pipeline

        job_repo.update_status(job_id, "running")
        session.commit()
        records_processed = 0
        for frame in iter_ingestion_pipeline(scenario=scenario):
            records_processed += len(frame)
            del frame
        job_repo.update_status(
            job_id,
            "done",
            records_processed=records_processed,
            anomalies_found=0,
        )
        session.commit()
        logger.info("Pipeline job %d completed: %d rows", job_id, records_processed)
    except Exception as exc:
        logger.exception("Pipeline job %d failed: %s", job_id, exc)
        try:
            job_repo.update_status(job_id, "failed", error_msg=str(exc))
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Could not persist failure for pipeline job %d", job_id)
    finally:
        session.close()
        engine.dispose()


def _start_pipeline_process(job_id: int, scenario: str, database_url: str) -> None:
    """Start the heavy pipeline outside the API worker thread."""
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_pipeline_task,
        args=(job_id, scenario, database_url),
        daemon=True,
    )
    process.start()
    logger.info("Started pipeline worker pid=%s for job %d", process.pid, job_id)


@router.post("/run", response_model=PipelineJobOut, status_code=202)
def trigger_pipeline(
    body: PipelineTriggerIn,
    background_tasks: BackgroundTasks,
    job_repo: JobRepository = Depends(get_job_repo),
) -> PipelineJobOut:
    """Trigger the ingestion pipeline asynchronously.

    Returns the created job immediately (status=pending). Poll
    GET /pipeline/jobs/{id} to track progress.
    """
    job = job_repo.create(body.scenario)
    # Persist the job before the background task starts; request dependencies
    # are finalized only after the response and must not be reused by the task.
    job_repo.commit()
    background_tasks.add_task(
        _start_pipeline_process,
        job.id,
        body.scenario,
        get_settings().database_url,
    )
    return PipelineJobOut.model_validate(job)


@router.get("/jobs/{job_id}", response_model=PipelineJobOut)
def get_job(
    job_id: int,
    job_repo: JobRepository = Depends(get_job_repo),
) -> PipelineJobOut:
    """Return the current status of a pipeline job."""
    job = job_repo.get_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return PipelineJobOut.model_validate(job)
