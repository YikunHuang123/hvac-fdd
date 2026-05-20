from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from hvac_fdd.api.deps import get_job_repo
from hvac_fdd.api.schemas import PipelineJobOut, PipelineTriggerIn
from hvac_fdd.db.jobs import JobRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


def _run_pipeline_task(job_id: int, scenario: str, job_repo: JobRepository) -> None:
    """Background task: runs ingestion pipeline and persists job status."""
    try:
        from hvac_fdd.ingestion.pipeline import run_ingestion_pipeline

        job_repo.update_status(job_id, "running")
        df = run_ingestion_pipeline(save=False)
        job_repo.update_status(
            job_id,
            "done",
            records_processed=len(df),
            anomalies_found=0,
        )
        logger.info("Pipeline job %d completed: %d rows", job_id, len(df))
    except Exception as exc:
        logger.exception("Pipeline job %d failed: %s", job_id, exc)
        job_repo.update_status(job_id, "failed", error_msg=str(exc))


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
    background_tasks.add_task(_run_pipeline_task, job.id, body.scenario, job_repo)
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
