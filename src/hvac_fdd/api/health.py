from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text

router = APIRouter(tags=["Health"])


@router.get("/health/live")
def health_live() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
def health_ready(request: Request) -> dict:
    try:
        with request.app.state.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
