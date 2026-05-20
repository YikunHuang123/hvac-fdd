from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from hvac_fdd.api import detections, health, pipeline, stats
from hvac_fdd.api.middleware import RequestTracingMiddleware
from hvac_fdd.config import get_settings
from hvac_fdd.db.base import Base, make_engine, make_session_factory
from hvac_fdd.exceptions import DatabaseError, HVACFDDError


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = make_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    Base.metadata.create_all(engine)
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    yield
    engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="HVAC FDD API", version="3.0.0", lifespan=lifespan)

    settings = get_settings()
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(detections.router)
    app.include_router(pipeline.router)
    app.include_router(stats.router)

    @app.exception_handler(HVACFDDError)
    async def hvac_error_handler(request: Request, exc: HVACFDDError) -> JSONResponse:
        status_code = 503 if isinstance(exc, DatabaseError) else 500
        return JSONResponse(
            status_code=status_code,
            content={"detail": str(exc)},
        )

    return app
