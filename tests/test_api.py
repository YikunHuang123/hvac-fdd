"""
Integration tests for the API layer.

Uses starlette.testclient.TestClient (synchronous) because the entire DB
layer is synchronous SQLAlchemy.  Each test receives an `api_client` fixture
that overrides `get_db_session` to inject the transactional test session from
conftest.py — so DB state is rolled back after every test.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator

import pytest
from sqlalchemy.orm import Session
from starlette.testclient import TestClient  # sync; no anyio needed for sync repos

from hvac_fdd.api import create_app
from hvac_fdd.api.deps import get_db_session
from hvac_fdd.db.detections import DetectionRepository
from hvac_fdd.db.jobs import JobRepository
from hvac_fdd.domain import AlertLevel, DetectionEvent, FaultType


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_event(
    *,
    zone_id: str = "ZONE-1",
    equipment_id: str = "AHU-1",
    alert_level: str = AlertLevel.WARNING.value,
    predicted_fault: str | None = FaultType.COI_STUCK.value,
    ground_truth: str | None = FaultType.COI_STUCK.value,
    event_time: datetime | None = None,
) -> DetectionEvent:
    return DetectionEvent(
        event_time=event_time or datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        zone_id=zone_id,
        equipment_id=equipment_id,
        detector_source="rules",
        violated_policy="VALVE_STUCK",
        trigger_signal="chwc_valve_pct",
        anomaly_index=0.85,
        alert_level=alert_level,
        ground_truth=ground_truth,
        predicted_fault=predicted_fault,
        confidence=0.9,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def api_client(
    db_session: Session,
    test_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """TestClient with get_db_session overridden to use the transactional test session.

    Patches hvac_fdd.api.get_settings (the name bound by the `from ... import`)
    so the lifespan uses the SQLite test engine instead of the default PostgreSQL URL.
    """
    # Patch the name as it lives in the api module's global namespace.
    monkeypatch.setattr("hvac_fdd.api.get_settings", lambda: test_settings)

    app = create_app()

    def _session_override() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db_session] = _session_override
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_live_returns_ok(self, api_client: TestClient):
        resp = api_client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_ready_returns_ok(self, api_client: TestClient):
        resp = api_client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_request_id_header_present(self, api_client: TestClient):
        resp = api_client.get("/health/live")
        assert "x-request-id" in resp.headers

    def test_request_id_echoed_when_provided(self, api_client: TestClient):
        resp = api_client.get("/health/live", headers={"X-Request-ID": "test-123"})
        assert resp.headers["x-request-id"] == "test-123"


# ── Detections ────────────────────────────────────────────────────────────────


class TestDetectionsEndpoint:
    def _populate(self, db_session: Session) -> None:
        repo = DetectionRepository(db_session)
        repo.insert_bulk([
            _make_event(
                zone_id="ZONE-1",
                alert_level=AlertLevel.CRITICAL.value,
                predicted_fault=FaultType.COI_STUCK.value,
                event_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            ),
            _make_event(
                zone_id="ZONE-2",
                alert_level=AlertLevel.WARNING.value,
                predicted_fault=FaultType.DAMPER_STUCK.value,
                event_time=datetime(2024, 6, 1, 11, 0, tzinfo=timezone.utc),
            ),
            _make_event(
                zone_id="ZONE-1",
                alert_level=AlertLevel.INFO.value,
                predicted_fault=FaultType.OA_BIAS.value,
                event_time=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            ),
        ])

    def test_empty_db_returns_empty_list(self, api_client: TestClient):
        resp = api_client.get("/api/v1/detections/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_returns_all_detections(self, api_client: TestClient, db_session: Session):
        self._populate(db_session)
        resp = api_client.get("/api/v1/detections/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert len(body["items"]) == 3

    def test_response_schema_fields(self, api_client: TestClient, db_session: Session):
        DetectionRepository(db_session).insert_bulk([_make_event()])
        resp = api_client.get("/api/v1/detections/")
        item = resp.json()["items"][0]
        for field in ("id", "event_time", "zone_id", "equipment_id",
                      "alert_level", "anomaly_index"):
            assert field in item

    def test_filter_by_zone_id(self, api_client: TestClient, db_session: Session):
        self._populate(db_session)
        resp = api_client.get("/api/v1/detections/", params={"zone_id": "ZONE-1"})
        body = resp.json()
        assert body["total"] == 2
        assert all(item["zone_id"] == "ZONE-1" for item in body["items"])

    def test_filter_by_alert_level(self, api_client: TestClient, db_session: Session):
        self._populate(db_session)
        resp = api_client.get("/api/v1/detections/", params={"alert_level": "CRITICAL"})
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["alert_level"] == "CRITICAL"

    def test_filter_by_fault_type(self, api_client: TestClient, db_session: Session):
        self._populate(db_session)
        resp = api_client.get("/api/v1/detections/", params={"fault_type": "damper_stuck"})
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["predicted_fault"] == "damper_stuck"

    def test_filter_by_time_range(self, api_client: TestClient, db_session: Session):
        self._populate(db_session)
        resp = api_client.get("/api/v1/detections/", params={
            "start": "2024-06-01T10:30:00Z",
            "end": "2024-06-01T11:30:00Z",
        })
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["zone_id"] == "ZONE-2"

    def test_pagination_page_size(self, api_client: TestClient, db_session: Session):
        self._populate(db_session)
        resp = api_client.get("/api/v1/detections/", params={"page_size": 2, "skip": 0})
        body = resp.json()
        assert body["total"] == 3       # total reflects unfiltered count
        assert len(body["items"]) == 2
        assert body["skip"] == 0
        assert body["page_size"] == 2

    def test_pagination_skip(self, api_client: TestClient, db_session: Session):
        self._populate(db_session)
        resp = api_client.get("/api/v1/detections/", params={"page_size": 2, "skip": 2})
        body = resp.json()
        assert len(body["items"]) == 1

    def test_results_ordered_newest_first(self, api_client: TestClient, db_session: Session):
        self._populate(db_session)
        resp = api_client.get("/api/v1/detections/")
        times = [item["event_time"] for item in resp.json()["items"]]
        assert times == sorted(times, reverse=True)

    def test_invalid_page_size_returns_422(self, api_client: TestClient):
        resp = api_client.get("/api/v1/detections/", params={"page_size": 0})
        assert resp.status_code == 422

    def test_page_size_above_max_returns_422(self, api_client: TestClient):
        resp = api_client.get("/api/v1/detections/", params={"page_size": 1001})
        assert resp.status_code == 422


# ── Stats ─────────────────────────────────────────────────────────────────────


class TestStatsEndpoint:
    def test_stats_empty_db(self, api_client: TestClient):
        resp = api_client.get("/stats/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["by_alert_level"] == {}
        assert body["by_fault_type"] == {}

    def test_stats_total_count(self, api_client: TestClient, db_session: Session):
        DetectionRepository(db_session).insert_bulk(
            [_make_event() for _ in range(4)]
        )
        body = api_client.get("/stats/").json()
        assert body["total"] == 4

    def test_stats_by_alert_level(self, api_client: TestClient, db_session: Session):
        DetectionRepository(db_session).insert_bulk([
            _make_event(alert_level=AlertLevel.CRITICAL.value),
            _make_event(alert_level=AlertLevel.CRITICAL.value),
            _make_event(alert_level=AlertLevel.WARNING.value),
        ])
        body = api_client.get("/stats/").json()
        assert body["by_alert_level"]["CRITICAL"] == 2
        assert body["by_alert_level"]["WARNING"] == 1

    def test_stats_by_fault_type(self, api_client: TestClient, db_session: Session):
        DetectionRepository(db_session).insert_bulk([
            _make_event(predicted_fault=FaultType.COI_STUCK.value),
            _make_event(predicted_fault=None),
        ])
        body = api_client.get("/stats/").json()
        assert body["by_fault_type"]["coi_stuck"] == 1
        assert body["by_fault_type"]["unknown"] == 1

    def test_by_fault_type_endpoint(self, api_client: TestClient, db_session: Session):
        DetectionRepository(db_session).insert_bulk([
            _make_event(predicted_fault=FaultType.OA_BIAS.value),
            _make_event(predicted_fault=FaultType.OA_BIAS.value),
        ])
        resp = api_client.get("/stats/by-fault-type")
        assert resp.status_code == 200
        body = resp.json()
        assert "by_fault_type" in body
        assert body["by_fault_type"]["oa_bias"] == 2

    def test_stats_response_has_all_fields(self, api_client: TestClient):
        body = api_client.get("/stats/").json()
        assert set(body.keys()) >= {"total", "by_alert_level", "by_fault_type", "by_equipment"}


# ── Pipeline ──────────────────────────────────────────────────────────────────


class TestPipelineEndpoint:
    def test_trigger_returns_202(self, api_client: TestClient, monkeypatch):
        import hvac_fdd.api.pipeline as pm
        monkeypatch.setattr(pm, "_run_pipeline_task", lambda *_: None)
        resp = api_client.post("/pipeline/run", json={"scenario": "all"})
        assert resp.status_code == 202

    def test_trigger_returns_job_schema(self, api_client: TestClient, monkeypatch):
        import hvac_fdd.api.pipeline as pm
        monkeypatch.setattr(pm, "_run_pipeline_task", lambda *_: None)
        body = api_client.post("/pipeline/run", json={"scenario": "all"}).json()
        assert body["status"] == "pending"
        assert body["scenario"] == "all"
        assert body["id"] is not None
        assert body["finished_at"] is None

    def test_trigger_default_scenario(self, api_client: TestClient, monkeypatch):
        import hvac_fdd.api.pipeline as pm
        monkeypatch.setattr(pm, "_run_pipeline_task", lambda *_: None)
        body = api_client.post("/pipeline/run", json={}).json()
        assert body["scenario"] == "all"

    def test_get_job_found(self, api_client: TestClient, db_session: Session, monkeypatch):
        import hvac_fdd.api.pipeline as pm
        monkeypatch.setattr(pm, "_run_pipeline_task", lambda *_: None)
        created = api_client.post("/pipeline/run", json={"scenario": "coi_stuck"}).json()
        job_id = created["id"]

        resp = api_client.get(f"/pipeline/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == job_id
        assert body["scenario"] == "coi_stuck"

    def test_get_job_not_found_returns_404(self, api_client: TestClient):
        resp = api_client.get("/pipeline/jobs/99999")
        assert resp.status_code == 404

    def test_background_task_updates_status_on_failure(
        self, api_client: TestClient, db_session: Session, monkeypatch
    ):
        # Simulate a pipeline failure without loading real data files.
        # (Calling run_ingestion_pipeline for real would load the full LBNL
        # dataset via hvac_fdd.config.get_settings, which uses the real data
        # directory and can exhaust memory.)
        import hvac_fdd.api.pipeline as pm

        def _fail_task(job_id: int, scenario: str, job_repo: JobRepository) -> None:
            job_repo.update_status(job_id, "failed", error_msg="simulated failure")

        monkeypatch.setattr(pm, "_run_pipeline_task", _fail_task)

        resp = api_client.post("/pipeline/run", json={"scenario": "all"})
        assert resp.status_code == 202
        job_id = resp.json()["id"]

        job = JobRepository(db_session).get_by_id(job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.error_msg == "simulated failure"
        assert job.finished_at is not None
