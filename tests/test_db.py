"""
Integration tests for the db layer repositories.

Uses the in-memory SQLite engine/session provided by conftest.py.
Each test gets a fresh transactional session that is rolled back on teardown.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hvac_fdd.db.detections import DetectionFilter, DetectionRepository
from hvac_fdd.db.jobs import JobRepository
from hvac_fdd.domain import AlertLevel, DetectionEvent, FaultType
from hvac_fdd.exceptions import DatabaseError


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_event(
    *,
    zone_id: str = "ZONE-1",
    alert_level: str = AlertLevel.WARNING.value,
    predicted_fault: str | None = FaultType.COI_STUCK.value,
    ground_truth: str | None = FaultType.COI_STUCK.value,
    event_time: datetime | None = None,
) -> DetectionEvent:
    return DetectionEvent(
        event_time=event_time or datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        zone_id=zone_id,
        equipment_id="AHU-1",
        detector_source="rules",
        violated_policy="VALVE_STUCK",
        trigger_signal="chwc_valve_pct",
        anomaly_index=0.85,
        alert_level=alert_level,
        ground_truth=ground_truth,
        predicted_fault=predicted_fault,
        confidence=0.9,
    )


# ── DetectionRepository ────────────────────────────────────────────────────────


class TestDetectionRepositoryInsert:
    def test_insert_bulk_returns_count(self, db_session):
        repo = DetectionRepository(db_session)
        events = [_make_event(zone_id=f"ZONE-{i}") for i in range(3)]
        n = repo.insert_bulk(events)
        assert n == 3

    def test_insert_bulk_empty_returns_zero(self, db_session):
        repo = DetectionRepository(db_session)
        assert repo.insert_bulk([]) == 0

    def test_inserted_rows_queryable(self, db_session):
        repo = DetectionRepository(db_session)
        repo.insert_bulk([_make_event(zone_id="ZONE-X")])
        rows = repo.query(DetectionFilter(zone_id="ZONE-X"), page_size=10, skip=0)
        assert len(rows) == 1
        assert rows[0].zone_id == "ZONE-X"

    def test_enum_values_stored_as_strings(self, db_session):
        repo = DetectionRepository(db_session)
        repo.insert_bulk([_make_event()])
        rows = repo.query(DetectionFilter(), page_size=10, skip=0)
        assert rows[0].alert_level == "WARNING"
        assert rows[0].predicted_fault == "coi_stuck"
        assert rows[0].ground_truth == "coi_stuck"

    def test_none_fields_stored_as_null(self, db_session):
        repo = DetectionRepository(db_session)
        repo.insert_bulk([_make_event(predicted_fault=None, ground_truth=None)])
        rows = repo.query(DetectionFilter(), page_size=10, skip=0)
        assert rows[0].predicted_fault is None
        assert rows[0].ground_truth is None


class TestDetectionRepositoryQuery:
    def _populate(self, db_session):
        repo = DetectionRepository(db_session)
        events = [
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
        ]
        repo.insert_bulk(events)
        return repo

    def test_query_no_filter_returns_all(self, db_session):
        repo = self._populate(db_session)
        rows = repo.query(DetectionFilter(), page_size=100, skip=0)
        assert len(rows) == 3

    def test_query_filter_by_zone(self, db_session):
        repo = self._populate(db_session)
        rows = repo.query(DetectionFilter(zone_id="ZONE-1"), page_size=100, skip=0)
        assert len(rows) == 2
        assert all(r.zone_id == "ZONE-1" for r in rows)

    def test_query_filter_by_alert_level(self, db_session):
        repo = self._populate(db_session)
        rows = repo.query(DetectionFilter(alert_level="CRITICAL"), page_size=100, skip=0)
        assert len(rows) == 1
        assert rows[0].alert_level == "CRITICAL"

    def test_query_filter_by_fault_type(self, db_session):
        repo = self._populate(db_session)
        rows = repo.query(DetectionFilter(fault_type="damper_stuck"), page_size=100, skip=0)
        assert len(rows) == 1
        assert rows[0].predicted_fault == "damper_stuck"

    def test_query_filter_by_time_range(self, db_session):
        repo = self._populate(db_session)
        start = datetime(2024, 6, 1, 10, 30, tzinfo=timezone.utc)
        end   = datetime(2024, 6, 1, 11, 30, tzinfo=timezone.utc)
        rows = repo.query(DetectionFilter(start=start, end=end), page_size=100, skip=0)
        assert len(rows) == 1
        assert rows[0].zone_id == "ZONE-2"

    def test_query_pagination(self, db_session):
        repo = self._populate(db_session)
        page1 = repo.query(DetectionFilter(), page_size=2, skip=0)
        page2 = repo.query(DetectionFilter(), page_size=2, skip=2)
        assert len(page1) == 2
        assert len(page2) == 1

    def test_query_ordered_newest_first(self, db_session):
        repo = self._populate(db_session)
        rows = repo.query(DetectionFilter(), page_size=10, skip=0)
        times = [r.event_time for r in rows]
        assert times == sorted(times, reverse=True)

    def test_count_matches_query(self, db_session):
        repo = self._populate(db_session)
        f = DetectionFilter(zone_id="ZONE-1")
        assert repo.count(f) == len(repo.query(f, page_size=100, skip=0))

    def test_count_no_filter(self, db_session):
        repo = self._populate(db_session)
        assert repo.count(DetectionFilter()) == 3


class TestDetectionRepositoryStatistics:
    def test_statistics_total(self, db_session):
        repo = DetectionRepository(db_session)
        repo.insert_bulk([_make_event() for _ in range(4)])
        stats = repo.statistics()
        assert stats["total"] == 4

    def test_statistics_by_alert_level(self, db_session):
        repo = DetectionRepository(db_session)
        repo.insert_bulk([
            _make_event(alert_level=AlertLevel.CRITICAL.value),
            _make_event(alert_level=AlertLevel.CRITICAL.value),
            _make_event(alert_level=AlertLevel.WARNING.value),
        ])
        stats = repo.statistics()
        assert stats["by_alert_level"]["CRITICAL"] == 2
        assert stats["by_alert_level"]["WARNING"] == 1

    def test_statistics_by_fault_type(self, db_session):
        repo = DetectionRepository(db_session)
        repo.insert_bulk([
            _make_event(predicted_fault=FaultType.COI_STUCK.value),
            _make_event(predicted_fault=FaultType.COI_STUCK.value),
            _make_event(predicted_fault=None),
        ])
        stats = repo.statistics()
        assert stats["by_fault_type"]["coi_stuck"] == 2
        assert stats["by_fault_type"]["unknown"] == 1

    def test_statistics_empty_db(self, db_session):
        repo = DetectionRepository(db_session)
        stats = repo.statistics()
        assert stats["total"] == 0
        assert stats["by_alert_level"] == {}


# ── JobRepository ─────────────────────────────────────────────────────────────


class TestJobRepository:
    def test_create_returns_pending_job(self, db_session):
        repo = JobRepository(db_session)
        job = repo.create("all")
        assert job.id is not None
        assert job.scenario == "all"
        assert job.status == "pending"
        assert job.finished_at is None

    def test_get_by_id_found(self, db_session):
        repo = JobRepository(db_session)
        created = repo.create("coi_stuck")
        fetched = repo.get_by_id(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.scenario == "coi_stuck"

    def test_get_by_id_not_found(self, db_session):
        repo = JobRepository(db_session)
        assert repo.get_by_id(99999) is None

    def test_update_status_running(self, db_session):
        repo = JobRepository(db_session)
        job = repo.create("all")
        updated = repo.update_status(job.id, "running")
        assert updated.status == "running"
        assert updated.finished_at is None

    def test_update_status_done_sets_finished_at(self, db_session):
        repo = JobRepository(db_session)
        job = repo.create("all")
        updated = repo.update_status(
            job.id, "done", records_processed=1000, anomalies_found=42
        )
        assert updated.status == "done"
        assert updated.finished_at is not None
        assert updated.records_processed == 1000
        assert updated.anomalies_found == 42

    def test_update_status_failed_sets_error_msg(self, db_session):
        repo = JobRepository(db_session)
        job = repo.create("all")
        updated = repo.update_status(job.id, "failed", error_msg="disk full")
        assert updated.status == "failed"
        assert updated.error_msg == "disk full"
        assert updated.finished_at is not None

    def test_update_status_nonexistent_raises(self, db_session):
        repo = JobRepository(db_session)
        with pytest.raises(DatabaseError, match="not found"):
            repo.update_status(99999, "done")

    def test_list_recent_returns_newest_first(self, db_session):
        repo = JobRepository(db_session)
        for scenario in ["a", "b", "c"]:
            repo.create(scenario)
        jobs = repo.list_recent(limit=10)
        assert len(jobs) == 3
        # newest (last created) has the highest id
        ids = [j.id for j in jobs]
        assert ids == sorted(ids, reverse=True)

    def test_list_recent_respects_limit(self, db_session):
        repo = JobRepository(db_session)
        for i in range(5):
            repo.create(f"scenario-{i}")
        jobs = repo.list_recent(limit=3)
        assert len(jobs) == 3
