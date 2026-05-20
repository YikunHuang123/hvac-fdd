from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

FAULT_LABELS: dict[str, str] = {
    "coi_bias": "Coil Sensor Bias",
    "coi_leakage": "Coil Leakage",
    "coi_stuck": "Valve Stuck",
    "damper_stuck": "Damper Stuck",
    "oa_bias": "OA Sensor Bias",
    "normal": "Normal",
    "unknown": "Unknown",
}

ALERT_PALETTE: dict[str, str] = {
    "CRITICAL": "#EF4444",
    "WARNING": "#F59E0B",
    "INFO": "#3B82F6",
}

FAULT_PALETTE: dict[str, str] = {
    "coi_bias": "#A78BFA",
    "coi_leakage": "#60A5FA",
    "coi_stuck": "#F87171",
    "damper_stuck": "#FBBF24",
    "oa_bias": "#34D399",
    "normal": "#10B981",
    "unknown": "#6B7280",
}

SCENARIOS = ["all", "coi_bias", "coi_leakage", "coi_stuck", "damper_stuck", "oa_bias"]


class HVACAPIClient:
    """Thin synchronous wrapper around the HVAC FDD REST API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 5) -> None:
        self.base_url = base_url.rstrip("/")
        self._s = requests.Session()
        self._timeout = timeout

    def _get(self, path: str, params: dict | None = None) -> Any:
        r = self._s.get(f"{self.base_url}{path}", params=params, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: dict | None = None) -> Any:
        r = self._s.post(f"{self.base_url}{path}", json=json, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def is_live(self) -> bool:
        try:
            self._get("/health/live")
            return True
        except Exception:
            return False

    def get_stats(self) -> dict:
        return self._get("/stats/")

    def get_detections(
        self,
        start: str | None = None,
        end: str | None = None,
        zone_id: str | None = None,
        alert_level: str | None = None,
        fault_type: str | None = None,
        page_size: int = 500,
        skip: int = 0,
    ) -> dict:
        params: dict[str, Any] = {"page_size": page_size, "skip": skip}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if zone_id:
            params["zone_id"] = zone_id
        if alert_level:
            params["alert_level"] = alert_level
        if fault_type:
            params["fault_type"] = fault_type
        return self._get("/api/v1/detections/", params)

    def trigger_pipeline(self, scenario: str = "all") -> dict:
        return self._post("/pipeline/run", {"scenario": scenario})

    def get_job(self, job_id: int) -> dict:
        return self._get(f"/pipeline/jobs/{job_id}")
