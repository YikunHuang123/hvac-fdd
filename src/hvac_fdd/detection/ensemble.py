"""
Ensemble detector that fuses rules, Isolation Forest, and (optionally) a classifier.

Fusion strategy:
  1. Rules detector  — low latency, high explainability
  2. GMM — catches anomalies not covered by rules
  3. Classifier      — annotates each detected event with a predicted FaultType

Deduplication: per (event_time, zone_id), keep the row with the highest anomaly_index.
Annotation: classifier's predicted_fault overwrites any rule-inferred fault label.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from hvac_fdd.detection.classifier import FaultClassifier
from hvac_fdd.detection.base import DetectorBase

logger = logging.getLogger(__name__)

_DEDUP_KEYS = ["event_time", "zone_id"]


class EnsembleDetector:
    """
    Combine rules, an unsupervised anomaly detector (GMM, IF, KAN), and an optional FaultClassifier into one detector.

    Args:
        rules:        Fitted LBNLRulesDetector.
        unsupervised: Fitted unsupervised detector (e.g. GMMDetector, IsolationForestDetector, KANDetector).
        classifier:   Optional fitted FaultClassifier; skipped when None.
        equipment_id: Equipment identifier written to every DetectionEvent.
    """

    def __init__(
        self,
        rules: LBNLRulesDetector,
        unsupervised: DetectorBase,
        classifier: Optional[FaultClassifier] = None,
        *,
        equipment_id: str = "AHU-1",
    ) -> None:
        self._rules        = rules
        self._unsupervised = unsupervised
        self._classifier   = classifier
        self._equipment_id = equipment_id

    # ── Public interface ──────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> list[DetectionEvent]:
        """
        Run all sub-detectors on df and return deduplicated DetectionEvents.

        Args:
            df: Feature-engineered DataFrame (output of build_feature_set).

        Returns:
            List of DetectionEvent domain objects, one per unique (event_time, zone_id).
        """
        rules_hits = self._rules.predict(df)
        unsup_hits = self._unsupervised.predict(df)
        combined   = self._merge_and_dedup(rules_hits, unsup_hits)

        if combined.empty:
            return []

        if self._classifier is not None:
            combined = self._annotate_with_classifier(combined, df)

        if "fault_type" in df.columns:
            labels = df[["event_time", "zone_id", "fault_type"]].drop_duplicates(
                subset=["event_time", "zone_id"]
            )
            combined = combined.drop(columns=["ground_truth"], errors="ignore").merge(
                labels,
                on=["event_time", "zone_id"],
                how="left",
            ).rename(columns={"fault_type": "ground_truth"})

        events = [self._to_domain(row) for _, row in combined.iterrows()]
        logger.info("EnsembleDetector: %d events from %d input rows", len(events), len(df))
        return events

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _merge_and_dedup(
        rules_hits: pd.DataFrame,
        unsup_hits: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Concatenate rules and unsupervised hits; keep the highest anomaly_index row per
        (event_time, zone_id) pair. Initialise predicted_fault and confidence
        columns to None so downstream code can fill them.
        """
        frames = [f for f in (rules_hits, unsup_hits) if len(f) > 0]
        if not frames:
            combined = pd.DataFrame()
        else:
            combined = pd.concat(frames, ignore_index=True)
            combined = (
                combined
                .sort_values("anomaly_index", ascending=False)
                .drop_duplicates(subset=_DEDUP_KEYS, keep="first")
                .reset_index(drop=True)
            )

        if combined.empty:
            return combined

        # Ensure classifier annotation columns exist.
        combined["predicted_fault"] = None
        combined["confidence"]      = None
        return combined

    def _annotate_with_classifier(
        self,
        combined: pd.DataFrame,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the classifier on the original df rows that correspond to detected
        events; merge predicted_fault and confidence back into combined.
        """
        assert self._classifier is not None  # checked by caller

        # Filter original df to only the rows that have detected events.
        keys   = combined[_DEDUP_KEYS].drop_duplicates()
        subset = df.merge(keys, on=_DEDUP_KEYS, how="inner")
        if subset.empty:
            return combined

        clf_out = self._classifier.predict(subset)
        if clf_out.empty:
            return combined

        # Build lookup: (event_time, zone_id) → (predicted_fault, confidence).
        clf_out["_key"] = list(zip(clf_out["event_time"], clf_out["zone_id"]))
        lookup: dict = dict(
            zip(
                clf_out["_key"],
                zip(clf_out["predicted_fault"], clf_out["confidence"]),
            )
        )

        combined = combined.copy()
        combined["_key"] = list(zip(combined["event_time"], combined["zone_id"]))
        combined["predicted_fault"] = combined["_key"].map(
            lambda k: lookup.get(k, (None, None))[0]
        )
        combined["confidence"] = combined["_key"].map(
            lambda k: lookup.get(k, (None, None))[1]
        )
        combined.drop(columns=["_key"], inplace=True)
        return combined

    def _to_domain(self, row: pd.Series) -> DetectionEvent:
        """Convert a combined-DataFrame row into a DetectionEvent domain object."""

        def _to_fault_type(val: object) -> Optional[FaultType]:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            try:
                return FaultType(val)
            except ValueError:
                return None

        def _to_alert_level(val: object) -> AlertLevel:
            try:
                return AlertLevel(val)
            except (ValueError, TypeError):
                return AlertLevel.INFO

        return DetectionEvent(
            event_time      = row["event_time"],
            zone_id         = str(row["zone_id"]),
            equipment_id    = self._equipment_id,
            detector_source = str(row["detector_source"]),
            violated_policy = str(row["violated_policy"]),
            trigger_signal  = str(row["trigger_signal"]),
            anomaly_index   = float(row["anomaly_index"]),
            alert_level     = _to_alert_level(row["alert_level"]),
            ground_truth    = _to_fault_type(row.get("ground_truth")),
            predicted_fault = _to_fault_type(row.get("predicted_fault")),
            confidence      = float(row["confidence"]) if row.get("confidence") is not None else None,
        )
