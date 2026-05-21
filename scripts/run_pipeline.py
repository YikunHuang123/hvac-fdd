#!/usr/bin/env python3
"""
HVAC FDD Pipeline Integration Script (Stable Edition).

🚀 Usage Guide:
--------------------------------------------------
A. Training Mode:
   1. Train Isolation Forest only (using 100% Normal Jan-Sept data):
      $ python scripts/run_pipeline.py --train-if
   2. Train Classifier only (using 20% sampled Jan-Sept data):
      $ python scripts/run_pipeline.py --train-clf
   3. Train both models:
      $ python scripts/run_pipeline.py --train-if --train-clf

B. Detection Mode (Flexible combinations):
   1. Use Physics Rules only (No model file required):
      $ python scripts/run_pipeline.py --use-rules --evaluate
   2. Use Isolation Forest only (Requires pre-trained IF):
      $ python scripts/run_pipeline.py --use-if --evaluate
   3. Rules Detection + Classifier Annotation (Skip IF):
      $ python scripts/run_pipeline.py --use-rules --use-clf --evaluate
   4. Full Engine + Persistence:
      $ python scripts/run_pipeline.py --use-rules --use-if --use-clf --persist --evaluate

C. Logic Dependencies:
   - --persist and --evaluate must be paired with at least one --use-xxx flag.
   - --use-if depends on models/isolation_forest.joblib.
   - --use-clf depends on models/classifier.joblib.
--------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from hvac_fdd.config import get_settings
from hvac_fdd.db.base import make_engine, make_session_factory
from hvac_fdd.db.detections import DetectionRepository
from hvac_fdd.detection.classifier import FaultClassifier
from hvac_fdd.detection.isolation_forest import IsolationForestDetector
from hvac_fdd.detection.rules import LBNLRulesDetector
from hvac_fdd.domain import AlertLevel, DetectionEvent, FaultType
from hvac_fdd.evaluation.metrics import (
    classification_report_extended,
    detection_report,
)
from hvac_fdd.ingestion.pipeline import iter_ingestion_pipeline

# Configure English logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_pipeline")


# ── Training helpers ──────────────────────────────────────────────────────────
# Encapsulated in functions so the concatenated DataFrame is released from memory
# as soon as training completes (scope-based GC rather than explicit del).

def _train_isolation_forest(frames: list[pd.DataFrame], settings) -> None:
    df = pd.concat(frames, ignore_index=True)
    IsolationForestDetector(settings).fit(df).save(settings.models_dir / "isolation_forest.joblib")


def _train_classifier(frames: list[pd.DataFrame], settings) -> None:
    df = pd.concat(frames, ignore_index=True)
    FaultClassifier(settings).fit(df).save(settings.models_dir / "classifier.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="HVAC FDD Pipeline Runner")

    # -- Training Flags --
    parser.add_argument("--train-if", action="store_true", help="Train Isolation Forest independently (100% Normal data)")
    parser.add_argument("--train-clf", action="store_true", help="Train Classifier independently (20% sampled data)")

    # -- Detection Components --
    parser.add_argument("--use-rules", action="store_true", help="Enable physics-based rules detector")
    parser.add_argument("--use-if", action="store_true", help="Enable unsupervised Isolation Forest detector")
    parser.add_argument("--use-clf", action="store_true", help="Enable supervised fault classifier for annotation")

    # -- Action Flags --
    parser.add_argument("--persist", action="store_true", help="Persist detected events to PostgreSQL")
    parser.add_argument("--evaluate", action="store_true", help="Run P/R/F1 evaluation on Oct-Dec hold-out data")
    parser.add_argument("--models-dir", type=str, help="Override models directory path")

    args = parser.parse_args()

    uses_detector = args.use_rules or args.use_if or args.use_clf
    if args.persist and not uses_detector:
        parser.error("--persist requires at least one --use-rules/--use-if/--use-clf flag")
    if args.evaluate and not uses_detector:
        logger.warning("--evaluate without any --use-xxx flag will report zero detections")

    settings = get_settings()
    if args.models_dir:
        settings.models_dir = Path(args.models_dir)
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Ingestion & Temporal Splitting ──────────────────────────────
    logger.info("Phase 1: Ingestion & Temporal Splitting...")

    if_train_frames:  list[pd.DataFrame] = []
    clf_train_frames: list[pd.DataFrame] = []
    eval_frames:      list[pd.DataFrame] = []

    for i, chunk_df in enumerate(iter_ingestion_pipeline(settings)):
        chunk_df["event_time"] = pd.to_datetime(chunk_df["event_time"])

        # Anchor split on the minimum year found in each chunk so that the
        # Jan-Sept / Oct-Dec boundary stays correct even if data spans years.
        base_year = int(chunk_df["event_time"].dt.year.min())
        train_cutoff = pd.Timestamp(f"{base_year}-09-30 23:59:59")
        train_chunk = chunk_df[chunk_df["event_time"] <= train_cutoff]
        eval_chunk  = chunk_df[chunk_df["event_time"] > train_cutoff]

        if args.train_if and not train_chunk.empty:
            if_train_frames.append(train_chunk[train_chunk["fault_type"] == FaultType.NORMAL.value])

        if args.train_clf and not train_chunk.empty:
            # Vary random_state per chunk to avoid picking the same relative
            # row positions across structurally similar scenario files.
            clf_train_frames.append(train_chunk.sample(frac=0.2, random_state=42 + i))

        if not eval_chunk.empty and (uses_detector or args.evaluate):
            eval_frames.append(eval_chunk)

    # ── Phase 2: Differential Training ───────────────────────────────────────
    if args.train_if and if_train_frames:
        logger.info("Phase 2a: Training Isolation Forest on 100%% Normal data...")
        _train_isolation_forest(if_train_frames, settings)

    if args.train_clf and clf_train_frames:
        logger.info("Phase 2b: Training Classifier on 20%% Sampled data...")
        _train_classifier(clf_train_frames, settings)

    # ── Phase 3: Per-scenario Detection ──────────────────────────────────────
    # eval_frames is fully in memory from Phase 1. Iterating one frame at a time
    # limits the peak allocation per detection pass to a single scenario's data.
    all_events: list[DetectionEvent] = []
    if eval_frames and uses_detector:
        logger.info("Phase 3: Running per-scenario detection...")

        rules_det = LBNLRulesDetector(settings) if args.use_rules else None
        if_det = (
            IsolationForestDetector.load(settings.models_dir / "isolation_forest.joblib", settings=settings)
            if args.use_if else None
        )
        clf_det = (
            FaultClassifier.load(settings.models_dir / "classifier.joblib", settings=settings)
            if args.use_clf else None
        )

        for eval_chunk in eval_frames:
            hits = []
            if rules_det:
                hits.append(rules_det.predict(eval_chunk))
            if if_det:
                hits.append(if_det.predict(eval_chunk))

            if hits:
                combined = pd.concat(hits, ignore_index=True)
                if not combined.empty:
                    # Aggregate all detector sources before deduplication so the
                    # information about which detectors agreed is not lost.
                    source_agg = (
                        combined.groupby(["event_time", "zone_id"])["detector_source"]
                        .apply(lambda x: "|".join(sorted(set(x))))
                        .reset_index(name="all_sources")
                    )
                    combined = (
                        combined.sort_values("anomaly_index", ascending=False)
                        .drop_duplicates(subset=["event_time", "zone_id"], keep="first")
                        .merge(source_agg, on=["event_time", "zone_id"])
                    )
                    multi = combined["all_sources"].str.contains("|", regex=False).sum()
                    if multi:
                        logger.debug("  %d/%d events agreed by multiple detectors", multi, len(combined))

                    if clf_det:
                        keys = combined[["event_time", "zone_id"]]
                        to_classify = eval_chunk.merge(keys, on=["event_time", "zone_id"])
                        clf_results = clf_det.predict(to_classify)
                        if not clf_results.empty:
                            combined = combined.merge(
                                clf_results[["event_time", "zone_id", "predicted_fault", "confidence"]],
                                on=["event_time", "zone_id"], how="left",
                            )

                    all_events.extend(_to_domain_events(combined, settings.default_equipment_id))

        logger.info("Detection complete: %d events found.", len(all_events))

    # ── Phase 4: Persistence ─────────────────────────────────────────────────
    if args.persist and all_events:
        logger.info("Phase 4: Persisting %d events...", len(all_events))
        engine = make_engine(settings.database_url)
        session_factory = make_session_factory(engine)
        with session_factory() as session:
            DetectionRepository(session).insert_bulk(all_events)
            session.commit()

    # ── Phase 5: Vectorized Evaluation ───────────────────────────────────────
    if args.evaluate and eval_frames:
        logger.info("Phase 5: Starting memory-optimized evaluation...")
        mini_frames = [df[["event_time", "zone_id", "fault_type"]] for df in eval_frames]
        eval_df_full = pd.concat(mini_frames, ignore_index=True)
        _run_evaluation_vectorized(eval_df_full, all_events, used_classifier=args.use_clf)


def _to_domain_events(df: pd.DataFrame, equip_id: str) -> list[DetectionEvent]:
    events = []
    for record in df.to_dict("records"):
        try:
            events.append(DetectionEvent(
                event_time=record["event_time"],
                zone_id=str(record["zone_id"]),
                equipment_id=equip_id,
                detector_source=str(record["detector_source"]),
                violated_policy=str(record["violated_policy"]),
                trigger_signal=str(record["trigger_signal"]),
                anomaly_index=float(record["anomaly_index"]),
                alert_level=AlertLevel(record["alert_level"]),
                ground_truth=FaultType(record["ground_truth"]) if pd.notna(record.get("ground_truth")) else None,
                predicted_fault=FaultType(record.get("predicted_fault")) if pd.notna(record.get("predicted_fault")) else None,
                confidence=float(record.get("confidence")) if pd.notna(record.get("confidence")) else None,
            ))
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping malformed event row: %s — %s", record, exc)
    return events


def _run_evaluation_vectorized(df: pd.DataFrame, events: list, *, used_classifier: bool) -> None:
    """Vectorized evaluation to prevent Pandas .apply() memory bombs."""
    if not events:
        df["detected"] = False
        df["predicted_fault"] = FaultType.NORMAL.value
    else:
        events_df = pd.DataFrame([
            {
                "event_time": e.event_time,
                "zone_id": e.zone_id,
                "predicted_fault": e.predicted_fault if e.predicted_fault else FaultType.NORMAL.value,
            }
            for e in events
        ]).drop_duplicates(subset=["event_time", "zone_id"], keep="first")
        events_df["detected"] = True

        df = df.merge(events_df, on=["event_time", "zone_id"], how="left")
        df["detected"] = df["detected"].fillna(False)
        df["predicted_fault"] = df["predicted_fault"].fillna(FaultType.NORMAL.value)

    rep = detection_report(df)
    logger.info("--- Detection Report (Hold-out: Oct-Dec) ---")
    logger.info("Overall: %s", rep["overall"])

    if used_classifier:
        crep = classification_report_extended(df["fault_type"], df["predicted_fault"])
        logger.info("--- Classification Report (Hold-out: Oct-Dec) ---")
        logger.info("Accuracy: %.4f | Macro F1: %.4f", crep["accuracy"], crep["macro_f1"])


if __name__ == "__main__":
    main()
