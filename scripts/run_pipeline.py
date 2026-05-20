#!/usr/bin/env python3
"""
HVAC FDD Pipeline Integration Script (Optimized & Granular).

🚀 Usage Guide:
--------------------------------------------------
A. Training Mode:
   1. Train Isolation Forest only:
      $ python scripts/run_pipeline.py --train-if
   2. Train Classifier only:
      $ python scripts/run_pipeline.py --train-clf

B. Detection Mode:
   1. Use Physics Rules only (Fastest):
      $ python scripts/run_pipeline.py --use-rules --evaluate
   2. Full Engine (Rules + IF + Classifier):
      $ python scripts/run_pipeline.py --use-rules --use-if --use-clf --evaluate
--------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from hvac_fdd.config import get_settings
from hvac_fdd.db.base import make_engine, make_session_factory
from hvac_fdd.db.detections import DetectionRepository
from hvac_fdd.detection.classifier import FaultClassifier
from hvac_fdd.detection.isolation_forest import IsolationForestDetector
from hvac_fdd.detection.rules import LBNLRulesDetector
from hvac_fdd.domain import FaultType
from hvac_fdd.evaluation.metrics import (
    classification_report_extended,
    detection_report,
)
from hvac_fdd.ingestion.pipeline import iter_ingestion_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_pipeline")

def main() -> None:
    parser = argparse.ArgumentParser(description="HVAC FDD Pipeline Runner")
    parser.add_argument("--train-if", action="store_true")
    parser.add_argument("--train-clf", action="store_true")
    parser.add_argument("--use-rules", action="store_true")
    parser.add_argument("--use-if", action="store_true")
    parser.add_argument("--use-clf", action="store_true")
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--models-dir", type=str)

    args = parser.parse_args()
    settings = get_settings()
    if args.models_dir: settings.models_dir = Path(args.models_dir)
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    # -- Phase 1: Ingestion --
    logger.info("Phase 1: Ingestion & Temporal Splitting...")
    if_train_frames, clf_train_frames, eval_frames = [], [], []
    for i, chunk_df in enumerate(iter_ingestion_pipeline(settings)):
        chunk_df["event_time"] = pd.to_datetime(chunk_df["event_time"])
        month = chunk_df["event_time"].dt.month
        train_chunk = chunk_df[month <= 9]
        eval_chunk = chunk_df[month > 9]
        if args.train_if: if_train_frames.append(train_chunk[train_chunk["fault_type"] == FaultType.NORMAL.value].copy())
        if args.train_clf: clf_train_frames.append(train_chunk.sample(frac=0.2, random_state=42))
        if not eval_chunk.empty: eval_frames.append(eval_chunk.copy())

    # -- Phase 2: Training --
    if args.train_if and if_train_frames:
        logger.info("Phase 2a: Training IF...")
        IsolationForestDetector(settings).fit(pd.concat(if_train_frames)).save(settings.models_dir / "isolation_forest.joblib")
        if_train_frames = None
    if args.train_clf and clf_train_frames:
        logger.info("Phase 2b: Training Classifier...")
        FaultClassifier(settings).fit(pd.concat(clf_train_frames)).save(settings.models_dir / "classifier.joblib")
        clf_train_frames = None

    # -- Phase 3: Detection --
    event_frames = []
    if eval_frames and (args.use_rules or args.use_if or args.use_clf):
        logger.info("Phase 3: Running Surgical Detection (Memory Efficient)...")
        rules_det = LBNLRulesDetector(settings) if args.use_rules else None
        if_det = IsolationForestDetector.load(settings.models_dir / "isolation_forest.joblib", settings=settings) if args.use_if else None
        clf_det = FaultClassifier.load(settings.models_dir / "classifier.joblib", settings=settings) if args.use_clf else None

        for idx, full_chunk in enumerate(eval_frames):
            hits = []
            if rules_det: hits.append(rules_det.predict(full_chunk))
            if if_det: hits.append(if_det.predict(full_chunk))
            if hits:
                combined = pd.concat(hits, ignore_index=True).sort_values("anomaly_index", ascending=False)
                combined = combined.drop_duplicates(subset=["event_time", "zone_id"], keep="first")
                if clf_det:
                    keys = combined[["event_time", "zone_id"]]
                    clf_in = full_chunk.merge(keys, on=["event_time", "zone_id"])
                    clf_out = clf_det.predict(clf_in)
                    if not clf_out.empty:
                        combined = combined.merge(clf_out[["event_time", "zone_id", "predicted_fault", "confidence"]], on=["event_time", "zone_id"], how="left")
                event_frames.append(combined)
            eval_frames[idx] = full_chunk[["event_time", "zone_id", "fault_type"]].copy()

    # -- Phase 4: Persistence --
    if args.persist and event_frames:
        logger.info("Phase 4: Bulk Persisting...")
        _persist_from_df(pd.concat(event_frames, ignore_index=True), settings)

    # -- Phase 5: Evaluation --
    if args.evaluate and eval_frames:
        logger.info("Phase 5: Vectorized Evaluation on hold-out set (Oct-Dec)...")
        eval_df = pd.concat(eval_frames, ignore_index=True)
        results_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
        _run_evaluation_vectorized(eval_df, results_df)

def _persist_from_df(df: pd.DataFrame, settings):
    from hvac_fdd.domain import DetectionEvent, AlertLevel
    engine = make_engine(settings.database_url); session_factory = make_session_factory(engine)
    with session_factory() as session:
        repo = DetectionRepository(session)
        events = []
        for _, r in df.iterrows():
            events.append(DetectionEvent(
                event_time=r["event_time"], zone_id=str(r["zone_id"]), equipment_id="AHU-1",
                detector_source=str(r["detector_source"]), violated_policy=str(r["violated_policy"]),
                trigger_signal=str(r["trigger_signal"]), anomaly_index=float(r["anomaly_index"]),
                alert_level=AlertLevel(r["alert_level"]),
                ground_truth=FaultType(r["ground_truth"]) if pd.notna(r.get("ground_truth")) else None,
                predicted_fault=FaultType(r.get("predicted_fault")) if pd.notna(r.get("predicted_fault")) else None,
                confidence=float(r.get("confidence")) if pd.notna(r.get("confidence")) else None
            ))
        repo.insert_bulk(events)
        session.commit()

def _run_evaluation_vectorized(df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    if results_df.empty:
        df["detected"] = False
        df["predicted_fault"] = FaultType.NORMAL.value
    else:
        # Check if predicted_fault exists (it only exists if --use-clf was used)
        cols_to_extract = ["event_time", "zone_id"]
        if "predicted_fault" in results_df.columns:
            cols_to_extract.append("predicted_fault")
        
        res = results_df[cols_to_extract].copy()
        res["detected"] = True
        df = df.merge(res, on=["event_time", "zone_id"], how="left")
        df["detected"] = df["detected"].fillna(False)
        if "predicted_fault" in df.columns:
            df["predicted_fault"] = df["predicted_fault"].fillna(FaultType.NORMAL.value)

    rep = detection_report(df)
    logger.info("--- Detection Report (Binary: Oct-Dec Hold-out) ---")
    logger.info("Overall: %s", rep["overall"])
    
    if "predicted_fault" in df.columns:
        crep = classification_report_extended(df["fault_type"], df["predicted_fault"])
        logger.info("--- Classification Report (Multi-class) ---")
        logger.info("Accuracy: %.4f | Macro F1: %.4f", crep["accuracy"], crep["macro_f1"])

if __name__ == "__main__":
    main()
