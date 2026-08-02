#!/usr/bin/env python3
"""
HVAC FDD Pipeline Integration Script (Stable Edition).

Usage Guide:
--------------------------------------------------
A. Training Mode:
   1. Train Unsupervised Model (using 100% Normal training-window data):
      $ python scripts/run_pipeline.py --train-unsup
   2. Train Classifier only (using 20% sampled Jan-Sep data):
      $ python scripts/run_pipeline.py --train-clf
   3. Train both models:
      $ python scripts/run_pipeline.py --train-unsup --train-clf

B. Detection Mode (Flexible combinations):
   1. Use Physics Rules only (No model file required):
      $ python scripts/run_pipeline.py --use-rules --evaluate
   2. Use Unsupervised Model only (Requires pre-trained model):
      $ python scripts/run_pipeline.py --use-unsup --evaluate
   3. Rules Detection + Classifier Annotation (Skip Unsupervised):
      $ python scripts/run_pipeline.py --use-rules --use-clf --evaluate
   4. Full Engine + Persistence:
      $ python scripts/run_pipeline.py --use-rules --use-unsup --use-clf --persist --evaluate

C. Logic Dependencies:
   - --persist and --evaluate must be paired with at least one --use-xxx flag.
   - --use-unsup depends on models/{model}_detector.joblib.
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
from hvac_fdd.detection.rules import LBNLRulesDetector
from hvac_fdd.domain import AlertLevel, DetectionEvent, FaultType
from hvac_fdd.evaluation.metrics import (
    classification_report_extended,
    detection_report,
)
from hvac_fdd.ingestion.parquet import iter_parquet_pipeline
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

def _train_unsup(
    frames: list[pd.DataFrame],
    settings,
    *,
    calibration_frames: list[pd.DataFrame] | None = None,
    target_fpr: float | None = None,
) -> None:
    df = pd.concat(frames, ignore_index=True)
    if settings.unsupervised_model == "gmm":
        from hvac_fdd.detection.gmm_detector import GMMDetector
        detector = GMMDetector(settings).fit(df)
        if target_fpr is not None:
            if not calibration_frames:
                raise ValueError("GMM threshold calibration requires validation frames")
            calibration_df = pd.concat(calibration_frames, ignore_index=True)
            detector.calibrate_warning_threshold(calibration_df, target_fpr)
        detector.save(settings.models_dir / "gmm_detector.joblib")
    elif settings.unsupervised_model == "if":
        from hvac_fdd.detection.isolation_forest import IsolationForestDetector
        IsolationForestDetector(settings).fit(df).save(settings.models_dir / "if_detector.joblib")
    elif settings.unsupervised_model == "kan":
        from hvac_fdd.detection.kan_detector import KANDetector
        KANDetector(settings).fit(df).save(settings.models_dir / "kan_detector.joblib")
    else:
        raise ValueError(f"Unknown unsupervised model: {settings.unsupervised_model}")


def _train_classifier(
    frames: list[pd.DataFrame],
    settings,
    *,
    calibration_frames: list[pd.DataFrame] | None = None,
    target_fpr: float | None = None,
) -> None:
    df = pd.concat(frames, ignore_index=True)
    classifier = FaultClassifier(settings).fit(df)
    if target_fpr is not None:
        if not calibration_frames:
            raise ValueError("Classifier threshold calibration requires validation frames")
        classifier.calibrate_anomaly_threshold(
            pd.concat(calibration_frames, ignore_index=True), target_fpr
        )
    classifier.save(settings.models_dir / "classifier.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="HVAC FDD Pipeline Runner")

    # -- Training Flags --
    parser.add_argument("--train-unsup", action="store_true", help="Train unsupervised model independently on normal data")
    parser.add_argument("--train-clf", action="store_true", help="Train classifier independently on sampled data")

    # -- Detection Components --
    parser.add_argument("--use-rules", action="store_true", help="Enable physics-based rules detector")
    parser.add_argument("--use-unsup", action="store_true", help="Enable unsupervised detector")
    parser.add_argument("--use-clf", action="store_true", help="Enable supervised fault classifier for annotation")

    # -- Action Flags --
    parser.add_argument("--persist", action="store_true", help="Persist detected events to PostgreSQL")
    parser.add_argument("--evaluate", action="store_true", help="Run P/R/F1 evaluation on the selected evaluation window")
    parser.add_argument(
        "--evaluation-window",
        choices=("validation", "final"),
        default="final",
        help="Evaluate the validation window or the untouched final hold-out",
    )
    parser.add_argument(
        "--split-protocol",
        choices=("annual", "common"),
        default="annual",
        help=(
            "annual: train Jan-Sep, validate Oct-Nov, test Dec; "
            "common: train Apr-Aug, validate Sep, test Oct so every scenario is present"
        ),
    )
    parser.add_argument(
        "--holdout-scenario",
        type=str,
        help=(
            "Optional CSV filename to exclude from training and evaluate alone "
            "in the selected window"
        ),
    )
    parser.add_argument(
        "--include-normal-reference",
        action="store_true",
        help="When holding out a fault scenario, also evaluate AHU_annual as a normal reference",
    )
    parser.add_argument("--models-dir", type=str, help="Override models directory path")
    parser.add_argument(
        "--input-format",
        choices=("csv", "parquet"),
        default="csv",
        help="Input path: stream and engineer CSV files, or read projected Parquet caches",
    )
    parser.add_argument(
        "--gmm-target-fpr",
        type=float,
        help=(
            "Calibrate the GMM warning threshold on the validation window "
            "to target this normal-operation row-level FPR (requires --train-unsup)"
        ),
    )
    parser.add_argument(
        "--classifier-target-fpr",
        type=float,
        help="Calibrate classifier anomaly threshold on validation normal rows",
    )

    args = parser.parse_args()

    uses_detector = args.use_rules or args.use_unsup or args.use_clf
    if args.persist and not uses_detector:
        parser.error("--persist requires at least one --use-rules/--use-unsup/--use-clf flag")
    if args.evaluate and not uses_detector:
        logger.warning("--evaluate without any --use-xxx flag will report zero detections")
    if args.gmm_target_fpr is not None:
        if not args.train_unsup:
            parser.error("--gmm-target-fpr requires --train-unsup")
        if not 0.0 < args.gmm_target_fpr < 1.0:
            parser.error("--gmm-target-fpr must be between 0 and 1")
    if args.classifier_target_fpr is not None:
        if not args.train_clf:
            parser.error("--classifier-target-fpr requires --train-clf")
        if not 0.0 < args.classifier_target_fpr < 1.0:
            parser.error("--classifier-target-fpr must be between 0 and 1")

    settings = get_settings()
    if args.models_dir:
        settings.models_dir = Path(args.models_dir)
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Ingestion & Temporal Splitting ──────────────────────────────
    logger.info("Phase 1: Ingestion & Temporal Splitting...")

    unsup_train_frames: list[pd.DataFrame] = []
    clf_train_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    classifier_calibration_frames: list[pd.DataFrame] = []
    eval_frames:      list[pd.DataFrame] = []
    source_files = sorted(Path(settings.lbnl_data_dir).glob("*.csv"))
    if args.holdout_scenario and args.holdout_scenario not in {p.name for p in source_files}:
        parser.error(f"Unknown --holdout-scenario file: {args.holdout_scenario}")

    ingestion_iterator = (
        iter_parquet_pipeline(settings)
        if args.input_format == "parquet"
        else iter_ingestion_pipeline(settings)
    )
    for i, chunk_df in enumerate(ingestion_iterator):
        chunk_df["event_time"] = pd.to_datetime(chunk_df["event_time"])
        chunk_df["scenario_file"] = (
            source_files[i].name if i < len(source_files) else f"file_{i}"
        )

        base_year = int(chunk_df["event_time"].dt.year.min())
        if args.split_protocol == "common":
            train_start = pd.Timestamp(f"{base_year}-04-01 00:00:00")
            train_cutoff = pd.Timestamp(f"{base_year}-08-31 23:59:59")
            validation_cutoff = pd.Timestamp(f"{base_year}-09-30 23:59:59")
            final_cutoff = pd.Timestamp(f"{base_year}-10-31 23:59:59")
        else:
            train_start = pd.Timestamp(f"{base_year}-01-01 00:00:00")
            train_cutoff = pd.Timestamp(f"{base_year}-09-30 23:59:59")
            validation_cutoff = pd.Timestamp(f"{base_year}-11-30 23:59:59")
            final_cutoff = pd.Timestamp(f"{base_year}-12-31 23:59:59")
        if i == 0:
            logger.info(
                "Using %s split protocol: train=%s..%s, validation=%s..%s, final>%s..%s",
                args.split_protocol,
                train_start,
                train_cutoff,
                train_cutoff,
                validation_cutoff,
                validation_cutoff,
                final_cutoff,
            )

        train_chunk = chunk_df[
            (chunk_df["event_time"] >= train_start)
            & (chunk_df["event_time"] <= train_cutoff)
        ]
        if args.evaluation_window == "validation":
            eval_chunk = chunk_df[
                (chunk_df["event_time"] > train_cutoff)
                & (chunk_df["event_time"] <= validation_cutoff)
            ]
        else:
            eval_chunk = chunk_df[
                (chunk_df["event_time"] > validation_cutoff)
                & (chunk_df["event_time"] <= final_cutoff)
            ]

        validation_chunk = chunk_df[
            (chunk_df["event_time"] > train_cutoff)
            & (chunk_df["event_time"] <= validation_cutoff)
        ]

        if not eval_chunk.empty:
            eval_chunk = eval_chunk.copy()

        is_holdout = (
            args.holdout_scenario is not None
            and i < len(source_files)
            and source_files[i].name == args.holdout_scenario
        )
        is_normal_reference = source_files[i].stem.lower() == "ahu_annual"

        if args.train_unsup and not train_chunk.empty and not is_holdout:
            unsup_train_frames.append(train_chunk[train_chunk["fault_type"] == FaultType.NORMAL.value])
        if (
            args.gmm_target_fpr is not None
            and not validation_chunk.empty
            and not is_holdout
        ):
            calibration_frames.append(
                validation_chunk[validation_chunk["fault_type"] == FaultType.NORMAL.value]
            )
        if (
            args.classifier_target_fpr is not None
            and not validation_chunk.empty
            and not is_holdout
        ):
            classifier_calibration_frames.append(validation_chunk)

        if args.train_clf and not train_chunk.empty and not is_holdout:
            # Train the classifier on both normal and fault data so it can act as a 
            # standalone multi-class detector.
            if settings.supervised_model.lower() == "tcn":
                # TCN requires continuous time series for sliding windows.
                # Take the last 20% of the chunk continuously.
                split_idx = int(len(train_chunk) * 0.8)
                clf_train_frames.append(train_chunk.iloc[split_idx:].copy())
            else:
                # Vary random_state per chunk to avoid picking the same relative
                # row positions across structurally similar scenario files.
                clf_train_frames.append(train_chunk.sample(frac=0.2, random_state=42 + i))

        if (
            not eval_chunk.empty
            and (uses_detector or args.evaluate)
            and (
                args.holdout_scenario is None
                or is_holdout
                or (args.include_normal_reference and is_normal_reference)
            )
        ):
            eval_frames.append(eval_chunk)

    # ── Phase 2: Differential Training ───────────────────────────────────────
    if args.train_unsup and unsup_train_frames:
        logger.info("Phase 2a: Training %s on 100%% Normal data...", settings.unsupervised_model.upper())
        _train_unsup(
            unsup_train_frames,
            settings,
            calibration_frames=calibration_frames,
            target_fpr=args.gmm_target_fpr,
        )

    if args.train_clf and clf_train_frames:
        logger.info("Phase 2b: Training Classifier on 20%% Sampled data...")
        _train_classifier(
            clf_train_frames,
            settings,
            calibration_frames=classifier_calibration_frames,
            target_fpr=args.classifier_target_fpr,
        )

    # ── Phase 3: Per-scenario Detection ──────────────────────────────────────
    # eval_frames is fully in memory from Phase 1. Iterating one frame at a time
    # limits the peak allocation per detection pass to a single scenario's data.
    all_events: list[DetectionEvent] = []
    all_event_frames: list[pd.DataFrame] = []
    total_detected_events = 0
    if eval_frames and uses_detector:
        logger.info("Phase 3: Running per-scenario detection...")

        rules_det = LBNLRulesDetector(settings) if args.use_rules else None
        
        unsup_det = None
        if args.use_unsup:
            if settings.unsupervised_model == "gmm":
                from hvac_fdd.detection.gmm_detector import GMMDetector
                unsup_det = GMMDetector.load(settings.models_dir / "gmm_detector.joblib", settings=settings)
            elif settings.unsupervised_model == "if":
                from hvac_fdd.detection.isolation_forest import IsolationForestDetector
                unsup_det = IsolationForestDetector.load(settings.models_dir / "if_detector.joblib", settings=settings)
            elif settings.unsupervised_model == "kan":
                from hvac_fdd.detection.kan_detector import KANDetector
                unsup_det = KANDetector.load(settings.models_dir / "kan_detector.joblib", settings=settings)
            else:
                raise ValueError(f"Unknown unsupervised model: {settings.unsupervised_model}")
        clf_det = (
            FaultClassifier.load(settings.models_dir / "classifier.joblib", settings=settings)
            if args.use_clf else None
        )

        for i, eval_chunk in enumerate(eval_frames):
            hits = []
            if rules_det:
                hits.append(rules_det.predict(eval_chunk))
            if unsup_det:
                hits.append(unsup_det.predict(eval_chunk))

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
                    combined = _attach_ground_truth(combined, eval_chunk)
                    multi = combined["all_sources"].str.contains("|", regex=False).sum()
                    if multi:
                        logger.debug("  %d/%d events agreed by multiple detectors", multi, len(combined))

                    if clf_det:
                        keys = combined[["event_time", "zone_id"]]
                        to_classify = eval_chunk.merge(keys, on=["event_time", "zone_id"])
                        clf_results = clf_det.predict(to_classify)
                        if not clf_results.empty:
                            clf_results = _attach_ground_truth(clf_results, eval_chunk)
                            combined = combined.merge(
                                clf_results[["event_time", "zone_id", "predicted_fault", "confidence"]],
                                on=["event_time", "zone_id"], how="left",
                            )

                    total_detected_events += len(combined)
                    if args.persist:
                        all_events.extend(_to_domain_events(combined, settings.default_equipment_id))
                    
                    if args.evaluate:
                        combined_eval = combined.copy()
                        combined_eval["file_index"] = i
                        all_event_frames.append(combined_eval)
            else:
                # If no primary anomaly detectors were used (e.g. evaluating Classifier only)
                if clf_det:
                    clf_results = clf_det.predict(eval_chunk)
                    if not clf_results.empty:
                        clf_results = _attach_ground_truth(clf_results, eval_chunk)
                        total_detected_events += len(clf_results)
                        if args.persist:
                            all_events.extend(_to_domain_events(clf_results, settings.default_equipment_id))
                        if args.evaluate:
                            combined_eval = clf_results.copy()
                            combined_eval["file_index"] = i
                            all_event_frames.append(combined_eval)

        logger.info("Detection complete: %d events found.", total_detected_events)

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
        mini_frames = []
        for i, df in enumerate(eval_frames):
            mini = df[["event_time", "zone_id", "fault_type", "scenario_file"]].copy()
            mini["file_index"] = i
            mini_frames.append(mini)
        eval_df_full = pd.concat(mini_frames, ignore_index=True)
        
        events_df = pd.concat(all_event_frames, ignore_index=True) if all_event_frames else pd.DataFrame()
        _run_evaluation_vectorized(
            eval_df_full,
            events_df,
            used_classifier=args.use_clf,
            evaluation_window=args.evaluation_window,
            split_protocol=args.split_protocol,
        )


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


def _attach_ground_truth(events: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the scenario label after detector-specific deduplication."""
    if events.empty or "fault_type" not in source_df.columns:
        return events

    labels = source_df[["event_time", "zone_id", "fault_type"]].drop_duplicates(
        subset=["event_time", "zone_id"]
    )
    result = events.drop(columns=["ground_truth"], errors="ignore").merge(
        labels,
        on=["event_time", "zone_id"],
        how="left",
    )
    return result.rename(columns={"fault_type": "ground_truth"})


def _run_evaluation_vectorized(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    used_classifier: bool,
    evaluation_window: str,
    split_protocol: str,
) -> None:
    """Vectorized evaluation to prevent Pandas .apply() memory bombs."""
    if events_df.empty:
        df["detected"] = False
        df["predicted_fault"] = FaultType.NORMAL.value
    else:
        if "predicted_fault" not in events_df.columns:
            events_df["predicted_fault"] = None
            
        events_df = events_df[["file_index", "event_time", "zone_id", "predicted_fault"]].copy()
        events_df["predicted_fault"] = events_df["predicted_fault"].fillna(FaultType.NORMAL.value)
        events_df = events_df.drop_duplicates(subset=["file_index", "event_time", "zone_id"], keep="first")
        events_df["detected"] = True

        df = df.merge(events_df, on=["file_index", "event_time", "zone_id"], how="left")
        df["detected"] = df["detected"].fillna(False)
        df["predicted_fault"] = df["predicted_fault"].fillna(FaultType.NORMAL.value)

    rep = detection_report(df)
    logger.info("--- Detection Report (%s window) ---", evaluation_window)
    logger.info("Overall: %s", rep["overall"])
    for fault, metrics in rep.get("per_fault", {}).items():
        logger.info("  %s: recall=%.4f", fault, metrics["recall"])

    logger.info("--- Per-scenario report (%s protocol) ---", split_protocol)
    for scenario_file, scenario_df in df.groupby("scenario_file", sort=True):
        is_fault = scenario_df["fault_type"] != FaultType.NORMAL.value
        if is_fault.all():
            logger.info(
                "  %s: n=%d fault_recall=%.4f detected_rows=%d",
                scenario_file,
                len(scenario_df),
                float(scenario_df["detected"].mean()),
                int(scenario_df["detected"].sum()),
            )
        elif (~is_fault).all():
            logger.info(
                "  %s: n=%d normal_false_positive_rate=%.4f false_positives=%d",
                scenario_file,
                len(scenario_df),
                float(scenario_df["detected"].mean()),
                int(scenario_df["detected"].sum()),
            )
        else:
            logger.warning("  %s: mixed ground-truth labels; scenario metrics skipped", scenario_file)

    alert_bursts = _count_alert_bursts(df, gap_minutes=5)
    logger.info(
        "Alert bursts (detected rows grouped by file/zone, gap <= 5 min): %d",
        alert_bursts,
    )

    if used_classifier:
        crep = classification_report_extended(df["fault_type"], df["predicted_fault"])
        logger.info("--- Classification Report (%s window) ---", evaluation_window)
        logger.info("Accuracy: %.4f | Macro F1: %.4f", crep["accuracy"], crep["macro_f1"])


def _count_alert_bursts(df: pd.DataFrame, *, gap_minutes: int) -> int:
    """Count contiguous predicted-alert bursts; no ground-truth event labels are assumed."""
    detected = df[df["detected"]].sort_values(
        ["file_index", "zone_id", "event_time"]
    )
    if detected.empty:
        return 0
    delta = (
        detected.groupby(["file_index", "zone_id"])["event_time"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )
    return int(delta.isna().sum() + (delta > gap_minutes).sum())


if __name__ == "__main__":
    main()
