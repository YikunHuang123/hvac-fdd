"""
Fault detection and classification evaluation metrics.

Three public functions:
  detection_report            — binary detection quality (P/R/F1 overall + per fault type)
  classification_report_extended — full multi-class P/R/F1 + confusion matrix
  time_to_detect              — mean detection delay per fault type (minutes)

All functions skip rows where ground-truth labels are None/NaN so they can
safely receive production-mode data mixed with labelled research data.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from hvac_fdd.domain import FaultType
from hvac_fdd.exceptions import FeatureEngineeringError, SchemaValidationError

logger = logging.getLogger(__name__)

# Non-NORMAL fault types evaluated by default in detection_report.
_NON_NORMAL_FAULTS: list[FaultType] = [ft for ft in FaultType if ft != FaultType.NORMAL]


# ── Public functions ──────────────────────────────────────────────────────────


def detection_report(
    df: pd.DataFrame,
    fault_types: list[FaultType] | None = None,
) -> dict:
    """
    Evaluate binary fault detection quality.

    For each fault type the detector is treated as a binary one-vs-rest
    classifier: positive = rows whose ground truth is that fault type,
    score = whether the detector flagged the row.

    Args:
        df: DataFrame with at least two columns:
              fault_type — str, FaultType.value ground-truth label.
                           Rows where this is None or NaN are skipped.
              detected   — bool, True if any detector fired for this row.
        fault_types: Fault types to include in per_fault section.
                     Defaults to all five non-NORMAL FaultType values.

    Returns:
        {
            "n_evaluated": int,
            "n_skipped":   int,
            "overall": {"precision": float, "recall": float, "f1": float},
            "per_fault": {
                "coi_stuck": {"precision": float, "recall": float, "f1": float},
                ...
            }
        }

    Raises:
        SchemaValidationError: if required columns are missing.
        FeatureEngineeringError: if no rows remain after skipping unlabelled rows.
    """
    _require_columns(df, ["fault_type", "detected"])
    if fault_types is None:
        fault_types = _NON_NORMAL_FAULTS

    valid, n_skipped = _drop_unlabelled(df, "fault_type")
    if len(valid) == 0:
        raise FeatureEngineeringError(
            "detection_report: no rows with valid ground-truth labels remain"
        )

    # Binary ground truth: non-NORMAL = 1, NORMAL = 0.
    y_true = (valid["fault_type"] != FaultType.NORMAL.value).astype(int)
    y_pred = valid["detected"].astype(int)

    p_ov, r_ov, f_ov, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    overall = {
        "precision": _round(p_ov),
        "recall":    _round(r_ov),
        "f1":        _round(f_ov),
    }

    per_fault: dict[str, dict] = {}
    for ft in fault_types:
        ft_val   = ft.value
        y_true_x = (valid["fault_type"] == ft_val).astype(int)
        
        # Calculate recall manually: True Positives / Actual Positives
        # We don't report Precision/F1 here because the detector doesn't distinguish fault types.
        actual_positives = y_true_x.sum()
        true_positives = (y_true_x & y_pred).sum()
        recall = float(true_positives / actual_positives) if actual_positives > 0 else 0.0
        
        per_fault[ft_val] = {
            "recall": _round(recall),
        }

    logger.info(
        "detection_report: %d evaluated, %d skipped, overall F1=%.4f",
        len(valid), n_skipped, f_ov,
    )
    return {
        "n_evaluated": len(valid),
        "n_skipped":   n_skipped,
        "overall":     overall,
        "per_fault":   per_fault,
    }


def classification_report_extended(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """
    Full multi-class classification report.

    Rows where either y_true or y_pred is None/NaN are skipped.

    Args:
        y_true: Ground-truth FaultType.value labels.
        y_pred: Predicted FaultType.value labels.

    Returns:
        {
            "n_evaluated": int,
            "n_skipped":   int,
            "accuracy":    float,
            "macro_f1":    float,
            "per_class": {
                "<fault_type>": {
                    "precision": float,
                    "recall":    float,
                    "f1":        float,
                    "support":   int,
                },
                ...
            },
            "confusion_matrix": {
                "labels": list[str],
                "matrix": list[list[int]],
            },
        }

    Raises:
        FeatureEngineeringError: if no rows remain after skipping unlabelled rows.
    """
    y_true = pd.Series(y_true, name="y_true").reset_index(drop=True)
    y_pred = pd.Series(y_pred, name="y_pred").reset_index(drop=True)

    valid_mask = y_true.notna() & y_pred.notna()
    n_skipped  = int((~valid_mask).sum())
    y_true     = y_true[valid_mask]
    y_pred     = y_pred[valid_mask]

    if len(y_true) == 0:
        raise FeatureEngineeringError(
            "classification_report_extended: no rows with valid labels remain"
        )

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    # Per-class precision, recall, F1, support.
    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class: dict[str, dict] = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": _round(precision_arr[i]),
            "recall":    _round(recall_arr[i]),
            "f1":        _round(f1_arr[i]),
            "support":   int(support_arr[i]),
        }

    cm       = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    logger.info(
        "classification_report_extended: %d evaluated, %d skipped, "
        "accuracy=%.4f, macro_f1=%.4f",
        len(y_true), n_skipped, accuracy, macro_f1,
    )
    return {
        "n_evaluated": len(y_true),
        "n_skipped":   n_skipped,
        "accuracy":    _round(accuracy),
        "macro_f1":    _round(macro_f1),
        "per_class":   per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm,
        },
    }


def time_to_detect(
    anomaly_df: pd.DataFrame,
    fault_start_times: dict[FaultType, datetime],
) -> dict[str, float]:
    """
    Compute the mean detection delay per fault type in minutes.

    For each fault type, delay is:
        event_time − fault_start_times[fault_type]
    averaged over all detected events whose ground_truth matches that type.
    Rows where ground_truth is None/NaN are skipped.
    Fault types with zero matching detections are omitted from the result.

    Args:
        anomaly_df:        DataFrame with 'event_time' (datetime-like) and
                           'ground_truth' (FaultType.value string or None) columns.
        fault_start_times: Mapping of FaultType → datetime when the fault began.

    Returns:
        Dict mapping FaultType.value → mean delay in minutes (float).

    Raises:
        SchemaValidationError: if required columns are missing.
    """
    _require_columns(anomaly_df, ["event_time", "ground_truth"])

    valid, _ = _drop_unlabelled(anomaly_df, "ground_truth")
    result: dict[str, float] = {}

    for ft, start_time in fault_start_times.items():
        ft_val  = ft.value
        ft_rows = valid[valid["ground_truth"] == ft_val]
        if ft_rows.empty:
            continue

        delays_min = (
            pd.to_datetime(ft_rows["event_time"]) - pd.Timestamp(start_time)
        ).dt.total_seconds() / 60.0

        result[ft_val] = _round(float(delays_min.mean()))

    logger.info("time_to_detect: computed delays for %d fault types", len(result))
    return result


# ── Private helpers ───────────────────────────────────────────────────────────


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing columns for evaluation: {sorted(missing)}")


def _drop_unlabelled(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, int]:
    """Return (rows_with_valid_label, n_dropped)."""
    mask   = df[col].notna()
    n_drop = int((~mask).sum())
    return df[mask].copy(), n_drop


def _round(value: float, ndigits: int = 4) -> float:
    """Round to ndigits; propagate NaN."""
    if np.isnan(value):
        return value
    return round(value, ndigits)
