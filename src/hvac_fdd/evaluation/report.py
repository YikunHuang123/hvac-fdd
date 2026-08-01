"""
Evaluation report serialisation.

Writes any metrics dict produced by metrics.py to one or more formats:
  json     — pretty-printed JSON file
  markdown — human-readable Markdown table report
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from hvac_fdd.exceptions import PipelineError

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS: frozenset[str] = frozenset({"json", "markdown"})


def generate_report(
    metrics: dict,
    output_path: Path | str,
    formats: list[str] | None = None,
) -> None:
    """
    Serialise a metrics dict to JSON and/or Markdown files.

    The output files are written next to output_path with the appropriate
    suffix appended:
      <output_path>.json
      <output_path>.md

    Args:
        metrics:     Any dict returned by detection_report or
                     classification_report_extended.
        output_path: Base path (without extension).  Parent directories are
                     created automatically.
        formats:     List of format strings; supported values are "json" and
                     "markdown".  Defaults to both.

    Raises:
        PipelineError(stage="report"): if an unsupported format is requested
                                       or a file cannot be written.
    """
    if formats is None:
        formats = ["json", "markdown"]

    unknown = [f for f in formats if f not in _SUPPORTED_FORMATS]
    if unknown:
        raise PipelineError(
            f"Unsupported report format(s): {unknown}. "
            f"Supported: {sorted(_SUPPORTED_FORMATS)}",
            stage="report",
        )

    base = Path(output_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        if fmt == "json":
            _write_json(metrics, base.with_suffix(".json"))
        elif fmt == "markdown":
            _write_markdown(metrics, base.with_suffix(".md"))


# ── Format writers ────────────────────────────────────────────────────────────


def _write_json(metrics: dict, path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=_json_default)
        logger.info("Report written: %s", path)
    except OSError as exc:
        raise PipelineError(f"Cannot write JSON report to {path}: {exc}", stage="report") from exc


def _write_markdown(metrics: dict, path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as fh:
            fh.write(_build_markdown(metrics))
        logger.info("Report written: %s", path)
    except OSError as exc:
        raise PipelineError(f"Cannot write Markdown report to {path}: {exc}", stage="report") from exc


def _build_markdown(metrics: dict) -> str:
    """Render a metrics dict as a Markdown report string."""
    lines: list[str] = ["# HVAC FDD Evaluation Report", ""]

    # ── Summary row ───────────────────────────────────────────────────────────
    n_eval    = metrics.get("n_evaluated", "N/A")
    n_skip    = metrics.get("n_skipped",   "N/A")
    lines += [
        f"**Evaluated rows:** {n_eval}  |  **Skipped (no ground truth):** {n_skip}",
        "",
    ]

    # ── Overall metrics ───────────────────────────────────────────────────────
    if "overall" in metrics:
        ov = metrics["overall"]
        lines += [
            "## Overall Detection Metrics",
            "",
            "| Precision | Recall | F1 |",
            "|----------:|-------:|---:|",
            f"| {_fmt(ov.get('precision'))} | {_fmt(ov.get('recall'))} | {_fmt(ov.get('f1'))} |",
            "",
        ]

    # ── Per-fault detection table ─────────────────────────────────────────────
    if "per_fault" in metrics:
        lines += [
            "## Per-Fault Detection Metrics",
            "",
            "| Fault Type | Precision | Recall | F1 |",
            "|:-----------|----------:|-------:|---:|",
        ]
        for fault_name, vals in metrics["per_fault"].items():
            lines.append(
                f"| {fault_name} "
                f"| {_fmt(vals.get('precision'))} "
                f"| {_fmt(vals.get('recall'))} "
                f"| {_fmt(vals.get('f1'))} |"
            )
        lines.append("")

    # ── Accuracy / macro F1 ───────────────────────────────────────────────────
    if "accuracy" in metrics or "macro_f1" in metrics:
        lines += [
            "## Classification Summary",
            "",
            f"- **Accuracy:** {_fmt(metrics.get('accuracy'))}",
            f"- **Macro F1:** {_fmt(metrics.get('macro_f1'))}",
            "",
        ]

    # ── Per-class classification table ────────────────────────────────────────
    if "per_class" in metrics:
        lines += [
            "## Per-Class Classification Metrics",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|:------|----------:|-------:|---:|--------:|",
        ]
        for cls_name, vals in metrics["per_class"].items():
            lines.append(
                f"| {cls_name} "
                f"| {_fmt(vals.get('precision'))} "
                f"| {_fmt(vals.get('recall'))} "
                f"| {_fmt(vals.get('f1'))} "
                f"| {vals.get('support', 'N/A')} |"
            )
        lines.append("")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    if "confusion_matrix" in metrics:
        cm_labels = metrics["confusion_matrix"].get("labels", [])
        cm_matrix = metrics["confusion_matrix"].get("matrix", [])
        if cm_labels and cm_matrix:
            lines += ["## Confusion Matrix", ""]
            # Header row
            header = "| |" + "".join(f" {lb} |" for lb in cm_labels)
            lines.append(header)
            sep    = "|:---|" + "".join("---:|" for _ in cm_labels)
            lines.append(sep)
            for lb, row in zip(cm_labels, cm_matrix):
                row_str = f"| **{lb}** |" + "".join(f" {v} |" for v in row)
                lines.append(row_str)
            lines.append("")

    return "\n".join(lines)


# ── Utilities ─────────────────────────────────────────────────────────────────


def _fmt(value: object) -> str:
    """Format a numeric value for Markdown display."""
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _json_default(obj: object) -> object:
    """JSON serialisation fallback for NaN/Inf floats."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
