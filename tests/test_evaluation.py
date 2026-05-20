"""
Phase 4 evaluation tests.

`pytest -v tests/test_evaluation.py`

Coverage per REFACTORING_PLAN.md Phase 4:
  - detection_report: P/R/F1 correctness with known labels
  - classification_report_extended: confusion matrix + per-class AUC
  - time_to_detect: delay calculation
  - generate_report: JSON and Markdown output
  - ground_truth=None rows are skipped in all metric functions

All tests use small in-memory DataFrames — no actual CSV files are read.
The test_settings fixture (conftest.py) provides Settings with safe defaults.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from hvac_fdd.domain import FaultType
from hvac_fdd.evaluation.metrics import (
    classification_report_extended,
    detection_report,
    time_to_detect,
)
from hvac_fdd.evaluation.report import generate_report
from hvac_fdd.exceptions import FeatureEngineeringError, PipelineError, SchemaValidationError

# ── Shared fixture helpers ────────────────────────────────────────────────────

_NORMAL     = FaultType.NORMAL.value
_COI_STUCK  = FaultType.COI_STUCK.value
_OA_BIAS    = FaultType.OA_BIAS.value
_COI_BIAS   = FaultType.COI_BIAS.value
_DAMPER     = FaultType.DAMPER_STUCK.value
_LEAKAGE    = FaultType.COI_LEAKAGE.value


def _detection_df(
    fault_types: list[str | None],
    detected: list[bool],
) -> pd.DataFrame:
    """Build a minimal detection DataFrame for detection_report tests."""
    return pd.DataFrame({"fault_type": fault_types, "detected": detected})


# ── detection_report: schema validation ──────────────────────────────────────


class TestDetectionReportSchema:
    def test_missing_fault_type_column_raises(self):
        df = pd.DataFrame({"detected": [True, False]})
        with pytest.raises(SchemaValidationError, match="fault_type"):
            detection_report(df)

    def test_missing_detected_column_raises(self):
        df = pd.DataFrame({"fault_type": [_NORMAL, _COI_STUCK]})
        with pytest.raises(SchemaValidationError, match="detected"):
            detection_report(df)

    def test_all_none_ground_truth_raises(self):
        df = _detection_df([None, None, None], [True, False, True])
        with pytest.raises(FeatureEngineeringError, match="no rows with valid"):
            detection_report(df)


# ── detection_report: None/NaN rows are skipped ───────────────────────────────


class TestDetectionReportSkipsNone:
    def test_none_rows_excluded_from_n_evaluated(self):
        df = _detection_df(
            [_COI_STUCK, None, _NORMAL],
            [True,       True,  False],
        )
        result = detection_report(df)
        assert result["n_evaluated"] == 2
        assert result["n_skipped"]   == 1

    def test_nan_rows_excluded(self):
        import numpy as np
        df = _detection_df(
            [_COI_STUCK, float("nan"), _NORMAL],
            [True,        True,         False],
        )
        result = detection_report(df)
        assert result["n_evaluated"] == 2
        assert result["n_skipped"]   == 1

    def test_none_row_does_not_affect_metrics(self):
        """Inserting a None row should not change the computed F1."""
        df_clean = _detection_df(
            [_COI_STUCK, _NORMAL],
            [True,       False],
        )
        df_with_none = _detection_df(
            [_COI_STUCK, None, _NORMAL],
            [True,       True, False],   # the None row is detected but should be ignored
        )
        r_clean = detection_report(df_clean)
        r_none  = detection_report(df_with_none)
        assert r_clean["overall"]["f1"] == r_none["overall"]["f1"]


# ── detection_report: metric correctness ─────────────────────────────────────


class TestDetectionReportMetrics:
    def test_perfect_detection(self):
        """All faults detected, no false alarms → P=R=F1=1.0."""
        df = _detection_df(
            [_COI_STUCK, _NORMAL, _COI_STUCK, _NORMAL],
            [True,        False,   True,        False],
        )
        result = detection_report(df)
        ov = result["overall"]
        assert ov["precision"] == 1.0
        assert ov["recall"]    == 1.0
        assert ov["f1"]        == 1.0

    def test_no_detections_gives_zero_precision_and_recall(self):
        df = _detection_df(
            [_COI_STUCK, _NORMAL],
            [False,       False],
        )
        result = detection_report(df)
        ov = result["overall"]
        assert ov["precision"] == 0.0
        assert ov["recall"]    == 0.0
        assert ov["f1"]        == 0.0

    def test_all_detected_with_no_faults_gives_zero_precision(self):
        """All NORMAL rows detected → all detections are false alarms."""
        df = _detection_df(
            [_NORMAL, _NORMAL, _NORMAL],
            [True,    True,    True],
        )
        result = detection_report(df)
        assert result["overall"]["precision"] == 0.0
        assert result["overall"]["recall"]    == 0.0  # no true positives

    def test_partial_detection_recall_equals_fraction(self):
        """2 of 4 fault rows detected → recall = 0.5."""
        df = _detection_df(
            [_COI_STUCK, _COI_STUCK, _COI_STUCK, _COI_STUCK],
            [True,        True,       False,       False],
        )
        result = detection_report(df)
        assert result["overall"]["recall"] == 0.5

    def test_f1_is_harmonic_mean_of_precision_and_recall(self):
        df = _detection_df(
            [_COI_STUCK, _NORMAL, _COI_STUCK, _NORMAL],
            [True,        True,    False,       False],   # 1 TP, 1 FP, 1 FN, 1 TN
        )
        result = detection_report(df)
        ov = result["overall"]
        p, r = ov["precision"], ov["recall"]
        expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        assert abs(ov["f1"] - expected_f1) < 1e-6

    def test_result_dict_has_required_keys(self):
        df = _detection_df([_COI_STUCK, _NORMAL], [True, False])
        result = detection_report(df)
        assert "n_evaluated" in result
        assert "n_skipped"   in result
        assert "overall"     in result
        assert "per_fault"   in result

    def test_overall_has_precision_recall_f1(self):
        df = _detection_df([_COI_STUCK, _NORMAL], [True, False])
        ov = detection_report(df)["overall"]
        assert "precision" in ov
        assert "recall"    in ov
        assert "f1"        in ov

    def test_per_fault_section_contains_all_non_normal_types(self):
        df = _detection_df([_COI_STUCK, _NORMAL], [True, False])
        result = detection_report(df)
        for ft in FaultType:
            if ft == FaultType.NORMAL:
                continue
            assert ft.value in result["per_fault"], f"{ft.value} missing from per_fault"

    def test_per_fault_recall_for_undetected_type_is_zero(self):
        """coi_leakage has rows but none detected → recall = 0."""
        df = _detection_df(
            [_LEAKAGE, _LEAKAGE, _NORMAL],
            [False,     False,    False],
        )
        result = detection_report(df)
        assert result["per_fault"][_LEAKAGE]["recall"] == 0.0

    def test_per_fault_recall_for_fully_detected_type_is_one(self):
        df = _detection_df(
            [_LEAKAGE, _LEAKAGE, _NORMAL],
            [True,     True,     False],
        )
        result = detection_report(df)
        assert result["per_fault"][_LEAKAGE]["recall"] == 1.0

    def test_custom_fault_types_filters_output(self):
        df = _detection_df([_COI_STUCK, _NORMAL], [True, False])
        result = detection_report(df, fault_types=[FaultType.COI_STUCK])
        assert list(result["per_fault"].keys()) == [_COI_STUCK]

    def test_metrics_are_floats_in_0_1_range(self):
        df = _detection_df(
            [_COI_STUCK, _NORMAL, _COI_STUCK, _NORMAL],
            [True,        False,   True,        True],
        )
        result = detection_report(df)
        for metric_val in result["overall"].values():
            assert 0.0 <= metric_val <= 1.0


# ── classification_report_extended ───────────────────────────────────────────


class TestClassificationReportExtended:
    def test_all_none_labels_raises(self):
        with pytest.raises(FeatureEngineeringError, match="no rows with valid"):
            classification_report_extended(
                pd.Series([None, None]),
                pd.Series([None, None]),
            )

    def test_none_rows_skipped(self):
        y_true = pd.Series([_COI_STUCK, None, _NORMAL])
        y_pred = pd.Series([_COI_STUCK, _OA_BIAS, _NORMAL])
        result = classification_report_extended(y_true, y_pred)
        assert result["n_evaluated"] == 2
        assert result["n_skipped"]   == 1

    def test_perfect_classification(self):
        y_true = pd.Series([_COI_STUCK, _NORMAL, _OA_BIAS])
        y_pred = pd.Series([_COI_STUCK, _NORMAL, _OA_BIAS])
        result = classification_report_extended(y_true, y_pred)
        assert result["accuracy"]  == 1.0
        assert result["macro_f1"]  == 1.0
        for cls_vals in result["per_class"].values():
            assert cls_vals["precision"] == 1.0
            assert cls_vals["recall"]    == 1.0
            assert cls_vals["f1"]        == 1.0

    def test_confusion_matrix_shape(self):
        labels = [_COI_STUCK, _NORMAL, _OA_BIAS]
        y_true = pd.Series(labels)
        y_pred = pd.Series(labels)
        result = classification_report_extended(y_true, y_pred)
        cm = result["confusion_matrix"]
        n  = len(cm["labels"])
        assert len(cm["matrix"]) == n
        assert all(len(row) == n for row in cm["matrix"])

    def test_confusion_matrix_counts_correctly(self):
        """One misclassification: coi_stuck predicted as oa_bias."""
        y_true = pd.Series([_COI_STUCK, _NORMAL])
        y_pred = pd.Series([_OA_BIAS,   _NORMAL])
        result = classification_report_extended(y_true, y_pred)
        labels = result["confusion_matrix"]["labels"]
        matrix = result["confusion_matrix"]["matrix"]
        oa_idx  = labels.index(_OA_BIAS)
        coi_idx = labels.index(_COI_STUCK)
        # row = actual, col = predicted; (coi_stuck, oa_bias) should be 1
        assert matrix[coi_idx][oa_idx] == 1

    def test_per_class_support_sums_to_n_evaluated(self):
        y_true = pd.Series([_COI_STUCK, _NORMAL, _COI_STUCK])
        y_pred = pd.Series([_COI_STUCK, _NORMAL, _COI_STUCK])
        result = classification_report_extended(y_true, y_pred)
        total_support = sum(v["support"] for v in result["per_class"].values())
        assert total_support == result["n_evaluated"]

    def test_auc_is_one_for_perfect_classifier(self):
        y_true = pd.Series([_COI_STUCK, _NORMAL, _OA_BIAS])
        y_pred = pd.Series([_COI_STUCK, _NORMAL, _OA_BIAS])
        result = classification_report_extended(y_true, y_pred)
        for cls_vals in result["per_class"].values():
            assert abs(cls_vals["auc"] - 1.0) < 1e-6

    def test_auc_is_nan_when_class_absent_from_true(self):
        """If a label appears only in y_pred (not in y_true), AUC cannot be computed."""
        y_true = pd.Series([_NORMAL, _NORMAL])
        y_pred = pd.Series([_NORMAL, _COI_STUCK])   # coi_stuck only in y_pred
        result = classification_report_extended(y_true, y_pred)
        coi_auc = result["per_class"].get(_COI_STUCK, {}).get("auc", None)
        if coi_auc is not None:
            assert math.isnan(coi_auc), f"Expected NaN for absent class, got {coi_auc}"

    def test_result_has_required_top_level_keys(self):
        y_true = pd.Series([_COI_STUCK, _NORMAL])
        y_pred = pd.Series([_COI_STUCK, _NORMAL])
        result = classification_report_extended(y_true, y_pred)
        for key in ("n_evaluated", "n_skipped", "accuracy", "macro_f1",
                    "per_class", "confusion_matrix"):
            assert key in result


# ── time_to_detect ────────────────────────────────────────────────────────────


class TestTimeToDetect:
    _T0 = datetime(2018, 1, 1, 0, 0, 0)

    def _anomaly_df(
        self,
        ground_truths: list[str | None],
        delays_min: list[float],
    ) -> pd.DataFrame:
        """Build an anomaly DataFrame with event_times offset from _T0."""
        times = [
            self._T0 + timedelta(minutes=d) if d is not None else None
            for d in delays_min
        ]
        return pd.DataFrame({"event_time": times, "ground_truth": ground_truths})

    def test_missing_column_raises(self):
        df = pd.DataFrame({"event_time": [self._T0]})
        with pytest.raises(SchemaValidationError, match="ground_truth"):
            time_to_detect(df, {})

    def test_mean_delay_single_fault_type(self):
        """Two COI_STUCK events at +10 and +20 min → mean delay = 15 min."""
        df = self._anomaly_df(
            [_COI_STUCK, _COI_STUCK],
            [10.0,       20.0],
        )
        result = time_to_detect(df, {FaultType.COI_STUCK: self._T0})
        assert abs(result[_COI_STUCK] - 15.0) < 1e-6

    def test_none_ground_truth_rows_skipped(self):
        """None rows must not inflate the mean."""
        df = self._anomaly_df(
            [_COI_STUCK, None],
            [10.0,       100.0],  # if None row were included, mean = 55, not 10
        )
        result = time_to_detect(df, {FaultType.COI_STUCK: self._T0})
        assert abs(result[_COI_STUCK] - 10.0) < 1e-6

    def test_fault_type_with_no_events_omitted(self):
        df = self._anomaly_df([_NORMAL], [0.0])
        result = time_to_detect(df, {FaultType.COI_STUCK: self._T0})
        assert _COI_STUCK not in result

    def test_multiple_fault_types(self):
        df = self._anomaly_df(
            [_COI_STUCK, _OA_BIAS],
            [5.0,        15.0],
        )
        result = time_to_detect(
            df,
            {FaultType.COI_STUCK: self._T0, FaultType.OA_BIAS: self._T0},
        )
        assert abs(result[_COI_STUCK] - 5.0)  < 1e-6
        assert abs(result[_OA_BIAS]   - 15.0) < 1e-6

    def test_empty_anomaly_df_returns_empty_dict(self):
        df = pd.DataFrame({"event_time": [], "ground_truth": []})
        result = time_to_detect(df, {FaultType.COI_STUCK: self._T0})
        assert result == {}

    def test_delay_can_be_negative(self):
        """Detection before fault start → negative delay is valid (early warning)."""
        t_start = datetime(2018, 1, 1, 1, 0, 0)
        df = pd.DataFrame({
            "event_time":   [datetime(2018, 1, 1, 0, 50, 0)],  # 10 min before start
            "ground_truth": [_COI_STUCK],
        })
        result = time_to_detect(df, {FaultType.COI_STUCK: t_start})
        assert result[_COI_STUCK] < 0


# ── generate_report ───────────────────────────────────────────────────────────


class TestGenerateReport:
    @pytest.fixture
    def sample_metrics(self) -> dict:
        return {
            "n_evaluated": 100,
            "n_skipped":   5,
            "overall":     {"precision": 0.87, "recall": 0.82, "f1": 0.845},
            "per_fault": {
                _COI_STUCK: {"precision": 0.9, "recall": 0.8,  "f1": 0.847},
                _LEAKAGE:   {"precision": 0.8, "recall": 0.75, "f1": 0.774},
            },
        }

    def test_json_file_created(self, sample_metrics, tmp_path):
        base = tmp_path / "report"
        generate_report(sample_metrics, base, formats=["json"])
        assert (tmp_path / "report.json").exists()

    def test_markdown_file_created(self, sample_metrics, tmp_path):
        base = tmp_path / "report"
        generate_report(sample_metrics, base, formats=["markdown"])
        assert (tmp_path / "report.md").exists()

    def test_both_formats_created_by_default(self, sample_metrics, tmp_path):
        base = tmp_path / "report"
        generate_report(sample_metrics, base)
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.md").exists()

    def test_json_content_is_valid_and_matches_input(self, sample_metrics, tmp_path):
        base = tmp_path / "report"
        generate_report(sample_metrics, base, formats=["json"])
        loaded = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert loaded["n_evaluated"] == 100
        assert loaded["n_skipped"]   == 5
        assert loaded["overall"]["f1"] == pytest.approx(0.845)

    def test_json_preserves_per_fault_keys(self, sample_metrics, tmp_path):
        base = tmp_path / "report"
        generate_report(sample_metrics, base, formats=["json"])
        loaded = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert _COI_STUCK in loaded["per_fault"]

    def test_markdown_contains_overall_section(self, sample_metrics, tmp_path):
        base = tmp_path / "report"
        generate_report(sample_metrics, base, formats=["markdown"])
        text = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Overall" in text

    def test_markdown_contains_fault_names(self, sample_metrics, tmp_path):
        base = tmp_path / "report"
        generate_report(sample_metrics, base, formats=["markdown"])
        text = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert _COI_STUCK in text
        assert _LEAKAGE   in text

    def test_unsupported_format_raises_pipeline_error(self, sample_metrics, tmp_path):
        with pytest.raises(PipelineError, match="Unsupported report format"):
            generate_report(sample_metrics, tmp_path / "report", formats=["pdf"])

    def test_parent_directory_created_automatically(self, sample_metrics, tmp_path):
        nested = tmp_path / "deep" / "nested" / "report"
        generate_report(sample_metrics, nested, formats=["json"])
        assert nested.with_suffix(".json").exists()

    def test_nan_float_serialised_in_json(self, tmp_path):
        """NaN values (e.g. from AUC when class absent) must not crash JSON serialisation."""
        metrics = {"auc": float("nan"), "n_evaluated": 10, "n_skipped": 0}
        base = tmp_path / "report"
        generate_report(metrics, base, formats=["json"])
        text = (tmp_path / "report.json").read_text(encoding="utf-8")
        assert "nan" in text.lower()

    def test_markdown_contains_confusion_matrix_section(self, tmp_path):
        metrics = {
            "n_evaluated": 10,
            "n_skipped":   0,
            "confusion_matrix": {
                "labels": [_NORMAL, _COI_STUCK],
                "matrix": [[8, 1], [0, 1]],
            },
        }
        base = tmp_path / "report"
        generate_report(metrics, base, formats=["markdown"])
        text = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Confusion Matrix" in text
        assert _NORMAL    in text
        assert _COI_STUCK in text


# ── Integration: detection_report → generate_report roundtrip ────────────────


class TestEvaluationRoundtrip:
    def test_detection_report_output_is_serialisable(self, tmp_path):
        """The dict returned by detection_report must survive JSON serialisation."""
        df = _detection_df(
            [_COI_STUCK, _NORMAL, _COI_STUCK, _NORMAL, None],
            [True,        False,   True,        True,   True],
        )
        metrics = detection_report(df)
        generate_report(metrics, tmp_path / "report")
        loaded = json.loads((tmp_path / "report.json").read_text())
        assert loaded["n_evaluated"] == metrics["n_evaluated"]

    def test_classification_report_output_is_serialisable(self, tmp_path):
        y_true = pd.Series([_COI_STUCK, _NORMAL, _OA_BIAS, None])
        y_pred = pd.Series([_COI_STUCK, _NORMAL, _COI_BIAS, _NORMAL])
        metrics = classification_report_extended(y_true, y_pred)
        generate_report(metrics, tmp_path / "clf_report")
        loaded = json.loads((tmp_path / "clf_report.json").read_text())
        assert loaded["n_evaluated"] == metrics["n_evaluated"]
