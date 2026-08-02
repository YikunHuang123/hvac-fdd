#!/usr/bin/env python3
"""Run reproducible leave-one-severity-out experiments and collect summaries."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path


OVERALL_RE = re.compile(r"Overall: (\{.*\})")
CLASSIFICATION_RE = re.compile(
    r"classification_report_extended: .*accuracy=([0-9.]+), macro_f1=([0-9.]+)"
)
NORMAL_FPR_RE = re.compile(r"normal_false_positive_rate=([0-9.]+)")


def _scenario_files(data_dir: Path) -> list[Path]:
    return sorted(
        path for path in data_dir.glob("*.csv") if path.stem.lower() != "ahu_annual"
    )


def _command_for(model: str, scenario: str, models_dir: Path, target_fpr: float) -> list[str]:
    base = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--split-protocol",
        "common",
        "--evaluation-window",
        "final",
        "--holdout-scenario",
        scenario,
        "--include-normal-reference",
        "--evaluate",
        "--models-dir",
        str(models_dir),
    ]
    if model == "tcn":
        return base + [
            "--train-clf",
            "--use-clf",
            "--classifier-target-fpr",
            str(target_fpr),
        ]
    if model == "xgboost":
        return base + ["--train-clf", "--use-clf"]
    if model == "rules_gmm":
        return base + ["--train-unsup", "--use-rules", "--use-unsup"]
    raise ValueError(f"Unsupported model: {model}")


def _parse_summary(output: str) -> dict[str, float]:
    summary: dict[str, float] = {}
    overall = OVERALL_RE.search(output)
    if overall:
        values = ast.literal_eval(overall.group(1))
        summary.update({f"detection_{k}": float(v) for k, v in values.items()})
    classification = CLASSIFICATION_RE.search(output)
    if classification:
        summary["classification_accuracy"] = float(classification.group(1))
        summary["classification_macro_f1"] = float(classification.group(2))
    normal_fpr = NORMAL_FPR_RE.search(output)
    if normal_fpr:
        summary["normal_reference_fpr"] = float(normal_fpr.group(1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tcn", "xgboost", "rules_gmm"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/LBNL_FDD_Data_Sets_SDAHU_all_3/LBNL_FDD_Dataset_SDAHU"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/holdout_severity"))
    parser.add_argument("--limit", type=int, help="Run only the first N scenarios for a smoke test")
    parser.add_argument("--target-fpr", type=float, default=0.10)
    args = parser.parse_args()

    scenarios = _scenario_files(args.data_dir)
    if args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        parser.error("No fault scenario CSV files found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    env = os.environ.copy()
    if args.model == "tcn":
        env["SUPERVISED_MODEL"] = "tcn"

    for scenario_path in scenarios:
        run_dir = args.output_dir / args.model / scenario_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        command = _command_for(args.model, scenario_path.name, run_dir, args.target_fpr)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        output = completed.stdout + "\n" + completed.stderr
        (run_dir / "run.log").write_text(output, encoding="utf-8")
        row = {
            "scenario": scenario_path.name,
            "model": args.model,
            "returncode": completed.returncode,
            **_parse_summary(output),
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

    (args.output_dir / f"summary_{args.model}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if any(row["returncode"] != 0 for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
