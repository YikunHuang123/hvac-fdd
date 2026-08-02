"""Benchmark CSV ingestion against projected Parquet ingestion."""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd
import psutil

from hvac_fdd.config import get_settings
from hvac_fdd.ingestion.parquet import iter_parquet_pipeline
from hvac_fdd.ingestion.pipeline import iter_ingestion_pipeline


@dataclass
class BenchmarkResult:
    name: str
    elapsed_seconds: float
    peak_rss_mb: float
    rows: int
    frames: int
    output_columns: int


class PeakMemorySampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self._process = psutil.Process()
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._peak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "PeakMemorySampler":
        self._peak = self._process.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self) -> float:
        return self._peak / 1024 / 1024

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._process.memory_info().rss)
            self._stop.wait(self._interval)


def _run_benchmark(
    name: str,
    iterator_factory: Callable[[], Iterator[pd.DataFrame]],
) -> BenchmarkResult:
    rows = 0
    frames = 0
    output_columns = 0
    started = time.perf_counter()
    with PeakMemorySampler() as sampler:
        for frame in iterator_factory():
            rows += len(frame)
            frames += 1
            output_columns = max(output_columns, len(frame.columns))
            del frame
    return BenchmarkResult(
        name=name,
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_mb=sampler.peak_mb,
        rows=rows,
        frames=frames,
        output_columns=output_columns,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/parquet_benchmark.json"),
        help="JSON output path",
    )
    args = parser.parse_args()

    settings = get_settings()
    csv_result = _run_benchmark(
        "csv_pandas_stream",
        lambda: iter_ingestion_pipeline(settings),
    )
    parquet_result = _run_benchmark(
        "parquet_duckdb_projection",
        lambda: iter_parquet_pipeline(settings),
    )

    result = {
        "scenario": "all",
        "results": [asdict(csv_result), asdict(parquet_result)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
