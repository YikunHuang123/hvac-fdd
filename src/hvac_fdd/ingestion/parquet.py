"""Read feature-engineered Parquet caches with column projection."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from hvac_fdd.config import Settings
from hvac_fdd.exceptions import DataLoadError
from hvac_fdd.ingestion.features import ENG_FEATURE_COLUMNS

_CACHE_PATTERN = "lbnl_features_*.parquet"
_IDENTIFIER_COLUMNS = ["event_time", "zone_id", "fault_type"]
_RULE_COLUMNS = [
    "chwc_valve_pct",
    "temp_supply_celsius",
    "temp_mixed_celsius",
    "sf_power_w",
    "sa_temp_error_c_60_std",
    "sf_power_w_60_std",
]
_DEFAULT_COLUMNS = list(
    dict.fromkeys(_IDENTIFIER_COLUMNS + _RULE_COLUMNS + ENG_FEATURE_COLUMNS)
)


def iter_parquet_pipeline(
    settings: Settings | None = None,
    *,
    columns: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield projected feature frames from the per-file Parquet cache.

    DuckDB performs the column projection inside the Parquet scan, so unused
    raw and engineered columns are not materialized into Pandas.
    """
    if settings is None:
        from hvac_fdd.config import get_settings

        settings = get_settings()

    paths = _find_cache_paths(Path(settings.processed_data_dir))
    if not paths:
        raise DataLoadError(
            f"No Parquet cache files found in {settings.processed_data_dir}; "
            "run scripts/preprocess_data.py first"
        )

    selected = list(dict.fromkeys(_DEFAULT_COLUMNS if columns is None else columns))
    _validate_columns(selected)

    try:
        import duckdb
    except ImportError as exc:
        raise DataLoadError(
            "DuckDB is required for Parquet cache reads; install project dependencies"
        ) from exc

    connection = duckdb.connect()
    try:
        projection = ", ".join(_quote_identifier(column) for column in selected)
        for path in paths:
            available = {
                row[0]
                for row in connection.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)", [str(path.resolve())]
                ).fetchall()
            }
            missing = [column for column in selected if column not in available]
            if missing:
                raise DataLoadError(
                    f"Parquet cache {path.name} is missing required columns: {missing}; "
                    "regenerate it with scripts/preprocess_data.py --force"
                )
            query = f"SELECT {projection} FROM read_parquet(?)"
            yield connection.execute(query, [str(path.resolve())]).fetch_df()
    finally:
        connection.close()


def _validate_columns(columns: Sequence[str]) -> None:
    if not columns:
        raise ValueError("At least one Parquet column must be selected")
    if any(not column or '"' in column for column in columns):
        raise ValueError("Invalid Parquet projection column")


def _find_cache_paths(processed_data_dir: Path) -> list[Path]:
    """Find per-file caches produced by either current or earlier scripts."""
    current = sorted(processed_data_dir.glob(_CACHE_PATTERN))
    if current:
        return current

    return sorted(
        path
        for path in processed_data_dir.glob("*.parquet")
        if path.name != "lbnl_features.parquet"
    )


def _quote_identifier(column: str) -> str:
    return f'"{column}"'
