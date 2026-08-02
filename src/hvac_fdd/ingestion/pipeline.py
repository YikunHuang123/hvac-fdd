"""
Ingestion pipeline orchestrator.

Wires together: load → wide_zones_to_long → fahrenheit_to_celsius
                → normalize_column_names → build_feature_set → (optional save)

Detection (Phase 3) and DB persistence (Phase 5) are intentionally
absent here; they will call run_ingestion_pipeline() and consume its output.
"""
from __future__ import annotations

import logging
from pathlib import Path

import typing
import pandas as pd

from hvac_fdd.config import Settings, get_settings
from hvac_fdd.exceptions import PipelineError
from hvac_fdd.ingestion.features import build_feature_set
from hvac_fdd.ingestion.lbnl_loader import LBNLDataLoader
from hvac_fdd.ingestion.transforms import (
    TEMP_COLS_RAW,
    fahrenheit_to_celsius,
    normalize_column_names,
    wide_zones_to_long,
)

logger = logging.getLogger(__name__)

# Name of the output parquet file written to settings.processed_data_dir.
_OUTPUT_FILENAME = "lbnl_features.parquet"


def run_ingestion_pipeline(
    settings: Settings | None = None,
    *,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run the full ingestion pipeline and return the feature-engineered DataFrame.

    Args:
        settings: Settings instance; uses get_settings() when None.
        save:     If True, persist the result to processed_data_dir as parquet.

    Raises PipelineError with a 'stage' attribute identifying the failing step:
        "load"      — CSV reading or schema validation failed
        "transform" — wide_zones_to_long / F→C / rename failed
        "features"  — feature engineering failed
        "save"      — parquet write failed
    """
    if settings is None:
        settings = get_settings()

    # ── Stage 1: Load ─────────────────────────────────────────────────────────
    try:
        loader = LBNLDataLoader(settings.lbnl_data_dir)
        df = loader.load_all()
        logger.info("Load complete: %d raw rows", len(df))
    except Exception as exc:
        raise PipelineError(str(exc), stage="load") from exc

    # ── Stage 2: Transform ────────────────────────────────────────────────────
    try:
        df = wide_zones_to_long(df)
        df = fahrenheit_to_celsius(df, columns=TEMP_COLS_RAW)
        df = normalize_column_names(df)
        logger.info("Transform complete: %d rows × %d cols", len(df), len(df.columns))
    except Exception as exc:
        raise PipelineError(str(exc), stage="transform") from exc

    # ── Stage 3: Feature engineering ──────────────────────────────────────────
    try:
        df = build_feature_set(df, settings)
        logger.info("Features complete: %d cols total", len(df.columns))
    except Exception as exc:
        raise PipelineError(str(exc), stage="features") from exc

    # ── Stage 4: Persist (optional) ───────────────────────────────────────────
    if save:
        try:
            _write_parquet(df, settings.processed_data_dir)
        except Exception as exc:
            raise PipelineError(str(exc), stage="save") from exc

    logger.info(
        "Ingestion pipeline complete: %d rows, %d columns",
        len(df), len(df.columns),
    )
    return df


def _write_parquet(df: pd.DataFrame, output_dir: Path | str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _OUTPUT_FILENAME
    df.to_parquet(out_path, index=False)
    logger.info("Saved processed features to %s", out_path)


def iter_ingestion_pipeline(
    settings: Settings | None = None,
    *,
    scenario: str | None = None,
) -> typing.Iterator[pd.DataFrame]:
    """
    Chunked ingestion pipeline. Yields one fully processed DataFrame per input CSV.
    Downcasts float64 to float32 to reduce memory footprint by 50%.
    """
    if settings is None:
        settings = get_settings()

    loader = LBNLDataLoader(settings.lbnl_data_dir)
    for i, df in enumerate(loader.iter_files(scenario=scenario)):
        try:
            df = wide_zones_to_long(df)
            df = fahrenheit_to_celsius(df, columns=TEMP_COLS_RAW)
            df = normalize_column_names(df)
            df = build_feature_set(df, settings)
            
            # Memory optimization: Downcast floats
            float_cols = df.select_dtypes(include=["float64"]).columns
            if len(float_cols) > 0:
                df[float_cols] = df[float_cols].astype("float32")
                
            yield df
        except Exception as exc:
            logger.exception("Failed to process file chunk %d: %s", i, exc)
