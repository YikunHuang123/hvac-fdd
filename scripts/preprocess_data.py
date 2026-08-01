#!/usr/bin/env python
"""
Preprocess LBNL CSV files to Parquet.
"""
import argparse
import logging
from pathlib import Path

from hvac_fdd.config import get_settings
from hvac_fdd.ingestion.pipeline import iter_ingestion_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("preprocess_data")

def main():
    parser = argparse.ArgumentParser(description="Preprocess LBNL CSV data to Parquet")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-file Parquet outputs")
    args = parser.parse_args()

    settings = get_settings()
    output_dir = Path(settings.processed_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting streaming ingestion pipeline...")
    for index, frame in enumerate(iter_ingestion_pipeline(settings)):
        output_path = output_dir / f"lbnl_features_{index:02d}.parquet"
        if output_path.exists() and not args.force:
            logger.info("Skipping existing output: %s", output_path)
            continue
        frame.to_parquet(output_path, index=False)
        logger.info("Wrote %d rows to %s", len(frame), output_path)

    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
