from __future__ import annotations

import logging
from pathlib import Path

import typing
import pandas as pd

from hvac_fdd.domain import FaultType
from hvac_fdd.exceptions import DataLoadError, SchemaValidationError
from hvac_fdd.ingestion.base import DataLoaderBase

logger = logging.getLogger(__name__)


class LBNLDataLoader(DataLoaderBase):
    """
    Load all 21 LBNL SDAHU fault-detection CSVs.

    Responsibilities:
    - Glob all *.csv files under data_dir
    - Infer FaultType from the filename stem (substring match)
    - Attach fault_type column (enum .value string) to each batch
    - Validate that all 31 required LBNL columns are present
    - Support chunked reading to handle ~140 MB per file
    """

    # Lowercase substring → FaultType mapping.
    # Keys are ordered from most specific to least; all are distinct substrings.
    FILENAME_FAULT_MAP: dict[str, FaultType] = {
        "ahu_annual":    FaultType.NORMAL,
        "coi_bias":      FaultType.COI_BIAS,
        "coi_leakage":   FaultType.COI_LEAKAGE,
        "coi_stuck":     FaultType.COI_STUCK,
        "damper_stuck":  FaultType.DAMPER_STUCK,
        "oa_bias":       FaultType.OA_BIAS,
    }

    REQUIRED_COLUMNS: list[str] = [
        "Datetime",
        "CHWC_VLV", "CHWC_VLV_DM",
        "MA_TEMP",
        "OA_CFM", "OA_DMPR", "OA_DMPR_DM", "OA_TEMP",
        "RA_CFM", "RA_DMPR", "RA_DMPR_DM", "RA_TEMP",
        "RF_CS", "RF_SPD", "RF_SPD_DM", "RF_WAT",
        "SA_CFM", "SA_SP", "SA_SPSPT", "SA_TEMP", "SA_TEMPSPT",
        "SF_CS", "SF_SPD", "SF_SPD_DM", "SF_WAT",
        "SYS_CTL",
        "ZONE_TEMP_1", "ZONE_TEMP_2", "ZONE_TEMP_3", "ZONE_TEMP_4", "ZONE_TEMP_5",
    ]

    def __init__(self, data_dir: Path | str) -> None:
        self._data_dir = Path(data_dir)

    # ── Public interface (DataLoaderBase) ─────────────────────────────────────

    def load_all(self, chunksize: int | None = 50_000) -> pd.DataFrame:
        """
        Concatenate all scenario CSVs into one DataFrame.
        Adds a 'fault_type' column (FaultType.value string).
        Raises DataLoadError if no CSV files are found.
        Raises SchemaValidationError if a required column is missing.
        """
        csv_paths = sorted(self._data_dir.glob("*.csv"))
        if not csv_paths:
            raise DataLoadError(f"No CSV files found in {self._data_dir}")

        frames: list[pd.DataFrame] = []
        for path in csv_paths:
            fault_type = self._infer_fault_type(path.stem)
            df = self._read_csv(path, chunksize)
            df["fault_type"] = fault_type.value
            frames.append(df)
            logger.debug("Loaded %s: %d rows → %s", path.name, len(df), fault_type.value)

        combined = pd.concat(frames, ignore_index=True)
        self.validate_schema(combined)
        logger.info("load_all: %d total rows from %d files", len(combined), len(frames))
        return combined

    def load_scenario(self, fault_type: FaultType) -> pd.DataFrame:
        """
        Load all CSV files belonging to one fault scenario.
        Raises DataLoadError if no matching files are found.
        """
        key = self._fault_type_to_key(fault_type)
        paths = [
            p for p in sorted(self._data_dir.glob("*.csv"))
            if key in p.stem.lower()
        ]
        if not paths:
            raise DataLoadError(
                f"No CSV files for fault_type={fault_type.value!r} "
                f"(key={key!r}) in {self._data_dir}"
            )

        frames = [self._read_csv(p) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        df["fault_type"] = fault_type.value
        self.validate_schema(df)
        logger.info(
            "load_scenario(%s): %d rows from %d files",
            fault_type.value, len(df), len(paths),
        )
        return df

    def iter_files(self, chunksize: int | None = None) -> typing.Iterator[pd.DataFrame]:
        """
        Yield one DataFrame per CSV file.
        Memory-efficient alternative to load_all().
        Loading 10818901 data at one time is too much memory overhead
        """
        csv_paths = sorted(self._data_dir.glob("*.csv"))
        if not csv_paths:
            raise DataLoadError(f"No CSV files found in {self._data_dir}")

        for path in csv_paths:
            fault_type = self._infer_fault_type(path.stem)
            df = self._read_csv(path, chunksize)
            df["fault_type"] = fault_type.value
            self.validate_schema(df)
            logger.info("iter_files: yielded %s (%d rows)", path.name, len(df))
            yield df

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise SchemaValidationError(f"Missing columns: {sorted(missing)}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _infer_fault_type(self, stem: str) -> FaultType:
        """Match a filename stem (no extension) to a FaultType via substring lookup."""
        lower = stem.lower()
        for key, fault_type in self.FILENAME_FAULT_MAP.items():
            if key in lower:
                return fault_type
        raise DataLoadError(f"Cannot infer fault type from filename stem: {stem!r}")

    def _fault_type_to_key(self, fault_type: FaultType) -> str:
        """Reverse-lookup: FaultType → filename substring key."""
        for key, ft in self.FILENAME_FAULT_MAP.items():
            if ft == fault_type:
                return key
        raise DataLoadError(f"No filename key registered for fault_type={fault_type.value!r}")

    def _read_csv(self, path: Path, chunksize: int | None = None) -> pd.DataFrame:
        kwargs: dict = {"parse_dates": ["Datetime"]}
        if chunksize:
            return pd.concat(
                pd.read_csv(path, chunksize=chunksize, **kwargs),
                ignore_index=True,
            )
        return pd.read_csv(path, **kwargs)
