"""Tests for projected Parquet cache reads."""
from __future__ import annotations

import pytest

from hvac_fdd.ingestion.parquet import _quote_identifier, _validate_columns


def test_quote_identifier_preserves_column_name() -> None:
    assert _quote_identifier("event_time") == '"event_time"'


def test_validate_columns_rejects_empty_projection() -> None:
    with pytest.raises(ValueError, match="At least one"):
        _validate_columns([])


def test_validate_columns_rejects_unsafe_identifier() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        _validate_columns(['event_time"; DROP TABLE detections;--'])
