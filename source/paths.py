"""Canonical project paths. Single source of truth for raw/prepared/results dirs."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # = source/
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PREPARED = PROJECT_ROOT / "data" / "prepared"
RESULTS = PROJECT_ROOT / "results"


def dataset_prepared(name: str) -> Path:
    return DATA_PREPARED / name


def experiment_results(name: str) -> Path:
    return RESULTS / name
