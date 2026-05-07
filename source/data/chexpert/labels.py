"""CheXpert label loading and pair construction (pandas-only, no torch)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from paths import DATA_RAW

GT_RADS = ("bc1", "bc2", "bc3", "bc5", "bc7")
BENCHMARK_RADS = ("bc4", "bc6", "bc8")
ALL_RADS = GT_RADS + BENCHMARK_RADS

N_STUDIES = 500
CONDITIONS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)

_RAW_LABELS_DIR = DATA_RAW / "cheXpert-test-set-labels" / "radiologists" / "groundtruth"
_RAW_IMAGES_DIR = DATA_RAW / "CheXpert"


def loo_voters(h_rad: str) -> list[str]:
    """GT voters for LOO majority: GT_RADS minus h_rad (4 for GT readers, 5 for benchmark)."""
    return [r for r in GT_RADS if r != h_rad]


def available_pairs(
    subset: Literal["gt", "bm", "all"] = "all",
) -> list[tuple[str, str]]:
    """All (condition, h_rad) pairs for the requested subset.

    "gt"        -> 5 GT readers × 5 conditions = 25 pairs
    "bm"        -> 3 benchmark readers × 5 conditions = 15 pairs
    "all"       -> 8 readers × 5 conditions = 40 pairs (default)
    """
    if subset == "gt":
        rads = GT_RADS
    elif subset == "bm":
        rads = BENCHMARK_RADS
    elif subset == "all":
        rads = ALL_RADS
    else:
        raise ValueError(f"Unknown subset {subset!r}. Valid: 'gt', 'bm', 'all'")
    return [(c, r) for c in CONDITIONS for r in rads]


def load_groundtruth(labels_dir: Path | None = None) -> pd.DataFrame:
    """Load the official majority-of-5 ground-truth CSV.

    Returns a DataFrame with columns: study_id, plus one column per condition.
    """
    ldir = Path(labels_dir) if labels_dir is not None else _RAW_LABELS_DIR
    gt_path = ldir.parents[1] / "groundtruth.csv"

    df = pd.read_csv(gt_path).rename(columns={"Study": "study_id"})
    return df.reset_index(drop=True)


def _load_rad_csv(path: Path, rad: str) -> pd.DataFrame:
    """Read a single radiologist CSV, validate condition columns, and rename them."""
    df = pd.read_csv(path).rename(columns={"Study": "study_id"})
    missing = set(CONDITIONS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing condition columns {missing}")
    return df[["study_id"] + list(CONDITIONS)].rename(  # type: ignore[call-overload]
        columns={c: f"{rad}_{c}" for c in CONDITIONS}
    )


def load_labels(labels_dir: Path | None = None) -> pd.DataFrame:
    """Load and merge per-radiologist CSVs (GT + benchmark readers).

    Returns a DataFrame with columns: study_id, image_path, {rad}_{condition}, ...
    for all 8 readers.
    """
    ldir = Path(labels_dir) if labels_dir is not None else _RAW_LABELS_DIR
    benchmark_dir = ldir.parent / "benchmark_radiologists"

    per_rad: list[pd.DataFrame] = []

    # GT readers: files are {rad}_gt.csv inside ldir
    for rad in GT_RADS:
        per_rad.append(_load_rad_csv(ldir / f"{rad}_gt.csv", rad))

    # Benchmark readers: files are {rad}.csv inside benchmark_dir
    for rad in BENCHMARK_RADS:
        per_rad.append(_load_rad_csv(benchmark_dir / f"{rad}.csv", rad))

    merged = per_rad[0]
    for df in per_rad[1:]:
        merged = merged.merge(df, on="study_id", how="inner")

    merged["image_path"] = merged["study_id"].map(_image_path)
    return merged.reset_index(drop=True)


def _image_path(study_id: str) -> str:
    """Resolve the frontal image path for a study_id (format: 'CheXpert-v1.0/test/patientXXXX/studyY')."""
    rel = Path(study_id).relative_to("CheXpert-v1.0")
    return str(_RAW_IMAGES_DIR / rel / "view1_frontal.jpg")
