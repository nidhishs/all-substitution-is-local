"""CIFAR-10H label loading and pair construction (pandas-only, no torch)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from paths import DATA_RAW

CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

N_TEST: int = 10_000
TOP_N_ANNOTATORS: int = 50
MIN_POSITIVES: int = 10

_RAW_DIR = DATA_RAW / "cifar10h"
_RAW_LABELS_CSV = _RAW_DIR / "cifar10h-raw.csv"

# Column name in cifar10h-raw.csv for the test image index.
_IDX_COL = "cifar10_test_test_idx"


def select_top_annotators(
    raw_df: pd.DataFrame, n: int = TOP_N_ANNOTATORS
) -> tuple[str, ...]:
    """Return the n annotators with the most labels, sorted descending by count."""
    counts = raw_df.groupby("annotator_id").size().sort_values(ascending=False)
    return tuple(counts.head(n).index.astype(str).tolist())


def _load_raw(raw_csv: Path | None = None) -> pd.DataFrame:
    path = Path(raw_csv) if raw_csv is not None else _RAW_LABELS_CSV
    df = pd.read_csv(path)
    if "is_attn_check" in df.columns:
        df = df[df["is_attn_check"] == 0]
    return df


def available_pairs(
    subset: Literal["all"] = "all",
    raw_csv: Path | None = None,
) -> list[tuple[str, str]]:
    """All (class_name, annotator_id) pairs.

    "all" -> 10 classes × top-50 annotators = 500 pairs (default).

    Raises:
        ValueError: if subset is not "all".
    """
    if subset != "all":
        raise ValueError(f"Unknown subset {subset!r}. Valid: 'all'")
    raw = _load_raw(raw_csv)
    annotators = select_top_annotators(raw)
    return [(c, a) for c in CLASSES for a in annotators]


def load_groundtruth(raw_csv: Path | None = None) -> pd.DataFrame:
    """Load the CIFAR-10 official test labels from the CIFAR-10H raw CSV.

    Returns a DataFrame with columns:
        image_id, airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

    Each class column is a binary (0/1) indicator: 1 iff the true class is that class.
    """
    raw = _load_raw(raw_csv)

    gt = (
        raw[[_IDX_COL, "true_label"]]
        .drop_duplicates(_IDX_COL)
        .rename(columns={_IDX_COL: "image_id"})
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    for i, cls in enumerate(CLASSES):
        gt[cls] = (gt["true_label"] == i).astype(int)

    return gt.drop(columns=["true_label"])


def load_labels(raw_csv: Path | None = None) -> pd.DataFrame:
    """Load per-annotator labels for the top-N most prolific annotators.

    Returns a long-format DataFrame with columns:
        image_id, image_path, annotator_id, chosen_class

    One row per (image, annotator) annotation, restricted to top-N annotators.
    `image_path` is a virtual handle "cifar10:<image_id>" — inference reads via
    torchvision.datasets.CIFAR10, not from individual image files.
    """
    raw = _load_raw(raw_csv)
    annotators = select_top_annotators(raw)
    annotator_set = set(annotators)

    df = raw[raw["annotator_id"].astype(str).isin(annotator_set)].copy()
    df["annotator_id"] = df["annotator_id"].astype(str)
    df["chosen_class"] = df["chosen_label"].apply(lambda i: CLASSES[int(i)])
    df["image_path"] = df[_IDX_COL].apply(lambda i: f"cifar10:{int(i)}")
    df = df.rename(columns={_IDX_COL: "image_id"})

    return df[["image_id", "image_path", "annotator_id", "chosen_class"]].reset_index(
        drop=True
    )
