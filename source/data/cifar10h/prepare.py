"""CIFAR-10H pair preparation: load predictions, build pair DataFrames, fit pairs."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

import utils
from data.preparation import (
    fit_augmented_beliefs,
    make_cal_eval_masks,
    temperature_scale,
    write_pair_file,
)
from paths import dataset_prepared

from .labels import (
    CLASSES,
    MIN_POSITIVES,
    available_pairs,
    load_groundtruth,
    load_labels,
)

_PREPARED_ROOT = dataset_prepared("cifar10h")

logger = logging.getLogger("prepare")


def load_predictions(path: Path) -> pd.DataFrame:
    """Read a competition-format inference CSV and rename columns for downstream use.

    Expected input columns: image_id, airplane, automobile, bird, cat, deer,
    dog, frog, horse, ship, truck.

    Returns a DataFrame with:
        image_id, b_x_raw_airplane, b_x_raw_automobile, ..., b_x_raw_truck
    """
    df = pd.read_csv(Path(path))
    rename_map = {c: f"b_x_raw_{c}" for c in CLASSES}
    return df.rename(columns=rename_map)


def make_pair_df(
    labels: pd.DataFrame,
    groundtruth: pd.DataFrame,
    class_name: str,
    annotator_id: str,
) -> pd.DataFrame:
    """Construct the (image_id, image_path, h, y) alignment DataFrame for one pair.

    Args:
        labels:       Long-format frame from load_labels() with columns
                      image_id, image_path, annotator_id, chosen_class.
        groundtruth:  Official label frame from load_groundtruth() with columns
                      image_id plus one binary column per class.
        class_name:   One of the 10 CIFAR-10 class names.
        annotator_id: Annotator identifier string.

    Returns:
        DataFrame with columns: image_id, image_path, h, y.

    Raises:
        ValueError: if class_name not in CLASSES.
    """
    if class_name not in CLASSES:
        raise ValueError(f"Unknown class {class_name!r}. Valid: {CLASSES}")

    ann_df = labels[labels["annotator_id"] == annotator_id][
        ["image_id", "image_path", "chosen_class"]
    ].copy()
    ann_df["h"] = (ann_df["chosen_class"] == class_name).astype(int)

    right = groundtruth[["image_id", class_name]].rename(columns={class_name: "y"})
    merged = ann_df.merge(right, on="image_id", how="inner")
    merged["y"] = merged["y"].astype(int)

    return merged[["image_id", "image_path", "h", "y"]].reset_index(drop=True)


def fit_pair(
    class_name: str,
    annotator_id: str,
    labels: pd.DataFrame,
    groundtruth: pd.DataFrame,
    inference_df: pd.DataFrame,
    pairs_dir: Path,
) -> Path | None:
    """Build and write the validated pair artefact for one (class_name, annotator_id).

    Pipeline: align inference rows -> stratified cal/eval split ->
    temperature-scale b_x -> cross-fit b_xh -> write_pair_file.

    Returns the path to the written .npz file, or None if the pair is skipped
    due to insufficient positives or negatives.

    Args:
        class_name:   One of the 10 CIFAR-10 class names.
        annotator_id: Annotator identifier string.
        labels:       Long-format frame from load_labels().
        groundtruth:  Official label frame from load_groundtruth().
        inference_df: Competition-format DataFrame after load_predictions().
                      Columns: image_id, b_x_raw_<class>, ...
        pairs_dir:    Output directory for .npz/.json pair artefacts.
    """
    pair_df = make_pair_df(labels, groundtruth, class_name, annotator_id)

    y = pair_df["y"].values.astype(int)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos < MIN_POSITIVES or n_neg < MIN_POSITIVES:
        logger.warning(
            f"Skipping {class_name}__{annotator_id}: "
            f"only {n_pos} positives and {n_neg} negatives (min={MIN_POSITIVES})"
        )
        return None

    indexed = inference_df.set_index("image_id")
    b_x_raw = indexed.loc[
        pair_df["image_id"].values, f"b_x_raw_{class_name}"
    ].values.astype(np.float64)

    h = pair_df["h"].values.astype(int)

    cal_mask, eval_mask = make_cal_eval_masks(y)
    T_opt, b_x = temperature_scale(b_x_raw, y, cal_mask)
    b_xh = fit_augmented_beliefs(b_x, h, y, eval_mask)

    tag = f"{class_name}__{annotator_id}"
    meta = {
        "n_classes": 2,
        "dataset": "cifar10h",
        "model": Path(pairs_dir).parent.name,
        "task": class_name,
        "annotator_id": annotator_id,
        "extra": {
            "T_opt": round(float(T_opt), 6),
        },
    }
    return write_pair_file(Path(pairs_dir), tag, b_x, b_xh, h, y, meta)


# fmt: off
@click.command()
@click.option("--predictions", required=True, type=click.Path(exists=True, path_type=Path), help="Path to a competition-format predictions CSV.")
@click.option("--limit", type=int, default=None, help="Prepare only the first N pairs (for quick testing).")
# fmt: on
def main(predictions: Path, limit: int | None) -> None:
    """Prepare pair artefacts from a predictions CSV. Writes to prepared/cifar10h/<model>/pairs/."""
    model = predictions.stem
    pairs_dir = _PREPARED_ROOT / model / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    utils.setup_logging(_PREPARED_ROOT / model, "prepare")
    utils.log_run_args(logger.info)

    labels = load_labels()
    groundtruth = load_groundtruth()
    inference_df = load_predictions(predictions)

    pairs = available_pairs()
    if limit is not None:
        pairs = pairs[:limit]

    n = len(pairs)
    n_written = 0
    for i, (class_name, annotator_id) in enumerate(pairs, 1):
        tag = f"{class_name}__{annotator_id}"
        logger.info(f"[{i:3d}/{n}] {tag}")
        result = fit_pair(
            class_name, annotator_id, labels, groundtruth, inference_df, pairs_dir
        )
        if result is not None:
            n_written += 1

    logger.info(f"Done. {n_written}/{n} pair artefact(s) written to {pairs_dir}")
