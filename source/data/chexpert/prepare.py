"""CheXpert consumer + orchestrator: load predictions, build pair DataFrames, fit pairs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

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

from .labels import ALL_RADS, CONDITIONS, available_pairs, load_groundtruth, load_labels

_PREPARED_ROOT = dataset_prepared("chexpert")

logger = logging.getLogger("prepare")


def load_predictions(path: Path) -> pd.DataFrame:
    """Read a competition-format inference CSV and rename columns for downstream use.

    Expected input columns: Study, Atelectasis, Cardiomegaly, Consolidation,
    Edema, Pleural Effusion.

    Returns a DataFrame with:
        study_id, b_x_raw_Atelectasis, b_x_raw_Cardiomegaly,
        b_x_raw_Consolidation, b_x_raw_Edema, b_x_raw_Pleural Effusion
    """
    df = pd.read_csv(Path(path))
    rename_map: dict[str, str] = {"Study": "study_id"}
    rename_map.update({c: f"b_x_raw_{c}" for c in CONDITIONS})
    return df.rename(columns=rename_map)


def make_pair_df(
    labels: pd.DataFrame,
    groundtruth: pd.DataFrame,
    condition: str,
    h_rad: str,
) -> pd.DataFrame:
    """Construct the (study_id, image_path, h, y) alignment DataFrame for one pair.

    Args:
        labels:      8-reader merged frame from load_labels() with columns study_id, image_path, {rad}_{condition} for all readers.
        groundtruth: Official majority-of-5 frame from load_groundtruth() with columns study_id + one column per condition.
        condition:   One of the 5 CheXpert conditions.
        h_rad:       Radiologist identifier (one of ALL_RADS).

    Returns:
        DataFrame with columns: study_id, image_path, h, y.

    Raises:
        ValueError: if condition not in CONDITIONS, or h_rad not in ALL_RADS.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}. Valid: {CONDITIONS}")
    if h_rad not in ALL_RADS:
        raise ValueError(f"Unknown radiologist {h_rad!r}. Valid: {ALL_RADS}")

    h_col = f"{h_rad}_{condition}"
    left = labels[["study_id", "image_path", h_col]]
    right = groundtruth[["study_id", condition]]

    merged = left.merge(right, on="study_id", how="inner")
    merged = merged.rename(columns={h_col: "h", condition: "y"})
    merged["h"] = merged["h"].astype(int)
    merged["y"] = merged["y"].astype(int)
    return merged.reset_index(drop=True)


def fit_pair(
    condition: str,
    h_rad: str,
    labels: pd.DataFrame,
    groundtruth: pd.DataFrame,
    inference_df: pd.DataFrame,
    pairs_dir: Path,
) -> Path:
    """Build and write the validated pair artefact for one (condition, h_rad).

    Pipeline: align inference rows -> stratified cal/eval split ->
    temperature-scale b_x -> cross-fit b_xh -> write_pair_file.

    Args:
        condition:    One of the 5 CheXpert conditions.
        h_rad:        Radiologist identifier (one of ALL_RADS).
        labels:       8-reader merged frame from load_labels().
        groundtruth:  Official majority-of-5 frame from load_groundtruth().
        inference_df: Competition-format DataFrame after load_predictions(). columns: study_id, b_x_raw_{condition}, ...
        pairs_dir:    Output directory for .npz/.json pair artefacts.

    Returns:
        Path to the written .npz file.
    """
    pair_df = make_pair_df(labels, groundtruth, condition, h_rad)

    inference_df = inference_df.set_index("study_id")
    b_x_raw = inference_df.loc[
        pair_df["study_id"].values, f"b_x_raw_{condition}"
    ].values.astype(np.float32)

    y = pair_df["y"].values.astype(int)
    h = pair_df["h"].values.astype(int)

    cal_mask, eval_mask = make_cal_eval_masks(y)
    T_opt, b_x = temperature_scale(b_x_raw, y, cal_mask)
    b_xh = fit_augmented_beliefs(b_x, h, y, eval_mask)

    tag = f"{condition.replace(' ', '_')}__{h_rad}"
    meta = {
        "n_classes": 2,
        "dataset": "chexpert",
        "model": Path(pairs_dir).parent.name,
        "task": condition,
        "annotator_id": h_rad,
        "extra": {
            "T_opt": round(float(T_opt), 6),
        },
    }
    return write_pair_file(Path(pairs_dir), tag, b_x, b_xh, h, y, meta)


# fmt: off
@click.command()
@click.option("--predictions", required=True, type=click.Path(exists=True, path_type=Path), help="Path to a competition-format predictions CSV.")
@click.option("--readers", type=click.Choice(["gt", "benchmark", "all"]), default="all", show_default=True, help="Which reader subset to produce pairs for.")
@click.option("--limit", type=int, default=None, help="Prepare only the first N pairs (for quick testing).")
# fmt: on
def main(
    predictions: Path, readers: Literal["gt", "benchmark", "all"], limit: int | None
) -> None:
    """Prepare pair artefacts from a predictions CSV. Writes to prepared/chexpert/<model>/pairs/."""
    model = predictions.stem
    pairs_dir = _PREPARED_ROOT / model / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    utils.setup_logging(_PREPARED_ROOT / model, "prepare")
    utils.log_run_args(logger.info)

    labels = load_labels()
    groundtruth = load_groundtruth()
    inference_df = load_predictions(predictions)

    pairs = available_pairs(subset=readers)
    if limit is not None:
        pairs = pairs[:limit]

    n = len(pairs)
    for i, (condition, h_rad) in enumerate(pairs, 1):
        tag = f"{condition.replace(' ', '_')}__{h_rad}"
        logger.info(f"[{i:2d}/{n}] {tag}")
        fit_pair(condition, h_rad, labels, groundtruth, inference_df, pairs_dir)

    logger.info(f"Done. {n} pair artefact(s) written to {pairs_dir}")
