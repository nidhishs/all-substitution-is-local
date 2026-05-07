"""Tests for data/cifar10h: label loading, pair construction, and pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from data.cifar10h.labels import (
    CLASSES,
    MIN_POSITIVES,
    N_TEST,
    available_pairs,
    load_groundtruth,
    load_labels,
    select_top_annotators,
)
from data.cifar10h.prepare import fit_pair, load_predictions, make_pair_df
from utils import validate_pair_array

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_IMAGES = 200  # small synthetic dataset
_N_ANNOTATORS = 5
_LABELS_PER_ANN = 80  # each annotator labels this many images


@pytest.fixture(scope="module")
def raw_csv(tmp_path_factory):
    """Synthetic cifar10h-raw.csv with 5 annotators × 80 labels each."""
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("cifar10h_data")

    rows = []
    for ann_idx in range(_N_ANNOTATORS):
        annotator_id = f"ann_{1000 + ann_idx}"
        image_indices = rng.choice(_N_IMAGES, size=_LABELS_PER_ANN, replace=False)
        true_labels = rng.integers(0, 10, size=_N_IMAGES)  # one per image slot
        for img_idx in image_indices:
            chosen = int(rng.integers(0, 10))
            rows.append(
                {
                    "cifar10_test_test_idx": int(img_idx),
                    "chosen_label": chosen,
                    "true_label": int(true_labels[img_idx]),
                    "annotator_id": annotator_id,
                    "reaction_time_ms": int(rng.integers(200, 2000)),
                }
            )

    csv_path = root / "cifar10h-raw.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def annotators(raw_csv):
    raw = pd.read_csv(raw_csv)
    return select_top_annotators(raw, n=_N_ANNOTATORS)


@pytest.fixture(scope="module")
def labels(raw_csv):
    return load_labels(raw_csv)


@pytest.fixture(scope="module")
def groundtruth(raw_csv):
    return load_groundtruth(raw_csv)


@pytest.fixture(scope="module")
def synthetic_inference_df(groundtruth):
    """Synthetic competition-format predictions DataFrame."""
    rng = np.random.default_rng(1)
    n = len(groundtruth)
    df = pd.DataFrame({"image_id": groundtruth["image_id"].values})
    for cls in CLASSES:
        df[f"b_x_raw_{cls}"] = rng.uniform(0.05, 0.95, size=n).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# select_top_annotators
# ---------------------------------------------------------------------------


def test_select_top_annotators_returns_n(raw_csv):
    raw = pd.read_csv(raw_csv)
    result = select_top_annotators(raw, n=3)
    assert len(result) == 3
    assert isinstance(result, tuple)
    assert all(isinstance(a, str) for a in result)


def test_select_top_annotators_sorted_descending(raw_csv):
    raw = pd.read_csv(raw_csv)
    result = select_top_annotators(raw, n=_N_ANNOTATORS)
    counts = raw.groupby("annotator_id").size()
    count_list = [counts[a] for a in result]
    assert count_list == sorted(count_list, reverse=True)


# ---------------------------------------------------------------------------
# load_groundtruth
# ---------------------------------------------------------------------------


def test_load_groundtruth_columns(groundtruth):
    assert "image_id" in groundtruth.columns
    for cls in CLASSES:
        assert cls in groundtruth.columns


def test_load_groundtruth_binary_values(groundtruth):
    for cls in CLASSES:
        assert set(groundtruth[cls].unique()).issubset({0, 1})


def test_load_groundtruth_unique_image_ids(groundtruth):
    assert groundtruth["image_id"].is_unique


def test_load_groundtruth_one_hot_per_row(groundtruth):
    total = groundtruth[[*CLASSES]].sum(axis=1)
    assert (total == 1).all(), "each image should be assigned exactly one true class"


# ---------------------------------------------------------------------------
# load_labels
# ---------------------------------------------------------------------------


def test_load_labels_columns(labels):
    assert set(labels.columns) >= {
        "image_id",
        "image_path",
        "annotator_id",
        "chosen_class",
    }


def test_load_labels_chosen_class_valid(labels):
    assert set(labels["chosen_class"].unique()).issubset(set(CLASSES))


def test_load_labels_image_path_format(labels):
    assert labels["image_path"].str.startswith("cifar10:").all()


def test_load_labels_restricted_to_top_annotators(raw_csv, labels):
    raw = pd.read_csv(raw_csv)
    expected = set(select_top_annotators(raw, n=_N_ANNOTATORS))
    actual = set(labels["annotator_id"].unique())
    assert actual.issubset(expected)


# ---------------------------------------------------------------------------
# available_pairs
# ---------------------------------------------------------------------------


def test_available_pairs_count(raw_csv):
    pairs = available_pairs(raw_csv=raw_csv)
    assert len(pairs) == len(CLASSES) * _N_ANNOTATORS


def test_available_pairs_all_classes_represented(raw_csv):
    pairs = available_pairs(raw_csv=raw_csv)
    pair_classes = {c for c, _ in pairs}
    assert pair_classes == set(CLASSES)


def test_available_pairs_rejects_unknown_subset(raw_csv):
    with pytest.raises(ValueError, match="Unknown subset"):
        available_pairs(subset="gt", raw_csv=raw_csv)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------


def test_load_predictions(tmp_path):
    csv_data = {"image_id": [0, 1, 2]}
    for cls in CLASSES:
        csv_data[cls] = [0.1] * 3
    csv_path = tmp_path / "test_preds.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)

    df = load_predictions(csv_path)
    assert "image_id" in df.columns
    for cls in CLASSES:
        assert f"b_x_raw_{cls}" not in df or f"b_x_raw_{cls}" in df.columns
        assert f"b_x_raw_{cls}" in df.columns


# ---------------------------------------------------------------------------
# make_pair_df
# ---------------------------------------------------------------------------


def test_make_pair_df_columns(labels, groundtruth, annotators):
    ann = annotators[0]
    df = make_pair_df(labels, groundtruth, CLASSES[0], ann)
    assert set(df.columns) >= {"image_id", "image_path", "h", "y"}


def test_make_pair_df_binary_h_y(labels, groundtruth, annotators):
    ann = annotators[0]
    df = make_pair_df(labels, groundtruth, CLASSES[0], ann)
    assert set(df["h"].unique()).issubset({0, 1})
    assert set(df["y"].unique()).issubset({0, 1})


def test_make_pair_df_h_reflects_annotator_choice(labels, groundtruth, annotators):
    """h == 1 iff the annotator chose that class."""
    ann = annotators[0]
    cls = CLASSES[1]
    df = make_pair_df(labels, groundtruth, cls, ann)
    ann_rows = labels[labels["annotator_id"] == ann].set_index("image_id")
    for _, row in df.iterrows():
        expected_h = int(ann_rows.loc[row["image_id"], "chosen_class"] == cls)
        assert row["h"] == expected_h


def test_make_pair_df_rejects_unknown_class(labels, groundtruth, annotators):
    with pytest.raises(ValueError, match="Unknown class"):
        make_pair_df(labels, groundtruth, "spaceship", annotators[0])


def test_make_pair_df_unknown_annotator_returns_empty(labels, groundtruth):
    df = make_pair_df(labels, groundtruth, CLASSES[0], "nonexistent_annotator_xyz")
    assert len(df) == 0


# ---------------------------------------------------------------------------
# fit_pair — end-to-end with stub inference (no real model)
# ---------------------------------------------------------------------------


def _make_dense_inference_df(groundtruth):
    """Synthetic predictions with balanced probs so both classes have enough support."""
    rng = np.random.default_rng(42)
    n = len(groundtruth)
    df = pd.DataFrame({"image_id": groundtruth["image_id"].values})
    for cls in CLASSES:
        p = rng.uniform(0.1, 0.9, size=n).astype(np.float32)
        df[f"b_x_raw_{cls}"] = p
    return df


def test_fit_pair_produces_valid_artefact(tmp_path, labels, groundtruth, annotators):
    inference_df = _make_dense_inference_df(groundtruth)
    ann = annotators[0]

    # Find a class + annotator combo with enough positives
    for cls in CLASSES:
        pair_df = make_pair_df(labels, groundtruth, cls, ann)
        if (
            pair_df["y"].sum() >= MIN_POSITIVES
            and (1 - pair_df["y"].values).sum() >= MIN_POSITIVES
        ):
            break
    else:
        pytest.skip(
            "No (class, annotator) pair with sufficient positives in synthetic data"
        )

    npz = fit_pair(
        class_name=cls,
        annotator_id=ann,
        labels=labels,
        groundtruth=groundtruth,
        inference_df=inference_df,
        pairs_dir=tmp_path / "pairs",
    )
    assert npz is not None
    assert npz.exists()
    assert npz.with_suffix(".json").exists()
    validate_pair_array(npz, npz.with_suffix(".json"))


def test_fit_pair_metadata_keys(tmp_path, labels, groundtruth, annotators):
    inference_df = _make_dense_inference_df(groundtruth)
    pairs_dir = tmp_path / "my_model" / "pairs"
    ann = annotators[0]

    for cls in CLASSES:
        pair_df = make_pair_df(labels, groundtruth, cls, ann)
        if (
            pair_df["y"].sum() >= MIN_POSITIVES
            and (1 - pair_df["y"].values).sum() >= MIN_POSITIVES
        ):
            break
    else:
        pytest.skip("No pair with sufficient positives in synthetic data")

    npz = fit_pair(
        class_name=cls,
        annotator_id=ann,
        labels=labels,
        groundtruth=groundtruth,
        inference_df=inference_df,
        pairs_dir=pairs_dir,
    )
    assert npz is not None
    meta = json.loads(npz.with_suffix(".json").read_text())
    assert meta["n_classes"] == 2
    assert meta["dataset"] == "cifar10h"
    assert meta["task"] == cls
    assert meta["annotator_id"] == ann
    assert "T_opt" in meta["extra"]
    assert meta["model"] == "my_model"


def test_fit_pair_skips_insufficient_positives(
    tmp_path, labels, groundtruth, annotators
):
    """fit_pair should return None when a class has fewer than MIN_POSITIVES positives."""
    inference_df = _make_dense_inference_df(groundtruth)
    ann = annotators[0]

    # Force a class where the annotator has no positives: 'airplane' labeled by an annotator
    # whose labels are all class 0 (airplane). We mock labels to contain h=1 for all -> y=1 for all
    # -> no negatives => should skip.
    mock_labels = labels[labels["annotator_id"] == ann].copy()
    # Replace all chosen_class with a rare class to make positives < MIN_POSITIVES
    mock_labels["chosen_class"] = "truck"
    # Also make all groundtruth class airplane have y=0 (so no positives)
    mock_gt = groundtruth.copy()
    mock_gt["airplane"] = 0  # no positives for airplane

    result = fit_pair(
        class_name="airplane",
        annotator_id=ann,
        labels=mock_labels,
        groundtruth=mock_gt,
        inference_df=inference_df,
        pairs_dir=tmp_path / "pairs",
    )
    assert result is None
