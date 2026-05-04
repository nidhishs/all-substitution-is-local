"""Tests for data/chexpert: label loading, pair construction, and pipeline."""

import json

import numpy as np
import pandas as pd
import pytest
from data.chexpert.labels import (
    ALL_RADS,
    BENCHMARK_RADS,
    CONDITIONS,
    GT_RADS,
    N_STUDIES,
    available_pairs,
    load_groundtruth,
    load_labels,
)
from data.chexpert.prepare import fit_pair, load_predictions, make_pair_df
from utils import validate_pair_array

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def labels_dir(tmp_path_factory):
    """Synthetic chexpert_data/ directory tree with CSVs for all 8 readers."""
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("chexpert_data")
    study_ids = [
        f"CheXpert-v1.0/test/patient{64741 + i}/study1" for i in range(N_STUDIES)
    ]

    # Official groundtruth.csv
    gt_rows = [{"Study": sid} for sid in study_ids]
    for row in gt_rows:
        for c in CONDITIONS:
            row[c] = int(rng.integers(0, 2))
    pd.DataFrame(gt_rows).to_csv(root / "groundtruth.csv", index=False)

    # GT reader CSVs: root/radiologists/groundtruth/{rad}_gt.csv
    gt_rads_dir = root / "radiologists" / "groundtruth"
    gt_rads_dir.mkdir(parents=True)
    for rad in GT_RADS:
        rows = [{"Study": sid} for sid in study_ids]
        for row in rows:
            for c in CONDITIONS:
                row[c] = int(rng.integers(0, 2))
        pd.DataFrame(rows).to_csv(gt_rads_dir / f"{rad}_gt.csv", index=False)

    # Benchmark reader CSVs: root/radiologists/benchmark_radiologists/{rad}.csv
    bench_dir = root / "radiologists" / "benchmark_radiologists"
    bench_dir.mkdir(parents=True)
    for rad in BENCHMARK_RADS:
        rows = [{"Study": sid} for sid in study_ids]
        for row in rows:
            for c in CONDITIONS:
                row[c] = int(rng.integers(0, 2))
        pd.DataFrame(rows).to_csv(bench_dir / f"{rad}.csv", index=False)

    return gt_rads_dir  # same semantics as before: labels_dir.parents[1] = root


@pytest.fixture(scope="module")
def labels(labels_dir):
    return load_labels(labels_dir)


@pytest.fixture(scope="module")
def groundtruth(labels_dir):
    return load_groundtruth(labels_dir)


@pytest.fixture(scope="module")
def synthetic_inference_df(labels):
    """Synthetic inference DataFrame matching label study_id order."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"study_id": labels["study_id"]})
    for c in CONDITIONS:
        df[f"b_x_raw_{c}"] = rng.uniform(0.05, 0.95, size=N_STUDIES).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# load_labels
# ---------------------------------------------------------------------------


def test_load_labels_returns_correct_study_count(labels):
    assert len(labels) == N_STUDIES


def test_load_labels_has_expected_columns(labels):
    for rad in ALL_RADS:
        for cond in CONDITIONS:
            assert f"{rad}_{cond}" in labels.columns


def test_load_labels_has_image_path_column(labels):
    assert "image_path" in labels.columns
    assert labels["image_path"].iloc[0].endswith("view1_frontal.jpg")


def test_load_labels_image_path_prefix_check(labels):
    # All image_path values should be strings (no NaN, no failures).
    assert labels["image_path"].notna().all()


def test_load_labels_raises_on_missing_csv(tmp_path):
    gt_dir = tmp_path / "radiologists" / "groundtruth"
    gt_dir.mkdir(parents=True)
    bench_dir = tmp_path / "radiologists" / "benchmark_radiologists"
    bench_dir.mkdir(parents=True)
    # Only one GT CSV — bc2 and others missing.
    pd.DataFrame(
        {
            "Study": ["CheXpert-v1.0/test/p1/s1"],
            "Atelectasis": [0],
            "Cardiomegaly": [0],
            "Consolidation": [0],
            "Edema": [0],
            "Pleural Effusion": [0],
        }
    ).to_csv(gt_dir / "bc1_gt.csv", index=False)
    with pytest.raises(FileNotFoundError, match="bc2"):
        load_labels(gt_dir)


def test_load_labels_raises_on_missing_condition_column(tmp_path):
    gt_dir = tmp_path / "radiologists" / "groundtruth"
    gt_dir.mkdir(parents=True)
    bench_dir = tmp_path / "radiologists" / "benchmark_radiologists"
    bench_dir.mkdir(parents=True)
    for rad in GT_RADS:
        # Missing 'Edema' column.
        pd.DataFrame(
            {
                "Study": [f"CheXpert-v1.0/test/p{i}/s1" for i in range(N_STUDIES)],
                "Atelectasis": [0] * N_STUDIES,
                "Cardiomegaly": [0] * N_STUDIES,
                "Consolidation": [0] * N_STUDIES,
                "Pleural Effusion": [0] * N_STUDIES,
            }
        ).to_csv(gt_dir / f"{rad}_gt.csv", index=False)
    with pytest.raises(ValueError, match="missing condition"):
        load_labels(gt_dir)


# ---------------------------------------------------------------------------
# load_groundtruth
# ---------------------------------------------------------------------------


def test_load_groundtruth(groundtruth):
    assert len(groundtruth) == N_STUDIES
    assert "study_id" in groundtruth.columns
    for cond in CONDITIONS:
        assert cond in groundtruth.columns


# ---------------------------------------------------------------------------
# available_pairs
# ---------------------------------------------------------------------------


def test_available_pairs_count():
    assert len(available_pairs()) == len(CONDITIONS) * len(ALL_RADS)  # 40
    assert len(available_pairs(subset="gt")) == len(CONDITIONS) * len(GT_RADS)  # 25
    assert len(available_pairs(subset="benchmark")) == len(CONDITIONS) * len(
        BENCHMARK_RADS
    )  # 15


def test_available_pairs_all_conditions_and_readers():
    pairs = available_pairs()
    pair_conditions = {c for c, _ in pairs}
    pair_readers = {r for _, r in pairs}
    assert pair_conditions == set(CONDITIONS)
    assert pair_readers == set(ALL_RADS)


def test_available_pairs_rejects_unknown_subset():
    with pytest.raises(ValueError, match="Unknown subset"):
        available_pairs(subset="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# make_pair_df
# ---------------------------------------------------------------------------


def test_make_pair_df_shape(labels, groundtruth):
    df = make_pair_df(labels, groundtruth, "Cardiomegaly", "bc1")
    assert len(df) == N_STUDIES
    assert set(df.columns) >= {"study_id", "image_path", "h", "y"}


def test_make_pair_df_y_from_groundtruth(labels, groundtruth):
    """y must be taken verbatim from groundtruth regardless of which reader is held out."""
    df = make_pair_df(labels, groundtruth, "Cardiomegaly", "bc1")
    # Align groundtruth to the same row order as df
    gt_aligned = (
        groundtruth.set_index("study_id").loc[df["study_id"], "Cardiomegaly"].values
    )
    assert list(df["y"]) == list(gt_aligned.astype(int))


def test_make_pair_df_rejects_unknown_condition(labels, groundtruth):
    with pytest.raises(ValueError, match="Unknown condition"):
        make_pair_df(labels, groundtruth, "Pneumonia", "bc1")


def test_make_pair_df_rejects_unknown_reader(labels, groundtruth):
    with pytest.raises(ValueError, match="Unknown radiologist"):
        make_pair_df(labels, groundtruth, "Cardiomegaly", "bc99")


def test_make_pair_df_accepts_benchmark_reader(labels, groundtruth):
    df = make_pair_df(labels, groundtruth, "Cardiomegaly", "bc4")
    assert len(df) == N_STUDIES
    assert set(df.columns) >= {"study_id", "image_path", "h", "y"}


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------


def test_load_predictions(tmp_path):
    # Write a tiny competition-format CSV
    csv_data = {
        "Study": ["CheXpert-v1.0/test/p1/s1", "CheXpert-v1.0/test/p2/s1"],
        "Atelectasis": [0.1, 0.2],
        "Cardiomegaly": [0.3, 0.4],
        "Consolidation": [0.5, 0.6],
        "Edema": [0.7, 0.8],
        "Pleural Effusion": [0.9, 0.95],
    }
    csv_path = tmp_path / "test_preds.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)

    df = load_predictions(csv_path)
    assert "study_id" in df.columns
    assert "Study" not in df.columns
    for c in CONDITIONS:
        assert f"b_x_raw_{c}" in df.columns


# ---------------------------------------------------------------------------
# fit_pair — end-to-end with stub inference (no DenseNet)
# ---------------------------------------------------------------------------


def test_fit_pair_produces_valid_artefact(
    tmp_path, labels, groundtruth, synthetic_inference_df
):
    npz = fit_pair(
        condition="Cardiomegaly",
        h_rad="bc1",
        labels=labels,
        groundtruth=groundtruth,
        inference_df=synthetic_inference_df,
        pairs_dir=tmp_path / "pairs",
    )
    assert npz.exists()
    assert npz.with_suffix(".json").exists()
    validate_pair_array(npz, npz.with_suffix(".json"))


def test_fit_pair_metadata_contains_required_keys(
    tmp_path, labels, groundtruth, synthetic_inference_df
):
    pairs_dir = tmp_path / "drnet_predictions" / "pairs"
    npz = fit_pair(
        condition="Atelectasis",
        h_rad="bc2",
        labels=labels,
        groundtruth=groundtruth,
        inference_df=synthetic_inference_df,
        pairs_dir=pairs_dir,
    )
    meta = json.loads(npz.with_suffix(".json").read_text())
    assert meta["n_classes"] == 2
    assert meta["dataset"] == "chexpert"
    assert meta["task"] == "Atelectasis"
    assert meta["annotator_id"] == "bc2"
    assert "T_opt" in meta["extra"]
    assert meta["model"] == "drnet_predictions"  # pairs_dir.parent.name
