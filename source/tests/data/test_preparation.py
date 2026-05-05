"""Unit tests for data/preparation.py — dataset-agnostic pair-prep helpers."""

import json

import numpy as np
import pytest

from data.preparation import (
    CAL_FRAC,
    fit_augmented_beliefs,
    make_cal_eval_masks,
    temperature_scale,
    write_pair_file,
)
from utils import validate_pair_array


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# make_cal_eval_masks
# ---------------------------------------------------------------------------


def test_make_cal_eval_masks_complementary(rng):
    N = 200
    y = rng.integers(0, 2, size=N)
    cal, ev = make_cal_eval_masks(y)
    assert cal.shape == (N,)
    assert ev.shape == (N,)
    assert np.all(cal ^ ev)  # exactly one True per row


def test_make_cal_eval_masks_fraction(rng):
    N = 400
    y = rng.integers(0, 2, size=N)
    cal, _ = make_cal_eval_masks(y)
    actual_frac = cal.mean()
    # allow ±5% of the target fraction
    assert abs(actual_frac - CAL_FRAC) < 0.05


def test_make_cal_eval_masks_both_classes_in_both_splits():
    y = np.array([0, 1] * 100)
    cal, ev = make_cal_eval_masks(y)
    for yval in (0, 1):
        mask = y == yval
        assert cal[mask].any() and ev[mask].any()


# ---------------------------------------------------------------------------
# temperature_scale
# ---------------------------------------------------------------------------


def test_temperature_scale_shape(rng):
    N = 150
    b_x_raw = rng.uniform(0.01, 0.99, size=N)
    y = rng.integers(0, 2, size=N)
    cal_mask = np.zeros(N, dtype=bool)
    cal_mask[: int(N * CAL_FRAC)] = True
    T_opt, b_x = temperature_scale(b_x_raw, y, cal_mask)
    assert isinstance(T_opt, float)
    assert T_opt > 0.0
    assert b_x.shape == (N,)
    assert np.all((b_x > 0) & (b_x < 1))


def test_temperature_scale_unit_temperature():
    # When model is already well-calibrated, T_opt should be near 1.
    # Build a dataset where T=1 is the true generative temperature:
    # draw logits, compute p = sigmoid(logit), sample labels from Bernoulli(p).
    rng2 = np.random.default_rng(7)
    N = 500
    logits = rng2.normal(0, 1.0, size=N)
    p_true = 1.0 / (1.0 + np.exp(-logits))
    y = rng2.binomial(1, p_true).astype(int)
    b_x_raw = np.clip(p_true, 0.05, 0.95)
    cal_mask = np.zeros(N, dtype=bool)
    cal_mask[: int(N * CAL_FRAC)] = True
    T_opt, _ = temperature_scale(b_x_raw, y, cal_mask)
    assert T_opt == pytest.approx(1.0, abs=0.3)


# ---------------------------------------------------------------------------
# fit_augmented_beliefs
# ---------------------------------------------------------------------------


def test_fit_augmented_beliefs_shape(rng):
    N = 200
    b_x = rng.uniform(0.1, 0.9, size=N)
    y = rng.integers(0, 2, size=N)
    h = rng.integers(0, 2, size=N)
    _, eval_mask = make_cal_eval_masks(y)
    b_xh = fit_augmented_beliefs(b_x, h, y, eval_mask)
    assert b_xh.shape == (N,)
    assert not np.isnan(b_xh[eval_mask]).any()
    assert np.isnan(b_xh[~eval_mask]).all()


def test_fit_augmented_beliefs_range(rng):
    N = 300
    b_x = rng.uniform(0.1, 0.9, size=N)
    y = rng.integers(0, 2, size=N)
    h = rng.integers(0, 2, size=N)
    _, eval_mask = make_cal_eval_masks(y)
    b_xh = fit_augmented_beliefs(b_x, h, y, eval_mask)
    assert np.all((b_xh[eval_mask] > 0) & (b_xh[eval_mask] < 1))


# ---------------------------------------------------------------------------
# write_pair_file
# ---------------------------------------------------------------------------


_BASE_META = {
    "n_classes": 2,
    "dataset": "test",
    "model": "m",
    "task": "Foo",
    "annotator_id": "r1",
}


def test_write_pair_file_passes_validation(tmp_path, rng):
    N = 200
    b_x = rng.uniform(0.1, 0.9, size=N)
    y = rng.integers(0, 2, size=N)
    h = rng.integers(0, 2, size=N)
    _, eval_mask = make_cal_eval_masks(y)
    b_xh = fit_augmented_beliefs(b_x, h, y, eval_mask)

    npz = write_pair_file(tmp_path, "Foo__r1", b_x, b_xh, h, y, _BASE_META)

    assert npz.exists()
    assert npz.with_suffix(".json").exists()
    validate_pair_array(npz, npz.with_suffix(".json"))  # must not raise


def test_write_pair_file_metadata_keys(tmp_path, rng):
    N = 200
    b_x = rng.uniform(0.1, 0.9, size=N)
    y = rng.integers(0, 2, size=N)
    h = rng.integers(0, 2, size=N)
    _, eval_mask = make_cal_eval_masks(y)
    b_xh = fit_augmented_beliefs(b_x, h, y, eval_mask)

    write_pair_file(tmp_path, "tag", b_x, b_xh, h, y, _BASE_META)

    written = json.loads((tmp_path / "tag.json").read_text())
    assert "n_classes" in written


def test_write_pair_file_rejects_missing_n_classes(tmp_path, rng):
    N = 100
    b_x = rng.uniform(0.1, 0.9, size=N)
    b_xh = rng.uniform(0.1, 0.9, size=N)
    h = rng.integers(0, 2, size=N)
    y = rng.integers(0, 2, size=N)
    with pytest.raises(ValueError, match="n_classes"):
        write_pair_file(tmp_path, "tag", b_x, b_xh, h, y, meta={"dataset": "test"})


def test_write_pair_file_only_eval_rows_written(tmp_path, rng):
    N = 200
    b_x = rng.uniform(0.1, 0.9, size=N)
    y = rng.integers(0, 2, size=N)
    h = rng.integers(0, 2, size=N)
    _, eval_mask = make_cal_eval_masks(y)
    b_xh = fit_augmented_beliefs(b_x, h, y, eval_mask)

    npz = write_pair_file(tmp_path, "tag", b_x, b_xh, h, y, _BASE_META)

    data = np.load(npz)
    assert data["b_x"].shape[0] == int(eval_mask.sum())
