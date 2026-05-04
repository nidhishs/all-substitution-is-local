"""Tests for utils.py — logging and pair-array validation."""

import json
import logging

import numpy as np
import pytest
from utils import setup_logging, validate_pair_array

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_pair(tmp_path):
    """Write a valid pair .npz + .json and return (npz_path, json_path)."""
    rng = np.random.default_rng(0)
    N, K, n_cls = 10, 2, 2
    b = rng.dirichlet(np.ones(K), size=N).astype(np.float32)
    npz = tmp_path / "pair.npz"
    jpath = tmp_path / "pair.json"
    np.savez(
        npz,
        b_x=b,
        b_xh=b,
        h=rng.integers(0, n_cls, size=N),
        y=rng.integers(0, 2, size=N),
    )
    jpath.write_text(
        json.dumps(
            {
                "n_classes": n_cls,
                "dataset": "test",
                "model": "test_model",
                "task": "Cardiomegaly",
                "annotator_id": "bc1",
            }
        )
    )
    return npz, jpath


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


def test_setup_logging_returns_logger(tmp_path):
    log = setup_logging(tmp_path, "test_run")
    assert isinstance(log, logging.Logger)
    assert log.name == "test_run"


def test_setup_logging_creates_log_file(tmp_path):
    setup_logging(tmp_path, "myexp")
    assert (tmp_path / "myexp.log").exists()


def test_setup_logging_writes_to_file(tmp_path):
    log = setup_logging(tmp_path, "writetest")
    log.info("hello from test")
    assert "hello from test" in (tmp_path / "writetest.log").read_text()


def test_setup_logging_does_not_duplicate_handlers(tmp_path):
    log = setup_logging(tmp_path, "dedup")
    n = len(log.handlers)
    setup_logging(tmp_path, "dedup")  # second call must be a no-op
    assert len(log.handlers) == n


def test_setup_logging_creates_output_dir_if_absent(tmp_path):
    new_dir = tmp_path / "nested" / "dir"
    setup_logging(new_dir, "x")
    assert new_dir.exists()


# ---------------------------------------------------------------------------
# validate_pair_array
# ---------------------------------------------------------------------------


def test_validate_pair_array_accepts_valid_pair(valid_pair):
    validate_pair_array(*valid_pair)  # must not raise


def test_validate_pair_array_rejects_missing_metadata_key(valid_pair):
    npz, jpath = valid_pair
    meta = json.loads(jpath.read_text())
    del meta["n_classes"]
    jpath.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="missing 'n_classes'"):
        validate_pair_array(npz, jpath)


def test_validate_pair_array_rejects_missing_array(valid_pair):
    npz, jpath = valid_pair
    data = dict(np.load(npz))
    del data["b_xh"]
    np.savez(npz, **data)
    with pytest.raises(ValueError, match="missing array"):
        validate_pair_array(npz, jpath)


def test_validate_pair_array_rejects_shape_mismatch(valid_pair):
    npz, jpath = valid_pair
    data = dict(np.load(npz))
    data["b_xh"] = data["b_xh"][:, :1]  # wrong number of columns
    np.savez(npz, **data)
    with pytest.raises(ValueError, match="b_xh shape"):
        validate_pair_array(npz, jpath)


def test_validate_pair_array_rejects_h_out_of_range(valid_pair):
    npz, jpath = valid_pair
    data = dict(np.load(npz))
    data["h"] = np.full(10, 5, dtype=int)  # 5 is outside [0, 2)
    np.savez(npz, **data)
    with pytest.raises(ValueError, match="h values outside"):
        validate_pair_array(npz, jpath)


@pytest.mark.parametrize(
    "key", ["n_classes", "dataset", "model", "task", "annotator_id"]
)
def test_validate_pair_array_rejects_missing_required_key(valid_pair, key):
    npz, jpath = valid_pair
    meta = json.loads(jpath.read_text())
    del meta[key]
    jpath.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=f"missing '{key}'"):
        validate_pair_array(npz, jpath)


def test_validate_pair_array_rejects_wrong_type_n_classes(valid_pair):
    npz, jpath = valid_pair
    meta = json.loads(jpath.read_text())
    meta["n_classes"] = "2"  # string instead of int
    jpath.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="'n_classes' must be int"):
        validate_pair_array(npz, jpath)


def test_validate_pair_array_rejects_wrong_type_dataset(valid_pair):
    npz, jpath = valid_pair
    meta = json.loads(jpath.read_text())
    meta["dataset"] = 42
    jpath.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="'dataset' must be str"):
        validate_pair_array(npz, jpath)


def test_validate_pair_array_rejects_extra_not_dict(valid_pair):
    npz, jpath = valid_pair
    meta = json.loads(jpath.read_text())
    meta["extra"] = [1, 2, 3]
    jpath.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="'extra' must be a dict"):
        validate_pair_array(npz, jpath)


def test_validate_pair_array_accepts_extra_dict(valid_pair):
    npz, jpath = valid_pair
    meta = json.loads(jpath.read_text())
    meta["extra"] = {"T_opt": 1.5, "n_eval": 100}
    jpath.write_text(json.dumps(meta))
    validate_pair_array(npz, jpath)  # must not raise


def test_validate_pair_array_accepts_no_extra(valid_pair):
    validate_pair_array(*valid_pair)  # fixture has no extra; must not raise


def test_validate_pair_array_rejects_y_out_of_range(valid_pair):
    npz, jpath = valid_pair
    data = dict(np.load(npz))
    data["y"] = np.full(10, 5, dtype=int)  # 5 is outside [0, 2)
    np.savez(npz, **data)
    with pytest.raises(ValueError, match="y values outside"):
        validate_pair_array(npz, jpath)
