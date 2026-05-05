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


# ---------------------------------------------------------------------------
# load_pair
# ---------------------------------------------------------------------------


def test_load_pair_returns_arrays_and_meta(valid_pair):
    from utils import load_pair

    npz, jpath = valid_pair
    b_x, b_xh, h, y, meta = load_pair(npz)
    assert b_x.shape == (10, 2)
    assert b_xh.shape == (10, 2)
    assert h.shape == (10,)
    assert y.shape == (10,)
    assert meta["dataset"] == "test"
    assert meta["n_classes"] == 2


# ---------------------------------------------------------------------------
# make_run_dir
# ---------------------------------------------------------------------------


def test_make_run_dir_uses_default_when_override_none(tmp_path, monkeypatch):
    import paths
    from utils import make_run_dir

    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    out = make_run_dir("experiment_X", None)
    assert out.parent == tmp_path / "experiment_X"
    assert out.name.startswith("run_") and len(out.name) == 10  # "run_" + 6 hex


def test_make_run_dir_honours_override(tmp_path):
    from utils import make_run_dir

    out = make_run_dir("experiment_X", tmp_path / "custom")
    assert out.parent == tmp_path / "custom"
    assert out.name.startswith("run_") and len(out.name) == 10


# ---------------------------------------------------------------------------
# start_run / finalize_run
# ---------------------------------------------------------------------------


def test_start_run_creates_dir_and_logger_under_click_context(tmp_path):
    import click
    from click.testing import CliRunner

    from utils import start_run

    out = tmp_path / "run_abc"

    @click.command()
    @click.option("--foo", default=1)
    def cmd(foo):
        logger = start_run(out, "test_exp", "synthetic regime")
        logger.info("payload-marker")

    result = CliRunner().invoke(cmd, [], catch_exceptions=False)
    assert result.exit_code == 0
    log_text = (out / "test_exp.log").read_text()
    assert "test_exp: synthetic regime" in log_text
    assert "foo=1" in log_text  # log_run_args output
    assert "payload-marker" in log_text


def test_finalize_run_writes_json_and_symlink(tmp_path):
    import logging

    from utils import finalize_run

    out = tmp_path / "run_xyz"
    out.mkdir()
    logger = logging.getLogger("test_finalize_run")
    payload = {"foo": 1, "bar": [1.0, 2.0]}
    finalize_run(out, payload, logger)
    written = (out / "results.json").read_text()
    assert '"foo": 1' in written
    assert (tmp_path / "latest").is_symlink()
    assert (tmp_path / "latest").readlink().name == "run_xyz"


def test_finalize_run_rejects_numpy_payload(tmp_path):
    import logging

    import numpy as np

    from utils import finalize_run

    out = tmp_path / "run_np"
    out.mkdir()
    logger = logging.getLogger("test_finalize_np")
    with pytest.raises(TypeError):
        finalize_run(out, {"val": np.float32(1.0)}, logger)
