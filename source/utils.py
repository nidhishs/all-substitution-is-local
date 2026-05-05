"""Project-internal utilities: logging and data-contract validation."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from secrets import token_hex

import click
import numpy as np

from paths import experiment_results

_REQUIRED_DATA_META: dict[str, type] = {
    "n_classes": int,
    "dataset": str,
    "model": str,
    "task": str,
    "annotator_id": str,
}


def setup_logging(output_dir: Path, name: str) -> logging.Logger:
    """Return a logger writing to stdout and output_dir/{name}.log.

    Args:
        output_dir: Log directory; created if absent.
        name: Logger name and log filename stem.

    Returns:
        Logger at INFO level. Idempotent — repeated calls return the same logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in (
        logging.FileHandler(output_dir / f"{name}.log"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_run_args(log_fn: Callable[[str], None]) -> None:
    """Log the current Click command's parameters as a single INFO line."""
    params = click.get_current_context().params
    log_fn("  " + "  ".join(f"{k}={v}" for k, v in params.items()))


def update_latest_symlink(run_dir: Path) -> None:
    """Point <run_dir.parent>/latest at run_dir.name.

    Replaces an existing symlink (including dangling) or regular file.
    """
    latest = run_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)


def torch_device() -> "torch.device":
    """Return best available torch device (cuda > mps > cpu)."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_pair_array(npz_path: Path, json_path: Path) -> None:
    """Raise ValueError if the pair-array file pair fails the data contract."""
    meta = json.loads(Path(json_path).read_text())

    for key, typ in _REQUIRED_DATA_META.items():
        if key not in meta:
            raise ValueError(f"{json_path}: missing '{key}'")
        if not isinstance(meta[key], typ):
            raise ValueError(
                f"{json_path}: '{key}' must be {typ.__name__}, got {type(meta[key]).__name__}"
            )

    if "extra" in meta and not isinstance(meta["extra"], dict):
        raise ValueError(
            f"{json_path}: 'extra' must be a dict, got {type(meta['extra']).__name__}"
        )

    data = np.load(npz_path)
    for key in ("b_x", "b_xh", "h", "y"):
        if key not in data.files:
            raise ValueError(f"{npz_path}: missing array '{key}'")

    N, K = data["b_x"].shape
    if data["b_xh"].shape != (N, K):
        raise ValueError(f"b_xh shape {data['b_xh'].shape} != b_x shape {(N, K)}")
    if data["h"].shape != (N,) or data["y"].shape != (N,):
        raise ValueError(f"h and y must be shape ({N},)")
    n_cls = meta["n_classes"]
    if not np.all((data["h"] >= 0) & (data["h"] < n_cls)):
        raise ValueError(f"h values outside [0, {n_cls})")
    if not np.all((data["y"] >= 0) & (data["y"] < n_cls)):
        raise ValueError(f"y values outside [0, {n_cls})")


def load_pair(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load a pair artefact written by write_pair_file. Returns (b_x, b_xh, h, y, meta).

    Assumes the artefact is already valid (validated at write time by write_pair_file).
    """
    data = np.load(npz_path)
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    return data["b_x"], data["b_xh"], data["h"], data["y"], meta


_LOGIT_EPS: float = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    """Logit of p, clipped to [_LOGIT_EPS, 1 - _LOGIT_EPS] to avoid ±inf."""
    p = np.clip(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(p / (1.0 - p))


def make_run_dir(experiment_name: str, output_dir: Path | None) -> Path:
    """Return <override or RESULTS/<experiment_name>>/run_<hex6>. Does not mkdir."""
    base = (
        experiment_results(experiment_name) if output_dir is None else Path(output_dir)
    )
    return base / f"run_{token_hex(3)}"


def start_run(out_dir: Path, logger_name: str, label: str) -> logging.Logger:
    """mkdir + setup_logging + log experiment header + log_run_args. Returns the logger.

    Must be called from inside a Click command context (log_run_args reads it).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir, logger_name)
    logger.info(f"{logger_name}: {label}")
    log_run_args(logger.info)
    return logger


def finalize_run(out_dir: Path, payload: dict, logger: logging.Logger) -> None:
    """Write payload as results.json (no numpy fallback — caller must cast), update latest."""
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Results written to {results_path}")
    update_latest_symlink(out_dir)
    logger.info(f"latest -> {out_dir.name}")
