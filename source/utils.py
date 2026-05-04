"""Project-internal utilities: logging and data-contract validation."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np

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
