"""Dataset-agnostic pair-preparation helpers.

All functions operate on plain numpy arrays and are called by dataset-specific
pipeline modules (e.g. data.chexpert). Nothing here knows which dataset it is.

Paper-fixed constants:
    CAL_FRAC: calibration fraction: 30% of studies per pair
    CV_FOLDS: cross-fitting folds for augmented-belief estimation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from utils import validate_pair_array

CAL_FRAC: float = 0.30
CV_FOLDS: int = 5

_EPS = 1e-7
_RANDOM_STATE = 42


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def make_cal_eval_masks(
    y: np.ndarray, *, seed: int = _RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified calibration / evaluation split (30 / 70).

    Args:
        y:    (N,) binary labels, int.
        seed: Random seed for the shuffle split.

    Returns:
        (cal_mask, eval_mask) -- complementary boolean arrays of shape (N,).
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=1.0 - CAL_FRAC, random_state=seed
    )
    cal_idx, _ = next(sss.split(np.zeros(len(y)), y))
    cal_mask = np.zeros(len(y), dtype=bool)
    cal_mask[cal_idx] = True
    return cal_mask, ~cal_mask


def temperature_scale(
    b_x_raw: np.ndarray, y: np.ndarray, cal_mask: np.ndarray
) -> tuple[float, np.ndarray]:
    """Temperature-scale raw sigmoid outputs on the calibration split.

    Args:
        b_x_raw:  (N,) raw model probabilities in (0, 1).
        y:        (N,) binary labels, int.
        cal_mask: (N,) bool — True for calibration instances.

    Returns:
        (T_opt, b_x_calibrated) — scalar temperature and (N,) calibrated beliefs.
    """
    logits = _logit(b_x_raw)
    y_cal = y[cal_mask].astype(float)

    def _nll(T: float) -> float:
        p = np.clip(1.0 / (1.0 + np.exp(-logits[cal_mask] / T)), _EPS, 1 - _EPS)
        return float(-np.mean(y_cal * np.log(p) + (1 - y_cal) * np.log(1 - p)))

    T_opt = float(minimize_scalar(_nll, bounds=(0.1, 10.0), method="bounded").x)  # type: ignore
    return T_opt, 1.0 / (1.0 + np.exp(-logits / T_opt))


def fit_augmented_beliefs(
    b_x: np.ndarray, h: np.ndarray, y: np.ndarray, eval_mask: np.ndarray
) -> np.ndarray:
    """5-fold cross-fitted P(y=1 | b_x, h) on the evaluation fold.

    Features: [logit(b_x), h, logit(b_x)*h].

    Args:
        b_x:       (N,) calibrated model probabilities.
        h:         (N,) human signal, int (binary).
        y:         (N,) labels, int.
        eval_mask: (N,) bool — rows to cross-fit; calibration rows are set to NaN.

    Returns:
        (N,) augmented beliefs; NaN for calibration rows.
    """
    b_e = b_x[eval_mask]
    h_e = h[eval_mask].astype(float)
    y_e = y[eval_mask].astype(int)

    logit_b = _logit(b_e)
    X = np.column_stack([logit_b, h_e, logit_b * h_e])

    b_xh_e = np.full(len(y_e), np.nan)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=_RANDOM_STATE)
    for train_idx, val_idx in skf.split(X, y_e):
        clf = LogisticRegression(max_iter=1000).fit(X[train_idx], y_e[train_idx])
        b_xh_e[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    b_xh = np.full(len(b_x), np.nan)
    b_xh[eval_mask] = b_xh_e
    return b_xh


def write_pair_file(
    out_dir: Path,
    tag: str,
    b_x: np.ndarray,
    b_xh: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    meta: dict,
) -> Path:
    """Write validated pair artefacts to out_dir/<tag>.{npz,json}.

    Scalar probabilities are written as (N, 2) arrays [1-p, p]; meta must
    include 'n_classes'. Only evaluation rows (b_xh not NaN) are written.
    Self-validates via validate_pair_array before returning.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_mask = ~np.isnan(b_xh)
    bx_e = b_x[eval_mask]
    bxh_e = b_xh[eval_mask]

    npz_path = out_dir / f"{tag}.npz"
    json_path = out_dir / f"{tag}.json"

    np.savez(
        npz_path,
        b_x=np.column_stack([1.0 - bx_e, bx_e]).astype(np.float64),
        b_xh=np.column_stack([1.0 - bxh_e, bxh_e]).astype(np.float64),
        h=h[eval_mask].astype(np.int32),
        y=y[eval_mask].astype(np.int32),
    )
    json_path.write_text(json.dumps(meta, indent=2))

    validate_pair_array(npz_path, json_path)
    return npz_path
