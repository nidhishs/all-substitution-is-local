"""Pure scoring functions for Experiment 3 review-allocation policies.

All scorers return shape (N,) arrays where higher = higher review priority.
Cross-fitting is the caller's responsibility; the fitter helpers
`fit_scoring_models` (P(h|b_x) and P(y|b_x, h)) operate on a single
training fold's arrays.

Policy list (matches SPEC narrative; 7 policies):
    score_br_hat       : ex-ante boundary regret E_{h|x}[BR(b_x, b_xh, R)]
    score_residual     : expected KL E_{h|x}[KL(b_xh || b_x)]  (residual expertise)
    score_margin       : -facet_distance(b_x, R)   (close-to-boundary)
    score_entropy      : Shannon entropy of b_x
    score_l2d          : logistic-proxy P(should defer | b_x)
    score_random       : uniform random
    score_oracle       : realized R[a_xh, y] - R[a_x, y]   (hindsight; uses y)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

import core

_EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _hx_features(b_x: np.ndarray) -> np.ndarray:
    """P(h | b_x) features: [logit(b_x[:, 1])] for binary K=2."""
    if b_x.shape[1] != 2:
        raise ValueError(f"Only binary K=2 supported; got K={b_x.shape[1]}")
    return _logit(b_x[:, 1]).reshape(-1, 1)


def _yxh_features(b_x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """P(y | b_x, h) features: [logit(b_x[:, 1]), h, logit(b_x[:, 1]) * h].

    Mirrors data.preparation.fit_augmented_beliefs feature set exactly.
    """
    z = _logit(b_x[:, 1])
    h_f = h.astype(float)
    return np.stack([z, h_f, z * h_f], axis=1)


def fit_scoring_models(
    b_x_train: np.ndarray, h_train: np.ndarray, y_train: np.ndarray
) -> tuple[LogisticRegression, LogisticRegression]:
    """Fit P(h | b_x) and P(y | b_x, h) on a training fold.

    Returns (h_model, y_model). Both are sklearn LogisticRegression(max_iter=1000).
    The y_model uses features identical to data.preparation.fit_augmented_beliefs.
    Falls back gracefully when training labels are degenerate (single class):
    a model that always predicts the observed class.
    """
    h_model = LogisticRegression(max_iter=1000)
    h_model.fit(_hx_features(b_x_train), h_train)

    y_model = LogisticRegression(max_iter=1000)
    y_model.fit(_yxh_features(b_x_train, h_train), y_train)

    return h_model, y_model


def _predict_b_xh_for_h(
    b_x_test: np.ndarray, h_value: int, y_model: LogisticRegression
) -> np.ndarray:
    """Predict (N, 2) augmented belief assuming h = h_value for every row."""
    h_filled = np.full(b_x_test.shape[0], h_value, dtype=int)
    proba = y_model.predict_proba(_yxh_features(b_x_test, h_filled))
    # Align proba columns with class labels [0, 1] regardless of training order.
    out = np.zeros_like(b_x_test)
    for col_idx, cls in enumerate(y_model.classes_):
        out[:, int(cls)] = proba[:, col_idx]
    # Fill missing class column if y_model saw only one class in training.
    if len(y_model.classes_) == 1:
        seen = int(y_model.classes_[0])
        out[:, 1 - seen] = 0.0
        out[:, seen] = 1.0
    return out


def _h_marginal(b_x_test: np.ndarray, h_model: LogisticRegression) -> np.ndarray:
    """Return (N, h_card) array of P(h_value | b_x_test). h_card = 2 for binary."""
    proba = h_model.predict_proba(_hx_features(b_x_test))
    out = np.zeros((b_x_test.shape[0], 2))
    for col_idx, cls in enumerate(h_model.classes_):
        out[:, int(cls)] = proba[:, col_idx]
    if len(h_model.classes_) == 1:
        seen = int(h_model.classes_[0])
        out[:, 1 - seen] = 0.0
        out[:, seen] = 1.0
    return out


def score_br_hat(
    b_x: np.ndarray,
    R: np.ndarray,
    h_model: LogisticRegression,
    y_model: LogisticRegression,
) -> np.ndarray:
    """E_{h|x}[BR(b_x, b_xh, R)] = ex-ante boundary regret."""
    p_h = _h_marginal(b_x, h_model)  # (N, 2)
    out = np.zeros(b_x.shape[0])
    for h_val in (0, 1):
        b_xh = _predict_b_xh_for_h(b_x, h_val, y_model)
        out += p_h[:, h_val] * core.boundary_regret(b_x, b_xh, R)
    return out


def score_residual(
    b_x: np.ndarray,
    h_model: LogisticRegression,
    y_model: LogisticRegression,
) -> np.ndarray:
    """E_{h|x}[KL(b_xh || b_x)] -- residual expertise (mutual info I(h; y | x))."""
    p_h = _h_marginal(b_x, h_model)  # (N, 2)
    out = np.zeros(b_x.shape[0])
    log_b_x = np.log(np.clip(b_x, _EPS, 1.0))
    for h_val in (0, 1):
        b_xh = _predict_b_xh_for_h(b_x, h_val, y_model)
        log_b_xh = np.log(np.clip(b_xh, _EPS, 1.0))
        kl = (b_xh * (log_b_xh - log_b_x)).sum(axis=1)  # (N,)
        out += p_h[:, h_val] * kl
    return out


def score_margin(b_x: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Higher score = closer to the reward-induced facet."""
    return -core.facet_distance(b_x, R)


def score_entropy(b_x: np.ndarray) -> np.ndarray:
    """Shannon entropy of b_x. Reward-blind."""
    return -(b_x * np.log(np.clip(b_x, _EPS, 1.0))).sum(axis=1)


def score_l2d(
    b_x_train: np.ndarray,
    h_train: np.ndarray,
    y_train: np.ndarray,
    R: np.ndarray,
    b_x_test: np.ndarray | None = None,
) -> np.ndarray:
    """Logistic-proxy learning-to-defer: predict P(should defer | b_x).

    The deferral label is `R[a_human, y] > R[a_model, y]`, computed on the
    training fold using a logistic predictor of b_xh from (b_x, h). The
    second-stage classifier maps logit(b_x) -> defer label. Returns
    P(defer = 1) for the test fold.
    """
    if b_x_test is None:
        b_x_test = b_x_train

    _, y_model = fit_scoring_models(b_x_train, h_train, y_train)
    bxh_train = _predict_b_xh_for_h(b_x_train, 0, y_model)
    bxh_train_realized = np.where(
        h_train.reshape(-1, 1) == 0,
        bxh_train,
        _predict_b_xh_for_h(b_x_train, 1, y_model),
    )
    a_model = core.model_action(b_x_train, R)
    a_human = core.model_action(bxh_train_realized, R)
    defer = (R[a_human, y_train] > R[a_model, y_train]).astype(int)

    if defer.max() == defer.min():
        return np.full(b_x_test.shape[0], float(defer[0]))

    clf = LogisticRegression(max_iter=1000)
    clf.fit(_hx_features(b_x_train), defer)
    proba = clf.predict_proba(_hx_features(b_x_test))
    pos_idx = int(np.where(clf.classes_ == 1)[0][0])
    return proba[:, pos_idx]


def score_random(b_x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform random scores."""
    return rng.random(b_x.shape[0])


def score_oracle(
    b_x: np.ndarray, b_xh: np.ndarray, y: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Hindsight realised gain: R[a_xh, y] - R[a_x, y]. Uses y; not deployable."""
    a_x = core.model_action(b_x, R)
    a_xh = core.model_action(b_xh, R)
    return R[a_xh, y] - R[a_x, y]
