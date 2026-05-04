"""Tests for Experiment 2: rewards, metrics, and aggregation."""

from __future__ import annotations

import core
import numpy as np
import pytest
from data.preparation import fit_augmented_beliefs, make_cal_eval_masks, write_pair_file
from experiments.experiment_2.metrics import compute_pair_metrics
from experiments.experiment_2.rewards import (
    FACET_THRESHOLDS,
    R1,
    R2,
    R3,
    REWARDS,
    indifference_threshold,
)
from experiments.experiment_2.runner import aggregate, run_pairs_dir

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_pair():
    rng = np.random.default_rng(42)
    N = 200
    b_x = rng.dirichlet([1, 1], size=N)
    b_xh = rng.dirichlet([2, 1], size=N)  # slightly shifted from b_x
    h = rng.integers(0, 2, N)
    y = rng.integers(0, 2, N)
    return b_x, b_xh, h, y


@pytest.fixture
def zero_br_pair():
    # b_xh == b_x -> no augmentation -> BR^ = 0 everywhere.
    rng = np.random.default_rng(0)
    N = 100
    b_x = rng.dirichlet([1, 1], size=N)
    h = rng.integers(0, 2, N)
    y = rng.integers(0, 2, N)
    return b_x, b_x.copy(), h, y


# ---------------------------------------------------------------------------
# Reward thresholds
# ---------------------------------------------------------------------------


def test_indifference_threshold_r1():
    assert indifference_threshold(R1) == pytest.approx(0.5)


def test_indifference_threshold_r2():
    assert indifference_threshold(R2) == pytest.approx(5 / 6)


def test_indifference_threshold_r3():
    assert indifference_threshold(R3) == pytest.approx(1 / 6)


def test_facet_thresholds_match_derived():
    for name, R in REWARDS.items():
        assert FACET_THRESHOLDS[name] == pytest.approx(indifference_threshold(R))


# ---------------------------------------------------------------------------
# compute_pair_metrics
# ---------------------------------------------------------------------------


def test_compute_pair_metrics_keys_present(synthetic_pair):
    b_x, b_xh, h, y = synthetic_pair
    result = compute_pair_metrics(b_x, b_xh, h, y)
    for name in REWARDS:
        assert name in result
        for key in (
            "rho_ll_br",
            "rho_ll_br_ci_lo",
            "rho_ll_br_ci_hi",
            "rho_facet_br",
            "top_decile_frac_br_zero",
            "mean_ll_gain",
            "all_zero_br",
        ):
            assert key in result[name], f"Missing key {key!r} in {name}"


def test_compute_pair_metrics_all_zero_br_gives_nan_rho(zero_br_pair):
    b_x, b_xh, h, y = zero_br_pair
    result = compute_pair_metrics(b_x, b_xh, h, y)
    for name in REWARDS:
        assert result[name]["all_zero_br"] is True
        assert np.isnan(result[name]["rho_ll_br"])


def test_compute_pair_metrics_top_decile_from_disagreement(synthetic_pair):
    """top_decile_frac_br_zero must match an explicit disagreement-based computation."""
    b_x, b_xh, h, y = synthetic_pair
    N = len(h)
    result = compute_pair_metrics(b_x, b_xh, h, y)

    for name, R in REWARDS.items():
        tol = max(1e-12, 1e-8 * float(R.max() - R.min()))
        br_hat = core.boundary_regret(b_x, b_xh, R)
        disagreement = 1.0 - b_x[np.arange(N), h.astype(int)]
        top_mask = disagreement >= np.percentile(disagreement, 90)
        expected = float((br_hat[top_mask] <= tol).mean())
        assert result[name]["top_decile_frac_br_zero"] == pytest.approx(
            expected
        ), f"top_decile_frac_br_zero for {name} does not match disagreement-based computation"


def test_br_tolerance_scales_with_reward_range():
    r1_range = float(R1.max() - R1.min())
    r2_range = float(R2.max() - R2.min())
    tol_r1 = max(1e-12, 1e-8 * r1_range)
    tol_r2 = max(1e-12, 1e-8 * r2_range)
    assert tol_r2 == pytest.approx(5 * tol_r1)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def _make_dummy_results(n_pairs: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_pairs):
        metrics = {}
        for name in REWARDS:
            rho = float(rng.uniform(-1, 1))
            metrics[name] = {
                "rho_ll_br": rho,
                "rho_ll_br_ci_lo": rho - 0.1,
                "rho_ll_br_ci_hi": rho + 0.1,
                "rho_facet_br": float(rng.uniform(-1, 1)),
                "top_decile_frac_br_zero": float(rng.uniform(0, 1)),
                "mean_ll_gain": float(rng.uniform(0, 1)),
                "all_zero_br": False,
            }
        results.append({"tag": f"pair_{i}", "metrics": metrics})
    return results


def _make_result_with_zero_br(name_to_zero: str) -> dict:
    rng = np.random.default_rng(99)
    metrics = {}
    for name in REWARDS:
        if name == name_to_zero:
            metrics[name] = {
                "rho_ll_br": float("nan"),
                "rho_ll_br_ci_lo": float("nan"),
                "rho_ll_br_ci_hi": float("nan"),
                "rho_facet_br": float("nan"),
                "top_decile_frac_br_zero": 1.0,
                "mean_ll_gain": 0.1,
                "all_zero_br": True,
            }
        else:
            rho = float(rng.uniform(-1, 1))
            metrics[name] = {
                "rho_ll_br": rho,
                "rho_ll_br_ci_lo": rho - 0.1,
                "rho_ll_br_ci_hi": rho + 0.1,
                "rho_facet_br": float(rng.uniform(-1, 1)),
                "top_decile_frac_br_zero": float(rng.uniform(0, 1)),
                "mean_ll_gain": 0.1,
                "all_zero_br": False,
            }
    return {"tag": "zero_br_pair", "metrics": metrics}


def test_aggregate_reports_both_rho_stats():
    results = _make_dummy_results(10)
    agg = aggregate(results)
    for name in REWARDS:
        assert "mean_rho_ll_br" in agg[name]
        assert "mean_rho_ll_br_all" in agg[name]


def test_aggregate_zero_br_pair_excluded_from_mean_but_included_in_all():
    good_results = _make_dummy_results(4, seed=0)
    zero_result = _make_result_with_zero_br("R2")
    all_results = good_results + [zero_result]

    agg = aggregate(all_results)

    assert agg["R2"]["n_valid"] == 4
    good_rhos = np.array([r["metrics"]["R2"]["rho_ll_br"] for r in good_results])
    assert agg["R2"]["mean_rho_ll_br"] == pytest.approx(float(np.nanmean(good_rhos)))
    expected_all = float(
        np.nan_to_num(
            np.array([r["metrics"]["R2"]["rho_ll_br"] for r in all_results]), nan=0.0
        ).mean()
    )
    assert agg["R2"]["mean_rho_ll_br_all"] == pytest.approx(expected_all)


def test_aggregate_n_pairs():
    results = _make_dummy_results(7)
    agg = aggregate(results)
    assert agg["n_pairs"] == 7


# ---------------------------------------------------------------------------
# e2e: runner consumes any contract-conforming artefact (dataset-agnostic)
# ---------------------------------------------------------------------------


def test_runner_raises_on_empty_pairs_dir(tmp_path):
    """Empty pair directories must fail loudly, not silently produce NaN aggregates."""
    empty_dir = tmp_path / "pairs"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No .npz pair files"):
        run_pairs_dir(empty_dir)


def test_runner_raises_on_missing_pairs_dir(tmp_path):
    """Nonexistent pairs dir behaves the same as empty (glob returns []) and fails."""
    with pytest.raises(FileNotFoundError, match="No .npz pair files"):
        run_pairs_dir(tmp_path / "does_not_exist")


def test_runner_consumes_minimal_contract_artefact(tmp_path):
    """Runner reads any artefact passing the 5-key contract, no chexpert specifics."""
    rng = np.random.default_rng(7)
    N = 200
    b_x = rng.uniform(0.1, 0.9, size=N)
    y = rng.integers(0, 2, size=N)
    h = rng.integers(0, 2, size=N)
    _, eval_mask = make_cal_eval_masks(y)
    b_xh = fit_augmented_beliefs(b_x, h, y, eval_mask)

    pairs_dir = tmp_path / "pairs"
    meta = {
        "n_classes": 2,
        "dataset": "synthetic",
        "model": "test_model",
        "task": "Widget",
        "annotator_id": "ann1",
    }
    write_pair_file(pairs_dir, "Widget__ann1", b_x, b_xh, h, y, meta)

    results = run_pairs_dir(pairs_dir)

    assert len(results) == 1
    assert results[0]["meta"]["dataset"] == "synthetic"
    assert results[0]["meta"]["task"] == "Widget"
    assert "condition" not in results[0]["meta"]
    assert "h_rad" not in results[0]["meta"]

    agg = aggregate(results)
    assert agg["n_pairs"] == 1
    for name in ("R1", "R2", "R3"):
        assert name in agg
