"""Tests for Experiment 3 policy scorers and allocation engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner as ClickTestRunner

import core
from experiments.experiment_2.rewards import R1, R3
from experiments.experiment_3 import allocation, policies, synthetic


def _toy_arrays(rng: np.random.Generator, N: int = 200, K: int = 2):
    """Synthetic (b_x, b_xh, h, y) for unit tests."""
    b_x_pos = rng.beta(2.0, 2.0, size=N)
    b_x = np.stack([1.0 - b_x_pos, b_x_pos], axis=1)
    h = rng.integers(0, K, size=N)
    y = (rng.uniform(size=N) < b_x_pos).astype(int)
    # Augmented belief: shrink/expand toward h-as-evidence.
    bxh_pos = np.clip(0.5 * b_x_pos + 0.5 * h, 1e-3, 1.0 - 1e-3)
    b_xh = np.stack([1.0 - bxh_pos, bxh_pos], axis=1)
    return b_x, b_xh, h, y


def test_score_entropy_shape_and_finite():
    rng = np.random.default_rng(0)
    b_x, *_ = _toy_arrays(rng)
    s = policies.score_entropy(b_x)
    assert s.shape == (b_x.shape[0],)
    assert np.isfinite(s).all()
    assert (s >= 0).all()


def test_score_entropy_max_at_uniform():
    b_uniform = np.full((1, 2), 0.5)
    b_extreme = np.array([[0.99, 0.01]])
    assert policies.score_entropy(b_uniform)[0] > policies.score_entropy(b_extreme)[0]


def test_score_margin_picks_close_to_boundary():
    # b on the R1 boundary (0.5) should outrank b at 0.9.
    b_close = np.array([[0.5, 0.5]])
    b_far = np.array([[0.9, 0.1]])
    assert policies.score_margin(b_close, R1)[0] > policies.score_margin(b_far, R1)[0]


def test_score_oracle_matches_realized_gain():
    rng = np.random.default_rng(1)
    b_x, b_xh, h, y = _toy_arrays(rng)
    s = policies.score_oracle(b_x, b_xh, y, R1)
    a_x = core.model_action(b_x, R1)
    a_xh = core.model_action(b_xh, R1)
    expected = R1[a_xh, y] - R1[a_x, y]
    np.testing.assert_array_almost_equal(s, expected)


def test_score_random_seeded_reproducibility():
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    b_x, *_ = _toy_arrays(np.random.default_rng(0))
    np.testing.assert_array_equal(
        policies.score_random(b_x, rng_a), policies.score_random(b_x, rng_b)
    )


def test_score_br_hat_nonnegative():
    rng = np.random.default_rng(2)
    b_x, _, h, y = _toy_arrays(rng, N=400)
    h_model, y_model = policies.fit_scoring_models(b_x, h, y)
    s = policies.score_br_hat(b_x, R1, h_model, y_model)
    assert (s >= -1e-9).all()  # BR is non-negative; allow tiny float slack
    assert s.shape == (b_x.shape[0],)


def test_score_residual_nonnegative():
    """E[KL(b_xh || b_x)] >= 0 by Jensen."""
    rng = np.random.default_rng(3)
    b_x, _, h, y = _toy_arrays(rng, N=400)
    h_model, y_model = policies.fit_scoring_models(b_x, h, y)
    s = policies.score_residual(b_x, h_model, y_model)
    assert (s >= -1e-9).all()


def test_score_l2d_in_unit_interval():
    rng = np.random.default_rng(4)
    b_x, b_xh, h, y = _toy_arrays(rng, N=400)
    s = policies.score_l2d(b_x, y, b_xh, R1)
    assert s.shape == (b_x.shape[0],)
    assert ((s >= 0) & (s <= 1)).all()


def test_top_q_indices_correct_count():
    scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    idx = allocation.top_q_indices(scores, q=0.4)
    assert len(idx) == 2  # ceil(0.4 * 5) = 2
    assert set(idx.tolist()) == {2, 4}  # top-2 by score


def test_utility_gain_random_baseline():
    """Random policy at q=1 should yield mean gain over all instances."""
    rng = np.random.default_rng(0)
    b_x, b_xh, h, y = _toy_arrays(rng, N=500)
    selected_all = np.arange(len(y))
    g = allocation.per_instance_review_gain(b_x, b_xh, y, R1)
    util = allocation.utility_gain(g, selected_all)
    assert abs(util - g.mean()) < 1e-12


def test_compute_scores_returns_all_policies():
    rng = np.random.default_rng(0)
    b_x, b_xh, h, y = _toy_arrays(rng, N=400)
    scores = allocation.compute_scores(b_x, b_xh, h, y, R1, rng=rng, n_folds=5)
    expected_keys = {
        "BR_hat",
        "Residual",
        "Margin",
        "Entropy",
        "L2D",
        "Random",
        "Oracle",
    }
    assert set(scores.keys()) == expected_keys
    for k, v in scores.items():
        assert v.shape == (400,)
        assert np.isfinite(v).all()


def test_paired_bootstrap_returns_finite_ci():
    rng = np.random.default_rng(0)
    n = 200
    diff = rng.normal(0.05, 0.1, size=n)
    lo, hi = allocation.paired_bootstrap_ci(diff, n_boot=200, rng=rng)
    assert lo < hi
    assert np.isfinite(lo) and np.isfinite(hi)


def test_evaluate_budget_yields_expected_keys():
    rng = np.random.default_rng(0)
    b_x, b_xh, h, y = _toy_arrays(rng, N=400)
    scores = allocation.compute_scores(b_x, b_xh, h, y, R1, rng=rng, n_folds=5)
    out = allocation.evaluate_budget(
        scores,
        b_x,
        b_xh,
        y,
        R1,
        q=0.2,
        baseline_policy="Random",
        rng=rng,
        n_boot=200,
    )
    for policy_name in (
        "BR_hat",
        "Residual",
        "Margin",
        "Entropy",
        "L2D",
        "Random",
        "Oracle",
    ):
        entry = out[policy_name]
        for key in (
            "utility_gain",
            "n_selected",
            "delta_vs_baseline",
            "delta_ci_lo",
            "delta_ci_hi",
        ):
            assert key in entry


def test_synthetic_configs_listed():
    names = synthetic.config_names()
    assert names == ("A_balanced", "A_low_signal", "A_off_threshold")


def test_generate_synthetic_pair_shapes():
    rng = np.random.default_rng(0)
    b_x, b_xh, h, y, meta = synthetic.generate_synthetic_pair(
        "A_balanced", n=500, rng=rng
    )
    assert b_x.shape == (500, 2)
    assert b_xh.shape == (500, 2)
    assert h.shape == (500,)
    assert y.shape == (500,)
    np.testing.assert_array_almost_equal(b_x.sum(axis=1), 1.0)
    np.testing.assert_array_almost_equal(b_xh.sum(axis=1), 1.0)
    assert meta["task"] == "A_balanced"
    assert meta["n_classes"] == 2


def test_synthetic_off_threshold_high_zero_br_under_r3():
    """A_off_threshold under R3 should have very high zero-BR mass."""
    rng = np.random.default_rng(0)
    b_x, b_xh, _, _, _ = synthetic.generate_synthetic_pair(
        "A_off_threshold",
        n=2000,
        rng=rng,
    )
    br = core.boundary_regret(b_x, b_xh, R3)
    tol = core.reward_tolerance(R3)
    assert (br <= tol).mean() > 0.85  # loose floor; geometry guarantees ~all-zero


# ---------------------------------------------------------------------------
# Seed reproducibility — --seed must control fold splits + bootstrap jointly
# ---------------------------------------------------------------------------


def test_compute_scores_same_seed_identical():
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    b_x, b_xh, h, y = _toy_arrays(np.random.default_rng(0), N=400)
    sa = allocation.compute_scores(b_x, b_xh, h, y, R1, rng=rng_a, n_folds=5)
    sb = allocation.compute_scores(b_x, b_xh, h, y, R1, rng=rng_b, n_folds=5)
    for k in sa:
        np.testing.assert_array_equal(sa[k], sb[k])


def test_compute_scores_different_seeds_diverge_on_cv_policies():
    """BR_hat / Residual / L2D depend on fold splits — must differ across seeds."""
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(8)
    b_x, b_xh, h, y = _toy_arrays(np.random.default_rng(0), N=400)
    sa = allocation.compute_scores(b_x, b_xh, h, y, R1, rng=rng_a, n_folds=5)
    sb = allocation.compute_scores(b_x, b_xh, h, y, R1, rng=rng_b, n_folds=5)
    for k in ("BR_hat", "Residual", "L2D"):
        assert not np.array_equal(
            sa[k], sb[k]
        ), f"{k} did not change across seeds — CV folds are not seed-driven"


def test_cli_synthetic_smoke(tmp_path: Path):
    from experiments.experiment_3.runner import main

    runner = ClickTestRunner()
    result = runner.invoke(
        main,
        [
            "synthetic",
            "--n-samples",
            "300",
            "--seed",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and not p.is_symlink()]
    assert len(run_dirs) == 1
    out_dir = run_dirs[0]
    payload = json.loads((out_dir / "results.json").read_text())
    assert "results" in payload
    assert "aggregate" in payload
    assert payload["aggregate"]["n_pairs"] == 3  # 3 synthetic configs
    first = payload["results"][0]
    assert "R1" in first["metrics"]
    assert "0.20" in first["metrics"]["R1"]
    assert "BR_hat" in first["metrics"]["R1"]["0.20"]


def test_cli_real_uses_dataset_prepared(tmp_path: Path, monkeypatch):
    """Smoke: write a fake pair artefact, monkeypatch dataset_prepared, invoke real."""
    import experiments.experiment_3.runner as runner_mod
    from data.preparation import write_pair_file
    from experiments.experiment_3.runner import main

    rng = np.random.default_rng(0)
    n = 350
    b_x_pos = rng.beta(2.0, 2.0, size=n)
    b_xh_pos = b_x_pos.copy()
    h = rng.integers(0, 2, size=n).astype(int)
    y = rng.integers(0, 2, size=n).astype(int)
    pairs_dir = tmp_path / "fakeds" / "fakemodel" / "pairs"
    pairs_dir.mkdir(parents=True)
    meta = {
        "dataset": "fakeds",
        "model": "fakemodel",
        "task": "T1",
        "annotator_id": "A1",
        "n_classes": 2,
    }
    write_pair_file(pairs_dir, "T1__A1", b_x_pos, b_xh_pos, h, y, meta)

    monkeypatch.setattr(
        runner_mod, "dataset_prepared", lambda name: tmp_path / "fakeds"
    )

    out_dir = tmp_path / "out"
    runner = ClickTestRunner()
    result = runner.invoke(
        main,
        ["real", "--dataset", "fakeds", "--seed", "0", "--output-dir", str(out_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    run_subdir = next(out_dir.iterdir())
    payload = json.loads((run_subdir / "results.json").read_text())
    assert payload["aggregate"]["n_pairs"] == 1


def test_evaluate_pair_and_aggregate_pairs_roundtrip():
    from experiments.experiment_2.rewards import REWARDS

    rng = np.random.default_rng(0)
    b_x, b_xh, h, y = _toy_arrays(rng, N=400)
    metrics = allocation.evaluate_pair(b_x, b_xh, h, y, REWARDS, rng=rng)
    assert set(metrics.keys()) == set(REWARDS.keys())
    assert "0.20" in metrics["R1"]
    assert "BR_hat" in metrics["R1"]["0.20"]
    # _baseline_policy must NOT leak into per-policy output anymore
    assert "_baseline_policy" not in metrics["R1"]["0.20"]

    agg = allocation.aggregate_pairs(
        [{"tag": "x", "meta": {}, "metrics": metrics}],
        REWARDS,
    )
    assert agg["n_pairs"] == 1
    assert agg["baseline_policy"] == "Margin"
    assert "BR_hat" in agg["policies"]["R1"]["0.20"]
