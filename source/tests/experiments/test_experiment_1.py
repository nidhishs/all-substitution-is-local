"""Tests for experiment_1: condition geometry, estimator, runner, and CLI."""

import json

import numpy as np
from click.testing import CliRunner

import core
from experiments.experiment_1.conditions import design_c1, design_c2, design_c3
from experiments.experiment_1.estimator import estimate_br
from experiments.experiment_1.runner import main, run_grid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oracle_positive_frac(problem, M, H) -> float:
    tol = max(1e-12, 1e-8 * float(problem.R.max() - problem.R.min()))
    positive = sum(
        1
        for m_s in range(M.kernel.shape[0])
        for h_s in range(H.kernel.shape[0])
        if core.boundary_regret(
            core.posterior(problem.prior, M.kernel[m_s])[None],
            core.joint_posterior(problem.prior, M.kernel[m_s], H.kernel[h_s])[None],
            problem.R,
        )[0]
        > tol
    )
    return positive / (M.kernel.shape[0] * H.kernel.shape[0])


# ---------------------------------------------------------------------------
# Condition geometry
# ---------------------------------------------------------------------------


def test_c3_zero_br_certificate():
    G, K = 3, 6
    problem, M, H = design_c3(K, G)
    res = estimate_br(problem, M, H, n=500, rng=np.random.default_rng(42), n_folds=2)
    np.testing.assert_allclose(res["br_oracle"], 0.0, atol=1e-10)


def test_c1_positive_br_fraction():
    G, K = 3, 6
    problem, M, H = design_c1(K, G)
    frac = _oracle_positive_frac(problem, M, H)
    assert frac > 0.10, f"C1 positive BR fraction {frac:.1%} should be > 10%"


def test_c2_weak_null_bound():
    G, K = 3, 6
    problem, M, H = design_c2(K, G)
    frac = _oracle_positive_frac(problem, M, H)
    assert frac <= 0.05, f"C2 positive BR fraction {frac:.1%} should be <= 5%"


def test_ll_gain_dissociation():
    """C3's human is more informative than C1's yet has zero boundary regret."""
    G, K = 3, 6
    rng = np.random.default_rng(0)
    res_c1 = estimate_br(*design_c1(K, G), n=1000, rng=rng, n_folds=5)
    res_c3 = estimate_br(*design_c3(K, G), n=1000, rng=rng, n_folds=5)
    ll_c1 = float(
        core.log_loss_gain(
            res_c1["b_x_oracle"], res_c1["b_xh_oracle"], res_c1["y"]
        ).mean()
    )
    ll_c3 = float(
        core.log_loss_gain(
            res_c3["b_x_oracle"], res_c3["b_xh_oracle"], res_c3["y"]
        ).mean()
    )
    assert ll_c3 > ll_c1, f"C3 LL gain ({ll_c3:.3f}) should exceed C1 ({ll_c1:.3f})"
    np.testing.assert_allclose(res_c3["br_oracle"], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Runner / CLI
# ---------------------------------------------------------------------------


def test_run_grid_smoke():
    df = run_grid(N=100, rng=np.random.default_rng(0), n_folds=2)
    assert len(df) == 18
    assert bool(df["condition"].isin(["C1", "C2", "C3"]).all())
    assert bool((df["N"] == 100).all())


def test_cli_writes_outputs(tmp_path):
    result = CliRunner().invoke(
        main, ["--output-dir", str(tmp_path), "--n-samples", "200"]
    )
    assert result.exit_code == 0, result.output

    latest = tmp_path / "latest"
    assert latest.is_symlink()
    out = latest.resolve()

    assert (out / "results.json").exists()

    payload = json.loads((out / "results.json").read_text())
    rows = payload["results"]
    assert len(rows) == 18
    assert set(rows[0].keys()) == {
        "condition",
        "G",
        "K",
        "N",
        "spearman_rho",
        "fpr",
        "frac_br_pos_true",
        "frac_br_pos_hat",
        "mean_ll_gain",
    }
