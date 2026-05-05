"""Grid runner and CLI for Experiment 1."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import click
import numpy as np
import pandas as pd

import core
import utils

from .conditions import design_c1, design_c2, design_c3
from .estimator import estimate_br

logger = logging.getLogger("experiment_1")

_CONFIGS = [(G, K) for G in (2, 3, 5) for K in (G, 2 * G)]
_CONDITIONS = [("C1", design_c1), ("C2", design_c2), ("C3", design_c3)]


def run_cell(
    condition: str,
    design_fn: Callable[[int, int], tuple],
    G: int,
    K: int,
    N: int,
    rng: np.random.Generator,
    n_folds: int = 5,
) -> dict:
    problem, M, H = design_fn(K, G)
    res = estimate_br(problem, M, H, N, rng, n_folds=n_folds)

    tol = core.reward_tolerance(problem.R)
    frac_true = float((res["br_oracle"] > tol).mean())
    frac_hat = float((res["br_hat"] > tol).mean())
    ll_gain = float(
        core.log_loss_gain(res["b_x_oracle"], res["b_xh_oracle"], res["y"]).mean()
    )

    # C1 has genuine BR variation so rank correlation is the primary diagnostic;
    # FPR is reported for all conditions to check false-positive behaviour.
    if condition == "C1":
        rho, _ = core.spearman_corr(res["br_hat"], res["br_oracle"])
    else:
        rho = np.nan
    fpr = core.false_positive_rate(res["br_oracle"], res["br_hat"], threshold=tol)

    return {
        "condition": condition,
        "G": G,
        "K": K,
        "N": N,
        "spearman_rho": rho,
        "fpr": fpr,
        "frac_br_pos_true": frac_true,
        "frac_br_pos_hat": frac_hat,
        "mean_ll_gain": ll_gain,
    }


def run_grid(N: int, rng: np.random.Generator, n_folds: int = 5) -> pd.DataFrame:
    rows = [
        run_cell(cond, fn, G, K, N, rng, n_folds=n_folds)
        for G, K in _CONFIGS
        for cond, fn in _CONDITIONS
    ]
    return pd.DataFrame(rows)


def aggregate_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in results.groupby("condition"):
        rows.append(
            {
                "condition": cond,
                "n_configs": len(g),
                "mean_spearman_rho": g["spearman_rho"].mean(),
                "mean_fpr": g["fpr"].mean(),
                "mean_frac_br_pos_true": g["frac_br_pos_true"].mean(),
                "mean_frac_br_pos_hat": g["frac_br_pos_hat"].mean(),
                "mean_ll_gain": g["mean_ll_gain"].mean(),
            }
        )
    return pd.DataFrame(rows)


# fmt: off
@click.command()
@click.option("--output-dir", default=None, type=click.Path(path_type=Path), help="Output directory.")
@click.option("--n-samples", default=10_000, show_default=True)
@click.option("--seed", default=0, show_default=True)
# fmt: on
def main(output_dir: Path | None, n_samples: int, seed: int) -> None:
    out = utils.make_run_dir("experiment_1", output_dir)
    utils.start_run(out, "experiment_1", "synthetic estimator validation")

    rng = np.random.default_rng(seed)
    results = run_grid(N=n_samples, rng=rng)
    summary = aggregate_summary(results)

    utils.finalize_run(
        out,
        {"results": results.to_dict("records"), "summary": summary.to_dict("records")},
        logger,
    )
    logger.info("\n" + summary.to_markdown(index=False, floatfmt=".3f"))
