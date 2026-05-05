"""Grid runner and CLI for Experiment 1."""

from __future__ import annotations

import logging
from pathlib import Path
from secrets import token_hex
from typing import Callable

import click
import numpy as np
import pandas as pd

import core
import utils
from paths import experiment_results

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
@click.option("--output-dir", default=experiment_results("experiment_1"), show_default=True, type=click.Path(path_type=Path))
@click.option("--n-samples", default=10_000, show_default=True)
@click.option("--seed", default=0, show_default=True)
# fmt: on
def main(output_dir: Path, n_samples: int, seed: int) -> None:
    run_id = f"run_{token_hex(3)}"
    out = output_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    utils.setup_logging(out, "experiment_1")

    logger.info("Experiment 1: synthetic estimator validation")
    utils.log_run_args(logger.info)

    rng = np.random.default_rng(seed)
    results = run_grid(N=n_samples, rng=rng)
    summary = aggregate_summary(results)

    results_path = out / "results.json"
    results.to_json(results_path, orient="records", indent=2)
    logger.info(f"Wrote {results_path}")

    summary_path = out / "summary.json"
    summary.to_json(summary_path, orient="records", indent=2)
    logger.info(f"Wrote {summary_path}")

    utils.update_latest_symlink(out)
    logger.info(f"latest -> {out.name}")

    logger.info("\n" + summary.to_markdown(index=False, floatfmt=".3f"))
