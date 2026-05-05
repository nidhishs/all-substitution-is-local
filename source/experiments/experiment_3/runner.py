"""Review allocation under scarce human input — Experiment 3."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np

import utils
from experiments.experiment_2.rewards import REWARDS
from paths import dataset_prepared

from .allocation import BASELINE, aggregate_pairs, evaluate_pair
from .synthetic import config_names, generate_synthetic_pair

logger = logging.getLogger("experiment_3")


@click.group()
def main() -> None:
    """Review allocation experiment (Experiment 3)."""


# fmt: off
@main.command()
@click.option("--n-samples", default=10_000, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--output-dir", default=None, type=click.Path(path_type=Path), help="Output directory.")
# fmt: on
def synthetic(n_samples: int, seed: int, output_dir: Path | None) -> None:
    """Synthetic regime: binary-state binary-action allocation."""
    rng = np.random.default_rng(seed)
    out_dir = utils.make_run_dir("experiment_3", output_dir)
    utils.start_run(out_dir, "experiment_3", "synthetic regime")

    pair_results: list[dict] = []
    for name in config_names():
        b_x, b_xh, h, y, meta = generate_synthetic_pair(name, n=n_samples, rng=rng)
        metrics = evaluate_pair(b_x, b_xh, h, y, REWARDS, rng=rng)
        pair_results.append({"tag": name, "meta": meta, "metrics": metrics})

    payload = {
        "results": pair_results,
        "aggregate": aggregate_pairs(pair_results, REWARDS),
    }
    utils.finalize_run(out_dir, payload, logger)
    _log_summary(payload["aggregate"], q_key="0.20")


# fmt: off
@main.command()
@click.option("--dataset", required=True, help="Dataset name (e.g. chexpert).")
@click.option("--model", default=None, help="Model directory name.")
@click.option("--seed", default=0, show_default=True)
@click.option("--output-dir", default=None, type=click.Path(path_type=Path), help="Output directory.")
# fmt: on
def real(dataset: str, model: str | None, seed: int, output_dir: Path | None) -> None:
    """Real regime: walk data/prepared/<dataset>/<model>/pairs/*.npz."""
    rng = np.random.default_rng(seed)
    prepared_root = dataset_prepared(dataset)

    if model is not None:
        model_dirs = [prepared_root / model]
    else:
        model_dirs = sorted(
            d for d in prepared_root.iterdir() if (d / "pairs").is_dir()
        )
        if not model_dirs:
            raise click.ClickException(
                f"No model directories with pairs/ found under {prepared_root}"
            )

    out_dir = utils.make_run_dir("experiment_3", output_dir)
    utils.start_run(out_dir, "experiment_3", f"real regime (dataset={dataset})")

    all_npz: list[Path] = []
    for model_dir in model_dirs:
        pairs_dir = model_dir / "pairs"
        npz_paths = sorted(pairs_dir.glob("*.npz"))
        if not npz_paths:
            raise click.ClickException(f"No .npz pair files found in {pairs_dir}")
        all_npz.extend(npz_paths)

    n_total = len(all_npz)
    pair_results: list[dict] = []
    for i, npz_path in enumerate(all_npz, 1):
        logger.info(f"[{i}/{n_total}] {npz_path.stem}")
        b_x, b_xh, h, y, meta = utils.load_pair(npz_path)
        metrics = evaluate_pair(b_x, b_xh, h, y, REWARDS, rng=rng)
        pair_results.append({"tag": npz_path.stem, "meta": meta, "metrics": metrics})

    payload = {
        "results": pair_results,
        "aggregate": aggregate_pairs(pair_results, REWARDS),
    }
    utils.finalize_run(out_dir, payload, logger)
    _log_summary(payload["aggregate"], q_key="0.20")


def _log_summary(aggregate: dict, *, q_key: str) -> None:
    """One-line per-reward summary at a chosen budget."""
    for reward_name in REWARDS:
        row = aggregate["policies"][reward_name][q_key]
        br_gain = row["BR_hat"]["mean_utility_gain"]
        br_win = row["BR_hat"]["win_rate_vs_baseline"]
        mg_gain = row[BASELINE]["mean_utility_gain"]
        ent_gain = row["Entropy"]["mean_utility_gain"]
        logger.info(
            f"  {reward_name} q={q_key}: BR_hat gain={br_gain:+.4f}  "
            f"win_vs_{BASELINE}={br_win:.1%}  "
            f"{BASELINE}={mg_gain:+.4f}  Entropy={ent_gain:+.4f}"
        )
