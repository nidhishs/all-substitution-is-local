"""Review allocation under scarce human input — Experiment 3."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from secrets import token_hex

import click
import numpy as np

import utils
from paths import dataset_prepared, experiment_results

from .allocation import POLICY_NAMES, compute_scores, evaluate_budget
from .synthetic import config_names, generate_synthetic_pair
from experiments.experiment_2.rewards import REWARDS

logger = logging.getLogger("experiment_3")

_BUDGETS = (0.05, 0.10, 0.20, 0.50)
_BASELINE = "Margin"  # paired-comparison baseline; matches narrative claim


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def _evaluate_pair(
    b_x: np.ndarray,
    b_xh: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Per-pair metrics: {reward_name: {budget_str: {policy: stats}}}."""
    metrics: dict = {}
    for reward_name, R in REWARDS.items():
        scores = compute_scores(b_x, b_xh, h, y, R, rng=rng)
        per_reward: dict = {}
        for q in _BUDGETS:
            per_reward[f"{q:.2f}"] = evaluate_budget(
                scores, b_x, b_xh, y, R, q,
                baseline_policy=_BASELINE, rng=rng,
            )
        metrics[reward_name] = per_reward
    return metrics


def _load_pair_npz(npz_path: Path):
    """Return (b_x, b_xh, h, y, meta)."""
    data = np.load(npz_path)
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    return data["b_x"], data["b_xh"], data["h"], data["y"], meta


def _aggregate(results: list[dict]) -> dict:
    """Cross-pair aggregation per (reward, budget)."""
    n_pairs = len(results)
    out: dict = {"n_pairs": n_pairs, "baseline_policy": _BASELINE, "policies": {}}
    for reward_name in REWARDS:
        out["policies"][reward_name] = {}
        for q in _BUDGETS:
            q_key = f"{q:.2f}"
            per_q: dict = {}
            for policy in POLICY_NAMES:
                gains = np.array(
                    [r["metrics"][reward_name][q_key][policy]["utility_gain"]
                     for r in results],
                    dtype=float,
                )
                deltas = np.array(
                    [r["metrics"][reward_name][q_key][policy]["delta_vs_baseline"]
                     for r in results],
                    dtype=float,
                )
                wins = int((deltas > 0).sum())
                losses = int((deltas < 0).sum())
                ties = int((deltas == 0).sum())
                per_q[policy] = {
                    "mean_utility_gain": float(np.nanmean(gains)),
                    "mean_delta_vs_baseline": float(np.nanmean(deltas)),
                    "win_rate_vs_baseline": float(wins / max(n_pairs, 1)),
                    "n_wins": wins,
                    "n_losses": losses,
                    "n_ties": ties,
                }
            out["policies"][reward_name][q_key] = per_q
    return out


def _setup_run(out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    utils.setup_logging(out_dir, "experiment_3")
    logger.info(f"Experiment 3: {label}")
    utils.log_run_args(logger.info)


def _write_outputs(out_dir: Path, payload: dict) -> None:
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    logger.info(f"Results written to {results_path}")
    utils.update_latest_symlink(out_dir)
    logger.info(f"latest -> {out_dir.name}")


def _resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        return experiment_results("experiment_3") / f"run_{token_hex(3)}"
    return Path(output_dir) / f"run_{token_hex(3)}"


@click.group()
def main() -> None:
    """Review allocation experiment (Experiment 3)."""


# fmt: off
@main.command()
@click.option("--n-samples", default=10_000, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--output-dir", default=None, type=click.Path(path_type=Path),
              help="Output directory. Defaults to results/experiment_3/run_<hex6>.")
# fmt: on
def synthetic(n_samples: int, seed: int, output_dir: Path | None) -> None:
    """Synthetic regime: binary-state binary-action allocation."""
    rng = np.random.default_rng(seed)
    out_dir = _resolve_output_dir(output_dir)
    _setup_run(out_dir, label="synthetic regime")

    pair_results: list[dict] = []
    for name in config_names():
        b_x, b_xh, h, y, meta = generate_synthetic_pair(name, n=n_samples, rng=rng)
        metrics = _evaluate_pair(b_x, b_xh, h, y, rng)
        pair_results.append({"tag": name, "meta": meta, "metrics": metrics})

    payload = {"results": pair_results, "aggregate": _aggregate(pair_results)}
    _write_outputs(out_dir, payload)

    for reward_name in REWARDS:
        row = payload["aggregate"]["policies"][reward_name]["0.20"]
        br = row["BR_hat"]["mean_utility_gain"]
        mg = row["Margin"]["mean_utility_gain"]
        ent = row["Entropy"]["mean_utility_gain"]
        logger.info(
            f"  {reward_name} q=0.20: BR_hat={br:+.4f}  "
            f"Margin={mg:+.4f}  Entropy={ent:+.4f}"
        )


# fmt: off
@main.command()
@click.option("--dataset", required=True, help="Dataset name (e.g. chexpert).")
@click.option("--model", default=None,
              help="Model directory name. If omitted, all models under the dataset are processed.")
@click.option("--seed", default=0, show_default=True)
@click.option("--output-dir", default=None, type=click.Path(path_type=Path),
              help="Output directory. Defaults to results/experiment_3/run_<hex6>.")
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

    out_dir = _resolve_output_dir(output_dir)
    _setup_run(out_dir, label=f"real regime (dataset={dataset})")

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
        b_x, b_xh, h, y, meta = _load_pair_npz(npz_path)
        metrics = _evaluate_pair(b_x, b_xh, h, y, rng)
        pair_results.append({"tag": npz_path.stem, "meta": meta, "metrics": metrics})

    payload = {"results": pair_results, "aggregate": _aggregate(pair_results)}
    _write_outputs(out_dir, payload)

    for reward_name in REWARDS:
        row = payload["aggregate"]["policies"][reward_name]["0.20"]
        br_gain = row["BR_hat"]["mean_utility_gain"]
        br_win = row["BR_hat"]["win_rate_vs_baseline"]
        mg_gain = row["Margin"]["mean_utility_gain"]
        logger.info(
            f"  {reward_name} q=0.20: BR_hat mean_gain={br_gain:+.4f}  "
            f"win_vs_Margin={br_win:.1%}  "
            f"Margin mean_gain={mg_gain:+.4f}"
        )
