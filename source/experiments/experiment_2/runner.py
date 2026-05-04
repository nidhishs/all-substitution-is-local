"""Pair loading and aggregation for Experiment 2."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from secrets import token_hex

import click
import numpy as np
import utils
from paths import dataset_prepared, experiment_results

from .metrics import compute_pair_metrics
from .rewards import REWARDS

logger = logging.getLogger("experiment_2")


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


def load_pair(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load a validated pair artefact. Returns (b_x, b_xh, h, y, meta)."""
    data = np.load(npz_path)
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    return data["b_x"], data["b_xh"], data["h"], data["y"], meta


def run_pairs_dir(
    pairs_dir: Path,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Compute dissociation metrics for every pair artefact in a directory.

    Raises FileNotFoundError if no .npz pair files are found, since aggregating
    over zero pairs produces structurally valid but entirely-NaN results.
    """
    npz_paths = sorted(pairs_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz pair files found in {pairs_dir}")

    results = []
    for i, npz_path in enumerate(npz_paths, 1):
        b_x, b_xh, h, y, meta = load_pair(npz_path)
        metrics = compute_pair_metrics(b_x, b_xh, h, y, rng)
        r2 = metrics["R2"]
        logger.info(
            f"[{i:2d}/{len(npz_paths)}] {npz_path.stem:<45s}  "
            f"ρ_R2={r2['rho_ll_br']:+.3f}  "
            f"zero_br={r2['top_decile_frac_br_zero']:.1%}"
        )
        results.append({"tag": npz_path.stem, "metrics": metrics, "meta": meta})
    return results


def aggregate(results: list[dict]) -> dict:
    """Aggregate Spearman stats across pairs.

    For each reward, reports:
      mean_rho_ll_br      -- nanmean over valid pairs (BR=0 pairs excluded).
      mean_rho_ll_br_all  -- mean treating BR=0 pairs as rho=0 (full denominator).
      n_valid             -- pairs with non-degenerate BR^ under this reward.
      n_rho_positive/negative -- sign counts across all pairs.
      mean_top_decile_frac_br_zero -- mean fraction of top-disagreement-decile with BR^=0.
    """
    agg: dict = {"n_pairs": len(results)}
    for name in REWARDS:
        rhos = np.array([r["metrics"][name]["rho_ll_br"] for r in results], dtype=float)
        top_decile = np.array(
            [r["metrics"][name]["top_decile_frac_br_zero"] for r in results],
            dtype=float,
        )
        n_valid = int(sum(not r["metrics"][name]["all_zero_br"] for r in results))

        agg[name] = {
            "mean_rho_ll_br": float(np.nanmean(rhos)),
            "mean_rho_ll_br_all": float(np.nan_to_num(rhos, nan=0.0).mean()),
            "n_valid": n_valid,
            "n_rho_positive": int(np.nansum(rhos > 0)),
            "n_rho_negative": int(np.nansum(rhos < 0)),
            "mean_top_decile_frac_br_zero": float(np.nanmean(top_decile)),
        }
    return agg


# fmt: off
@click.command()
@click.option("--dataset", required=True, help="Dataset name (e.g. chexpert).")
@click.option("--model", default=None, help="Model directory name. If omitted, all models under the dataset are processed.")
@click.option("--output-dir", default=None, type=click.Path(path_type=Path), help="Output directory. Defaults to results/experiment_2/<run_id>/.")
@click.option("--seed", default=0, show_default=True, help="RNG seed for bootstrap CI.")
# fmt: on
def main(dataset: str, model: str | None, output_dir: Path | None, seed: int) -> None:
    """Compute BR̂ dissociation metrics across all pair artefacts."""
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

    if output_dir is None:
        output_dir = experiment_results("experiment_2") / f"run_{token_hex(3)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    utils.setup_logging(output_dir, "experiment_2")
    logger.info("Experiment 2: BR^ dissociation on real reader data")
    utils.log_run_args(logger.info)
    rng = np.random.default_rng(seed)

    all_results: list[dict] = []
    for model_dir in model_dirs:
        pairs_dir = model_dir / "pairs"
        logger.info(f"=== Model: {model_dir.name} ({pairs_dir}) ===")
        try:
            results = run_pairs_dir(pairs_dir, rng)
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc
        all_results.extend(results)

    agg = aggregate(all_results)

    logger.info("=" * 60)
    logger.info(f"AGGREGATE ({agg['n_pairs']} pairs)")
    for reward_name in REWARDS:
        r = agg[reward_name]
        logger.info(
            f"  {reward_name}: mean_rho={r['mean_rho_ll_br']:+.3f}  "
            f"mean_rho_all={r['mean_rho_ll_br_all']:+.3f}  "
            f"n_valid={r['n_valid']}  "
            f"zero_br={r['mean_top_decile_frac_br_zero']:.1%}"
        )
    logger.info("=" * 60)

    out = {"results": all_results, "aggregate": agg}
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(out, indent=2, default=_json_default))
    logger.info(f"Results written to {results_path}")

    utils.update_latest_symlink(output_dir)
    logger.info(f"latest -> {output_dir.name}")
