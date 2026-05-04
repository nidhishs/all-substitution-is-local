#!/usr/bin/env python3
"""
Experiment 2 (Full): Boundary Regret on CheXpert – 5 readers × 4 conditions
=============================================================================

Full design:
  - 5 GT radiologists as h: bc1, bc2, bc3, bc5, bc7
  - 4 conditions: Cardiomegaly, Edema, Atelectasis, Pleural Effusion
  - For each (h_rad, condition): y = majority vote of the other 4 GT readers
  - 20 independent (reader, condition) experiments, N≈350 eval each

DenseNet inference runs once on all 500 images. Temperature calibration and
b_xh cross-fitting run independently per (condition, h_rad) pair (different y
→ different calibration → different b_x). Results are aggregated across all
20 pairs to produce the full Experiment 2 verdict.

Usage:
  cd source && python experiment2_full.py --data-dir data/
  cd source && python experiment2_full.py --data-dir data/ --skip-inference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from utils import REWARD_MATRICES, FACET_THRESHOLDS, boundary_regret, model_action, log_loss_gain


# ── Constants ─────────────────────────────────────────────────────────────────

GT_RADS = ["bc1", "bc2", "bc3", "bc5", "bc7"]

# CSV column name → (DenseNet pathology label, expected model index)
CONDITION_CONFIG: dict[str, dict] = {
    "Cardiomegaly":    {"model_label": "Cardiomegaly", "model_idx": 10},
    "Edema":           {"model_label": "Edema",        "model_idx": 4},
    "Atelectasis":     {"model_label": "Atelectasis",  "model_idx": 0},
    "Pleural Effusion":{"model_label": "Effusion",     "model_idx": 7},
}
CONDITIONS = list(CONDITION_CONFIG.keys())


# ── Data loading ──────────────────────────────────────────────────────────────

def load_labels(data_dir: Path) -> pd.DataFrame:
    """Load all GT reader labels for all conditions.

    Returns wide DataFrame: columns Study, image_path, bc1..bc7 × condition.
    """
    rad_dir = data_dir / "cheXpert-test-set-labels" / "radiologists" / "groundtruth"
    img_root = data_dir / "CheXpert" / "test"

    frames = []
    for rad in GT_RADS:
        df = pd.read_csv(rad_dir / f"{rad}_gt.csv", usecols=["Study"] + CONDITIONS)
        df = df.rename(columns={c: f"{rad}_{c}" for c in CONDITIONS})
        frames.append(df.set_index("Study"))

    merged = frames[0].join(frames[1:], how="inner").reset_index()
    study_prefix = "CheXpert-v1.0/test/"
    merged["image_path"] = (
        merged["Study"].str.replace(study_prefix, str(img_root) + "/", regex=False)
        + "/view1_frontal.jpg"
    )
    missing = [p for p in merged["image_path"] if not Path(p).is_file()]
    assert not missing, f"{len(missing)} image paths missing from disk"
    assert len(merged) == 500, f"Expected 500 studies, got {len(merged)}"
    return merged.reset_index(drop=True)


def make_pair_df(labels: pd.DataFrame, condition: str, h_rad: str) -> pd.DataFrame:
    """Build the (study_id, image_path, h, y) DataFrame for one (condition, h_rad) pair."""
    other_rads = [r for r in GT_RADS if r != h_rad]
    y_cols = [f"{r}_{condition}" for r in other_rads]
    df = pd.DataFrame({
        "study_id":   labels["Study"],
        "image_path": labels["image_path"],
        "h":          labels[f"{h_rad}_{condition}"].astype(int),
        "y":          (labels[y_cols].sum(axis=1) >= 2).astype(int),
    })
    assert df["h"].sum() >= 20, f"{h_rad}/{condition}: too few h positives ({df['h'].sum()})"
    assert df["y"].sum() >= 20, f"{h_rad}/{condition}: too few y positives ({df['y'].sum()})"
    return df.reset_index(drop=True)


# ── DenseNet inference (runs once for all conditions) ─────────────────────────

def run_inference_all(labels: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Run DenseNet on all 500 images; extract 4 condition raw probs.

    Saves inference.csv with columns: Study, b_x_raw_{condition}.
    Returns DataFrame with those columns.
    """
    import skimage.io
    import torch
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.eval()

    for cond, cfg in CONDITION_CONFIG.items():
        actual = model.pathologies[cfg["model_idx"]]
        assert actual == cfg["model_label"], (
            f"Index mismatch for {cond}: expected '{cfg['model_label']}', got '{actual}'"
        )

    def _preprocess(path: str):
        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 3:
            img = img.mean(2)
        img = img[np.newaxis, :, :]
        img = xrv.datasets.XRayCenterCrop()(img)
        img = xrv.datasets.XRayResizer(224)(img)
        return torch.from_numpy(img).unsqueeze(0)

    paths = labels["image_path"].tolist()
    n = len(paths)
    raw = {c: np.zeros(n, dtype=np.float32) for c in CONDITIONS}

    print(f"  Running DenseNet on {n} images ...")
    batch_size = 16
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_paths = paths[start:start + batch_size]
            tensors = torch.cat([_preprocess(p) for p in batch_paths], dim=0)
            out = model(tensors).numpy()
            for cond, cfg in CONDITION_CONFIG.items():
                raw[cond][start:start + len(batch_paths)] = out[:, cfg["model_idx"]]

    inference_df = pd.DataFrame({"Study": labels["Study"]})
    for cond in CONDITIONS:
        inference_df[f"b_x_raw_{cond}"] = raw[cond]

    output_dir.mkdir(parents=True, exist_ok=True)
    inference_df.to_csv(output_dir / "inference.csv", index=False)
    print(f"  Saved inference.csv ({n} rows × {len(CONDITIONS)} conditions)")
    return inference_df


def load_inference(output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / "inference.csv")


# ── Per-pair pipeline ─────────────────────────────────────────────────────────

def _nll(T: float, logits: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(1 / (1 + np.exp(-logits / T)), 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def calibrate(b_x_raw: np.ndarray, y: np.ndarray, cal_mask: np.ndarray) -> tuple[float, np.ndarray]:
    """Temperature-scale b_x_raw on cal_mask; return (T_opt, b_x)."""
    raw_clipped = np.clip(b_x_raw, 1e-7, 1 - 1e-7)
    logits = np.log(raw_clipped / (1 - raw_clipped))
    result = minimize_scalar(
        lambda T: _nll(T, logits[cal_mask], y[cal_mask].astype(float)),
        bounds=(0.1, 10.0), method="bounded",
    )
    T_opt = float(result.x)
    b_x = 1 / (1 + np.exp(-logits / T_opt))
    return T_opt, b_x


def _mce(b_x: np.ndarray, y: np.ndarray) -> float:
    quantiles = np.percentile(b_x, np.linspace(0, 100, 11))
    bin_ids = np.digitize(b_x, quantiles[1:-1])
    errs = []
    for i in range(10):
        mask = bin_ids == i
        if mask.sum() > 0:
            errs.append(abs(b_x[mask].mean() - y[mask].mean()))
    return float(np.max(errs)) if errs else float("nan")


def fit_augmented(b_x: np.ndarray, h: np.ndarray, y: np.ndarray,
                  eval_mask: np.ndarray) -> np.ndarray:
    """5-fold cross-fitted b_xh on eval set. Returns b_xh for all rows (NaN for cal)."""
    b_x_eval = b_x[eval_mask]
    h_eval   = h[eval_mask].astype(float)
    y_eval   = y[eval_mask].astype(int)

    eps = 1e-7
    X = np.column_stack([
        np.log(np.clip(b_x_eval, eps, 1 - eps) / np.clip(1 - b_x_eval, eps, 1 - eps)),
        h_eval,
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    b_xh_eval = np.full(len(y_eval), np.nan)
    for train_idx, val_idx in skf.split(X, y_eval):
        clf = LogisticRegression(max_iter=1000).fit(X[train_idx], y_eval[train_idx])
        b_xh_eval[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    assert not np.isnan(b_xh_eval).any()
    b_xh = np.full(len(b_x), np.nan)
    b_xh[eval_mask] = b_xh_eval
    return b_xh


def compute_br(b_x: np.ndarray, b_xh: np.ndarray, h: np.ndarray,
               y: np.ndarray, eval_mask: np.ndarray) -> pd.DataFrame:
    """Compute BR_hat and log-loss gain for eval rows."""
    bx_e  = b_x[eval_mask]
    bxh_e = b_xh[eval_mask]
    y_e   = y[eval_mask].astype(int)

    bx_2d  = np.column_stack([1 - bx_e,  bx_e])
    bxh_2d = np.column_stack([1 - bxh_e, bxh_e])

    rows = {}
    for name, R in REWARD_MATRICES.items():
        rows[f"BR_hat_{name}"] = boundary_regret(bx_2d, bxh_2d, R)
        rows[f"a_x_{name}"]    = model_action(bx_2d, R)
    rows["log_loss_gain"] = log_loss_gain(bx_e, bxh_e, y_e)
    rows["b_x"]  = bx_e
    rows["b_xh"] = bxh_e
    rows["h"]    = h[eval_mask]
    rows["y"]    = y_e
    return pd.DataFrame(rows)


def _bootstrap_spearman_ci(x, y_arr, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    rhos = [spearmanr(x[idx := rng.integers(0, len(x), len(x))], y_arr[idx]).statistic
            for _ in range(n)]
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def compute_stats(eval_df: pd.DataFrame) -> dict:
    lkg = eval_df["log_loss_gain"].values
    b_x = eval_df["b_x"].values
    stats = {}

    for name, threshold in FACET_THRESHOLDS.items():
        br = eval_df[f"BR_hat_{name}"].values
        dist = np.abs(b_x - threshold)
        all_zero = bool((br == 0).all())
        if all_zero:
            # Constant BR = 0: correlation undefined; treat as rho = -1 (strongest methods signal)
            rho, pval, lo, hi = -1.0, 0.0, -1.0, -1.0
            dist_rho = float(spearmanr(dist, br).statistic) if not all_zero else float("nan")
        else:
            rho, pval = spearmanr(lkg, br)
            lo, hi = _bootstrap_spearman_ci(lkg, br)
            dist_rho, _ = spearmanr(dist, br)
        top_mask = lkg >= np.percentile(lkg, 90)
        frac_zero_br = float((br[top_mask] == 0).mean()) if top_mask.sum() else float("nan")
        stats[name] = {
            "rho": float(rho), "pval": float(pval),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "frac_zero_br": frac_zero_br,
            "facet_dist_rho": float(dist_rho) if not np.isnan(dist_rho) else float("nan"),
            "all_zero_br": all_zero,
        }

    top70 = lkg >= np.percentile(lkg, 70)
    br_r1, br_r2 = eval_df["BR_hat_R1"].values, eval_df["BR_hat_R2"].values
    stats["reward_sensitivity"] = float(((br_r1 > 0) & (br_r2 == 0) & top70).sum() / top70.sum())
    return stats


def _is_strongly_methods(stats: dict) -> bool:
    r2 = stats["R2"]
    # all_zero_br means BR=0 everywhere — trivially satisfies both conditions
    if r2.get("all_zero_br", False):
        return True
    return r2["rho"] < 0.70 and r2["frac_zero_br"] >= 0.30


def run_pair(
    pair_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    condition: str,
    h_rad: str,
    output_dir: Path,
) -> dict:
    """Run the full pipeline for one (condition, h_rad) pair. Returns stats dict."""
    tag = f"{condition.replace(' ', '_')}_{h_rad}"
    pair_out = output_dir / tag
    pair_out.mkdir(parents=True, exist_ok=True)

    # Align inference to pair_df via Study column
    inf_aligned = inference_df.set_index("Study").loc[pair_df["study_id"].values]
    b_x_raw = inf_aligned[f"b_x_raw_{condition}"].values.astype(np.float32)

    y   = pair_df["y"].values.astype(int)
    h   = pair_df["h"].values.astype(int)

    # Cal/eval split stratified by this pair's y
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.70, random_state=42)
    cal_idx, _ = next(sss.split(np.zeros(len(y)), y))
    cal_mask  = np.zeros(len(y), dtype=bool)
    cal_mask[cal_idx] = True
    eval_mask = ~cal_mask

    T_opt, b_x = calibrate(b_x_raw, y, cal_mask)
    mce = _mce(b_x[eval_mask], y[eval_mask])

    b_xh = fit_augmented(b_x, h, y, eval_mask)
    eval_df = compute_br(b_x, b_xh, h, y, eval_mask)

    n_eval = eval_mask.sum()
    h_acc  = float((h[eval_mask] == y[eval_mask]).mean())
    bx_acc = float(((b_x[eval_mask] > 0.5).astype(int) == y[eval_mask]).mean())

    stats = compute_stats(eval_df)
    verdict = "strongly_methods" if _is_strongly_methods(stats) else "other"

    print(
        f"  [{tag:45s}] ρ_R2={stats['R2']['rho']:+.3f}  "
        f"zero_br={stats['R2']['frac_zero_br']:.1%}  "
        f"T={T_opt:.3f}  MCE={mce:.3f}  {verdict}"
    )

    return {
        "condition": condition,
        "h_rad": h_rad,
        "n_eval": int(n_eval),
        "h_positives": int(h[eval_mask].sum()),
        "y_positives": int(y[eval_mask].sum()),
        "h_accuracy": h_acc,
        "b_x_accuracy": bx_acc,
        "T_opt": T_opt,
        "MCE": mce,
        "stats": stats,
        "verdict": verdict,
    }


# ── Aggregation and reporting ─────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    rho_r2   = np.array([r["stats"]["R2"]["rho"] for r in results], dtype=float)
    zero_br  = np.array([r["stats"]["R2"]["frac_zero_br"] for r in results], dtype=float)
    rho_r1   = np.array([r["stats"]["R1"]["rho"] for r in results], dtype=float)
    rho_r3   = np.array([r["stats"]["R3"]["rho"] for r in results], dtype=float)
    fdist_r2 = np.array([r["stats"]["R2"]["facet_dist_rho"] for r in results], dtype=float)
    n_sm     = sum(1 for r in results if r["verdict"] == "strongly_methods")
    return {
        "n_pairs": len(results),
        "n_strongly_methods": n_sm,
        "frac_strongly_methods": n_sm / len(results),
        "mean_rho_R1": float(np.nanmean(rho_r1)),
        "mean_rho_R2": float(np.nanmean(rho_r2)),
        "mean_rho_R3": float(np.nanmean(rho_r3)),
        "median_rho_R2": float(np.nanmedian(rho_r2)),
        "mean_frac_zero_br_R2": float(np.nanmean(zero_br)),
        "mean_facet_dist_rho_R2": float(np.nanmean(fdist_r2)),
        "rho_R2_by_pair": rho_r2.tolist(),
        "zero_br_by_pair": zero_br.tolist(),
    }


def write_report(results: list[dict], agg: dict, output_dir: Path) -> None:
    lines = [
        "# Experiment 2 (Full) — Summary Report",
        "",
        f"**Design:** {len(GT_RADS)} readers × {len(CONDITIONS)} conditions = {len(results)} (reader, condition) pairs",
        f"**Readers (h):** {', '.join(GT_RADS)}",
        f"**Conditions:** {', '.join(CONDITIONS)}",
        "",
        "## Per-pair Results",
        "",
        "| condition | h_rad | n_eval | y+ | h+ | T_opt | MCE | ρ(R1) | ρ(R2) | ρ(R3) | zero_BR_R2 | facet_dist_R2 | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r["stats"]
        lines.append(
            f"| {r['condition']} | {r['h_rad']} | {r['n_eval']} "
            f"| {r['y_positives']} | {r['h_positives']} "
            f"| {r['T_opt']:.3f} | {r['MCE']:.3f} "
            f"| {s['R1']['rho']:+.3f} | {s['R2']['rho']:+.3f} | {s['R3']['rho']:+.3f} "
            f"| {s['R2']['frac_zero_br']:.1%} | {s['R2']['facet_dist_rho']:+.3f} "
            f"| {r['verdict']} |"
        )

    lines += [
        "",
        "## Aggregate",
        "",
        f"- Pairs with 'strongly_methods' verdict: **{agg['n_strongly_methods']}/{agg['n_pairs']}** ({agg['frac_strongly_methods']:.1%})",
        f"- Mean Spearman ρ(lkg, BR_R2) across pairs: **{agg['mean_rho_R2']:+.3f}**",
        f"- Median Spearman ρ(lkg, BR_R2): **{agg['median_rho_R2']:+.3f}**",
        f"- Mean fraction zero-BR (top decile, R2): **{agg['mean_frac_zero_br_R2']:.1%}**",
        f"- Mean facet-distance ρ (R2): **{agg['mean_facet_dist_rho_R2']:+.3f}**",
        "",
        "## Verdict",
        "",
    ]

    if agg["frac_strongly_methods"] >= 0.70:
        lines.append(
            f"**Strongly methods** — {agg['frac_strongly_methods']:.0%} of pairs satisfy the pre-registered criterion "
            f"(ρ_R2 < 0.70 AND top-decile zero-BR ≥ 30%)."
        )
    elif agg["frac_strongly_methods"] >= 0.50:
        lines.append(
            f"**Mixed (majority)** — {agg['frac_strongly_methods']:.0%} of pairs pass. "
            "The directional claim holds for most conditions and readers, but not uniformly."
        )
    else:
        lines.append(
            f"**Inconclusive** — only {agg['frac_strongly_methods']:.0%} of pairs pass. "
            "Revisit reward structure or check calibration."
        )

    report = "\n".join(lines) + "\n"
    out_path = output_dir / "experiment2_full_report.md"
    out_path.write_text(report)
    print(f"\nReport written to {out_path}")


def scatter_rho_matrix(results: list[dict], output_dir: Path) -> None:
    """Heatmap of Spearman ρ (R2) across conditions × readers."""
    rho_matrix = np.zeros((len(CONDITIONS), len(GT_RADS)))
    for r in results:
        ci = CONDITIONS.index(r["condition"])
        ri = GT_RADS.index(r["h_rad"])
        rho_matrix[ci, ri] = r["stats"]["R2"]["rho"]

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(rho_matrix, vmin=-0.5, vmax=0.5, cmap="RdBu", aspect="auto")
    ax.set_xticks(range(len(GT_RADS)));   ax.set_xticklabels(GT_RADS)
    ax.set_yticks(range(len(CONDITIONS))); ax.set_yticklabels(CONDITIONS)
    for ci in range(len(CONDITIONS)):
        for ri in range(len(GT_RADS)):
            ax.text(ri, ci, f"{rho_matrix[ci,ri]:+.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Spearman ρ (lkg vs BR_R2)")
    ax.set_title("Spearman ρ(log-loss gain, BR̂) under R2\n(target: < 0, confirms methods claim)")
    fig.tight_layout()
    fig.savefig(output_dir / "rho_r2_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to {output_dir / 'rho_r2_heatmap.png'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 (Full): 5 readers × 4 conditions")
    parser.add_argument("--data-dir",       type=Path, default=Path("data"))
    parser.add_argument("--output-dir",     type=Path, default=Path("results/full_exp2"))
    parser.add_argument("--skip-inference", action="store_true",
                        help="Load existing inference.csv (re-use prior DenseNet run)")
    parser.add_argument("--conditions",     nargs="+", default=CONDITIONS,
                        help=f"Subset of conditions to run (default: all {len(CONDITIONS)})")
    parser.add_argument("--readers",        nargs="+", default=GT_RADS,
                        help=f"Subset of readers to use as h (default: all {len(GT_RADS)})")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading CheXpert labels ...")
    labels = load_labels(args.data_dir)
    print(f"  Loaded {len(labels)} studies, {len(GT_RADS)} readers, {len(CONDITIONS)} conditions")

    if args.skip_inference:
        print("Step 2: Loading existing inference.csv ...")
        inference_df = load_inference(args.output_dir)
    else:
        print("Step 2: Running DenseNet inference on all images ...")
        inference_df = run_inference_all(labels, args.output_dir)

    # Align labels → inference on Study column
    inference_df = inference_df.set_index("Study").loc[labels["Study"].values].reset_index()

    print(f"\nStep 3: Running {len(args.conditions)} × {len(args.readers)} = "
          f"{len(args.conditions)*len(args.readers)} (condition, reader) pairs ...")

    results = []
    for condition in args.conditions:
        for h_rad in args.readers:
            pair_df = make_pair_df(labels, condition, h_rad)
            # pass study IDs aligned to inference
            pair_df["study_id"] = labels["Study"].values
            result = run_pair(pair_df, inference_df, condition, h_rad, args.output_dir)
            results.append(result)

    agg = aggregate(results)

    print(f"\n{'='*60}")
    print(f"AGGREGATE ({agg['n_pairs']} pairs)")
    print(f"  Strongly methods: {agg['n_strongly_methods']}/{agg['n_pairs']} ({agg['frac_strongly_methods']:.0%})")
    print(f"  Mean ρ_R2:        {agg['mean_rho_R2']:+.3f}")
    print(f"  Median ρ_R2:      {agg['median_rho_R2']:+.3f}")
    print(f"  Mean zero_BR_R2:  {agg['mean_frac_zero_br_R2']:.1%}")
    print(f"{'='*60}")

    out = {"results": results, "aggregate": agg}
    json_path = args.output_dir / "experiment2_full_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"Full results saved to {json_path}")

    scatter_rho_matrix(results, args.output_dir)
    write_report(results, agg, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
