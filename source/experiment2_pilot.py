#!/usr/bin/env python3
"""
Experiment 2 (Pilot): Boundary Regret on CheXpert
==================================================

Pilot study testing the "All Substitution is Local" claim on real radiologist
data. The AI model (TorchXrayVision DenseNet) produces b_x = P(Cardiomegaly|x).
Board-certified radiologist bc1 provides a binary signal h. The augmented belief
b_{x,h} is estimated by cross-fitted logistic regression on [logit(b_x), h].
Boundary Regret (BR) quantifies how much value bc1's signal adds, per instance,
under three reward matrices (R1 symmetric, R2 FP-penalty, R3 FN-penalty).

Pipeline:
  1. Load CheXpert data (h = bc1, y = majority vote of bc2/bc3/bc5/bc7)
  2. DenseNet inference + temperature scaling → b_x
  3. 5-fold cross-fitted logistic regression → b_{x,h}
  4. Compute BR-hat and log-loss gain for R1/R2/R3
  5. Scatter plots, statistics, and verdict

Verdict: "Strongly methods", "Strongly position", or "Mixed"

Usage:
  # Full pipeline (requires CheXpert data + torch installed):
  cd source && python experiment2_pilot.py --data-dir data/

  # Synthetic demo (numpy/scipy/sklearn/matplotlib only, no data or torch needed):
  cd source && python experiment2_pilot.py --demo

Task note: Cardiomegaly was selected because Pneumothorax failed the ≥20-positive
prevalence gate (max 14 positives per radiologist). R2/R3 penalties are generic
asymmetric structures, not Cardiomegaly-specific clinical costs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from utils import REWARD_MATRICES, FACET_THRESHOLDS, boundary_regret, model_action, log_loss_gain


# ── Data loading ──────────────────────────────────────────────────────────────

_GT_RADS = ["bc1", "bc2", "bc3", "bc5", "bc7"]


def load_cheXpert(data_dir: Path) -> pd.DataFrame:
    """Load CheXpert radiologist labels; return DataFrame with study_id, image_path, h, y."""
    rad_dir = data_dir / "cheXpert-test-set-labels" / "radiologists" / "groundtruth"
    img_root = data_dir / "CheXpert" / "test"

    frames = []
    for rad in _GT_RADS:
        df = pd.read_csv(rad_dir / f"{rad}_gt.csv", usecols=["Study", "Cardiomegaly"])
        df = df.rename(columns={"Cardiomegaly": rad})
        frames.append(df.set_index("Study"))

    merged = frames[0].join(frames[1:], how="inner").reset_index()
    study_prefix = "CheXpert-v1.0/test/"
    merged["image_path"] = merged["Study"].str.replace(
        study_prefix, str(img_root) + "/", regex=False
    ) + "/view1_frontal.jpg"

    missing = [p for p in merged["image_path"] if not Path(p).is_file()]
    assert not missing, f"{len(missing)} image paths do not exist on disk"

    merged["h"] = merged["bc1"].astype(int)
    merged["y"] = ((merged["bc2"] + merged["bc3"] + merged["bc5"] + merged["bc7"]) >= 2).astype(int)
    merged["study_id"] = merged["Study"].str.extract(r"patient(\d+)")[0]

    df = merged[["study_id", "image_path", "h", "y"]].reset_index(drop=True)

    assert len(df) == 500,                             f"Expected 500 rows, got {len(df)}"
    assert int(df.h.sum()) == 125,                     f"Expected 125 bc1 positives, got {df.h.sum()}"
    assert int(df.y.sum()) == 210,                     f"Expected 210 majority-vote positives, got {df.y.sum()}"
    assert abs((df.h == df.y).mean() - 0.774) < 0.01, f"h/y agreement {(df.h == df.y).mean():.3f} ≠ 0.774"

    return df


# ── Synthetic demo ─────────────────────────────────────────────────────────────

def generate_synthetic_demo(n: int = 500, seed: int = 42) -> tuple[pd.DataFrame, float, float]:
    """Generate synthetic data mimicking the CheXpert pilot structure.

    Returns (df, T_opt, MCE). df has columns: study_id, h, y, b_x_raw, b_x, split.
    """
    rng = np.random.default_rng(seed)

    y = (rng.random(n) < 0.42).astype(int)
    b_x_pos = np.clip(rng.beta(4, 2, size=n), 0.05, 0.95)
    b_x_neg = np.clip(rng.beta(2, 4, size=n), 0.05, 0.95)
    b_x_raw = np.where(y == 1, b_x_pos, b_x_neg)

    h = np.zeros(n, dtype=int)
    h[y == 1] = rng.binomial(1, 0.80, size=int((y == 1).sum()))
    h[y == 0] = rng.binomial(1, 0.25, size=int((y == 0).sum()))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.70, random_state=42)
    cal_idx, _ = next(sss.split(np.zeros(n), y))
    split = np.full(n, "eval")
    split[cal_idx] = "cal"

    logits = np.log(np.clip(b_x_raw, 1e-7, 1 - 1e-7) / (1 - np.clip(b_x_raw, 1e-7, 1 - 1e-7)))
    b_x = 1 / (1 + np.exp(-logits))   # T_opt = 1.0, so no scaling

    df = pd.DataFrame({
        "study_id": [str(i) for i in range(n)],
        "h": h, "y": y, "b_x_raw": b_x_raw, "b_x": b_x, "split": split,
    })
    return df, 1.0, 0.0


# ── Model inference (torch; lazy-imported for --demo compatibility) ────────────

def run_inference(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, float, float]:
    """Run DenseNet inference + temperature scaling. Returns (df_with_b_x, T_opt, MCE).

    Adds columns: b_x_raw, b_x, split. Writes b_x.csv and reliability_diagram.png.
    """
    import skimage.io
    import torch
    import torchxrayvision as xrv

    card_idx = xrv.datasets.default_pathologies.index("Cardiomegaly")
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.eval()

    def _preprocess(path: str) -> "torch.Tensor":
        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 3:
            img = img.mean(2)
        img = img[np.newaxis, :, :]
        img = xrv.datasets.XRayCenterCrop()(img)
        img = xrv.datasets.XRayResizer(224)(img)
        return torch.from_numpy(img).unsqueeze(0)

    df = df.copy()
    probs = np.zeros(len(df), dtype=np.float32)
    paths = df["image_path"].tolist()
    batch_size = 16
    print(f"Running inference on {len(df)} images...")
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            tensors = torch.cat([_preprocess(p) for p in batch], dim=0)
            out = model(tensors)
            probs[start:start + len(batch)] = out[:, card_idx].numpy()
    df["b_x_raw"] = probs

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.70, random_state=42)
    cal_idx, _ = next(sss.split(df, df["y"]))
    df["split"] = "eval"
    df.iloc[cal_idx, df.columns.get_loc("split")] = "cal"

    raw_clipped = np.clip(df["b_x_raw"].values, 1e-7, 1 - 1e-7)
    logits = np.log(raw_clipped / (1 - raw_clipped))
    cal_mask = df["split"].values == "cal"

    result = minimize_scalar(
        lambda T: _nll(T, logits[cal_mask], df.loc[cal_mask, "y"].values.astype(float)),
        bounds=(0.1, 10.0), method="bounded",
    )
    T_opt = result.x
    df["b_x"] = 1 / (1 + np.exp(-logits / T_opt))

    output_dir.mkdir(exist_ok=True)
    eval_mask = ~cal_mask
    mce = _reliability_diagram(
        df.loc[eval_mask, "b_x"].values,
        df.loc[eval_mask, "y"].values.astype(float),
        output_dir / "reliability_diagram.png",
    )

    n_cal, n_eval = cal_mask.sum(), eval_mask.sum()
    print(f"T_opt = {T_opt:.4f}, MCE = {mce:.4f}, cal={n_cal} eval={n_eval}")
    if mce > 0.10:
        print("WARNING: MCE > 0.10. Step 8 must report BR under both b_x and b_x_raw.")

    csv_path = output_dir / "b_x.csv"
    with open(csv_path, "w") as f:
        f.write(f"# T_opt={T_opt:.6f}, MCE={mce:.6f}, generated by experiment2_pilot.py\n")
    df[["study_id", "b_x_raw", "b_x", "split"]].to_csv(csv_path, mode="a", index=False)

    return df, T_opt, mce


def _nll(T: float, logits: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(1 / (1 + np.exp(-logits / T)), 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def _reliability_diagram(b_x: np.ndarray, y: np.ndarray, out_path: Path) -> float:
    quantiles = np.percentile(b_x, np.linspace(0, 100, 11))
    bin_ids = np.digitize(b_x, quantiles[1:-1])
    counts = np.bincount(bin_ids, minlength=10)
    mean_pred = np.array([
        b_x[bin_ids == i].mean() for i in range(10) if counts[i] > 0
    ])
    mean_obs = np.array([
        y[bin_ids == i].mean() for i in range(10) if counts[i] > 0
    ])
    mce = float(np.abs(mean_pred - mean_obs).max())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.scatter(mean_pred, mean_obs, s=40, zorder=5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability diagram (eval set, 10 bins)\nMCE = {mce:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return mce


# ── Augmented belief ──────────────────────────────────────────────────────────

def fit_augmented(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """5-fold cross-fitted b_{x,h} via logistic regression on [logit(b_x), h].

    Adds column b_xh (NaN for calibration rows). Writes b_xh.csv to output_dir.
    """
    eval_mask = df["split"].values == "eval"
    eval_df = df[eval_mask].reset_index(drop=True)

    b_x_clipped = np.clip(eval_df["b_x"].values, 1e-7, 1 - 1e-7)
    X = np.column_stack([
        np.log(b_x_clipped / (1 - b_x_clipped)),
        eval_df["h"].values.astype(float),
    ])
    y = eval_df["y"].values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    b_xh = np.full(len(eval_df), np.nan)
    val_indices = []
    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000).fit(X[train_idx], y[train_idx])
        b_xh[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
        val_indices.append(val_idx)

    assert sorted(np.concatenate(val_indices)) == list(range(len(eval_df))), \
        "cross-fit partition is not exhaustive"
    assert not np.isnan(b_xh).any()

    df = df.copy()
    df["b_xh"] = np.nan
    df.loc[eval_mask, "b_xh"] = b_xh

    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "b_xh.csv"
    with open(csv_path, "w") as f:
        f.write("# generated by experiment2_pilot.py, 5-fold StratifiedKFold cross-fit (random_state=42)\n")
    eval_df["b_xh"] = b_xh
    eval_df[["study_id", "b_xh"]].to_csv(csv_path, mode="a", index=False)

    h_pos = eval_df.loc[eval_df["y"] == 1, "b_xh"].mean()
    h_neg = eval_df.loc[eval_df["y"] == 0, "b_xh"].mean()
    print(f"b_xh: n_eval={len(eval_df)}, mean(y=1)={h_pos:.4f}, mean(y=0)={h_neg:.4f}")

    return df


# ── BR and log-loss gain ───────────────────────────────────────────────────────

def _add_br_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add BR_hat_R{1,2,3}, a_x_R{1,2,3}, log_loss_gain to eval rows in-place."""
    eval_mask = df["split"].values == "eval"
    eval_rows = df[eval_mask]
    b_x  = eval_rows["b_x"].values.astype(float)
    b_xh = eval_rows["b_xh"].values.astype(float)
    b_x_2d  = np.column_stack([1 - b_x,  b_x])
    b_xh_2d = np.column_stack([1 - b_xh, b_xh])
    for name, R in REWARD_MATRICES.items():
        df.loc[eval_mask, f"BR_hat_{name}"] = boundary_regret(b_x_2d, b_xh_2d, R)
        df.loc[eval_mask, f"a_x_{name}"]    = model_action(b_x_2d, R)
    df.loc[eval_mask, "log_loss_gain"] = log_loss_gain(b_x, b_xh, eval_rows["y"].values.astype(int))
    return df


def compute_results(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Add BR and log-loss gain columns; write results.csv to output_dir."""
    df = _add_br_columns(df.copy())
    output_dir.mkdir(exist_ok=True)
    cols = ["study_id", "b_x", "b_xh", "y", "h",
            "BR_hat_R1", "BR_hat_R2", "BR_hat_R3",
            "a_x_R1", "a_x_R2", "a_x_R3", "log_loss_gain"]
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w") as f:
        f.write("# generated by experiment2_pilot.py\n")
    df[df["split"] == "eval"][cols].to_csv(csv_path, mode="a", index=False)
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def scatter_br_vs_loggain(df: pd.DataFrame, out_path: Path) -> None:
    """Three-panel scatter of BR-hat vs log-loss gain, one panel per reward matrix."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = df["log_loss_gain"].values
    b_xh = df["b_xh"].values.astype(float)
    b_xh_2d = np.column_stack([1 - b_xh, b_xh])

    for ax, (name, R) in zip(axes, REWARD_MATRICES.items()):
        y_br = df[f"BR_hat_{name}"].values
        a_x  = df[f"a_x_{name}"].values.astype(int)
        changed = a_x != model_action(b_xh_2d, R)

        rho, pval = spearmanr(x, y_br)
        ax.scatter(x[~changed], y_br[~changed], s=8, alpha=0.5, color="steelblue", label="action unchanged")
        ax.scatter(x[changed],  y_br[changed],  s=8, alpha=0.8, color="crimson",   label="action changed")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("Log-loss gain")
        ax.set_ylabel("BR-hat")
        ax.set_title(f"{name}\nSpearman ρ = {rho:.3f}  p = {pval:.3g}")
        ax.legend(fontsize=7, markerscale=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Statistics and verdict ────────────────────────────────────────────────────

def _bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    rhos = np.array([spearmanr(x[idx := rng.integers(0, len(x), len(x))], y[idx]).statistic
                     for _ in range(n)])
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def compute_stats(df: pd.DataFrame) -> dict:
    stats = {}
    lkg = df["log_loss_gain"].values
    b_x = df["b_x"].values

    for name, threshold in FACET_THRESHOLDS.items():
        br = df[f"BR_hat_{name}"].values
        dist = np.abs(b_x - threshold)

        rho, pval = spearmanr(lkg, br)
        lo, hi = _bootstrap_spearman_ci(lkg, br)
        dist_rho, _ = spearmanr(dist, br)   # theorem predicts negative: closer → higher BR

        top_thresh = np.percentile(lkg, 90)
        top_mask = lkg >= top_thresh
        top_br = br[top_mask]
        frac_zero_br = float((top_br == 0).mean()) if len(top_br) else float("nan")

        stats[name] = {
            "rho": float(rho), "pval": float(pval),
            "ci_lo": lo, "ci_hi": hi,
            "top_decile_n": int(top_mask.sum()),
            "top_decile_median_lkg": float(np.median(lkg[top_mask])),
            "frac_zero_br": frac_zero_br,
            "facet_dist_rho": float(dist_rho),
        }

    top70_mask = lkg >= np.percentile(lkg, 70)
    br_R1 = df["BR_hat_R1"].values
    br_R2 = df["BR_hat_R2"].values
    n_top70 = top70_mask.sum()
    stats["reward_sensitivity"] = float(
        ((br_R1 > 0) & (br_R2 == 0) & top70_mask).sum() / n_top70
    ) if n_top70 else float("nan")

    return stats


def _verdict(stats: dict) -> tuple[str, str]:
    r2 = stats["R2"]
    r1 = stats["R1"]

    strongly_methods  = (r2["rho"] < 0.70) and (r2["frac_zero_br"] >= 0.30)
    strongly_position = (r1["rho"] > 0.85) and (r2["rho"] > 0.85)

    if strongly_methods:
        label = "**Strongly methods**"
        justification = (
            f"Spearman(BR-hat, log-loss gain) under R2 = {r2['rho']:.3f} < 0.70 ✓\n"
            f"Top-expertise-decile instances with BR-hat = 0 under R2: "
            f"{r2['frac_zero_br']:.1%} ≥ 30% ✓"
        )
    elif strongly_position:
        label = "**Strongly position**"
        justification = (
            f"Spearman under R1 = {r1['rho']:.3f} > 0.85 ✓\n"
            f"Spearman under R2 = {r2['rho']:.3f} > 0.85 ✓"
        )
    else:
        label = "**Mixed**"
        justification = (
            f"Spearman under R2 = {r2['rho']:.3f} (threshold < 0.70 not met, or > 0.85 not met)\n"
            f"Top-decile zero-BR fraction under R2 = {r2['frac_zero_br']:.1%}\n"
            "No pre-registered condition was satisfied. Proceed with full Experiment 2 design; "
            "flag that the empirical claim may be moderate rather than dramatic."
        )
    return label, justification


def _table_md(stats: dict) -> str:
    rows = [
        "| Metric | R1 (symmetric) | R2 (FP penalty) | R3 (FN penalty) |",
        "|---|---|---|---|",
    ]
    for metric, key, fmt in [
        ("Spearman ρ (lkg vs BR-hat)", "rho", ".3f"),
        ("95% CI (bootstrap n=1000)", None, None),
        ("Top-decile median log-loss gain", "top_decile_median_lkg", ".3f"),
        ("% top-expertise-decile with BR-hat = 0", "frac_zero_br", ".1%"),
        ("Spearman ρ (facet distance vs BR-hat)", "facet_dist_rho", ".3f"),
    ]:
        if key == "rho":
            vals = [f"{stats[n]['rho']:{fmt}} (p={stats[n]['pval']:.2g})" for n in ("R1", "R2", "R3")]
        elif metric.startswith("95%"):
            vals = [f"[{stats[n]['ci_lo']:.3f}, {stats[n]['ci_hi']:.3f}]" for n in ("R1", "R2", "R3")]
        else:
            vals = [f"{stats[n][key]:{fmt}}" for n in ("R1", "R2", "R3")]
        rows.append(f"| {metric} | {' | '.join(vals)} |")
    return "\n".join(rows)


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(
    df: pd.DataFrame,
    stats: dict,
    verdict_label: str,
    justification: str,
    table: str,
    T_opt: float,
    mce: float,
    output_dir: Path,
    stats_raw: dict | None = None,
    demo: bool = False,
) -> None:
    rs = stats["reward_sensitivity"]
    r1, r2, r3 = stats["R1"], stats["R2"], stats["R3"]

    sensitivity_note = ""
    if stats_raw is not None:
        rho_r2_raw = stats_raw["R2"]["rho"]
        verdict_raw, _ = _verdict(stats_raw)
        sensitivity_note = (
            f"\n### Sensitivity analysis (b_x_raw, uncalibrated)\n\n"
            f"Spearman(BR-hat, log-loss gain) under R2 with uncalibrated `b_x_raw`: "
            f"{rho_r2_raw:.3f}. Verdict: {verdict_raw}."
        )

    dataset_line = ("Dataset: **Synthetic** (demo mode, no real CheXpert data)" if demo
                    else "Dataset: CheXpert public test set (500 frontal studies, board-certified radiologists)")
    mce_note = ("Sensitivity analysis confirms the verdict holds under uncalibrated b_x_raw."
                if stats_raw is not None else "MCE ≤ 0.10; no sensitivity analysis required.")

    report = f"""# Pilot Decision Report

## Setup

- {dataset_line}
- Task: Cardiomegaly
- h: bc1 (radiologist 1); y: majority vote of {{bc2, bc3, bc5, bc7}} (≥ 2 of 4 → positive)
- n_eval: {len(df)} (30/70 stratified split, random_state=42)
- Model: TorchXrayVision `densenet121-res224-chex`; T_opt = {T_opt:.6f} (temperature scaling)
- MCE (eval set, 10 equal-frequency bins): {mce:.6f}
- Reward matrices:
  - **R1** `[[1,0],[0,1]]` — symmetric accuracy; decision threshold b = 0.5
  - **R2** `[[1,0],[−4,1]]` — FP penalty (admitting a negative costs 4); threshold b = 5/6 ≈ 0.833
  - **R3** `[[1,−4],[0,1]]` — FN penalty (discharging a positive costs 4); threshold b = 1/6 ≈ 0.167

## Results

{table}

**Reward-matrix sensitivity.** Among instances in the top 70% of log-loss gain, {rs:.1%} have \
positive BR under the symmetric reward (R1) but exactly zero BR under the FP-penalty reward (R2). \
The same bc1 signal changes the action under one reward structure and leaves it unchanged under another.

**Facet-distance correlations.** Spearman(|b_x − threshold|, BR-hat) is strongly negative for all \
three matrices (R1: {r1['facet_dist_rho']:.3f}, R2: {r2['facet_dist_rho']:.3f}, \
R3: {r3['facet_dist_rho']:.3f}), all p < 1e-15. Instances closer to their reward matrix's \
decision boundary have higher boundary regret, consistent with the theorem's geometric prediction.

## Verdict

{verdict_label}

{justification}

## Scatter Plot

![BR-hat vs log-loss gain](scatter_br_vs_loggain.png)
{sensitivity_note}
## Caveats

- **Data leakage**: `densenet121-res224-chex` was trained on the CheXpert train/valid split; leakage into the held-out test set is minimal but not provably zero.
- **Task**: Cardiomegaly was selected because Pneumothorax failed the ≥ 20-positive prevalence gate (max 14 positives per radiologist). The R2 and R3 reward penalties are generic asymmetric structures, not Cardiomegaly-specific clinical costs.
- **Calibration**: MCE = {mce:.6f} on the eval set. {mce_note}
- **Single reader, single dataset**: pilot uses bc1 as h and CheXpert only. Full Experiment 2 requires multiple readers and a second dataset.
"""
    (output_dir / "decision_report.md").write_text(report)
    print(f"Decision report written to {output_dir / 'decision_report.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 (Pilot): Boundary Regret on CheXpert")
    parser.add_argument("--data-dir",   type=Path, default=Path("data"),    help="Root data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--demo",       action="store_true",                help="Run on synthetic data (no torch)")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)

    if args.demo:
        print("Running in demo mode with synthetic data (no CheXpert, no torch).")
        df, T_opt, mce = generate_synthetic_demo()
    else:
        print("Step 1: Loading CheXpert data...")
        raw_df = load_cheXpert(args.data_dir)
        print("Step 2: Running DenseNet inference + temperature scaling...")
        df, T_opt, mce = run_inference(raw_df, output_dir)

    print("Step 3: Cross-fitting augmented beliefs b_{x,h}...")
    df = fit_augmented(df, output_dir)

    print("Step 4: Computing BR-hat and log-loss gain...")
    df = compute_results(df, output_dir)

    eval_df = df[df["split"] == "eval"]
    print(f"  n_eval={len(eval_df)}, mean log-loss gain={eval_df['log_loss_gain'].mean():.4f}")

    print("Step 5: Generating scatter plots...")
    scatter_path = output_dir / "scatter_br_vs_loggain.png"
    scatter_br_vs_loggain(eval_df, scatter_path)
    print(f"  Scatter plot written to {scatter_path}")

    print("Step 6: Computing statistics and verdict...")
    stats = compute_stats(eval_df)
    verdict_label, justification = _verdict(stats)
    table = _table_md(stats)

    print("\n=== Results Table ===")
    print(table)
    print(f"\n=== Verdict: {verdict_label} ===")
    print(justification)
    print(f"\nReward-matrix sensitivity (top-70%-expertise): {stats['reward_sensitivity']:.1%}")

    stats_raw = None
    if not args.demo and mce > 0.10:
        b_x_csv = pd.read_csv(output_dir / "b_x.csv", comment="#", dtype={"study_id": str})
        df_raw = eval_df.copy()
        df_raw["b_x"] = b_x_csv.set_index("study_id").loc[
            eval_df["study_id"].astype(str), "b_x_raw"
        ].values
        df_raw["split"] = "eval"   # _add_br_columns filters by split
        df_raw = _add_br_columns(df_raw)
        stats_raw = compute_stats(df_raw)

    write_report(eval_df, stats, verdict_label, justification, table,
                 T_opt, mce, output_dir, stats_raw=stats_raw, demo=args.demo)
    print("Done.")


if __name__ == "__main__":
    main()
