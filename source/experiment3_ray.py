#!/usr/bin/env python3
"""
Experiment 3: Ray-parallelized Review Allocation Driver
========================================================

Parallelizes at the (dataset × reward_matrix) level:
  - 3 reward matrices × 2 datasets = 6 concurrent allocation tasks (CPU-only)
  - 1 CIFAR-10H data prep task (1 GPU, runs before CIFAR-10H allocation tasks)

Usage (after running extract_chexpert_arrays.py locally):
  python experiment3_ray.py \\
      --chexpert-data data/chexpert_pilot.npz \\
      --rewards R1 R2 R3 \\
      --budgets 0.05 0.10 0.20 0.50 \\
      --bootstrap 1000 \\
      --output-dir /tmp/exp3_results

Inside ray job submit the script calls ray.init(address="auto") to connect
to the running cluster. Do not call with --address when submitting via
ray job submit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import ray


# ──────────────────────────────────────────────────────────────
# Vectorized bootstrap (used inside run_allocation_task)
# ──────────────────────────────────────────────────────────────

def _bootstrap_ci_vec(
    b_x: np.ndarray,
    b_xh: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    scores_dict: dict,
    budget: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Bootstrap 95% CIs via vectorized numpy (no Python inner loops)."""
    N = len(y)
    n_review = max(1, int(N * budget))

    # Precompute per-instance utilities
    a_model = (b_x @ R.T).argmax(axis=1)
    a_human = (b_xh @ R.T).argmax(axis=1)
    u_model = R[a_model, y]
    u_human = R[a_human, y]

    rng = np.random.default_rng(seed)
    accum = {name: [] for name in scores_dict}

    for _ in range(n_bootstrap):
        idx = rng.integers(N, size=N)
        u_m = u_model[idx]
        u_h = u_human[idx]
        baseline = u_m.mean()

        for name, scores in scores_dict.items():
            s = scores[idx]
            top_k = np.argpartition(s, -n_review)[-n_review:]
            mask = np.zeros(N, dtype=bool)
            mask[top_k] = True
            total_u = np.where(mask, u_h, u_m).mean()
            accum[name].append(float(total_u - baseline))

    ci = {}
    for name, gains in accum.items():
        g = np.array(gains)
        ci[name] = [float(g.mean()), float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]
    return ci


# ──────────────────────────────────────────────────────────────
# Ray remote task: CIFAR-10H data preparation (1 GPU)
# ──────────────────────────────────────────────────────────────

@ray.remote(num_gpus=1, num_cpus=4)
def prepare_cifar10h_data(output_dir: str) -> str:
    """
    Download CIFAR-10H labels, run ImageNet-pretrained ResNet-18 on CIFAR-10
    test set (with temperature scaling), and save cifar10h.npz.

    Returns the absolute path to the saved .npz file.
    """
    import urllib.request
    from pathlib import Path

    import numpy as np
    import torch
    import torchvision
    import torchvision.transforms as T
    from scipy.optimize import minimize_scalar

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "cifar10h_v2.npz"

    if npz_path.exists():
        print(f"cifar10h_v2.npz already exists at {npz_path}, skipping prep.")
        return str(npz_path)

    # 1. Download CIFAR-10H human label distribution
    probs_path = out_dir / "cifar10h-probs.npy"
    if not probs_path.exists():
        url = "https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy"
        print(f"Downloading CIFAR-10H labels from {url} ...")
        urllib.request.urlretrieve(url, probs_path)
    cifar10h_probs = np.load(probs_path)  # (10000, 10)
    assert cifar10h_probs.shape == (10000, 10), f"Unexpected shape: {cifar10h_probs.shape}"
    print(f"Loaded CIFAR-10H probs: {cifar10h_probs.shape}")

    # 2. CIFAR-10 test set + ground truth labels
    # Use CIFAR-10-native normalization (not ImageNet) — no resize needed
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar10_root = str(out_dir / "cifar10")
    print("Downloading CIFAR-10 test set ...")
    dataset = torchvision.datasets.CIFAR10(root=cifar10_root, train=False, download=True, transform=transform)
    y = np.array(dataset.targets)
    N = len(y)  # 10000

    # 3. CIFAR-10 pretrained ResNet-20 (outputs 10 classes natively)
    # chenyaofo/pytorch-cifar-models: ~91% test accuracy on CIFAR-10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CIFAR-10 ResNet-20 (pretrained) on {device} ...")
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet20",
        pretrained=True, verbose=False, trust_repo=True,
    )
    model.eval().to(device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    all_logits = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device)).cpu().numpy()
            all_logits.append(logits)
    logits = np.concatenate(all_logits, axis=0)  # (10000, 10)
    del model
    torch.cuda.empty_cache()

    # 4. Temperature scaling: optimize T on a 20% held-out calibration split
    rng = np.random.default_rng(0)
    val_idx = rng.choice(N, size=N // 5, replace=False)
    val_mask = np.zeros(N, dtype=bool)
    val_mask[val_idx] = True

    def neg_log_likelihood(T: float) -> float:
        ls = logits[val_mask] / T
        # Numerically stable log-softmax NLL
        ls_shifted = ls - ls.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(ls_shifted).sum(axis=1))
        nll = -(ls_shifted[np.arange(val_mask.sum()), y[val_mask]] - log_sum_exp)
        return float(nll.mean())

    result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 10.0), method="bounded")
    T_opt = float(result.x)
    print(f"Temperature scaling: T_opt = {T_opt:.4f}")

    # Apply and convert to probabilities
    logits_scaled = logits / T_opt
    ls_shifted = logits_scaled - logits_scaled.max(axis=1, keepdims=True)
    exp_ls = np.exp(ls_shifted)
    b_x = exp_ls / exp_ls.sum(axis=1, keepdims=True)  # (10000, 10)

    # 5. Human signal: sample one annotator draw per instance
    h = np.array([rng.choice(10, p=p) for p in cifar10h_probs])

    # 6. Augmented beliefs b_{x,h}: 5-fold cross-fitted multinomial logistic regression
    # on [logit(b_x), one_hot(h)] → P(y|x,h). Mirrors CheXpert's fit_augmented() exactly.
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

    eps = 1e-7
    log_odds_bx = np.log(
        np.clip(b_x, eps, 1 - eps) / np.clip(1 - b_x, eps, 1 - eps)
    )  # (N, 10)
    h_onehot = np.eye(10, dtype=np.float32)[h]  # (N, 10)
    X_aug = np.hstack([log_odds_bx, h_onehot])  # (N, 20)

    b_xh = np.zeros_like(b_x)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_aug, y):
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
        clf.fit(X_aug[train_idx], y[train_idx])
        b_xh[val_idx] = clf.predict_proba(X_aug[val_idx])
    b_xh = np.clip(b_xh, 1e-8, None)
    b_xh /= b_xh.sum(axis=1, keepdims=True)

    np.savez(
        npz_path,
        b_x=b_x.astype(np.float32),
        b_xh=b_xh.astype(np.float32),
        h=h.astype(np.int32),
        y=y.astype(np.int32),
        dataset_name="cifar10h",
        cifar10h_probs=cifar10h_probs.astype(np.float32),
        T_opt=np.array(T_opt),
    )

    model_acc = (b_x.argmax(axis=1) == y).mean()
    human_acc = (h == y).mean()
    bxh_acc   = (b_xh.argmax(axis=1) == y).mean()
    print(f"Saved {npz_path}  (N={N}, K=10)")
    print(f"  ResNet-20 accuracy (b_x):                {model_acc:.1%}")
    print(f"  Single-draw human accuracy (h):          {human_acc:.1%}")
    print(f"  Cross-fitted b_xh accuracy:              {bxh_acc:.1%}  (should exceed b_x)")
    return str(npz_path)


# ──────────────────────────────────────────────────────────────
# Ray remote task: allocation experiment (CPU-only)
# ──────────────────────────────────────────────────────────────

@ray.remote(num_cpus=2)
def run_allocation_task(
    npz_path: str,
    reward_name: str,
    budgets: list[float],
    bootstrap_n: int,
    seed: int,
) -> dict:
    """
    Run full allocation experiment for one (dataset, reward_matrix) pair.

    Returns a JSON-serializable dict with allocation results and bootstrap CIs.
    """
    # Limit BLAS thread count to match the 2 CPUs claimed from Ray
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")

    # Ensure local imports resolve (working dir is source/ on cluster)
    src_dir = str(Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from experiment3_allocation import (  # noqa: PLC0415
        load_data,
        make_binary_rewards,
        make_multiclass_rewards,
        run_allocation,
        score_entropy,
        score_margin,
    )

    data = load_data(npz_path)
    b_x = data["b_x"]
    b_xh = data["b_xh"]
    h = data["h"]
    y = data["y"]
    ds_name = data["dataset_name"]
    K = b_x.shape[1]

    if K == 2:
        R = make_binary_rewards(reward_name)
    else:
        R = make_multiclass_rewards(K, reward_name)

    c_review = 0.1 if reward_name == "R4" else 0.0

    print(f"[{ds_name}/{reward_name}] Starting allocation (N={len(y)}, K={K}) ...")
    results = run_allocation(
        b_x, b_xh, h, y, R,
        reward_name=reward_name,
        dataset_name=ds_name,
        budgets=budgets,
        c_review=c_review,
        seed=seed,
    )
    print(f"[{ds_name}/{reward_name}] Allocation done.")

    results_dicts = [
        {
            "dataset": r.dataset,
            "reward_name": r.reward_name,
            "budget": r.budget,
            "utility": r.utility,
            "utility_gain": r.utility_gain,
            "gain_per_review": r.gain_per_review,
            "oracle_overlap": r.oracle_overlap,
        }
        for r in results
    ]

    # Vectorized bootstrap CIs for non-parametric baselines
    ci_by_budget: dict[str, dict] = {}
    if bootstrap_n > 0:
        entropy_scores = score_entropy(b_x)
        margin_scores = score_margin(b_x, R)
        rng = np.random.default_rng(seed)
        scores_dict = {
            "Entropy": entropy_scores,
            "Margin": margin_scores,
            "Random": rng.random(len(y)),
        }
        for budget in budgets:
            ci_by_budget[str(budget)] = _bootstrap_ci_vec(
                b_x, b_xh, y, R, scores_dict, budget, bootstrap_n, seed
            )
        print(f"[{ds_name}/{reward_name}] Bootstrap ({bootstrap_n} samples) done.")

    return {
        "dataset": ds_name,
        "reward": reward_name,
        "results": results_dicts,
        "bootstrap_ci": ci_by_budget,
    }


# ──────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: Ray-parallelized allocation")
    parser.add_argument("--chexpert-data", default="data/chexpert_pilot.npz",
                        help="Path to chexpert_pilot.npz (relative to working dir)")
    parser.add_argument("--rewards", nargs="+", default=["R1", "R2", "R3"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.50])
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Bootstrap samples for CIs (0 = skip; use 1000 for real data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/tmp/exp3_results",
                        help="Directory for output JSON files and CIFAR-10H cache")
    parser.add_argument("--skip-cifar10h", action="store_true",
                        help="Only run CheXpert (skip CIFAR-10H entirely)")
    parser.add_argument("--demo", action="store_true",
                        help="Run synthetic demo instead of real data (for pipeline testing)")
    args = parser.parse_args()

    ray.init(address="auto")
    print(f"Ray cluster resources: {ray.cluster_resources()}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Demo mode: skip real data, test Ray wiring only ──────────
    if args.demo:
        print("\n=== DEMO MODE: synthetic data ===\n")
        src_dir = str(Path(__file__).parent)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from experiment3_allocation import generate_synthetic_demo, save_results_json, make_binary_rewards, run_allocation
        data = generate_synthetic_demo(N=500, seed=args.seed)
        b_x, b_xh, h, y = data["b_x"], data["b_xh"], data["h"], data["y"]
        demo_npz = out_dir / "demo.npz"
        np.savez(demo_npz, b_x=b_x, b_xh=b_xh, h=h, y=y, dataset_name="demo")
        futures = [
            run_allocation_task.remote(str(demo_npz), rn, args.budgets, 0, args.seed)
            for rn in args.rewards
        ]
        all_results_raw = ray.get(futures)
        _save_and_print(all_results_raw, "demo", out_dir)
        ray.shutdown()
        return

    # ── Resolve dataset paths ─────────────────────────────────────
    chexpert_path = Path(args.chexpert_data)
    if not chexpert_path.exists():
        raise FileNotFoundError(
            f"CheXpert data not found at {chexpert_path}. "
            "Run: python extract_chexpert_arrays.py first."
        )
    print(f"CheXpert data: {chexpert_path}")

    # ── CIFAR-10H data prep (GPU task, runs first) ────────────────
    cifar10h_path = out_dir / "cifar10h_v2.npz"
    cifar10h_future = None
    if not args.skip_cifar10h:
        cifar10h_future = prepare_cifar10h_data.remote(str(out_dir))
        print("Submitted CIFAR-10H prep task (1 GPU). Starting CheXpert tasks immediately.")

    # ── CheXpert allocation tasks (6 tasks: 3 rewards × 2 datasets) ──
    chexpert_futures = {
        rn: run_allocation_task.remote(
            str(chexpert_path), rn, args.budgets, args.bootstrap, args.seed
        )
        for rn in args.rewards
    }
    print(f"Submitted {len(chexpert_futures)} CheXpert allocation tasks.")

    # ── Collect CheXpert results (don't block on CIFAR-10H prep) ─────
    print("Waiting for CheXpert allocation tasks ...")
    chexpert_results = ray.get(list(chexpert_futures.values()))
    _save_and_print(chexpert_results, "chexpert_pilot", out_dir)

    # ── Wait for CIFAR-10H prep, then submit + collect allocation tasks ──
    cifar10h_futures: dict[str, ray.ObjectRef] = {}
    if cifar10h_future is not None:
        print("Waiting for CIFAR-10H prep task ...")
        cifar10h_npz_path = ray.get(cifar10h_future)
        print(f"CIFAR-10H prep complete: {cifar10h_npz_path}")
        cifar10h_futures = {
            rn: run_allocation_task.remote(
                cifar10h_npz_path, rn, args.budgets, args.bootstrap, args.seed
            )
            for rn in args.rewards
        }
        print(f"Submitted {len(cifar10h_futures)} CIFAR-10H allocation tasks.")

    if cifar10h_futures:
        print("Waiting for CIFAR-10H allocation tasks ...")
        cifar10h_results = ray.get(list(cifar10h_futures.values()))
        _save_and_print(cifar10h_results, "cifar10h", out_dir)

    print(f"\nAll done. Results in {out_dir}")
    ray.shutdown()


def _save_and_print(raw_results: list[dict], dataset_tag: str, out_dir: Path) -> None:
    """Merge task results → save JSON + print summary table."""
    all_result_dicts = []
    all_bootstrap = {}

    for task_out in raw_results:
        all_result_dicts.extend(task_out["results"])
        rn = task_out["reward"]
        if task_out["bootstrap_ci"]:
            all_bootstrap[rn] = task_out["bootstrap_ci"]

    out_path = out_dir / f"experiment3_{dataset_tag}.json"
    with open(out_path, "w") as f:
        json.dump({"allocation": all_result_dicts, "bootstrap_ci": all_bootstrap}, f, indent=2)
    print(f"Saved {out_path}")

    # Print summary: utility gain per policy per (reward, budget)
    print(f"\n── {dataset_tag.upper()} — Utility gain over model-only ──")
    policies = list(all_result_dicts[0]["utility_gain"].keys()) if all_result_dicts else []
    for r in all_result_dicts:
        rn, bgt = r["reward_name"], r["budget"]
        gains = "  ".join(f"{p}:{r['utility_gain'][p]:>+.4f}" for p in policies)
        print(f"  {rn} q={bgt:.0%}:  {gains}")


if __name__ == "__main__":
    main()
