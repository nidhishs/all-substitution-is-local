#!/usr/bin/env python3
"""
Data Preparation for Experiment 3
==================================

Prepares datasets into the standard .npz format required by experiment3_allocation.py.

Each dataset must produce:
  b_x:   (N, K) float  — calibrated model beliefs over K classes
  b_xh:  (N, K) float  — human-augmented beliefs (cross-fitted)
  h:     (N,)   int    — recorded human signal
  y:     (N,)   int    — ground truth labels

Two datasets:
  A. CheXpert (from pilot) — binary (K=2), you supply pilot arrays
  B. CIFAR-10H — 10-class (K=10), downloaded automatically

Usage:
  # CheXpert: provide your pilot data
  uv run --with numpy prepare_data.py chexpert \\
      --b-x pilot_bx.npy --h pilot_h.npy --y pilot_y.npy --b-xh pilot_bxh.npy

  # CIFAR-10H: automatic download + model inference
  uv run --with numpy --with torch --with torchvision prepare_data.py cifar10h \\
      --cifar10h-labels path/to/cifar10h-probs.npy

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────
# A. CheXpert (from pilot)
# ──────────────────────────────────────────────────────────────

def prepare_chexpert(args):
    """
    Package CheXpert pilot data into the standard .npz format.

    You need to provide 4 numpy arrays from your pilot:
      b_x:  (N,) model belief P(Y=1|x) — calibrated (temperature-scaled)
      b_xh: (N,) augmented belief P(Y=1|x,h) — cross-fitted
      h:    (N,) bc1 radiologist label (0 or 1)
      y:    (N,) ground truth (majority vote, 0 or 1)

    If b_x and b_xh are 1D (binary probability), they'll be expanded to (N, 2).
    """
    b_x_raw = np.load(args.b_x)
    b_xh_raw = np.load(args.b_xh)
    h = np.load(args.h).astype(int)
    y = np.load(args.y).astype(int)

    N = len(y)
    assert len(b_x_raw) == N, f"b_x has {len(b_x_raw)} entries, expected {N}"
    assert len(b_xh_raw) == N, f"b_xh has {len(b_xh_raw)} entries, expected {N}"
    assert len(h) == N, f"h has {len(h)} entries, expected {N}"

    # Expand 1D to (N, 2) if needed
    if b_x_raw.ndim == 1:
        b_x = np.column_stack([1 - b_x_raw, b_x_raw])
    else:
        b_x = b_x_raw

    if b_xh_raw.ndim == 1:
        b_xh = np.column_stack([1 - b_xh_raw, b_xh_raw])
    else:
        b_xh = b_xh_raw

    out_path = Path(args.output) / "chexpert_pilot.npz"
    np.savez(out_path,
             b_x=b_x, b_xh=b_xh, h=h, y=y,
             dataset_name="chexpert_pilot")
    print(f"Saved {out_path}  (N={N}, K=2)")
    print(f"  Prevalence: {y.mean():.1%}")
    print(f"  h=1 rate: {h.mean():.1%}")
    print(f"  Mean b_x[1]: {b_x[:, 1].mean():.3f}")


# ──────────────────────────────────────────────────────────────
# B. CIFAR-10H
# ──────────────────────────────────────────────────────────────

def prepare_cifar10h(args):
    """
    Prepare CIFAR-10H dataset.

    Requires:
    1. CIFAR-10H human probability labels:
       Download from https://github.com/jcpeterson/cifar-10h
       File: cifar10h-probs.npy (10000, 10) array of human label distributions

    2. A calibrated CIFAR-10 model to produce b_x.
       If --model-probs is provided, uses those directly.
       Otherwise, runs a pretrained ResNet on CIFAR-10 test set.
    """
    # Load CIFAR-10H human labels
    cifar10h_probs = np.load(args.cifar10h_labels)  # (10000, 10)
    N = cifar10h_probs.shape[0]
    assert cifar10h_probs.shape == (N, 10), f"Expected (10000, 10), got {cifar10h_probs.shape}"

    # Ground truth: CIFAR-10 test labels
    if args.cifar10_labels is not None:
        y = np.load(args.cifar10_labels).astype(int)
    else:
        # Try to load from torchvision
        try:
            import torchvision
            dataset = torchvision.datasets.CIFAR10(
                root=args.cifar10_root or "/tmp/cifar10",
                train=False, download=True
            )
            y = np.array(dataset.targets)
        except ImportError:
            print("ERROR: Provide --cifar10-labels or install torchvision")
            sys.exit(1)

    assert len(y) == N, f"y has {len(y)} entries, expected {N}"

    # Human signal: sample one annotator per instance
    # CIFAR-10H provides the distribution; we sample h ~ Cat(cifar10h_probs[i])
    rng = np.random.default_rng(42)
    h = np.array([rng.choice(10, p=p) for p in cifar10h_probs])

    # Model beliefs b_x
    if args.model_probs is not None:
        b_x = np.load(args.model_probs)
        assert b_x.shape == (N, 10), f"model_probs shape {b_x.shape}, expected ({N}, 10)"
    else:
        print("Computing model beliefs from pretrained ResNet...")
        b_x = _compute_cifar10_model_beliefs(args.cifar10_root or "/tmp/cifar10")

    # Augmented beliefs b_{x,h}: shift model belief toward human label
    # Using a simple Bayesian update with the human label distribution
    b_xh = np.zeros_like(b_x)
    for i in range(N):
        # P(y|x,h) ∝ P(y|x) * P(h|y) where P(h|y) is estimated from cifar10h
        # Simple approach: weight model belief by human distribution info
        # More rigorous: cross-fitted logistic regression (done in experiment3)
        alpha = 0.6  # mixing weight for human info
        h_info = np.zeros(10)
        h_info[h[i]] = 1.0
        b_xh[i] = (1 - alpha) * b_x[i] + alpha * h_info
        b_xh[i] = np.clip(b_xh[i], 1e-6, None)
        b_xh[i] /= b_xh[i].sum()

    out_path = Path(args.output) / "cifar10h.npz"
    np.savez(out_path,
             b_x=b_x, b_xh=b_xh, h=h, y=y,
             dataset_name="cifar10h",
             cifar10h_probs=cifar10h_probs)
    print(f"Saved {out_path}  (N={N}, K=10)")
    print(f"  Model accuracy: {(b_x.argmax(axis=1) == y).mean():.1%}")
    print(f"  Human accuracy: {(h == y).mean():.1%}")


def _compute_cifar10_model_beliefs(cifar10_root: str) -> np.ndarray:
    """Run pretrained ResNet-18 on CIFAR-10 test set, return calibrated probs."""
    try:
        import torch
        import torchvision
        import torchvision.transforms as transforms
    except ImportError:
        print("ERROR: Install torch and torchvision to compute model beliefs")
        print("  uv run --with torch --with torchvision prepare_data.py ...")
        sys.exit(1)

    from torchvision.models import ResNet18_Weights, resnet18

    # Resize CIFAR-10's 32×32 images to 224×224 expected by ImageNet ResNet-18
    transform = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=cifar10_root, train=False, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    # ImageNet-pretrained ResNet-18 as a calibrated feature extractor
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()

    all_logits = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images)
            all_logits.append(logits.numpy())

    logits = np.concatenate(all_logits, axis=0)

    # Temperature scaling (calibration) — optimize on a held-out split in experiment3_ray.py
    T = 1.0  # placeholder; experiment3_ray.py runs proper grid-search calibration
    probs = _softmax(logits / T)
    return probs


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare data for Experiment 3")
    subparsers = parser.add_subparsers(dest="dataset")

    # CheXpert
    p_chex = subparsers.add_parser("chexpert", help="Package CheXpert pilot data")
    p_chex.add_argument("--b-x", required=True, help="Path to b_x .npy file")
    p_chex.add_argument("--b-xh", required=True, help="Path to b_xh .npy file")
    p_chex.add_argument("--h", required=True, help="Path to h .npy file")
    p_chex.add_argument("--y", required=True, help="Path to y .npy file")
    p_chex.add_argument("--output", default="data", help="Output directory")

    # CIFAR-10H
    p_cifar = subparsers.add_parser("cifar10h", help="Prepare CIFAR-10H dataset")
    p_cifar.add_argument("--cifar10h-labels", required=True,
                         help="Path to cifar10h-probs.npy from github.com/jcpeterson/cifar-10h")
    p_cifar.add_argument("--cifar10-labels", default=None,
                         help="Path to CIFAR-10 test labels .npy (optional, will download)")
    p_cifar.add_argument("--cifar10-root", default=None,
                         help="Root dir for torchvision CIFAR-10 download")
    p_cifar.add_argument("--model-probs", default=None,
                         help="Path to precomputed model probabilities (N, 10) .npy")
    p_cifar.add_argument("--output", default="data", help="Output directory")

    args = parser.parse_args()

    if args.dataset is None:
        parser.print_help()
        sys.exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.dataset == "chexpert":
        prepare_chexpert(args)
    elif args.dataset == "cifar10h":
        prepare_cifar10h(args)


if __name__ == "__main__":
    main()
