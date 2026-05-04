#!/usr/bin/env python3
"""
Extract CheXpert pilot arrays from results.csv → data/chexpert_pilot.npz

results.csv columns (from experiment2_pilot.py):
  b_x:   scalar P(Y=1|x) — calibrated, temperature-scaled DenseNet belief
  b_xh:  scalar P(Y=1|x,h) — cross-fitted augmented belief
  h:     bc1 radiologist label (0 or 1)
  y:     majority-vote ground truth (0 or 1)

Output: data/chexpert_pilot.npz with (N, 2) arrays for b_x and b_xh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).parent
    csv_path = root / "results" / "results.csv"
    out_dir = root / "data"
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path, comment="#")
    N = len(df)

    b_x_1d = df["b_x"].to_numpy(dtype=np.float64)
    b_xh_1d = df["b_xh"].to_numpy(dtype=np.float64)
    h = df["h"].to_numpy(dtype=np.int32)
    y = df["y"].to_numpy(dtype=np.int32)

    # Expand scalars to (N, 2): column 0 = P(Y=0), column 1 = P(Y=1)
    b_x = np.column_stack([1.0 - b_x_1d, b_x_1d])
    b_xh = np.column_stack([1.0 - b_xh_1d, b_xh_1d])

    assert b_x.shape == (N, 2)
    assert b_xh.shape == (N, 2)
    assert np.allclose(b_x.sum(axis=1), 1.0)
    assert np.allclose(b_xh.sum(axis=1), 1.0)

    out_path = out_dir / "chexpert_pilot.npz"
    np.savez(out_path, b_x=b_x, b_xh=b_xh, h=h, y=y, dataset_name="chexpert_pilot")

    print(f"Saved {out_path}  (N={N}, K=2)")
    print(f"  Prevalence (y=1):  {y.mean():.1%}")
    print(f"  h=1 rate:          {h.mean():.1%}")
    print(f"  Mean b_x[pos]:     {b_x[:, 1].mean():.3f}")
    print(f"  Mean b_xh[pos]:    {b_xh[:, 1].mean():.3f}")


if __name__ == "__main__":
    main()
