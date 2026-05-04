"""
Theory figure for Section 3 of the ASIL/boundary-regret NeurIPS paper.

Shows the geometry of BR(x, h) = V(b_{x,h}) - r_{a_x} . b_{x,h} for a binary
decision problem (K=2 states, 2 actions) with an asymmetric reward matrix.
The key visual contrast: when the human-updated posterior stays in the same
decision region as the model belief, BR = 0; when it crosses the decision
facet, BR > 0 (visible as a gap between V and the model-action line).

Regenerate: python figures/theory_figure.py  (from the repo root)
Outputs: figures/theory_figure.pdf  (vector, for LaTeX)
         figures/theory_figure.png  (200 DPI, for quick preview)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os

# ---------------------------------------------------------------------------
# Constants — reward geometry and belief points
# ---------------------------------------------------------------------------

# Asymmetric reward matrix: false positive (discharging a sick patient) is
# cheap; false negative (admitting a healthy patient) is mildly penalised.
# This shifts the decision threshold away from 0.5, to around b* ≈ 0.6,
# making the BR=0 vs BR>0 contrast more vivid and closer to real deployment.
#
# Action a0 ("discharge"): reward 1.0 if healthy (state 0), 0.0 if sick (state 1)
# Action a1 ("admit"):     reward 0.2 if healthy (state 0), 1.0 if sick (state 1)
#
# a0 expected reward: 1.0*(1-b) + 0.0*b  =  1.0 - b
# a1 expected reward: 0.2*(1-b) + 1.0*b  =  0.2 + 0.8*b
# Cross-over: 1.0 - b = 0.2 + 0.8*b  →  0.8 = 1.8*b  →  b* = 4/9 ≈ 0.444

# Wait — let's pick an even more asymmetric example where the facet is ~0.62,
# giving clearer visual separation:
#   a0: rewards (1.0, 0.0)  ← cheap to discharge healthy, costly if sick
#   a1: rewards (0.0, 0.62) ← admit is sub-optimal unless P(sick) is high
#
# Actually, let's use a clinically-motivated 5:1 miss-detection penalty:
#   a0 ("discharge"): (1, 0)
#   a1 ("admit"):     (0, 5)
# Cross-over: 1*(1-b) = 5*b  →  b* = 1/6 ≈ 0.167 — too far left, hard to see.
#
# Best visual: keep it asymmetric but in a readable region.
# Use:
#   a0 ("discharge"): (1.0, 0.0)
#   a1 ("admit"):     (0.0, 1.5)
# Cross-over: 1-b = 1.5*b  →  1 = 2.5*b  →  b* = 0.4
#
# Model belief b_x = 0.25 (D_{a0}: discharge region)
# Posterior h0: b_{x,h0} = 0.30  (still in D_{a0}, BR=0)
# Posterior h1: b_{x,h1} = 0.65  (in D_{a1}, BR>0)

R_A0 = np.array([1.0, 0.0])   # discharge rewards: (r[state=0], r[state=1])
R_A1 = np.array([0.0, 1.5])   # admit rewards

# Decision facet: r_a0 . (1-b, b) = r_a1 . (1-b, b)
# 1.0*(1-b) = 1.5*b  →  1 = 2.5*b  →  b* = 0.4
B_STAR = R_A0[0] / (R_A0[0] - R_A0[1] + R_A1[1] - R_A1[0])
# = 1.0 / (1.0 - 0.0 + 1.5 - 0.0) = 1.0 / 2.5 = 0.4

# Belief points
B_X    = 0.25   # model belief — in D_{a0} (discharge region)
B_XH0  = 0.32   # posterior after h0 — still in D_{a0}, so BR=0
B_XH1  = 0.68   # posterior after h1 — in D_{a1}, so BR>0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def er_a0(b: float | np.ndarray) -> float | np.ndarray:
    """Expected reward of a0 (discharge) at scalar or array belief b.

    Belief vector is (1-b, b); state 0 = healthy, state 1 = sick.
    """
    return R_A0[0] * (1.0 - b) + R_A0[1] * b


def er_a1(b: float | np.ndarray) -> float | np.ndarray:
    """Expected reward of a1 (admit) at scalar or array belief b."""
    return R_A1[0] * (1.0 - b) + R_A1[1] * b


def V(b: float | np.ndarray) -> float | np.ndarray:
    """V(b) = max_a  r_a . (1-b, b) — the upper envelope of the two action lines.

    We compute this directly (not via information_model.terminal_value) because
    that function uses a fixed 3-state problem; for a binary problem it is
    cleaner to inline the computation here.
    """
    return np.maximum(er_a0(b), er_a1(b))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_figure() -> None:
    # -- Font & style setup ------------------------------------------------
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    plt.style.use("default")
    # Re-apply after style reset
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(5, 4))

    b = np.linspace(0.0, 1.0, 500)

    # -- Plot action lines -------------------------------------------------
    # a1 line (gray, context only)
    ax.plot(b, er_a1(b), color="#AAAAAA", lw=1.2, ls="--",
            label=r"$r_{a_1} \cdot b$ (admit)", zorder=2)

    # a0 line extended across all b (red — this is the model-action line)
    ax.plot(b, er_a0(b), color="#CC0000", lw=1.4, ls="-",
            label=r"$r_{a_0} \cdot b$ (discharge, model action)", zorder=3)

    # V(b) upper envelope (thick black)
    ax.plot(b, V(b), color="black", lw=2.5, ls="-",
            label=r"$V(b)$ (value function)", zorder=4)

    # -- Decision facet ----------------------------------------------------
    ax.axvline(B_STAR, color="#888888", lw=0.9, ls=":", zorder=1)
    ax.text(B_STAR + 0.012, 0.06,
            r"$b^*\!\approx\!0.40$" + "\n" + r"(facet $\mathcal{D}_{a_0}|\mathcal{D}_{a_1}$)",
            fontsize=8, color="#555555", va="bottom", ha="left", linespacing=1.4)

    # -- b_x marker --------------------------------------------------------
    y_bx = er_a0(B_X)  # = V(B_X) since B_X is in D_{a0}
    ax.axvline(B_X, color="#4477AA", lw=1.0, ls="--", zorder=2, alpha=0.8)
    ax.scatter([B_X], [y_bx], color="#4477AA", s=40, zorder=5)
    ax.text(B_X - 0.018, y_bx + 0.06, r"$b_x$", fontsize=10,
            color="#4477AA", ha="right", va="bottom")

    # -- b_{x,h0}: BR = 0 --------------------------------------------------
    v_bxh0 = V(B_XH0)
    er_a0_bxh0 = er_a0(B_XH0)
    # At B_XH0, both V and er_a0 coincide (same decision region)
    ax.axvline(B_XH0, color="#44AA77", lw=1.0, ls="--", zorder=2, alpha=0.8)
    ax.scatter([B_XH0], [v_bxh0], color="#44AA77", s=40, zorder=5)
    # Place label to the right to avoid overlap with b_x label
    ax.text(B_XH0 + 0.013, v_bxh0 + 0.04, r"$b_{x,h_0}$", fontsize=10,
            color="#44AA77", ha="left", va="bottom")
    # BR=0 label: small annotation pointing to the coincidence point
    ax.annotate(r"$\mathrm{BR}=0$",
                xy=(B_XH0, v_bxh0), xytext=(B_XH0 - 0.17, v_bxh0 + 0.16),
                fontsize=9, color="#44AA77",
                arrowprops=dict(arrowstyle="-|>", color="#44AA77",
                                lw=0.9, shrinkB=4),
                ha="center", va="bottom")

    # -- b_{x,h1}: BR > 0 --------------------------------------------------
    v_bxh1   = V(B_XH1)
    er_a0_bxh1 = er_a0(B_XH1)
    br_h1    = v_bxh1 - er_a0_bxh1   # positive

    ax.axvline(B_XH1, color="#EE7733", lw=1.0, ls="--", zorder=2, alpha=0.8)
    ax.scatter([B_XH1], [v_bxh1],    color="#EE7733", s=40, zorder=5)
    ax.scatter([B_XH1], [er_a0_bxh1], color="#CC0000", s=40, zorder=5)

    # Vertical bracket for BR — draw as a line segment with ticks at each end
    brace_x = B_XH1 + 0.018
    tick_w  = 0.015   # half-width of horizontal tick marks
    ax.plot([brace_x, brace_x], [er_a0_bxh1, v_bxh1],
            color="#EE7733", lw=1.8, zorder=6)
    ax.plot([brace_x - tick_w, brace_x + tick_w], [er_a0_bxh1, er_a0_bxh1],
            color="#EE7733", lw=1.5, zorder=6)
    ax.plot([brace_x - tick_w, brace_x + tick_w], [v_bxh1, v_bxh1],
            color="#EE7733", lw=1.5, zorder=6)
    ax.text(brace_x + 0.025, (v_bxh1 + er_a0_bxh1) / 2,
            r"$\mathrm{BR}(x,h_1)>0$", fontsize=9, color="#EE7733",
            va="center", ha="left")

    ax.text(B_XH1 - 0.013, v_bxh1 + 0.05, r"$b_{x,h_1}$", fontsize=10,
            color="#EE7733", ha="right", va="bottom")

    # -- Shaded decision regions (subtle) ----------------------------------
    ax.axvspan(0.0, B_STAR, alpha=0.04, color="#4477AA", zorder=0)
    ax.axvspan(B_STAR, 1.0, alpha=0.04, color="#EE7733", zorder=0)
    ax.text(B_STAR / 2, 0.96, r"$\mathcal{D}_{a_0}$", fontsize=9,
            color="#4477AA", alpha=0.7, ha="center", va="top",
            transform=ax.get_xaxis_transform())
    ax.text((B_STAR + 1.0) / 2, 0.96, r"$\mathcal{D}_{a_1}$", fontsize=9,
            color="#EE7733", alpha=0.7, ha="center", va="top",
            transform=ax.get_xaxis_transform())

    # -- Axes & labels -----------------------------------------------------
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.15)
    ax.set_xlabel(r"$b$ (probability of positive class)", fontsize=11)
    ax.set_ylabel(r"Expected reward", fontsize=11)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color="black",   lw=2.5, ls="-",
               label=r"$V(b)$ — value function"),
        Line2D([0], [0], color="#CC0000", lw=1.4, ls="-",
               label=r"$r_{a_0}\!\cdot\! b$ — model-action line"),
        Line2D([0], [0], color="#AAAAAA", lw=1.2, ls="--",
               label=r"$r_{a_1}\!\cdot\! b$ — rival action"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              framealpha=0.9, edgecolor="#CCCCCC")

    fig.tight_layout()

    # -- Save --------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
    pdf_path = os.path.join(out_dir, "theory_figure.pdf")
    png_path = os.path.join(out_dir, "theory_figure.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"Saved PDF : {pdf_path}")
    print(f"Saved PNG : {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Sanity-check the reward geometry before plotting
    assert abs(B_STAR - 0.4) < 1e-9, f"Unexpected facet location: {B_STAR}"
    assert B_X    < B_STAR, "b_x must be in D_{a0}"
    assert B_XH0  < B_STAR, "b_{x,h0} must be in D_{a0} for BR=0"
    assert B_XH1  > B_STAR, "b_{x,h1} must be in D_{a1} for BR>0"
    br0 = V(B_XH0) - er_a0(B_XH0)
    br1 = V(B_XH1) - er_a0(B_XH1)
    assert abs(br0) < 1e-9, f"BR(x,h0) should be 0, got {br0}"
    assert br1 > 0,          f"BR(x,h1) should be >0, got {br1}"
    print(f"Reward geometry check:")
    print(f"  Facet b* = {B_STAR:.4f}")
    print(f"  b_x = {B_X}, b_{{x,h0}} = {B_XH0}, b_{{x,h1}} = {B_XH1}")
    print(f"  BR(x,h0) = {br0:.6f} (should be 0)")
    print(f"  BR(x,h1) = {br1:.6f} (should be > 0)")
    make_figure()
