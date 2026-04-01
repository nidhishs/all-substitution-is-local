#!/usr/bin/env python3
"""Generate illustration figures for the interior complementarity paper.

Usage:
    uv run --with numpy --with matplotlib generate_figures.py

Outputs two PDFs into the same directory as this script:
    interaction_decomposition.pdf  -- Three-panel figure (delta-VoI + complement/substitute forces)
    force_transition_ray.pdf       -- Phase transition along a belief ray
"""

import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from information_model import (
    bregman_forces,
    delta_voi,
    verify_known_values,
)
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR

# ---------------------------------------------------------------------------
# Simplex geometry
# ---------------------------------------------------------------------------

VERTEX_S1 = np.array([0.0, 0.0])
VERTEX_S2 = np.array([1.0, 0.0])
VERTEX_S3 = np.array([0.5, math.sqrt(3) / 2])

SIMPLEX_VERTICES = [VERTEX_S1, VERTEX_S2, VERTEX_S3]
SIMPLEX_PATH = Path(
    [VERTEX_S1, VERTEX_S2, VERTEX_S3, VERTEX_S1],
    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
)

# Decision boundaries: each runs from the triple point (1/3,1/3,1/3) to a simplex edge.
TRIPLE_POINT = np.array([1 / 3, 1 / 3, 1 / 3])
BOUNDARY_ENDPOINTS = {
    "$a_1/a_2$": np.array([1 / 2, 1 / 2, 0.0]),
    "$a_1/a_3$": np.array([2 / 5, 0.0, 3 / 5]),
    "$a_2/a_3$": np.array([0.0, 2 / 5, 3 / 5]),
}

EXTEND_T = 1.15  # fraction past simplex edge for boundary label placement

# Decision region polygons (vertices in Cartesian space).
_TP_XY = (0.5 * (1 / 3) + 0.5 * (1 / 3), (math.sqrt(3) / 2) * (1 / 3))
_EP12_XY = (0.5, 0.0)
_EP13_XY = (0.0 * 1 + 0.5 * (3 / 5), (math.sqrt(3) / 2) * (3 / 5))
_EP23_XY = ((2 / 5) + 0.5 * (3 / 5), (math.sqrt(3) / 2) * (3 / 5))

REGION_POLYS = {
    0: [(0.0, 0.0), _EP12_XY, _TP_XY, _EP13_XY],  # R1
    1: [_EP12_XY, (1.0, 0.0), _EP23_XY, _TP_XY],  # R2
    2: [_EP13_XY, _TP_XY, _EP23_XY, (0.5, math.sqrt(3) / 2)],  # R3
}
REGION_COLORS = {0: "#e0e0e0", 1: "#ededed", 2: "#f8f8f8"}

# Two sets of region-label positions (different visual weight per figure type).
REGION_LABEL_POS_DEFAULT = [
    ("$\\mathcal{R}_1$", (0.72, 0.08, 0.20)),
    ("$\\mathcal{R}_2$", (0.08, 0.72, 0.20)),
    ("$\\mathcal{R}_3$", (0.07, 0.07, 0.86)),
]
# Marked beliefs from the paper.
BELIEF_B1 = np.array([1 / 11, 2 / 11, 8 / 11])  # Theorem 2 (interior)
BELIEF_B2 = np.array([1 / 4, 1 / 6, 7 / 12])  # Decomposition (near boundary)
BELIEF_B3 = np.array([5 / 12, 5 / 12, 1 / 6])  # Substitution (on boundary)


def bary_to_xy(b1, b2, b3):
    """Barycentric coordinates -> Cartesian (x, y) on the equilateral triangle."""
    x = b2 + 0.5 * b3
    y = (math.sqrt(3) / 2) * b3
    return x, y


def xy_to_bary(x, y):
    """Cartesian (x, y) -> barycentric coordinates (b1, b2, b3)."""
    b3 = y / (math.sqrt(3) / 2)
    b2 = x - 0.5 * b3
    b1 = 1.0 - b2 - b3
    return b1, b2, b3


def clamp_to_interior(b1, b2, b3, margin=0.005):
    """Push barycentric coords into the simplex interior and renormalize."""
    coords = np.array([max(b1, margin), max(b2, margin), max(b3, margin)])
    return coords / coords.sum()


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------


def evaluate_on_grid(func, resolution=300, margin=0.005):
    """Evaluate a scalar function on a Cartesian meshgrid covering the simplex.

    Points outside the simplex are clamped to the nearest interior point so
    that contourf gets continuous data everywhere (no NaN holes). The caller
    clips the rendered output to the triangle polygon.

    Returns (grid_x, grid_y, values) -- 2D arrays for contourf.
    """
    pad = 0.08  # extend well past simplex edges so clipping has no gaps
    xs = np.linspace(-pad, 1.0 + pad, resolution)
    ys = np.linspace(-pad, VERTEX_S3[1] + pad, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    values = np.empty(grid_x.shape)

    for i in range(resolution):
        for j in range(resolution):
            b1, b2, b3 = xy_to_bary(grid_x[i, j], grid_y[i, j])
            belief = clamp_to_interior(b1, b2, b3, margin)
            values[i, j] = func(belief)

    return grid_x, grid_y, values


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------


def add_simplex_clip(ax):
    """Add an invisible simplex polygon to the axes and return it for clipping."""
    clip_poly = Polygon(
        SIMPLEX_VERTICES,
        closed=True,
        fill=False,
        edgecolor="none",
        linewidth=0,
        transform=ax.transData,
    )
    ax.add_patch(clip_poly)
    return clip_poly


def draw_simplex_frame(ax):
    """Draw the triangle border and vertex labels."""
    border = Polygon(
        SIMPLEX_VERTICES, fill=False, edgecolor="black", linewidth=1.8, zorder=4
    )
    ax.add_patch(border)

    for label, pos, offset in [
        ("$s_1$", VERTEX_S1, (-10, -10)),
        ("$s_2$", VERTEX_S2, (10, -10)),
        ("$s_3$", VERTEX_S3, (0, 8)),
    ]:
        ax.annotate(
            label,
            pos,
            textcoords="offset points",
            xytext=offset,
            fontsize=14,
            fontweight="bold",
            ha="center",
        )


def draw_decision_boundaries(ax, linewidth=1.2, alpha=0.5, extend=0.0):
    """Draw the three dashed decision boundary lines.

    If extend > 0, lines continue past the simplex edge by that fraction
    (e.g. extend=0.15 adds 15% beyond the endpoint).
    """
    t = np.linspace(0, 1 + extend, 200)
    for _name, endpoint in BOUNDARY_ENDPOINTS.items():
        path_bary = np.outer(1 - t, TRIPLE_POINT) + np.outer(t, endpoint)
        x, y = bary_to_xy(path_bary[:, 0], path_bary[:, 1], path_bary[:, 2])
        ax.plot(x, y, color="black", linestyle="--", linewidth=linewidth, alpha=alpha)


def _simplex_interior_mask(grid_x, grid_y):
    """Boolean mask: True for grid points inside the simplex (all bary coords > 0)."""
    b3 = grid_y / (math.sqrt(3) / 2)
    b2 = grid_x - 0.5 * b3
    b1 = 1.0 - b2 - b3
    return (b1 > 0) & (b2 > 0) & (b3 > 0)


def setup_simplex_axes(ax, pad=0.06):
    """Configure axes for a simplex plot: equal aspect, no frame."""
    ax.set_xlim(-pad, 1.0 + pad)
    ax.set_ylim(-pad, VERTEX_S3[1] + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_boundary_labels(ax, fontsize=9):
    """Annotate decision-boundary names at extended tips past the simplex edge."""
    for label, endpoint, ha, va, offset in [
        ("$a_1/a_2$", BOUNDARY_ENDPOINTS["$a_1/a_2$"], "center", "top", (0, -4)),
        ("$a_1/a_3$", BOUNDARY_ENDPOINTS["$a_1/a_3$"], "right", "center", (-6, 0)),
        ("$a_2/a_3$", BOUNDARY_ENDPOINTS["$a_2/a_3$"], "left", "center", (6, 0)),
    ]:
        tip = (1 - EXTEND_T) * TRIPLE_POINT + EXTEND_T * endpoint
        x, y = bary_to_xy(*tip)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=fontsize,
            color="0.3",
            ha=ha,
            va=va,
        )


def draw_region_labels(ax, positions=None):
    """Draw R1/R2/R3 region labels at the given barycentric positions."""
    if positions is None:
        positions = REGION_LABEL_POS_DEFAULT
    for label, bary in positions:
        rx, ry = bary_to_xy(*bary)
        ax.text(
            rx, ry, label,
            ha="center", va="center", fontsize=13, color="0.35", fontweight="bold",
        )


def draw_simplex_panel(ax, region_labels=None):
    """Full simplex panel: shaded regions, boundaries, labels, and frame."""
    for action_idx, verts in REGION_POLYS.items():
        poly = Polygon(
            verts, closed=True,
            facecolor=REGION_COLORS[action_idx], edgecolor="none", zorder=0,
        )
        ax.add_patch(poly)
    draw_decision_boundaries(ax, linewidth=0.8, alpha=0.8, extend=0.15)
    draw_simplex_frame(ax)
    draw_boundary_labels(ax)
    draw_region_labels(ax, region_labels)
    setup_simplex_axes(ax)


# ---------------------------------------------------------------------------
# Force grid evaluation
# ---------------------------------------------------------------------------


def _evaluate_forces_on_grid(resolution=500, margin=0.005):
    """Evaluate both Bregman forces in a single pass over the grid."""
    pad = 0.08
    xs = np.linspace(-pad, 1.0 + pad, resolution)
    ys = np.linspace(-pad, VERTEX_S3[1] + pad, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    complement_grid = np.empty(grid_x.shape)
    substitute_grid = np.empty(grid_x.shape)

    for i in range(resolution):
        for j in range(resolution):
            b1, b2, b3 = xy_to_bary(grid_x[i, j], grid_y[i, j])
            belief = clamp_to_interior(b1, b2, b3, margin)
            complement_grid[i, j], substitute_grid[i, j] = bregman_forces(belief)

    return grid_x, grid_y, complement_grid, substitute_grid


# ---------------------------------------------------------------------------
# Panel C: Phase transition along a ray
# ---------------------------------------------------------------------------


def render_force_transition_ray(save_path: str) -> None:
    t = np.linspace(0, 0.18, 500)

    complement_values = np.empty_like(t)
    substitute_values = np.empty_like(t)

    for idx, ti in enumerate(t):
        belief = np.array([1 / 4 + ti, 1 / 6, 7 / 12 - ti])
        complement_values[idx], substitute_values[idx] = bregman_forces(belief)

    # Find crossing point (interaction boundary)
    diff = complement_values - substitute_values
    t_cross = None
    for i in range(len(diff) - 1):
        if diff[i] >= 0 and diff[i + 1] < 0:
            t_cross = t[i] + (t[i + 1] - t[i]) * diff[i] / (diff[i] - diff[i + 1])
            break

    t_db = 7 / 60  # Decision boundary

    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)

    # Full-height background shading for complement/substitute regions
    if t_cross is not None:
        ax.axvspan(0, t_cross, color="firebrick", alpha=0.05, zorder=0)
        ax.axvspan(t_cross, t[-1], color="steelblue", alpha=0.05, zorder=0)

    # Two force curves
    ax.plot(
        t,
        complement_values,
        "-",
        color="firebrick",
        linewidth=1.8,
        label="$\\mathbb{E}[D_g]$ (complement force)",
    )
    ax.plot(
        t,
        substitute_values,
        "-",
        color="steelblue",
        linewidth=1.8,
        label="$\\mathbb{E}[D_h]$ (substitute force)",
    )

    # Boundary lines — visually distinct styles
    if t_cross is not None:
        ax.axvline(t_cross, color="0.3", linewidth=1.0, linestyle=":")
        ax.annotate(
            "interaction\nboundary",
            (t_cross, 0.92),
            fontsize=8,
            color="0.3",
            ha="right",
            va="center",
            xytext=(-4, 0),
            textcoords="offset points",
        )

    ax.axvline(t_db, color="0.3", linewidth=1.0, linestyle="--")
    ax.annotate(
        "decision\nboundary",
        (t_db, 0.92),
        fontsize=8,
        color="0.3",
        ha="left",
        va="center",
        xytext=(4, 0),
        textcoords="offset points",
    )

    # Mark starting belief b₂ at t=0
    ax.annotate(
        "$b_2$",
        (0, complement_values[0]),
        textcoords="offset points",
        xytext=(-12, 0),
        fontsize=10,
        color="0.3",
        ha="right",
        va="center",
    )

    # Region labels at top of background spans
    if t_cross is not None:
        t_comp_mid = t_cross * 0.5
        ax.text(
            t_comp_mid,
            0.95,
            "complements",
            fontsize=9,
            color="firebrick",
            ha="center",
            va="top",
            fontstyle="italic",
            alpha=0.8,
            transform=ax.get_xaxis_transform(),
        )
        t_sub_mid = (t_cross + t[-1]) / 2
        ax.text(
            t_sub_mid,
            0.95,
            "substitutes",
            fontsize=9,
            color="steelblue",
            ha="center",
            va="top",
            fontstyle="italic",
            alpha=0.8,
            transform=ax.get_xaxis_transform(),
        )

    # Decision region labels
    ax.text(
        0.03,
        0.72,
        "$\\mathcal{R}_3$",
        fontsize=12,
        color="0.5",
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.text(
        t_db + (t[-1] - t_db) / 2,
        0.72,
        "$\\mathcal{R}_1$",
        fontsize=12,
        color="0.5",
        ha="center",
        va="center",
        fontweight="bold",
    )

    ax.set_xlabel(
        "$t$ along $b(t) = (\\frac{1}{4}+t,\\; \\frac{1}{6},\\; \\frac{7}{12}-t)$",
        fontsize=10,
    )
    ax.set_ylabel("Force magnitude", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_xlim(0, t[-1])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.15)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Hero figure: combined three-panel layout
# ---------------------------------------------------------------------------


def render_interaction_decomposition(save_path: str) -> None:
    """Combined three-panel hero figure for the paper.

    (a) Delta-VoI simplex  (b) Complement force  (c) Substitute force
    Horizontal colorbars: one under (a), one shared under (b)+(c).
    """
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as gridspec

    # ---- Compute grids ----
    gx_a, gy_a, vals_a = evaluate_on_grid(delta_voi, resolution=800)
    vals_a = np.where(np.abs(vals_a) < 1e-6, 1e-6, vals_a)

    gx_f, gy_f, cg, sg = _evaluate_forces_on_grid(resolution=500)

    # Color norms
    interior_a = _simplex_interior_mask(gx_a, gy_a)
    cap_a = min(np.percentile(np.abs(vals_a[interior_a]), 97), 1.5)
    norm_a = TwoSlopeNorm(vmin=-cap_a, vcenter=0, vmax=cap_a)

    interior_f = _simplex_interior_mask(gx_f, gy_f)
    force_cap = min(
        np.percentile(np.concatenate([cg[interior_f], sg[interior_f]]), 98), 3.5
    )
    norm_f = mcolors.Normalize(vmin=0, vmax=force_cap)

    # ---- Layout: 2 rows (panels + colorbars), 3 columns ----
    fig = plt.figure(figsize=(14, 5.2))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[1, 0.04],
        width_ratios=[1, 1, 1],
        wspace=0.15,
        hspace=0.22,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    cax_a = fig.add_subplot(gs[1, 0])
    cax_bc = fig.add_subplot(gs[1, 1:])

    # ---- Panel (a): Delta-VoI ----
    clip_a = add_simplex_clip(ax_a)
    mesh_a = ax_a.pcolormesh(
        gx_a,
        gy_a,
        vals_a,
        cmap="RdBu_r",
        norm=norm_a,
        shading="auto",
        rasterized=True,
    )
    mesh_a.set_clip_path(clip_a)
    draw_decision_boundaries(ax_a, linewidth=0.8, alpha=0.8, extend=0.15)
    draw_simplex_frame(ax_a)
    setup_simplex_axes(ax_a)

    draw_region_labels(ax_a)
    draw_boundary_labels(ax_a)

    # Marked belief points
    for belief, color, label, offset in [
        (BELIEF_B1, "#b2182b", "$b_1$", (0, 8)),  # dark red — complement
        (BELIEF_B2, "#e08214", "$b_2$", (0, 8)),  # amber — boundary complement
        (BELIEF_B3, "#6baed6", "$b_3$", (0, 8)),  # light blue — substitute
    ]:
        x, y = bary_to_xy(*belief)
        ax_a.plot(
            x,
            y,
            "*",
            color=color,
            markersize=11,
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=5,
        )
        ax_a.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=10,
            color=color,
            fontweight="bold",
            zorder=5,
        )

    # Ray b(t)
    t_ray = np.linspace(0, 0.35, 200)
    t_ray = t_ray[7 / 12 - t_ray > 0.02]
    ray_x, ray_y = bary_to_xy(1 / 4 + t_ray, np.full_like(t_ray, 1 / 6), 7 / 12 - t_ray)
    ax_a.plot(ray_x, ray_y, ":", color="0.3", linewidth=1.0, alpha=0.5)
    mid = len(t_ray) // 2
    ax_a.annotate(
        "$b(t)$",
        (ray_x[mid], ray_y[mid]),
        textcoords="offset points",
        xytext=(-22, -4),
        fontsize=8,
        color="0.15",
        fontstyle="italic",
    )

    # ---- Panel (b): Complement force ----
    clip_b = add_simplex_clip(ax_b)
    mesh_b = ax_b.pcolormesh(
        gx_f,
        gy_f,
        cg,
        cmap="YlOrRd",
        norm=norm_f,
        shading="auto",
        rasterized=True,
    )
    mesh_b.set_clip_path(clip_b)
    draw_decision_boundaries(ax_b, linewidth=0.8, alpha=0.8, extend=0.15)
    draw_simplex_frame(ax_b)
    setup_simplex_axes(ax_b)

    # ---- Panel (c): Substitute force ----
    clip_c = add_simplex_clip(ax_c)
    mesh_c = ax_c.pcolormesh(
        gx_f,
        gy_f,
        sg,
        cmap="YlOrRd",
        norm=norm_f,
        shading="auto",
        rasterized=True,
    )
    mesh_c.set_clip_path(clip_c)
    draw_decision_boundaries(ax_c, linewidth=0.8, alpha=0.8, extend=0.15)
    draw_simplex_frame(ax_c)
    setup_simplex_axes(ax_c)

    # Boundary labels on force panels
    for ax in [ax_b, ax_c]:
        draw_boundary_labels(ax, fontsize=8)

    # ---- Colorbars (horizontal, below panels) ----
    cb_a = fig.colorbar(mesh_a, cax=cax_a, orientation="horizontal")
    cb_a.set_label("$\\Delta\\mathrm{VoI}(j \\mid i,\\, b)$", fontsize=10)

    cb_bc = fig.colorbar(mesh_b, cax=cax_bc, orientation="horizontal")
    cb_bc.set_label("Force magnitude", fontsize=10)

    # ---- Panel labels and titles ----
    titles = [
        ("Interaction value $\\Delta\\mathrm{VoI}$"),
        ("Complement force $\\mathbb{E}[D_g]$"),
        ("Substitute force $\\mathbb{E}[D_h]$"),
    ]
    for ax, title in zip([ax_a, ax_b, ax_c], titles):
        ax.set_title(title, fontsize=11, pad=8)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Verifying exact values...")
    verify_known_values()

    print("Generating figures...")
    render_interaction_decomposition(os.path.join(OUTPUT_DIR, "interaction_decomposition.pdf"))
    render_force_transition_ray(os.path.join(OUTPUT_DIR, "force_transition_ray.pdf"))
    print("Done.")
