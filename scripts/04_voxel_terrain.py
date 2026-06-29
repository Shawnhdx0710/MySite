#!/usr/bin/env python3
"""
3D Voxel Terrain: Expected Threat (xT) Grid
============================================
Renders the 32×24 xT grid as a series of 3D pillars on a soccer pitch —
a "block heatmap" in the style of modern architectural data visualization.

Output: output/xt_voxel_terrain.png  (high-res PNG)

Usage:
    python scripts/04_voxel_terrain.py

Prerequisites: Run scripts/02_train_xt.py first to generate the grid.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRID_X, GRID_Y = 32, 24
PITCH_LENGTH = 105.0  # metres
PITCH_WIDTH = 68.0

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
GRID_NPY = OUTPUT_DIR / "xt_grid_32x24.npy"
OUTPUT_PNG = OUTPUT_DIR / "xt_voxel_terrain.png"

# Visual scaling: max xT value → bar height in plot units
BAR_HEIGHT_SCALE = 12.0  # tallest bar will be this tall

# Figure
FIG_W, FIG_H = 20, 12
DPI = 200

# Pitch colour (dark base to make glowing bars pop)
PITCH_FACE = "#1a1d23"
PITCH_LINE = "#3a3d44"
BACKGROUND = "#111318"


# ---------------------------------------------------------------------------
# Custom colormap: deep navy → teal → gold → white-hot
# ---------------------------------------------------------------------------
def make_cmap():
    return LinearSegmentedColormap("xt_3d", {
        "red": [
            (0.00, 0.08, 0.08),
            (0.25, 0.00, 0.00),
            (0.50, 0.00, 0.00),
            (0.70, 0.95, 0.95),
            (0.85, 1.00, 1.00),
            (1.00, 1.00, 1.00),
        ],
        "green": [
            (0.00, 0.12, 0.12),
            (0.25, 0.30, 0.30),
            (0.50, 0.70, 0.70),
            (0.70, 0.75, 0.75),
            (0.85, 0.85, 0.85),
            (1.00, 0.95, 0.95),
        ],
        "blue": [
            (0.00, 0.30, 0.30),
            (0.25, 0.55, 0.55),
            (0.50, 0.35, 0.35),
            (0.70, 0.15, 0.15),
            (0.85, 0.10, 0.10),
            (1.00, 0.20, 0.20),
        ],
    }, N=256)


# ---------------------------------------------------------------------------
# Pitch markings on the base plane
# ---------------------------------------------------------------------------
def draw_pitch_base(ax):
    """Draw pitch lines on the z=0 plane."""
    kw = dict(color=PITCH_LINE, linewidth=0.5, alpha=0.5, zorder=0)

    # Outer boundary
    ax.plot([0, PITCH_LENGTH], [0, 0], [0, 0], **kw)
    ax.plot([0, PITCH_LENGTH], [PITCH_WIDTH, PITCH_WIDTH], [0, 0], **kw)
    ax.plot([0, 0], [0, PITCH_WIDTH], [0, 0], **kw)
    ax.plot([PITCH_LENGTH, PITCH_LENGTH], [0, PITCH_WIDTH], [0, 0], **kw)

    # Halfway line
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2],
            [0, PITCH_WIDTH], [0, 0], **kw)

    # Centre circle
    theta = np.linspace(0, 2 * np.pi, 80)
    cx, cy = PITCH_LENGTH / 2, PITCH_WIDTH / 2
    ax.plot(cx + 9.15 * np.cos(theta), cy + 9.15 * np.sin(theta),
            np.zeros(80), **kw)

    # Penalty areas
    for x0, direction in [(0, 1), (PITCH_LENGTH, -1)]:
        pa_x = x0 if direction == 1 else x0 - 16.5
        pa_y = PITCH_WIDTH / 2 - 20.15
        xs = [pa_x, pa_x + 16.5, pa_x + 16.5, pa_x, pa_x]
        ys = [pa_y, pa_y, pa_y + 40.3, pa_y + 40.3, pa_y]
        ax.plot(xs, ys, np.zeros(5), **kw)

        # 6-yard box
        g6_x = x0 if direction == 1 else x0 - 5.5
        g6_y = PITCH_WIDTH / 2 - 9.16
        xs = [g6_x, g6_x + 5.5, g6_x + 5.5, g6_x, g6_x]
        ys = [g6_y, g6_y, g6_y + 18.32, g6_y + 18.32, g6_y]
        ax.plot(xs, ys, np.zeros(5), **kw)

    # Goals (filled rectangles)
    for x0 in [0, PITCH_LENGTH]:
        ax.bar3d(
            x0 - 0.3, PITCH_WIDTH / 2 - 3.66, 0,
            0.3, 7.32, 0.02,
            color="#555a60", alpha=0.6, shade=False, zorder=0,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load
    if not GRID_NPY.exists():
        raise FileNotFoundError(f"{GRID_NPY} not found. Run 02_train_xt.py first.")
    grid = np.load(str(GRID_NPY))  # shape (24, 32) = (y-bins, x-bins)

    # Cell dimensions
    cell_w = PITCH_LENGTH / GRID_X
    cell_h = PITCH_WIDTH / GRID_Y
    gap_w = cell_w * 0.08   # subtle gap between bars
    gap_h = cell_h * 0.08

    bar_w = cell_w - gap_w
    bar_d = cell_h - gap_h

    # Normalise xT → bar height
    max_xt = grid.max()
    heights = grid / max_xt * BAR_HEIGHT_SCALE

    # Build positions for every bar (centred in its cell)
    xs = np.arange(GRID_X) * cell_w + gap_w / 2
    ys = np.arange(GRID_Y) * cell_h + gap_h / 2
    xpos, ypos = np.meshgrid(xs, ys)  # each is (24, 32)

    xpos_flat = xpos.ravel()
    ypos_flat = ypos.ravel()
    zpos_flat = np.zeros_like(xpos_flat)
    heights_flat = heights.ravel()
    values_flat = grid.ravel()

    # Colour normalisation
    norm = Normalize(vmin=0, vmax=max_xt)
    cmap = make_cmap()
    facecolors = cmap(norm(values_flat))

    # ---- figure ----
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BACKGROUND)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BACKGROUND)

    # ---- bars ----
    bar_container = ax.bar3d(
        xpos_flat, ypos_flat, zpos_flat,
        bar_w, bar_d, heights_flat,
        color=facecolors,
        shade=True,           # <-- 3D lighting: sides darker than top faces
        edgecolor="none",
        alpha=0.92,
        zorder=1,
    )

    # ---- pitch base ----
    draw_pitch_base(ax)

    # Ground plane (semi-transparent to catch light and ground the bars)
    xx = np.array([0, PITCH_LENGTH])
    yy = np.array([0, PITCH_WIDTH])
    xxg, yyg = np.meshgrid(xx, yy)
    ax.plot_surface(
        xxg, yyg, np.zeros_like(xxg),
        color=PITCH_FACE, alpha=0.35, shade=False,
        zorder=0, antialiased=True,
    )

    # ---- camera ----
    # Isometric-ish view: elevated, angled so goal peaks don't hide midfield
    ax.view_init(elev=38, azim=-55)
    # Make sure camera is somewhat centered on the attacking third
    ax.set_proj_type("persp", focal_length=0.15)

    # ---- axes cleanup ----
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_zlim(0, BAR_HEIGHT_SCALE * 1.05)

    # Remove panes / grids / ticks for clean look
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(PITCH_LINE)
    ax.yaxis.pane.set_edgecolor(PITCH_LINE)
    ax.zaxis.pane.set_edgecolor(PITCH_LINE)
    ax.xaxis.pane.set_alpha(0.15)
    ax.yaxis.pane.set_alpha(0.15)
    ax.zaxis.pane.set_alpha(0.15)
    ax.grid(False)

    # ---- colour bar ----
    cbar_ax = fig.add_axes([0.88, 0.18, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Expected Threat (xT)", color="#aaa", fontsize=11,
                   fontfamily="sans-serif", labelpad=10)
    cbar.ax.yaxis.set_tick_params(color="#888", labelsize=9)
    cbar.outline.set_edgecolor("#444")
    for label in cbar.ax.get_yticklabels():
        label.set_color("#aaa")

    # ---- title ----
    fig.suptitle(
        "Expected Threat — 32×24 Grid",
        x=0.40, y=0.92,
        fontsize=18, fontweight="bold", color="#ddd",
        fontfamily="sans-serif",
    )
    ax.set_title(
        f"3,927 matches · 13.7M actions · Max xT = {max_xt:.4f} · "
        f"grid = {GRID_X}×{GRID_Y} ({GRID_X * GRID_Y} cells)",
        fontsize=10, color="#777", pad=0,
        fontfamily="sans-serif",
    )

    # ---- attribution ----
    fig.text(
        0.40, 0.06,
        "Attack →",
        fontsize=10, color="#555", fontfamily="sans-serif",
        ha="center",
    )

    # ---- save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(OUTPUT_PNG), dpi=DPI,
        facecolor=fig.get_facecolor(), edgecolor="none",
        bbox_inches="tight",
    )
    print(f"Saved: {OUTPUT_PNG}")
    print(f"  {GRID_X}×{GRID_Y} grid — {GRID_X * GRID_Y} pillars")
    print(f"  Height range: [{heights_flat.min():.2f}, {heights_flat.max():.2f}]")
    print(f"  xT range:     [{grid.min():.6f}, {grid.max():.6f}]")


if __name__ == "__main__":
    main()
