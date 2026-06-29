#!/usr/bin/env python3
"""
3D Voxel Terrain: Expected Threat (xT) Grid
============================================
Renders the 32×24 xT grid as a series of 3D pillars on a soccer pitch —
a publication-ready "block heatmap" on a light architectural background.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRID_X, GRID_Y = 32, 24
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
GRID_NPY = OUTPUT_DIR / "xt_grid_32x24.npy"
OUTPUT_PNG = OUTPUT_DIR / "xt_voxel_terrain.png"

BAR_HEIGHT_SCALE = 12.0
DPI = 200

# ---- light-theme palette ----
BACKGROUND = "#f5f3ef"       # warm paper
PITCH_FACE = "#e8e4dc"       # base-plane tone
PITCH_LINE = "#c4bfb4"       # pitch markings
GRID_LINE  = "#ddd9d0"       # cell-boundary grid
PANE_COLOR = (0.88, 0.86, 0.82, 0.30)  # barely-there pane
TEXT_DARK  = "#3a3732"       # titles / labels
TEXT_MID   = "#8a8580"       # subtitle / ticks
BAR_EDGE   = "#000000"       # thin dark edge per bar for definition


# ---------------------------------------------------------------------------
# Custom colormap: cool seafoam → amber → crimson (light-bg friendly)
# ---------------------------------------------------------------------------
def make_cmap():
    return LinearSegmentedColormap("xt_light", {
        "red": [
            (0.00, 0.10, 0.10),
            (0.30, 0.00, 0.00),
            (0.55, 0.00, 0.00),
            (0.70, 0.95, 0.95),
            (0.85, 1.00, 1.00),
            (1.00, 0.75, 0.75),
        ],
        "green": [
            (0.00, 0.55, 0.55),
            (0.30, 0.65, 0.65),
            (0.55, 0.60, 0.60),
            (0.70, 0.55, 0.55),
            (0.85, 0.25, 0.25),
            (1.00, 0.05, 0.05),
        ],
        "blue": [
            (0.00, 0.70, 0.70),
            (0.30, 0.60, 0.60),
            (0.55, 0.05, 0.05),
            (0.70, 0.00, 0.00),
            (0.85, 0.00, 0.00),
            (1.00, 0.00, 0.00),
        ],
    }, N=256)


# ---------------------------------------------------------------------------
# Pitch markings on the z=0 base plane
# ---------------------------------------------------------------------------
def draw_pitch_base(ax):
    kw = dict(color=PITCH_LINE, linewidth=0.6, alpha=0.55)

    # boundary
    ax.plot([0, PITCH_LENGTH], [0, 0],              [0, 0], **kw)
    ax.plot([0, PITCH_LENGTH], [PITCH_WIDTH]*2,     [0, 0], **kw)
    ax.plot([0, 0],            [0, PITCH_WIDTH],     [0, 0], **kw)
    ax.plot([PITCH_LENGTH]*2,  [0, PITCH_WIDTH],     [0, 0], **kw)

    # halfway
    ax.plot([PITCH_LENGTH/2]*2, [0, PITCH_WIDTH], [0, 0], **kw)

    # centre circle
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(PITCH_LENGTH/2 + 9.15*np.cos(t),
            PITCH_WIDTH/2  + 9.15*np.sin(t),
            np.zeros(100), **kw)

    # penalty areas + 6-yard boxes
    for x0, dx in [(0, 1), (PITCH_LENGTH, -1)]:
        px = x0 if dx == 1 else x0 - 16.5
        py = PITCH_WIDTH/2 - 20.15
        xs = [px, px+16.5, px+16.5, px, px]
        ys = [py, py, py+40.3, py+40.3, py]
        ax.plot(xs, ys, np.zeros(5), **kw)

        gx = x0 if dx == 1 else x0 - 5.5
        gy = PITCH_WIDTH/2 - 9.16
        xs = [gx, gx+5.5, gx+5.5, gx, gx]
        ys = [gy, gy, gy+18.32, gy+18.32, gy]
        ax.plot(xs, ys, np.zeros(5), **kw)

    # goals
    for x0 in [0, PITCH_LENGTH]:
        ax.bar3d(x0 - 0.3, PITCH_WIDTH/2 - 3.66, 0,
                 0.3, 7.32, 0.02,
                 color="#b0aba0", alpha=0.45, shade=False, zorder=0)

    # cell grid on base
    cw, ch = PITCH_LENGTH / GRID_X, PITCH_WIDTH / GRID_Y
    gkw = dict(color=GRID_LINE, linewidth=0.25, alpha=0.45)
    for i in range(GRID_X + 1):
        ax.plot([i*cw]*2, [0, PITCH_WIDTH], [0, 0], **gkw)
    for j in range(GRID_Y + 1):
        ax.plot([0, PITCH_LENGTH], [j*ch]*2, [0, 0], **gkw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not GRID_NPY.exists():
        raise FileNotFoundError(f"{GRID_NPY} not found. Run 02_train_xt.py first.")
    grid = np.load(str(GRID_NPY))

    # ---- geometry ----
    cw, ch = PITCH_LENGTH / GRID_X, PITCH_WIDTH / GRID_Y
    gap = 0.10
    bar_w, bar_d = cw * (1 - gap), ch * (1 - gap)
    ox, oy = cw * gap / 2, ch * gap / 2

    max_xt = grid.max()
    heights = grid / max_xt * BAR_HEIGHT_SCALE

    xs = np.arange(GRID_X) * cw + ox
    ys = np.arange(GRID_Y) * ch + oy
    xpos, ypos = np.meshgrid(xs, ys)

    norm = Normalize(vmin=0, vmax=max_xt)
    cmap = make_cmap()
    facecolors = cmap(norm(grid.ravel()))

    # ---- figure ----
    fig = plt.figure(figsize=(22, 13), facecolor=BACKGROUND)

    # 3D axes — shifted rightward so left edge of pitch is fully visible
    ax = fig.add_axes([0.22, -0.02, 0.60, 1.00], projection="3d")
    ax.set_facecolor(BACKGROUND)

    # ---- bars ----
    ax.bar3d(
        xpos.ravel(), ypos.ravel(), np.zeros(GRID_X * GRID_Y),
        bar_w, bar_d, heights.ravel(),
        color=facecolors,
        shade=True,
        edgecolor=BAR_EDGE,
        linewidth=0.15,
        alpha=0.95,
        zorder=2,
    )

    # ---- pitch ----
    draw_pitch_base(ax)
    xx, yy = np.meshgrid([0, PITCH_LENGTH], [0, PITCH_WIDTH])
    ax.plot_surface(xx, yy, np.zeros_like(xx),
                    color=PITCH_FACE, alpha=0.55, shade=False,
                    zorder=0, antialiased=True)

    # ---- camera ----
    ax.view_init(elev=22, azim=-55)
    ax.set_proj_type("persp", focal_length=0.15)

    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_zlim(0, BAR_HEIGHT_SCALE * 1.05)

    # ---- panes ----
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_major_formatter(NullFormatter())
        axis.pane.fill = True
        axis.pane.set_facecolor(PANE_COLOR)
        axis.pane.set_edgecolor("none")
    ax.grid(False)

    # ---- colorbar (thin, right) ----
    cbar_ax = fig.add_axes([0.85, 0.18, 0.010, 0.58])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Expected Threat (xT)", color=TEXT_DARK, fontsize=10,
                   fontfamily="sans-serif", labelpad=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_MID, labelsize=8)
    cbar.outline.set_edgecolor(PITCH_LINE)
    for l in cbar.ax.get_yticklabels():
        l.set_color(TEXT_MID)

    # ---- title / subtitle ----
    fig.suptitle("Expected Threat (xT) Grid  |  BY @shawnhdx0710",
                 x=0.50, y=0.955, fontsize=20, fontweight="bold",
                 color=TEXT_DARK, fontfamily="sans-serif")
    fig.text(0.50, 0.918, "13.7M actions from Statsbomb Open Data",
             fontsize=11, color=TEXT_MID, fontfamily="sans-serif", ha="center")

    # ---- attack hint ----
    fig.text(0.50, 0.025, "Attack →",
             fontsize=10, color=TEXT_MID, fontfamily="sans-serif", ha="center")

    # ---- save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_PNG), dpi=DPI,
                facecolor=fig.get_facecolor(), edgecolor="none",
                bbox_inches="tight")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"  {GRID_X}×{GRID_Y} grid — {GRID_X * GRID_Y} pillars")
    print(f"  Height range: [{heights.ravel().min():.2f}, {heights.ravel().max():.2f}]")
    print(f"  xT range:     [{grid.min():.6f}, {grid.max():.6f}]")


if __name__ == "__main__":
    main()
