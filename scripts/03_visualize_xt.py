#!/usr/bin/env python3
"""
Phase 4: xT Grid Visualization
================================
Loads the trained xT grid and renders a 2D heatmap showing how expected
threat varies across the pitch. Threat should increase logically toward the
opponent's goal and central areas.

Output: output/xt_heatmap_32x24.png

Usage:
    python scripts/03_visualize_xt.py

Prerequisites: Run scripts/02_train_xt.py first to generate the grid.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRID_X = 32
GRID_Y = 24
PITCH_LENGTH = 105.0  # metres (SPADL convention)
PITCH_WIDTH = 68.0

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
GRID_NPY = OUTPUT_DIR / f"xt_grid_{GRID_X}x{GRID_Y}.npy"
OUTPUT_PNG = OUTPUT_DIR / f"xt_heatmap_{GRID_X}x{GRID_Y}.png"

# Figure dimensions (inches) — wide to match pitch aspect ratio
FIG_W, FIG_H = 18, 8
DPI = 150


# ---------------------------------------------------------------------------
# Custom colormap
# ---------------------------------------------------------------------------
def make_xt_cmap() -> LinearSegmentedColormap:
    """Build a perceptually uniform colormap from cool (low threat) to
    hot (high threat), with a muted cream background for zero."""
    cdict = {
        "red": [
            (0.00, 0.97, 0.97),  # cream
            (0.05, 0.20, 0.20),  # deep blue
            (0.30, 0.00, 0.00),  # teal
            (0.55, 0.00, 0.00),  # green
            (0.75, 1.00, 1.00),  # yellow
            (0.90, 1.00, 1.00),  # orange
            (1.00, 0.65, 0.65),  # red
        ],
        "green": [
            (0.00, 0.98, 0.98),
            (0.05, 0.30, 0.30),
            (0.30, 0.60, 0.60),
            (0.55, 0.80, 0.80),
            (0.75, 0.85, 0.85),
            (0.90, 0.40, 0.40),
            (1.00, 0.00, 0.00),
        ],
        "blue": [
            (0.00, 0.94, 0.94),
            (0.05, 0.60, 0.60),
            (0.30, 0.70, 0.70),
            (0.55, 0.30, 0.30),
            (0.75, 0.00, 0.00),
            (0.90, 0.00, 0.00),
            (1.00, 0.00, 0.00),
        ],
    }
    return LinearSegmentedColormap("xt_cmap", cdict, N=256)


# ---------------------------------------------------------------------------
# Pitch drawing helpers
# ---------------------------------------------------------------------------
def draw_pitch(ax: plt.Axes) -> None:
    """Draw a simplified football pitch outline on the given axes."""
    # Pitch boundary
    ax.add_patch(mpatches.Rectangle(
        (0, 0), PITCH_LENGTH, PITCH_WIDTH,
        fill=False, edgecolor="white", linewidth=1.2, zorder=3,
    ))
    # Halfway line
    ax.axvline(PITCH_LENGTH / 2, color="white", linewidth=1.0, zorder=3)
    # Centre circle
    centre = plt.Circle(
        (PITCH_LENGTH / 2, PITCH_WIDTH / 2), 9.15,
        fill=False, edgecolor="white", linewidth=0.8, zorder=3,
    )
    ax.add_patch(centre)
    # Centre dot
    ax.plot(PITCH_LENGTH / 2, PITCH_WIDTH / 2, "o", color="white",
            markersize=3, zorder=3)

    # Penalty areas
    for x, direction in [(0, 1), (PITCH_LENGTH, -1)]:
        pa_x = x if direction == 1 else x - 16.5
        ax.add_patch(mpatches.Rectangle(
            (pa_x, PITCH_WIDTH / 2 - 20.15), 16.5, 40.3,
            fill=False, edgecolor="white", linewidth=0.8, zorder=3,
        ))
        # Six-yard box
        g6_x = x if direction == 1 else x - 5.5
        ax.add_patch(mpatches.Rectangle(
            (g6_x, PITCH_WIDTH / 2 - 9.16), 5.5, 18.32,
            fill=False, edgecolor="white", linewidth=0.8, zorder=3,
        ))
        # Goal
        goal_x = x - 0.5 * direction
        ax.add_patch(mpatches.Rectangle(
            (goal_x, PITCH_WIDTH / 2 - 3.66),
            0.5, 7.32,
            fill=True, facecolor="white", edgecolor="white",
            linewidth=0.5, zorder=3,
        ))

    # Attack direction arrow
    ax.annotate(
        "ATTACK →", xy=(PITCH_LENGTH - 3, 1.5),
        fontsize=8, color="white", alpha=0.6,
        ha="right", va="bottom",
    )


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------
def main() -> None:
    # Load grid
    if not GRID_NPY.exists():
        raise FileNotFoundError(
            f"{GRID_NPY} not found. Run scripts/02_train_xt.py first."
        )
    grid = np.load(str(GRID_NPY))  # shape (24, 32) = (y-bins, x-bins)

    # Cell dimensions
    cell_w = PITCH_LENGTH / GRID_X
    cell_h = PITCH_WIDTH / GRID_Y

    # Build (x, y) mesh for pcolormesh — edges, not centres
    x_edges = np.linspace(0, PITCH_LENGTH, GRID_X + 1)
    y_edges = np.linspace(0, PITCH_WIDTH, GRID_Y + 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    # Colormap
    cmap = make_xt_cmap()

    # Heatmap — transpose grid so rows = y (width), cols = x (length)
    # grid shape is (24, 32) = (y, x) — pcolormesh expects (y, x)
    vmax = np.percentile(grid[grid > 0], 98) if (grid > 0).any() else grid.max()
    im = ax.pcolormesh(
        x_edges, y_edges, grid,
        cmap=cmap, shading="flat",
        vmin=0, vmax=max(vmax, 0.001),
        zorder=1, alpha=0.9,
    )

    # Draw pitch markings
    draw_pitch(ax)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
    cbar.set_label("Expected Threat (xT)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # Labels
    ax.set_xlabel("Pitch length (m)  —  attacking direction →",
                  color="white", fontsize=10)
    ax.set_ylabel("Pitch width (m)", color="white", fontsize=10)

    # Title
    nonzero_pct = (grid > 0).sum() / grid.size * 100
    ax.set_title(
        f"Expected Threat (xT) Grid — {GRID_X}×{GRID_Y}  |  BY @shawnhdx0710",
        color="white", fontsize=12, fontweight="bold", pad=12,
    )

    # Tidy up
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_aspect("equal")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(OUTPUT_PNG), dpi=DPI, facecolor=fig.get_facecolor(),
        edgecolor="none", bbox_inches="tight",
    )
    print(f"Saved: {OUTPUT_PNG}")
    print(f"  Dimensions: {FIG_W}×{FIG_H}″ @ {DPI} dpi")
    print(f"  Grid shape: {grid.shape}  range=[{grid.min():.6f}, {grid.max():.6f}]")


if __name__ == "__main__":
    main()