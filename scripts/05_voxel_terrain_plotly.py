#!/usr/bin/env python3
"""
Interactive 3D Voxel Terrain: Expected Threat (xT) Grid (Plotly)
=================================================================
Single-Mesh3d, fully interactive — drag to rotate, scroll to zoom,
hover any pillar to see its grid cell and exact xT value.

Output: output/xt_voxel_terrain.html  (self-contained HTML)

Usage:
    python scripts/05_voxel_terrain_plotly.py
"""

import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRID_X, GRID_Y = 32, 24
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0

OUTPUT_DIR  = Path(__file__).resolve().parent.parent / "output"
GRID_NPY    = OUTPUT_DIR / "xt_grid_32x24.npy"
OUTPUT_HTML = OUTPUT_DIR / "xt_voxel_terrain.html"

BAR_HEIGHT_SCALE = 12.0
COLORSCALE = "plasma"

# Light-theme palette
BG          = "#f5f3ef"
PITCH_COLOR = "#c4bfb4"
GRID_COLOR  = "#ddd9d0"
TEXT_DARK   = "#3a3732"
TEXT_MID    = "#8a8580"
PANE_COLOR  = "rgba(220,216,208,0.25)"

# ---------------------------------------------------------------------------
# Cuboid template  (8 vertices, 12 triangles)
# ---------------------------------------------------------------------------
# Vertex layout (unit cube at origin):
#   4-------7   y
#  /|      /|   |
# 0-------3 |   o---x
# | 5-----|-6  /
# |/      |/  z
# 1-------2
V_UNIT = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom (z=0)
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top    (z=1)
], dtype=float)

# Face triangulation (12 triangles × 3 vertex indices per cuboid)
TRI_UNIT = np.array([
    [0,1,2], [0,2,3],   # bottom
    [4,5,6], [4,6,7],   # top
    [0,1,5], [0,5,4],   # front
    [2,3,7], [2,7,6],   # back
    [0,3,7], [0,7,4],   # left
    [1,2,6], [1,6,5],   # right
], dtype=int)

N_CELLS = GRID_X * GRID_Y   # 768
V_PER   = 8
T_PER   = 12

# ---------------------------------------------------------------------------
# Pitch markings
# ---------------------------------------------------------------------------
def pitch_lines():
    traces = []
    lw, lc = 1.5, PITCH_COLOR
    def line(xs, ys, zs=0):
        return go.Scatter3d(
            x=xs, y=ys, z=[zs]*len(xs),
            mode="lines", line=dict(width=lw, color=lc),
            hoverinfo="skip", showlegend=False,
        )
    # boundary
    traces.append(line([0, PITCH_LENGTH], [0, 0]))
    traces.append(line([0, PITCH_LENGTH], [PITCH_WIDTH, PITCH_WIDTH]))
    traces.append(line([0, 0], [0, PITCH_WIDTH]))
    traces.append(line([PITCH_LENGTH, PITCH_LENGTH], [0, PITCH_WIDTH]))
    # halfway
    traces.append(line([PITCH_LENGTH/2]*2, [0, PITCH_WIDTH]))
    # centre circle
    t = np.linspace(0, 2*np.pi, 100)
    traces.append(line(PITCH_LENGTH/2 + 9.15*np.cos(t),
                       PITCH_WIDTH/2  + 9.15*np.sin(t)))
    # penalty areas + 6-yard boxes
    for x0, dx in [(0, 1), (PITCH_LENGTH, -1)]:
        px = x0 if dx == 1 else x0 - 16.5
        py = PITCH_WIDTH/2 - 20.15
        traces.append(line([px, px+16.5, px+16.5, px, px],
                           [py, py, py+40.3, py+40.3, py]))
        gx = x0 if dx == 1 else x0 - 5.5
        gy = PITCH_WIDTH/2 - 9.16
        traces.append(line([gx, gx+5.5, gx+5.5, gx, gx],
                           [gy, gy, gy+18.32, gy+18.32, gy]))
    # cell grid
    cw, ch = PITCH_LENGTH/GRID_X, PITCH_WIDTH/GRID_Y
    for i in range(GRID_X + 1):
        traces.append(go.Scatter3d(
            x=[i*cw]*2, y=[0, PITCH_WIDTH], z=[0, 0],
            mode="lines", line=dict(width=0.4, color=GRID_COLOR),
            hoverinfo="skip", showlegend=False,
        ))
    for j in range(GRID_Y + 1):
        traces.append(go.Scatter3d(
            x=[0, PITCH_LENGTH], y=[j*ch]*2, z=[0, 0],
            mode="lines", line=dict(width=0.4, color=GRID_COLOR),
            hoverinfo="skip", showlegend=False,
        ))
    return traces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not GRID_NPY.exists():
        raise FileNotFoundError(f"{GRID_NPY} not found. Run 02_train_xt.py first.")
    grid = np.load(str(GRID_NPY))  # (24, 32)  rows=y, cols=x

    cw, ch = PITCH_LENGTH / GRID_X, PITCH_WIDTH / GRID_Y
    gap = 0.10
    bw, bd = cw * (1 - gap), ch * (1 - gap)
    ox, oy = cw * gap / 2, ch * gap / 2

    max_xt = grid.max()
    heights = grid / max_xt * BAR_HEIGHT_SCALE

    # ---- pre-allocate master arrays ----
    n_verts  = N_CELLS * V_PER   # 6,144
    n_tris   = N_CELLS * T_PER   # 9,216

    x_master = np.empty(n_verts, dtype=float)
    y_master = np.empty(n_verts, dtype=float)
    z_master = np.empty(n_verts, dtype=float)
    i_master = np.empty(n_tris, dtype=int)
    j_master = np.empty(n_tris, dtype=int)
    k_master = np.empty(n_tris, dtype=int)
    intensity = np.empty(n_verts, dtype=float)
    customdata = np.empty((n_verts, 3), dtype=object)  # [col, row, xt_value]

    for p in range(N_CELLS):
        col = p % GRID_X
        row = p // GRID_X
        x0  = col * cw + ox
        y0  = row * ch + oy
        h   = heights[row, col]
        val = grid[row, col]

        v0 = p * V_PER
        v  = V_UNIT * [bw, bd, h] + [x0, y0, 0.0]
        x_master[v0:v0+V_PER] = v[:, 0]
        y_master[v0:v0+V_PER] = v[:, 1]
        z_master[v0:v0+V_PER] = v[:, 2]
        intensity[v0:v0+V_PER] = val
        customdata[v0:v0+V_PER] = [col, row, val]

        t0 = p * T_PER
        i_master[t0:t0+T_PER] = TRI_UNIT[:, 0] + v0
        j_master[t0:t0+T_PER] = TRI_UNIT[:, 1] + v0
        k_master[t0:t0+T_PER] = TRI_UNIT[:, 2] + v0

    # ---- single consolidated Mesh3d ----
    mesh = go.Mesh3d(
        x=x_master, y=y_master, z=z_master,
        i=i_master, j=j_master, k=k_master,
        intensity=intensity,
        colorscale=COLORSCALE,
        cmin=0, cmax=max_xt,
        flatshading=True,
        customdata=customdata,
        hovertemplate=(
            "Coordinate: (%{customdata[0]}, %{customdata[1]})<br>"
            "Expected Threat: %{customdata[2]:.4f}"
            "<extra></extra>"
        ),
        showlegend=False,
        name="",
        colorbar=dict(
            title=dict(
                text="Expected<br>Threat (xT)",
                font=dict(size=12, color=TEXT_DARK, family="Inter, sans-serif"),
            ),
            tickfont=dict(size=10, color=TEXT_MID),
            thickness=14,
            len=0.60,
            outlinecolor=PITCH_COLOR,
            outlinewidth=1,
            tickformat=".3f",
        ),
    )

    # ---- ground plane ----
    ground = go.Mesh3d(
        x=[0, PITCH_LENGTH, PITCH_LENGTH, 0],
        y=[0, 0, PITCH_WIDTH, PITCH_WIDTH],
        z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 3], k=[2, 1],
        color="rgba(220,216,208,0.55)",
        flatshading=True,
        hoverinfo="skip",
        showlegend=False,
    )

    # ---- figure ----
    fig = go.Figure(
        data=[ground, mesh] + pitch_lines(),
    )

    # ---- layout ----
    fig.update_layout(
        title=dict(
            text="Expected Threat (xT) Grid  |  BY @shawnhdx0710",
            font=dict(size=22, color=TEXT_DARK, family="Inter, sans-serif"),
            x=0.5, y=0.97,
        ),
        scene=dict(
            xaxis=dict(range=[0, PITCH_LENGTH], visible=False,
                       backgroundcolor=PANE_COLOR),
            yaxis=dict(range=[0, PITCH_WIDTH], visible=False,
                       backgroundcolor=PANE_COLOR),
            zaxis=dict(range=[0, BAR_HEIGHT_SCALE * 1.05], visible=False,
                       backgroundcolor=PANE_COLOR),
            camera=dict(
                eye=dict(x=0.9, y=-0.5, z=0.55),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.05),
            ),
            aspectmode="manual",
            aspectratio=dict(x=PITCH_LENGTH, y=PITCH_WIDTH,
                             z=BAR_HEIGHT_SCALE),
            bgcolor=BG,
        ),
        paper_bgcolor=BG,
        hovermode="closest",
        margin=dict(l=10, r=10, t=70, b=10),
    )

    # ---- save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUTPUT_HTML), include_plotlyjs=True, full_html=True)
    print(f"Saved: {OUTPUT_HTML}")
    print(f"  {GRID_X}×{GRID_Y} grid — {N_CELLS} pillars")
    print(f"  1 Mesh3d trace — {n_verts:,} vertices, {n_tris:,} triangles")
    print(f"  xT range: [{grid.min():.6f}, {grid.max():.6f}]")


if __name__ == "__main__":
    main()
