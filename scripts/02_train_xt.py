#!/usr/bin/env python3
"""
Phase 3: xT Model Training
============================
Loads all cached SPADL Parquet files and trains a 32×24 Expected Threat grid
using socceraction's `ExpectedThreat` model.

The model is trained only on successful 'moving' actions (passes, dribbles,
crosses) to compute transition probabilities. Shots are used to compute
the per-cell scoring probability.

Outputs (written to output/):
  - xt_grid_32x24.npy    — NumPy binary (2D array, shape 24×32)
  - xt_grid_32x24.csv    — human-readable CSV
  - xt_grid_32x24.json   — socceraction-native JSON (loadable with load_model)

Usage:
    python scripts/02_train_xt.py

Prerequisites: Run scripts/01_cache_spadl.py first to populate the cache.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from socceraction.xthreat import ExpectedThreat
from socceraction.spadl.config import actiontypes, results

import pyarrow.parquet as pq
from xt_utils import CACHE_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRID_X = 32  # bins along the pitch length (105m → ~3.28m per cell)
GRID_Y = 24  # bins along the pitch width (68m → ~2.83m per cell)
EPS = 1e-5    # convergence threshold for value iteration

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_NPY = OUTPUT_DIR / f"xt_grid_{GRID_X}x{GRID_Y}.npy"
OUTPUT_CSV = OUTPUT_DIR / f"xt_grid_{GRID_X}x{GRID_Y}.csv"
OUTPUT_JSON = OUTPUT_DIR / f"xt_grid_{GRID_X}x{GRID_Y}.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_all_spadl() -> pd.DataFrame:
    """
    Load all cached SPADL Parquet files into a single DataFrame.

    Uses pyarrow's ParquetDataset for efficient multi-file read — only the
    columns needed by the xT model are loaded, minimising RAM pressure on M1.
    """
    parquet_files = sorted(CACHE_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found in {CACHE_DIR}. "
            "Run scripts/01_cache_spadl.py first."
        )

    print(f"Loading {len(parquet_files):,} cached Parquet files ...")

    # The xT model only needs: start_x, start_y, end_x, end_y, type_id, result_id
    columns_needed = ["start_x", "start_y", "end_x", "end_y", "type_id", "result_id"]

    # Read all files as a single pyarrow table, then convert to pandas
    t0 = time.perf_counter()
    table = pq.read_table(
        str(CACHE_DIR),
        columns=columns_needed,
    )
    df = table.to_pandas()
    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(df):,} SPADL actions in {elapsed:.1f}s "
          f"({df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB RAM)")

    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_xt(actions: pd.DataFrame) -> ExpectedThreat:
    """Train the ExpectedThreat model on the given SPADL actions."""

    # Quick diagnostics before training
    n_shots = (actions["type_id"] == actiontypes.index("shot")).sum()
    n_goals = (
        (actions["type_id"] == actiontypes.index("shot"))
        & (actions["result_id"] == results.index("success"))
    ).sum()

    move_ids = [actiontypes.index(a) for a in ["pass", "dribble", "cross"]]
    n_moves = actions["type_id"].isin(move_ids).sum()
    n_moves_success = (
        actions["type_id"].isin(move_ids)
        & (actions["result_id"] == results.index("success"))
    ).sum()

    print(f"\nTraining data summary:")
    print(f"  Total actions loaded:  {len(actions):,}")
    print(f"  Shots:                 {n_shots:,} ({n_goals:,} goals)")
    print(f"  Moving actions:        {n_moves:,} ({n_moves_success:,} successful)")
    print(f"  Shot conversion:       {n_goals/n_shots*100:.1f}%" if n_shots > 0 else "")

    # Initialise and fit
    print(f"\nTraining xT model (grid={GRID_X}×{GRID_Y}, eps={EPS}) ...")
    t0 = time.perf_counter()

    model = ExpectedThreat(l=GRID_X, w=GRID_Y, eps=EPS)
    model.fit(actions)

    elapsed = time.perf_counter() - t0
    print(f"  Converged in {elapsed:.1f}s ({len(model.heatmaps)} iterations)")

    return model


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
def save_outputs(model: ExpectedThreat) -> None:
    """Persist the xT grid in multiple formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = model.xT  # shape: (w, l) = (24, 32) — rows=y, cols=x

    # NumPy binary
    np.save(str(OUTPUT_NPY), grid)
    print(f"\nSaved: {OUTPUT_NPY}  shape={grid.shape}  dtype={grid.dtype}")

    # CSV (with header row and index column for orientation)
    df_csv = pd.DataFrame(
        grid,
        index=[f"y={i}" for i in range(grid.shape[0])],
        columns=[f"x={j}" for j in range(grid.shape[1])],
    )
    df_csv.to_csv(str(OUTPUT_CSV))
    print(f"Saved: {OUTPUT_CSV}")

    # socceraction-native JSON
    model.save_model(str(OUTPUT_JSON))
    print(f"Saved: {OUTPUT_JSON}")

    # Quick stats
    print(f"\nxT grid stats:")
    print(f"  Min:      {grid.min():.6f}")
    print(f"  Max:      {grid.max():.6f}")
    print(f"  Mean:     {grid.mean():.6f}")
    print(f"  Non-zero: {(grid > 0).sum()} / {grid.size} cells "
          f"({(grid > 0).sum() / grid.size * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    start = time.perf_counter()

    # 1. Load all cached SPADL
    actions = load_all_spadl()

    # 2. Train
    model = train_xt(actions)

    # 3. Save
    save_outputs(model)

    elapsed = time.perf_counter() - start
    print(f"\n{'='*50}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()