#!/usr/bin/env python3
"""
Phase 2: Data Engineering & Caching Layer
==========================================
Fetches all available StatsBomb Open Data matches, converts event data to
SPADL, and persists as Parquet files in `sb_cache/spadl_parquet/`.

Key design:
- Checks cache before downloading — never re-downloads a match.
- Uses a custom SPADL converter (xt_utils.py) that works with modern
  statsbombpy's flattened column format.
- On-disk format is Apache Parquet (columnar, compressed) — fast reads,
  low memory pressure, and M1 SSD friendly.
- tqdm progress bar over all matches with detailed per-match logging.

Usage:
    python scripts/01_cache_spadl.py

Resume-friendly: re-run any time; already-cached matches are skipped.
"""

import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add project root to path so we can import xt_utils
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xt_utils import (
    convert_events_to_spadl,
    ensure_cache_dir,
    cache_path,
    match_cached,
    CACHE_DIR,
)

# statsbombpy emits a noisy warning on open-data access; mute it
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from statsbombpy import sb


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Competitions to fetch. These are the major StatsBomb Open Data comps.
# We include all to maximise xT training data.
# The full list is dynamic — fetched at runtime via sb.competitions().
# We exclude youth/international duplicates via the flags below.
INCLUDE_YOUTH = True   # e.g. FIFA U20 World Cup
INCLUDE_WOMEN = True   # e.g. FA Women's Super League, Women's World Cup

# Request retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds between retries


# ---------------------------------------------------------------------------
# Fetch pipeline
# ---------------------------------------------------------------------------
def fetch_all_matches() -> pd.DataFrame:
    """Return a DataFrame of all available matches across all competitions."""
    print("Fetching competition list from StatsBomb Open Data ...")
    comps = sb.competitions()
    print(f"  Found {len(comps)} competition-seasons in total.")

    # Optionally filter
    if not INCLUDE_YOUTH:
        comps = comps[~comps["competition_youth"].fillna(False)]
    if not INCLUDE_WOMEN:
        comps = comps[comps["competition_gender"] == "male"]

    print(f"  → {len(comps)} after filtering (youth={INCLUDE_YOUTH}, women={INCLUDE_WOMEN}).")

    all_matches: list[pd.DataFrame] = []
    for _, row in comps.iterrows():
        cid, sid = int(row["competition_id"]), int(row["season_id"])
        cname = row["competition_name"]
        try:
            matches = sb.matches(competition_id=cid, season_id=sid)
            if len(matches) == 0:
                continue
            matches["_competition_name"] = cname
            matches["_season_name"] = row.get("season_name", "")
            all_matches.append(matches)
        except Exception:
            # Some comps may fail (API issues) — log and continue
            print(f"  ⚠ Could not fetch matches for {cname} (id={cid}, season={sid})")
            continue

    full = pd.concat(all_matches, ignore_index=True)
    print(f"  → {len(full)} total matches across all competition-seasons.")
    return full


def process_match(match_row: pd.Series) -> tuple[int, bool, str]:
    """
    Download + convert + cache a single match.

    Returns (match_id, success, message).
    """
    match_id = int(match_row["match_id"])
    comp_name = match_row.get("_competition_name", "?")
    home = match_row.get("home_team", "?")
    away = match_row.get("away_team", "?")
    label = f"[{comp_name}] {home} vs {away} ({match_id})"

    # --- cache hit ---
    if match_cached(match_id):
        return match_id, True, "cached"

    # --- download with retries ---
    events = None
    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            events = sb.events(match_id=match_id)
            break
        except Exception as exc:
            last_err = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return match_id, False, f"download-failed: {last_err[:120]}"

    if events is None or len(events) == 0:
        return match_id, False, "empty-events"

    # --- convert to SPADL ---
    try:
        actions = convert_events_to_spadl(events, match_id)
    except Exception as exc:
        return match_id, False, f"spadl-convert-failed: {exc!s}"

    if len(actions) == 0:
        return match_id, False, "no-spadl-actions"

    # --- persist ---
    try:
        ensure_cache_dir()
        actions.to_parquet(str(cache_path(match_id)), index=False, compression="zstd")
    except Exception as exc:
        return match_id, False, f"write-failed: {exc!s}"

    # Estimate storage savings
    events_mb = events.memory_usage(deep=True).sum() / (1024 * 1024)
    actions_mb = actions.memory_usage(deep=True).sum() / (1024 * 1024)
    parquet_mb = cache_path(match_id).stat().st_size / (1024 * 1024)

    return match_id, True, (
        f"{len(events)} events ({events_mb:.1f}MB) → "
        f"{len(actions)} actions → {parquet_mb:.2f}MB Parquet"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    start = time.perf_counter()

    matches = fetch_all_matches()
    match_ids = matches["match_id"].unique()
    n_total = len(match_ids)

    # Quick pre-scan for cache hits
    n_cached = sum(1 for mid in match_ids if match_cached(int(mid)))
    print(f"\nCache status: {n_cached}/{n_total} already cached, "
          f"{n_total - n_cached} to process.\n")

    # Process sequentially with tqdm
    succeeded, skipped, failed = 0, 0, 0
    fail_list: list[tuple[int, str]] = []
    total_actions = 0

    pbar = tqdm(
        matches.iterrows(),
        total=n_total,
        desc="Caching SPADL",
        unit="match",
        ncols=100,
    )

    for _, row in pbar:
        mid = int(row["match_id"])

        # Update description to show current match
        pbar.set_postfix_str(
            f"{row.get('_competition_name','?')[:20]} — "
            f"{row.get('home_team','?')} v {row.get('away_team','?')}"
        )

        match_id, ok, msg = process_match(row)

        if ok and msg == "cached":
            skipped += 1
        elif ok:
            succeeded += 1
            # Try to get action count from parquet metadata
            try:
                pf = pd.read_parquet(str(cache_path(match_id)), columns=["type_id"])
                total_actions += len(pf)
            except Exception:
                pass
        else:
            failed += 1
            fail_list.append((match_id, msg))

        # Update tqdm stats
        pbar.set_description(
            f"✓{succeeded} ↪{skipped} ✗{failed}  "
        )

    elapsed = time.perf_counter() - start

    # ---- summary ----
    print(f"\n{'='*60}")
    print(f"Pipeline complete.  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Succeeded (new): {succeeded}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Failed:           {failed}")
    print(f"  Total actions:    ~{total_actions:,}")
    print(f"  Cache directory:  {CACHE_DIR.resolve()}")

    # Disk usage
    total_mb = sum(
        f.stat().st_size for f in CACHE_DIR.glob("*.parquet")
    ) / (1024 * 1024)
    print(f"  Cache size:       {total_mb:.1f} MB ({len(list(CACHE_DIR.glob('*.parquet')))} files)")

    if fail_list:
        print(f"\n  Failed matches ({len(fail_list)}):")
        for mid, reason in fail_list[:20]:
            print(f"    {mid}: {reason}")
        if len(fail_list) > 20:
            print(f"    ... and {len(fail_list) - 20} more.")


if __name__ == "__main__":
    main()