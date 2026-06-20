"""
Shared utilities for the Expected Threat (xT) pipeline.

Contains a custom StatsBomb→SPADL converter that works with modern statsbombpy
(flattened column format: pass_end_location, shot_outcome, etc.) and produces
SPADL-format DataFrames compatible with socceraction's xT model.

Why a custom converter? socceraction v1.5.3 expects the legacy statsbombpy
format with a nested `extra` dict column. Modern statsbombpy (v1.x) flattens
those dicts into prefixed columns. Rather than reconstruct the legacy format,
this converter maps modern columns directly to SPADL.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# SPADL action types and result types (must match socceraction.spadl.config)
# ---------------------------------------------------------------------------
# fmt: off
ACTION_TYPES = [
    "pass",              # 0
    "cross",             # 1
    "throw_in",          # 2
    "freekick_crossed",  # 3
    "freekick_short",    # 4
    "corner_crossed",    # 5
    "corner_short",      # 6
    "take_on",           # 7
    "foul",              # 8
    "tackle",            # 9
    "interception",      # 10
    "shot",              # 11
    "shot_penalty",      # 12
    "shot_freekick",     # 13
    "keeper_save",       # 14
    "keeper_claim",      # 15
    "keeper_punch",      # 16
    "keeper_pick_up",    # 17
    "clearance",         # 18
    "bad_touch",         # 19
    "non_action",        # 20
    "dribble",           # 21
    "goalkick",          # 22
]
RESULT_TYPES = ["fail", "success", "offside", "owngoal"]

# Quick lookup dicts
TYPE2ID = {name: i for i, name in enumerate(ACTION_TYPES)}
RESULT2ID = {name: i for i, name in enumerate(RESULT_TYPES)}

# Pitch dimensions
SB_X, SB_Y = 120.0, 80.0        # StatsBomb raw (120×80 grid)
SPADL_X, SPADL_Y = 105.0, 68.0  # SPADL normalized (actual pitch)
SCALE_X = SPADL_X / SB_X
SCALE_Y = SPADL_Y / SB_Y
# ---------------------------------------------------------------------------
# Body-part mapping
# ---------------------------------------------------------------------------
BODYPART_MAP = {
    "Right Foot": 0, "Left Foot": 0,  # "foot"
    "Head": 1,                         # "head"
    "Other": 2, "Keeper Arm": 2,
    "Both Hands": 2, "Left Hand": 2, "Right Hand": 2,
}
BODYPART_DEFAULT = 0  # foot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_loc(arr, idx, default=np.nan):
    """Extract the idx-th element from each location list, or default."""
    if arr is None or len(arr) == 0:
        return default
    if isinstance(arr, (list, tuple)):
        return arr[idx] if len(arr) > idx else default
    return default


def _parse_location_column(col):
    """
    Parse a pandas column of [x, y] or [x, y, z] lists into separate x, y Series.
    Returns (x_series, y_series) with SPADL-normalized coordinates.
    """
    # Extract first two elements safely
    x_vals = col.apply(lambda v: float(v[0]) if isinstance(v, (list, tuple)) and len(v) >= 1 else np.nan)
    y_vals = col.apply(lambda v: float(v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else np.nan)
    # Normalize to SPADL pitch
    return x_vals * SCALE_X, y_vals * SCALE_Y


# ---------------------------------------------------------------------------
# Primary conversion entry point
# ---------------------------------------------------------------------------
def convert_events_to_spadl(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """
    Convert a modern-statsbombpy events DataFrame into SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        Raw events from statsbombpy (flattened format). Must be from a single
        match and include all standard columns (type, location, pass_*, shot_*,
        carry_*, dribble_*, etc.).
    match_id : int
        StatsBomb match ID, used as the SPADL game_id.

    Returns
    -------
    pd.DataFrame
        SPADL-format actions with columns: game_id, period_id, time_seconds,
        team_id, player_id, type_id, result_id, bodypart_id,
        start_x, start_y, end_x, end_y.
    """
    df = events.copy()

    # ---- basic identifiers ------------------------------------------------
    df["game_id"] = match_id
    df["period_id"] = df["period"]
    df["time_seconds"] = pd.to_timedelta(df["timestamp"]).dt.total_seconds()
    df["team_id"] = df["team_id"]
    df["player_id"] = df["player_id"]

    # ---- start locations (normalized) -------------------------------------
    start_x, start_y = _parse_location_column(df["location"])
    df["start_x"] = start_x
    df["start_y"] = start_y

    # ---- end locations (depends on event type) ----------------------------
    df["end_x"] = df["start_x"]  # default: ball stays (non-actions, etc.)
    df["end_y"] = df["start_y"]

    # Passes
    mask_pass = df["type"] == "Pass"
    if mask_pass.any():
        px, py = _parse_location_column(df.loc[mask_pass, "pass_end_location"])
        df.loc[mask_pass, "end_x"] = px
        df.loc[mask_pass, "end_y"] = py

    # Shots
    mask_shot = df["type"] == "Shot"
    if mask_shot.any():
        sx, sy = _parse_location_column(df.loc[mask_shot, "shot_end_location"])
        df.loc[mask_shot, "end_x"] = sx
        df.loc[mask_shot, "end_y"] = sy

    # Carries
    mask_carry = df["type"] == "Carry"
    if mask_carry.any():
        cx, cy = _parse_location_column(df.loc[mask_carry, "carry_end_location"])
        df.loc[mask_carry, "end_x"] = cx
        df.loc[mask_carry, "end_y"] = cy

    # Goalkeeper
    mask_gk = df["type"] == "Goal Keeper"
    if mask_gk.any() and "goalkeeper_end_location" in df.columns:
        gx, gy = _parse_location_column(df.loc[mask_gk, "goalkeeper_end_location"])
        df.loc[mask_gk, "end_x"] = gx.fillna(df.loc[mask_gk, "end_x"])
        df.loc[mask_gk, "end_y"] = gy.fillna(df.loc[mask_gk, "end_y"])

    # ---- type_id -----------------------------------------------------------
    _assign_type_ids(df)

    # ---- result_id ---------------------------------------------------------
    _assign_result_ids(df)

    # ---- bodypart_id -------------------------------------------------------
    _assign_bodypart_ids(df)

    # ---- assemble output ---------------------------------------------------
    columns = [
        "game_id", "period_id", "time_seconds", "team_id", "player_id",
        "type_id", "result_id", "bodypart_id",
        "start_x", "start_y", "end_x", "end_y",
    ]
    result = df[columns].copy()

    # Drop rows with NaN in critical fields
    result = result.dropna(subset=["start_x", "start_y"])
    result = result.reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Internal: assign type_id
# ---------------------------------------------------------------------------
def _assign_type_ids(df: pd.DataFrame) -> None:
    """Set SPADL type_id in-place based on event type and sub-type columns."""
    # Default: non_action (20)
    df["type_id"] = TYPE2ID["non_action"]

    # Passes — differentiated by pass_type and pass_cross
    mask_pass = df["type"] == "Pass"
    if mask_pass.any():
        # Start all passes as type 0 (pass)
        df.loc[mask_pass, "type_id"] = TYPE2ID["pass"]
        pt = df.loc[mask_pass, "pass_type"]

        # Throw-in
        is_throw = pt == "Throw-in"
        df.loc[mask_pass & is_throw, "type_id"] = TYPE2ID["throw_in"]

        # Goal kick
        is_gk = pt == "Goal Kick"
        df.loc[mask_pass & is_gk, "type_id"] = TYPE2ID["goalkick"]

        # Free kick (crossed vs short)
        is_fk = pt == "Free Kick"
        fk_cross = df.loc[mask_pass, "pass_cross"].fillna(False).infer_objects(copy=False).astype(bool)
        df.loc[mask_pass & is_fk & fk_cross, "type_id"] = TYPE2ID["freekick_crossed"]
        df.loc[mask_pass & is_fk & ~fk_cross, "type_id"] = TYPE2ID["freekick_short"]

        # Corner (crossed vs short)
        is_corner = pt == "Corner"
        df.loc[mask_pass & is_corner & fk_cross, "type_id"] = TYPE2ID["corner_crossed"]
        df.loc[mask_pass & is_corner & ~fk_cross, "type_id"] = TYPE2ID["corner_short"]

        # Open-play cross (pass_cross=True, not set-piece)
        is_open_play_cross = (
            fk_cross & (pt.isna() | ~pt.isin([
                "Throw-in", "Goal Kick", "Free Kick", "Corner",
            ]))
        )
        df.loc[mask_pass & is_open_play_cross, "type_id"] = TYPE2ID["cross"]

    # Shots
    mask_shot = df["type"] == "Shot"
    if mask_shot.any():
        df.loc[mask_shot, "type_id"] = TYPE2ID["shot"]
        st = df.loc[mask_shot, "shot_type"]
        df.loc[mask_shot & (st == "Penalty"), "type_id"] = TYPE2ID["shot_penalty"]
        df.loc[mask_shot & (st == "Free Kick"), "type_id"] = TYPE2ID["shot_freekick"]

    # Carry → dribble (21)
    mask_carry = df["type"] == "Carry"
    df.loc[mask_carry, "type_id"] = TYPE2ID["dribble"]

    # Dribble → take_on (7)
    mask_dribble = df["type"] == "Dribble"
    df.loc[mask_dribble, "type_id"] = TYPE2ID["take_on"]

    # Foul Committed → foul (8)
    df.loc[df["type"] == "Foul Committed", "type_id"] = TYPE2ID["foul"]

    # Interception (10)
    df.loc[df["type"] == "Interception", "type_id"] = TYPE2ID["interception"]

    # Duel → tackle (9) if duel_type is Tackle, else non_action
    mask_duel = df["type"] == "Duel"
    if mask_duel.any():
        df.loc[mask_duel, "type_id"] = TYPE2ID["non_action"]
        is_tackle = df.loc[mask_duel, "duel_type"] == "Tackle"
        df.loc[mask_duel & is_tackle, "type_id"] = TYPE2ID["tackle"]

    # Clearance (18)
    df.loc[df["type"] == "Clearance", "type_id"] = TYPE2ID["clearance"]

    # Miscontrol → bad_touch (19)
    df.loc[df["type"] == "Miscontrol", "type_id"] = TYPE2ID["bad_touch"]

    # Goal Keeper — differentiated by goalkeeper_type
    mask_gk = df["type"] == "Goal Keeper"
    if mask_gk.any():
        gk_type = df.loc[mask_gk, "goalkeeper_type"]
        df.loc[mask_gk, "type_id"] = TYPE2ID["non_action"]
        df.loc[mask_gk & gk_type.isin(["Shot Saved", "Shot Faced"]), "type_id"] = TYPE2ID["keeper_save"]
        df.loc[mask_gk & (gk_type == "Collected"), "type_id"] = TYPE2ID["keeper_claim"]
        df.loc[mask_gk & (gk_type == "Punch"), "type_id"] = TYPE2ID["keeper_punch"]
        df.loc[mask_gk & gk_type.isin(["Keeper Sweeper", "Smother"]), "type_id"] = TYPE2ID["keeper_pick_up"]
        df.loc[mask_gk & (gk_type == "Goal Conceded"), "type_id"] = TYPE2ID["keeper_save"]

    # Own Goal Against → shot (treated as shot by opponent)
    df.loc[df["type"] == "Own Goal Against", "type_id"] = TYPE2ID["shot"]

    # Everything else stays as non_action (20)


# ---------------------------------------------------------------------------
# Internal: assign result_id
# ---------------------------------------------------------------------------
def _assign_result_ids(df: pd.DataFrame) -> None:
    """Set SPADL result_id in-place.  All outcome checks are
    case-insensitive and handle NaN gracefully."""
    df["result_id"] = np.nan
    tid = df["type_id"]

    # Passes, crosses, throw-ins, freekicks, corners, goalkicks
    is_pass_like = tid.isin([TYPE2ID[n] for n in [
        "pass", "cross", "throw_in", "freekick_crossed", "freekick_short",
        "corner_crossed", "corner_short", "goalkick",
    ]])
    if is_pass_like.any():
        outcome = df.loc[is_pass_like, "pass_outcome"]
        # Fail: any non-null outcome means the pass didn't succeed
        df.loc[is_pass_like, "result_id"] = outcome.notna().map(
            {True: RESULT2ID["fail"], False: RESULT2ID["success"]}
        )

    # Shots (including penalties and free kicks)
    is_shot = tid.isin([TYPE2ID[n] for n in ["shot", "shot_penalty", "shot_freekick"]])
    if is_shot.any():
        shot_outcome = df.loc[is_shot, "shot_outcome"]
        # Default: fail (0)
        df.loc[is_shot, "result_id"] = RESULT2ID["fail"]
        df.loc[is_shot & (shot_outcome == "Goal"), "result_id"] = RESULT2ID["success"]

    # Own goal
    is_own_goal = df["type"] == "Own Goal Against"
    if is_own_goal.any():
        df.loc[is_own_goal, "result_id"] = RESULT2ID["owngoal"]

    # Carries and dribbles — always success if they exist
    df.loc[tid == TYPE2ID["dribble"], "result_id"] = RESULT2ID["success"]  # carries
    df.loc[tid == TYPE2ID["take_on"], "result_id"] = RESULT2ID["fail"]      # default fail
    mask_dribble = df["type"] == "Dribble"
    if mask_dribble.any():
        df.loc[mask_dribble & (df["dribble_outcome"] == "Complete"), "result_id"] = RESULT2ID["success"]

    # Fouls
    df.loc[tid == TYPE2ID["foul"], "result_id"] = RESULT2ID["fail"]

    # Tackles (from Duels)
    mask_tackle = tid == TYPE2ID["tackle"]
    if mask_tackle.any():
        df.loc[mask_tackle, "result_id"] = RESULT2ID["fail"]
        duel_out = df.loc[mask_tackle, "duel_outcome"]
        # "Won" or "Success In Play" → success
        won = duel_out.str.lower().str.contains("won|success", na=False)
        df.loc[mask_tackle & won, "result_id"] = RESULT2ID["success"]

    # Interceptions
    mask_interc = tid == TYPE2ID["interception"]
    if mask_interc.any():
        df.loc[mask_interc, "result_id"] = RESULT2ID["fail"]
        int_out = df.loc[mask_interc, "interception_outcome"]
        won = int_out.str.lower().str.contains("won|success", na=False)
        df.loc[mask_interc & won, "result_id"] = RESULT2ID["success"]

    # Keeper actions
    keeper_tids = [TYPE2ID[n] for n in ["keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up"]]
    is_keeper = tid.isin(keeper_tids)
    if is_keeper.any():
        df.loc[is_keeper, "result_id"] = RESULT2ID["fail"]
        gk_out = df.loc[is_keeper, "goalkeeper_outcome"]
        # Success-ish outcomes
        ok = gk_out.str.lower().str.contains(
            "success|safe|collected|touched out", na=False,
        )
        df.loc[is_keeper & ok, "result_id"] = RESULT2ID["success"]
        # Punch that stays in play is still "success" for SPADL
        df.loc[is_keeper & gk_out.isin(["In Play Safe", "In Play Danger"]), "result_id"] = RESULT2ID["success"]

    # Clearance — always success
    df.loc[tid == TYPE2ID["clearance"], "result_id"] = RESULT2ID["success"]

    # Bad touch — always fail
    df.loc[tid == TYPE2ID["bad_touch"], "result_id"] = RESULT2ID["fail"]

    # Non-actions — fail
    df.loc[tid == TYPE2ID["non_action"], "result_id"] = RESULT2ID["fail"]

    # Fill any remaining NaN
    df["result_id"] = df["result_id"].fillna(RESULT2ID["fail"]).astype(int)


# ---------------------------------------------------------------------------
# Internal: assign bodypart_id
# ---------------------------------------------------------------------------
def _assign_bodypart_ids(df: pd.DataFrame) -> None:
    """Set SPADL bodypart_id in-place (0=foot, 1=head, 2=other)."""
    df["bodypart_id"] = BODYPART_DEFAULT  # foot

    # Pass body part
    if "pass_body_part" in df.columns:
        mask = df["pass_body_part"].notna()
        df.loc[mask, "bodypart_id"] = df.loc[mask, "pass_body_part"].map(
            BODYPART_MAP
        ).fillna(BODYPART_DEFAULT).astype(int)

    # Shot body part (overrides)
    if "shot_body_part" in df.columns:
        mask = df["shot_body_part"].notna()
        df.loc[mask, "bodypart_id"] = df.loc[mask, "shot_body_part"].map(
            BODYPART_MAP
        ).fillna(BODYPART_DEFAULT).astype(int)

    # Goalkeeper body part (overrides)
    if "goalkeeper_body_part" in df.columns:
        mask = df["goalkeeper_body_part"].notna()
        df.loc[mask, "bodypart_id"] = df.loc[mask, "goalkeeper_body_part"].map(
            BODYPART_MAP
        ).fillna(BODYPART_DEFAULT).astype(int)


# ---------------------------------------------------------------------------
# Parquet cache path helper
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).resolve().parent.parent / "sb_cache" / "spadl_parquet"


def cache_path(match_id: int) -> Path:
    """Return the Parquet file path for a given match ID."""
    return CACHE_DIR / f"{match_id}.parquet"


def match_cached(match_id: int) -> bool:
    """Check if a match has already been cached."""
    return cache_path(match_id).exists()


def ensure_cache_dir() -> None:
    """Create the cache directory tree if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)