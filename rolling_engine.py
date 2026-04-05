"""
PropEdge V18.0 — rolling_engine.py
Live rolling stat computation. Never reads pre-computed L*_* CSV columns.
All features computed strictly from prior played games only.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Optional

from config import (
    get_pos_group, get_dvp, season_progress,
    SEASON_START, SEASON_DAYS, MIN_PRIOR_GAMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def filter_played(df: pd.DataFrame) -> pd.DataFrame:
    """Remove DNP and zero-minute rows. MUST be called before any rolling window."""
    mask = (df["DNP"].fillna(0) == 0) & (df["MIN_NUM"].fillna(0) > 0)
    return df[mask].copy()


def _sm(arr: np.ndarray, n: int) -> float:
    a = arr[-n:]
    return float(np.mean(a)) if len(a) > 0 else 0.0


def _ss(arr: np.ndarray, n: int) -> float:
    a = arr[-n:]
    return float(np.std(a)) if len(a) > 1 else 0.0


def _parse_min(v) -> float:
    """Parse NBA API minute strings: PT36M14.00S | 36:14 | float."""
    import re
    s = str(v).strip()
    if s in ("", "None", "nan", "0", "PT00M00.00S"):
        return 0.0
    if s.startswith("PT") and "M" in s:
        m = re.match(r"PT(\d+)M([\d.]+)S", s)
        return float(m.group(1)) + float(m.group(2)) / 60 if m else 0.0
    if ":" in s:
        p = s.split(":")
        return float(p[0]) + float(p[1]) / 60
    try:
        return float(s)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE EXTRACTOR  (V14 — 56 features)
# ─────────────────────────────────────────────────────────────────────────────

def extract_prediction_features(
    prior_played: pd.DataFrame,
    line: float,
    opponent: str,
    rest_days: int,
    pos_raw: str,
    game_date,          # datetime or pd.Timestamp
    min_line: Optional[float] = None,
    max_line: Optional[float] = None,
    dvp_rank_cache: Optional[dict] = None,
    pace_rank_cache: Optional[dict] = None,
) -> Optional[dict]:
    """
    Compute all 56 V14 features from prior played game history.
    Returns None if insufficient history (< MIN_PRIOR_GAMES).
    NEVER reads pre-computed L*_* columns from CSV.
    """
    prior = filter_played(prior_played)
    n = len(prior)
    if n < MIN_PRIOR_GAMES:
        return None

    # ── Raw arrays ───────────────────────────────────────────────────────────
    pts   = prior["PTS"].fillna(0).values.astype(float)
    mins  = prior["MIN_NUM"].fillna(0).values.astype(float)
    fga   = prior["FGA"].fillna(0).values.astype(float)
    fg3a  = prior["FG3A"].fillna(0).values.astype(float)
    fg3m  = prior["FG3M"].fillna(0).values.astype(float)
    fta   = prior["FTA"].fillna(0).values.astype(float)
    usage = prior["USAGE_APPROX"].fillna(0).values.astype(float)
    is_home = prior["IS_HOME"].fillna(0).values.astype(float)

    # ── Rolling averages ─────────────────────────────────────────────────────
    L3  = _sm(pts, 3)
    L5  = _sm(pts, 5)
    L10 = _sm(pts, 10)
    L20 = _sm(pts, 20)
    L30 = _sm(pts, 30)

    std10 = _ss(pts, 10)
    std30 = _ss(pts, 30)
    eps = 1e-6

    # ── Orthogonal decomposition (V12+) ──────────────────────────────────────
    level        = L30
    reversion    = L10 - L30
    momentum     = L5  - L30
    acceleration = L3  - L5
    try:
        level_ewm = float(prior["PTS"].tail(10).ewm(span=5).mean().iloc[-1])
    except Exception:
        level_ewm = L10

    # ── Z-score momentum (V14 NEW) ───────────────────────────────────────────
    z_momentum   = momentum      / (std30 + eps)
    z_reversion  = reversion     / (std30 + eps)
    z_accel      = acceleration  / (std10 + eps)

    # ── Mean reversion risk (V14 NEW) ────────────────────────────────────────
    abs_mom = abs(momentum)
    if abs_mom > 6:
        mean_reversion_risk = 1.0
    elif abs_mom > 3:
        mean_reversion_risk = 0.5
    else:
        mean_reversion_risk = 0.0
    extreme_hot  = 1.0 if momentum  >  6 else 0.0
    extreme_cold = 1.0 if momentum  < -6 else 0.0

    # ── Season context (V14 NEW) ─────────────────────────────────────────────
    if hasattr(game_date, "strftime"):
        date_str = game_date.strftime("%Y-%m-%d")
    else:
        date_str = str(game_date)[:10]
    sp = season_progress(date_str)
    early_season_w = 1.0 - math.exp(-max(0, n - 5) / 15.0)
    games_depth    = min(float(n) / 30.0, 1.0)

    # ── Volume / trend / consistency ─────────────────────────────────────────
    volume      = L30 - line
    trend       = L5  - L30
    consistency = 1.0 / (std10 + 1.0)
    hr10 = float(np.mean(pts[-10:] > line)) if n >= 10 else float(np.mean(pts > line))
    hr30 = float(np.mean(pts[-30:] > line)) if n >= 30 else float(np.mean(pts > line))

    # ── Minutes ──────────────────────────────────────────────────────────────
    min_l10 = _sm(mins, 10)
    min_l30 = _sm(mins, 30)
    min_cv  = _ss(mins, 10) / (min_l10 + eps)
    recent_min_trend = _sm(mins, 3) - min_l10
    pts_per_min      = L10 / (min_l10 + eps)

    # ── Shot volume ──────────────────────────────────────────────────────────
    fga_l10  = _sm(fga,  10)
    fg3a_l10 = _sm(fg3a, 10)
    fg3m_l10 = _sm(fg3m, 10)
    fta_l10  = _sm(fta,  10)
    ft_rate     = fta_l10 / (fga_l10 + eps)
    fga_per_min = fga_l10 / (min_l10 + eps)
    ppfga_l10   = L10     / (fga_l10 + eps)

    # ── Usage / role ─────────────────────────────────────────────────────────
    usage_l10      = _sm(usage, 10)
    usage_l30      = _sm(usage, 30)
    role_intensity = usage_l10 * min_l10 / 100.0

    # ── Home / away split ────────────────────────────────────────────────────
    h_mask = is_home[-n:] == 1
    home_pts = pts[-n:][h_mask][-10:]
    away_pts = pts[-n:][~h_mask][-10:]
    home_l10 = float(np.mean(home_pts)) if len(home_pts) > 0 else L10
    away_l10 = float(np.mean(away_pts)) if len(away_pts) > 0 else L10
    home_away_split = home_l10 - away_l10

    # ── Rest ─────────────────────────────────────────────────────────────────
    is_b2b = 1.0 if rest_days == 1 else 0.0
    rd_clip = min(rest_days, 10)
    if rest_days <= 1:    rc = 0
    elif rest_days <= 2:  rc = 1
    elif rest_days <= 4:  rc = 2
    elif rest_days <= 6:  rc = 3
    else:                  rc = 4
    is_long_rest = 1.0 if rest_days >= 6 else 0.0

    # ── Line features ────────────────────────────────────────────────────────
    line_vs_l30 = line - L30
    line_bucket = float(int(min(max(line // 5, 0), 5)))

    # ── Line sharpness (V14 NEW) ─────────────────────────────────────────────
    if min_line is not None and max_line is not None and max_line > min_line:
        line_spread    = float(max_line - min_line)
        line_sharpness = 1.0 / (line_spread + 1.0)
    else:
        line_spread    = 0.5
        line_sharpness = 0.67

    # ── Volatility risk (V14 NEW) ────────────────────────────────────────────
    vol_risk = std10 * line / 100.0

    # ── DVP + pace (from cache or live lookup) ────────────────────────────────
    pos = get_pos_group(pos_raw)
    if dvp_rank_cache:
        defP_dynamic = float(dvp_rank_cache.get((opponent, pos), 15))
    else:
        defP_dynamic = float(get_dvp(opponent, pos_raw))

    if pace_rank_cache:
        pace_rank = float(pace_rank_cache.get(opponent, 15))
    else:
        pace_rank = 15.0

    return {
        # Orthogonal rolling
        "level": level, "reversion": reversion, "momentum": momentum,
        "acceleration": acceleration, "level_ewm": level_ewm,
        # Z-score (V14)
        "z_momentum": z_momentum, "z_reversion": z_reversion, "z_accel": z_accel,
        # Mean reversion (V14)
        "mean_reversion_risk": mean_reversion_risk,
        "extreme_hot": extreme_hot, "extreme_cold": extreme_cold,
        # Season context (V14)
        "season_progress": sp, "early_season_weight": early_season_w,
        "games_depth": games_depth,
        # Standard
        "volume": volume, "trend": trend, "std10": std10, "consistency": consistency,
        "hr10": hr10, "hr30": hr30,
        # Minutes
        "min_l10": min_l10, "min_l30": min_l30, "min_cv": min_cv,
        "recent_min_trend": recent_min_trend, "pts_per_min": pts_per_min,
        # Shot volume
        "fga_l10": fga_l10, "fg3a_l10": fg3a_l10, "fg3m_l10": fg3m_l10,
        "fta_l10": fta_l10, "ft_rate": ft_rate, "fga_per_min": fga_per_min,
        "ppfga_l10": ppfga_l10,
        # Usage
        "usage_l10": usage_l10, "usage_l30": usage_l30, "role_intensity": role_intensity,
        # Home/away
        "home_l10": home_l10, "away_l10": away_l10, "home_away_split": home_away_split,
        # Rest
        "is_b2b": is_b2b, "rest_days": float(rd_clip), "rest_cat": float(rc),
        "is_long_rest": is_long_rest,
        # Line
        "line": line, "line_vs_l30": line_vs_l30, "line_bucket": line_bucket,
        # Line quality (V14)
        "line_spread": line_spread, "line_sharpness": line_sharpness,
        # Volatility (V14)
        "vol_risk": vol_risk,
        # Opponent
        "defP_dynamic": defP_dynamic, "pace_rank": pace_rank,
        # Display (not in ML feature vector — stored separately)
        "_l3": L3, "_l5": L5, "_l10": L10, "_l20": L20, "_l30": L30,
        "_std10": std10, "_min_l10": min_l10,
        "_n_games": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING APPEND (for Batch 0 after new game rows are written)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_for_new_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling L3/L5/L10/L30 for newly appended game rows.
    Uses explicit loops — never groupby().apply().
    Called only by batch0_grade.py after append_gamelogs().
    """
    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)
    roll_cols = ["L3_PTS", "L5_PTS", "L10_PTS", "L30_PTS", "STD10_PTS",
                 "L10_MIN", "L10_FGA", "L10_USAGE"]
    for c in roll_cols:
        if c not in df.columns:
            df[c] = float("nan")

    played_mask = (df["DNP"].fillna(0) == 0) & (df["MIN_NUM"].fillna(0) > 0)

    for player, grp in df.groupby("PLAYER_NAME"):
        played_idx = grp.index[played_mask.loc[grp.index]]
        pts_hist   = []
        min_hist   = []
        fga_hist   = []
        usg_hist   = []

        for idx in grp.index:
            row = grp.loc[idx]
            if idx in played_idx:
                # Compute rolling BEFORE appending this game
                df.at[idx, "L3_PTS"]  = float(np.mean(pts_hist[-3:]))  if pts_hist else float("nan")
                df.at[idx, "L5_PTS"]  = float(np.mean(pts_hist[-5:]))  if pts_hist else float("nan")
                df.at[idx, "L10_PTS"] = float(np.mean(pts_hist[-10:])) if pts_hist else float("nan")
                df.at[idx, "L30_PTS"] = float(np.mean(pts_hist[-30:])) if pts_hist else float("nan")
                df.at[idx, "STD10_PTS"] = (
                    float(np.std(pts_hist[-10:])) if len(pts_hist) >= 2 else float("nan")
                )
                df.at[idx, "L10_MIN"]   = float(np.mean(min_hist[-10:])) if min_hist else float("nan")
                df.at[idx, "L10_FGA"]   = float(np.mean(fga_hist[-10:])) if fga_hist else float("nan")
                df.at[idx, "L10_USAGE"] = float(np.mean(usg_hist[-10:])) if usg_hist else float("nan")

                pts_hist.append(float(row.get("PTS", 0) or 0))
                min_hist.append(float(row.get("MIN_NUM", 0) or 0))
                fga_hist.append(float(row.get("FGA", 0) or 0))
                usg_hist.append(float(row.get("USAGE_APPROX", 0) or 0))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE A — RULE-BASED COMPOSITE (10 signals)
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_NAMES = [
    "Volume", "HR_L30", "HR_L10", "Trend", "Context",
    "Defense", "H2H", "Pace", "FG_Trend", "Min_Trend",
]


def compute_composite(
    feats: dict,
    line: float,
    direction: str,
    pos_group: str,
    h2h_avg: float = 0.0,
    use_h2h: bool = False,
) -> tuple[float, int, list[dict]]:
    """
    Compute Engine A rule-based composite score.
    Returns (composite_score [-1,+1], flag_count [0-10], flag_details).
    """
    from config import POS_WEIGHTS
    weights = POS_WEIGHTS.get(pos_group, POS_WEIGHTS["Guard"])

    L30  = feats.get("_l30",  feats.get("level", line))
    L10  = feats.get("_l10",  L30)
    L5   = feats.get("_l5",   L10)
    L3   = feats.get("_l3",   L5)
    hr30 = feats.get("hr30", 0.5)
    hr10 = feats.get("hr10", 0.5)
    dvp  = feats.get("defP_dynamic", 15)
    pace = feats.get("pace_rank", 15)
    fga_l10 = feats.get("fga_l10", 10)
    fga_l30 = feats.get("fga_l10", 10)   # proxy
    min_l3  = feats.get("recent_min_trend", 0) + feats.get("min_l10", 30)
    min_l10 = feats.get("min_l10", 30)
    std10   = feats.get("std10", 5)
    books   = feats.get("_books", 5)

    # Signal values (positive = bullish / supports OVER)
    s1 = 1.0 if L30 > line else (-1.0 if L30 < line else 0.0)
    s2 = 1.0 if hr30 > 0.55 else (-1.0 if hr30 < 0.45 else 0.0)
    s3 = 1.0 if hr10 > 0.55 else (-1.0 if hr10 < 0.45 else 0.0)
    s4 = 1.0 if L5 > L30 + 1 else (-1.0 if L5 < L30 - 1 else 0.0)
    s5 = 1.0 if fga_l10 > fga_l30 else -1.0
    s6 = 1.0 if dvp >= 22 else (-1.0 if dvp <= 8 else 0.0)   # weak def = OVER signal
    s7 = (1.0 if h2h_avg > line + 1 else (-1.0 if h2h_avg < line - 1 else 0.0)) if use_h2h else 0.0
    s8 = 1.0 if pace >= 22 else (-1.0 if pace <= 8 else 0.0)
    s9_fga  = fga_l10 / max(fga_l30, 1)
    s9 = 1.0 if s9_fga > 1.05 else (-1.0 if s9_fga < 0.95 else 0.0)
    s10 = 1.0 if min_l3 > min_l10 + 1 else (-1.0 if min_l3 < min_l10 - 1 else 0.0)

    signals = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]

    # Exclude H2H from denominator if not used
    total_w = sum(weights[i] for i in range(10) if not (i == 6 and not use_h2h))

    composite = sum(signals[i] * weights[i] for i in range(10)) / max(total_w, 1.0)
    composite = max(-1.0, min(1.0, composite))

    # Determine direction for flag counting
    is_over = "OVER" in direction.upper()
    flag_count = sum(
        1 for i, s in enumerate(signals)
        if (is_over and s > 0) or (not is_over and s < 0)
    )

    flag_details = [
        {"name": _SIGNAL_NAMES[i], "value": signals[i],
         "weight": weights[i], "agrees": (is_over and signals[i] > 0) or (not is_over and signals[i] < 0)}
        for i in range(10)
    ]

    return composite, flag_count, flag_details
