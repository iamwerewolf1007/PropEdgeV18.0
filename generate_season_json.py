#!/usr/bin/env python3
"""
PropEdge V18.0 — Generate Season JSONs
========================================
One-time run (via `python3 run.py generate`) that builds the full season
JSON files from scratch using the V14 Adaptive Fusion prediction engine.

Steps:
  1. Load game logs (2024-25 + 2025-26) and H2H database
  2. Train V14 models (OOF 5-fold, no leakage) — or load existing
  3. Generate 2024-25 synthetic prop lines (synthetic_lines.py)
  4. Load 2025-26 real prop lines (Excel)
  5. Run V14 engine over BOTH seasons → score + grade every play
  6. Write:
       data/season_2024_25.json    — 2024-25 backtest (LOCKED after this)
       data/season_2025_26.json    — 2025-26 full season
       data/today.json             — most recent date's plays
       data/backtest_summary.json  — aggregate stats for dashboard header

Non-negotiable rules (inherited from V12):
  1. Never groupby().apply() for rolling — silent wrong results on large datasets
  2. Always parse_dates=['GAME_DATE'] in every read_csv
  3. Never read L*_* CSV columns at predict time — they go stale
  4. Always filter_played() before rolling — DNP zeros contaminate averages
  5. Rolling windows span BOTH seasons (no season reset)
  6. H2H lookup dicts use .to_dict() per row — prevents Series truth crash
  7. Graded plays are permanently immutable — never overwrite result/actualPts
  8. clean_json() before every json.dump()
  9. Deduplicate H2H by (PLAYER_NAME, OPPONENT) keep='last'
 10. pace_rank computed from real team FGA — never a constant
"""

import json
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    VERSION, DATA_DIR, MODEL_DIR,
    FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_PROPS,
    FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST,
    FILE_TODAY, FILE_SEASON_2526, FILE_SEASON_2425,
    season_progress, get_pos_group, fusion_weights,
    LEAN_ZONE, UNIT_MAP, HIGH_LINE_THRESHOLD, TRUST_THRESHOLD,
    clean_json,
)
from rolling_engine import filter_played, extract_prediction_features
from reasoning_engine import generate_pre_match_reason, generate_post_match_reason
from synthetic_lines import generate_season_lines
from audit import log_event

# ─── Feature list (must match model_trainer.ML_FEATURES exactly) ─────────────
ML_FEATURES = [
    "level", "reversion", "momentum", "acceleration", "level_ewm",
    "z_momentum", "z_reversion", "z_accel",
    "mean_reversion_risk", "extreme_hot", "extreme_cold",
    "season_progress", "early_season_weight", "games_depth",
    "volume", "trend", "std10", "consistency", "hr10", "hr30",
    "min_l10", "min_l30", "min_cv", "recent_min_trend", "pts_per_min",
    "fga_l10", "fg3a_l10", "fg3m_l10", "fta_l10", "ft_rate", "fga_per_min", "ppfga_l10",
    "usage_l10", "usage_l30", "role_intensity",
    "home_l10", "away_l10", "home_away_split",
    "is_b2b", "rest_days", "rest_cat", "is_long_rest",
    "defP_dynamic", "pace_rank",
    "h2h_ts_dev", "h2h_fga_dev", "h2h_min_dev", "h2h_conf", "h2h_games", "h2h_trend",
    "line", "line_vs_l30", "line_bucket",
    "line_spread", "line_sharpness", "vol_risk",
]


# =============================================================================
# HELPERS
# =============================================================================

def _s(v):
    """Safe scalar for JSON serialisation."""
    if v is None:
        return None
    try:
        import numpy as _np
        if isinstance(v, float) and v != v:
            return None
        if isinstance(v, _np.integer):
            return int(v)
        if isinstance(v, _np.floating):
            return None if _np.isnan(v) else round(float(v), 4)
        if isinstance(v, _np.bool_):
            return bool(v)
    except ImportError:
        pass
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return v


def _save(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(clean_json(data), f, separators=(",", ":"))


# =============================================================================
# STEP 1: LOAD ALL DATA
# =============================================================================

def load_all_data():
    print("\n[1/5] Loading game logs...")
    gl24 = pd.read_csv(FILE_GL_2425, parse_dates=["GAME_DATE"])
    gl25 = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    gl   = pd.concat([gl24, gl25], ignore_index=True).sort_values(["PLAYER_NAME", "GAME_DATE"])
    gl["DNP"] = gl["DNP"].fillna(0)
    played = filter_played(gl)
    print(f"  Played rows: {len(played):,}  |  Players: {played['PLAYER_NAME'].nunique():,}")

    # Player history index (keyed by name → sorted DataFrame of played rows)
    player_idx = {
        pname: grp.sort_values("GAME_DATE").reset_index(drop=True)
        for pname, grp in played.groupby("PLAYER_NAME")
    }

    # H2H lookup — rule 9: deduplicate keep='last'
    h2h_df  = pd.read_csv(FILE_H2H).drop_duplicates(subset=["PLAYER_NAME", "OPPONENT"], keep="last")
    h2h_lkp = {
        (r["PLAYER_NAME"], r["OPPONENT"]): r.to_dict()
        for _, r in h2h_df.iterrows()
    }

    # Live DVP rank (rule 10: from real FGA data)
    dvp_dict: dict = {}
    for (opp, pos), g in played.groupby(["OPPONENT", "PLAYER_POSITION"]):
        dvp_dict[(opp, pos)] = g["PTS"].mean()
    dvp_rank: dict = {}
    for pos in played["PLAYER_POSITION"].unique():
        subset = {k: v for k, v in dvp_dict.items() if k[1] == pos}
        for rank, (opp, p) in enumerate(
            sorted(subset, key=lambda k: subset[k], reverse=True), 1
        ):
            dvp_rank[(opp, p)] = rank

    # Pace rank (rule 10)
    team_fga   = played.groupby("OPPONENT")["FGA"].mean()
    pace_cache = {t: i+1 for i, (t, _) in
                  enumerate(team_fga.sort_values(ascending=False).items())}

    # B2B rest-days map
    b2b_map: dict = {}
    for pname, grp in played.groupby("PLAYER_NAME"):
        dates = grp["GAME_DATE"].values
        for i, d in enumerate(dates):
            rd = int((d - dates[i-1]).astype("timedelta64[D]").astype(int)) if i > 0 else 99
            b2b_map[(pname, pd.Timestamp(d).strftime("%Y-%m-%d"))] = rd

    # Recent 20 scores per player (sparkline / recent20 array)
    recent_idx: dict = {}
    for pname, grp in played.groupby("PLAYER_NAME"):
        g = grp.sort_values("GAME_DATE")
        recent_idx[pname] = list(zip(
            g["GAME_DATE"].dt.strftime("%Y-%m-%d").tolist(),
            g["PTS"].fillna(0).tolist(),
            g["IS_HOME"].fillna(0).astype(int).tolist(),
            g["OPPONENT"].fillna("").tolist(),
        ))

    # ── 2025-26: real prop lines from Excel ───────────────────────────────────
    print("  Loading 2025-26 prop lines from Excel...")
    props_2526 = _load_excel_props()
    print(f"    {len(props_2526):,} real prop lines")

    # ── 2024-25: synthetic prop lines (via synthetic_lines.py) ────────────────
    print("  Generating synthetic prop lines for 2024-25...")
    played_2425 = played[played["GAME_DATE"] < pd.Timestamp("2025-10-01")].copy()
    props_2425  = generate_season_lines(played_2425, season="2024-25")
    print(f"    {len(props_2425):,} synthetic prop lines")

    return player_idx, h2h_lkp, dvp_rank, pace_cache, b2b_map, recent_idx, props_2526, props_2425


def _load_excel_props() -> list[dict]:
    """Load real 2025-26 prop lines from the Excel source file."""
    if not FILE_PROPS.exists():
        print("  *** Excel props file not found — 2025-26 will use synthetic lines ***")
        return []
    xl = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
    xl["Date"] = pd.to_datetime(xl["Date"])
    xl = xl.dropna(subset=["Line"])
    props: list[dict] = []
    for _, r in xl.iterrows():
        try:
            raw_time = str(r.get("Game_Time_ET", "") or "").strip()
            game_time = raw_time.replace(" ET", "").strip() if raw_time and raw_time != "nan" else ""
            props.append({
                "player":    str(r["Player"]).strip(),
                "date":      r["Date"],
                "line":      float(r["Line"]),
                "min_line":  float(r["Min Line"])  if pd.notna(r.get("Min Line", ""))  else None,
                "max_line":  float(r["Max Line"])  if pd.notna(r.get("Max Line", ""))  else None,
                "over_odds":  float(r.get("Over Odds",   -110) or -110),
                "under_odds": float(r.get("Under Odds",  -110) or -110),
                "game":      str(r.get("Game", "")).strip(),
                "home":      str(r.get("Home", "")).strip(),
                "away":      str(r.get("Away", "")).strip(),
                "game_time": game_time,
                "books":     int(r.get("Books", 1) or 1),
                "season":    "2025-26",
                "source":    "real",
            })
        except Exception:
            continue
    return props


# =============================================================================
# STEP 2: BUILD FEATURE ROWS
# =============================================================================

def build_feature_rows(
    player_idx, h2h_lkp, dvp_rank, pace_cache, b2b_map, props
) -> pd.DataFrame:
    """
    Build one feature-rich row per prop.

    FIX: actual_pts is now OPTIONAL.
    - Props with a game log result → actual_pts set, target_cls set (used for training)
    - Props with NO game log result yet (future/incomplete log) → actual_pts = NaN,
      target_cls = -1 (excluded from model training, but still scored for the dashboard)
    - opponent derived from Excel home/away fields FIRST; game log used as fallback.
      This means props are never skipped just because the game hasn't been played yet.
    """
    rows: list[dict] = []
    skip = {"no_player": 0, "thin_history": 0, "no_feats": 0}
    ungraded = 0

    for prop in props:
        pname    = prop["player"]
        date     = prop["date"]
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        line     = prop["line"]

        hist = player_idx.get(pname)
        if hist is None:
            skip["no_player"] += 1; continue

        prior = hist[hist["GAME_DATE"] < pd.Timestamp(date)]
        if len(prior) < 5:
            skip["thin_history"] += 1; continue

        pos = str(prior["PLAYER_POSITION"].iloc[-1])

        # ── Derive opponent from Excel home/away (FIX: no game log dependency) ──
        # Excel has Home and Away team abbreviations already.
        # Determine which side the player is on from their most recent team abbreviation.
        home = str(prop.get("home", "")).strip()
        away = str(prop.get("away", "")).strip()
        my_team = str(prior["GAME_TEAM_ABBREVIATION"].iloc[-1]).strip() if "GAME_TEAM_ABBREVIATION" in prior.columns else ""

        if home and away and my_team:
            opponent = home if my_team.upper() == away.upper() else away
        elif home and away:
            opponent = away  # default: player is home team
        else:
            # Last resort: check game log for this exact date
            actual_game_for_opp = hist[hist["GAME_DATE"] == pd.Timestamp(date)]
            opponent = str(actual_game_for_opp["OPPONENT"].values[0]) if len(actual_game_for_opp) else ""

        # ── Look up actual result from game log (optional — does NOT skip if missing) ──
        actual_game = hist[hist["GAME_DATE"] == pd.Timestamp(date)]
        if len(actual_game) > 0 and pd.notna(actual_game["PTS"].values[0]):
            actual_pts = float(actual_game["PTS"].values[0])
            target_cls = 1 if actual_pts > line else 0
            # Use game log opponent as ground truth if available (more reliable)
            if len(actual_game) > 0 and "OPPONENT" in actual_game.columns:
                opponent = str(actual_game["OPPONENT"].values[0])
        else:
            actual_pts = float("nan")
            target_cls = -1   # sentinel: exclude from model training
            ungraded += 1

        rest_d  = b2b_map.get((pname, date_str), 99)
        pos_grp = get_pos_group(pos)

        feats = extract_prediction_features(
            prior_played=prior,
            line=line,
            opponent=opponent,
            rest_days=rest_d,
            pos_raw=pos,
            game_date=pd.Timestamp(date),
            min_line=prop.get("min_line"),
            max_line=prop.get("max_line"),
            dvp_rank_cache={(opponent, pos_grp): dvp_rank.get((opponent, pos), 15)},
            pace_rank_cache=pace_cache,
        )
        if feats is None:
            skip["no_feats"] += 1; continue

        hk = h2h_lkp.get((pname, opponent), {})
        feats["h2h_ts_dev"]  = float(hk.get("H2H_TS_VS_OVERALL",  0) or 0)
        feats["h2h_fga_dev"] = float(hk.get("H2H_FGA_VS_OVERALL", 0) or 0)
        feats["h2h_min_dev"] = float(hk.get("H2H_MIN_VS_OVERALL", 0) or 0)
        feats["h2h_conf"]    = float(hk.get("H2H_CONFIDENCE",     0) or 0)
        feats["h2h_games"]   = float(hk.get("H2H_GAMES",          0) or 0)
        feats["h2h_trend"]   = float(hk.get("H2H_PTS_TREND",      0) or 0)

        row = {
            **feats,
            "actual_pts": actual_pts,
            "target_cls": target_cls,
            "player":     pname,
            "date":       pd.Timestamp(date),
            "date_str":   date_str,
            "pos":        pos,
            "opponent":   opponent,
            "h2h_avg":    float(hk.get("H2H_AVG_PTS", 0) or 0),
            **{k: prop[k] for k in (
                "line", "min_line", "max_line", "over_odds", "under_odds",
                "game", "home", "away", "game_time", "books", "season", "source"
            )},
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # fillna(0) for feature columns only — preserve NaN in actual_pts for ungraded rows
    feat_cols = [c for c in df.columns if c != "actual_pts"]
    df[feat_cols] = df[feat_cols].fillna(0)

    graded    = df[df["target_cls"] >= 0]
    over_rate = graded["target_cls"].mean() if len(graded) else float("nan")
    print(f"    Rows built: {len(df):,}  |  Graded: {len(graded):,}  |  Ungraded: {ungraded:,}  |  Skipped: {skip}")
    if len(graded):
        print(f"    OVER rate (graded only): {over_rate:.1%}")
    return df


# =============================================================================
# STEP 3: TRAIN MODELS (OOF — no leakage)
# =============================================================================

def train_models_oof(df: pd.DataFrame, skip_train: bool = False):
    """
    Train V14 GBT classifier + regressor using TimeSeriesSplit OOF.
    Fit isotonic calibrator on OOF predictions.
    Compute player trust scores from OOF accuracy.
    Save all models to MODEL_DIR.

    FIX: df may contain ungraded rows (target_cls == -1, actual_pts == NaN).
    These are excluded from training but still receive OOF scores for the dashboard.
    oof_prob and oof_reg are returned indexed to the FULL df (n rows), with zeros
    for ungraded rows so apply_v14_scoring can handle them uniformly.
    """
    # Split into graded (has actual result) and ungraded (future / no CSV match)
    graded_mask = df["target_cls"].values >= 0
    df_tr  = df[graded_mask].reset_index(drop=True)
    n_full = len(df)
    n_tr   = len(df_tr)
    print(f"  Training on {n_tr:,} graded rows  ({n_full - n_tr:,} ungraded rows will be scored only)")

    X_tr   = df_tr[ML_FEATURES].values
    y_cls  = df_tr["target_cls"].values
    y_reg  = df_tr["actual_pts"].values
    lines  = df_tr["line"].values

    if skip_train and FILE_CLF.exists():
        print("  --no-train: using existing models")
        with open(FILE_CLF, "rb") as f: clf = pickle.load(f)
        with open(FILE_REG, "rb") as f: reg = pickle.load(f)
        with open(FILE_CAL, "rb") as f: cal = pickle.load(f)
        trust = json.loads(FILE_TRUST.read_text()) if FILE_TRUST.exists() else {}
        # Score ALL rows (graded + ungraded) using existing models
        X_full   = df[ML_FEATURES].values
        oof_prob = clf.predict_proba(X_full)[:, 1]
        oof_reg  = reg.predict(X_full)
        return clf, reg, cal, trust, oof_prob, oof_reg

    # Sample weights — recency + quality adjustments (training rows only)
    w = 1.0 + (np.arange(n_tr) / n_tr)
    w[df_tr["date"].dt.month.values == 10] *= 0.4   # Oct early-season downweight
    w[df_tr["date"].dt.month.values == 11] *= 0.7
    w[df_tr["h2h_conf"].values > 0.6] *= 1.2
    w[df_tr["mean_reversion_risk"].values == 1.0] *= 0.8
    w = w / w.mean()

    clf_kw = dict(
        n_estimators=400, max_depth=3, learning_rate=0.035,
        min_samples_leaf=15, subsample=0.75,
        n_iter_no_change=30, validation_fraction=0.1, tol=1e-4, random_state=42,
    )
    reg_kw = dict(
        n_estimators=400, max_depth=4, learning_rate=0.035,
        min_samples_leaf=15, subsample=0.75, loss="huber", alpha=0.9,
        n_iter_no_change=30, validation_fraction=0.1, tol=1e-4, random_state=42,
    )

    # OOF loop — trained and evaluated on graded rows only
    tscv         = TimeSeriesSplit(n_splits=5)
    oof_prob_tr  = np.zeros(n_tr)   # indexed to df_tr (graded rows)
    oof_reg_tr   = np.zeros(n_tr)

    for fold, (tr, va) in enumerate(tscv.split(X_tr), 1):
        cf = GradientBoostingClassifier(**clf_kw)
        rf = GradientBoostingRegressor(**reg_kw)
        cf.fit(X_tr[tr], y_cls[tr], sample_weight=w[tr])
        rf.fit(X_tr[tr], y_reg[tr], sample_weight=w[tr])
        oof_prob_tr[va] = cf.predict_proba(X_tr[va])[:, 1]
        oof_reg_tr[va]  = rf.predict(X_tr[va])
        c = ((oof_prob_tr[va] > 0.5) == y_cls[va]).mean()
        r = ((oof_reg_tr[va]  > lines[va]) == y_cls[va]).mean()
        print(f"    Fold {fold}: clf={c:.3f}  reg={r:.3f}")

    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof_prob_tr, y_cls)

    # OOF trust scores per player (minimum 10 games, graded rows only)
    oof_dir_tr = (oof_prob_tr > 0.5).astype(int)
    df_tr["_oof_correct"] = (oof_dir_tr == y_cls).astype(int)
    trust = {
        p: round(float(g["_oof_correct"].mean()), 4)
        for p, g in df_tr.groupby("player")
        if len(g) >= 10
    }

    # Final models trained on graded data only
    clf = GradientBoostingClassifier(**clf_kw)
    reg = GradientBoostingRegressor(**reg_kw)
    clf.fit(X_tr, y_cls, sample_weight=w)
    reg.fit(X_tr, y_reg, sample_weight=w)
    print(f"  Final models: clf={clf.n_estimators_} trees  reg={reg.n_estimators_} trees")

    MODEL_DIR.mkdir(exist_ok=True)
    with open(FILE_CLF, "wb") as f: pickle.dump(clf, f)
    with open(FILE_REG, "wb") as f: pickle.dump(reg, f)
    with open(FILE_CAL, "wb") as f: pickle.dump(cal, f)
    FILE_TRUST.write_text(json.dumps(trust, indent=2))
    print(f"  Models saved → {MODEL_DIR}")

    # Score ALL rows (graded + ungraded) with the final fitted models
    # oof_prob/oof_reg are indexed to the FULL df (n_full rows)
    X_full   = df[ML_FEATURES].values
    oof_prob = clf.predict_proba(X_full)[:, 1]
    oof_reg  = reg.predict(X_full)
    # For graded rows, replace with the OOF estimates (unbiased, no leakage)
    graded_idx = np.where(graded_mask)[0]
    oof_prob[graded_idx] = oof_prob_tr
    oof_reg[graded_idx]  = oof_reg_tr

    return clf, reg, cal, trust, oof_prob, oof_reg


# =============================================================================
# STEP 4: APPLY V14 SCORING
# =============================================================================

def apply_v14_scoring(
    df: pd.DataFrame, cal, trust: dict, oof_prob: np.ndarray, oof_reg: np.ndarray
) -> pd.DataFrame:
    """
    Apply the V14 Adaptive Fusion scoring to the feature DataFrame.
    Populates: direction, tierLabel, tier, conf, predPts, predGap,
               calProb, units, result, actualPts, delta, enginesAgree, isLean.
    """
    lines    = df["line"].values
    actual   = df["actual_pts"].values
    y_cls    = df["target_cls"].values
    cal_prob = cal.transform(oof_prob)
    pred_pts = oof_reg
    gap      = np.abs(pred_pts - lines)
    gap_conf = np.clip(0.5 + gap * 0.04, 0.45, 0.90)
    sp       = df["season_progress"].values
    ew       = df["early_season_weight"].values

    # Engine A: composite signal — uses same compute_composite() as batch_predict
    # for consistency between historical backtest and live predictions
    from rolling_engine import compute_composite
    temp_dirs = ["OVER" if d == 1 else "UNDER" for d in (pred_pts > lines).astype(int)]
    composite_arr = np.zeros(len(df))
    comp_conf_arr = np.zeros(len(df))
    for i in range(len(df)):
        row_feats = {col: df[col].iloc[i] for col in df.columns if col in df.columns}
        pos_grp = str(df["pos"].iloc[i]) if "pos" in df.columns else "Guard"
        from config import get_pos_group
        pg = get_pos_group(pos_grp)
        h2h_avg_i   = float(df["h2h_avg"].iloc[i])   if "h2h_avg"   in df.columns else 0.0
        h2h_games_i = int(df["h2h_games"].iloc[i])   if "h2h_games" in df.columns else 0
        c, _, _ = compute_composite(row_feats, lines[i], temp_dirs[i], pg,
                                    h2h_avg=h2h_avg_i, use_h2h=(h2h_games_i >= 3))
        composite_arr[i] = c
        comp_conf_arr[i] = min(0.85, max(0.50, 0.5 + abs(c) * 0.3))
    composite = composite_arr
    comp_conf = comp_conf_arr

    # V14 Adaptive Fusion
    alpha = 0.55 + 0.05 * sp
    beta  = 0.30 - 0.05 * sp
    fc    = alpha * cal_prob + beta * gap_conf + 0.15 * comp_conf

    # Regime adjustments
    reg_dir = (pred_pts > lines).astype(int)
    xhot    = df["extreme_hot"].values
    xcold   = df["extreme_cold"].values
    mom     = df["momentum"].values
    fc = np.where((xhot == 1) & (reg_dir == 1), fc - 0.03, fc)
    fc = np.where((xcold == 1) & (reg_dir == 0), fc - 0.02, fc)
    fc = np.where((mom > 3) & (mom <= 6) & (reg_dir == 1),  fc + 0.015, fc)
    fc = np.where((mom < -3) & (mom >= -6) & (reg_dir == 0), fc + 0.015, fc)
    fc = fc * (0.70 + 0.30 * ew)
    fc = np.where(df["is_long_rest"].values == 1, fc - 0.03, fc)
    fc = np.where((lines >= HIGH_LINE_THRESHOLD) & (reg_dir == 1), fc - 0.03, fc)
    fc = np.where(df["line_sharpness"].values > 0.80, fc + 0.01, fc)
    fc = np.clip(fc, 0.40, 0.90)

    # Direction + lean
    clf_dir       = (oof_prob > 0.5).astype(int)
    engines_agree = (reg_dir == clf_dir)
    lo, hi        = LEAN_ZONE
    is_over  = (cal_prob >= hi) & engines_agree
    is_under = (cal_prob <= lo) & engines_agree
    is_lean  = ~(is_over | is_under)

    # H2H alignment gate
    h2h_ts = df["h2h_ts_dev"].values
    h2h_g  = df["h2h_games"].values
    h2h_ok = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if h2h_g[i] >= 5:  # V18: raised from 3 to 5 for statistical robustness
            if is_over[i]  and h2h_ts[i] < -3: h2h_ok[i] = False
            if is_under[i] and h2h_ts[i] >  3: h2h_ok[i] = False

    # Tier assignment
    std10    = df["std10"].values
    vol_risk = df["vol_risk"].values
    tl_arr: list[str] = []
    for i in range(len(df)):
        if is_lean[i]:
            tl_arr.append("T3_LEAN"); continue
        f  = fc[i]; g2 = gap[i]; s = std10[i]; ha = h2h_ok[i]
        hv = s > 9   # V18: vol_risk removed from high_vol gate
        if   f >= 0.73 and g2 >= 5.0 and s <= 6 and ha and not hv: tl_arr.append("T1_ULTRA")
        elif f >= 0.68 and g2 >= 4.0 and s <= 7 and ha and not hv: tl_arr.append("T1_PREMIUM")
        elif f >= 0.63 and g2 >= 3.0 and s <= 8 and ha and not hv: tl_arr.append("T1")
        elif f >= 0.56 and g2 >= 1.5 and s <= 9 and ha:            tl_arr.append("T2")  # V18: gap 2.0→1.5
        else:                                                         tl_arr.append("T3")

    # Trust demotion: T1 → T2 for low-trust players
    for i in range(len(df)):
        if trust.get(df["player"].iloc[i], 1.0) < TRUST_THRESHOLD and tl_arr[i].startswith("T1"):
            tl_arr[i] = "T2"

    tl_arr = np.array(tl_arr)
    units  = np.array([UNIT_MAP.get(t, 0.0) for t in tl_arr])
    dirs   = [
        ("LEAN OVER" if cal_prob[i] >= 0.5 else "LEAN UNDER") if is_lean[i]
        else ("OVER" if is_over[i] else "UNDER")
        for i in range(len(df))
    ]
    # FIX: ungraded rows (no actual_pts) get result="" and delta=NaN — never WIN/LOSS/LEAN
    graded_mask_sc = df["target_cls"].values >= 0   # graded rows in this df
    results = []
    for i in range(len(df)):
        if not graded_mask_sc[i]:
            results.append("")          # ungraded: no result yet
        elif is_lean[i]:
            results.append("LEAN")
        elif is_over[i]:
            results.append("WIN" if actual[i] > lines[i] else "LOSS")
        else:
            results.append("WIN" if actual[i] <= lines[i] else "LOSS")

    # delta only meaningful for graded rows
    delta_arr = np.where(graded_mask_sc, np.round(actual - lines, 1), float("nan"))

    df["direction"]    = dirs
    df["tierLabel"]    = tl_arr
    df["tier"]         = [1 if t.startswith("T1") else (2 if t == "T2" else 3) for t in tl_arr]
    df["conf"]         = np.round(fc, 4)
    df["predPts"]      = np.round(pred_pts, 1)
    df["predGap"]      = np.round(gap, 2)
    df["calProb"]      = np.round(cal_prob, 4)
    df["units"]        = units
    df["result"]       = results
    df["actualPts"]    = actual          # NaN for ungraded — _build_play handles this
    df["delta"]        = delta_arr
    df["enginesAgree"] = engines_agree
    df["isLean"]       = is_lean

    t12 = df[(df["tier"] <= 2) & df["result"].isin(["WIN", "LOSS"])]
    t1  = df[(df["tier"] == 1) & df["result"].isin(["WIN", "LOSS"])]
    if len(t12): print(f"  T1+T2 OOF accuracy: {(t12['result']=='WIN').mean():.1%}  ({len(t12):,})")
    if len(t1):  print(f"  T1    OOF accuracy: {(t1['result']=='WIN').mean():.1%}  ({len(t1):,})")
    return df


# =============================================================================
# STEP 5: BUILD JSON FILES
# =============================================================================

def build_json_files(df: pd.DataFrame, recent_idx: dict, target_date=None) -> None:
    """
    Convert scored feature DataFrame into play dicts and write JSON files.

    FIX — 2025-26 is MERGED not overwritten:
    - season_2025_26.json is merged with the existing file using the same
      immutability rules as batch_predict.append_season_json.
      Graded plays (WIN/LOSS/DNP) are never overwritten.
      Ungraded generate plays update ungraded batch_predict plays.
      Plays in the existing file but NOT in df (e.g. from batch_predict for
      future dates not yet in Excel) are preserved as-is.
    - season_2024_25.json is still a full overwrite (all historical, complete).
    - today.json is always overwritten — it's the live dashboard view.
    """
    DATA_DIR.mkdir(exist_ok=True)
    plays_2526_new: list[dict] = []
    plays_2425: list[dict] = []

    total = len(df)
    for i in range(total):
        if i % 5000 == 0 and i > 0:
            print(f"    {i}/{total}...")
        play = _build_play(df.iloc[i], recent_idx)
        if str(df["season"].iloc[i]).startswith("2025"):
            plays_2526_new.append(play)
        else:
            plays_2425.append(play)

    def _sort_key(p):
        return (p.get("tier", 9), -p.get("conf", 0))

    plays_2526_new.sort(key=lambda p: (p["date"], _sort_key(p)))
    plays_2425.sort(key=lambda p: (p["date"], _sort_key(p)))

    # ── Merge 2025-26 with existing season JSON (FIX: not a full overwrite) ──
    existing_2526: list[dict] = []
    if FILE_SEASON_2526.exists():
        try:
            with open(FILE_SEASON_2526) as f:
                existing_2526 = json.load(f)
        except Exception:
            existing_2526 = []

    def _key(p): return (p.get("player", ""), p.get("date", ""), str(p.get("line", "")))
    ex_map = {_key(p): i for i, p in enumerate(existing_2526)}

    # Apply new plays: respect immutability of graded plays
    for p in plays_2526_new:
        k = _key(p)
        if k in ex_map:
            old = existing_2526[ex_map[k]]
            if old.get("result") in ("WIN", "LOSS", "DNP"):
                continue   # immutable — never overwrite a graded play
            existing_2526[ex_map[k]] = p   # update ungraded with fresh scoring
        else:
            existing_2526.append(p)   # new play not seen before

    # Re-sort chronologically within 2025-26
    existing_2526.sort(key=lambda p: (p.get("date", ""), _sort_key(p)))
    plays_2526 = existing_2526

    _save(FILE_SEASON_2526, plays_2526)
    print(f"  season_2025_26.json: {len(plays_2526):,} plays  (from generate: {len(plays_2526_new):,} new/updated)")
    log_event("GEN", "SEASON_2526_GENERATED", FILE_SEASON_2526.name, rows_after=len(plays_2526))

    _save(FILE_SEASON_2425, plays_2425)
    print(f"  season_2024_25.json: {len(plays_2425):,} plays  — LOCKED")
    log_event("GEN", "SEASON_2425_GENERATED", FILE_SEASON_2425.name, rows_after=len(plays_2425))

    # today.json — most recent date or user-specified
    all_plays = plays_2526 + plays_2425
    if target_date:
        today_plays = [p for p in all_plays if p["date"] == target_date]
    else:
        # Most recent date that has any 2025-26 plays
        dates_2526 = sorted(set(p["date"] for p in plays_2526))
        best = dates_2526[-1] if dates_2526 else ""
        today_plays = [p for p in all_plays if p["date"] == best]

    today_plays.sort(key=lambda p: (p.get("tier", 9), -p.get("conf", 0)))
    _save(FILE_TODAY, today_plays)
    print(f"  today.json: {len(today_plays)} plays  date={today_plays[0]['date'] if today_plays else '?'}")

    # Backtest summary for dashboard header
    _write_summary(plays_2526, plays_2425)
    print("  backtest_summary.json written")


def _build_play(row: pd.Series, recent_idx: dict) -> dict:
    """Convert one scored feature row into a play dict for the dashboard."""
    pname    = str(row["player"])
    date_str = str(row["date_str"])
    line     = float(row["line"])

    # Recent 20 scores prior to this game
    all_s = recent_idx.get(pname, [])
    prior = [(d, pt, h, op) for d, pt, h, op in all_s if d < date_str]
    r20   = [
        {"date": d, "pts": float(pt), "home": bool(h), "opponent": str(op),
         "overLine": float(pt) > line}
        for d, pt, h, op in prior[-20:]
    ]

    # Reconstruct rolling display fields from ML features
    L30 = float(row.get("level", 0))
    L10 = L30 + float(row.get("reversion", 0))
    L5  = L30 + float(row.get("momentum", 0))
    L3  = L5  + float(row.get("acceleration", 0))
    flags = max(0, min(10, int(round(float(row.get("hr10", 0.5)) * 10))))

    play = {
        # Identity
        "player":   pname,
        "date":     date_str,
        "game":     str(row.get("game", "")),
        "home":     str(row.get("home", "")),
        "away":     str(row.get("away", "")),
        "opponent": str(row.get("opponent", "")),
        "position": str(row.get("pos", "")),
        "season":   str(row.get("season", "2025-26")),
        "source":   str(row.get("source", "real")),
        # Prop line
        "line":       line,
        "overOdds":   float(row.get("over_odds",  -110) or -110),
        "underOdds":  float(row.get("under_odds", -110) or -110),
        "books":      int(row.get("books", 1) or 1),
        "game_time":  str(row.get("game_time", "") or ""),
        # V14 scoring outputs
        "direction":  str(row.get("direction", "")),
        "tierLabel":  str(row.get("tierLabel", "T3")),
        "tier":       int(row.get("tier", 3)),
        "conf":       float(row.get("conf", 0.5)),
        "predPts":    float(row.get("predPts", line)),
        "predGap":    float(row.get("predGap", 0)),
        "calProb":    float(row.get("calProb", 0.5)),
        "units":      float(row.get("units", 0)),
        "flags":      flags,
        "enginesAgree":      bool(row.get("enginesAgree", True)),
        "meanReversionRisk": float(row.get("mean_reversion_risk", 0)),
        "seasonProgress":    float(row.get("season_progress", 0.5)),
        "earlySeasonW":      float(row.get("early_season_weight", 1.0)),
        "volRisk":           float(row.get("vol_risk", 0)),
        # Grade — actualPts and delta are None for ungraded (future) plays
        "result":    str(row.get("result", "")),
        "actualPts": _s(row.get("actual_pts")) if pd.isna(row.get("actual_pts", float("nan"))) else _s(float(row.get("actual_pts", 0))),
        "delta":     None if pd.isna(row.get("delta", float("nan"))) else _s(float(row.get("delta", 0))),
        # Rolling display
        "l30": round(L30, 1), "l10": round(L10, 1),
        "l5":  round(L5,  1), "l3":  round(L3,  1),
        "std10":    round(float(row.get("std10", 5)), 2),
        "hr10":     round(float(row.get("hr10", 0.5)), 3),
        "hr30":     round(float(row.get("hr30", 0.5)), 3),
        "min_l10":  round(float(row.get("min_l10", 30)), 1),
        "fga_l10":  round(float(row.get("fga_l10", 10)), 1),
        "momentum": round(float(row.get("momentum", 0)), 2),
        # Contextual
        "defP_dynamic": int(row.get("defP_dynamic", 15)),
        "pace_rank":    int(row.get("pace_rank", 15)),
        "h2h_avg":      round(float(row.get("h2h_avg", 0)), 1),
        "h2h_games":    int(row.get("h2h_games", 0) or 0),
        "h2h_ts_dev":   round(float(row.get("h2h_ts_dev", 0)), 4),
        "h2h_conf":     round(float(row.get("h2h_conf", 0)), 4),
        "lineSharpness": round(float(row.get("line_sharpness", 0.67)), 3),
        "is_b2b":   int(row.get("is_b2b", 0)),
        "rest_days": int(row.get("rest_days", 2)),
        "home_l10": round(float(row.get("home_l10", 0)), 1),
        "away_l10": round(float(row.get("away_l10", 0)), 1),
        "min_l30":  round(float(row.get("min_l30", 0)), 1),
        # Decision signal fields (exposed from ML features)
        "usage_l10":      round(float(row.get("usage_l10", 0)), 2),
        "usage_l30":      round(float(row.get("usage_l30", 0)), 2),
        "min_cv":         round(float(row.get("min_cv", 0)), 3),
        "fta_l10":        round(float(row.get("fta_l10", 0)), 1),
        "ft_rate":        round(float(row.get("ft_rate", 0)), 3),
        "fg3a_l10":       round(float(row.get("fg3a_l10", 0)), 1),
        "pts_per_min":    round(float(row.get("pts_per_min", 0)), 3),
        "home_away_split": round(float(row.get("home_away_split", 0)), 1),
        "level_ewm":      round(float(row.get("level_ewm", 0)), 1),
        "line_vs_l30":    round(float(row.get("line_vs_l30", 0)), 1),
        "extreme_hot":    int(row.get("extreme_hot", 0) or 0),
        "extreme_cold":   int(row.get("extreme_cold", 0) or 0),
        "is_long_rest":   int(row.get("is_long_rest", 0) or 0),
        "recent_min_trend": round(float(row.get("recent_min_trend", 0)), 1),
        "h2h_trend":      round(float(row.get("h2h_trend", 0)), 1),
        "lineHistory": [{"line": _s(line), "batch": 0, "ts": ""}],
        "recent20": r20,
        "preMatchReason":  "",
        "postMatchReason": "",
        "lossType":        "",
    }

    # Generate narratives (non-critical — catch all errors)
    p_for_reason = {
        **play,
        "_n_games": int(row.get("games_depth", 0.5) * 30),
        "is_long_rest": int(row.get("is_long_rest", 0)),
        "flagDetails": [],
    }
    try:
        play["preMatchReason"] = generate_pre_match_reason(p_for_reason)
    except Exception:
        play["preMatchReason"] = f"V14: {play['direction']} {line}"

    if play["result"] in ("WIN", "LOSS"):
        try:
            post, lt = generate_post_match_reason(p_for_reason)
            play["postMatchReason"] = post
            play["lossType"] = lt
        except Exception:
            play["postMatchReason"] = f"Result: {play['result']}. Actual: {play['actualPts']:.0f}pts."
            play["lossType"] = "MODEL_CORRECT" if play["result"] == "WIN" else "MODEL_FAILURE_GENERAL"

    return play


def _write_summary(plays_2526: list[dict], plays_2425: list[dict]) -> None:
    """Write backtest_summary.json for dashboard header stats."""
    def _s_season(plays):
        # Guard: use .get() so plays from any schema version (old JSON, batch_predict) are safe
        def _res(p): return p.get("result") or ""
        def _tier(p): return p.get("tier") or 3
        def _tl(p):   return p.get("tierLabel") or "T3"
        def _units(p): return float(p.get("units") or 0)

        g     = [p for p in plays if _res(p) in ("WIN", "LOSS")]
        def acc(sub): return round(sum(1 for p in sub if _res(p) == "WIN") / max(len(sub), 1), 4)
        t1    = [p for p in g if _tier(p) == 1]
        t2    = [p for p in g if _tl(p) == "T2"]
        t12   = t1 + t2
        ultra = [p for p in g if _tl(p) == "T1_ULTRA"]
        prem  = [p for p in g if _tl(p) == "T1_PREMIUM"]
        pl    = sum(
            _units(p) * 0.909 if _res(p) == "WIN" else -_units(p)
            for p in g if _units(p) > 0
        )
        monthly: dict = {}
        for p in t12:
            m = (p.get("date") or "")[:7]
            if not m: continue
            if m not in monthly: monthly[m] = {"wins": 0, "total": 0, "pl": 0.0}
            monthly[m]["total"] += 1
            if _res(p) == "WIN": monthly[m]["wins"] += 1
            monthly[m]["pl"] += _units(p) * 0.909 if _res(p) == "WIN" else -_units(p)
        return {
            "total": len(plays), "graded": len(g),
            "t1_ultra":   {"n": len(ultra), "acc": acc(ultra)},
            "t1_premium": {"n": len(prem),  "acc": acc(prem)},
            "t1":         {"n": len(t1),    "acc": acc(t1)},
            "t2":         {"n": len(t2),    "acc": acc(t2)},
            "t12":        {"n": len(t12),   "acc": acc(t12)},
            "pl": round(pl, 1), "monthly": monthly,
        }

    summary = {
        "generated":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "season_2526":  _s_season(plays_2526),
        "season_2425":  _s_season(plays_2425),
    }
    _save(DATA_DIR / "backtest_summary.json", summary)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PropEdge V14 — Generate Season JSONs")
    parser.add_argument("--season",   default="both", choices=["2526", "2425", "both"])
    parser.add_argument("--no-train", action="store_true", help="Skip retraining — use existing models")
    parser.add_argument("--date",     default=None,   help="Set today.json to this date (YYYY-MM-DD)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  PropEdge {VERSION} — Generate Season JSONs")
    print("=" * 60)
    t0 = time.time()

    # [1/5] Load data
    (player_idx, h2h_lkp, dvp_rank, pace_cache,
     b2b_map, recent_idx, props_2526, props_2425) = load_all_data()

    # [2/5] Build feature rows (both seasons combined for training)
    print("\n[2/5] Building feature rows for training (both seasons)...")
    df_train = build_feature_rows(
        player_idx, h2h_lkp, dvp_rank, pace_cache, b2b_map,
        props_2526 + props_2425
    )

    # [3/5] Train V14 models (OOF 5-fold)
    print(f"\n[3/5] Training V14 models (OOF 5-fold) on {len(df_train):,} rows...")
    clf, reg, cal, trust, oof_prob, oof_reg = train_models_oof(df_train, skip_train=args.no_train)

    # [4/5] Apply V14 scoring (using OOF predictions — no leakage)
    print("\n[4/5] Applying V14 Adaptive Fusion scoring...")
    df_scored = apply_v14_scoring(df_train, cal, trust, oof_prob, oof_reg)

    # [5/5] Write JSON files
    print("\n[5/5] Writing JSON output files...")
    build_json_files(df_scored, recent_idx, target_date=args.date)

    # Print season summaries from scored df (excludes ungraded rows cleanly)
    for season_label, season_filter in [("2025-26", "2025"), ("2024-25", "2024")]:
        rows    = df_scored[df_scored["season"].str.startswith(season_filter)]
        graded  = rows[rows["result"].isin(["WIN", "LOSS"])]
        wins    = int((graded["result"] == "WIN").sum())
        losses  = int(len(graded) - wins)
        pct     = f"{wins/len(graded)*100:.1f}%" if len(graded) else "—"
        t12     = graded[graded["tier"] <= 2]
        t12_w   = int((t12["result"] == "WIN").sum())
        t12_pct = f"{t12_w/len(t12)*100:.1f}%" if len(t12) else "—"
        ungr    = int((rows["result"] == "").sum()) + int(rows["result"].isna().sum())
        print(f"\n  {season_label}: {len(rows):,} rows  |  "
              f"Graded: {wins}W/{losses}L = {pct}  |  "
              f"T1+T2: {t12_w}W/{len(t12)-t12_w}L = {t12_pct}  |  "
              f"Ungraded: {ungr}")

    elapsed = time.time() - t0
    print(f"\n  ✓ Generate complete  ({elapsed:.0f}s)")
    print("=" * 60)
    log_event("GEN", "GENERATE_COMPLETE", detail=f"elapsed={elapsed:.0f}s")

    # Git push — upload all updated JSON files to GitHub
    print("\n  Pushing to GitHub...")
    try:
        from batch_predict import git_push
        git_push("GEN: full season JSON rebuild")
    except Exception as e:
        print(f"  ⚠ Git push error: {e}")


if __name__ == "__main__":
    main()
