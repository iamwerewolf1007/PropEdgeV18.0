"""
PropEdge V18.0 — model_trainer.py
Full training pipeline: GBR classifier + regressor + OOF isotonic calibrator + player trust.

Training data sources:
  - 2025-26 season: real bookmaker lines from PropEdge Excel (Player_Points_Props sheet)
  - 2024-25 season: synthetic lines generated from each player's L30 rolling average
                    (no real lines exist for that season in the Excel file)

Both seasons are concatenated before training. Temporal sample weights downweight
October rows (thin rolling windows) and upweight recent high-confidence plays.
OOF 5-fold TimeSeriesSplit — NO in-sample leakage for calibrator or trust scores.
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

from config import (
    FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_PROPS,
    FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST,
    MIN_PRIOR_GAMES, season_progress, get_pos_group,
)
from rolling_engine import filter_played, extract_prediction_features

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
]  # 56 features


# ─────────────────────────────────────────────────────────────────────────────
# PROP SOURCES
# ─────────────────────────────────────────────────────────────────────────────

def _load_real_props() -> list[dict]:
    """Load 2025-26 real bookmaker lines from Excel."""
    if not FILE_PROPS.exists():
        print("  ⚠ Excel props file missing — 2025-26 will use synthetic lines.")
        return []
    try:
        xl = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
        xl["Date"] = pd.to_datetime(xl["Date"])
        xl = xl.dropna(subset=["Line"])
        props = []
        for _, r in xl.iterrows():
            try:
                props.append({
                    "player":    str(r["Player"]).strip(),
                    "date":      r["Date"],
                    "line":      float(r["Line"]),
                    "min_line":  float(r["Min Line"])  if pd.notna(r.get("Min Line", ""))  else None,
                    "max_line":  float(r["Max Line"])  if pd.notna(r.get("Max Line", ""))  else None,
                    "season":    "2025-26",
                    "source":    "real",
                })
            except Exception:
                continue
        print(f"  Real lines (2025-26): {len(props):,}")
        return props
    except Exception as e:
        print(f"  ⚠ Excel read error: {e}")
        return []


def _generate_synthetic_props(played: pd.DataFrame) -> list[dict]:
    """
    Generate synthetic prop lines for 2024-25 season.
    One prop per player per game (after they have ≥5 prior games).
    Line = round(L30 × 2) / 2, minimum 3.5 pts.
    These approximate bookmaker lines from the player's rolling baseline.
    """
    played = played[played["GAME_DATE"] < pd.Timestamp("2025-10-01")].copy()
    played = played.sort_values(["PLAYER_NAME", "GAME_DATE"])
    props = []

    for pname, grp in played.groupby("PLAYER_NAME"):
        pts_list = grp["PTS"].fillna(0).tolist()
        dates    = grp["GAME_DATE"].tolist()
        history  = []

        for i, (d, pts) in enumerate(zip(dates, pts_list)):
            if i >= 5 and len(history) >= 5:
                l30        = np.mean(history[-30:]) if len(history) >= 30 else np.mean(history)
                synth_line = max(3.5, round(l30 * 2) / 2)
                props.append({
                    "player":   pname,
                    "date":     d,
                    "line":     synth_line,
                    "min_line": synth_line - 0.5,
                    "max_line": synth_line + 0.5,
                    "season":   "2024-25",
                    "source":   "synthetic",
                })
            history.append(pts)

    print(f"  Synthetic lines (2024-25): {len(props):,}")
    return props


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_training_data() -> pd.DataFrame:
    """
    Build training frame from BOTH seasons.
    2025-26: real bookmaker lines from Excel.
    2024-25: synthetic lines from L30 rolling average.
    """
    print("  Loading game logs (both seasons)...")
    gl24 = pd.read_csv(FILE_GL_2425, parse_dates=["GAME_DATE"])
    gl25 = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    gl   = pd.concat([gl24, gl25], ignore_index=True).sort_values(["PLAYER_NAME", "GAME_DATE"])
    gl["DNP"] = gl["DNP"].fillna(0)
    played = filter_played(gl)
    print(f"  Played rows: {len(played):,}  |  Players: {played['PLAYER_NAME'].nunique():,}")

    # Player index (full career history across both seasons)
    player_idx = {
        pname: grp.sort_values("GAME_DATE").reset_index(drop=True)
        for pname, grp in played.groupby("PLAYER_NAME")
    }

    # H2H lookup
    h2h_df = pd.read_csv(FILE_H2H)
    h2h_df = h2h_df.drop_duplicates(subset=["PLAYER_NAME", "OPPONENT"], keep="last")
    h2h_lkp = {
        (r["PLAYER_NAME"], r["OPPONENT"]): r.to_dict()
        for _, r in h2h_df.iterrows()
    }

    # DVP rank (live from combined CSV)
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

    # Pace rank
    team_fga = played.groupby("OPPONENT")["FGA"].mean()
    pace_cache = {
        t: i + 1
        for i, (t, _) in enumerate(team_fga.sort_values(ascending=False).items())
    }

    # B2B map
    b2b_map: dict = {}
    for pname, grp in played.groupby("PLAYER_NAME"):
        dates = grp["GAME_DATE"].values
        for i, d in enumerate(dates):
            rd = int((d - dates[i-1]).astype("timedelta64[D]").astype(int)) if i > 0 else 99
            b2b_map[(pname, pd.Timestamp(d).strftime("%Y-%m-%d"))] = rd

    # Prop lines — both seasons
    print("  Loading prop lines...")
    props_real      = _load_real_props()
    props_synthetic = _generate_synthetic_props(played)
    all_props       = props_real + props_synthetic
    print(f"  Total props: {len(all_props):,}")

    # Build feature rows
    print("  Extracting features...")
    rows = []
    skipped = {"no_player": 0, "thin_history": 0, "no_actual": 0, "no_feats": 0}

    for prop in all_props:
        pname    = prop["player"]
        date     = prop["date"]
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        line     = prop["line"]

        hist = player_idx.get(pname)
        if hist is None:
            skipped["no_player"] += 1; continue

        prior = hist[hist["GAME_DATE"] < pd.Timestamp(date)]
        if len(prior) < MIN_PRIOR_GAMES:
            skipped["thin_history"] += 1; continue

        actual_game = hist[hist["GAME_DATE"] == pd.Timestamp(date)]
        if len(actual_game) == 0:
            skipped["no_actual"] += 1; continue

        actual_pts = float(actual_game["PTS"].values[0])
        if pd.isna(actual_pts):
            skipped["no_actual"] += 1; continue

        pos       = str(prior["PLAYER_POSITION"].iloc[-1])
        opponent  = str(actual_game["OPPONENT"].values[0])
        rest_days = b2b_map.get((pname, date_str), 99)
        pos_grp   = get_pos_group(pos)

        feats = extract_prediction_features(
            prior_played=prior,
            line=line,
            opponent=opponent,
            rest_days=rest_days,
            pos_raw=pos,
            game_date=pd.Timestamp(date),
            min_line=prop.get("min_line"),
            max_line=prop.get("max_line"),
            dvp_rank_cache={(opponent, pos_grp): dvp_rank.get((opponent, pos), 15)},
            pace_rank_cache=pace_cache,
        )
        if feats is None:
            skipped["no_feats"] += 1; continue

        hk = h2h_lkp.get((pname, opponent), {})
        feats["h2h_ts_dev"]  = float(hk.get("H2H_TS_VS_OVERALL",  0) or 0)
        feats["h2h_fga_dev"] = float(hk.get("H2H_FGA_VS_OVERALL", 0) or 0)
        feats["h2h_min_dev"] = float(hk.get("H2H_MIN_VS_OVERALL", 0) or 0)
        feats["h2h_conf"]    = float(hk.get("H2H_CONFIDENCE",     0) or 0)
        feats["h2h_games"]   = float(hk.get("H2H_GAMES",          0) or 0)
        feats["h2h_trend"]   = float(hk.get("H2H_PTS_TREND",      0) or 0)

        feats["actual_pts"] = actual_pts
        feats["target_cls"] = 1 if actual_pts > line else 0
        feats["player"]     = pname
        feats["date"]       = pd.Timestamp(date)
        feats["position"]   = pos
        feats["line"]       = line
        feats["season"]     = prop.get("season", "2025-26")
        feats["source"]     = prop.get("source", "real")

        rows.append(feats)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df = df.fillna(0)
    print(f"  Training rows: {len(df):,}  (skipped: {skipped})")
    print(f"  Season split — 2025-26: {(df['season']=='2025-26').sum():,}  "
          f"2024-25: {(df['season']=='2024-25').sum():,}")
    print(f"  OVER rate — real: {df[df['source']=='real']['target_cls'].mean():.1%}  "
          f"synthetic: {df[df['source']=='synthetic']['target_cls'].mean():.1%}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def _build_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Temporal + quality sample weights.
    - Recent rows weighted higher (1.0 → 2.0 temporal ramp)
    - October downweighted 60% (thin rolling windows, model less reliable)
    - November downweighted 30%
    - Synthetic lines downweighted 40% (less reliable targets than real bookmaker lines)
    - High-confidence H2H rows boosted 20%
    - Extreme momentum rows downweighted 20% (model known to misjudge)
    """
    n = len(df)
    w = 1.0 + (np.arange(n) / n)  # 1.0 → 2.0

    w[df["date"].dt.month.values == 10] *= 0.40   # early-season penalty
    w[df["date"].dt.month.values == 11] *= 0.70

    # Synthetic lines are less reliable training signal than real bookmaker lines
    synthetic_mask = df.get("source", pd.Series(["real"] * n)).values == "synthetic"
    w[synthetic_mask] *= 0.60

    w[df["h2h_conf"].values > 0.6] *= 1.20        # high-confidence H2H
    w[df["mean_reversion_risk"].values == 1.0] *= 0.80  # extreme momentum

    return w / w.mean()  # normalise to mean = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save(
    clf_path: Path = FILE_CLF,
    reg_path: Path = FILE_REG,
    cal_path: Path = FILE_CAL,
    trust_path: Path = FILE_TRUST,
) -> None:
    """
    Full training pipeline using BOTH 2024-25 and 2025-26 seasons.
    Saves 4 model files. Called by:
      - python3 run.py generate    (first-time setup)
      - python3 run.py retrain     (manual retrain)
      - batch0_grade.py            (daily nightly retrain after grading)
    """
    df      = build_training_data()
    X       = df[ML_FEATURES].values
    y_cls   = df["target_cls"].values
    y_reg   = df["actual_pts"].values
    weights = _build_weights(df)
    lines   = df["line"].values

    # ── OOF 5-fold for calibration + trust (no leakage) ──────────────────────
    print("  OOF 5-fold training...")
    tscv     = TimeSeriesSplit(n_splits=5)
    oof_prob = np.zeros(len(df))
    oof_pred = np.zeros(len(df))

    clf_params = dict(
        n_estimators=400, max_depth=3, learning_rate=0.035,
        min_samples_leaf=15, subsample=0.75,
        n_iter_no_change=30, validation_fraction=0.1, tol=1e-4, random_state=42,
    )
    reg_params = dict(
        n_estimators=400, max_depth=4, learning_rate=0.035,
        min_samples_leaf=15, subsample=0.75, loss="huber", alpha=0.9,
        n_iter_no_change=30, validation_fraction=0.1, tol=1e-4, random_state=42,
    )

    fold_accs_clf, fold_accs_reg = [], []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        cf = GradientBoostingClassifier(**clf_params)
        rf = GradientBoostingRegressor(**reg_params)
        cf.fit(X[tr_idx], y_cls[tr_idx], sample_weight=weights[tr_idx])
        rf.fit(X[tr_idx], y_reg[tr_idx], sample_weight=weights[tr_idx])
        oof_prob[va_idx] = cf.predict_proba(X[va_idx])[:, 1]
        oof_pred[va_idx] = rf.predict(X[va_idx])
        c_acc = ((oof_prob[va_idx] > 0.5) == y_cls[va_idx]).mean()
        r_acc = ((oof_pred[va_idx] > lines[va_idx]) == y_cls[va_idx]).mean()
        fold_accs_clf.append(c_acc)
        fold_accs_reg.append(r_acc)
        print(f"    Fold {fold}: clf={c_acc:.3f}  reg={r_acc:.3f}")

    print(f"  OOF clf acc: {np.mean(fold_accs_clf):.3f}  |  OOF reg acc: {np.mean(fold_accs_reg):.3f}")

    # ── Calibrator on OOF probs — no in-sample leakage ───────────────────────
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_prob, y_cls)

    # ── Player trust: OOF per-player direction accuracy ───────────────────────
    oof_dir = (oof_prob > 0.5).astype(int)
    df["_oof_correct"] = (oof_dir == y_cls).astype(int)
    trust: dict[str, float] = {}
    for player, grp in df.groupby("player"):
        if len(grp) >= 10:
            trust[player] = round(float(grp["_oof_correct"].mean()), 4)
    print(f"  Player trust scores: {len(trust):,} players")

    # ── Final models on full dataset ──────────────────────────────────────────
    print("  Training final models on full dataset...")
    clf_final = GradientBoostingClassifier(**clf_params)
    reg_final = GradientBoostingRegressor(**reg_params)
    clf_final.fit(X, y_cls, sample_weight=weights)
    reg_final.fit(X, y_reg, sample_weight=weights)
    print(f"  Clf trees: {clf_final.n_estimators_}  |  Reg trees: {reg_final.n_estimators_}")

    # In-sample reference (informational only — calibrator is OOF)
    is_pred = (clf_final.predict_proba(X)[:, 1] > 0.5).astype(int)
    print(f"  In-sample clf direction acc: {(is_pred == y_cls).mean():.1%}  ← optimistic (use OOF figures)")

    # ── Save ─────────────────────────────────────────────────────────────────
    clf_path.parent.mkdir(exist_ok=True)
    with open(clf_path, "wb") as f: pickle.dump(clf_final,  f)
    with open(reg_path, "wb") as f: pickle.dump(reg_final,  f)
    with open(cal_path, "wb") as f: pickle.dump(calibrator, f)
    with open(trust_path, "w") as f: json.dump(trust, f, indent=2)

    print(f"  ✓ Models saved → {clf_path.parent}")
    print(f"  ✓ Training rows used: {len(df):,}  "
          f"(2025-26 real: {(df['source']=='real').sum():,}  "
          f"2024-25 synthetic: {(df['source']=='synthetic').sum():,})")
