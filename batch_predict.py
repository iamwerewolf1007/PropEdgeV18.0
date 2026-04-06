"""
PropEdge V17.0 — batch_predict.py
Fetch props → predict → save today.json + season JSON + Excel.
Usage: python3 batch_predict.py 1 | 2 | 3 | 4 | 5
"""

import json
import math
import pickle
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    VERSION, FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_PROPS,
    FILE_TODAY, FILE_SEASON_2526, FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST,
    ODDS_API_KEY, ODDS_BASE_URL, CREDIT_ALERT,
    today_et, et_window, season_progress, get_pos_group,
    TIER_GATES, UNIT_MAP, LEAN_ZONE, TRUST_THRESHOLD, HIGH_LINE_THRESHOLD,
    fusion_weights, clean_json, GIT_REMOTE, REPO_DIR,
)
from rolling_engine import filter_played, extract_prediction_features, compute_composite
from dvp_updater import compute_and_save_dvp
from reasoning_engine import generate_pre_match_reason
from audit import log_event
from model_trainer import ML_FEATURES

import requests


def _parse_batch() -> int:
    """Parse batch number from argv — safe even when imported by run.py."""
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            pass
    return 2

BATCH = _parse_batch()

# ─────────────────────────────────────────────────────────────────────────────
# NICKNAME NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
_NICKNAMES = {
    "nick": "nicolas", "naz": "nazreon", "cam": "cameron", "tj": "terence",
    "cj": "carl", "jj": "james", "aj": "anthony", "rj": "randy",
    "pj": "paul", "og": "ogugua", "kj": "kevin", "zach": "zachary",
    "sue": "suzanne", "mo": "moritz",
}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[.'']", "", s)
    tokens = s.lower().split()
    tokens = [t for t in tokens if t not in _SUFFIXES]
    if tokens and tokens[0] in _NICKNAMES:
        tokens[0] = _NICKNAMES[tokens[0]]
    return " ".join(tokens)


def resolve_name(odds_name: str, player_index: dict) -> str | None:
    if odds_name in player_index:
        return odds_name
    n = _norm(odds_name)
    for k in player_index:
        if _norm(k) == n:
            return k
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    clf = reg = cal = trust = None
    try:
        with open(FILE_CLF, "rb") as f: clf = pickle.load(f)
        with open(FILE_REG, "rb") as f: reg = pickle.load(f)
        with open(FILE_CAL, "rb") as f: cal = pickle.load(f)
        with open(FILE_TRUST) as f:     trust = json.load(f)
        print(f"  ✓ V14 Adaptive Fusion models loaded ({len(trust):,} player trust scores)")
    except Exception as e:
        print(f"  ⚠ Model load error: {e}. Run `python3 run.py generate` first.")
    return clf, reg, cal, trust


# ─────────────────────────────────────────────────────────────────────────────
# FETCH PROPS — Excel primary, Odds API fallback
# ─────────────────────────────────────────────────────────────────────────────

def fetch_props_from_excel(date_str: str) -> list[dict]:
    """Read Tom's Excel file for today's props."""
    if not FILE_PROPS.exists():
        return []
    try:
        xl = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
        xl["Date"] = pd.to_datetime(xl["Date"]).dt.date
        today = pd.Timestamp(date_str).date()
        subset = xl[xl["Date"] == today].dropna(subset=["Line"])
        props = []
        for _, r in subset.iterrows():
            # Parse game time — strip 'ET' suffix so _tMin() in dashboard works
            raw_time = str(r.get("Game_Time_ET", "")).strip()
            game_time = raw_time.replace(" ET", "").strip() if raw_time and raw_time != "nan" else ""
            props.append({
                "player":    str(r["Player"]).strip(),
                "game":      str(r.get("Game", "")),
                "home":      str(r.get("Home", "")),
                "away":      str(r.get("Away", "")),
                "game_time": game_time,
                "line":      float(r["Line"]),
                "over_odds":  float(r["Over Odds"])  if pd.notna(r.get("Over Odds"))  else -110,
                "under_odds": float(r["Under Odds"]) if pd.notna(r.get("Under Odds")) else -110,
                "books":     int(r["Books"])          if pd.notna(r.get("Books"))      else 1,
                "min_line":  float(r["Min Line"])     if pd.notna(r.get("Min Line"))   else None,
                "max_line":  float(r["Max Line"])     if pd.notna(r.get("Max Line"))   else None,
                "source":    "excel",
            })
        print(f"  Excel: {len(props)} props for {date_str}")
        return props
    except Exception as e:
        print(f"  Excel read error: {e}")
        return []


def fetch_props_from_api(date_str: str) -> list[dict]:
    """Fetch from The Odds API (fallback)."""
    fr_utc, to_utc = et_window(date_str)
    fr_str = fr_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    to_str = to_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        events_url = f"{ODDS_BASE_URL}/sports/basketball_nba/events"
        r = requests.get(events_url, params={"apiKey": ODDS_API_KEY,
            "commenceTimeFrom": fr_str, "commenceTimeTo": to_str}, timeout=15)
        remaining = int(r.headers.get("x-requests-remaining", 9999))
        if remaining <= CREDIT_ALERT:
            print(f"  ⚠ LOW CREDITS: {remaining} remaining")
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        print(f"  API events error: {e}")
        return []

    props = []
    for event in events:
        eid  = event["id"]
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        game = f"{away} @ {home}"

        try:
            odds_url = f"{ODDS_BASE_URL}/sports/basketball_nba/events/{eid}/odds"
            r2 = requests.get(odds_url, params={
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": "player_points",
                "oddsFormat": "american",
            }, timeout=15)
            r2.raise_for_status()
            data = r2.json()
        except Exception as e:
            print(f"  Odds error for {game}: {e}")
            continue

        player_lines: dict[str, list] = {}
        player_odds:  dict[str, dict] = {}
        for bm in data.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "player_points":
                    continue
                for outcome in mkt.get("outcomes", []):
                    pname = str(outcome.get("description", outcome.get("name", ""))).strip()
                    point = outcome.get("point")
                    side  = str(outcome.get("name", "")).upper()
                    price = outcome.get("price")
                    if not pname or point is None:
                        continue
                    if pname not in player_lines:
                        player_lines[pname] = []
                        player_odds[pname]  = {"over": None, "under": None}
                    if side == "OVER":
                        player_lines[pname].append(float(point))
                        if player_odds[pname]["over"] is None:
                            player_odds[pname]["over"] = price
                    elif side == "UNDER":
                        if player_odds[pname]["under"] is None:
                            player_odds[pname]["under"] = price

        for pname, line_list in player_lines.items():
            if not line_list:
                continue
            avg_line = round(sum(line_list) / len(line_list) * 2) / 2
            od = player_odds.get(pname, {})
            props.append({
                "player":     pname,
                "game":       game,
                "home":       home,
                "away":       away,
                "line":       avg_line,
                "over_odds":  od.get("over")  or -110,
                "under_odds": od.get("under") or -110,
                "books":      len(line_list),
                "min_line":   min(line_list),
                "max_line":   max(line_list),
                "source":     "api",
            })

    print(f"  Odds API: {len(props)} props for {date_str}")
    return props


def append_to_excel(props: list[dict], date_str: str) -> None:
    """Append today's fetched props to the Excel source file."""
    if not props:
        return
    try:
        import openpyxl
        new_rows = []
        for p in props:
            new_rows.append({
                "Date": date_str, "Player": p["player"], "Game": p["game"],
                "Home": p["home"], "Away": p["away"], "Line": p["line"],
                "Over Odds": p["over_odds"], "Under Odds": p["under_odds"],
                "Books": p["books"], "Min Line": p["min_line"], "Max Line": p["max_line"],
            })
        new_df = pd.DataFrame(new_rows)

        if FILE_PROPS.exists():
            existing = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
            existing["Date"] = pd.to_datetime(existing["Date"]).dt.strftime("%Y-%m-%d")
            new_df["Date"] = date_str
            # Dedup: keep new over old for same (Date, Player, Game)
            def mk(df):
                return df["Date"].astype(str) + "|" + df["Player"].astype(str) + "|" + df["Game"].astype(str)
            kept = existing[~mk(existing).isin(mk(new_df))]
            combined = pd.concat([kept, new_df], ignore_index=True)
        else:
            combined = new_df

        with pd.ExcelWriter(FILE_PROPS, engine="openpyxl") as w:
            combined.to_excel(w, sheet_name="Player_Points_Props", index=False)
        print(f"  Excel updated: {len(combined)} total rows")
    except Exception as e:
        print(f"  Excel append error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ONE PLAY  (V14 Adaptive Fusion)
# ─────────────────────────────────────────────────────────────────────────────

def score_play(
    feats: dict,
    line: float,
    pos_raw: str,
    clf, reg, cal,
    trust: dict,
    player_name: str,
    date_str: str,
    h2h_row: dict,
    opponent: str,
) -> dict:
    """Apply V14 Adaptive Fusion engine. Returns scored play dict."""

    sp = season_progress(date_str)
    ew = feats.get("early_season_weight", 1.0)

    # ── Engine B: Regressor ────────────────────────────────────────────────
    pred_pts: float | None = None
    gap = 0.0
    gap_conf = 0.5
    if reg is not None:
        X = pd.DataFrame([feats])[ML_FEATURES].fillna(0).values
        pred_pts = float(reg.predict(X)[0])
        gap      = abs(pred_pts - line)
        gap_conf = min(0.90, max(0.45, 0.5 + gap * 0.04))
    reg_dir = 1 if (pred_pts or line) > line else 0   # 1=OVER, 0=UNDER

    # ── Engine C: Classifier ───────────────────────────────────────────────
    prob_over = 0.5
    cal_prob  = 0.5
    if clf is not None and cal is not None:
        X = pd.DataFrame([feats])[ML_FEATURES].fillna(0).values
        prob_over = float(clf.predict_proba(X)[0, 1])
        cal_prob  = float(cal.transform([prob_over])[0])
    clf_dir = 1 if prob_over > 0.5 else 0

    # ── Engine A: Composite ────────────────────────────────────────────────
    h2h_avg   = float(h2h_row.get("H2H_AVG_PTS", 0) or 0)
    h2h_games = int(h2h_row.get("H2H_GAMES", 0) or 0)
    temp_dir  = "OVER" if reg_dir == 1 else "UNDER"
    pos_grp   = get_pos_group(pos_raw)
    composite, flag_count, flag_details = compute_composite(
        feats, line, temp_dir, pos_grp,
        h2h_avg=h2h_avg, use_h2h=(h2h_games >= 3),
    )
    comp_conf = min(0.85, max(0.50, 0.5 + abs(composite) * 0.3))

    # ── V14 Adaptive Fusion ────────────────────────────────────────────────
    alpha, beta, gamma = fusion_weights(sp)
    fusion_conf = alpha * cal_prob + beta * gap_conf + gamma * comp_conf

    # Regime adjustments
    momentum  = feats.get("momentum", 0)
    extreme_h = feats.get("extreme_hot", 0)
    extreme_c = feats.get("extreme_cold", 0)
    if extreme_h and reg_dir == 1: fusion_conf -= 0.03
    if extreme_c and reg_dir == 0: fusion_conf -= 0.02
    if 3 < momentum <= 6 and reg_dir == 1:  fusion_conf += 0.015
    if -6 <= momentum < -3 and reg_dir == 0: fusion_conf += 0.015
    fusion_conf *= (0.70 + 0.30 * ew)          # early-season scaling
    if feats.get("is_long_rest", 0): fusion_conf -= 0.03
    if line >= HIGH_LINE_THRESHOLD and reg_dir == 1: fusion_conf -= 0.03
    if feats.get("line_sharpness", 0.5) > 0.80: fusion_conf += 0.01
    fusion_conf = max(0.40, min(0.90, fusion_conf))

    # ── Adaptive direction ─────────────────────────────────────────────────
    lo, hi = LEAN_ZONE
    engines_agree = (reg_dir == clf_dir)
    if cal_prob >= hi and engines_agree:
        direction = "OVER";  is_lean = False
    elif cal_prob <= lo and engines_agree:
        direction = "UNDER"; is_lean = False
    else:
        direction = "LEAN OVER" if cal_prob >= 0.50 else "LEAN UNDER"
        is_lean   = True

    # ── H2H alignment gate ────────────────────────────────────────────────
    h2h_ts = float(h2h_row.get("H2H_TS_VS_OVERALL", 0) or 0)
    h2h_aligned = True
    if h2h_games >= 3:
        if direction == "OVER"  and h2h_ts < -3: h2h_aligned = False
        if direction == "UNDER" and h2h_ts >  3: h2h_aligned = False

    # ── Tier assignment ────────────────────────────────────────────────────
    std10    = feats.get("std10", 5)
    vol_risk = feats.get("vol_risk", 0)
    high_vol = (std10 > 8) or (vol_risk > 1.5)
    fc = fusion_conf

    tier_label = "T3_LEAN" if is_lean else "T3"
    if not is_lean:
        if   fc >= 0.73 and gap >= 5.0 and std10 <= 6 and h2h_aligned and not high_vol: tier_label = "T1_ULTRA"
        elif fc >= 0.68 and gap >= 4.0 and std10 <= 7 and h2h_aligned and not high_vol: tier_label = "T1_PREMIUM"
        elif fc >= 0.63 and gap >= 3.0 and std10 <= 8 and h2h_aligned and not high_vol: tier_label = "T1"
        elif fc >= 0.56 and gap >= 2.0 and std10 <= 9 and h2h_aligned:                  tier_label = "T2"

    # Trust demotion
    player_trust = trust.get(player_name)
    if player_trust is not None and player_trust < TRUST_THRESHOLD and tier_label in ("T1_ULTRA","T1_PREMIUM","T1"):
        tier_label = "T2"

    units = UNIT_MAP.get(tier_label, 0.0)

    return {
        "direction": direction,
        "tierLabel":  tier_label,
        "tier":       1 if tier_label.startswith("T1") else (2 if tier_label == "T2" else 3),
        "conf":       round(fusion_conf, 4),
        "predPts":    round(pred_pts, 1) if pred_pts is not None else None,
        "predGap":    round(gap, 2),
        "calProb":    round(cal_prob, 4),
        "gapConf":    round(gap_conf, 4),
        "compConf":   round(comp_conf, 4),
        "composite":  round(composite, 4),
        "flags":      flag_count,
        "flagDetails": flag_details,
        "units":      units,
        "h2hAligned": h2h_aligned,
        "enginesAgree": engines_agree,
        "earlySeasonW": round(ew, 3),
        "seasonProgress": round(sp, 3),
        "meanReversionRisk": feats.get("mean_reversion_risk", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / MERGE today.json
# ─────────────────────────────────────────────────────────────────────────────

def save_today(plays: list[dict]) -> None:
    """
    Merge new predictions into today.json.
    Rules:
    - Only keep plays from today's date (the batch date).
    - Graded plays (WIN/LOSS/DNP) are immutable — never overwritten.
    - Ungraded plays from a prior date are discarded (stale).
    - DNP plays from a prior date are also discarded — they may be incorrect
      from a failed API call and should not persist across days.
    """
    existing: list[dict] = []
    if FILE_TODAY.exists():
        try:
            with open(FILE_TODAY) as f:
                existing = json.load(f)
        except Exception:
            existing = []

    # Current batch date
    batch_date = plays[0].get("date", "") if plays else ""

    def key(p): return (p.get("player",""), p.get("date",""), str(p.get("line","")))

    # Preserve graded plays from TODAY — WIN/LOSS regardless of direction (inc. LEAN props).
    # DNP excluded: those come from the live game and will be re-graded by batch0.
    # LEAN (direction only, not result) plays with result="" are NOT preserved here
    # because they are ungraded — batch_predict will re-score them with fresh models.
    graded = {
        key(p): p for p in existing
        if p.get("result") in ("WIN", "LOSS")
        and p.get("date") == batch_date
    }

    merged = list(graded.values())
    for p in plays:
        k = key(p)
        if k in graded:
            continue  # immutable
        # Stitch lineHistory from prior ungraded for same date
        old = next(
            (e for e in existing
             if key(e) == k and e.get("date") == batch_date),
            None
        )
        if old:
            p["lineHistory"] = old.get("lineHistory", [])
            if not any(h.get("batch") == BATCH for h in p["lineHistory"]):
                p["lineHistory"].append(
                    {"line": p["line"], "batch": BATCH, "ts": p.get("batchTs","")}
                )
        else:
            p["lineHistory"] = [
                {"line": p["line"], "batch": BATCH, "ts": p.get("batchTs","")}
            ]
        merged.append(p)

    merged.sort(key=lambda p: (p.get("tier", 9), -p.get("conf", 0)))
    with open(FILE_TODAY, "w") as f:
        json.dump(clean_json(merged), f, indent=2)
    print(f"  today.json: {len(merged)} plays saved "
          f"({len(graded)} graded from today preserved)")


def append_season_json(plays: list[dict]) -> None:
    """Append / update plays in season_2025_26.json."""
    existing: list[dict] = []
    if FILE_SEASON_2526.exists():
        try:
            with open(FILE_SEASON_2526) as f:
                existing = json.load(f)
        except Exception:
            existing = []

    def key(p): return (p.get("player",""), p.get("date",""), str(p.get("line","")))
    ex_map = {key(p): i for i, p in enumerate(existing)}

    for p in plays:
        k = key(p)
        if k in ex_map:
            old = existing[ex_map[k]]
            if old.get("result") in ("WIN","LOSS","DNP"):
                continue   # immutable
            existing[ex_map[k]] = p
        else:
            existing.append(p)

    with open(FILE_SEASON_2526, "w") as f:
        json.dump(clean_json(existing), f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# GIT PUSH
# ─────────────────────────────────────────────────────────────────────────────

def git_push(message: str) -> None:
    import subprocess, os
    repo = REPO_DIR if REPO_DIR.exists() else Path(__file__).parent
    # Force SSH — never HTTPS — so no credential prompts
    env = {
        **os.environ,
        "GIT_SSH_COMMAND": "ssh -o BatchMode=yes -o StrictHostKeyChecking=no",
    }
    try:
        # Ensure remote is SSH URL (not HTTPS)
        subprocess.run(
            ["git", "-C", str(repo), "remote", "set-url", "origin", GIT_REMOTE],
            capture_output=True, env=env,
        )
        subprocess.run(
            ["git", "-C", str(repo), "add", "-A"],
            capture_output=True, env=env,
        )
        result = subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", message],
            capture_output=True, text=True, env=env,
        )
        if "nothing to commit" in (result.stdout + result.stderr):
            print("  Git: nothing to commit."); return
        push = subprocess.run(
            ["git", "-C", str(repo), "push", "--set-upstream", "origin", "main"],
            capture_output=True, text=True, timeout=60, env=env,
        )
        if push.returncode == 0:
            print("  Git push ✓")
        else:
            print(f"  ⚠ Git push failed: {push.stderr.strip()[:120]}")
    except subprocess.TimeoutExpired:
        print("  ⚠ Git push timed out — local files are correct.")
    except Exception as e:
        print(f"  ⚠ Git push error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# RECENT20 BUILDER  (sparkline + score pills with home/away flags)
# ─────────────────────────────────────────────────────────────────────────────

def _build_recent20(prior: pd.DataFrame, line: float) -> list[dict]:
    """
    Extract last 20 played games from prior history for dashboard sparkline.
    Returns list of {date, pts, home, opponent, overLine} objects, oldest first.
    Matches the format written by generate_season_json._build_play().
    """
    if prior is None or len(prior) == 0:
        return []
    tail = prior.tail(20)
    result = []
    for _, r in tail.iterrows():
        pts = float(r.get("PTS", 0) or 0)
        result.append({
            "date":     pd.Timestamp(r["GAME_DATE"]).strftime("%Y-%m-%d"),
            "pts":      pts,
            "home":     bool(int(r.get("IS_HOME", 0) or 0)),
            "opponent": str(r.get("OPPONENT", "") or ""),
            "overLine": pts > line,
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from datetime import datetime as dt
    date_str  = today_et()
    batch_ts  = dt.now(timezone.utc).isoformat()

    print(f"\n{'='*60}")
    print(f"  PropEdge {VERSION} — Batch {BATCH}  |  {date_str} UK")
    print(f"{'='*60}")
    log_event(f"B{BATCH}", "BATCH_START", detail=f"date={date_str}")

    # ── Update DVP ────────────────────────────────────────────────────────────
    compute_and_save_dvp()

    # ── Load game logs ────────────────────────────────────────────────────────
    print("  Loading game logs...")
    gl24 = pd.read_csv(FILE_GL_2425, parse_dates=["GAME_DATE"])
    gl25 = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    gl = pd.concat([gl24, gl25], ignore_index=True)
    gl["DNP"] = gl["DNP"].fillna(0)
    played = filter_played(gl)
    played = played.sort_values(["PLAYER_NAME","GAME_DATE"])

    player_idx = {
        pname: grp.sort_values("GAME_DATE").reset_index(drop=True)
        for pname, grp in played.groupby("PLAYER_NAME")
    }

    # ── Build caches ──────────────────────────────────────────────────────────
    dvp_dict: dict = {}
    for (opp, pos), g in played.groupby(["OPPONENT","PLAYER_POSITION"]):
        dvp_dict[(opp, pos)] = g["PTS"].mean()
    dvp_rank: dict = {}
    for pos in played["PLAYER_POSITION"].unique():
        subset = {k:v for k,v in dvp_dict.items() if k[1]==pos}
        for rank,(opp,p) in enumerate(sorted(subset,key=lambda k:subset[k],reverse=True),1):
            dvp_rank[(opp,p)] = rank

    team_fga = played.groupby("OPPONENT")["FGA"].mean()
    pace_cache = {t: i+1 for i,(t,_) in enumerate(team_fga.sort_values(ascending=False).items())}

    # B2B map
    b2b_map: dict = {}
    for pname, grp in played.groupby("PLAYER_NAME"):
        dates = grp["GAME_DATE"].values
        for i, d in enumerate(dates):
            rd = int((d - dates[i-1]).astype("timedelta64[D]").astype(int)) if i > 0 else 99
            b2b_map[(pname, pd.Timestamp(d).strftime("%Y-%m-%d"))] = rd

    # H2H lookup
    h2h_df = pd.read_csv(FILE_H2H)
    h2h_df = h2h_df.drop_duplicates(subset=["PLAYER_NAME","OPPONENT"], keep="last")
    h2h_lkp = {
        (r["PLAYER_NAME"], r["OPPONENT"]): r.to_dict()
        for _, r in h2h_df.iterrows()
    }

    # ── Load models ───────────────────────────────────────────────────────────
    clf, reg, cal, trust = load_models()

    # ── Fetch props ───────────────────────────────────────────────────────────
    props = fetch_props_from_excel(date_str)
    if not props:
        print("  Excel empty — falling back to Odds API...")
        props = fetch_props_from_api(date_str)
    if not props:
        print("  No props found. Exiting."); return

    if [p for p in props if p.get("source") == "api"]:
        append_to_excel(props, date_str)

    log_event(f"B{BATCH}", "PROPS_FETCHED", detail=f"{len(props)} props")

    # ── Score each player ─────────────────────────────────────────────────────
    plays = []
    skipped: dict[str, int] = {"no_player": 0, "no_history": 0, "no_feats": 0}

    for prop in props:
        pname    = prop["player"]
        line     = prop["line"]
        game     = prop.get("game", "")
        home     = prop.get("home", "")
        away     = prop.get("away", "")

        resolved = resolve_name(pname, player_idx)
        if resolved is None:
            skipped["no_player"] += 1; continue

        hist = player_idx[resolved]
        prior = hist[hist["GAME_DATE"] < pd.Timestamp(date_str)]
        if len(prior) < 5:
            skipped["no_history"] += 1; continue

        pos = str(prior["PLAYER_POSITION"].iloc[-1])
        # Determine opponent from game string
        opp_team = ""
        if home and away:
            my_team = prior["GAME_TEAM_ABBREVIATION"].iloc[-1] if "GAME_TEAM_ABBREVIATION" in prior.columns else ""
            # if player's team matches away → opponent is home; and vice versa
            if my_team:
                opp_team = home if my_team.upper() == away.upper() else away
            else:
                opp_team = away  # default: treat as away opponent

        rest_days = b2b_map.get((resolved, date_str), 99)

        feats = extract_prediction_features(
            prior_played=prior,
            line=line,
            opponent=opp_team,
            rest_days=rest_days,
            pos_raw=pos,
            game_date=pd.Timestamp(date_str),
            min_line=prop.get("min_line"),
            max_line=prop.get("max_line"),
            dvp_rank_cache={(opp_team, get_pos_group(pos)): dvp_rank.get((opp_team, pos), 15)},
            pace_rank_cache=pace_cache,
        )
        if feats is None:
            skipped["no_feats"] += 1; continue

        h2h_row = h2h_lkp.get((resolved, opp_team), {})
        feats["h2h_ts_dev"]  = float(h2h_row.get("H2H_TS_VS_OVERALL",  0) or 0)
        feats["h2h_fga_dev"] = float(h2h_row.get("H2H_FGA_VS_OVERALL", 0) or 0)
        feats["h2h_min_dev"] = float(h2h_row.get("H2H_MIN_VS_OVERALL", 0) or 0)
        feats["h2h_conf"]    = float(h2h_row.get("H2H_CONFIDENCE",     0) or 0)
        feats["h2h_games"]   = float(h2h_row.get("H2H_GAMES",          0) or 0)
        feats["h2h_trend"]   = float(h2h_row.get("H2H_PTS_TREND",      0) or 0)

        scored = score_play(feats, line, pos, clf, reg, cal, trust or {}, resolved, date_str, h2h_row, opp_team)

        play = {
            "player":     resolved,
            "date":       date_str,
            "game":       game,
            "home":       home,
            "away":       away,
            "opponent":   opp_team,
            "line":       line,
            "overOdds":   prop.get("over_odds", -110),
            "underOdds":  prop.get("under_odds", -110),
            "books":      prop.get("books", 1),
            "position":   pos,
            "batchTs":    batch_ts,
            **scored,
            # Display rolling fields
            "l30": round(feats.get("_l30", 0), 1),
            "l10": round(feats.get("_l10", 0), 1),
            "l5":  round(feats.get("_l5",  0), 1),
            "l3":  round(feats.get("_l3",  0), 1),
            "std10":  round(feats.get("std10", 0), 2),
            "hr10":   round(feats.get("hr10",  0), 3),
            "hr30":   round(feats.get("hr30",  0), 3),
            "min_l10": round(feats.get("min_l10", 0), 1),
            "fga_l10": round(feats.get("fga_l10", 0), 1),
            "defP_dynamic": feats.get("defP_dynamic", 15),
            "pace_rank":    feats.get("pace_rank", 15),
            "h2h_avg":      float(h2h_row.get("H2H_AVG_PTS", 0) or 0),
            "h2h_games":    int(h2h_row.get("H2H_GAMES", 0) or 0),
            "h2h_ts_dev":   float(feats.get("h2h_ts_dev",  0) or 0),
            "h2h_fga_dev":  float(feats.get("h2h_fga_dev", 0) or 0),
            "h2h_conf":     float(feats.get("h2h_conf",    0) or 0),
            "game_time":    prop.get("game_time", ""),
            "home_l10":     round(feats.get("home_l10", 0), 1),
            "away_l10":     round(feats.get("away_l10", 0), 1),
            "min_l30":      round(feats.get("min_l30", 0), 1),
            "fga_l10":      round(feats.get("fga_l10", 0), 1),
            "vol_risk":     round(feats.get("vol_risk", 0), 3),
            # Decision signal fields (for dashboard signals + trends)
            "usage_l10":      round(feats.get("usage_l10", 0), 2),
            "usage_l30":      round(feats.get("usage_l30", 0), 2),
            "min_cv":         round(feats.get("min_cv", 0), 3),
            "fta_l10":        round(feats.get("fta_l10", 0), 1),
            "ft_rate":        round(feats.get("ft_rate", 0), 3),
            "fg3a_l10":       round(feats.get("fg3a_l10", 0), 1),
            "pts_per_min":    round(feats.get("pts_per_min", 0), 3),
            "home_away_split": round(feats.get("home_away_split", 0), 1),
            "level_ewm":      round(feats.get("level_ewm", 0), 1),
            "line_vs_l30":    round(feats.get("line_vs_l30", 0), 1),
            "extreme_hot":    int(feats.get("extreme_hot", 0) or 0),
            "extreme_cold":   int(feats.get("extreme_cold", 0) or 0),
            "is_long_rest":   int(feats.get("is_long_rest", 0) or 0),
            "recent_min_trend": round(feats.get("recent_min_trend", 0), 1),
            "h2h_trend":      round(feats.get("h2h_trend", 0), 1),
            # Recent 20 scores — needed by dashboard for sparkline + score pills with home/away flags
            "recent20": _build_recent20(prior, line),
        }

        play["preMatchReason"] = generate_pre_match_reason(play)
        plays.append(play)

    print(f"  Scored: {len(plays)} plays | Skipped: {skipped}")
    t1_plays = [p for p in plays if p.get("tier") == 1]
    t2_plays = [p for p in plays if p.get("tierLabel") == "T2"]
    print(f"  T1: {len(t1_plays)}  T2: {len(t2_plays)}")

    log_event(f"B{BATCH}", "PREDICTIONS", detail=f"{len(plays)} plays, skipped={skipped}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_today(plays)
    append_season_json(plays)
    git_push(f"V17 B{BATCH}: {date_str} — {len(plays)} plays")

    log_event(f"B{BATCH}", "BATCH_COMPLETE", detail=f"{len(plays)} plays written")
    print(f"\n  ✓ Batch {BATCH} complete — {len(plays)} plays.\n")


if __name__ == "__main__":
    main()
