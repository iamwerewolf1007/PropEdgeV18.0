"""
PropEdge V18.0 — injury_ingest.py
NBA injury / availability ingestion pipeline.

Data sources (both proven UK-compatible):
  1. cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json
     -> Today's games + game IDs (public CDN, not geo-blocked)
  2. nba_api BoxScoreTraditionalV3 per game_id (same call V14 uses for grading)
     -> Returns full roster with INACTIVE status pre-game, DNP=0min post-game
  3. nba_api ScoreboardV3 fallback for game IDs (V14-proven UK-compatible)

Why not stats.nba.com or NBA CMS PDFs:
  - stats.nba.com is geo-blocked from UK
  - NBA CMS PDFs only exist during season on game days
  - cdn.nba.com is the public NBA app CDN — not geo-blocked anywhere

Status model:
  INACTIVE -> player confirmed not playing (in inactives list)
  ACTIVE   -> player confirmed in game roster
  UNKNOWN  -> player not found in any game today

CLI:
  python3 injury_ingest.py           -- fetch + update
  python3 injury_ingest.py --status  -- show current state (no fetch)
  python3 injury_ingest.py --date 2026-10-22 -- specific date
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    FILE_INJURIES_CURRENT, FILE_INJURY_HISTORY, FILE_INJURY_MANIFEST,
    today_et,
)
from audit import log_event

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NBA_CDN_SCOREBOARD = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"

import random as _random
_UA_POOL = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

def _cdn_headers() -> dict:
    return {
        "User-Agent":      _random.choice(_UA_POOL),
        "Accept":          "application/json, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.nba.com/",
        "Origin":          "https://www.nba.com",
    }

HISTORY_COLUMNS = [
    "fetch_ts", "game_date", "game_id", "matchup",
    "team", "opponent",
    "player_name", "player_name_normalized",
    "status_normalized", "status_raw",
    "source_url", "source_hash",
    "first_seen_ts", "last_seen_ts", "status_changed",
]

# ─────────────────────────────────────────────────────────────────────────────
# NAME NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

_SUFFIX_RE = re.compile(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", re.IGNORECASE)
_PUNCT_RE  = re.compile(r"[.\-']")

def normalize_player_name(raw: str) -> str:
    if not raw or str(raw).strip().lower() in ("", "nan", "none"):
        return ""
    name = str(raw).strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = _SUFFIX_RE.sub("", name).strip()
    name = name.lower()
    name = _PUNCT_RE.sub("", name)
    return " ".join(name.split())


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: NBA CDN Scoreboard -> game IDs
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_today_games_cdn() -> list[dict]:
    """Fetch today's games from NBA public CDN. Not geo-blocked."""
    try:
        r = requests.get(NBA_CDN_SCOREBOARD, headers=_cdn_headers(), timeout=15)
        r.raise_for_status()
        data  = r.json()
        games = data.get("scoreboard", {}).get("games", [])
        result = []
        for g in games:
            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            result.append({
                "game_id":   g.get("gameId", ""),
                "home_team": home.get("teamTricode", ""),
                "away_team": away.get("teamTricode", ""),
                "matchup":   f"{away.get('teamTricode','')} @ {home.get('teamTricode','')}",
                "status":    g.get("gameStatusText", ""),
            })
        print(f"  [injury] CDN scoreboard: {len(result)} games today")
        return result
    except Exception as e:
        print(f"  [injury] CDN scoreboard: {type(e).__name__}: {str(e)[:80]}")
        return []


def _fetch_today_games_nba_api(date_str: str) -> list[dict]:
    """Fallback: nba_api ScoreboardV3 -- same call V14 uses for grading."""
    try:
        from nba_api.stats.endpoints import ScoreboardV3
        time.sleep(1)
        sb = ScoreboardV3(game_date=date_str, league_id="00")
        gh = sb.game_header.get_data_frame()
        if gh.empty:
            return []
        result = []
        for _, row in gh.iterrows():
            home = str(row.get("HOME_TEAM_ABBREVIATION", ""))
            away = str(row.get("VISITOR_TEAM_ABBREVIATION", ""))
            result.append({
                "game_id":   str(row.get("GAME_ID", "")),
                "home_team": home,
                "away_team": away,
                "matchup":   f"{away} @ {home}",
                "status":    str(row.get("GAME_STATUS_TEXT", "")),
            })
        print(f"  [injury] nba_api ScoreboardV3: {len(result)} games")
        return result
    except Exception as e:
        print(f"  [injury] ScoreboardV3: {type(e).__name__}: {str(e)[:80]}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: NBA CDN Boxscore -> inactives per game
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_game_inactives(game: dict, fetch_ts: str) -> list[dict]:
    """
    Fetch inactives for one game using nba_api BoxScoreTraditionalV3.
    This is the same call V14 uses for grading — proven UK-compatible.
    Pre-game: returns all rostered players; inactives have status="INACTIVE"
              or are listed in the inactive sub-list.
    Post-game: returns players with minutes=0 as DNP (also treated as inactive).
    """
    game_id  = game["game_id"]
    home     = game["home_team"]
    away     = game["away_team"]
    matchup  = game["matchup"]
    source_url  = f"nba_api://BoxScoreTraditionalV3/{game_id}"
    source_hash = hashlib.sha256(game_id.encode()).hexdigest()[:16]

    try:
        from nba_api.stats.endpoints import BoxScoreTraditionalV3
        time.sleep(0.8)   # V14 uses 0.8s between box calls
        box = BoxScoreTraditionalV3(game_id=game_id)
        ps  = box.player_stats.get_data_frame()

        if ps is None or ps.empty:
            print(f"  [injury]   {matchup}: no data yet (pre-game roster not loaded)")
            return []

        rows = []
        for _, row in ps.iterrows():
            # Build player name from firstName + familyName columns
            fn    = str(row.get("firstName",   row.get("FN",  ""))).strip()
            ln    = str(row.get("familyName",  row.get("LN",  ""))).strip()
            pname = f"{fn} {ln}".strip()
            if not pname:
                continue

            team_code  = str(row.get("teamTricode",       row.get("TEAM_ABBREVIATION", ""))).strip()
            opp        = away if team_code == home else home
            status_raw = str(row.get("status", "")).strip()

            # Determine active/inactive
            # "INACTIVE" status = not in game
            # minutes=0 post-game = DNP (also inactive for prop purposes)
            from rolling_engine import _parse_min
            mins = _parse_min(row.get("minutes", row.get("MR", 0)))

            if status_raw.upper() == "INACTIVE":
                status_norm = "INACTIVE"
            elif mins <= 0 and status_raw.upper() not in ("", "ACTIVE"):
                status_norm = "INACTIVE"
            else:
                status_norm = "ACTIVE"

            rows.append(_make_row(pname, team_code, opp, matchup, game_id,
                                  status_raw or status_norm, status_norm,
                                  source_url, source_hash, fetch_ts))

        inactive_ct = sum(1 for r in rows if r["status_normalized"] == "INACTIVE")
        print(f"  [injury]   {matchup}: {len(rows)} players, {inactive_ct} inactive")
        return rows

    except Exception as e:
        print(f"  [injury]   {matchup} ({game_id}): {type(e).__name__}: {str(e)[:80]}")
        return []


def _make_row(player_name, team, opponent, matchup, game_id,
              status_raw, status_normalized, source_url, source_hash, fetch_ts) -> dict:
    return {
        "fetch_ts":               fetch_ts,
        "game_date":              today_et(),
        "game_id":                game_id,
        "matchup":                matchup,
        "team":                   team,
        "opponent":               opponent,
        "player_name":            player_name,
        "player_name_normalized": normalize_player_name(player_name),
        "status_normalized":      status_normalized,
        "status_raw":             status_raw,
        "source_url":             source_url,
        "source_hash":            source_hash,
        "first_seen_ts":          fetch_ts,
        "last_seen_ts":           fetch_ts,
        "status_changed":         False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _load_current() -> dict:
    if not FILE_INJURIES_CURRENT.exists():
        return {}
    try:
        data = json.loads(FILE_INJURIES_CURRENT.read_text())
        if isinstance(data, dict) and "players" in data:
            return {p["player_name_normalized"]: p
                    for p in data["players"] if p.get("player_name_normalized")}
        return data
    except Exception:
        return {}


def _save_current(state: dict) -> None:
    FILE_INJURIES_CURRENT.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "n_players":    len(state),
        "players":      list(state.values()),
    }
    FILE_INJURIES_CURRENT.write_text(json.dumps(out, indent=2, default=str))


def _append_history(rows: list[dict]) -> None:
    FILE_INJURY_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FILE_INJURY_HISTORY.exists()
    with open(FILE_INJURY_HISTORY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)


def _load_manifest() -> list:
    if not FILE_INJURY_MANIFEST.exists():
        return []
    try:
        return json.loads(FILE_INJURY_MANIFEST.read_text())
    except Exception:
        return []


def _save_manifest(m: list) -> None:
    FILE_INJURY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    FILE_INJURY_MANIFEST.write_text(json.dumps(m[-500:], indent=2, default=str))


def _content_hash(rows: list[dict]) -> str:
    key = json.dumps(
        sorted([(r.get("player_name",""), r.get("status_normalized","")) for r in rows]),
        sort_keys=True
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _detect_changes(new_rows: list[dict], old_state: dict) -> tuple[list[dict], list[str]]:
    changes = []
    for row in new_rows:
        key  = row.get("player_name_normalized", "")
        prev = old_state.get(key)
        if prev and prev.get("status_normalized") != row.get("status_normalized"):
            row["status_changed"] = True
            row["first_seen_ts"]  = prev.get("first_seen_ts", row["fetch_ts"])
            changes.append(
                f"{row['player_name']} ({row['team']}): "
                f"{prev['status_normalized']} -> {row['status_normalized']}"
            )
        elif prev:
            row["first_seen_ts"] = prev.get("first_seen_ts", row["fetch_ts"])
    return new_rows, changes


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_store(date_str: Optional[str] = None) -> dict:
    """
    Main entry point. Fetch today's NBA inactives and update output files.
    Uses NBA CDN (public, not geo-blocked) with nba_api as fallback.
    """
    date_str  = date_str or today_et()
    fetch_ts  = datetime.now(timezone.utc).isoformat()
    old_state = _load_current()
    manifest  = _load_manifest()

    log_event("INJ", "FETCH_START", detail=f"date={date_str}")

    # Step 1: Get today's games
    print(f"  [injury] Fetching games for {date_str}...")
    games = _fetch_today_games_cdn()

    if not games:
        print("  [injury] CDN failed -- trying nba_api ScoreboardV3...")
        games = _fetch_today_games_nba_api(date_str)

    if not games:
        print(f"  [injury] No games found for {date_str}")
        print(f"  [injury] Note: NBA only publishes data on game days during the season")
        log_event("INJ", "NO_GAMES", detail=f"date={date_str}")
        return {"n_rows": len(old_state), "n_changed": 0, "status": "NO_GAMES",
                "source": "none", "rows": len(old_state), "changes": [],
                "cached": True, "error": False}

    # Step 2: Fetch inactives per game
    all_rows: list[dict] = []
    for game in games:
        time.sleep(0.5)
        rows = _fetch_game_inactives(game, fetch_ts)
        all_rows.extend(rows)

    if not all_rows:
        print(f"  [injury] Got {len(games)} games but 0 player rows")
        print(f"  [injury] BoxScoreTraditionalV3 returned no rows — rosters may not be loaded yet (typically available ~1hr before tip)")
        return {"n_rows": len(old_state), "n_changed": 0, "status": "NOT_READY",
                "source": "cdn", "rows": len(old_state), "changes": [],
                "cached": True, "error": False}

    # Step 3: Dedup
    content_hash = _content_hash(all_rows)
    if manifest and manifest[-1].get("source_hash") == content_hash:
        print(f"  [injury] No change since last fetch ({len(all_rows)} players)")
        return {"n_rows": len(all_rows), "n_changed": 0, "status": "UNCHANGED",
                "source": "cdn", "rows": len(all_rows), "changes": [],
                "cached": True, "error": False}

    # Step 4: Change detection
    all_rows, changes = _detect_changes(all_rows, old_state)

    # Step 5: Build new state (prefer INACTIVE over ACTIVE on dedup)
    new_state: dict = {}
    for row in all_rows:
        key = row.get("player_name_normalized", "")
        if not key:
            continue
        existing = new_state.get(key)
        if existing is None or row["status_normalized"] == "INACTIVE":
            new_state[key] = row

    # Step 6: Save
    _save_current(new_state)
    _append_history(all_rows)
    manifest.append({
        "fetch_ts":    fetch_ts,
        "game_date":   date_str,
        "n_games":     len(games),
        "n_players":   len(new_state),
        "source_hash": content_hash,
        "n_changed":   len(changes),
        "status":      "ok",
    })
    _save_manifest(manifest)

    inactive_ct = sum(1 for v in new_state.values() if v["status_normalized"] == "INACTIVE")
    active_ct   = sum(1 for v in new_state.values() if v["status_normalized"] == "ACTIVE")

    if changes:
        for c in changes[:5]:
            print(f"    STATUS CHANGE: {c}")

    log_event("INJ", "FETCH_OK",
              detail=f"players={len(new_state)} inactive={inactive_ct} changes={len(changes)}")
    print(f"  [injury] Done: {len(new_state)} players | INACTIVE={inactive_ct} ACTIVE={active_ct} | {len(changes)} changes")

    return {"n_rows": len(new_state), "n_changed": len(changes), "status": "OK",
            "source": "cdn", "rows": len(new_state), "changes": changes,
            "cached": False, "error": False}


# ─────────────────────────────────────────────────────────────────────────────
# QUERY INTERFACE (used by batch_predict and reasoning_engine)
# ─────────────────────────────────────────────────────────────────────────────

def load_injury_state(date_str: Optional[str] = None) -> dict:
    """Load current state. Fetches if no cached state exists."""
    state = _load_current()
    if not state:
        print("  [injury] No cached state -- fetching...")
        fetch_and_store(date_str)
        state = _load_current()
    return state


# Alias
get_current_injuries = load_injury_state


def get_team_injury_summary(injury_state: dict, team: str) -> dict:
    """Team-level availability summary for reasoning engine."""
    team_rows = [v for v in injury_state.values()
                 if v.get("team", "").upper() == team.upper()]
    inactive  = [v["player_name"] for v in team_rows if v.get("status_normalized") == "INACTIVE"]
    n = len(inactive)
    return {
        "n_inactive": n, "inactive_players": inactive,
        "has_star_out": n > 0,
        "risk_level": "HIGH" if n >= 2 else "MEDIUM" if n == 1 else "NONE",
        # Backwards-compat aliases for reasoning_engine
        "n_out": n, "n_questionable": 0, "n_doubtful": 0,
        "out_players": inactive, "questionable_players": [],
    }


def get_teammate_load_boost(
    injury_state: dict,
    home_team: str,
    away_team: str,
    player_name_norm: str,
    usage_threshold: float = 0.25,
) -> float:
    """Return pts boost if a teammate is INACTIVE on the same team."""
    from config import TEAMMATE_BOOST_MAGNITUDE
    player_entry = injury_state.get(player_name_norm)
    if not player_entry:
        return 0.0
    my_team = player_entry.get("team", "")
    if not my_team:
        return 0.0
    inactive_teammates = [
        v for v in injury_state.values()
        if v.get("team", "").upper() == my_team.upper()
        and v.get("status_normalized") == "INACTIVE"
        and v.get("player_name_normalized", "") != player_name_norm
    ]
    return float(TEAMMATE_BOOST_MAGNITUDE) if inactive_teammates else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PropEdge V18 -- Injury Ingest")
    parser.add_argument("--status", action="store_true", help="Show state, no fetch")
    parser.add_argument("--date",   type=str, default=None)
    args = parser.parse_args()

    if args.status:
        state    = load_injury_state(args.date)
        inactive = [v for v in state.values() if v.get("status_normalized") == "INACTIVE"]
        active   = [v for v in state.values() if v.get("status_normalized") == "ACTIVE"]
        print(f"\n  NBA Inactives -- {args.date or today_et()}")
        print(f"  {'--'*27}")
        print(f"  INACTIVE ({len(inactive)}) -- will not play:")
        for v in inactive[:20]:
            print(f"    {v.get('player_name',''):<25} {v.get('team',''):<5} vs {v.get('opponent','')}")
        print(f"  ACTIVE: {len(active)} confirmed playing")
        print(f"  Total tracked: {len(state)}")
    else:
        result = fetch_and_store(date_str=args.date)
        print(f"\n  Result: {result['status']} | {result['n_rows']} players | {result['n_changed']} changes")


if __name__ == "__main__":
    main()
