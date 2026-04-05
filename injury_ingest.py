"""
PropEdge V18.0 — injury_ingest.py
NBA Official Injury Report ingestion pipeline.

Architecture:
  Primary source: nba_api.LeagueInjuryReports (structured JSON — no PDF parsing needed)
  Secondary source: NBA CMS PDF (ak-static.cms.nba.com) — for audit + timestamp
  Outputs (flat files — no DB):
    data/injuries_current.json      — latest state per player (keyed by normalized name)
    data/injury_report_history.csv  — append-only history of every row from every report
    data/injury_report_manifest.json — report-level metadata (URL, hash, timestamps)

Point-in-time integrity:
  Every row carries report_ts. Downstream jobs query:
    get_injury_status(player, as_of_ts) → status at that moment.
  This prevents leakage in backtests.

Name matching:
  Uses PropEdge normalize_name() — compatible with game log CSV names.
  Handles Jr./Sr./II/III suffixes, punctuation (P.J.→PJ), accents.

Status buckets:
  OUT, QUESTIONABLE, DOUBTFUL, PROBABLE, AVAILABLE, NOT_YET_SUBMITTED, UNKNOWN

CLI:
  python3 injury_ingest.py fetch          — fetch latest report, update files
  python3 injury_ingest.py fetch --date YYYY-MM-DD  — fetch for specific date
  python3 injury_ingest.py status         — show current injury state
  python3 injury_ingest.py changes        — show status changes from last report
  python3 injury_ingest.py history PLAYER — show history for a player
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FILE_INJURIES_CURRENT, FILE_INJURY_HISTORY, FILE_INJURY_MANIFEST,
    FILE_AUDIT, ODDS_API_KEY, get_uk,
)
from audit import log_event

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NBA_INJURY_API = "https://stats.nba.com/stats/leagueinjuries"
NBA_INJURY_PAGE = "https://www.nba.com/players/injury-report"
NBA_CMS_PDF_BASE = "https://ak-static.cms.nba.com/wp-content/uploads/injury-report"

_NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

# Status normalisation map — raw NBA text → bucket
_STATUS_MAP = {
    "out":                  "OUT",
    "doubtful":             "DOUBTFUL",
    "questionable":         "QUESTIONABLE",
    "probable":             "PROBABLE",
    "available":            "AVAILABLE",
    "not yet submitted":    "NOT_YET_SUBMITTED",
    "gtd":                  "QUESTIONABLE",   # game-time decision = questionable
    "game time decision":   "QUESTIONABLE",
    "day to day":           "QUESTIONABLE",
    "inactive":             "OUT",
    "dnp":                  "OUT",
    "will not play":        "OUT",
}

# Reason categories (mapped from reason keywords)
_REASON_CATS = [
    ("rest",         ["rest","load management","maintenance"]),
    ("knee",         ["knee","acl","mcl","meniscus","patellar"]),
    ("ankle",        ["ankle","achilles"]),
    ("hamstring",    ["hamstring","quad","quadricep"]),
    ("back",         ["back","lumbar","spine","spinal"]),
    ("shoulder",     ["shoulder","rotator"]),
    ("foot",         ["foot","plantar","toe","heel"]),
    ("hand_wrist",   ["hand","wrist","finger","thumb"]),
    ("illness",      ["illness","flu","covid","cold","personal","non-covid"]),
    ("hip_groin",    ["hip","groin","pelvis"]),
    ("concussion",   ["concussion","head","neck"]),
    ("suspension",   ["suspension","suspended"]),
    ("personal",     ["personal","family"]),
    ("not_submitted",["not yet submitted"]),
]


# ─────────────────────────────────────────────────────────────────────────────
# NAME NORMALISATION (V18 extended — compatible with game log CSV names)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_player_name(raw: str) -> str:
    """
    Canonical join key compatible with PropEdge game log CSV player names.
    Handles: accents, Jr/Sr/II/III suffixes, punctuation (P.J.→PJ),
             extra spaces, case.
    """
    if not raw:
        return ""
    # Unicode normalisation → ASCII
    s = unicodedata.normalize("NFKD", raw)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Remove periods inside initials (P.J. → PJ, T.J. → TJ)
    s = re.sub(r"(?<=[A-Z])\.(?=[A-Z])", "", s)
    s = re.sub(r"(?<=\s[A-Z])\.(?=\s|$)", "", s)
    # Remove common suffixes (Jr., Sr., II, III, IV)
    s = re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV)$", "", s, flags=re.IGNORECASE)
    # Collapse multiple spaces, strip
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


# ─────────────────────────────────────────────────────────────────────────────
# STATUS / REASON NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_status(raw: str) -> str:
    if not raw:
        return "UNKNOWN"
    key = raw.strip().lower()
    for pattern, bucket in _STATUS_MAP.items():
        if pattern in key:
            return bucket
    return "UNKNOWN"


def categorize_reason(raw: str) -> str:
    if not raw:
        return "unknown"
    key = raw.strip().lower()
    for cat, keywords in _REASON_CATS:
        if any(kw in key for kw in keywords):
            return cat
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY FETCH — nba_api LeagueInjuries endpoint
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_nba_api(date_str: str) -> list[dict] | None:
    """
    Fetch injury data from NBA stats API.
    Returns list of raw row dicts or None on failure.
    """
    try:
        time.sleep(1)
        r = requests.get(
            NBA_INJURY_API,
            headers=_NBA_HEADERS,
            params={
                "LeagueID":   "00",
                "Date":       date_str,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()

        # Try nba_api library as fallback
        result_sets = data.get("resultSets", [])
        if not result_sets:
            return None

        rs = result_sets[0]
        headers = rs.get("headers", [])
        rows    = rs.get("rowSet", [])
        if not rows:
            return None

        return [dict(zip(headers, row)) for row in rows]

    except Exception as e:
        print(f"  [injury] NBA API error: {e}")
        return None


def _fetch_from_nba_api_lib(date_str: str) -> list[dict] | None:
    """Use nba_api library as alternative endpoint."""
    try:
        from nba_api.stats.endpoints import leagueinjurystatus
        time.sleep(1)
        resp = leagueinjurystatus.LeagueInjuryStatus(league_id="00")
        df = resp.get_data_frames()[0]
        if df.empty:
            return None
        return df.to_dict("records")
    except Exception as e:
        print(f"  [injury] nba_api lib error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECONDARY FETCH — NBA CMS PDF discovery + download
# ─────────────────────────────────────────────────────────────────────────────

def _discover_pdf_urls(date_str: str) -> list[str]:
    """
    Enumerate likely NBA CMS PDF URLs for date_str.
    NBA publishes 3–6 reports per day at standard time slots.
    Pattern: Injury-Report_YYYY-MM-DD_HH_MMam/pm.pdf
    """
    slots = [
        "01_00AM","02_00AM","03_00AM","05_00AM","06_00AM",
        "10_00AM","11_00AM","12_00PM","01_00PM","02_00PM",
        "03_00PM","04_00PM","05_00PM","06_00PM","07_00PM",
        "08_00PM","09_00PM","10_00PM","11_00PM",
    ]
    return [
        f"{NBA_CMS_PDF_BASE}/Injury-Report_{date_str}_{slot}.pdf"
        for slot in slots
    ]


def _probe_pdf_urls(date_str: str) -> list[str]:
    """
    HEAD-probe candidate PDF URLs. Return those that exist (200 OK).
    Respectful: 0.3s delay between probes.
    """
    candidates = _discover_pdf_urls(date_str)
    found = []
    for url in candidates:
        try:
            r = requests.head(url, timeout=8)
            if r.status_code == 200:
                found.append(url)
        except Exception:
            pass
        time.sleep(0.3)
    return found


def _download_pdf(url: str) -> bytes | None:
    """Download PDF bytes. Returns None on failure."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"  [injury] PDF download error: {e}")
        return None


def _parse_pdf_bytes(pdf_bytes: bytes, source_url: str) -> list[dict]:
    """
    Parse NBA injury report PDF using pdfplumber.
    Returns list of raw dicts with team, player, status, reason fields.
    """
    rows = []
    try:
        import io
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                # Try table extraction first
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if not row or len(row) < 4:
                            continue
                        # NBA PDF columns: Game Date | Game Time | Matchup | Team | Player | Status | Reason
                        # Column positions vary — use heuristics
                        cleaned = [str(c or "").strip() for c in row]
                        # Skip header rows
                        if any(h in cleaned[0].upper() for h in
                               ["GAME DATE","TEAM","STATUS","REPORT"]):
                            continue
                        # NOT YET SUBMITTED rows
                        joined = " ".join(cleaned).upper()
                        if "NOT YET SUBMITTED" in joined:
                            rows.append({
                                "team": cleaned[0] if len(cleaned) > 0 else "",
                                "player": "NOT YET SUBMITTED",
                                "status_raw": "NOT YET SUBMITTED",
                                "reason_raw": "",
                                "source_url": source_url,
                            })
                            continue
                        # Standard row — try to find player name (longest non-empty field)
                        if len(cleaned) >= 5:
                            rows.append({
                                "team":       cleaned[-4] if len(cleaned) >= 4 else "",
                                "player":     cleaned[-3] if len(cleaned) >= 3 else "",
                                "status_raw": cleaned[-2] if len(cleaned) >= 2 else "",
                                "reason_raw": cleaned[-1] if len(cleaned) >= 1 else "",
                                "source_url": source_url,
                            })
    except Exception as e:
        print(f"  [injury] PDF parse error: {e}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# ROW NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_api_row(raw: dict, report_ts: str, source_url: str, source_hash: str) -> dict:
    """
    Normalise a row from the NBA API response into the standard schema.
    NBA API field names vary — handle both snake_case and CamelCase.
    """
    def _g(row, *keys):
        for k in keys:
            v = row.get(k) or row.get(k.lower()) or row.get(k.upper())
            if v is not None:
                return str(v).strip()
        return ""

    player_raw  = _g(raw, "PlayerName", "PLAYER_NAME", "player_name", "Name")
    team_raw    = _g(raw, "TeamName", "TEAM_NAME", "team_name", "Team", "TeamCity")
    status_raw  = _g(raw, "PlayerStatus", "PLAYER_STATUS", "Status", "InjuryStatus")
    reason_raw  = _g(raw, "InjuryDescription", "INJURY_DESCRIPTION", "Reason",
                     "InjuryReason", "Comment")
    game_date   = _g(raw, "GameDate", "GAME_DATE", "game_date")
    game_time   = _g(raw, "GameTime", "GAME_TIME", "game_time")
    matchup     = _g(raw, "Matchup", "MATCHUP", "matchup")
    opponent    = _g(raw, "Opponent", "OPPONENT", "opponent")

    player_norm = normalize_name(player_raw)
    status_norm = normalize_status(status_raw)
    reason_cat  = categorize_reason(reason_raw)

    now_ts = datetime.now(timezone.utc).isoformat()

    return {
        "report_ts":              report_ts,
        "report_date":            report_ts[:10] if report_ts else "",
        "game_date":              game_date,
        "game_time_local":        game_time,
        "matchup":                matchup,
        "team":                   team_raw,
        "opponent":               opponent,
        "player_name":            player_raw,
        "player_name_normalized": player_norm,
        "status_raw":             status_raw,
        "status_normalized":      status_norm,
        "reason_raw":             reason_raw,
        "reason_category":        reason_cat,
        "source_url":             source_url,
        "source_file_name":       source_url.split("/")[-1] if source_url else "",
        "source_hash":            source_hash,
        "first_seen_ts":          now_ts,
        "last_seen_ts":           now_ts,
        "is_latest":              True,
        "status_changed":         False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STORAGE — current state + append-only history + manifest
# ─────────────────────────────────────────────────────────────────────────────

_HISTORY_COLS = [
    "report_ts","report_date","game_date","game_time_local","matchup",
    "team","opponent","player_name","player_name_normalized",
    "status_raw","status_normalized","reason_raw","reason_category",
    "source_url","source_file_name","source_hash",
    "first_seen_ts","last_seen_ts","is_latest","status_changed",
]


def _load_current() -> dict:
    """Load injuries_current.json → dict keyed by player_name_normalized."""
    if FILE_INJURIES_CURRENT.exists():
        try:
            return json.load(open(FILE_INJURIES_CURRENT))
        except Exception:
            return {}
    return {}


def _save_current(state: dict) -> None:
    FILE_INJURIES_CURRENT.parent.mkdir(parents=True, exist_ok=True)
    with open(FILE_INJURIES_CURRENT, "w") as f:
        json.dump(state, f, indent=2)


def _append_history(rows: list[dict]) -> None:
    """Append rows to injury_report_history.csv (append-only)."""
    FILE_INJURY_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FILE_INJURY_HISTORY.exists()
    with open(FILE_INJURY_HISTORY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_HISTORY_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)


def _load_manifest() -> list[dict]:
    if FILE_INJURY_MANIFEST.exists():
        try:
            return json.load(open(FILE_INJURY_MANIFEST))
        except Exception:
            return []
    return []


def _save_manifest(manifest: list[dict]) -> None:
    FILE_INJURY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(FILE_INJURY_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def _content_hash(content: bytes | str) -> str:
    if isinstance(content, str):
        content = content.encode()
    return hashlib.sha256(content).hexdigest()[:16]


def _already_processed(source_hash: str, manifest: list[dict]) -> bool:
    return any(m.get("source_hash") == source_hash for m in manifest)


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_changes(new_rows: list[dict], old_state: dict) -> tuple[list[dict], list[str]]:
    """
    Compare new rows against old current state.
    Mark status_changed=True when a player's status differs from previous.
    Returns (updated_rows, list of change description strings).
    """
    changes = []
    for row in new_rows:
        key = row["player_name_normalized"]
        old = old_state.get(key)
        if old and old.get("status_normalized") != row["status_normalized"]:
            row["status_changed"] = True
            row["first_seen_ts"]  = old.get("first_seen_ts", row["first_seen_ts"])
            changes.append(
                f"{row['player_name']}: {old.get('status_normalized','?')} → "
                f"{row['status_normalized']} ({row['reason_raw'][:50]})"
            )
        elif old:
            row["first_seen_ts"] = old.get("first_seen_ts", row["first_seen_ts"])
    return new_rows, changes


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FETCH ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_store(date_str: str | None = None) -> dict:
    """
    Main entry point. Fetch latest NBA injury report, normalise, store.
    Returns summary dict: {rows, changes, source, cached, error}.

    Fail-safe: on any error, returns cached data with error flag.
    """
    if date_str is None:
        from config import today_et
        date_str = today_et()

    fetch_ts = datetime.now(timezone.utc).isoformat()
    log_event("INJ", "FETCH_START", detail=f"date={date_str}")

    manifest   = _load_manifest()
    old_state  = _load_current()

    # ── Try primary source: nba_api library ──────────────────────────────────
    raw_rows = None
    source_url = f"nba_api://leagueinjurystatus/{date_str}"
    source_label = "nba_api_lib"

    raw_rows = _fetch_from_nba_api_lib(date_str)
    if not raw_rows:
        # Try direct stats API endpoint
        raw_rows = _fetch_from_nba_api(date_str)
        source_label = "nba_api_direct"

    # ── If API failed, return cached state ───────────────────────────────────
    if not raw_rows:
        print(f"  [injury] ⚠ All sources failed — using cached state ({len(old_state)} players)")
        log_event("INJ", "FETCH_FAILED_USING_CACHE", detail=f"date={date_str} cached={len(old_state)}")
        return {"rows": len(old_state), "changes": [], "source": "cache",
                "cached": True, "error": True}

    # ── Hash for dedup ────────────────────────────────────────────────────────
    content_str = json.dumps(raw_rows, sort_keys=True)
    report_hash = _content_hash(content_str.encode())

    if _already_processed(report_hash, manifest):
        print(f"  [injury] No new report (hash match) — {len(old_state)} players in cache")
        log_event("INJ", "FETCH_NO_CHANGE", detail=f"hash={report_hash}")
        return {"rows": len(old_state), "changes": [], "source": source_label,
                "cached": True, "error": False}

    # ── Normalise rows ────────────────────────────────────────────────────────
    normalised = []
    for raw in raw_rows:
        try:
            row = _normalise_api_row(raw, fetch_ts, source_url, report_hash)
            if row["player_name"] and row["player_name"] != "NOT YET SUBMITTED":
                normalised.append(row)
        except Exception:
            continue

    if not normalised:
        print(f"  [injury] ⚠ Fetch returned data but 0 rows normalised")
        log_event("INJ", "PARSE_EMPTY", detail=f"date={date_str}")
        return {"rows": 0, "changes": [], "source": source_label,
                "cached": False, "error": True}

    # ── Change detection ──────────────────────────────────────────────────────
    normalised, changes = _detect_changes(normalised, old_state)

    # ── Update current state ──────────────────────────────────────────────────
    # Mark all existing as not latest, then overwrite with new
    new_state = {r["player_name_normalized"]: r for r in normalised}
    _save_current(new_state)

    # ── Append history ────────────────────────────────────────────────────────
    _append_history(normalised)

    # ── Update manifest ───────────────────────────────────────────────────────
    manifest.append({
        "fetch_ts":      fetch_ts,
        "report_date":   date_str,
        "source_url":    source_url,
        "source_label":  source_label,
        "source_hash":   report_hash,
        "rows_parsed":   len(normalised),
        "players_unique": len(new_state),
        "changes":       len(changes),
        "status":        "ok",
    })
    _save_manifest(manifest)

    # ── Audit + summary ───────────────────────────────────────────────────────
    out_ct = sum(1 for r in normalised if r["status_normalized"] == "OUT")
    q_ct   = sum(1 for r in normalised if r["status_normalized"] == "QUESTIONABLE")
    d_ct   = sum(1 for r in normalised if r["status_normalized"] == "DOUBTFUL")

    log_event("INJ", "FETCH_OK",
              detail=f"rows={len(normalised)} out={out_ct} q={q_ct} d={d_ct} changes={len(changes)}")

    print(f"  [injury] {len(normalised)} players  |  "
          f"OUT={out_ct}  Q={q_ct}  D={d_ct}  changes={len(changes)}")

    if changes:
        for c in changes[:5]:
            print(f"    ⚡ {c}")
        if len(changes) > 5:
            print(f"    ... and {len(changes)-5} more")

    return {
        "rows":    len(normalised),
        "changes": changes,
        "source":  source_label,
        "cached":  False,
        "error":   False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUERY FUNCTIONS (used by batch_predict and reasoning_engine)
# ─────────────────────────────────────────────────────────────────────────────

def load_injury_state(date_str: str | None = None) -> dict:
    """
    Load latest injury state for date_str.
    Returns dict keyed by player_name_normalized → injury row.
    If no cached data exists, attempts a fresh fetch.
    Fail-safe: returns {} on complete failure.
    """
    if FILE_INJURIES_CURRENT.exists():
        try:
            state = json.load(open(FILE_INJURIES_CURRENT))
            # Filter to rows for date_str if provided
            if date_str:
                filtered = {k: v for k, v in state.items()
                            if not v.get("game_date") or v.get("game_date") == date_str
                            or v.get("report_date") == date_str}
                return filtered if filtered else state
            return state
        except Exception:
            pass

    # No cache — attempt fresh fetch
    print("  [injury] No cached state — fetching...")
    fetch_and_store(date_str)
    try:
        return json.load(open(FILE_INJURIES_CURRENT))
    except Exception:
        return {}


def get_player_injury_status(injury_state: dict, player_name: str) -> str:
    """
    Look up a player's current injury status.
    Returns status_normalized string: OUT, QUESTIONABLE, DOUBTFUL, PROBABLE,
    AVAILABLE, NOT_YET_SUBMITTED, UNKNOWN, or "" if not in report.
    """
    key = normalize_name(player_name)
    row = injury_state.get(key)
    if row is None:
        # Try fuzzy: check if key is a substring of any name in state
        for k, v in injury_state.items():
            if key in k or k in key:
                return v.get("status_normalized", "")
    return row.get("status_normalized", "") if row else ""


def get_teammate_load_boost(
    injury_state: dict,
    home_team: str,
    away_team: str,
    player_name: str,
    usage_threshold: float = 0.25,
) -> float:
    """
    Compute teammate load boost for player_name.
    Returns a continuous score 0.0–1.0:
      0.0 = no boost (no OUT teammates with significant usage)
      0.5 = one notable teammate OUT
      1.0 = multiple high-usage teammates OUT

    Logic: if a teammate on the same team is OUT and had usage > threshold,
    the player may absorb minutes/shots. This is the highest-value V18 signal.

    Note: teammate usage is estimated from team name matching in injury report.
    For precise usage, would need lineup data — this is a useful approximation.
    """
    out_count = 0
    # Find teammates who are OUT
    for key, row in injury_state.items():
        if row.get("status_normalized") != "OUT":
            continue
        # Check if on same team (compare team name)
        team = row.get("team", "").lower()
        our_teams = [home_team.lower(), away_team.lower()]
        if not any(t in team or team in t for t in our_teams if t):
            continue
        # Skip the player themselves
        if normalize_name(row.get("player_name", "")) == normalize_name(player_name):
            continue
        out_count += 1

    if out_count == 0:
        return 0.0
    elif out_count == 1:
        return 0.5
    else:
        return min(1.0, 0.5 + (out_count - 1) * 0.2)


def get_point_in_time_status(
    player_name: str,
    as_of_ts: str,
    game_date: str | None = None,
) -> dict | None:
    """
    Point-in-time query: what was the injury status for player_name
    as of timestamp as_of_ts?

    Used by backtests to avoid leakage — only use reports published
    BEFORE the prediction was made.

    Returns the most recent injury row with report_ts <= as_of_ts.
    Returns None if no record found before that timestamp.
    """
    if not FILE_INJURY_HISTORY.exists():
        return None

    player_key = normalize_name(player_name)
    best = None
    best_ts = ""

    try:
        with open(FILE_INJURY_HISTORY, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("player_name_normalized") != player_key:
                    continue
                if game_date and row.get("game_date") != game_date:
                    continue
                row_ts = row.get("report_ts", "")
                if row_ts <= as_of_ts and (best is None or row_ts > best_ts):
                    best = row
                    best_ts = row_ts
    except Exception:
        return None

    return best


def get_team_injury_summary(injury_state: dict, team: str) -> dict:
    """
    Summary of injury status for a team — used by reasoning engine.
    Returns: {out: [names], questionable: [names], doubtful: [names], total_affected: int}
    """
    result: dict = {"out": [], "questionable": [], "doubtful": [], "total_affected": 0}
    team_lower = team.lower()
    for key, row in injury_state.items():
        row_team = row.get("team", "").lower()
        if not (team_lower in row_team or row_team in team_lower):
            continue
        status = row.get("status_normalized", "")
        name   = row.get("player_name", key)
        if status == "OUT":
            result["out"].append(name)
        elif status == "QUESTIONABLE":
            result["questionable"].append(name)
        elif status == "DOUBTFUL":
            result["doubtful"].append(name)
    result["total_affected"] = (
        len(result["out"]) + len(result["questionable"]) + len(result["doubtful"])
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# VOL_RISK ANALYSIS (P-4 — gate validation against historical data)
# ─────────────────────────────────────────────────────────────────────────────

def run_volrisk_analysis(season_json_path: Path) -> None:
    """
    P-4: Analyse plays that were blocked from T1/T2 by the vol_risk gate.
    Checks old gate (vol_risk>1.5) vs new gate (std10>9) against actual results.
    Run once manually: python3 injury_ingest.py volrisk-analysis
    """
    import json as _json

    if not season_json_path.exists():
        print(f"  ✗ Not found: {season_json_path}")
        return

    plays = _json.load(open(season_json_path))
    graded = [p for p in plays if p.get("result") in ("WIN","LOSS")]

    if not graded:
        print("  No graded plays found.")
        return

    old_blocked = []  # blocked by old gate (vol_risk>1.5) but not new (std10>9)
    new_blocked = []  # still blocked by new gate (std10>9)
    would_upgrade = []  # old blocked → now T1/T2 eligible

    for p in graded:
        vol_risk = p.get("volRisk", 0) or 0
        std10    = p.get("std10", 5) or 5
        gap      = p.get("predGap", 0) or 0
        fc       = p.get("conf", 0) or 0
        tier     = p.get("tierLabel", "T3")

        old_hv = (std10 > 8) or (vol_risk > 1.5)
        new_hv = std10 > 9

        # Would this play have been T1 eligible under new gate?
        could_be_t1 = (fc >= 0.63 and gap >= 3.0 and std10 <= 8)

        if old_hv and not new_hv and could_be_t1:
            would_upgrade.append(p)
        if old_hv:
            old_blocked.append(p)
        if new_hv:
            new_blocked.append(p)

    print(f"\n  VOL_RISK GATE ANALYSIS — {season_json_path.name}")
    print(f"  {'─'*55}")
    print(f"  Total graded plays:       {len(graded):>6}")
    print(f"  Blocked by OLD gate:      {len(old_blocked):>6}  (std10>8 OR vol_risk>1.5)")
    print(f"  Blocked by NEW gate:      {len(new_blocked):>6}  (std10>9 only)")
    print(f"  Would upgrade to T1/T2:   {len(would_upgrade):>6}  (freed by gate change)")

    if would_upgrade:
        wins   = sum(1 for p in would_upgrade if p.get("result") == "WIN")
        losses = len(would_upgrade) - wins
        hr     = wins / len(would_upgrade) * 100 if would_upgrade else 0
        print(f"\n  Freed plays performance:")
        print(f"    {wins}W / {losses}L = {hr:.1f}% HR")
        if hr >= 62:
            print(f"    ✓ Gate change VALIDATED — freed plays hit at {hr:.1f}% (above T1 threshold)")
        elif hr >= 55:
            print(f"    ⚠ Marginal — freed plays hit at {hr:.1f}% (T2 range)")
        else:
            print(f"    ✗ Gate change NOT validated — freed plays only {hr:.1f}%")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# POLLING CADENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def should_poll_now(date_str: str | None = None) -> tuple[bool, str]:
    """
    Return (should_poll, reason) based on NBA injury reporting windows.
    NBA windows (ET):
      Day-before: after 5pm ET
      Game-day morning: 11am–1pm ET
      Pre-tip final: 2hrs before first tip
    UK times (ET+5 summer / ET+4 winter — approximate):
      Day-before: after 22:00 UK
      Game-day morning: 16:00–18:00 UK
      Pre-tip: 18:00–21:00 UK (covers 1pm–4pm ET tips)
    """
    from zoneinfo import ZoneInfo
    now_uk = datetime.now(ZoneInfo("Europe/London"))
    hour   = now_uk.hour

    # Off-hours — no NBA games typically start before 6pm ET (23:00 UK)
    if hour < 11:
        return False, "too early (before 11:00 UK)"
    if hour >= 23:
        return False, "after last tip window"

    # Game-day morning window (UK 16:00–18:00)
    if 16 <= hour < 18:
        return True, "game-day morning window (16:00–18:00 UK)"

    # Pre-tip window (UK 18:00–22:00)
    if 18 <= hour < 23:
        return True, "pre-tip window (18:00–23:00 UK)"

    # Day-before evening (UK 20:00–23:00)
    if 20 <= hour:
        return True, "day-before evening window (20:00+ UK)"

    # Mid-day (11:00–16:00) — light polling
    if 11 <= hour < 16:
        return True, "mid-day light polling"

    return False, "outside polling window"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse, sys
    from config import FILE_SEASON_2526

    parser = argparse.ArgumentParser(description="PropEdge V18 — Injury Ingest")
    parser.add_argument("command", nargs="?", default="fetch",
                        choices=["fetch","status","changes","history",
                                 "poll-check","volrisk-analysis","help"])
    parser.add_argument("--date",   help="YYYY-MM-DD (default: today ET)")
    parser.add_argument("--player", help="Player name (for history command)")
    parser.add_argument("--force",  action="store_true", help="Force fetch even if no-op")
    args = parser.parse_args()

    if args.command == "fetch":
        if not args.force:
            should, reason = should_poll_now(args.date)
            if not should:
                print(f"  [injury] Skipping — {reason}")
                return
        result = fetch_and_store(args.date)
        print(f"  Result: {result}")

    elif args.command == "status":
        state = load_injury_state(args.date)
        if not state:
            print("  No injury data cached.")
            return
        out_list  = [(k,v) for k,v in state.items() if v.get("status_normalized")=="OUT"]
        q_list    = [(k,v) for k,v in state.items() if v.get("status_normalized")=="QUESTIONABLE"]
        d_list    = [(k,v) for k,v in state.items() if v.get("status_normalized")=="DOUBTFUL"]
        print(f"\n  Injury Status ({args.date or 'latest'})")
        print(f"  {'─'*50}")
        for label, lst in [("OUT",out_list),("QUESTIONABLE",q_list),("DOUBTFUL",d_list)]:
            if lst:
                print(f"\n  {label} ({len(lst)}):")
                for _, v in lst:
                    print(f"    {v.get('player_name','?'):<28} {v.get('team',''):<20} {v.get('reason_raw','')[:40]}")

    elif args.command == "changes":
        manifest = _load_manifest()
        if not manifest:
            print("  No manifest found.")
            return
        last = manifest[-1]
        print(f"  Last report: {last.get('fetch_ts','')} | {last.get('changes',0)} changes")
        # Read most recent changed rows from history
        changed = []
        if FILE_INJURY_HISTORY.exists():
            with open(FILE_INJURY_HISTORY, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("status_changed") == "True":
                        changed.append(row)
        if changed:
            print(f"\n  Recent status changes:")
            for r in changed[-10:]:
                print(f"    {r.get('player_name','?'):<28} → {r.get('status_normalized','?')} "
                      f"({r.get('report_ts','')[:16]})")

    elif args.command == "history":
        name = args.player or (sys.argv[2] if len(sys.argv) > 2 else "")
        if not name:
            print("  Usage: injury_ingest.py history --player 'Player Name'")
            return
        key = normalize_name(name)
        if FILE_INJURY_HISTORY.exists():
            rows = []
            with open(FILE_INJURY_HISTORY, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("player_name_normalized") == key:
                        rows.append(row)
            if rows:
                print(f"\n  Injury history for {name} ({len(rows)} records):")
                for r in rows[-20:]:
                    print(f"    {r.get('report_ts','')[:16]}  {r.get('status_normalized','?'):<15} "
                          f"{r.get('reason_raw','')[:40]}")
            else:
                print(f"  No history found for '{name}' (key: {key})")

    elif args.command == "poll-check":
        should, reason = should_poll_now(args.date)
        print(f"  Should poll: {should} — {reason}")

    elif args.command == "volrisk-analysis":
        run_volrisk_analysis(FILE_SEASON_2526)

    else:
        print(__doc__)


if __name__ == "__main__":
    main()


# Alias for backwards compatibility
get_current_injuries = load_injury_state
