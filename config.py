"""
PropEdge V14.0 — config.py
All constants, paths, timezone helpers, DVP cache, clean_json.
"""

import json
import math
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo  # Python 3.9+

# ─────────────────────────────────────────────────────────────────────────────
# VERSION & PATHS
# ─────────────────────────────────────────────────────────────────────────────
VERSION   = "V18.0"
ROOT      = Path(__file__).parent.resolve()
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR   = ROOT / "logs"
DAILY_DIR = ROOT / "daily"
SRC_DIR   = ROOT / "source-files"

for d in (DATA_DIR, MODEL_DIR, LOG_DIR, DAILY_DIR, SRC_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE FILES
# ─────────────────────────────────────────────────────────────────────────────
FILE_GL_2425 = SRC_DIR / "nba_gamelogs_2024_25.csv"
FILE_GL_2526 = SRC_DIR / "nba_gamelogs_2025_26.csv"
FILE_H2H     = SRC_DIR / "h2h_database.csv"
FILE_PROPS   = SRC_DIR / "PropEdge_-_Match_and_Player_Prop_lines_.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
# DATA / MODEL FILES
# ─────────────────────────────────────────────────────────────────────────────
FILE_TODAY       = DATA_DIR / "today.json"
FILE_SEASON_2526 = DATA_DIR / "season_2025_26.json"
FILE_SEASON_2425 = DATA_DIR / "season_2024_25.json"
FILE_AUDIT       = DATA_DIR / "audit_log.csv"
FILE_DVP         = DATA_DIR / "dvp_rankings.json"

# V18 Injury intelligence
FILE_INJURIES_CURRENT  = DATA_DIR / "injuries_current.json"
FILE_INJURY_HISTORY    = DATA_DIR / "injury_report_history.csv"
FILE_INJURY_MANIFEST   = DATA_DIR / "injury_report_manifest.json"

FILE_INJURIES_CURRENT  = DATA_DIR / "injuries_current.json"
FILE_INJURY_HISTORY    = DATA_DIR / "injury_report_history.csv"
FILE_INJURY_MANIFEST   = DATA_DIR / "injury_report_manifest.json"

FILE_CLF         = MODEL_DIR / "direction_classifier.pkl"
FILE_REG         = MODEL_DIR / "projection_model.pkl"
FILE_CAL         = MODEL_DIR / "calibrator.pkl"
FILE_TRUST       = MODEL_DIR / "player_trust.json"

# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────
ODDS_API_KEY    = "c0bab20a574208a41a6e0d930cdaf313"  # update if rotated
ODDS_BASE_URL   = "https://api.the-odds-api.com/v4"
CREDIT_ALERT    = 170

GIT_REMOTE = "git@github.com:iamwerewolf1007/PropEdgeV18.0.git"
REPO_DIR   = Path.home() / "Documents" / "GitHub" / "PropEdgeV18.0"

# ─────────────────────────────────────────────────────────────────────────────
# SEASON CONTEXT (for season_progress feature)
# ─────────────────────────────────────────────────────────────────────────────
SEASON_START = datetime(2026, 10, 1, tzinfo=timezone.utc)
SEASON_END   = datetime(2027, 4, 20, tzinfo=timezone.utc)
SEASON_DAYS  = (SEASON_END - SEASON_START).days  # ~201 days

# ─────────────────────────────────────────────────────────────────────────────
# TIMEZONE HELPERS (correct US + UK DST — fixed in V12, inherited)
# ─────────────────────────────────────────────────────────────────────────────
_ET = ZoneInfo("America/New_York")
_UK = ZoneInfo("Europe/London")


def get_et() -> ZoneInfo:
    return _ET


def get_uk() -> ZoneInfo:
    return _UK


def now_et() -> datetime:
    return datetime.now(_ET)


def now_uk() -> datetime:
    return datetime.now(_UK)


def today_et() -> str:
    """Current NBA calendar date as YYYY-MM-DD (Eastern Time)."""
    return now_et().strftime("%Y-%m-%d")


def et_window(date_str: str):
    """Return (from_utc, to_utc) covering full ET calendar day ± 90-min buffer."""
    y, mo, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
    midnight_et = datetime(y, mo, d, 0, 0, 0, tzinfo=_ET)
    midnight_utc = midnight_et.astimezone(timezone.utc)
    return midnight_utc - timedelta(minutes=90), midnight_utc + timedelta(hours=25, minutes=90)


def season_progress(date_str: str) -> float:
    """0.0 at season start (Oct 1) → 1.0 at season end (Apr 20)."""
    try:
        y, mo, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
        dt = datetime(y, mo, d, tzinfo=timezone.utc)
        prog = (dt - SEASON_START).days / SEASON_DAYS
        return max(0.0, min(1.0, prog))
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# POSITION MAPPING
# ─────────────────────────────────────────────────────────────────────────────
POS_MAP = {
    "PG": "Guard",  "SG": "Guard",  "G": "Guard",
    "SF": "Forward","PF": "Forward","F": "Forward",
    "C":  "Center", "FC": "Forward","GF": "Guard",
}

# Position weights for Engine A composite (10 signals)
POS_WEIGHTS = {
    "Guard":   [3.0, 2.5, 2.0, 2.0, 1.0, 1.5, 1.2, 0.5, 1.5, 1.0],
    "Forward": [3.0, 2.5, 2.0, 1.5, 1.5, 1.5, 1.0, 0.5, 1.0, 0.75],
    "Center":  [2.5, 2.0, 2.0, 1.0, 1.5, 2.5, 1.0, 1.0, 0.5, 1.5],
}


def get_pos_group(pos_raw: str) -> str:
    return POS_MAP.get(str(pos_raw).strip().upper(), "Guard")


# ─────────────────────────────────────────────────────────────────────────────
# DVP CACHE (process-scoped, invalidated after dvp_updater runs)
# ─────────────────────────────────────────────────────────────────────────────
_dvp_cache: dict | None = None


def _load_dvp_cache() -> dict:
    global _dvp_cache
    if _dvp_cache is None:
        if FILE_DVP.exists():
            with open(FILE_DVP) as f:
                _dvp_cache = json.load(f)
        else:
            _dvp_cache = {}
    return _dvp_cache


def invalidate_dvp_cache():
    global _dvp_cache
    _dvp_cache = None


def get_dvp(team: str, pos_raw: str, fallback: int = 15) -> int:
    cache = _load_dvp_cache()
    pos = get_pos_group(pos_raw)
    key = f"{team}|{pos}"
    return int(cache.get(key, fallback))


# ─────────────────────────────────────────────────────────────────────────────
# TEAM NAME / ABBREVIATION MAPS
# ─────────────────────────────────────────────────────────────────────────────
TEAM_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN JSON (numpy types → native Python, NaN → None)
# ─────────────────────────────────────────────────────────────────────────────
def clean_json(obj):
    """Recursively convert numpy/pandas types and NaN for JSON serialisation."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [clean_json(x) for x in obj.tolist()]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# V14 FUSION WEIGHT FORMULA
# ─────────────────────────────────────────────────────────────────────────────
def fusion_weights(sp: float) -> tuple[float, float, float]:
    """
    Season-adaptive fusion weights.
    sp = season_progress (0.0 Oct → 1.0 Apr)
    Returns (alpha_clf, beta_reg, gamma_comp)
    """
    alpha = 0.55 + 0.05 * sp   # 0.55 → 0.60
    beta  = 0.30 - 0.05 * sp   # 0.30 → 0.25
    gamma = 0.15
    return alpha, beta, gamma


# ─────────────────────────────────────────────────────────────────────────────
# TIER THRESHOLDS (V14)
# ─────────────────────────────────────────────────────────────────────────────
TIER_GATES = {
    "T1_ULTRA":   {"conf": 0.73, "gap": 5.0, "std10": 6, "vol_risk": 1.5},
    "T1_PREMIUM": {"conf": 0.68, "gap": 4.0, "std10": 7, "vol_risk": 1.5},
    "T1":         {"conf": 0.63, "gap": 3.0, "std10": 8, "vol_risk": 1.5},
    "T2":         {"conf": 0.56, "gap": 2.0, "std10": 9, "vol_risk": 99},
}

UNIT_MAP = {
    "T1_ULTRA": 3.0, "T1_PREMIUM": 2.0, "T1": 2.0, "T2": 1.0,
    "T3": 0.0, "T3_LEAN": 0.0,
}

LEAN_ZONE = (0.46, 0.54)        # cal_prob must be outside this for hard call
TRUST_THRESHOLD = 0.42          # player trust below this → demote T1 → T2
HIGH_LINE_THRESHOLD = 25        # lines ≥ this get OVER confidence penalty
MIN_PRIOR_GAMES = 5             # minimum games needed to score a player

# V18 injury confidence penalties
INJURY_PENALTY_QUESTIONABLE = 0.05   # cal_prob penalty for QUESTIONABLE
INJURY_PENALTY_DOUBTFUL     = 0.08   # cal_prob penalty for DOUBTFUL
# OUT players are skipped entirely before scoring
MIN_PRIOR_GAMES = 5             # fewer than this → skip prediction

# ─────────────────────────────────────────────────────────────────────────────
# V18 INJURY CONFIDENCE MODIFIERS
# ─────────────────────────────────────────────────────────────────────────────
INJURY_CONF_PENALTY = {
    "OUT":          None,    # Skipped entirely — not scored
    "DOUBTFUL":     0.10,    # -0.10 fusion_conf — near-disqualifying
    "QUESTIONABLE": 0.06,    # -0.06 fusion_conf — meaningful but not fatal
    "PROBABLE":     0.01,    # -0.01 — minor flag, expected to play
    "AVAILABLE":    0.00,    # No penalty
    "UNKNOWN":      0.03,    # -0.03 — treat as mild concern
}
TEAMMATE_BOOST_USAGE_THRESHOLD = 0.22   # teammate usage % that qualifies as "primary scorer"
TEAMMATE_BOOST_MAGNITUDE       = 2.5    # pts added to L10 estimate for usage boost calc
