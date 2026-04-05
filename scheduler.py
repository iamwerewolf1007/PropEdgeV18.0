"""
PropEdge V18.0 — scheduler.py
Smart macOS launchd scheduler — 6 prediction batches.

Weekday schedule (Mon-Fri):
  B0  07:30 — Grade yesterday + retrain (earlier = more time before B1)
  B1  08:30 — First morning scan (overnight lines, early sharp value)
  B2  11:00 — Mid-morning refresh (injury news absorbed, line movement)
  B3  16:00 — Afternoon sweep (most US props posted by ~11am ET = 4pm UK)
  B4  18:30 — Pre-game final (1.5hr before typical 7pm ET first tip)
  B5  21:00 — Late/West-Coast top-up (10pm ET games, late line moves)

Weekend schedule (Sat-Sun):
  B0  07:30 — Grade + retrain (FIXED)
  B1-B5 shift relative to first NBA tip-off detected from Odds API
  Recalculated daily at 05:55 UK via daily-recalc agent

Usage:
  python3 scheduler.py install
  python3 scheduler.py uninstall
  python3 scheduler.py status
  python3 scheduler.py next
  python3 scheduler.py daily-recalc
  python3 scheduler.py weekend-check [YYYY-MM-DD]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
PLIST_DIR  = Path.home() / "Library" / "LaunchAgents"
PYTHON     = sys.executable
LOG_DIR    = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

_UK = ZoneInfo("Europe/London")
_ET = ZoneInfo("America/New_York")

# Launchd plist labels (v17 namespace — won't conflict with v14)
AGENTS = {
    "b0": "com.propedge.v18.batch0",   # grade + retrain
    "b1": "com.propedge.v18.batch1",   # morning scan
    "b2": "com.propedge.v18.batch2",   # mid-morning refresh
    "b3": "com.propedge.v18.batch3",   # afternoon sweep
    "b4": "com.propedge.v18.batch4",   # pre-game final
    "b5": "com.propedge.v18.batch5",   # late/west-coast
    "inj": "com.propedge.v18.injury",  # V18: injury polling (11:00-21:00, every 10min via repeat)
    "db": "com.propedge.v18.daily",    # daily recalc (05:55 UK)
}

# Fixed weekday times (UK local)
WEEKDAY_TIMES = {
    "b0": (7,  30),   # 07:30 — grade + retrain
    "ij": (8,   0),   # 08:00 — injury ingest (between grade and B1)
    "b1": (8,  30),   # 08:30 — first morning scan
    "b2": (11,  0),   # 11:00 — mid-morning refresh
    "b3": (16,  0),   # 16:00 — afternoon sweep
    "b4": (18, 30),   # 18:30 — pre-game final
    "b5": (21,  0),   # 21:00 — late / West-Coast
}

# Weekend: B1-B5 offsets relative to first NBA tip (ET→UK)
# Designed so B4 is the key pre-game run (60min before tip)
WEEKEND_OFFSETS_MINS = {
    "b1": -180,  # 3hr before tip  → early props + injury news
    "b2": -120,  # 2hr before tip  → line movement settled
    "b3":  -60,  # 60min before tip → pre-game final
    "b4":  +60,  # 60min after tip  → second wave / early-tip latecomers
    "b5": +180,  # 3hr after tip    → West-Coast games in play
}

WEEKEND_FLOOR = {
    "b1": (10,  0),
    "b2": (12,  0),
    "b3": (16,  0),
    "b4": (19,  0),
    "b5": (21,  0),
}

WEEKEND_CEIL = {
    "b1": (14,  0),
    "b2": (17,  0),
    "b3": (19,  0),
    "b4": (22,  0),
    "b5": (23, 55),
}

ODDS_API_KEY = None


# ─────────────────────────────────────────────────────────────────────────────
# TIP-OFF DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    global ODDS_API_KEY
    if ODDS_API_KEY is None:
        try:
            sys.path.insert(0, str(ROOT))
            from config import ODDS_API_KEY as KEY
            ODDS_API_KEY = KEY
        except Exception:
            ODDS_API_KEY = ""
    return ODDS_API_KEY


def fetch_first_tip_et(date_str: str) -> datetime | None:
    """Return earliest NBA tip-off as ET-aware datetime. None on failure."""
    key = _get_api_key()
    if not key:
        return None
    from config import et_window
    fr_utc, to_utc = et_window(date_str)
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/events",
            params={
                "apiKey": key,
                "commenceTimeFrom": fr_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "commenceTimeTo":   to_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            timeout=10,
        )
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        print(f"  [scheduler] Tip-off API error: {e}")
        return None

    if not events:
        return None

    earliest = None
    for ev in events:
        ts = ev.get("commence_time", "")
        if not ts:
            continue
        try:
            from datetime import timezone
            dt_utc = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            dt_et  = dt_utc.astimezone(_ET)
            if earliest is None or dt_et < earliest:
                earliest = dt_et
        except Exception:
            continue
    return earliest


def compute_weekend_times(date_str: str) -> dict[str, tuple[int, int]]:
    """Compute B1-B5 UK times for a weekend date. Falls back to weekday times."""
    first_tip_et = fetch_first_tip_et(date_str)

    if first_tip_et is None:
        print("  [scheduler] Tip-off detection failed — using weekday fallback times.")
        return {k: WEEKDAY_TIMES[k] for k in ("b1","b2","b3","b4","b5")}

    first_tip_uk = first_tip_et.astimezone(_UK)
    print(f"  [scheduler] First tip: {first_tip_et.strftime('%H:%M ET')} "
          f"= {first_tip_uk.strftime('%H:%M UK')}")

    result: dict[str, tuple[int, int]] = {}
    for batch, offset_mins in WEEKEND_OFFSETS_MINS.items():
        target = first_tip_uk + timedelta(minutes=offset_mins)
        hour, minute = target.hour, target.minute

        fl_h, fl_m = WEEKEND_FLOOR[batch]
        if (hour, minute) < (fl_h, fl_m):
            hour, minute = fl_h, fl_m

        ce_h, ce_m = WEEKEND_CEIL[batch]
        if (hour, minute) > (ce_h, ce_m):
            hour, minute = ce_h, ce_m

        result[batch] = (hour, minute)
        print(f"    {batch.upper()}: {hour:02d}:{minute:02d} UK (offset {offset_mins:+d}min from tip)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PLIST GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _plist_content(label: str, script: str, hour: int, minute: int,
                   log_name: str, args: list[str] | None = None) -> str:
    prog_args = f"        <string>{PYTHON}</string>\n        <string>{ROOT / script}</string>"
    if args:
        for a in args:
            prog_args += f"\n        <string>{a}</string>"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
{prog_args}
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>   <integer>{hour}</integer>
        <key>Minute</key> <integer>{minute}</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{LOG_DIR / log_name}.log</string>
    <key>StandardErrorPath</key>
    <string>{LOG_DIR / log_name}_err.log</string>

    <key>RunAtLoad</key>  <false/>
    <key>WorkingDirectory</key>
    <string>{ROOT}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{Path(PYTHON).parent}</string>
        <key>HOME</key>
        <string>{Path.home()}</string>
    </dict>
</dict>
</plist>"""


def _injury_plist() -> str:
    """Injury polling agent — runs every 15 minutes via StartInterval."""
    label = AGENTS["inj"]
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{PYTHON}</string>
        <string>{ROOT / "injury_ingest.py"}</string>
    </array>
    <key>StartInterval</key><integer>900</integer>
    <key>StandardOutPath</key><string>{LOG_DIR / "injury"}.log</string>
    <key>StandardErrorPath</key><string>{LOG_DIR / "injury"}_err.log</string>
    <key>RunAtLoad</key><false/>
    <key>WorkingDirectory</key><string>{ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{Path(PYTHON).parent}</string>
        <key>HOME</key><string>{Path.home()}</string>
    </dict>
</dict>
</plist>"""


def _injury_poll_plist() -> str:
    """Injury polling agent — runs every 600s (10 min), window-aware (checks internally)."""
    nl = "\n"
    prog_args = (
        "        <string>" + str(PYTHON) + "</string>" + nl +
        "        <string>" + str(ROOT / "injury_ingest.py") + "</string>" + nl +
        "        <string>fetch</string>"
    )
    home_str = str(Path.home())
    python_parent = str(Path(PYTHON).parent)
    log_poll = str(LOG_DIR / "injury_poll")
    root_str = str(ROOT)
    label = AGENTS["ip"]
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
{prog_args}
    </array>
    <key>StartInterval</key>
    <integer>600</integer>
    <key>StandardOutPath</key>
    <string>{log_poll}.log</string>
    <key>StandardErrorPath</key>
    <string>{log_poll}_err.log</string>
    <key>RunAtLoad</key>  <false/>
    <key>WorkingDirectory</key>
    <string>{root_str}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{python_parent}</string>
        <key>HOME</key>
        <string>{home_str}</string>
    </dict>
</dict>
</plist>"""


def _daily_runner_plist() -> str:
    """05:55 UK — daily schedule recalculator."""
    return _plist_content(
        AGENTS["db"], "scheduler.py", 5, 55, "daily_recalc",
        args=["daily-recalc"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCHCTL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _launchctl(cmd: list[str]) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def _load_plist(path: Path) -> None:
    _launchctl(["launchctl", "unload", str(path)])
    if _launchctl(["launchctl", "load", str(path)]):
        print(f"  ✓ Loaded: {path.name}")
    else:
        print(f"  ✗ Failed: {path.name}")


def _unload_plist(path: Path) -> None:
    if _launchctl(["launchctl", "unload", str(path)]):
        print(f"  ✓ Unloaded: {path.name}")
    if path.exists():
        path.unlink()
        print(f"  ✓ Deleted: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# INSTALL / UNINSTALL
# ─────────────────────────────────────────────────────────────────────────────

def install(times: dict[str, tuple[int, int]] | None = None) -> None:
    """Write all plists and load via launchctl."""
    if times is None:
        times = WEEKDAY_TIMES

    PLIST_DIR.mkdir(parents=True, exist_ok=True)
    plists = {}

    # B0 — always fixed at 07:30
    plists["b0"] = PLIST_DIR / f"{AGENTS['b0']}.plist"
    plists["b0"].write_text(_plist_content(
        AGENTS["b0"], "batch0_grade.py", *WEEKDAY_TIMES["b0"], "batch0"))

    # Injury ingest — 08:00 (between grade and B1), runs fetch + force
    plists["inj"] = PLIST_DIR / f"{AGENTS['inj']}.plist"
    plists["inj"].write_text(_plist_content(
        AGENTS["ij"], "injury_ingest.py", *WEEKDAY_TIMES["ij"], "injury_ingest",
        args=["fetch", "--force"]))

    # B1-B5 — prediction batches
    for bk in ("b1","b2","b3","b4","b5"):
        plists[bk] = PLIST_DIR / f"{AGENTS[bk]}.plist"
        arg = str({"b1":"1","b2":"2","b3":"3","b4":"4","b5":"5"}[bk])
        plists[bk].write_text(_plist_content(
            AGENTS[bk], "batch_predict.py", *times[bk], bk, args=[arg]))

    # V18: Injury polling agent — runs every 15 min 11:00-21:00 UK on game days
    # launchd doesn't support intervals + time windows natively, so we use
    # StartInterval (seconds) + run-time check in injury_ingest itself.
    # The agent starts at 11:00 and injury_ingest exits early if no games today.
    plists["inj"] = PLIST_DIR / f"{AGENTS['inj']}.plist"
    plists["inj"].write_text(_injury_plist())

    # Injury polling — window-aware, runs every 10min 11:00-22:00 via repeat interval
    # launchd StartInterval (seconds): installed as single 11:00 agent + repeat
    # Simpler: install at 11:00, 13:00, 15:00, 16:30, 17:30, 18:00, 19:00, 20:00, 21:00
    _injury_poll_times = [(11,0),(13,0),(15,0),(16,30),(17,30),(18,0),(19,0),(20,0),(21,0)]
    # Use a single repeat-interval plist for cleaner install
    _ip_path = PLIST_DIR / f"{AGENTS['ip']}.plist"
    _ip_path.write_text(_injury_poll_plist())
    _load_plist(_ip_path)
    plists["ip"] = _ip_path

    # Daily recalculator
    plists["db"] = PLIST_DIR / f"{AGENTS['db']}.plist"
    plists["db"].write_text(_daily_runner_plist())

    print("\n  Loading launchd agents...")
    for key, path in plists.items():
        _load_plist(path)

    print(f"\n  PropEdge V18.0 — Schedule installed:")
    print(f"    B0 Grade:       07:30 UK (fixed)")
    labels = {
        "b1": "Morning scan",
        "b2": "Mid-morning",
        "b3": "Afternoon",
        "b4": "Pre-game",
        "b5": "Late/West-Coast",
    }
    for bk in ("b1","b2","b3","b4","b5"):
        h, m = times[bk]
        print(f"    {bk.upper()} {labels[bk]:<16} {h:02d}:{m:02d} UK")
    print(f"    Daily recalc:   05:55 UK")


def uninstall() -> None:
    """Unload and remove all PropEdge V17 launchd agents."""
    for label in AGENTS.values():
        path = PLIST_DIR / f"{label}.plist"
        if path.exists():
            _unload_plist(path)
    print("  All V17 agents removed.")


# ─────────────────────────────────────────────────────────────────────────────
# DAILY RECALC
# ─────────────────────────────────────────────────────────────────────────────

def daily_recalc() -> None:
    """
    Called at 05:55 UK every morning by the daily agent.
    Weekdays: safety-reinstall weekday times.
    Weekends: fetch tip-off and rewrite B1-B5 plists.
    """
    now_uk   = datetime.now(_UK)
    weekday  = now_uk.weekday()  # 0=Mon, 5=Sat, 6=Sun
    date_str = now_uk.strftime("%Y-%m-%d")

    print(f"[daily-recalc] {date_str}  weekday={weekday}")

    if weekday not in (5, 6):
        print("  Weekday — reinstalling fixed times.")
        _reinstall_predict_plists(WEEKDAY_TIMES)
        return

    print(f"  Weekend — computing game-relative schedule...")
    weekend_times = compute_weekend_times(date_str)
    _reinstall_predict_plists(weekend_times)
    print(f"  Weekend schedule applied for {date_str}")


def _reinstall_predict_plists(times: dict[str, tuple[int, int]]) -> None:
    """Rewrite and reload B1-B5 plists only (injury agent is interval-based, not time-based)."""
    for bk in ("b1","b2","b3","b4","b5"):
        if bk not in times:
            continue
        path = PLIST_DIR / f"{AGENTS[bk]}.plist"
        arg  = str({"b1":"1","b2":"2","b3":"3","b4":"4","b5":"5"}[bk])
        path.write_text(_plist_content(
            AGENTS[bk], "batch_predict.py", *times[bk], bk, args=[arg]))
        _launchctl(["launchctl", "unload", str(path)])
        _load_plist(path)


# ─────────────────────────────────────────────────────────────────────────────
# STATUS / NEXT
# ─────────────────────────────────────────────────────────────────────────────

def status() -> None:
    print(f"\n  {'Agent':<45} {'Status':>12}")
    print(f"  {'─'*59}")
    for key, label in AGENTS.items():
        path = PLIST_DIR / f"{label}.plist"
        result = subprocess.run(["launchctl","list",label], capture_output=True, text=True)
        if result.returncode == 0:
            state = "LOADED ✓"
        elif path.exists():
            state = "NOT LOADED"
        else:
            state = "NOT INSTALLED"
        print(f"  {label:<45} {state:>12}")


def show_next() -> None:
    print(f"\n  {'Agent':<45} {'Next run (UK)':>20}")
    print(f"  {'─'*67}")
    now_uk = datetime.now(_UK)
    for key, label in AGENTS.items():
        path = PLIST_DIR / f"{label}.plist"
        if not path.exists():
            print(f"  {label:<45} {'NOT INSTALLED':>20}")
            continue
        try:
            import plistlib
            with open(path, "rb") as f:
                pl = plistlib.load(f)
            sci = pl.get("StartCalendarInterval", {})
            h   = sci.get("Hour", 0)
            m   = sci.get("Minute", 0)
            candidate = now_uk.replace(hour=h, minute=m, second=0, microsecond=0)
            if candidate <= now_uk:
                candidate += timedelta(days=1)
            print(f"  {label:<45} {candidate.strftime('%a %d %b  %H:%M UK'):>20}")
        except Exception as e:
            print(f"  {label:<45} {'ERROR: '+str(e):>20}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "install":
        print("\n  Installing PropEdge V17 launchd agents (weekday schedule)...")
        install()

    elif cmd == "uninstall":
        print("\n  Uninstalling PropEdge V17 agents...")
        uninstall()

    elif cmd == "reinstall":
        print("\n  Reinstalling V17 agents...")
        uninstall()
        install()

    elif cmd == "status":
        status()

    elif cmd == "next":
        show_next()

    elif cmd == "daily-recalc":
        daily_recalc()

    elif cmd == "weekend-check":
        date_str = sys.argv[2] if len(sys.argv) > 2 else datetime.now(_UK).strftime("%Y-%m-%d")
        print(f"\n  Weekend schedule preview for {date_str}:")
        times = compute_weekend_times(date_str)
        for bk, (h, m) in times.items():
            print(f"    {bk.upper()}: {h:02d}:{m:02d} UK")

    else:
        print(f"""
PropEdge V18.0 — Scheduler

Commands:
  python3 scheduler.py install          Install all launchd agents
  python3 scheduler.py uninstall        Remove all agents
  python3 scheduler.py reinstall        Remove + reinstall
  python3 scheduler.py status           Show all agent states
  python3 scheduler.py next             Print next run times
  python3 scheduler.py daily-recalc     Run the daily schedule recalculator
  python3 scheduler.py weekend-check    Preview weekend schedule
  python3 scheduler.py weekend-check YYYY-MM-DD

Weekday schedule (Mon-Fri):
  B0  07:30 UK — Grade yesterday + retrain
  B1  08:30 UK — Morning scan (overnight lines)
  B2  11:00 UK — Mid-morning refresh (injury news, line moves)
  B3  16:00 UK — Afternoon sweep (~11am ET, most props posted)
  B4  18:30 UK — Pre-game final (1.5hr before first tip)
  B5  21:00 UK — Late/West-Coast top-up

Weekend schedule (Sat-Sun, tip-relative):
  B0  07:30 UK — Grade + retrain (FIXED)
  B1  3hr  before first tip  (floor: 10:00, ceil: 14:00)
  B2  2hr  before first tip  (floor: 12:00, ceil: 17:00)
  B3  60min before first tip (floor: 16:00, ceil: 19:00)
  B4  60min after  first tip (floor: 19:00, ceil: 22:00)
  B5  3hr  after   first tip (floor: 21:00, ceil: 23:55)
  Weekend times recalculated daily at 05:55 UK.
""")


if __name__ == "__main__":
    main()
