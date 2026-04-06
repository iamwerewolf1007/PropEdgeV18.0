"""
PropEdge V18.0 — scheduler.py
Smart macOS launchd scheduler — 6 prediction batches + injury ingest.

Weekday schedule (Mon-Fri):
  B0   07:30 — Grade yesterday + retrain (fixed)
  INJ  08:00 — Injury report fetch (between grade and B1)
  B1   08:30 — Morning scan (overnight lines)
  B2   11:00 — Mid-morning refresh (injury news, line moves)
  B3   16:00 — Afternoon sweep (~11am ET, most props posted)
  B4   18:30 — Pre-game final (1.5hr before first tip)
  B5   21:00 — Late/West-Coast top-up
  INJ* every 15min — injury polling agent (StartInterval=900)
  DB   05:55 — Daily schedule recalculator

Weekend schedule (Sat-Sun):
  B0/INJ fixed. B1-B5 tip-relative. Recalculated daily at 05:55 UK.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT      = Path(__file__).parent.resolve()
PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
PYTHON    = sys.executable
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

_UK = ZoneInfo("Europe/London")
_ET = ZoneInfo("America/New_York")

# ── Agent labels ──────────────────────────────────────────────────────────────
AGENTS = {
    "b0":  "com.propedge.v18.batch0",
    "inj": "com.propedge.v18.injury",
    "b1":  "com.propedge.v18.batch1",
    "b2":  "com.propedge.v18.batch2",
    "b3":  "com.propedge.v18.batch3",
    "b4":  "com.propedge.v18.batch4",
    "b5":  "com.propedge.v18.batch5",
    "db":  "com.propedge.v18.daily",
}

# ── Fixed weekday times (UK local) ────────────────────────────────────────────
WEEKDAY_TIMES = {
    "b0":  (7,  30),
    "inj": (8,   0),
    "b1":  (8,  30),
    "b2":  (11,  0),
    "b3":  (16,  0),
    "b4":  (18, 30),
    "b5":  (21,  0),
}

# ── Weekend offsets from first tip ────────────────────────────────────────────
WEEKEND_OFFSETS_MINS = {
    "b1": -180, "b2": -120, "b3": -60, "b4": 60, "b5": 180,
}
WEEKEND_FLOOR = {
    "b1": (10, 0), "b2": (12, 0), "b3": (16, 0), "b4": (19, 0), "b5": (21, 0),
}
WEEKEND_CEIL = {
    "b1": (14, 0), "b2": (17, 0), "b3": (19, 0), "b4": (22, 0), "b5": (23, 55),
}

ODDS_API_KEY = None


# ── Tip-off detection ─────────────────────────────────────────────────────────

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


def fetch_first_tip_et(date_str: str):
    key = _get_api_key()
    if not key:
        return None
    from config import et_window
    from datetime import timezone
    fr_utc, to_utc = et_window(date_str)
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/events",
            params={"apiKey": key,
                    "commenceTimeFrom": fr_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "commenceTimeTo":   to_utc.strftime("%Y-%m-%dT%H:%M:%SZ")},
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


def compute_weekend_times(date_str: str) -> dict:
    first_tip_et = fetch_first_tip_et(date_str)
    if first_tip_et is None:
        print("  [scheduler] Tip-off detection failed — using weekday fallback.")
        return {k: WEEKDAY_TIMES[k] for k in ("b1","b2","b3","b4","b5")}
    first_tip_uk = first_tip_et.astimezone(_UK)
    print(f"  [scheduler] First tip: {first_tip_et.strftime('%H:%M ET')} = {first_tip_uk.strftime('%H:%M UK')}")
    result = {}
    for bk, offset in WEEKEND_OFFSETS_MINS.items():
        target = first_tip_uk + timedelta(minutes=offset)
        h, m   = target.hour, target.minute
        fl_h, fl_m = WEEKEND_FLOOR[bk]
        if (h, m) < (fl_h, fl_m): h, m = fl_h, fl_m
        ce_h, ce_m = WEEKEND_CEIL[bk]
        if (h, m) > (ce_h, ce_m): h, m = ce_h, ce_m
        result[bk] = (h, m)
        print(f"    {bk.upper()}: {h:02d}:{m:02d} UK (offset {offset:+d}min)")
    return result


# ── Plist generators ──────────────────────────────────────────────────────────

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
    <key>Label</key><string>{label}</string>
    <key>ProgramArguments</key>
    <array>
{prog_args}
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>   <integer>{hour}</integer>
        <key>Minute</key> <integer>{minute}</integer>
    </dict>
    <key>StandardOutPath</key><string>{LOG_DIR / log_name}.log</string>
    <key>StandardErrorPath</key><string>{LOG_DIR / log_name}_err.log</string>
    <key>RunAtLoad</key>  <false/>
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
    """Injury polling — StartInterval every 15 minutes all day."""
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
    <key>StandardOutPath</key><string>{LOG_DIR / "injury_poll"}.log</string>
    <key>StandardErrorPath</key><string>{LOG_DIR / "injury_poll"}_err.log</string>
    <key>RunAtLoad</key>  <false/>
    <key>WorkingDirectory</key><string>{ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{Path(PYTHON).parent}</string>
        <key>HOME</key><string>{Path.home()}</string>
    </dict>
</dict>
</plist>"""


def _daily_runner_plist() -> str:
    return _plist_content(
        AGENTS["db"], "scheduler.py", 5, 55, "daily_recalc",
        args=["daily-recalc"]
    )


# ── launchctl helpers ─────────────────────────────────────────────────────────

def _launchctl(cmd: list[str]) -> bool:
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0

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


# ── Install / Uninstall ───────────────────────────────────────────────────────

def install(times: dict | None = None) -> None:
    if times is None:
        times = WEEKDAY_TIMES
    PLIST_DIR.mkdir(parents=True, exist_ok=True)

    # B0 — grade + retrain (fixed 07:30)
    p_b0 = PLIST_DIR / f"{AGENTS['b0']}.plist"
    p_b0.write_text(_plist_content(AGENTS["b0"], "batch0_grade.py", 7, 30, "batch0"))
    _load_plist(p_b0)

    # INJ — injury polling (StartInterval 15min)
    p_inj = PLIST_DIR / f"{AGENTS['inj']}.plist"
    p_inj.write_text(_injury_poll_plist())
    _load_plist(p_inj)

    # B1-B5 — prediction batches
    for bk in ("b1","b2","b3","b4","b5"):
        p = PLIST_DIR / f"{AGENTS[bk]}.plist"
        h, m = times[bk]
        arg  = str({"b1":"1","b2":"2","b3":"3","b4":"4","b5":"5"}[bk])
        p.write_text(_plist_content(AGENTS[bk], "batch_predict.py", h, m, bk, args=[arg]))
        _load_plist(p)

    # DB — daily recalculator (05:55)
    p_db = PLIST_DIR / f"{AGENTS['db']}.plist"
    p_db.write_text(_daily_runner_plist())
    _load_plist(p_db)

    print(f"\n  PropEdge V18.0 — Schedule installed:")
    print(f"    B0  Grade:          07:30 UK (fixed)")
    print(f"    INJ Injury poll:    every 15min (StartInterval)")
    labels = {"b1":"Morning scan","b2":"Mid-morning","b3":"Afternoon",
              "b4":"Pre-game","b5":"Late/West-Coast"}
    for bk in ("b1","b2","b3","b4","b5"):
        h, m = times[bk]
        print(f"    {bk.upper()}  {labels[bk]:<16} {h:02d}:{m:02d} UK")
    print(f"    DB  Daily recalc:   05:55 UK")


def uninstall() -> None:
    for label in AGENTS.values():
        path = PLIST_DIR / f"{label}.plist"
        if path.exists():
            _unload_plist(path)
    print("  All V18 agents removed.")


# ── Daily recalc ──────────────────────────────────────────────────────────────

def daily_recalc() -> None:
    now_uk   = datetime.now(_UK)
    weekday  = now_uk.weekday()
    date_str = now_uk.strftime("%Y-%m-%d")
    print(f"[daily-recalc] {date_str}  weekday={weekday}")
    if weekday not in (5, 6):
        print("  Weekday — reinstalling fixed times.")
        _reinstall_predict_plists(WEEKDAY_TIMES)
        return
    print("  Weekend — computing game-relative schedule...")
    weekend_times = compute_weekend_times(date_str)
    _reinstall_predict_plists(weekend_times)
    print(f"  Weekend schedule applied for {date_str}")


def _reinstall_predict_plists(times: dict) -> None:
    for bk in ("b1","b2","b3","b4","b5"):
        if bk not in times:
            continue
        path = PLIST_DIR / f"{AGENTS[bk]}.plist"
        h, m = times[bk]
        arg  = str({"b1":"1","b2":"2","b3":"3","b4":"4","b5":"5"}[bk])
        path.write_text(_plist_content(AGENTS[bk], "batch_predict.py", h, m, bk, args=[arg]))
        _launchctl(["launchctl", "unload", str(path)])
        _load_plist(path)


# ── Status / Next ─────────────────────────────────────────────────────────────

def status() -> None:
    print(f"\n  {'Agent':<45} {'Status':>12}")
    print(f"  {'─'*59}")
    for key, label in AGENTS.items():
        path   = PLIST_DIR / f"{label}.plist"
        result = subprocess.run(["launchctl","list",label], capture_output=True, text=True)
        if result.returncode == 0:   state = "LOADED ✓"
        elif path.exists():           state = "NOT LOADED"
        else:                         state = "NOT INSTALLED"
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
            if sci:
                h   = sci.get("Hour", 0)
                m   = sci.get("Minute", 0)
                candidate = now_uk.replace(hour=h, minute=m, second=0, microsecond=0)
                if candidate <= now_uk:
                    candidate += timedelta(days=1)
                print(f"  {label:<45} {candidate.strftime('%a %d %b  %H:%M UK'):>20}")
            else:
                interval = pl.get("StartInterval", 0)
                print(f"  {label:<45} {'every '+str(interval)+'s':>20}")
        except Exception as e:
            print(f"  {label:<45} {'ERROR: '+str(e)[:20]:>20}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if   cmd == "install":       print("\n  Installing PropEdge V18 agents..."); install()
    elif cmd == "uninstall":     print("\n  Uninstalling V18 agents...");        uninstall()
    elif cmd == "reinstall":     uninstall(); install()
    elif cmd == "status":        status()
    elif cmd == "next":          show_next()
    elif cmd == "daily-recalc":  daily_recalc()
    elif cmd == "weekend-check":
        date_str = sys.argv[2] if len(sys.argv) > 2 else datetime.now(_UK).strftime("%Y-%m-%d")
        print(f"\n  Weekend schedule preview for {date_str}:")
        for bk, (h, m) in compute_weekend_times(date_str).items():
            print(f"    {bk.upper()}: {h:02d}:{m:02d} UK")
    else:
        print("""
PropEdge V18.0 — Scheduler
  install          Install all launchd agents
  uninstall        Remove all agents
  reinstall        Remove + reinstall
  status           Show all agent states
  next             Print next run times
  daily-recalc     Run daily schedule recalculator
  weekend-check [YYYY-MM-DD]
""")

if __name__ == "__main__":
    main()
