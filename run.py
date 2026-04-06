#!/usr/bin/env python3
"""
PropEdge V18.0 — Master Orchestrator
======================================
Repo:        git@github.com:iamwerewolf1007/PropEdgeV18.0.git
Working dir: ~/Documents/GitHub/PropEdgeV18.0

Commands:
  python3 run.py setup      — First-time: train models + build season JSONs (~10-15 min)
  python3 run.py generate   — Alias for setup (rebuild everything from scratch)
  python3 run.py grade      — Batch 0: grade yesterday via NBA API + retrain (08:00 UK)
  python3 run.py predict    — Batch 4: pre-game final         (18:30 UK, default)
  python3 run.py predict 1  — Batch 1: morning scan           (08:30 UK)
  python3 run.py predict 2  — Batch 2: mid-morning refresh    (11:00 UK)
  python3 run.py predict 3  — Batch 3: afternoon sweep        (16:00 UK)
  python3 run.py predict 4  — Batch 4: pre-game final         (18:30 UK)
  python3 run.py predict 5  — Batch 5: late/West-Coast        (21:00 UK)

Overrides:
  python3 run.py grade --date 2026-04-02            — Grade a specific date
  python3 run.py grade --date 2026-04-02 --no-retrain
  python3 run.py grade-csv --date 2026-04-02        — Grade from game log CSV (no NBA API)
  python3 run.py grade-csv --date 2026-04-02 --no-retrain

Utilities:
  python3 run.py retrain    — Retrain models only
  python3 run.py dvp        — Rebuild DVP rankings from game log
  python3 run.py h2h        — Rebuild H2H database
  python3 run.py check      — Data integrity report
  python3 run.py install    — Install launchd scheduler (macOS)
  python3 run.py uninstall  — Remove launchd scheduler
  python3 run.py status     — Scheduler + model status
  python3 run.py weekend    — Preview weekend batch schedule

Batch schedule (launchd agents, UK time):
  07:30 — batch0_grade.py         (grade + retrain)
  08:30 — batch_predict.py 1      (morning scan — overnight lines)
  11:00 — batch_predict.py 2      (mid-morning — injury news, line moves)
  16:00 — batch_predict.py 3      (afternoon sweep — most US props posted)
  18:30 — batch_predict.py 4      (pre-game final — 1.5hr before first tip)
  21:00 — batch_predict.py 5      (late/West-Coast top-up)
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    VERSION, FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST,
    FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_DVP,
    FILE_TODAY, FILE_SEASON_2526, FILE_PROPS, clean_json,
)


# ─────────────────────────────────────────────────────────────────────────────
# SETUP / GENERATE  (run once — builds everything from source files)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_setup():
    """
    First-time setup. Calls generate_season_json.py to:
      1. Load game logs + H2H
      2. Generate 2024-25 synthetic props (synthetic_lines.py)
      3. Load 2025-26 real prop lines (Excel)
      4. Train V14 models (OOF 5-fold — ~10-15 min)
      5. Score + grade all plays → season_2024_25.json + season_2025_26.json
      6. Write today.json + backtest_summary.json
    After this, season_2024_25.json is permanently locked.
    Daily grade/predict batches append new days on top.
    """
    print(f"\n  PropEdge {VERSION}  —  Setup (first-time)\n")

    # Validate source files
    missing = [f for f in (FILE_GL_2425, FILE_GL_2526, FILE_PROPS) if not f.exists()]
    if missing:
        print("  ✗ Missing source files:")
        for f in missing:
            print(f"    {f}")
        print("  Add these to source-files/ and retry.")
        return

    import pandas as pd
    gl26 = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    gl25 = pd.read_csv(FILE_GL_2425, parse_dates=["GAME_DATE"])
    xl   = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
    xl["Date"] = pd.to_datetime(xl["Date"])
    print("  Source files:")
    print(f"    Excel props    : {len(xl):,} rows | "
          f"{xl['Date'].min().date()} → {xl['Date'].max().date()}")
    print(f"    Game log 25-26 : {len(gl26):,} rows | "
          f"latest: {gl26['GAME_DATE'].max().date()}")
    print(f"    Game log 24-25 : {len(gl25):,} rows")
    print()

    # ── Step 1: generate_season_json.py does all the heavy lifting ────────────
    print("  [1/2] Building full season JSON + training models...")
    print("        (generate_season_json.py — ~10-15 min)\n")

    result = subprocess.run(
        [sys.executable, str(ROOT / "generate_season_json.py")],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print("  ✗ generate_season_json.py failed — aborting setup.")
        sys.exit(1)

    # Ensure today.json exists
    if not FILE_TODAY.exists():
        FILE_TODAY.write_text("[]")
        print("  Created: today.json (empty)")

    print()

    # ── Step 2: check if yesterday needs grading from CSV ─────────────────────
    print("  [2/2] Checking if yesterday needs grading from game log...")
    from datetime import datetime, timedelta
    from config import get_uk
    uk_now   = datetime.now(get_uk())
    yest_str = (uk_now - timedelta(days=1)).strftime("%Y-%m-%d")

    yest_gl   = gl26[gl26["GAME_DATE"].dt.strftime("%Y-%m-%d") == yest_str]
    yest_play = yest_gl[
        (yest_gl["DNP"].fillna(0) == 0) & (yest_gl["MIN_NUM"].fillna(0) > 0)
    ]

    with open(FILE_SEASON_2526) as f:
        season = json.load(f)
    yest_ungraded = [
        p for p in season
        if p.get("date") == yest_str and not p.get("result")
    ]

    if len(yest_ungraded) > 0 and len(yest_play) > 0:
        print(f"  Found {len(yest_ungraded)} ungraded plays for {yest_str} "
              f"with {len(yest_play)} CSV results — grading now...")
        result = _grade_from_csv(yest_str, no_retrain=True)
        print(f"  Graded: {result['wins']}W / {result['losses']}L / {result['dnps']} DNP")
    elif len(yest_ungraded) > 0:
        print(f"  {len(yest_ungraded)} ungraded plays for {yest_str} "
              f"but no CSV results yet — will grade after games finish.")
    else:
        print(f"  Yesterday ({yest_str}) already graded.")

    print()
    print("  ✓ Setup complete. Season JSONs fully built and graded.")
    print("  Daily schedule:")
    print("    07:30 UK → python3 run.py grade    (grade yesterday)")
    print("    08:30 UK → python3 run.py predict  (B1: morning scan)")
    print("    11:00 UK → python3 run.py predict 2 (B2: mid-morning)")
    print("    16:00 UK → python3 run.py predict 3 (B3: afternoon)")
    print("    18:30 UK → python3 run.py predict 4 (B4: pre-game)")
    print("    21:00 UK → python3 run.py predict 5 (B5: late/west-coast)")
    print()

    # Note: generate_season_json.py already calls git_push internally.
    # run.py does NOT double-push here — the push happened inside the subprocess.


# ─────────────────────────────────────────────────────────────────────────────
# GRADE  (daily Batch 0 — NBA API)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_grade():
    """Grade yesterday via NBA API + retrain. Supports --date and --no-retrain."""
    import batch0_grade
    batch0_grade.main()


# ─────────────────────────────────────────────────────────────────────────────
# GRADE-CSV  (NBA API bypass — grade from game log CSV)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_grade_from_csv():
    """
    Grade a date from game log CSV — no NBA API needed.
    Usage: python3 run.py grade-csv --date 2026-04-02 [--no-retrain]
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",       required=True)
    parser.add_argument("--no-retrain", action="store_true")
    args, _ = parser.parse_known_args()

    print(f"\n  Grading {args.date} from game log CSV...")
    result = _grade_from_csv(args.date, no_retrain=args.no_retrain)

    from batch_predict import git_push
    git_push(f"B0-csv: grade {args.date}")
    print(f"  ✓ {result['wins']}W / {result['losses']}L / {result['dnps']} DNP\n")


def _grade_from_csv(grade_date: str, no_retrain: bool = False) -> dict:
    """
    Grade plays for grade_date using game log CSV.
    Matches plays in today.json and season JSON against CSV results.
    Returns dict with wins / losses / dnps counts.
    """
    import pandas as pd
    from reasoning_engine import generate_post_match_reason

    gl     = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    day_gl = gl[gl["GAME_DATE"].dt.strftime("%Y-%m-%d") == grade_date]
    played = day_gl[
        (day_gl["DNP"].fillna(0) == 0) & (day_gl["MIN_NUM"].fillna(0) > 0)
    ]

    if len(played) == 0:
        latest = gl["GAME_DATE"].max().date() if len(gl) else "unknown"
        print(f"  ⚠ No played rows in CSV for {grade_date} (latest: {latest})")
        return {"wins": 0, "losses": 0, "dnps": 0}

    results_map    = {}
    players_in_box = set()
    for _, r in played.iterrows():
        name = r["PLAYER_NAME"]
        players_in_box.add(name)
        results_map[name] = r.to_dict()

    print(f"  CSV: {len(played)} played rows, {len(players_in_box)} players")

    wins = losses = dnps = 0

    for path in (FILE_TODAY, FILE_SEASON_2526):
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        changed = False

        for play in data:
            if play.get("date") != grade_date:
                continue
            if play.get("result") in ("WIN", "LOSS", "DNP"):
                continue  # immutable

            pname   = play.get("player", "")
            line    = float(play.get("line", 20))
            dr      = str(play.get("direction", ""))
            is_over  = "OVER"  in dr.upper() and "LEAN" not in dr.upper()
            is_under = "UNDER" in dr.upper() and "LEAN" not in dr.upper()

            box = results_map.get(pname)

            if pname not in players_in_box or box is None:
                play["result"]          = "DNP"
                play["actualPts"]       = None
                play["actualMin"]       = 0
                post, lt                = generate_post_match_reason(play)
                play["postMatchReason"] = post
                play["lossType"]        = "DNP"
                dnps += 1
            else:
                actual     = float(box.get("PTS", 0))
                actual_min = float(box.get("MIN_NUM", 0))
                play["actualPts"] = actual
                play["actualMin"] = round(actual_min, 1)
                play["delta"]     = round(actual - line, 1)

                if is_over:
                    win = actual > line
                elif is_under:
                    win = actual <= line
                else:
                    win = (actual > line and "OVER" in dr) or \
                          (actual <= line and "UNDER" in dr)

                play["result"] = "WIN" if win else "LOSS"
                if win: wins += 1
                else:   losses += 1

                post, lt = generate_post_match_reason(play, {
                    "actual_pts":    actual,
                    "actual_min":    actual_min,
                    "actual_fga":    box.get("FGA", 0),
                    "actual_fgm":    box.get("FGM", 0),
                    "actual_fg_pct": box.get("FGM", 0) / max(box.get("FGA", 1), 1),
                })
                play["postMatchReason"] = post
                play["lossType"]        = lt

            changed = True

        if changed:
            with open(path, "w") as f:
                json.dump(clean_json(data), f, indent=2)

    print(f"  Graded: W:{wins}  L:{losses}  DNP:{dnps}")

    if not no_retrain:
        from model_trainer import train_and_save
        print("  Retraining models...")
        train_and_save(FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST)

    return {"wins": wins, "losses": losses, "dnps": dnps}


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT  (daily Batches 1-4)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_predict(batch_num: int = 2):
    """Run a prediction batch (1-4). Default: 2."""
    import importlib
    sys.argv = ["batch_predict.py", str(batch_num)]
    import batch_predict
    importlib.reload(batch_predict)
    batch_predict.main()


# ─────────────────────────────────────────────────────────────────────────────
# RETRAIN
# ─────────────────────────────────────────────────────────────────────────────

def cmd_retrain():
    """Retrain V14 models from source CSVs only."""
    from model_trainer import train_and_save
    print(f"  Retraining {VERSION} models...")
    train_and_save(FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST)
    print("  ✓ Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# DVP / H2H
# ─────────────────────────────────────────────────────────────────────────────

def cmd_dvp():
    """Rebuild DVP rankings from game log."""
    from dvp_updater import compute_and_save_dvp
    compute_and_save_dvp(FILE_GL_2526, FILE_DVP)


def cmd_h2h():
    """Rebuild H2H database from both game logs."""
    from h2h_builder import build_h2h
    build_h2h(FILE_GL_2425, FILE_GL_2526, FILE_H2H)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK
# ─────────────────────────────────────────────────────────────────────────────

def cmd_injury():
    """Fetch latest injury report and update data/injuries_current.json."""
    from injury_ingest import fetch_and_store, load_injury_state
    status   = "--status" in sys.argv
    date_arg = next((sys.argv[i+1] for i,a in enumerate(sys.argv) if a == "--date"), None)
    if status:
        state = load_injury_state(date_str=date_arg)
        if not state:
            print("  No injury data available."); return
        out   = [v for v in state.values() if v.get("status_normalized") == "OUT"]
        doubt = [v for v in state.values() if v.get("status_normalized") == "DOUBTFUL"]
        ques  = [v for v in state.values() if v.get("status_normalized") == "QUESTIONABLE"]
        print(f"\n  Injury report — {date_arg or 'today'}")
        print(f"  {'─'*55}")
        print(f"  OUT ({len(out)}):")
        for v in out[:15]:
            print(f"    {v.get('player_name',''):<25} {v.get('team','')}  {v.get('reason_raw','')[:35]}")
        print(f"  DOUBTFUL ({len(doubt)}):")
        for v in doubt[:10]:
            print(f"    {v.get('player_name',''):<25} {v.get('team','')}  {v.get('reason_raw','')[:35]}")
        print(f"  QUESTIONABLE ({len(ques)}):")
        for v in ques[:10]:
            print(f"    {v.get('player_name',''):<25} {v.get('team','')}  {v.get('reason_raw','')[:35]}")
    else:
        result = fetch_and_store(date_str=date_arg)
        print(f"  Injury: {result.get('status','?')} | {result.get('n_rows',0)} players | {result.get('n_changed',0)} changes")


def cmd_check():
    """Data integrity report."""
    import datetime as _dt
    import pandas as pd
    from config import FILE_AUDIT

    print(f"\n  PropEdge {VERSION} — Data Integrity Check")
    print(f"  {'─'*55}")

    # Source files
    for label, path in [
        ("GL 2025-26",  FILE_GL_2526),
        ("GL 2024-25",  FILE_GL_2425),
        ("H2H DB",      FILE_H2H),
        ("Excel props", FILE_PROPS),
    ]:
        if path.exists():
            try:
                if path.suffix == ".csv":
                    df = pd.read_csv(path)
                    if "GAME_DATE" in df.columns:
                        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
                        extra = f"{len(df):,} rows | latest: {df['GAME_DATE'].max().date()}"
                    else:
                        extra = f"{len(df):,} rows"
                elif path.suffix == ".xlsx":
                    df = pd.read_excel(path, sheet_name="Player_Points_Props")
                    df["Date"] = pd.to_datetime(df["Date"])
                    extra = (f"{len(df):,} rows | "
                             f"{df['Date'].min().date()} → {df['Date'].max().date()}")
                else:
                    extra = ""
            except Exception:
                extra = f"{path.stat().st_size/1024:.0f}KB"
            print(f"  ✓ {label:<15} {extra}")
        else:
            print(f"  ✗ {label:<15} MISSING — {path.name}")

    print(f"  {'─'*55}")

    # Models
    for label, path in [
        ("Classifier",  FILE_CLF),
        ("Regressor",   FILE_REG),
        ("Calibrator",  FILE_CAL),
        ("Trust",       FILE_TRUST),
    ]:
        if path.exists():
            age = (_dt.datetime.now() -
                   _dt.datetime.fromtimestamp(path.stat().st_mtime))
            print(f"  ✓ Model {label:<12} {path.stat().st_size/1024:>7.0f}KB  "
                  f"({int(age.total_seconds()/3600)}h ago)")
        else:
            print(f"  ✗ Model {label:<12} MISSING — run `python3 run.py setup`")

    print(f"  {'─'*55}")

    # Data JSONs
    for label, path in [
        ("season_2025_26", FILE_SEASON_2526),
        ("today.json",     FILE_TODAY),
    ]:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if not data:
                print(f"  ⚠ {label:<18} EMPTY — run `python3 run.py setup`")
                continue
            dates  = sorted(set(p["date"] for p in data))
            wins   = sum(1 for p in data if p.get("result") == "WIN")
            losses = sum(1 for p in data if p.get("result") == "LOSS")
            graded = wins + losses
            ungrad = sum(1 for p in data if not p.get("result"))
            hr     = f"{wins/graded*100:.1f}%" if graded else "—"
            print(f"  ✓ {label:<18} {len(data):>6} plays | "
                  f"{dates[0]} → {dates[-1]} | "
                  f"W:{wins} L:{losses} HR:{hr} | ungraded:{ungrad}")
        else:
            print(f"  ✗ {label:<18} MISSING")

    print(f"  {'─'*55}")

    # Audit
    try:
        audit  = pd.read_csv(FILE_AUDIT)
        alerts = audit[audit["event"].str.contains("FAIL|ALERT|ABORT", na=False)]
        print(f"  Audit: {len(audit):,} events | {len(alerts)} warnings")
        for _, row in alerts.tail(3).iterrows():
            print(f"    ⚠ [{row.get('ts','')}] {row.get('event','')}: "
                  f"{str(row.get('detail',''))[:60]}")
    except Exception:
        print("  Audit: no log yet")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def cmd_install():
    """Install launchd scheduler agents (macOS)."""
    from scheduler import install
    install()


def cmd_uninstall():
    """Remove launchd scheduler agents."""
    from scheduler import uninstall
    uninstall()


def cmd_status():
    """Show scheduler agent status + model status + integrity check."""
    from scheduler import status, show_next
    status()
    show_next()
    cmd_check()


def cmd_weekend():
    """Preview weekend batch schedule for a given date."""
    from scheduler import compute_weekend_times
    from datetime import datetime
    from zoneinfo import ZoneInfo
    date_arg = (
        sys.argv[2] if len(sys.argv) > 2
        else datetime.now(ZoneInfo("Europe/London")).strftime("%Y-%m-%d")
    )
    print(f"\n  Weekend schedule preview for {date_arg}:")
    times = compute_weekend_times(date_arg)
    for bk, (h, m) in times.items():
        print(f"    {bk.upper()}: {h:02d}:{m:02d} UK")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DISPATCH
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    print(f"\n  PropEdge {VERSION}  —  {cmd}")

    dispatch = {
        "setup":      cmd_setup,
        "generate":   cmd_setup,      # alias — same as setup
        "grade":      cmd_grade,
        "grade-csv":  cmd_grade_from_csv,
        "retrain":    cmd_retrain,
        "dvp":        cmd_dvp,
        "h2h":        cmd_h2h,
        "injury":     cmd_injury,
        "check":      cmd_check,
        "status":     cmd_status,
        "install":    cmd_install,
        "uninstall":  cmd_uninstall,
        "weekend":    cmd_weekend,
    }

    if cmd == "predict":
        n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 2
        cmd_predict(n)
    elif cmd in dispatch:
        dispatch[cmd]()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
