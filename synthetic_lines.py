"""
PropEdge V18.0 — Synthetic Line Generator
==========================================
Generates realistic prop lines for 2024-25 season backtest.
Mimics sportsbook line-setting from rolling history.

Called by generate_season_json.py for the 2024-25 backtest season.
"""

import numpy as np
import pandas as pd


def generate_season_lines(game_logs_df: pd.DataFrame, season: str = "2024-25") -> list[dict]:
    """
    Generate synthetic prop lines for an entire season of game logs.

    Methodology per row:
      1. Require ≥ 5 prior games with known PTS
      2. Baseline = mean of last 30 played games
      3. Synthetic line = round(baseline × 2) / 2, floored at 3.5
      4. min/max line = ±0.5 around the line (mimic book spread)

    Returns list of prop dicts compatible with generate_season_json.py
    (same schema as _load_excel_props output).
    """
    df = game_logs_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Only consider rows where the player actually played
    played_mask = (df["DNP"].fillna(0) == 0) & (df["MIN_NUM"].fillna(0) > 0)
    played = df[played_mask].copy()
    played = played.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

    props: list[dict] = []

    for pname, grp in played.groupby("PLAYER_NAME"):
        pts_list = grp["PTS"].fillna(0).tolist()
        dates    = grp["GAME_DATE"].tolist()
        opps     = grp["OPPONENT"].fillna("").tolist()
        homes    = grp["IS_HOME"].fillna(0).astype(int).tolist()
        history: list[float] = []

        for i, (d, pts, opp, home) in enumerate(zip(dates, pts_list, opps, homes)):
            if i >= 5 and len(history) >= 5:
                l30        = float(np.mean(history[-30:]))
                synth_line = max(3.5, round(l30 * 2) / 2)

                # Derive game string
                team = str(grp.iloc[i].get("GAME_TEAM_ABBREVIATION", ""))
                if home:
                    game_str   = f"{opp} @ {team}"
                    home_team  = team
                    away_team  = opp
                else:
                    game_str   = f"{team} @ {opp}"
                    home_team  = opp
                    away_team  = team

                props.append({
                    "player":    str(pname),
                    "date":      d,
                    "line":      synth_line,
                    "min_line":  synth_line - 0.5,
                    "max_line":  synth_line + 0.5,
                    "over_odds": -110,
                    "under_odds": -110,
                    "game":      game_str,
                    "home":      home_team,
                    "away":      away_team,
                    "game_time": "",
                    "books":     1,
                    "season":    season,
                    "source":    "synthetic",
                })

            history.append(float(pts))

    return props
