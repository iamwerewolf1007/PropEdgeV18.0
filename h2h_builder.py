"""
PropEdge V18.0 — h2h_builder.py
Full vectorised rebuild of h2h_database.csv from both season CSVs.
Called by Batch 0 after append_gamelogs().
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import FILE_GL_2425, FILE_GL_2526, FILE_H2H
from rolling_engine import filter_played


def build_h2h(
    file_gl_2425: Path = FILE_GL_2425,
    file_gl_2526: Path = FILE_GL_2526,
    output_path: Path = FILE_H2H,
) -> None:
    """
    Build h2h_database.csv from both season CSVs.
    Produces one row per (PLAYER_NAME, OPPONENT) pair.
    """
    print("  Building H2H database...")

    gl24 = pd.read_csv(file_gl_2425, parse_dates=["GAME_DATE"])
    gl25 = pd.read_csv(file_gl_2526, parse_dates=["GAME_DATE"])
    gl = pd.concat([gl24, gl25], ignore_index=True)

    played = filter_played(gl)
    played = played.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

    # ── Per-player overall stats ──────────────────────────────────────────────
    overall = (
        played.groupby("PLAYER_NAME")
        .agg(
            overall_avg_pts=("PTS", "mean"),
            overall_avg_fga=("FGA", "mean"),
            overall_avg_min=("MIN_NUM", "mean"),
            overall_avg_fgm=("FGM", "mean"),
            overall_avg_fta=("FTA", "mean"),
            overall_usage=("USAGE_APPROX", "mean"),
        )
        .reset_index()
    )
    played = played.merge(overall, on="PLAYER_NAME", how="left")

    # ── H2H groupby ──────────────────────────────────────────────────────────
    results = []
    for (player, opp), grp in played.groupby(["PLAYER_NAME", "OPPONENT"]):
        grp = grp.sort_values("GAME_DATE")
        n = len(grp)

        avg_pts    = grp["PTS"].mean()
        median_pts = grp["PTS"].median()
        std_pts    = grp["PTS"].std() if n > 1 else 0.0
        avg_fga    = grp["FGA"].mean()
        avg_min    = grp["MIN_NUM"].mean()
        avg_fta    = grp["FTA"].mean()

        # Efficiency
        tot_fgm = grp["FGM"].sum()   if "FGM" in grp else 0
        tot_fga = grp["FGA"].sum()
        tot_fta = grp["FTA"].sum()
        tot_pts = grp["PTS"].sum()
        fg_pct  = tot_fgm / max(tot_fga, 1)
        ts_pct  = tot_pts / max(2 * (tot_fga + 0.44 * tot_fta), 1)

        # Deviations from overall
        oa_pts = grp["overall_avg_pts"].iloc[0]
        oa_fga = grp["overall_avg_fga"].iloc[0]
        oa_min = grp["overall_avg_min"].iloc[0]

        ts_vs_overall  = ts_pct - (oa_pts / max(2 * (oa_fga + 0.44 * grp["overall_avg_fta"].iloc[0]), 1))
        fga_vs_overall = avg_fga - oa_fga
        min_vs_overall = avg_min - oa_min

        # Recent form
        l3 = grp.tail(3)["PTS"].mean() if n >= 3 else avg_pts
        l5 = grp.tail(5)["PTS"].mean() if n >= 5 else avg_pts
        pts_trend = l3 - avg_pts

        # Season split
        current_season = grp[grp["SEASON"] == "2025-26"] if "SEASON" in grp.columns else grp.tail(max(1, n // 2))
        cur_games = len(current_season)
        cur_avg   = current_season["PTS"].mean() if cur_games > 0 else avg_pts

        # Home/away
        home_games = grp[grp["IS_HOME"] == 1]
        away_games = grp[grp["IS_HOME"] == 0]
        home_avg = home_games["PTS"].mean() if len(home_games) > 0 else avg_pts
        away_avg = away_games["PTS"].mean() if len(away_games) > 0 else avg_pts

        # Confidence score: sample size × consistency
        sample_frac = min(n / 10.0, 1.0)
        consistency = 1.0 / (1.0 + (std_pts / max(avg_pts, 1)))
        confidence  = round(0.6 * sample_frac + 0.4 * consistency, 4)

        # Scoring profile
        if abs(fga_vs_overall) > 1.5:
            profile = "VOLUME"
        elif abs(min_vs_overall) > 2.0:
            profile = "MINUTES"
        elif abs(ts_vs_overall) > 0.03:
            profile = "EFFICIENCY"
        elif abs(fga_vs_overall) > 0.5 and abs(ts_vs_overall) > 0.01:
            profile = "MIXED"
        else:
            profile = "NEUTRAL"

        # Days since last H2H
        days_since = (pd.Timestamp.now() - grp["GAME_DATE"].max()).days

        results.append({
            "PLAYER_NAME": player,
            "OPPONENT": opp,
            "H2H_GAMES": n,
            "H2H_AVG_PTS": round(avg_pts, 2),
            "H2H_MEDIAN_PTS": round(median_pts, 2),
            "H2H_STD_PTS": round(std_pts, 2),
            "H2H_FG_PCT": round(fg_pct, 4),
            "H2H_TS_PCT": round(ts_pct, 4),
            "H2H_TS_VS_OVERALL": round(ts_vs_overall, 4),
            "H2H_FGA_VS_OVERALL": round(fga_vs_overall, 2),
            "H2H_MIN_VS_OVERALL": round(min_vs_overall, 2),
            "H2H_AVG_FGA": round(avg_fga, 2),
            "H2H_AVG_MIN": round(avg_min, 2),
            "L3_H2H_AVG_PTS": round(l3, 2),
            "L5_H2H_AVG_PTS": round(l5, 2),
            "H2H_PTS_TREND": round(pts_trend, 2),
            "H2H_CURRENT_SEASON_AVG_PTS": round(cur_avg, 2),
            "H2H_GAMES_CURRENT_SEASON": cur_games,
            "H2H_HOME_AVG_PTS": round(home_avg, 2),
            "H2H_AWAY_AVG_PTS": round(away_avg, 2),
            "H2H_CONFIDENCE": confidence,
            "H2H_SCORING_PROFILE": profile,
            "DAYS_SINCE_LAST_H2H": days_since,
        })

    h2h_df = pd.DataFrame(results)
    # Deduplicate (keep last — most recent matchup is most relevant)
    h2h_df = h2h_df.drop_duplicates(subset=["PLAYER_NAME", "OPPONENT"], keep="last")
    h2h_df.to_csv(output_path, index=False)
    print(f"  H2H database: {len(h2h_df):,} rows → {output_path.name}")
