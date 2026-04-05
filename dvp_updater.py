"""
PropEdge V18.0 — dvp_updater.py
Computes live DVP rankings from game log CSV.
Called at start of each batch predict and after grading in Batch 0.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from config import FILE_GL_2526, FILE_DVP, get_pos_group, invalidate_dvp_cache

# Hardcoded season-start fallback (used when CSV data is sparse)
DVP_RAW_FALLBACK = {
    ("ATL", "Guard"): 24, ("ATL", "Forward"): 22, ("ATL", "Center"): 20,
    ("BOS", "Guard"):  5, ("BOS", "Forward"):  7, ("BOS", "Center"):  6,
    ("BKN", "Guard"): 28, ("BKN", "Forward"): 26, ("BKN", "Center"): 25,
    ("CHA", "Guard"): 22, ("CHA", "Forward"): 20, ("CHA", "Center"): 18,
    ("CHI", "Guard"): 18, ("CHI", "Forward"): 16, ("CHI", "Center"): 14,
    ("CLE", "Guard"):  8, ("CLE", "Forward"):  9, ("CLE", "Center"):  7,
    ("DAL", "Guard"): 12, ("DAL", "Forward"): 11, ("DAL", "Center"): 13,
    ("DEN", "Guard"): 15, ("DEN", "Forward"): 14, ("DEN", "Center"): 12,
    ("DET", "Guard"): 26, ("DET", "Forward"): 24, ("DET", "Center"): 23,
    ("GSW", "Guard"): 16, ("GSW", "Forward"): 18, ("GSW", "Center"): 17,
    ("HOU", "Guard"): 20, ("HOU", "Forward"): 19, ("HOU", "Center"): 21,
    ("IND", "Guard"): 17, ("IND", "Forward"): 17, ("IND", "Center"): 16,
    ("LAC", "Guard"):  9, ("LAC", "Forward"): 10, ("LAC", "Center"):  9,
    ("LAL", "Guard"): 14, ("LAL", "Forward"): 13, ("LAL", "Center"): 15,
    ("MEM", "Guard"): 21, ("MEM", "Forward"): 21, ("MEM", "Center"): 19,
    ("MIA", "Guard"):  6, ("MIA", "Forward"):  8, ("MIA", "Center"):  8,
    ("MIL", "Guard"): 13, ("MIL", "Forward"): 12, ("MIL", "Center"): 11,
    ("MIN", "Guard"):  3, ("MIN", "Forward"):  4, ("MIN", "Center"):  3,
    ("NOP", "Guard"): 25, ("NOP", "Forward"): 25, ("NOP", "Center"): 24,
    ("NYK", "Guard"): 10, ("NYK", "Forward"):  6, ("NYK", "Center"):  5,
    ("OKC", "Guard"):  7, ("OKC", "Forward"):  5, ("OKC", "Center"):  4,
    ("ORL", "Guard"): 11, ("ORL", "Forward"): 15, ("ORL", "Center"): 10,
    ("PHI", "Guard"): 27, ("PHI", "Forward"): 27, ("PHI", "Center"): 26,
    ("PHX", "Guard"): 19, ("PHX", "Forward"): 23, ("PHX", "Center"): 22,
    ("POR", "Guard"): 23, ("POR", "Forward"): 28, ("POR", "Center"): 27,
    ("SAC", "Guard"): 29, ("SAC", "Forward"): 29, ("SAC", "Center"): 28,
    ("SAS", "Guard"): 30, ("SAS", "Forward"): 30, ("SAS", "Center"): 30,
    ("TOR", "Guard"): 28, ("TOR", "Forward"): 26, ("TOR", "Center"): 29,
    ("UTA", "Guard"): 22, ("UTA", "Forward"): 20, ("UTA", "Center"): 20,
    ("WAS", "Guard"): 30, ("WAS", "Forward"): 30, ("WAS", "Center"): 29,
}


def compute_and_save_dvp(
    file_gl: Path = FILE_GL_2526,
    output_path: Path = FILE_DVP,
    recent_n: int = 20,
    min_games: int = 5,
):
    """
    Read last `recent_n` played games per player from the game log.
    Compute average PTS allowed per team per position group.
    Rank all teams per position (higher rank = weaker defence).
    Blend with hardcoded fallback for sparse teams.
    Write dvp_rankings.json and invalidate the in-process cache.
    """
    try:
        df = pd.read_csv(file_gl, parse_dates=["GAME_DATE"])
        df["DNP"] = df["DNP"].fillna(0)
        played = df[(df["DNP"] == 0) & (df["MIN_NUM"].fillna(0) > 0)].copy()
        played = played.sort_values("GAME_DATE")

        # Map positions
        played["pos_group"] = played["PLAYER_POSITION"].apply(get_pos_group)

        # Take last N games per player
        recent = played.groupby("PLAYER_NAME").tail(recent_n)

        # Average PTS allowed per opponent × position
        opp_pos_avg = (
            recent.groupby(["OPPONENT", "pos_group"])["PTS"]
            .agg(["mean", "count"])
            .reset_index()
        )

        # Rank per position (highest avg PTS allowed = rank 30 = weakest)
        rankings: dict[str, dict] = {}
        for pos_group in ["Guard", "Forward", "Center"]:
            subset = opp_pos_avg[opp_pos_avg["pos_group"] == pos_group].copy()
            subset = subset.sort_values("mean", ascending=True).reset_index(drop=True)
            # rank 1 = best defence, 30 = worst
            subset["rank"] = range(1, len(subset) + 1)

            for _, row in subset.iterrows():
                team = str(row["OPPONENT"])
                count = int(row["count"])
                live_rank = int(row["rank"])
                fallback_rank = DVP_RAW_FALLBACK.get((team, pos_group), 15)

                if count >= min_games:
                    final_rank = live_rank
                elif count >= 1:
                    # Blend: 70% live + 30% fallback
                    final_rank = round(0.7 * live_rank + 0.3 * fallback_rank)
                else:
                    final_rank = fallback_rank

                key = f"{team}|{pos_group}"
                rankings[key] = {"rank": final_rank, "games": count, "live_rank": live_rank}

        # Ensure all fallback teams present
        for (team, pos_group), rank in DVP_RAW_FALLBACK.items():
            key = f"{team}|{pos_group}"
            if key not in rankings:
                rankings[key] = {"rank": rank, "games": 0, "live_rank": rank}

        # Save flat format: key → rank (for get_dvp() compatibility)
        flat = {k: v["rank"] for k, v in rankings.items()}
        with open(output_path, "w") as f:
            json.dump(flat, f, indent=2)

        invalidate_dvp_cache()
        print(f"  DVP updated: {len(flat)} entries written → {output_path.name}")

    except Exception as e:
        print(f"  ⚠ DVP update failed: {e}. Using cached/fallback values.")
