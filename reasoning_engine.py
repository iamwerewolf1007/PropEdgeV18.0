"""
PropEdge V18.0 — reasoning_engine.py
Generates pre-match (5-part) and post-match (7-part) plain-English narratives.
V14 additions: mean reversion risk note, early-season confidence note, loss type for TREND_REVERSAL.
"""

from __future__ import annotations
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# PRE-MATCH REASONING
# ─────────────────────────────────────────────────────────────────────────────

def generate_pre_match_reason(play: dict) -> str:
    """5-part pre-match narrative. Returns a single string."""
    name      = play.get("player", "Player")
    line      = float(play.get("line", 20))
    direction = play.get("direction", "OVER")
    tier      = play.get("tierLabel", "T2")
    pred_pts  = play.get("predPts")
    pred_gap  = play.get("predGap", 0)
    conf      = play.get("conf", 0.6)
    flags     = play.get("flags", 5)
    flag_d    = play.get("flagDetails", [])

    L30  = float(play.get("l30",  line))
    L10  = float(play.get("l10",  L30))
    L5   = float(play.get("l5",   L10))
    L3   = float(play.get("l3",   L5))
    std10 = float(play.get("std10", 5))
    hr10  = float(play.get("hr10", 0.5))
    hr30  = float(play.get("hr30", 0.5))
    min_l10 = float(play.get("min_l10", 30))
    min_l30 = float(play.get("min_l30", 30))

    momentum = L5 - L30
    is_over  = "OVER" in direction.upper()
    is_lean  = "LEAN" in direction.upper()

    opp      = play.get("opponent", "")
    dvp      = int(play.get("defP_dynamic", 15))
    pace     = int(play.get("pace_rank",    15))
    h2h_avg  = float(play.get("h2h_avg",   0))
    h2h_g    = int(play.get("h2h_games",   0))
    h2h_ts   = float(play.get("h2h_ts_dev", 0))

    early_w  = float(play.get("early_season_weight", 1.0))
    mean_rev = float(play.get("mean_reversion_risk", 0.0))

    parts = []

    # ── S1: Lead signal ───────────────────────────────────────────────────────
    vol = L30 - line
    candidates = []
    if abs(vol) >= 1:
        score = abs(vol) * (1.5 if (is_over and vol > 0) or (not is_over and vol < 0) else 1.1)
        candidates.append((score, f"{name}'s L30 average of {L30:.1f}pts sits {'above' if vol>0 else 'below'} the {line} line by {abs(vol):.1f}pts."))
    if h2h_g >= 3 and abs(h2h_avg - line) >= 1:
        score = abs(h2h_avg - line) * (1.4 if (is_over and h2h_avg > line) or (not is_over and h2h_avg < line) else 1.1)
        candidates.append((score, f"Against {opp}, {name} averages {h2h_avg:.1f}pts over {h2h_g} matchups ({'above' if h2h_avg > line else 'below'} the {line} line)."))
    if abs(momentum) >= 2:
        score = abs(momentum) * (1.3 if (is_over and momentum > 0) or (not is_over and momentum < 0) else 1.0)
        candidates.append((score, f"{name} is {'trending up' if momentum>0 else 'trending down'} {abs(momentum):.1f}pts vs their L30 baseline (L5: {L5:.1f})."))
    if std10 <= 4:
        candidates.append((6 - std10, f"{name} is highly consistent with a std deviation of just {std10:.1f}pts over their last 10 games."))

    if candidates:
        parts.append(sorted(candidates, reverse=True)[0][1])
    else:
        parts.append(f"{name}'s L30 of {L30:.1f}pts vs the {line} line.")

    # ── S2: Matchup context ───────────────────────────────────────────────────
    ctx = []
    if h2h_g >= 3 and abs(h2h_ts) > 0.02:
        ts_dir = "better" if h2h_ts > 0 else "worse"
        ctx.append(f"Shooting efficiency runs {ts_dir} vs {opp} (TS% {h2h_ts:+.1%} vs overall).")
    if abs(min_l10 - min_l30) >= 2:
        m_dir = "increasing" if min_l10 > min_l30 else "decreasing"
        ctx.append(f"Minutes are {m_dir} (L10: {min_l10:.0f} vs L30: {min_l30:.0f}).")
    if dvp >= 22:
        ctx.append(f"{opp} rank {dvp}/30 in defensive vulnerability — one of the weaker units in the league.")
    elif dvp <= 8:
        ctx.append(f"{opp} rank {dvp}/30 defensively — a tough matchup.")
    if pace >= 22:
        ctx.append(f"{opp} play at a fast pace (rank {pace}/30) which typically boosts scoring volume.")
    if ctx:
        parts.append(" ".join(ctx[:2]))

    # ── S3: Signal audit ─────────────────────────────────────────────────────
    agree_names   = [fd["name"] for fd in flag_d if fd.get("agrees")]
    disagree_names = [fd["name"] for fd in flag_d if not fd.get("agrees") and fd.get("value", 0) != 0]
    if flags >= 8:
        parts.append(f"Full signal consensus: {flags}/10 signals agree ({', '.join(agree_names[:4])}).")
    elif flags >= 6:
        parts.append(f"Strong signal alignment: {flags}/10 in favour. Opposing: {', '.join(disagree_names[:2]) or 'none'}.")
    else:
        parts.append(f"Mixed signals: {flags}/10 agree. Key counter-signals: {', '.join(disagree_names[:3]) or 'none'}.")

    # ── S4: Model projection ──────────────────────────────────────────────────
    if pred_pts is not None:
        s4 = f"V14 model projects {pred_pts:.1f}pts ({'+' if pred_gap >= 0 else ''}{pred_pts - line:.1f} vs line). Fusion confidence: {conf:.0%} [{tier}]."
        if early_w < 0.5:
            s4 += f" [Early season — only {play.get('_n_games', '?')} games in window; reduced confidence applied.]"
        parts.append(s4)

    # ── S5: Risk factor ───────────────────────────────────────────────────────
    risk = None
    if mean_rev >= 1.0 and is_over and momentum > 6:
        risk = f"[Reversion risk: L5 is {momentum:+.1f}pts vs L30 — partial mean reversion is likely.]"
    elif mean_rev >= 1.0 and not is_over and momentum < -6:
        risk = f"[Reversion risk: player is in a {abs(momentum):.1f}pt cold spell — bounce-back risk exists.]"
    elif float(play.get("is_long_rest", 0)) == 1:
        risk = f"[Long rest (≥6 days): rust effect may suppress output vs L10 average.]"
    elif std10 > 7:
        risk = f"[High variance: std10={std10:.1f}pts — outcome could deviate significantly from projection.]"
    elif is_over and hr30 < 0.4:
        risk = f"[Low OVER hit rate: only {hr30:.0%} of last 30 games exceeded this line.]"
    elif line >= 25 and is_over:
        risk = f"[High-line shading: bookmakers typically shade {line}+ lines — adjusted confidence.]"
    if risk:
        parts.append(risk)

    if is_lean:
        return "[Low conviction — lean only] " + " ".join(parts)

    # ── V18: Injury context ─────────────────────────────────────────────────
    avail       = play.get("avail", "")
    inj_reason  = play.get("injuryReason", "")
    team_risk   = play.get("teamInjuryRisk", "NONE")
    t_boost     = play.get("teammateBoost", 0)
    inj_changed = play.get("injuryStatusChanged", False)

    if avail == "QUESTIONABLE":
        parts.append(f"Tagged QUESTIONABLE ({inj_reason[:40]})" if inj_reason else "Tagged QUESTIONABLE — minutes uncertain.")
    elif avail == "DOUBTFUL":
        parts.append(f"Listed DOUBTFUL ({inj_reason[:40]}) — heavy confidence penalty applied." if inj_reason else "Listed DOUBTFUL — heavy confidence penalty applied.")
    elif avail == "PROBABLE":
        parts.append("Listed PROBABLE — expected to play with minor flag.")
    if inj_changed:
        parts.append("⚠ Status changed since last report — elevated lineup volatility.")
    if t_boost > 0:
        parts.append(f"Teammate ruled OUT — usage boost candidate (+{t_boost:.1f}pt role load est).")
    if team_risk == "HIGH":
        parts.append("Team injury risk HIGH — lineup uncertainty elevated.")
    elif team_risk == "MEDIUM":
        parts.append("Team injury risk MEDIUM — monitor pre-tip.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# POST-MATCH REASONING
# ─────────────────────────────────────────────────────────────────────────────

def generate_post_match_reason(play: dict, box_data: Optional[dict] = None) -> tuple[str, str]:
    """
    7-part post-match narrative. Returns (narrative_str, loss_type_str).
    loss_type is computed fresh here — never read from stored play dict.
    """
    name       = play.get("player", "Player")
    line       = float(play.get("line", 20))
    direction  = play.get("direction", "OVER")
    pred_pts   = play.get("predPts")
    result     = play.get("result", "")

    box = box_data or {}
    actual_pts  = float(box.get("actual_pts",  play.get("actualPts",  0)) or 0)
    actual_min  = float(box.get("actual_min",  play.get("actualMin",  0)) or 0)
    actual_fga  = float(box.get("actual_fga",  0) or 0)
    actual_fgm  = float(box.get("actual_fgm",  0) or 0)
    actual_fg   = actual_fgm / max(actual_fga, 1)

    l10_min = float(play.get("min_l10", 30))
    l10_fga = float(play.get("fga_l10", 10))
    l10_fg  = float(play.get("l10_fg_pct", 0.45))
    flags   = int(play.get("flags", 5))
    momentum = float(play.get("l5", line)) - float(play.get("l30", line))

    is_over = "OVER" in direction.upper()
    delta   = actual_pts - line
    margin  = abs(delta)
    won     = result == "WIN"
    integrity_flag = box.get("integrity_flag", "")

    # ── Loss type (computed — not read from stored field) ─────────────────────
    loss_type = "MODEL_CORRECT"
    if not won:
        if actual_min < l10_min - 4:
            loss_type = "MINUTES_SHORTFALL"
        elif actual_fga > 0 and abs(actual_fg - l10_fg) >= 0.05:
            loss_type = "SHOOTING_VARIANCE"
        elif margin <= 2:
            loss_type = "CLOSE_CALL"
        elif margin > 8:
            loss_type = "BLOWOUT_EFFECT"
        elif abs(momentum) > 4:
            loss_type = "TREND_REVERSAL"
        elif flags >= 7:
            loss_type = "MODEL_FAILURE_CONSENSUS"
        else:
            loss_type = "MODEL_FAILURE_GENERAL"

    parts = []

    # S1: Outcome
    outcome_word = "HIT ✓" if won else "MISSED ✗"
    parts.append(
        f"{outcome_word} — {name} scored {actual_pts:.0f}pts vs {line} line "
        f"({'over' if delta > 0 else 'under'} by {abs(delta):.1f}pts, called {direction})."
    )

    # S2: Minutes
    if actual_min > 0:
        min_diff = actual_min - l10_min
        if abs(min_diff) >= 3:
            parts.append(
                f"Minutes: {actual_min:.0f} played vs {l10_min:.0f} L10 average "
                f"({'more' if min_diff > 0 else 'fewer'} than expected by {abs(min_diff):.0f}min)."
            )

    # S3: Efficiency
    if actual_fga > 0:
        fg_diff = actual_fg - l10_fg
        if abs(fg_diff) >= 0.05:
            parts.append(
                f"Shot efficiency: {actual_fg:.0%} FG% vs {l10_fg:.0%} L10 average — "
                f"{'hot' if fg_diff > 0 else 'cold'} shooting."
            )

    # S4: Signal audit
    if won:
        parts.append(f"Signal alignment ({flags}/10 agreed) proved correct.")
    else:
        opp_cnt = 10 - flags
        parts.append(
            f"Despite {flags}/10 signals agreeing, {opp_cnt} counter-signals proved relevant. "
            f"Loss type: {loss_type.replace('_', ' ').title()}."
        )

    # S5: Model accuracy
    if pred_pts is not None:
        model_err = abs(actual_pts - pred_pts)
        model_correct = (is_over and actual_pts > line) or (not is_over and actual_pts <= line)
        parts.append(
            f"Model projection: {pred_pts:.1f}pts (error: {model_err:.1f}pts, "
            f"direction {'correct' if model_correct else 'incorrect'})."
        )

    # S6: Loss classification explanation
    _loss_desc = {
        "MODEL_CORRECT":           "Prediction confirmed.",
        "CLOSE_CALL":              f"Margin of {margin:.1f}pts — nearly correct. Statistical noise at the boundary.",
        "MINUTES_SHORTFALL":       f"Player received {actual_min:.0f} min vs expected {l10_min:.0f} — usage was restricted.",
        "SHOOTING_VARIANCE":       f"Shooting efficiency deviated sharply — luck/defence disrupted scoring efficiency.",
        "BLOWOUT_EFFECT":          f"Margin of {margin:.0f}pts suggests game script disrupted normal patterns.",
        "TREND_REVERSAL":          f"Player showed {momentum:+.1f}pt momentum vs baseline — partial mean reversion occurred.",
        "MODEL_FAILURE_CONSENSUS": f"All {flags} signals agreed but the outcome was anomalous — possible outlier game.",
        "MODEL_FAILURE_GENERAL":   "Genuine model uncertainty — no dominant structural cause identified.",
    }
    parts.append(_loss_desc.get(loss_type, ""))

    # S7: Learning note
    opp = play.get("opponent", "this opponent")
    if loss_type == "CLOSE_CALL":
        parts.append(f"Consider treating sub-1pt gap plays vs {opp} as leans going forward.")
    elif loss_type == "MINUTES_SHORTFALL":
        parts.append(f"Monitor {name}'s minutes situation — rotation changes may be underway.")
    elif loss_type == "TREND_REVERSAL":
        parts.append(f"Extreme momentum ({momentum:+.1f}pt) plays carry reversion risk — V14 penalises these automatically.")
    elif won:
        parts.append(f"Model and signal alignment validated vs {opp}.")

    if integrity_flag:
        parts.append(f"⚠ Data integrity note: {integrity_flag}")

    # ── V18: Post-match injury context ─────────────────────────────────────
    avail       = play.get("avail", "")
    inj_changed = play.get("injuryStatusChanged", False)
    t_boost     = play.get("teammateBoost", 0)
    if avail in ("QUESTIONABLE","DOUBTFUL") and not won:
        parts.append("Questionable/Doubtful tag may have reflected true minutes limitation.")
    elif avail in ("QUESTIONABLE","DOUBTFUL") and won:
        parts.append("Player overcame injury tag — role intact despite pre-game concern.")
    if inj_changed:
        parts.append("Late status change detected — edge may have been invalidated by roster news.")
    if t_boost > 0 and won:
        parts.append("Teammate-out usage boost correctly identified — role expansion played out.")
    elif t_boost > 0 and not won:
        parts.append("Teammate-out usage boost did not materialise as expected.")

    return " ".join(p for p in parts if p), loss_type
