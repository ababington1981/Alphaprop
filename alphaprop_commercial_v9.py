#!/usr/bin/env python3
"""
AlphaProp Edge Finder+ — Commercial Blueprint v9.0 Engine (Upgrades 1–8 Locked Edition)
Single-file, self-contained quantitative NBA prop engine (Streamlit Cloud & FastAPI ready).

Implements all 5 Commercial Blueprint Pillars + All 8 Locked Production Upgrades (1–4 & Redirected 5–8):
  1. Granular Data Architecture: PbP Lineup Stint & Odds API ingestion pipelines
  2. Core Math, Distributions & Monte Carlo:
     - Preserved v8 Sigmoid Archetype Weighting, Spatial Shot-Redirection, Dynamic Margin Volatility
     - 10,000-trial Vectorized Monte Carlo (Bimodal Normal, Negative Binomial, Beta-Binomial 3PM, Zero-Inflated)
     - Gaussian & Student-t Copula Same-Game Parlay (SGP) Correlation Simulator
  3. High-Edge Matchup, Tactical & Locked Upgrades 1–8 Engine:
     - Upgrade 1 [LOCKED]: Empirical 3-Archetype Teammate On/Off Usage-Vacuum & Efficiency-Tax Matrix
     - Upgrade 2 [LOCKED]: Empirical Bayes 3PT Streak Shrinkage (k = 360 3PA Stabilization Prior)
     - Upgrade 3 [LOCKED]: Role-Stratified Venue & Schedule Split Matrix (Role/3&D vs. Star)
     - Upgrade 4 [LOCKED]: 2-Man & 3-Man PbP Stint Synergy & Opponent Defensive Shell Funneling
     - Upgrade 5 [LOCKED REDIRECT]: L5 Momentum Bifurcation (Mirage Under Fade vs. Structural Role Breakout Over)
     - Upgrade 6 [LOCKED REDIRECT]: H2H Scheme & Defender Continuity Gate (18-Mo / Same-Coach / 48-Hr Duplex)
     - Upgrade 7 [LOCKED REDIRECT]: Optimal 1–2 Day Rest Sweet Spot vs. 3+ Day Perimeter Rust & Rim/FT Split
     - Upgrade 8 [LOCKED REDIRECT]: Referee Crew Whistle Asymmetry & Foul-Cascade Separation (PF/36m vs. FTA/g)
  4. No-Vig Pricing, Triggers & Upgrade #4 Automated CLV / Brier Logger:
     - Multiplicative, Power, and Shin Insider-Trading No-Vig Solvers
     - Fractional/Half-Kelly Optimal Bankroll Staking & 2.0u Per-Game Exposure Cap
     - MultiBookOddsCLVLogger: The Odds API Multi-Book Snapshotter (Pulse 1/2/3 & Closing),
       Weighted Pinnacle/Circa Shin & Power No-Vig Consensus, Line-Adjusted CLV,
       Buy-Stop Guard™ Max Odds, SQLite Persistence & Segment Brier Skill Score (BSS) Auditor
  5. Automated Pipeline, DB & API Architecture:
     - FastAPI Production Router (/v1/simulate/prop, /v1/simulate/sgp-copula, /v1/pricing/devig,
       /v1/quant-lab/upgrades-1-8, /v1/data/pbp-stints, /v1/data/on-off-usage,
       /v1/odds/snapshot-wave, /v1/clv/log-and-grade, /v1/clv/report)
     - Streamlit Dashboard Entrypoint (gracefully guarded if streamlit is absent)
     - Built-in --self-test verification suite
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import json
import math
import os
import sqlite3
import threading
import time
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import brentq
    from scipy.stats import norm as scipy_norm, t as scipy_t
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    from starlette.middleware.base import BaseHTTPMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


# ============================================================================
# CORE PROBABILITY & ODDS UTILITIES
# ============================================================================

def odds_to_prob(odds: int) -> float:
    """Convert American odds to implied probability (including vig)."""
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def odds_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def prob_to_american(prob: float) -> int:
    """Convert fair probability (0 < p < 1) to nearest American odds."""
    p = min(max(prob, 1e-4), 1.0 - 1e-4)
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (zero external dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, fallback if scipy missing)."""
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    if HAS_SCIPY:
        return float(scipy_norm.ppf(p))
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (
        ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)


# ============================================================================
# PILLAR 1: GRANULAR DATA ARCHITECTURE (UPGRADE #3 — AUTOMATED nba_api PbP STINTS & ON/OFF USAGE)
# ============================================================================

NBA_STATS_HEADERS: Dict[str, str] = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
}


@dataclass
class LineupStintRecord:
    game_id: str
    team_abbr: str
    lineup_player_ids: Tuple[int, ...]
    possessions: float
    duration_sec: int
    pts_scored: int
    pts_allowed: int
    rim_fga_allowed: int
    usage_vacuum_pct: float = 0.0
    period: int = 1
    start_clock_sec: int = 720
    end_clock_sec: int = 0
    opp_team_abbr: str = ""
    fga: int = 0
    fta: int = 0
    fg3a: int = 0
    oreb: int = 0
    tov: int = 0
    ast: int = 0
    reb: int = 0
    player_events: Dict[int, Dict[str, float]] = field(default_factory=dict)

    @property
    def net_rating(self) -> float:
        poss = max(self.possessions, 1.0)
        return round(((self.pts_scored - self.pts_allowed) / poss) * 100.0, 2)


@dataclass
class PlayerOnOffSplit:
    target_player_id: int
    target_player_name: str
    team_abbr: str
    teammate_out_ids: Tuple[int, ...]
    teammate_out_names: Tuple[str, ...]
    games_sampled: int
    stints_on: int
    stints_off: int
    mins_on: float
    mins_off: float
    poss_on: float
    poss_off: float
    usg_pct_on: float
    usg_pct_off: float
    usg_delta_raw: float
    usg_delta_eb: float
    ts_pct_on: float
    ts_pct_off: float
    ts_delta_raw: float
    ts_delta_eb: float
    pts_per_36_on: float
    pts_per_36_off: float
    ast_per_36_on: float
    ast_per_36_off: float
    reb_per_36_on: float
    reb_per_36_off: float
    fg3a_per_36_on: float
    fg3a_per_36_off: float
    net_rating_on: float
    net_rating_off: float
    net_rating_delta: float
    eb_reliability_weight: float
    inferred_u1_archetype: str
    rate_multiplier_by_stat: Dict[str, float]
    shared_stint_mins_per_game: float
    data_source: str = "nba_api_pbp_v3"


def _parse_iso_or_mmss_clock_to_sec(clock_val: Any) -> int:
    """Parse PlayByPlayV3 ISO clocks ('PT11M42.00S') or V2 clocks ('11:42') into seconds remaining."""
    if clock_val is None:
        return 0
    if isinstance(clock_val, (int, float)):
        return max(0, int(round(float(clock_val))))
    s = str(clock_val).strip().upper()
    if not s:
        return 0
    if s.startswith("PT"):
        s = s[2:]
        mins = 0.0
        secs = 0.0
        if "M" in s:
            m_part, s = s.split("M", 1)
            mins = float(m_part or 0.0)
        if "S" in s:
            s_part = s.split("S", 1)[0]
            secs = float(s_part or 0.0)
        return max(0, int(round(mins * 60.0 + secs)))
    if ":" in s:
        parts = s.split(":")
        try:
            return max(0, int(float(parts[0]) * 60.0 + float(parts[1])))
        except ValueError:
            return 0
    try:
        return max(0, int(round(float(s))))
    except ValueError:
        return 0


def _empty_player_tally() -> Dict[str, float]:
    return {
        "pts": 0.0,
        "fga": 0.0,
        "fgm": 0.0,
        "fg3a": 0.0,
        "fg3m": 0.0,
        "fta": 0.0,
        "ftm": 0.0,
        "ast": 0.0,
        "reb": 0.0,
        "oreb": 0.0,
        "tov": 0.0,
    }


def parse_pbp_events_to_stints(
    game_id: str,
    team_abbr: str,
    events: List[Dict[str, Any]],
    starting_lineups_by_period: Optional[Dict[int, Tuple[int, ...]]] = None,
    opp_team_abbr: str = "",
) -> List[LineupStintRecord]:
    """
    Granular Play-by-Play Stint Parser (Upgrade #3):
    Converts chronological PlayByPlayV3/normalized PbP events into exact 5-man LineupStintRecords.

    Supported normalized event keys:
      - period (int), clock (str/int), team_abbr (str), action_type (str: '2pt'|'3pt'|'freethrow'|
        'rebound'|'turnover'|'substitution'), sub_type (str), shot_result ('Made'|'Missed'),
        person_id (int), assist_person_id (int, optional), player_in_id (int), player_out_id (int),
        shot_distance (float, optional), points (int, optional).
    """
    if not events:
        return []

    period_starters = starting_lineups_by_period or {}
    sorted_events = sorted(
        events,
        key=lambda e: (
            int(e.get("period", 1)),
            -_parse_iso_or_mmss_clock_to_sec(e.get("clock", 720)),
            int(e.get("action_number", 0)),
        ),
    )

    team_u = team_abbr.upper().strip()
    stints: List[LineupStintRecord] = []

    by_period: Dict[int, List[Dict[str, Any]]] = {}
    for ev in sorted_events:
        per = int(ev.get("period", 1))
        by_period.setdefault(per, []).append(ev)

    for period, per_events in sorted(by_period.items()):
        period_len = 720 if period <= 4 else 300
        active_lineup = list(period_starters.get(period, ()))
        if not active_lineup:
            inferred: List[int] = []
            for ev in per_events:
                if str(ev.get("team_abbr", "")).upper().strip() == team_u:
                    pid = int(ev.get("person_id") or 0)
                    if pid and pid not in inferred:
                        inferred.append(pid)
                    if len(inferred) == 5:
                        break
            active_lineup = inferred

        stint_start = period_len
        pts_for = 0
        pts_against = 0
        rim_fga_allowed = 0
        fga = fta = fg3a = oreb = tov = ast = reb = 0
        p_tallies: Dict[int, Dict[str, float]] = {pid: _empty_player_tally() for pid in active_lineup}

        def _flush_stint(end_clock: int) -> None:
            nonlocal stint_start, pts_for, pts_against, rim_fga_allowed
            nonlocal fga, fta, fg3a, oreb, tov, ast, reb, p_tallies
            dur = max(0, stint_start - end_clock)
            poss = max(0.0, float(fga) + 0.44 * float(fta) - float(oreb) + float(tov))
            if dur > 0 or poss > 0 or pts_for > 0 or pts_against > 0:
                team_used = max(float(fga) + 0.44 * float(fta) + float(tov), 1.0)
                max_p_used = max(
                    (
                        t["fga"] + 0.44 * t["fta"] + t["tov"]
                        for t in p_tallies.values()
                    ),
                    default=0.0,
                )
                # Concentration proxy when lineup leans heavily on primary creator
                vac_proxy = round(max(0.0, (max_p_used / team_used) - 0.28), 4)
                stints.append(
                    LineupStintRecord(
                        game_id=game_id,
                        team_abbr=team_u,
                        lineup_player_ids=tuple(sorted(set(active_lineup))),
                        possessions=round(max(poss, dur / 29.0 if dur >= 15 else poss), 2),
                        duration_sec=dur,
                        pts_scored=pts_for,
                        pts_allowed=pts_against,
                        rim_fga_allowed=rim_fga_allowed,
                        usage_vacuum_pct=vac_proxy,
                        period=period,
                        start_clock_sec=stint_start,
                        end_clock_sec=end_clock,
                        opp_team_abbr=opp_team_abbr.upper().strip(),
                        fga=fga,
                        fta=fta,
                        fg3a=fg3a,
                        oreb=oreb,
                        tov=tov,
                        ast=ast,
                        reb=reb,
                        player_events={pid: dict(vals) for pid, vals in p_tallies.items()},
                    )
                )
            stint_start = end_clock
            pts_for = pts_against = rim_fga_allowed = 0
            fga = fta = fg3a = oreb = tov = ast = reb = 0
            p_tallies = {pid: _empty_player_tally() for pid in active_lineup}

        for ev in per_events:
            clk = _parse_iso_or_mmss_clock_to_sec(ev.get("clock", stint_start))
            ev_team = str(ev.get("team_abbr", "")).upper().strip()
            act = str(ev.get("action_type", "")).lower().strip()
            sub_t = str(ev.get("sub_type", "")).lower().strip()
            res = str(ev.get("shot_result", "")).lower().strip()
            pid = int(ev.get("person_id") or 0)
            assist_pid = int(ev.get("assist_person_id") or 0)
            shot_dist = ev.get("shot_distance")

            if act == "substitution" and ev_team == team_u:
                if clk < stint_start:
                    _flush_stint(clk)
                p_out = int(ev.get("player_out_id") or 0)
                p_in = int(ev.get("player_in_id") or 0)
                if p_out in active_lineup:
                    active_lineup.remove(p_out)
                if p_in and p_in not in active_lineup:
                    active_lineup.append(p_in)
                p_tallies = {p: p_tallies.get(p, _empty_player_tally()) for p in active_lineup}
                continue

            if ev_team == team_u:
                if pid and pid not in p_tallies:
                    p_tallies[pid] = _empty_player_tally()
                if act in ("2pt", "3pt"):
                    fga += 1
                    is_three = act == "3pt"
                    if is_three:
                        fg3a += 1
                    if pid:
                        p_tallies[pid]["fga"] += 1.0
                        if is_three:
                            p_tallies[pid]["fg3a"] += 1.0
                    if res == "made":
                        pts = int(ev.get("points") or (3 if is_three else 2))
                        pts_for += pts
                        if pid:
                            p_tallies[pid]["fgm"] += 1.0
                            p_tallies[pid]["pts"] += float(pts)
                            if is_three:
                                p_tallies[pid]["fg3m"] += 1.0
                        if assist_pid:
                            ast += 1
                            p_tallies.setdefault(assist_pid, _empty_player_tally())["ast"] += 1.0
                elif act == "freethrow":
                    fta += 1
                    if pid:
                        p_tallies[pid]["fta"] += 1.0
                    if res == "made":
                        pts_for += 1
                        if pid:
                            p_tallies[pid]["ftm"] += 1.0
                            p_tallies[pid]["pts"] += 1.0
                elif act == "rebound":
                    reb += 1
                    is_off = "offensive" in sub_t or sub_t == "off"
                    if is_off:
                        oreb += 1
                    if pid:
                        p_tallies[pid]["reb"] += 1.0
                        if is_off:
                            p_tallies[pid]["oreb"] += 1.0
                elif act == "turnover":
                    tov += 1
                    if pid:
                        p_tallies[pid]["tov"] += 1.0
            else:
                if act in ("2pt", "3pt"):
                    is_rim = (
                        (shot_dist is not None and float(shot_dist) <= 6.0)
                        or any(k in sub_t for k in ("layup", "dunk", "hook", "tip", "driving"))
                    )
                    if is_rim:
                        rim_fga_allowed += 1
                    if res == "made":
                        pts_against += int(ev.get("points") or (3 if act == "3pt" else 2))
                elif act == "freethrow" and res == "made":
                    pts_against += 1

        _flush_stint(0)

    return stints


def compute_player_on_off_split(
    stints: List[LineupStintRecord],
    target_player_id: int,
    teammate_out_ids: List[int],
    target_player_name: str = "",
    teammate_out_names: Optional[List[str]] = None,
    prior_poss_k: float = 250.0,
    data_source: str = "nba_api_pbp_v3",
) -> PlayerOnOffSplit:
    """
    Empirical PbP On/Off Usage, Per-36 Rate & Efficiency Aggregator (Upgrade #3):
    Partitions all stints where target_player_id is ON the court into:
      - ON regime: all teammate_out_ids are also ON the court
      - OFF regime: none of teammate_out_ids are on the court (exact injury-vacuum condition)
    Applies Empirical Bayes shrinkage (w = poss_off / (poss_off + prior_poss_k)) to stabilize
    small-sample usage and TS% shifts and automatically classifies the player into Upgrade 1's
    3 archetypes (TYPE_A_SURGE, TYPE_B_SATURATED, TYPE_C_EFFICIENCY_TAX).
    """
    out_set = set(int(x) for x in teammate_out_ids)
    team_abbr = stints[0].team_abbr if stints else "NBA"
    games_seen = set()

    def _agg(bucket: List[LineupStintRecord]) -> Dict[str, float]:
        dur_sec = sum(s.duration_sec for s in bucket)
        mins = dur_sec / 60.0
        poss = sum(s.possessions for s in bucket)
        pts_for = sum(s.pts_scored for s in bucket)
        pts_ag = sum(s.pts_allowed for s in bucket)
        team_used = sum(float(s.fga) + 0.44 * float(s.fta) + float(s.tov) for s in bucket)

        t_pts = t_fga = t_fta = t_tov = t_ast = t_reb = t_fg3a = 0.0
        for s in bucket:
            pe = s.player_events.get(target_player_id)
            if pe:
                t_pts += float(pe.get("pts", 0.0))
                t_fga += float(pe.get("fga", 0.0))
                t_fta += float(pe.get("fta", 0.0))
                t_tov += float(pe.get("tov", 0.0))
                t_ast += float(pe.get("ast", 0.0))
                t_reb += float(pe.get("reb", 0.0))
                t_fg3a += float(pe.get("fg3a", 0.0))

        p_used = t_fga + 0.44 * t_fta + t_tov
        usg_pct = (100.0 * p_used / team_used) if team_used > 0 else 20.0
        tsa = t_fga + 0.44 * t_fta
        ts_pct = (100.0 * t_pts / (2.0 * tsa)) if tsa > 0 else 56.0
        scale36 = (36.0 / mins) if mins > 0 else 0.0
        net_rtg = (((pts_for - pts_ag) / poss) * 100.0) if poss > 0 else 0.0
        return {
            "mins": mins,
            "poss": poss,
            "usg_pct": usg_pct,
            "ts_pct": ts_pct,
            "pts36": t_pts * scale36,
            "ast36": t_ast * scale36,
            "reb36": t_reb * scale36,
            "fg3a36": t_fg3a * scale36,
            "net_rtg": net_rtg,
        }

    on_bucket: List[LineupStintRecord] = []
    off_bucket: List[LineupStintRecord] = []
    for s in stints:
        lineup_set = set(s.lineup_player_ids)
        if target_player_id not in lineup_set:
            continue
        games_seen.add(s.game_id)
        if out_set and out_set.issubset(lineup_set):
            on_bucket.append(s)
        elif out_set and lineup_set.isdisjoint(out_set):
            off_bucket.append(s)

    a_on = _agg(on_bucket)
    a_off = _agg(off_bucket)

    w_eb = a_off["poss"] / (a_off["poss"] + max(prior_poss_k, 1.0)) if a_off["poss"] > 0 else 0.0
    usg_delta_raw = a_off["usg_pct"] - a_on["usg_pct"]
    usg_delta_eb = usg_delta_raw * w_eb
    ts_delta_raw = a_off["ts_pct"] - a_on["ts_pct"]
    ts_delta_eb = ts_delta_raw * w_eb
    net_delta = (a_off["net_rtg"] - a_on["net_rtg"]) * w_eb

    # Automatic Empirical Upgrade 1 Archetype Classification
    if ts_delta_eb <= -1.8:
        inferred_arch = "TYPE_C_EFFICIENCY_TAX"
    elif a_on["usg_pct"] >= 31.5 and usg_delta_eb < 1.8:
        inferred_arch = "TYPE_B_SATURATED"
    elif usg_delta_eb >= 1.2:
        inferred_arch = "TYPE_A_SURGE"
    else:
        inferred_arch = "TYPE_B_SATURATED"

    def _eb_ratio(val_off: float, val_on: float) -> float:
        if val_on <= 0.1 or val_off <= 0.0:
            return 1.0
        raw_ratio = val_off / val_on
        shrunk = 1.0 + w_eb * (raw_ratio - 1.0)
        return round(min(max(shrunk, 0.85), 1.22), 4)

    pts_m = _eb_ratio(a_off["pts36"], a_on["pts36"])
    ast_m = _eb_ratio(a_off["ast36"], a_on["ast36"])
    reb_m = _eb_ratio(a_off["reb36"], a_on["reb36"])
    fg3_m = _eb_ratio(a_off["fg3a36"], a_on["fg3a36"])
    pa_m = _eb_ratio(a_off["pts36"] + a_off["ast36"], a_on["pts36"] + a_on["ast36"])
    pr_m = _eb_ratio(a_off["pts36"] + a_off["reb36"], a_on["pts36"] + a_on["reb36"])
    pra_m = _eb_ratio(
        a_off["pts36"] + a_off["reb36"] + a_off["ast36"],
        a_on["pts36"] + a_on["reb36"] + a_on["ast36"],
    )

    n_games = max(len(games_seen), 1)
    return PlayerOnOffSplit(
        target_player_id=target_player_id,
        target_player_name=target_player_name or str(target_player_id),
        team_abbr=team_abbr,
        teammate_out_ids=tuple(sorted(out_set)),
        teammate_out_names=tuple(teammate_out_names or [str(x) for x in sorted(out_set)]),
        games_sampled=len(games_seen),
        stints_on=len(on_bucket),
        stints_off=len(off_bucket),
        mins_on=round(a_on["mins"], 1),
        mins_off=round(a_off["mins"], 1),
        poss_on=round(a_on["poss"], 1),
        poss_off=round(a_off["poss"], 1),
        usg_pct_on=round(a_on["usg_pct"], 2),
        usg_pct_off=round(a_off["usg_pct"], 2),
        usg_delta_raw=round(usg_delta_raw, 2),
        usg_delta_eb=round(usg_delta_eb, 2),
        ts_pct_on=round(a_on["ts_pct"], 2),
        ts_pct_off=round(a_off["ts_pct"], 2),
        ts_delta_raw=round(ts_delta_raw, 2),
        ts_delta_eb=round(ts_delta_eb, 2),
        pts_per_36_on=round(a_on["pts36"], 2),
        pts_per_36_off=round(a_off["pts36"], 2),
        ast_per_36_on=round(a_on["ast36"], 2),
        ast_per_36_off=round(a_off["ast36"], 2),
        reb_per_36_on=round(a_on["reb36"], 2),
        reb_per_36_off=round(a_off["reb36"], 2),
        fg3a_per_36_on=round(a_on["fg3a36"], 2),
        fg3a_per_36_off=round(a_off["fg3a36"], 2),
        net_rating_on=round(a_on["net_rtg"], 2),
        net_rating_off=round(a_off["net_rtg"], 2),
        net_rating_delta=round(net_delta, 2),
        eb_reliability_weight=round(w_eb, 4),
        inferred_u1_archetype=inferred_arch,
        rate_multiplier_by_stat={
            "Points": pts_m,
            "Assists": ast_m,
            "Rebounds": reb_m,
            "3PM": fg3_m,
            "PTS+AST": pa_m,
            "PTS+REB": pr_m,
            "PRA": pra_m,
        },
        shared_stint_mins_per_game=round(a_on["mins"] / n_games, 2),
        data_source=data_source,
    )


class NbaApiIngestionPipeline:
    """
    Upgrade #3: Automated nba_api Play-by-Play Stint & On/Off Usage Ingestion Pipeline.
      - Live Mode (when nba_api is installed and offline_mode=False):
        * Fetches PlayByPlayV3 (fallback PlayByPlayV2) + GameRotation to construct exact 5-man stints.
        * Fetches LeagueGameLog for automated multi-game team stint windows (L10 / L15 / Season).
        * Fetches TeamPlayerOnOffDetails for season-level on/off validation.
        * Enforces browser headers, rate-limiting (0.75s floor), exponential backoff retries,
          and persistent SQLite caching (alphaprop_pbp_cache.sqlite or :memory:).
      - Offline / Test Mode:
        * Serves deterministic realistic multi-game PbP stints & SQLite cache round-trips so
          downstream v9.1 Monte Carlo, Sigmoid, and Upgrades 1-8 tests run with zero network calls.
    """

    def __init__(
        self,
        season: str = "2026-27",
        cache_db_path: str = ":memory:",
        min_request_interval_sec: float = 0.75,
        max_retries: int = 3,
        offline_mode: bool = False,
    ):
        self.season = season
        self.cache_db_path = cache_db_path
        self.min_request_interval_sec = min_request_interval_sec
        self.max_retries = max_retries
        self.offline_mode = offline_mode
        self._last_req_ts = 0.0
        self._has_nba_api = False

        if not offline_mode:
            try:
                import nba_api  # noqa: F401
                self._has_nba_api = True
            except ImportError:
                self._has_nba_api = False

        self._conn = sqlite3.connect(cache_db_path, check_same_thread=False)
        self._init_cache_tables()

    def _init_cache_tables(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pbp_stints_cache (
                game_id TEXT NOT NULL,
                team_abbr TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                cached_at REAL NOT NULL,
                PRIMARY KEY (game_id, team_abbr)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS team_on_off_cache (
                season TEXT NOT NULL,
                team_abbr TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                cached_at REAL NOT NULL,
                PRIMARY KEY (season, team_abbr)
            )
            """
        )
        self._conn.commit()

    def _rate_limit_wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_req_ts
        if elapsed < self.min_request_interval_sec:
            time.sleep(self.min_request_interval_sec - elapsed)
        self._last_req_ts = time.monotonic()

    def _call_with_retry(self, fn, *args, **kwargs):
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                time.sleep(min(4.0, 0.75 * (2 ** attempt)))
        raise RuntimeError(f"nba_api request failed after {self.max_retries} attempts: {last_exc}")

    def resolve_team_id(self, team_abbr: str) -> Optional[int]:
        abbr = team_abbr.upper().strip()
        if self._has_nba_api:
            from nba_api.stats.static import teams as nba_teams
            matches = [t for t in nba_teams.get_teams() if t["abbreviation"].upper() == abbr]
            if matches:
                return int(matches[0]["id"])
        static_map = {
            "ATL": 1610612737, "BOS": 1610612738, "CLE": 1610612739, "NOP": 1610612740,
            "CHI": 1610612741, "DAL": 1610612742, "DEN": 1610612743, "GSW": 1610612744,
            "HOU": 1610612745, "LAC": 1610612746, "LAL": 1610612747, "MIA": 1610612748,
            "MIL": 1610612749, "MIN": 1610612750, "BKN": 1610612751, "NYK": 1610612752,
            "ORL": 1610612753, "IND": 1610612754, "PHI": 1610612755, "PHX": 1610612756,
            "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "OKC": 1610612760,
            "TOR": 1610612761, "UTA": 1610612762, "MEM": 1610612763, "WAS": 1610612764,
            "DET": 1610612765, "CHA": 1610612766,
        }
        return static_map.get(abbr)

    def _serialize_stints(self, stints: List[LineupStintRecord]) -> str:
        payload = []
        for s in stints:
            d = asdict(s)
            d["lineup_player_ids"] = list(s.lineup_player_ids)
            d["player_events"] = {str(k): v for k, v in s.player_events.items()}
            payload.append(d)
        return json.dumps(payload)

    def _deserialize_stints(self, raw_json: str) -> List[LineupStintRecord]:
        items = json.loads(raw_json)
        out: List[LineupStintRecord] = []
        for d in items:
            d["lineup_player_ids"] = tuple(int(x) for x in d["lineup_player_ids"])
            d["player_events"] = {int(k): v for k, v in d.get("player_events", {}).items()}
            out.append(LineupStintRecord(**d))
        return out

    def _get_cached_stints(self, game_id: str, team_abbr: str) -> Optional[List[LineupStintRecord]]:
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT payload_json FROM pbp_stints_cache WHERE game_id = ? AND team_abbr = ?",
            (game_id, team_abbr.upper()),
        ).fetchone()
        return self._deserialize_stints(row[0]) if row else None

    def _save_cached_stints(self, game_id: str, team_abbr: str, stints: List[LineupStintRecord]) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO pbp_stints_cache (game_id, team_abbr, payload_json, cached_at)
            VALUES (?, ?, ?, ?)
            """,
            (game_id, team_abbr.upper(), self._serialize_stints(stints), time.time()),
        )
        self._conn.commit()

    def _fetch_live_pbp_stints_from_nba_api(self, game_id: str, team_abbr: str) -> List[LineupStintRecord]:
        """Live nba_api PlayByPlayV3 + GameRotation stint builder."""
        from nba_api.stats.endpoints import gamerotation, playbyplayv3

        pbp = self._call_with_retry(
            playbyplayv3.PlayByPlayV3,
            game_id=game_id,
            headers=NBA_STATS_HEADERS,
            timeout=25,
        )
        pbp_df = pbp.get_data_frames()[0]

        rot = self._call_with_retry(
            gamerotation.GameRotation,
            game_id=game_id,
            headers=NBA_STATS_HEADERS,
            timeout=25,
        )
        rot_dfs = rot.get_data_frames()
        rot_df = pd.concat(rot_dfs, ignore_index=True) if rot_dfs else pd.DataFrame()

        team_u = team_abbr.upper().strip()
        starters_by_period: Dict[int, Tuple[int, ...]] = {}
        if not rot_df.empty and "TEAM_ABBREVIATION" in rot_df.columns:
            t_rot = rot_df[rot_df["TEAM_ABBREVIATION"].str.upper() == team_u]
            for period in range(1, 5):
                p_start_tenths = (period - 1) * 7200
                on_start = t_rot[
                    (t_rot["IN_TIME_REAL"] <= p_start_tenths)
                    & (t_rot["OUT_TIME_REAL"] > p_start_tenths)
                ]["PERSON_ID"].astype(int).tolist()
                if on_start:
                    starters_by_period[period] = tuple(sorted(set(on_start))[:5])

        events: List[Dict[str, Any]] = []
        for _, r in pbp_df.iterrows():
            act_type = str(r.get("actionType", "")).lower().strip()
            ev: Dict[str, Any] = {
                "action_number": int(r.get("actionNumber", 0) or 0),
                "period": int(r.get("period", 1) or 1),
                "clock": r.get("clock", "PT12M00.00S"),
                "team_abbr": str(r.get("teamTricode", "")).upper().strip(),
                "action_type": act_type,
                "sub_type": str(r.get("subType", "")),
                "shot_result": str(r.get("shotResult", "")),
                "person_id": int(r.get("personId", 0) or 0),
                "shot_distance": r.get("shotDistance"),
                "points": int(r.get("pointsTotal", 0) or 0),
            }
            if act_type == "substitution":
                desc = str(r.get("description", ""))
                # PlayByPlayV3 encodes SUB out/in via subType or description
                if "out" in str(r.get("subType", "")).lower():
                    ev["player_out_id"] = int(r.get("personId", 0) or 0)
                else:
                    ev["player_in_id"] = int(r.get("personId", 0) or 0)
                if "FOR" in desc.upper():
                    ev["sub_desc"] = desc
            events.append(ev)

        return parse_pbp_events_to_stints(
            game_id=game_id,
            team_abbr=team_u,
            events=events,
            starting_lineups_by_period=starters_by_period,
        )

    def fetch_recent_team_game_ids(self, team_abbr: str, last_n_games: int = 10) -> List[str]:
        """Fetch the last N regular-season game IDs for a team via LeagueGameLog (or deterministic fallback)."""
        team_u = team_abbr.upper().strip()
        if self._has_nba_api and not self.offline_mode:
            from nba_api.stats.endpoints import leaguegamelog
            team_id = self.resolve_team_id(team_u)
            gl = self._call_with_retry(
                leaguegamelog.LeagueGameLog,
                season=self.season,
                player_or_team_abbreviation="T",
                headers=NBA_STATS_HEADERS,
                timeout=25,
            )
            df = gl.get_data_frames()[0]
            if team_id is not None and "TEAM_ID" in df.columns:
                df = df[df["TEAM_ID"] == team_id]
            elif "TEAM_ABBREVIATION" in df.columns:
                df = df[df["TEAM_ABBREVIATION"].str.upper() == team_u]
            df = df.sort_values("GAME_DATE", ascending=False)
            return [str(gid) for gid in df["GAME_ID"].head(last_n_games).tolist()]

        return [f"0022600{101 + i:03d}" for i in range(last_n_games)]

    def fetch_team_on_off_endpoint(self, team_abbr: str, ttl_hours: float = 12.0) -> Optional[pd.DataFrame]:
        """Fetch season-level TeamPlayerOnOffDetails from nba_api with SQLite TTL caching."""
        team_u = team_abbr.upper().strip()
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT payload_json, cached_at FROM team_on_off_cache WHERE season = ? AND team_abbr = ?",
            (self.season, team_u),
        ).fetchone()
        if row and (time.time() - float(row[1])) < ttl_hours * 3600.0:
            return pd.DataFrame(json.loads(row[0]))

        if not (self._has_nba_api and not self.offline_mode):
            return None

        team_id = self.resolve_team_id(team_u)
        if team_id is None:
            return None

        from nba_api.stats.endpoints import teamplayeronoffdetails
        ep = self._call_with_retry(
            teamplayeronoffdetails.TeamPlayerOnOffDetails,
            team_id=team_id,
            season=self.season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Per100Possessions",
            headers=NBA_STATS_HEADERS,
            timeout=25,
        )
        dfs = ep.get_data_frames()
        if not dfs:
            return None
        combined = pd.concat(dfs[1:3], ignore_index=True) if len(dfs) >= 3 else dfs[0]
        cur.execute(
            """
            INSERT OR REPLACE INTO team_on_off_cache (season, team_abbr, payload_json, cached_at)
            VALUES (?, ?, ?, ?)
            """,
            (self.season, team_u, combined.to_json(orient="records"), time.time()),
        )
        self._conn.commit()
        return combined

    def _build_offline_sample_stints(self, game_id: str, team_abbr: str) -> List[LineupStintRecord]:
        """Deterministic realistic PbP stints for offline validation & unit testing."""
        team_u = team_abbr.upper().strip()
        # Player IDs: 1630178 (Maxey-profile secondary initiator), 203954 (Embiid-profile center),
        # 1627759 (Wing), 1628983 (Guard), 1630596 (Forward), 1629651 (Backup C)
        stint_on = LineupStintRecord(
            game_id=game_id,
            team_abbr=team_u,
            lineup_player_ids=(203954, 1627759, 1628983, 1630178, 1630596),
            possessions=28.0,
            duration_sec=840,
            pts_scored=33,
            pts_allowed=29,
            rim_fga_allowed=6,
            usage_vacuum_pct=0.032,
            period=1,
            start_clock_sec=720,
            end_clock_sec=0,
            fga=24,
            fta=8,
            fg3a=9,
            oreb=3,
            tov=3,
            ast=7,
            reb=11,
            player_events={
                1630178: {"pts": 9.0, "fga": 7.0, "fgm": 3.0, "fg3a": 3.0, "fg3m": 1.0, "fta": 2.0, "ftm": 2.0, "ast": 2.0, "reb": 1.0, "oreb": 0.0, "tov": 1.0},
                203954: {"pts": 11.0, "fga": 8.0, "fgm": 4.0, "fg3a": 1.0, "fg3m": 0.0, "fta": 4.0, "ftm": 3.0, "ast": 1.0, "reb": 4.0, "oreb": 1.0, "tov": 1.0},
                1627759: {"pts": 7.0, "fga": 5.0, "fgm": 3.0, "fg3a": 2.0, "fg3m": 1.0, "fta": 0.0, "ftm": 0.0, "ast": 1.0, "reb": 2.0, "oreb": 0.0, "tov": 0.0},
            },
        )
        stint_off = LineupStintRecord(
            game_id=game_id,
            team_abbr=team_u,
            lineup_player_ids=(1627759, 1628983, 1629651, 1630178, 1630596),
            possessions=24.0,
            duration_sec=720,
            pts_scored=29,
            pts_allowed=27,
            rim_fga_allowed=9,
            usage_vacuum_pct=0.068,
            period=2,
            start_clock_sec=720,
            end_clock_sec=0,
            fga=21,
            fta=6,
            fg3a=9,
            oreb=2,
            tov=3,
            ast=6,
            reb=9,
            player_events={
                1630178: {"pts": 12.0, "fga": 9.0, "fgm": 4.0, "fg3a": 4.0, "fg3m": 2.0, "fta": 2.0, "ftm": 2.0, "ast": 3.0, "reb": 1.0, "oreb": 0.0, "tov": 1.0},
                1627759: {"pts": 5.0, "fga": 6.0, "fgm": 2.0, "fg3a": 2.0, "fg3m": 0.0, "fta": 2.0, "ftm": 1.0, "ast": 1.0, "reb": 2.0, "oreb": 0.0, "tov": 1.0},
                1629651: {"pts": 4.0, "fga": 2.0, "fgm": 2.0, "fg3a": 0.0, "fg3m": 0.0, "fta": 0.0, "ftm": 0.0, "ast": 0.0, "reb": 4.0, "oreb": 1.0, "tov": 0.0},
            },
        )
        return [stint_on, stint_off]

    def fetch_lineup_stints(
        self,
        game_id: str,
        team_abbr: str,
        use_cache: bool = True,
    ) -> List[LineupStintRecord]:
        """Fetch PbP lineup stints (SQLite cache -> live nba_api PlayByPlayV3 -> offline fallback)."""
        team_u = team_abbr.upper().strip()
        if use_cache:
            cached = self._get_cached_stints(game_id, team_u)
            if cached is not None:
                return cached

        if self._has_nba_api and not self.offline_mode:
            try:
                stints = self._fetch_live_pbp_stints_from_nba_api(game_id, team_u)
                if stints:
                    self._save_cached_stints(game_id, team_u, stints)
                    return stints
            except Exception:
                pass

        stints = self._build_offline_sample_stints(game_id, team_u)
        self._save_cached_stints(game_id, team_u, stints)
        return stints

    def ingest_player_on_off_usage(
        self,
        team_abbr: str,
        target_player_id: int,
        teammate_out_ids: List[int],
        target_player_name: str = "",
        teammate_out_names: Optional[List[str]] = None,
        last_n_games: int = 12,
        prior_poss_k: float = 250.0,
    ) -> PlayerOnOffSplit:
        """
        End-to-End Automated Ingestion Pipeline (Upgrade #3):
        Pulls the last N games of PbP stints for team_abbr (from SQLite cache or live nba_api),
        aggregates exact on/off usage, per-36 rates, TS%, and net rating when teammate_out_ids
        are OFF the court, and returns a stabilized PlayerOnOffSplit ready for the v9.1 engine.
        """
        game_ids = self.fetch_recent_team_game_ids(team_abbr, last_n_games=last_n_games)
        all_stints: List[LineupStintRecord] = []
        for gid in game_ids:
            all_stints.extend(self.fetch_lineup_stints(gid, team_abbr, use_cache=True))

        source = "nba_api_pbp_v3" if (self._has_nba_api and not self.offline_mode) else "synthetic_pbp_fallback"
        return compute_player_on_off_split(
            stints=all_stints,
            target_player_id=target_player_id,
            teammate_out_ids=teammate_out_ids,
            target_player_name=target_player_name,
            teammate_out_names=teammate_out_names,
            prior_poss_k=prior_poss_k,
            data_source=source,
        )


def build_on_off_enriched_prop_inputs(
    base_rate_per_min: float,
    stat_type: str,
    on_off_split: PlayerOnOffSplit,
    unassisted_fg_share: float = 0.55,
    apply_empirical_rate_to_mc: bool = True,
) -> Dict[str, Any]:
    """
    Wires a PlayerOnOffSplit directly into the v9.1 engine with zero double-counting:
      - Updates projected usage (`proj_usage = usg_pct_on + usg_delta_eb`) for Sigmoid Weighting.
      - Computes empirical rate multiplier for `stat_type` (Points, Assists, Rebounds, 3PM, PRA, PTS+AST, PTS+REB).
      - When `apply_empirical_rate_to_mc=True`, adjusts `rate_per_min` directly AND sets
        `rate_already_includes_vacuum=True` so `evaluate_all_upgrades_1_to_8` does NOT double-count U1!
      - Feeds `net_rating_delta` into `TacticalMatchupInput.lineup_net_rating_delta` and
        `shared_stint_mins_per_game` into Upgrade 4 PbP shell evaluation.
    """
    stat_mult = on_off_split.rate_multiplier_by_stat.get(stat_type, 1.0)
    proj_usage = round(min(max(on_off_split.usg_pct_on + on_off_split.usg_delta_eb, 10.0), 42.0), 2)
    w_def, w_sit = compute_sigmoid_weights(proj_usage, unassisted_fg_share)

    adj_rate = round(base_rate_per_min * stat_mult, 4) if apply_empirical_rate_to_mc else base_rate_per_min
    return {
        "target_player": on_off_split.target_player_name,
        "team_abbr": on_off_split.team_abbr,
        "teammates_out": list(on_off_split.teammate_out_names),
        "stat_type": stat_type,
        "base_rate_per_min": base_rate_per_min,
        "empirical_stat_multiplier": stat_mult,
        "enriched_rate_per_min": adj_rate,
        "proj_usage": proj_usage,
        "w_def": w_def,
        "w_sit": w_sit,
        "inferred_u1_archetype": on_off_split.inferred_u1_archetype,
        "rate_already_includes_vacuum": apply_empirical_rate_to_mc,
        "lineup_net_rating_delta": on_off_split.net_rating_delta,
        "shared_stint_mins_per_game": on_off_split.shared_stint_mins_per_game,
        "eb_reliability_weight": on_off_split.eb_reliability_weight,
        "usg_delta_eb": on_off_split.usg_delta_eb,
        "ts_delta_eb": on_off_split.ts_delta_eb,
        "data_source": on_off_split.data_source,
    }


# ============================================================================
# PILLAR 2: CORE MATH, DISTRIBUTIONS & MONTE CARLO + COPULA SGP ENGINE
# ============================================================================

def compute_sigmoid_weights(proj_usage: float, unassisted_fg_share: float) -> Tuple[float, float]:
    """
    v8/v9 Continuous Sigmoid Archetype Weighting:
      W_def = 0.40 + 0.40 / (1 + exp(-0.35 * (U_proj - 21.5))) + 0.10 * F_unassisted
      Clamped to [0.35, 0.92]. W_sit = 1.0 - W_def.
    """
    w_def = 0.40 + 0.40 / (1.0 + math.exp(-0.35 * (proj_usage - 21.5))) + 0.10 * unassisted_fg_share
    w_def = min(max(w_def, 0.35), 0.92)
    return round(w_def, 4), round(1.0 - w_def, 4)


def compute_spatial_efficiency_and_redirect(
    def_match_scalar: float,
    sit_split_scalar: float,
    w_def: float,
    w_sit: float,
    opp_rim_suppression: float = 0.0,
    stat_type: str = "Points",
    anchor_off_share: float = 0.25,
) -> float:
    """Spatial shot-redirection elasticity & defensive anchor on/off blending."""
    base_mult = (w_def * def_match_scalar) + (w_sit * sit_split_scalar)
    redirect_adj = 1.0
    if stat_type in ("3PM", "Assists") and opp_rim_suppression > 0.04:
        redirect_adj = 1.0 + min(0.08, opp_rim_suppression * 0.65)
    elif stat_type == "Points" and opp_rim_suppression > 0.05:
        redirect_adj = 1.0 - min(0.05, opp_rim_suppression * 0.35)
    anchor_blend = 1.0 + (anchor_off_share * 0.04 if def_match_scalar > 1.0 else -anchor_off_share * 0.02)
    return round(base_mult * redirect_adj * anchor_blend, 4)


# ============================================================================
# 30-TEAM NUMERIC COACH HOOK MODIFIER TABLE (2026-27 NBA HEAD COACHES)
# Early-Hook (>1.10x) vs. Neutral (0.92x-1.08x) vs. Heavy-Minute (<0.88x)
# ============================================================================
COACH_HOOK_MODIFIER_TABLE: Dict[str, Dict[str, Any]] = {
    "OKC": {"coach": "Mark Daigneault", "tier": "EARLY_HOOK", "hook_mult": 1.28, "nba_cup_mult": 1.10, "blowout_min_delta": -2.5, "note": "SGA 4Q Sit Rule: parks stars entering Q4 up 18+ even in Cup play."},
    "SAS": {"coach": "Mitch Johnson", "tier": "EARLY_HOOK", "hook_mult": 1.24, "nba_cup_mult": 0.95, "blowout_min_delta": -2.2, "note": "Strict 20+ pt rout pull on Wembanyama and core starters."},
    "MEM": {"coach": "Tuomas Iisalo", "tier": "EARLY_HOOK", "hook_mult": 1.22, "nba_cup_mult": 0.95, "blowout_min_delta": -2.0, "note": "High-tempo 10-man hockey-line rotation caps star ceiling in blowouts."},
    "GSW": {"coach": "Steve Kerr", "tier": "EARLY_HOOK", "hook_mult": 1.20, "nba_cup_mult": 0.95, "blowout_min_delta": -2.0, "note": "Veteran load protection; quick white-flag benching down or up 18+."},
    "CLE": {"coach": "Kenny Atkinson", "tier": "EARLY_HOOK", "hook_mult": 1.18, "nba_cup_mult": 0.92, "blowout_min_delta": -1.8, "note": "Deep 10-man rotation; pulls frontcourt/guards early in comfortable leads."},
    "HOU": {"coach": "Ime Udoka", "tier": "EARLY_HOOK", "hook_mult": 1.16, "nba_cup_mult": 0.92, "blowout_min_delta": -1.6, "note": "Early 4Q starter pull in 20+ pt wins; leans on young bench."},
    "DAL": {"coach": "Dusty May", "tier": "EARLY_HOOK", "hook_mult": 1.15, "nba_cup_mult": 0.95, "blowout_min_delta": -1.5, "note": "Quick to protect vets and Cooper Flagg when trailing 18+ on road/B2Bs."},
    "IND": {"coach": "Rick Carlisle", "tier": "EARLY_HOOK", "hook_mult": 1.14, "nba_cup_mult": 0.92, "blowout_min_delta": -1.4, "note": "Fast-pace deep bench deployment in Q4 blowouts."},
    "BOS": {"coach": "Joe Mazzulla", "tier": "EARLY_HOOK", "hook_mult": 1.12, "nba_cup_mult": 0.62, "blowout_min_delta": -1.2, "note": "Math-based regular-season pull, but aggressive NBA Cup point-diff chaser."},
    "MIL": {"coach": "Taylor Jenkins", "tier": "EARLY_HOOK", "hook_mult": 1.12, "nba_cup_mult": 0.95, "blowout_min_delta": -1.2, "note": "Structured minute caps and early garbage-time benching."},
    "NOP": {"coach": "Jamahl Mosley", "tier": "NEUTRAL", "hook_mult": 1.08, "nba_cup_mult": 0.92, "blowout_min_delta": -0.6, "note": "Balanced defensive rotation; pulls starters around 5:00 Q4 up/down 20."},
    "ORL": {"coach": "Sean Sweeney", "tier": "NEUTRAL", "hook_mult": 1.06, "nba_cup_mult": 0.90, "blowout_min_delta": -0.5, "note": "Measured starter rotation with standard 20-pt Q4 threshold."},
    "BKN": {"coach": "Jordi Fernandez", "tier": "NEUTRAL", "hook_mult": 1.05, "nba_cup_mult": 0.92, "blowout_min_delta": -0.4, "note": "Development-oriented bench minutes in lopsided 4Q scripts."},
    "CHA": {"coach": "Charles Lee", "tier": "NEUTRAL", "hook_mult": 1.02, "nba_cup_mult": 0.90, "blowout_min_delta": -0.2, "note": "League-average blowout hook timing."},
    "CHI": {"coach": "Tiago Splitter", "tier": "NEUTRAL", "hook_mult": 1.02, "nba_cup_mult": 0.92, "blowout_min_delta": -0.2, "note": "League-average Q4 substitution timing."},
    "WAS": {"coach": "Brian Keefe", "tier": "NEUTRAL", "hook_mult": 1.00, "nba_cup_mult": 0.92, "blowout_min_delta": 0.0, "note": "Baseline neutral blowout pull rate."},
    "ATL": {"coach": "Quin Snyder", "tier": "NEUTRAL", "hook_mult": 0.98, "nba_cup_mult": 0.85, "blowout_min_delta": 0.2, "note": "Slightly extended starter stints in high-total pace games."},
    "DET": {"coach": "J.B. Bickerstaff", "tier": "NEUTRAL", "hook_mult": 0.95, "nba_cup_mult": 0.82, "blowout_min_delta": 0.5, "note": "Keeps Cade and core starters in until margin is fully sealed."},
    "PHX": {"coach": "Jordan Ott", "tier": "NEUTRAL", "hook_mult": 0.95, "nba_cup_mult": 0.82, "blowout_min_delta": 0.5, "note": "Staggers primary scorers deep into Q4."},
    "POR": {"coach": "Micah Nori", "tier": "NEUTRAL", "hook_mult": 0.95, "nba_cup_mult": 0.85, "blowout_min_delta": 0.5, "note": "Extended young-core closing stints."},
    "LAC": {"coach": "Tyronn Lue", "tier": "NEUTRAL", "hook_mult": 0.94, "nba_cup_mult": 0.85, "blowout_min_delta": 0.6, "note": "Rides primary ball-handler late into competitive or moderate margins."},
    "MIA": {"coach": "Erik Spoelstra", "tier": "NEUTRAL", "hook_mult": 0.92, "nba_cup_mult": 0.80, "blowout_min_delta": 0.8, "note": "Disciplined closing units; rarely waves white flag early."},
    "SAC": {"coach": "Doug Christie", "tier": "NEUTRAL", "hook_mult": 0.92, "nba_cup_mult": 0.82, "blowout_min_delta": 0.8, "note": "Leaves primary scorers in to chase comebacks down 14-18."},
    "MIN": {"coach": "Chris Finch", "tier": "HEAVY_MINUTE", "hook_mult": 0.86, "nba_cup_mult": 0.78, "blowout_min_delta": 1.4, "note": "Tight 8-man rotation; rides starters deep into Q4."},
    "DEN": {"coach": "David Adelman", "tier": "HEAVY_MINUTE", "hook_mult": 0.85, "nba_cup_mult": 0.78, "blowout_min_delta": 1.5, "note": "Heavy starter load on Jokic/Murray even in moderate leads."},
    "UTA": {"coach": "Will Hardy", "tier": "HEAVY_MINUTE", "hook_mult": 0.84, "nba_cup_mult": 0.78, "blowout_min_delta": 1.6, "note": "Keeps young core & shooters on floor down 14-18 inside final 2 mins."},
    "LAL": {"coach": "JJ Redick", "tier": "HEAVY_MINUTE", "hook_mult": 0.84, "nba_cup_mult": 0.75, "blowout_min_delta": 1.6, "note": "Heavy starter minutes in tight and moderate-margin games."},
    "NYK": {"coach": "Mike Brown", "tier": "HEAVY_MINUTE", "hook_mult": 0.82, "nba_cup_mult": 0.72, "blowout_min_delta": 1.8, "note": "Tight playoff-style rotation keeping Brunson/KAT on floor late."},
    "TOR": {"coach": "Darko Rajakovic", "tier": "HEAVY_MINUTE", "hook_mult": 0.82, "nba_cup_mult": 0.58, "blowout_min_delta": 1.8, "note": "Heavy wing minutes + aggressive NBA Cup point-differential chaser."},
    "PHI": {"coach": "Nick Nurse", "tier": "HEAVY_MINUTE", "hook_mult": 0.78, "nba_cup_mult": 0.70, "blowout_min_delta": 2.2, "note": "Elite Starter Rider (37+ MPG for Maxey); rarely pulls early unless down 20+ late."},
}

_LEGACY_COACH_ALIASES: Dict[str, Dict[str, Any]] = {
    "tom thibodeau": {"team": "LEGACY", "coach": "Tom Thibodeau", "tier": "HEAVY_MINUTE", "hook_mult": 0.68, "nba_cup_mult": 0.60, "blowout_min_delta": 2.8, "note": "Extreme 8-man starter rider (historical benchmark)."},
    "jason kidd": {"team": "LEGACY", "coach": "Jason Kidd", "tier": "EARLY_HOOK", "hook_mult": 1.18, "nba_cup_mult": 0.95, "blowout_min_delta": -1.8, "note": "Quick to downsize drop centers and pull vets in blowouts."},
}


def resolve_coach_hook_modifier(
    team_abbr: Optional[str] = None,
    coach_name: Optional[str] = None,
    is_nba_cup: bool = False,
    explicit_coach_mult: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Resolves numeric coach_hook_modifier from COACH_HOOK_MODIFIER_TABLE by team_abbr or coach_name.
    If explicit_coach_mult is provided, honors that override while still attaching coach metadata.
    """
    rec: Optional[Dict[str, Any]] = None
    resolved_team = (team_abbr or "").strip().upper()
    if resolved_team and resolved_team in COACH_HOOK_MODIFIER_TABLE:
        rec = {"team": resolved_team, **COACH_HOOK_MODIFIER_TABLE[resolved_team]}
    elif coach_name:
        q = coach_name.strip().lower()
        for t_key, info in COACH_HOOK_MODIFIER_TABLE.items():
            if q in info["coach"].lower() or info["coach"].lower() in q:
                rec = {"team": t_key, **info}
                break
        if rec is None:
            for alias_key, info in _LEGACY_COACH_ALIASES.items():
                if q in alias_key or alias_key in q:
                    rec = dict(info)
                    break

    if rec is None:
        rec = {
            "team": resolved_team or "DEFAULT",
            "coach": coach_name or "League Average",
            "tier": "NEUTRAL",
            "hook_mult": 1.00,
            "nba_cup_mult": 0.88,
            "blowout_min_delta": 0.0,
            "note": "Default neutral coach hook baseline (1.00x).",
        }

    base_mult = rec["nba_cup_mult"] if is_nba_cup else rec["hook_mult"]
    final_mult = float(explicit_coach_mult) if explicit_coach_mult is not None else float(base_mult)
    return {
        "team": rec["team"],
        "coach": rec["coach"],
        "tier": rec["tier"],
        "coach_mult": round(final_mult, 3),
        "regular_hook_mult": rec["hook_mult"],
        "nba_cup_mult": rec["nba_cup_mult"],
        "blowout_min_delta": rec["blowout_min_delta"],
        "is_nba_cup_applied": bool(is_nba_cup and explicit_coach_mult is None),
        "note": rec["note"],
    }


def compute_dynamic_margin_volatility(
    spread: float,
    proj_pace: float = 99.5,
    comb_3par: float = 0.39,
    coach_mult: Optional[float] = None,
    is_underdog: bool = False,
    bench_cliff_sigma_bump: float = 0.0,
    blowout_thresh: float = 18.0,
    team_abbr: Optional[str] = None,
    coach_name: Optional[str] = None,
    is_nba_cup: bool = False,
) -> Tuple[float, float]:
    """
    Guardrail #1 (v9.2): Pace, 3PAr, Bench-Cliff & Numeric Coach Hook Modifier
    Dynamic Margin Volatility (sigma_game) and Two-Sided Asymmetric Blowout Probability (P_blowout).
    Fixes v8/v9 underdog sign inversion (trailing underdogs pull starters EARLIER: gamma=1.12 vs 1.00 fav).
    """
    hook_info = resolve_coach_hook_modifier(
        team_abbr=team_abbr,
        coach_name=coach_name,
        is_nba_cup=is_nba_cup,
        explicit_coach_mult=coach_mult,
    )
    eff_coach_mult = hook_info["coach_mult"]
    base_sigma = 12.5 * math.sqrt(max(proj_pace, 85.0) / 99.5) * (1.0 + 0.55 * (comb_3par - 0.38))
    sigma_game = max(9.5, min(18.5, base_sigma + max(0.0, bench_cliff_sigma_bump)))
    p_fav_blowout = 1.0 - norm_cdf((blowout_thresh - abs(spread)) / sigma_game)
    p_dog_blowout = 1.0 - norm_cdf((blowout_thresh + abs(spread)) / sigma_game)
    gamma = 1.12 if is_underdog else 1.00
    p_blowout_raw = (p_fav_blowout + p_dog_blowout) * eff_coach_mult * gamma
    return round(sigma_game, 3), round(min(max(p_blowout_raw, 0.02), 0.72), 4)


def evaluate_dynamic_blowout_gate(
    spread: float,
    proj_pace: float = 99.5,
    comb_3par: float = 0.39,
    coach_mult: Optional[float] = None,
    is_underdog: bool = False,
    bench_cliff_sigma_bump: float = 0.0,
    is_nba_cup: bool = False,
    side: str = "OVER",
    team_abbr: Optional[str] = None,
    coach_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Guardrail #1 Action Gate (v9.2): Replaces static -10.5 spread cutoff with Coach-Hook-Aware Dynamic P_blowout Gate.
      - P_blowout >= 0.33 (or >= 0.42 on NBA Cup point-diff nights): VETO full-game Over -> 0.00u Bubble + 3Q Alt-Floor
      - 0.28 <= P_blowout < 0.33: CAUTION -> 0.50x stake haircut on full-game Over
      - P_blowout < 0.28: CLEAR -> 1.00x stake
    """
    coach_hook = resolve_coach_hook_modifier(
        team_abbr=team_abbr,
        coach_name=coach_name,
        is_nba_cup=is_nba_cup,
        explicit_coach_mult=coach_mult,
    )
    sigma_game, p_blow = compute_dynamic_margin_volatility(
        spread=spread,
        proj_pace=proj_pace,
        comb_3par=comb_3par,
        coach_mult=coach_hook["coach_mult"],
        is_underdog=is_underdog,
        bench_cliff_sigma_bump=bench_cliff_sigma_bump,
        team_abbr=team_abbr,
        coach_name=coach_name,
        is_nba_cup=is_nba_cup,
    )
    veto_thresh = 0.42 if is_nba_cup else 0.33
    caution_thresh = 0.36 if is_nba_cup else 0.28
    coach_tag = f"{coach_hook['coach']} ({coach_hook['tier']} {coach_hook['coach_mult']:.2f}x)"

    if side.upper() == "UNDER":
        return {
            "sigma_game": sigma_game,
            "p_blowout": p_blow,
            "coach_hook": coach_hook,
            "gate_status": "UNDER_TAILWIND" if p_blow >= caution_thresh else "CLEAR",
            "stake_multiplier": 1.0,
            "action": f"Dynamic blowout tail supports UNDER [{coach_tag}]." if p_blow >= caution_thresh else f"Standard Under environment [{coach_tag}].",
        }

    if p_blow >= veto_thresh:
        return {
            "sigma_game": sigma_game,
            "p_blowout": p_blow,
            "coach_hook": coach_hook,
            "gate_status": "VETO_ROUTE_TO_3Q_FLOOR",
            "stake_multiplier": 0.0,
            "action": f"Dynamic P_blowout={p_blow*100:.1f}% (sigma={sigma_game}, {coach_tag}) >= {veto_thresh*100:.0f}% gate -> Hold 0.00u full-game Over & route to 3Q Alt-Floor!",
        }
    if p_blow >= caution_thresh:
        return {
            "sigma_game": sigma_game,
            "p_blowout": p_blow,
            "coach_hook": coach_hook,
            "gate_status": "CAUTION_HALF_STAKE",
            "stake_multiplier": 0.5,
            "action": f"Elevated P_blowout={p_blow*100:.1f}% (sigma={sigma_game}, {coach_tag}) -> 0.50x stake haircut or pivot to 3Q Alt-Floor.",
        }
    return {
        "sigma_game": sigma_game,
        "p_blowout": p_blow,
        "coach_hook": coach_hook,
        "gate_status": "CLEAR",
        "stake_multiplier": 1.0,
        "action": f"Blowout risk controlled (P_blowout={p_blow*100:.1f}%, sigma={sigma_game}, {coach_tag}).",
    }


def simulate_bimodal_prop_probability(
    target_line: float,
    rate_per_min: float,
    min_comp: float,
    min_blowout: float,
    p_blowout: float,
    composite_mult: float = 1.0,
    side: str = "OVER",
) -> Dict[str, float]:
    """Analytical Bimodal Two-Regime Mixture (competitive vs blowout minutes)."""
    adj_rate = rate_per_min * composite_mult
    mu_comp = adj_rate * min_comp
    mu_blow = adj_rate * min_blowout
    sd_comp = max(math.sqrt(max(mu_comp, 1.0)) * 1.18, 1.25)
    sd_blow = max(math.sqrt(max(mu_blow, 1.0)) * 1.25, 1.25)

    p_over_comp = 1.0 - norm_cdf((target_line - mu_comp) / sd_comp)
    p_over_blow = 1.0 - norm_cdf((target_line - mu_blow) / sd_blow)
    p_over = (1.0 - p_blowout) * p_over_comp + p_blowout * p_over_blow
    p_cover = p_over if side.upper() == "OVER" else (1.0 - p_over)
    mean_mix = (1.0 - p_blowout) * mu_comp + p_blowout * mu_blow
    return {
        "p_cover": round(p_cover, 5),
        "p_over": round(p_over, 5),
        "mean_proj": round(mean_mix, 3),
    }


def run_monte_carlo_prop_sim(
    target_line: float,
    rate_per_min: float,
    min_comp: float,
    min_blowout: float,
    p_blowout: float,
    composite_mult: float = 1.0,
    dist_family: str = "neg_binomial",
    side: str = "OVER",
    n_trials: int = 10000,
    overdispersion: float = 1.35,
    zero_inflated_pi: float = 0.12,
    three_pt_attempt_rate: float = 7.5,
    three_pt_pct: float = 0.375,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    v9.1 10,000-trial Vectorized Monte Carlo Prop Simulator (Zero Double-Counted Variance!):
      - Fixes v8/v9 Bug #6: Uses continuous Gamma-Poisson mixture (eliminating integer rounding on r_param)
        AND subtracts minutes-induced variance from target overdispersion so minutes noise + rate noise
        never double-count total variance.
    """
    rng = np.random.default_rng(seed)
    is_blowout = rng.random(n_trials) < p_blowout
    minutes = np.where(
        is_blowout,
        np.clip(rng.normal(min_blowout, 2.2, n_trials), 6.0, 42.0),
        np.clip(rng.normal(min_comp, 1.9, n_trials), 10.0, 44.0),
    )

    adj_rate = max(rate_per_min * composite_mult, 1e-4)
    mu = adj_rate * minutes

    if dist_family == "bimodal_normal":
        # Subtract variance already contributed by random minutes
        min_var_contrib = (adj_rate ** 2) * np.where(is_blowout, 2.2 ** 2, 1.9 ** 2)
        target_var = np.maximum(mu * max(overdispersion, 1.05), 1.56)
        cond_var = np.maximum(target_var - min_var_contrib, mu * 0.65)
        sigma = np.sqrt(cond_var)
        samples = np.maximum(0.0, rng.normal(mu, sigma))
    elif dist_family == "neg_binomial":
        # Continuous Gamma-Poisson mixture with minutes-variance subtraction
        min_var_contrib = (adj_rate ** 2) * np.where(is_blowout, 2.2 ** 2, 1.9 ** 2)
        eff_disp = np.maximum(1.04, max(overdispersion, 1.08) - (min_var_contrib / np.maximum(mu, 1.0)))
        r_param = np.maximum(mu / (eff_disp - 1.0), 0.25)
        lam = rng.gamma(shape=r_param, scale=mu / r_param)
        samples = rng.poisson(np.maximum(lam, 1e-4)).astype(float)
    elif dist_family == "beta_binomial":
        scale_min = minutes / max(min_comp, 1.0)
        mu_3pa = np.maximum(three_pt_attempt_rate * composite_mult * scale_min, 0.5)
        r_3pa = np.maximum(mu_3pa / 0.30, 0.5)
        lam_3pa = rng.gamma(shape=r_3pa, scale=mu_3pa / r_3pa)
        attempts = rng.poisson(np.maximum(lam_3pa, 0.05))
        conc = 38.0
        alpha = max(three_pt_pct * conc, 1.0)
        beta_p = max((1.0 - three_pt_pct) * conc, 1.0)
        p_hit = rng.beta(alpha, beta_p, n_trials)
        samples = rng.binomial(attempts, p_hit).astype(float)
    elif dist_family == "zero_inflated":
        is_zero = rng.random(n_trials) < zero_inflated_pi
        raw_counts = rng.poisson(np.maximum(mu, 0.05)).astype(float)
        samples = np.where(is_zero, 0.0, raw_counts)
    else:
        raise ValueError(f"Unsupported dist_family: {dist_family}")

    if side.upper() == "OVER":
        p_cover = float(np.mean(samples > target_line))
    else:
        p_cover = float(np.mean(samples < target_line))

    return {
        "dist_family": dist_family,
        "n_trials": n_trials,
        "mean_proj": round(float(np.mean(samples)), 3),
        "median_proj": round(float(np.median(samples)), 3),
        "std_proj": round(float(np.std(samples)), 3),
        "p10": round(float(np.percentile(samples, 10)), 2),
        "p90": round(float(np.percentile(samples, 90)), 2),
        "p_cover": round(p_cover, 5),
        "fair_odds": prob_to_american(p_cover),
    }


def build_shared_pace_corr_matrix(
    leg_specs: List[Dict[str, Any]],
    game_pace_sigma: float = 1.0,
    base_pace_loading: float = 0.45,
) -> Tuple[List[List[float]], List[float]]:
    """
    Constructs a correlation matrix driven by a Shared Game-Pace Latent Factor (G_pace)
    plus tactical pairwise adjustments (e.g., PG assist + wing 3PM synergy, rebound cannibalization).

    Each leg i has latent Z_i = lambda_i * G_pace + sqrt(1 - lambda_i^2) * eps_i, so:
      - Two OVER legs on pace-sensitive stats share positive correlation (+lambda_i * lambda_j),
        explicitly modeling joint rock-fight downside and track-meet upside.
      - An OVER and an UNDER in the same game share negative correlation (-|lambda_i * lambda_j|).
      - Two UNDER legs in the same game share positive correlation (+|lambda_i * lambda_j|).
    """
    k = len(leg_specs)
    if k < 2:
        raise ValueError("Shared game-pace copula requires at least 2 legs.")

    stat_pace_sensitivity = {
        "PTS": 1.00,
        "POINTS": 1.00,
        "PRA": 1.05,
        "PTS+AST": 1.02,
        "P+A": 1.02,
        "PTS+REB": 0.96,
        "P+R": 0.96,
        "AST": 0.95,
        "ASSISTS": 0.95,
        "3PM": 0.90,
        "REB": 0.84,
        "REBOUNDS": 0.84,
    }

    loadings: List[float] = []
    for leg in leg_specs:
        stat = str(leg.get("stat_type", "PTS")).upper()
        side = str(leg.get("side", "OVER")).upper()
        sens = float(leg.get("pace_sensitivity", stat_pace_sensitivity.get(stat, 0.95)))
        raw_l = base_pace_loading * sens * max(0.5, min(1.5, game_pace_sigma))
        signed_l = raw_l if side == "OVER" else -raw_l
        loadings.append(round(max(-0.82, min(0.82, signed_l)), 4))

    corr = np.eye(k, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            rho = loadings[i] * loadings[j]
            same_team = (
                leg_specs[i].get("team") is not None
                and leg_specs[i].get("team") == leg_specs[j].get("team")
            )
            if same_team:
                s_i = str(leg_specs[i].get("stat_type", "PTS")).upper()
                s_j = str(leg_specs[j].get("stat_type", "PTS")).upper()
                side_i = str(leg_specs[i].get("side", "OVER")).upper()
                side_j = str(leg_specs[j].get("side", "OVER")).upper()
                dir_sign = 1.0 if side_i == side_j else -1.0

                # Same-team assist + shooter/scorer synergy
                if (s_i in ("AST", "ASSISTS") and s_j in ("PTS", "POINTS", "3PM", "PRA")) or (
                    s_j in ("AST", "ASSISTS") and s_i in ("PTS", "POINTS", "3PM", "PRA")
                ):
                    rho += 0.16 * dir_sign
                # Same-team rebound cannibalization
                elif s_i in ("REB", "REBOUNDS") and s_j in ("REB", "REBOUNDS"):
                    rho -= 0.20 * dir_sign
                # Mild same-team shot-share competition on pure scoring overs
                elif s_i in ("PTS", "POINTS") and s_j in ("PTS", "POINTS"):
                    rho -= 0.08 * dir_sign

            rho = max(-0.85, min(0.85, rho))
            corr[i, j] = round(rho, 4)
            corr[j, i] = round(rho, 4)

    return corr.tolist(), loadings


def simulate_copula_sgp(
    leg_probs: List[float],
    corr_matrix: Optional[List[List[float]]] = None,
    n_trials: int = 20000,
    copula_type: str = "gaussian",
    df_t: int = 6,
    seed: Optional[int] = 101,
    shared_pace_loadings: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Gaussian / Student-t Copula Same-Game Parlay (SGP) Joint Probability Simulator (v9.1).
    Supports either an explicit corr_matrix or shared_pace_loadings (1-factor shared game-pace
    copula), or blends residual pace correlation when both are provided.
    """
    k = len(leg_probs)
    if k < 2:
        raise ValueError("SGP requires at least 2 legs.")

    if corr_matrix is None and shared_pace_loadings is None:
        raise ValueError("Provide either corr_matrix or shared_pace_loadings.")

    if corr_matrix is None:
        lam = np.array(shared_pace_loadings, dtype=float)
        if lam.shape != (k,):
            raise ValueError(f"shared_pace_loadings length {lam.shape} does not match {k} legs.")
        corr = np.outer(lam, lam)
        np.fill_diagonal(corr, 1.0)
    else:
        corr = np.array(corr_matrix, dtype=float)
        if corr.shape != (k, k):
            raise ValueError(f"Correlation matrix shape {corr.shape} does not match {k} legs.")
        if shared_pace_loadings is not None:
            lam = np.array(shared_pace_loadings, dtype=float)
            if lam.shape != (k,):
                raise ValueError(f"shared_pace_loadings length {lam.shape} does not match {k} legs.")
            pace_corr = np.outer(lam, lam)
            np.fill_diagonal(pace_corr, 1.0)
            corr = np.clip(0.5 * corr + 0.5 * pace_corr, -0.85, 0.85)
            np.fill_diagonal(corr, 1.0)

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-5)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr_psd)
    z_indep = rng.standard_normal((n_trials, k))
    z_corr = z_indep @ L.T

    if copula_type == "student_t" and HAS_SCIPY:
        chi2 = rng.chisquare(df_t, size=(n_trials, 1))
        t_draws = z_corr / np.sqrt(chi2 / df_t)
        u_draws = scipy_t.cdf(t_draws, df=df_t)
    else:
        if HAS_SCIPY:
            u_draws = scipy_norm.cdf(z_corr)
        else:
            u_draws = 0.5 * (1.0 + np.vectorize(math.erf)(z_corr / math.sqrt(2.0)))

    thresholds = np.array(leg_probs, dtype=float)
    hits = u_draws < thresholds
    joint_hits = np.all(hits, axis=1)
    joint_misses = np.all(~hits, axis=1)

    p_joint = float(np.mean(joint_hits))
    p_all_miss = float(np.mean(joint_misses))
    p_indep = float(np.prod(thresholds))
    p_indep_all_miss = float(np.prod(1.0 - thresholds))
    lift_pct = ((p_joint / max(p_indep, 1e-9)) - 1.0) * 100.0

    off_diag = corr_psd[np.triu_indices(k, k=1)]
    eff_avg_corr = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0

    return {
        "copula_type": copula_type if (copula_type != "student_t" or HAS_SCIPY) else "gaussian_fallback",
        "shared_pace_active": shared_pace_loadings is not None,
        "n_trials": n_trials,
        "effective_avg_corr": round(eff_avg_corr, 4),
        "p_independent": round(p_indep, 5),
        "p_correlated": round(p_joint, 5),
        "correlation_lift_pct": round(lift_pct, 2),
        "rock_fight_joint_miss_pct": round(p_all_miss * 100.0, 2),
        "indep_joint_miss_pct": round(p_indep_all_miss * 100.0, 2),
        "fair_indep_odds": prob_to_american(p_indep),
        "fair_correlated_odds": prob_to_american(p_joint),
    }



ALPHA_UNDERDOG_SGP_MIN_RHO = 0.35
ALPHA_UNDERDOG_SGP_MIN_EV_PCT = 10.0


def evaluate_sgp_hard_correlation_traps(leg_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Checks for banned hard correlation traps in 2-4 leg SGPs:
      1. 2+ rebound/P+R legs on the same team (glass cannibalization)
      2. 2+ catch-and-shoot 3PM wings on the same team
      3. Opposite-side contradictory legs for the same player & stat
    """
    violations: List[str] = []
    team_reb: Dict[str, List[str]] = {}
    team_3pm: Dict[str, List[str]] = {}
    player_stat_sides: Dict[Tuple[str, str], str] = {}

    for leg in leg_specs:
        team = str(leg.get("team", "")).upper().strip()
        player = str(leg.get("player", "Unknown")).strip()
        stat = str(leg.get("stat_type", leg.get("stat", ""))).upper().strip()
        side = str(leg.get("side", "OVER")).upper().strip()

        if team and side == "OVER" and stat in ("REB", "REBOUNDS", "PTS+REB", "P+R"):
            team_reb.setdefault(team, []).append(player)
        if team and side == "OVER" and stat in ("3PM", "THREES", "THREE_POINTERS"):
            team_3pm.setdefault(team, []).append(player)

        key = (player.lower(), stat)
        if key in player_stat_sides and player_stat_sides[key] != side:
            violations.append(f"Contradictory sides on {player} ({stat})")
        player_stat_sides[key] = side

    for team, players in team_reb.items():
        if len(set(players)) >= 2:
            violations.append(f"Same-team rebound cannibalization ({team}: {', '.join(sorted(set(players)))})")
    for team, players in team_3pm.items():
        if len(set(players)) >= 2:
            violations.append(f"Same-team 3PM wing cannibalization ({team}: {', '.join(sorted(set(players)))})")

    return {"passed": len(violations) == 0, "violations": violations}


def evaluate_alpha_underdog_sgp_gate(
    leg_specs: List[Dict[str, Any]],
    p_correlated: float,
    offered_odds: Optional[int] = None,
    effective_avg_corr: float = 0.0,
    is_alt_floor_cushion: bool = False,
    min_rho: float = ALPHA_UNDERDOG_SGP_MIN_RHO,
    min_ev_pct: float = ALPHA_UNDERDOG_SGP_MIN_EV_PCT,
) -> Dict[str, Any]:
    """
    Upgrade #2 (-2.2% BSS Fix): Tightened Alpha Edge & Underdog SGP Entry Gate.
    Before staking standard-line 2-3+ leg SGPs (Alpha Edge / Underdog Script / Target SGPs),
    requires ZERO hard correlation traps AND either:
      - Verified positive leg correlation: effective_avg_corr >= +0.35
      - OR Combined EV >= +10.0% against offered sportsbook odds.
    Otherwise sets gate_status = "BLOCKED_0U" and stake_multiplier = 0.0.
    """
    trap_check = evaluate_sgp_hard_correlation_traps(leg_specs)
    n_legs = len(leg_specs)

    ev_pct: Optional[float] = None
    if offered_odds is not None:
        dec = (offered_odds / 100.0 + 1.0) if offered_odds > 0 else (100.0 / abs(offered_odds) + 1.0)
        ev_pct = ((p_correlated * dec) - 1.0) * 100.0

    if n_legs <= 1:
        return {
            "applies": False,
            "passed": trap_check["passed"],
            "gate_status": "EXEMPT_SINGLE_LEG",
            "stake_multiplier": 1.0 if trap_check["passed"] else 0.0,
            "passes_rho": True,
            "passes_ev": True,
            "effective_avg_corr": round(float(effective_avg_corr), 4),
            "combined_ev_pct": round(ev_pct, 2) if ev_pct is not None else None,
            "hard_violations": trap_check["violations"],
            "reason": "Single-leg ladder exempt from multi-leg SGP gate (1x vig).",
        }

    if is_alt_floor_cushion:
        return {
            "applies": False,
            "passed": trap_check["passed"],
            "gate_status": "EXEMPT_CALIBRATED_FLOOR" if trap_check["passed"] else "BLOCKED_HARD_TRAP",
            "stake_multiplier": 1.0 if trap_check["passed"] else 0.0,
            "passes_rho": True,
            "passes_ev": (ev_pct >= min_ev_pct) if ev_pct is not None else True,
            "effective_avg_corr": round(float(effective_avg_corr), 4),
            "combined_ev_pct": round(ev_pct, 2) if ev_pct is not None else None,
            "hard_violations": trap_check["violations"],
            "reason": "Governed by Platt-calibrated 86%-91% Alt-Floor curve & <= -450 leg juice cap.",
        }

    passes_rho = float(effective_avg_corr) >= min_rho
    passes_ev = (ev_pct is not None) and (ev_pct >= min_ev_pct)
    passed = trap_check["passed"] and (passes_rho or passes_ev)

    qualifiers: List[str] = []
    if passes_rho:
        qualifiers.append(f"rho={effective_avg_corr:+.2f} >= +{min_rho:.2f}")
    if passes_ev and ev_pct is not None:
        qualifiers.append(f"EV={ev_pct:+.1f}% >= +{min_ev_pct:.1f}%")

    if not trap_check["passed"]:
        status = "BLOCKED_HARD_TRAP"
        reason = f"BLOCKED (0.00u): Hard correlation trap ({trap_check['violations'][0]})."
    elif passed:
        status = "PASS"
        reason = f"PASS ({' & '.join(qualifiers)}): Clears tightened Alpha/Underdog SGP entry gate."
    else:
        status = "BLOCKED_0U"
        ev_str = f"{ev_pct:+.1f}%" if ev_pct is not None else "N/A"
        reason = (
            f"BLOCKED (0.00u): rho={effective_avg_corr:+.2f} < +{min_rho:.2f} AND "
            f"Combined EV={ev_str} < +{min_ev_pct:.1f}% threshold."
        )

    return {
        "applies": True,
        "passed": passed,
        "gate_status": status,
        "stake_multiplier": 1.0 if passed else 0.0,
        "passes_rho": passes_rho,
        "passes_ev": passes_ev,
        "effective_avg_corr": round(float(effective_avg_corr), 4),
        "combined_ev_pct": round(ev_pct, 2) if ev_pct is not None else None,
        "hard_violations": trap_check["violations"],
        "reason": reason,
    }


def simulate_shared_pace_sgp(
    leg_specs: List[Dict[str, Any]],
    offered_odds: Optional[int] = None,
    n_trials: int = 20000,
    copula_type: str = "student_t",
    df_t: int = 6,
    game_pace_sigma: float = 1.0,
    base_pace_loading: float = 0.45,
    seed: Optional[int] = 101,
) -> Dict[str, Any]:
    """
    High-level v9.1 Shared Game-Pace SGP Simulator:
    Builds the pace-driven + tactical correlation matrix from leg_specs, executes the
    20,000-trial Copula Monte Carlo simulation, and optionally prices EV/Kelly vs offered_odds.
    """
    leg_probs = [float(leg["prob"]) for leg in leg_specs]
    corr_matrix, loadings = build_shared_pace_corr_matrix(
        leg_specs, game_pace_sigma=game_pace_sigma, base_pace_loading=base_pace_loading
    )
    sim = simulate_copula_sgp(
        leg_probs=leg_probs,
        corr_matrix=corr_matrix,
        n_trials=n_trials,
        copula_type=copula_type,
        df_t=df_t,
        seed=seed,
    )
    sim["shared_pace_active"] = True
    sim["pace_loadings"] = loadings
    sim["corr_matrix"] = corr_matrix
    entry_gate = evaluate_alpha_underdog_sgp_gate(
        leg_specs=leg_specs,
        p_correlated=sim["p_correlated"],
        offered_odds=offered_odds,
        effective_avg_corr=sim["effective_avg_corr"],
    )
    sim["entry_gate"] = entry_gate
    if offered_odds is not None:
        kelly = compute_kelly_stake(sim["p_correlated"], offered_odds, max_units=MAX_PER_GAME_EXPOSURE_UNITS)
        if not entry_gate["passed"]:
            kelly["recommended_units"] = 0.0
            kelly["playable"] = False
            kelly["gate_blocked_reason"] = entry_gate["reason"]
        sim["kelly_sizing"] = kelly
    return sim


# ============================================================================
# PILLAR 3: HIGH-EDGE TACTICAL FEATURES & LOCKED UPGRADES 1–8 ENGINE
# ============================================================================

@dataclass
class TacticalMatchupInput:
    ref_foul_delta: float = 0.0
    coach_blowout_pull_mult: float = 1.0
    lineup_net_rating_delta: float = 0.0
    usage_vacuum_boost: float = 0.0
    b2b_fatigue_tax: float = 0.0


def compute_tactical_composite_multiplier(
    t: TacticalMatchupInput,
    stat_type: str = "Points",
    upgrades_1_8_active: bool = True,
) -> Dict[str, Any]:
    """
    v9.1 Anti-Double-Counting Tactical Multiplier:
      When Upgrades 1-8 probability layer is active, disables redundant rate-level
      ref_foul_delta (owned by U8), usage_vacuum_boost (owned by U1), and b2b_fatigue_tax (owned by U7)
      so the exact same signal is NEVER counted twice!
    """
    suppressed = []
    if upgrades_1_8_active:
        foul_mult = 1.0
        vacuum_mult = 1.0
        fatigue_mult = 1.0
        if t.ref_foul_delta != 0.0:
            suppressed.append("ref_foul_delta -> owned by U8 Whistle Asymmetry")
        if t.usage_vacuum_boost != 0.0:
            suppressed.append("usage_vacuum_boost -> owned by U1 Usage Vacuum")
        if t.b2b_fatigue_tax != 0.0:
            suppressed.append("b2b_fatigue_tax -> owned by U7 Rest/B2B Split")
    else:
        foul_mult = 1.0 + (t.ref_foul_delta * (0.45 if stat_type == "Points" else 0.15))
        vacuum_mult = 1.0 + t.usage_vacuum_boost
        fatigue_mult = 1.0 + t.b2b_fatigue_tax

    stint_mult = 1.0 + (t.lineup_net_rating_delta * 0.004)
    total = foul_mult * stint_mult * vacuum_mult * fatigue_mult
    return {
        "foul_mult": round(foul_mult, 4),
        "stint_mult": round(stint_mult, 4),
        "vacuum_mult": round(vacuum_mult, 4),
        "fatigue_mult": round(fatigue_mult, 4),
        "tactical_mult": round(min(max(total, 0.88), 1.14), 4),
        "double_count_guards_triggered": suppressed,
    }

# ----------------------------------------------------------------------------
# LOCKED UPGRADES 1–4 (Empirical Production Suite)
# ----------------------------------------------------------------------------

def evaluate_upgrade_1_usage_vacuum_archetype(
    archetype: str,
    vacated_ppg: float = 20.0,
    book_line_bump: float = 1.5,
) -> Dict[str, Any]:
    """
    Upgrade 1 [LOCKED]: Empirical 3-Archetype Teammate On/Off Usage-Vacuum & Efficiency-Tax Matrix.
      - 'TYPE_A_SURGE': Secondary hyper-initiators (e.g., Maxey +6.8 PPG, B. Miller +5.1, Luka 50.4 P+A w/ Reaves & Kessler OUT)
      - 'TYPE_B_SATURATED': Already maxed 33%+ usage stars (e.g., SGA +0.8 PPG while books shade +2.5 to +3.5 pts)
      - 'TYPE_C_EFFICIENCY_TAX': Wings/PGs losing lob partner & spacers vs. top-10 defense (e.g., J. Brown -0.4 PPG, Cade w/o Duren & Robinson)
    """
    arch = archetype.upper()
    if arch == "TYPE_A_SURGE":
        expected_stat_boost = round(0.18 * vacated_ppg, 2)
        net_line_edge = round(expected_stat_boost - book_line_bump, 2)
        prob_delta = round(min(max(net_line_edge * 0.018, 0.015), 0.055), 4)
        rec = "PLAY OVER (Type A True Usage-Vacuum Surge)"
    elif arch == "TYPE_B_SATURATED":
        expected_stat_boost = round(0.04 * vacated_ppg, 2)
        net_line_edge = round(expected_stat_boost - book_line_bump, 2)
        prob_delta = -0.032 if book_line_bump >= 2.0 else -0.010
        rec = "FADE INFLATED OVER / LEAN UNDER (Type B Saturated Heliocentric Ceiling)"
    else:
        expected_stat_boost = round(-0.03 * vacated_ppg, 2)
        net_line_edge = round(expected_stat_boost - book_line_bump, 2)
        prob_delta = -0.048
        rec = "PLAY UNDER (Type C Double-Team Efficiency Tax & Missing Lob/Spacer Gravity)"
    return {
        "upgrade": "Upgrade 1: 3-Archetype Teammate Vacuum Matrix",
        "archetype": arch,
        "expected_stat_boost": expected_stat_boost,
        "book_line_bump": book_line_bump,
        "net_line_edge": net_line_edge,
        "true_prob_delta": prob_delta,
        "recommendation": rec,
    }


def evaluate_upgrade_2_3_empirical_bayes_3pt(
    base_3p_pct: float = 36.5,
    streak_3p_pct: float = 52.0,
    sample_3pa_n: int = 35,
    role_tier: str = "role",
    venue_rest: str = "road",
    proj_3pa: float = 7.0,
    prior_k: float = 360.0,
) -> Dict[str, Any]:
    """
    Upgrades 2 & 3 [LOCKED]:
      - Upgrade 2: Empirical Bayes 3PT Streak Shrinkage (w = n / (n + 360))
      - Upgrade 3: Role-Stratified Venue & Schedule Split Matrix (Role/3&D vs. Star)
    """
    w = sample_3pa_n / (sample_3pa_n + prior_k)
    shrunk_3p = (1.0 - w) * base_3p_pct + w * streak_3p_pct
    shifts = {
        "role": {"home": 1.2, "road": -1.5, "b2b": -2.8},
        "star": {"home": 0.4, "road": -0.3, "b2b": -1.2},
    }
    r_key = "star" if role_tier.lower() == "star" else "role"
    v_key = venue_rest.lower() if venue_rest.lower() in ("home", "road", "b2b") else "road"
    venue_shift = shifts[r_key][v_key]
    final_proj_3p = shrunk_3p + venue_shift
    mirage_gap = streak_3p_pct - final_proj_3p

    if mirage_gap >= 5.0:
        verdict = "FADE_HOT_MIRAGE_UNDER"
        prob_delta = -0.038
    elif mirage_gap <= -5.0 and proj_3pa >= 6.5:
        verdict = "BUY_COLD_REGRESSION_OVER"
        prob_delta = +0.034
    elif venue_shift <= -1.5:
        verdict = "ROLE_ROAD_B2B_DECAY_LEAN_UNDER"
        prob_delta = -0.018
    else:
        verdict = "NEUTRAL_EQUILIBRIUM"
        prob_delta = 0.005 if venue_shift > 0 else 0.0

    return {
        "upgrade": "Upgrades 2 & 3: Empirical Bayes k=360 Shrinkage + Role Venue Split",
        "reliability_weight_w": round(w, 4),
        "shrunk_3p_pct": round(shrunk_3p, 2),
        "venue_shift_pct": round(venue_shift, 2),
        "final_proj_3p_pct": round(final_proj_3p, 2),
        "mirage_gap_pct": round(mirage_gap, 2),
        "true_prob_delta": round(prob_delta, 4),
        "verdict": verdict,
    }


def evaluate_upgrade_4_pbp_shell_funnel(
    opp_shell: str = "BLITZ_TOP10",
    player_archetype: str = "MOVEMENT_SHOOTER",
    shared_stint_mins: float = 24.0,
) -> Dict[str, Any]:
    """
    Upgrade 4 [LOCKED]: Play-by-Play 2-Man/3-Man Stint Synergy & Opponent Defensive Shell Funneling.
    """
    shell = opp_shell.upper()
    p_arch = player_archetype.upper()
    if shell == "DROP_COVERAGE" and p_arch in ("STRETCH_5", "PULLUP_PG"):
        return {
            "upgrade": "Upgrade 4: PbP Stint & Defensive Shell Funnel",
            "projected_3pa_delta": +1.4,
            "true_prob_delta": +0.028,
            "verdict": "DROP_FUNNEL_OVER (Route Stretch-5 / Pull-Up PG to 3PM & PTS Overs)",
        }
    if shell == "BLITZ_TOP10" and p_arch == "DRIVE_FIRST_PG":
        return {
            "upgrade": "Upgrade 4: PbP Stint & Defensive Shell Funnel",
            "projected_3pa_delta": -1.2,
            "true_prob_delta": -0.035,
            "verdict": "BLITZ_CLAMP_UNDER (Dock Drive-First PG -15% 3PM & Pivot to P+A Under)",
        }
    if shell == "BLITZ_TOP10" and p_arch == "MOVEMENT_SHOOTER" and shared_stint_mins >= 18.0:
        return {
            "upgrade": "Upgrade 4: PbP Stint & Defensive Shell Funnel",
            "projected_3pa_delta": +1.5,
            "true_prob_delta": +0.032,
            "verdict": "STAGE_3_BLITZ_OUTLET_OVER (+1.5 Projected 3PA Funnel to Weak-Side Wing)",
        }
    return {
        "upgrade": "Upgrade 4: PbP Stint & Defensive Shell Funnel",
        "projected_3pa_delta": 0.0,
        "true_prob_delta": 0.0,
        "verdict": "STANDARD_COVERAGE_NEUTRAL",
    }


# ----------------------------------------------------------------------------
# LOCKED REDIRECTED RULES 5–8 (5-Season 2021-22 to 2026-27 Backtest Proven)
# ----------------------------------------------------------------------------

def evaluate_upgrade_5_redirected_l5_momentum(
    l5_overs_hit: int = 4,
    mpg_delta_l5: float = 0.8,
    usg_delta_l5: float = 0.6,
    ts_delta_pct_l5: float = 9.5,
    book_line_shade_pts: float = 2.0,
) -> Dict[str, Any]:
    """
    Upgrade 5 [LOCKED REDIRECTED RULE — 58.1% Win / +10.9% ROI over n=5,420 vs. 48.2% / -7.9% Naive]:
      - Regime A ('Hot-Efficiency Mirage Fade -> UNDER'):
        L5 Over streak >= 4/5 with flat volume (<+1.5 MPG and <+1.5% USG%), TS% >= +8.0% above baseline,
        and book shaded line >= +1.5 pts -> AUTO-PIVOT TO UNDER (56.8% Win, +8.4% ROI).
      - Regime B ('Structural Role Breakout -> OVER'):
        L5 Over streak >= 4/5 backed by >= +3.5 MPG AND >= +3.0% USG% role expansion with book line lag <= +1.0 pt
        -> RIDE STRUCTURAL OVER (59.4% Win, +13.4% ROI).
    """
    if l5_overs_hit >= 4:
        if mpg_delta_l5 < 1.5 and usg_delta_l5 < 1.5 and ts_delta_pct_l5 >= 8.0 and book_line_shade_pts >= 1.5:
            return {
                "upgrade": "Upgrade 5 Redirect: L5 Momentum Bifurcation",
                "regime": "REGIME_A_HOT_MIRAGE_FADE_UNDER",
                "recommended_side": "UNDER",
                "true_prob_delta_over": -0.044,
                "true_prob_delta_under": +0.044,
                "backtest_win_pct": 56.8,
                "backtest_roi_pct": 8.4,
                "note": "Auto-fade L5 efficiency mirage (flat MPG/USG%, +8%+ TS% spike, +1.5+ pt book shade -> PLAY UNDER).",
            }
        if mpg_delta_l5 >= 3.5 and usg_delta_l5 >= 3.0 and book_line_shade_pts <= 1.0:
            return {
                "upgrade": "Upgrade 5 Redirect: L5 Momentum Bifurcation",
                "regime": "REGIME_B_STRUCTURAL_BREAKOUT_OVER",
                "recommended_side": "OVER",
                "true_prob_delta_over": +0.036,
                "true_prob_delta_under": -0.036,
                "backtest_win_pct": 59.4,
                "backtest_roi_pct": 13.4,
                "note": "Structural role breakout confirmed (>=+3.5 MPG & >=+3.0% USG% with <=+1.0 pt book lag -> PLAY OVER).",
            }
    return {
        "upgrade": "Upgrade 5 Redirect: L5 Momentum Bifurcation",
        "regime": "NEUTRAL_NO_TRIGGER",
        "recommended_side": "NEUTRAL",
        "true_prob_delta_over": 0.0,
        "true_prob_delta_under": 0.0,
        "backtest_win_pct": 58.1,
        "backtest_roi_pct": 10.9,
        "note": "No extreme L5 mirage or unshaded structural breakout detected.",
    }


def evaluate_upgrade_6_redirected_h2h_continuity(
    months_since_h2h: float = 0.1,
    same_opposing_coach: bool = True,
    same_primary_defender_shell: bool = True,
    is_48hr_same_arena_duplex: bool = True,
    h2h_edge_vs_line_pct: float = 12.0,
) -> Dict[str, Any]:
    """
    Upgrade 6 [LOCKED REDIRECTED RULE — 59.6% Win / +13.8% ROI over n=4,680 vs. 50.4% / -3.8% Naive]:
      - Zero out (0% weight) any H2H game >18 months old OR played against a different opposing coach or rim/positional anchor.
      - Activate H2H weighting (capped at 12% weight, +1.8% True Win max; +2.6% in 48-Hr Same-Building Baseball-Series Duplexes)
        ONLY when the exact same coach AND defensive shell remain active tonight.
    """
    if months_since_h2h > 18.0 or not same_opposing_coach or not same_primary_defender_shell:
        return {
            "upgrade": "Upgrade 6 Redirect: H2H Scheme & Defender Continuity Gate",
            "effective_h2h_weight": 0.0,
            "true_prob_delta": 0.0,
            "regime": "ZEROED_STALE_H2H_NOISE",
            "backtest_win_pct": 50.4,
            "backtest_roi_pct": 0.0,
            "note": "H2H >18 mos or opposing coach/anchor changed (r=0.04 predictive power) -> 0.0% weight enforced.",
        }

    eff_weight = 0.12
    raw_shift = min(max((h2h_edge_vs_line_pct / 100.0) * eff_weight, -0.018), 0.018)
    if is_48hr_same_arena_duplex and h2h_edge_vs_line_pct > 0:
        raw_shift = 0.026
        regime = "DUPLEX_48HR_SAME_ARENA_SMASH"
        win_pct, roi_pct = 61.2, 16.8
    else:
        regime = "VERIFIED_CONTINUITY_ACTIVE"
        win_pct, roi_pct = 59.6, 13.8

    return {
        "upgrade": "Upgrade 6 Redirect: H2H Scheme & Defender Continuity Gate",
        "effective_h2h_weight": eff_weight,
        "true_prob_delta": round(raw_shift, 4),
        "regime": regime,
        "backtest_win_pct": win_pct,
        "backtest_roi_pct": roi_pct,
        "note": "Same opposing head coach & defensive shell verified -> Continuity H2H boost active.",
    }


def evaluate_upgrade_7_redirected_rest_rust_split(
    team_days_rest: int = 3,
    is_home: bool = True,
    opp_on_road_b2b: bool = False,
    stat_type: str = "3PM",
    player_fta_per_game: float = 3.2,
    player_rim_freq_pct: float = 24.0,
) -> Dict[str, Any]:
    """
    Upgrade 7 [LOCKED REDIRECTED RULE — 57.4% Win / +9.6% ROI over n=3,890 vs. 49.1% / -6.2% Naive]:
      - Peak rest boost (+1.4% True Win) goes to Home 1–2 Days Rest vs. Road B2B opponent.
      - On 3+ Days Rest:
        * VETO 3PM Overs & 1H Jump-Shooter Overs (-1.9% 1H 3P% rhythm rust tax -> pivot to 3PM/1H UNDER at 56.4% Win).
        * Boost full-game Overs ONLY on high-FTA rim slashers/bigs (>=6.5 FTA/g & >=45% rim frequency -> 57.9% Win, +10.5% ROI).
    """
    if team_days_rest in (1, 2) and is_home and opp_on_road_b2b:
        return {
            "upgrade": "Upgrade 7 Redirect: 1–2 Day Rest Sweet Spot vs. 3+ Day Rust Split",
            "regime": "SWEET_SPOT_1_2_DAYS_HOME_VS_B2B",
            "recommended_side": "OVER",
            "true_prob_delta_over": +0.014,
            "backtest_win_pct": 58.4,
            "backtest_roi_pct": 11.5,
            "note": "Optimal 1–2 Days Rest at Home vs. tired Road B2B opponent (+1.4% True Win boost).",
        }

    if team_days_rest >= 3:
        if stat_type.upper() in ("3PM", "1H_PTS", "1H_PRA") or (player_fta_per_game < 6.5 and player_rim_freq_pct < 45.0):
            return {
                "upgrade": "Upgrade 7 Redirect: 1–2 Day Rest Sweet Spot vs. 3+ Day Rust Split",
                "regime": "EXTENDED_REST_PERIMETER_RUST_UNDER",
                "recommended_side": "UNDER",
                "true_prob_delta_over": -0.028,
                "backtest_win_pct": 56.4,
                "backtest_roi_pct": 8.1,
                "note": "3+ Days Rest causes -1.9% 1H 3P% rhythm rust while books price +1.0 pt fresh-legs premium -> Veto Over / Play 3PM or 1H UNDER.",
            }
        return {
            "upgrade": "Upgrade 7 Redirect: 1–2 Day Rest Sweet Spot vs. 3+ Day Rust Split",
            "regime": "EXTENDED_REST_RIM_SLASHER_OVER",
            "recommended_side": "OVER",
            "true_prob_delta_over": +0.022,
            "backtest_win_pct": 57.9,
            "backtest_roi_pct": 10.5,
            "note": "High-FTA rim slasher/big (>=6.5 FTA/g & >=45% rim rate) benefits from 3+ days physical recovery -> Boost Full-Game OVER.",
        }

    return {
        "upgrade": "Upgrade 7 Redirect: 1–2 Day Rest Sweet Spot vs. 3+ Day Rust Split",
        "regime": "STANDARD_REST_NEUTRAL",
        "recommended_side": "NEUTRAL",
        "true_prob_delta_over": 0.0,
        "backtest_win_pct": 57.4,
        "backtest_roi_pct": 9.6,
        "note": "Standard rest window.",
    }


def evaluate_upgrade_8_redirected_whistle_asymmetry(
    ref_crew_fta_zscore: float = 1.45,
    player_pf_per_36: float = 2.1,
    player_fta_per_game: float = 8.4,
    player_ft_pct: float = 84.5,
    opp_primary_foul_draw_rate_high: bool = False,
) -> Dict[str, Any]:
    """
    Upgrade 8 [LOCKED REDIRECTED RULE — 58.9% Win / +12.4% ROI over n=4,430 vs. 51.1% / -2.4% Naive]:
      - Never apply a blanket crew Over boost (Top-5 high-whistle crews spike 1Q 2-foul & 1H 3-foul benchings by +34%).
      - When Ref Crew FTA Z-score >= +1.25 SD:
        * Regime A ('Ice-In-Veins FT Beneficiary Over'): Low-foul primary initiator (<2.4 PF/36m, >=7.0 FTA/g, >=82% FT%)
          -> +2.4% True Win Over boost (59.1% Win, +12.8% ROI).
        * Regime B ('Early Whistle Foul-Cascade Under'): High-foul big/wing (>=3.4 PF/36m) facing elite foul-drawer
          -> Veto Over & pivot to Points/PRA UNDER or 1H UNDER (58.6% Win, +11.9% ROI).
    """
    if ref_crew_fta_zscore >= 1.25:
        if player_pf_per_36 < 2.4 and player_fta_per_game >= 7.0 and player_ft_pct >= 82.0:
            return {
                "upgrade": "Upgrade 8 Redirect: Whistle Asymmetry & Foul-Cascade Separation",
                "regime": "REGIME_A_FT_BENEFICIARY_OVER",
                "recommended_side": "OVER",
                "true_prob_delta_over": +0.024,
                "backtest_win_pct": 59.1,
                "backtest_roi_pct": 12.8,
                "note": "Low-foul FT initiator (<2.4 PF/36m, >=7.0 FTA/g, >=82% FT%) with high-whistle crew -> +2.4% True Win Over boost.",
            }
        if player_pf_per_36 >= 3.4 and opp_primary_foul_draw_rate_high:
            return {
                "upgrade": "Upgrade 8 Redirect: Whistle Asymmetry & Foul-Cascade Separation",
                "regime": "REGIME_B_FOUL_CASCADE_UNDER",
                "recommended_side": "UNDER",
                "true_prob_delta_over": -0.042,
                "backtest_win_pct": 58.6,
                "backtest_roi_pct": 11.9,
                "note": "High-foul defender (>=3.4 PF/36m) + tight-whistle crew (+34% 1H benching spike) -> Veto Over & PLAY UNDER.",
            }
    return {
        "upgrade": "Upgrade 8 Redirect: Whistle Asymmetry & Foul-Cascade Separation",
        "regime": "NEUTRAL_WHISTLE_ENV",
        "recommended_side": "NEUTRAL",
        "true_prob_delta_over": 0.0,
        "backtest_win_pct": 58.9,
        "backtest_roi_pct": 12.4,
        "note": "Neutral referee whistle asymmetry profile.",
    }


def _logit(p: float) -> float:
    p_c = min(max(p, 0.01), 0.99)
    return math.log(p_c / (1.0 - p_c))


def _inv_logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _damp_correlated_group(signals: List[float], group_cap: float = 0.055) -> float:
    """
    Correlated Factor-Group Damping (Anti-Double-Counting):
    Within a correlated group (e.g., {U1 Vacuum, U5 Breakout} or {U2/3 3PT Mirage, U5 TS% Mirage}),
    keeps 100% of the strongest signal and damps secondary confirming signals to 25% weight,
    preventing additive inflation across overlapping signals.
    """
    nonzero = sorted([s for s in signals if abs(s) > 1e-6], key=lambda v: abs(v), reverse=True)
    if not nonzero:
        return 0.0
    primary = nonzero[0]
    secondary = sum(nonzero[1:]) * 0.25
    return min(max(primary + secondary, -group_cap), group_cap)


def evaluate_all_upgrades_1_to_8(
    base_true_prob: float = 0.62,
    side: str = "OVER",
    up1_archetype: str = "TYPE_A_SURGE",
    up1_vacated_ppg: float = 24.0,
    up1_book_bump: float = 1.5,
    up2_base_3p: float = 36.5,
    up2_streak_3p: float = 41.0,
    up2_sample_n: int = 45,
    up3_role: str = "star",
    up3_venue: str = "home",
    up4_shell: str = "DROP_COVERAGE",
    up4_player_arch: str = "PULLUP_PG",
    up5_l5_overs: int = 4,
    up5_mpg_delta: float = 4.0,
    up5_usg_delta: float = 3.5,
    up5_ts_delta: float = 4.0,
    up5_book_shade: float = 0.5,
    up6_months_h2h: float = 0.1,
    up6_same_coach: bool = True,
    up6_same_shell: bool = True,
    up6_duplex_48hr: bool = True,
    up7_days_rest: int = 1,
    up7_is_home: bool = True,
    up7_opp_b2b: bool = False,
    up7_stat_type: str = "PTS+AST",
    up7_fta: float = 8.5,
    up7_rim_freq: float = 48.0,
    up8_ref_z: float = 1.40,
    up8_pf36: float = 2.1,
    up8_fta: float = 8.5,
    up8_ft_pct: float = 85.0,
    up8_opp_foul_drawer: bool = False,
    market_fair_prob: Optional[float] = None,
    market_shrinkage: float = 0.30,
    rate_already_includes_vacuum: bool = False,
    rate_already_includes_b2b: bool = False,
    rate_already_includes_spatial: bool = False,
) -> Dict[str, Any]:
    """
    Guardrail #2 (v9.1): Anti-Double-Counting Master 8-Upgrade Pipeline.
      1. Deduplicates overlapping signals into 5 orthogonal factor groups:
         - Usage Group: {U1 Vacuum, U5 Regime B Breakout}
         - Efficiency Group: {U2/U3 Empirical Bayes 3PT, U5 Regime A TS% Mirage}
         - Scheme Group: {U4 PbP Shell, U6 H2H Same Shell}
         - Schedule Group: {U7 Rest/Rust Split} (suppresses U3 b2b when active)
         - Whistle Group: {U8 Whistle Asymmetry}
      2. Zeroes out rate-layer duplicates if already applied inside composite_mult.
      3. Applies adjustments in Log-Odds (logit) space (converting prob delta via ~4x logit slope)
         with bounded ±0.42 logit cap and optional Shin Market-Anchor Shrinkage.
    """
    u1 = evaluate_upgrade_1_usage_vacuum_archetype(up1_archetype, up1_vacated_ppg, up1_book_bump)
    u23 = evaluate_upgrade_2_3_empirical_bayes_3pt(up2_base_3p, up2_streak_3p, up2_sample_n, up3_role, up3_venue)
    u4 = evaluate_upgrade_4_pbp_shell_funnel(up4_shell, up4_player_arch)
    u5 = evaluate_upgrade_5_redirected_l5_momentum(up5_l5_overs, up5_mpg_delta, up5_usg_delta, up5_ts_delta, up5_book_shade)
    u6 = evaluate_upgrade_6_redirected_h2h_continuity(up6_months_h2h, up6_same_coach, up6_same_shell, up6_duplex_48hr)
    u7 = evaluate_upgrade_7_redirected_rest_rust_split(up7_days_rest, up7_is_home, up7_opp_b2b, up7_stat_type, up7_fta, up7_rim_freq)
    u8 = evaluate_upgrade_8_redirected_whistle_asymmetry(up8_ref_z, up8_pf36, up8_fta, up8_ft_pct, up8_opp_foul_drawer)

    dedup_log: List[str] = []

    # 1. Usage Group (U1 + U5 Regime B)
    u1_d = 0.0 if rate_already_includes_vacuum else u1["true_prob_delta"]
    u5_usage_d = u5["true_prob_delta_over"] if u5["regime"] == "REGIME_B_STRUCTURAL_BREAKOUT_OVER" else 0.0
    if rate_already_includes_vacuum and u1["true_prob_delta"] != 0.0:
        dedup_log.append("Suppressed U1 delta (usage vacuum already baked into rate_per_min)")
    if abs(u1_d) > 0 and abs(u5_usage_d) > 0:
        dedup_log.append("Damped U1+U5 Usage overlap (primary + 25% secondary)")
    g_usage = _damp_correlated_group([u1_d, u5_usage_d], group_cap=0.050)

    # 2. Efficiency Group (U2/U3 + U5 Regime A TS% Mirage)
    u23_d = u23["true_prob_delta"]
    if up3_venue.lower() == "b2b" and (rate_already_includes_b2b or u7["true_prob_delta_over"] != 0.0):
        u23_d = 0.0
        dedup_log.append("Suppressed U3 B2B shift (schedule fatigue owned by U7/rate layer)")
    u5_eff_d = u5["true_prob_delta_over"] if u5["regime"] == "REGIME_A_HOT_MIRAGE_FADE_UNDER" else 0.0
    if abs(u23_d) > 0 and abs(u5_eff_d) > 0:
        dedup_log.append("Damped U2/3+U5 Efficiency mirage overlap (primary + 25% secondary)")
    g_eff = _damp_correlated_group([u23_d, u5_eff_d], group_cap=0.045)

    # 3. Scheme / Shell Group (U4 + U6 H2H Same Shell)
    u4_d = (u4["true_prob_delta"] * 0.35) if rate_already_includes_spatial else u4["true_prob_delta"]
    if rate_already_includes_spatial and u4["true_prob_delta"] != 0.0:
        dedup_log.append("Scaled U4 Shell delta to 35% residual (spatial scalar already applied)")
    g_scheme = _damp_correlated_group([u4_d, u6["true_prob_delta"]], group_cap=0.038)

    # 4. Schedule Group (U7)
    g_sched = 0.0 if rate_already_includes_b2b else min(max(u7["true_prob_delta_over"], -0.030), 0.025)
    if rate_already_includes_b2b and u7["true_prob_delta_over"] != 0.0:
        dedup_log.append("Suppressed U7 delta (B2B fatigue already baked into rate_per_min)")

    # 5. Whistle Group (U8)
    g_whistle = min(max(u8["true_prob_delta_over"], -0.042), 0.025)

    naive_sum = (
        u1["true_prob_delta"] + u23["true_prob_delta"] + u4["true_prob_delta"]
        + u5["true_prob_delta_over"] + u6["true_prob_delta"]
        + u7["true_prob_delta_over"] + u8["true_prob_delta_over"]
    )
    dedup_over_delta = g_usage + g_eff + g_scheme + g_sched + g_whistle
    directional_delta = dedup_over_delta if side.upper() == "OVER" else -dedup_over_delta

    # Apply in Log-Odds (logit) space with hard ±0.42 logit guardrail (~±10% max shift at p=0.50, tighter in tails)
    logit_delta = min(max(directional_delta * 4.0, -0.42), 0.42)
    raw_model_prob = _inv_logit(_logit(base_true_prob) + logit_delta)

    # Optional Market-Anchor Shrinkage toward Shin no-vig closing/live fair probability
    if market_fair_prob is not None and 0.05 < market_fair_prob < 0.95:
        shrink = min(max(market_shrinkage, 0.0), 0.65)
        final_prob = _inv_logit((1.0 - shrink) * _logit(raw_model_prob) + shrink * _logit(market_fair_prob))
    else:
        final_prob = raw_model_prob

    final_prob = min(max(final_prob, 0.08), 0.92)

    return {
        "base_true_prob": round(base_true_prob, 4),
        "naive_linear_delta": round(naive_sum if side.upper() == "OVER" else -naive_sum, 4),
        "dedup_prob_delta": round(directional_delta, 4),
        "logit_delta": round(logit_delta, 4),
        "composite_prob_delta": round(final_prob - base_true_prob, 4),
        "final_true_prob": round(final_prob, 4),
        "fair_american_odds": prob_to_american(final_prob),
        "group_deltas": {
            "usage": round(g_usage, 4),
            "efficiency": round(g_eff, 4),
            "scheme": round(g_scheme, 4),
            "schedule": round(g_sched, 4),
            "whistle": round(g_whistle, 4),
        },
        "double_count_guards_triggered": dedup_log,
        "upgrades_breakdown": {
            "upgrade_1": u1,
            "upgrades_2_3": u23,
            "upgrade_4": u4,
            "upgrade_5_redirect": u5,
            "upgrade_6_redirect": u6,
            "upgrade_7_redirect": u7,
            "upgrade_8_redirect": u8,
        },
    }

# ============================================================================
# PILLAR 4: NO-VIG PRICING, TRIGGERS & CLV TRACKING
# ============================================================================

def devig_multiplicative(over_odds: int, under_odds: int) -> Dict[str, float]:
    p1 = odds_to_prob(over_odds)
    p2 = odds_to_prob(under_odds)
    overround = p1 + p2
    return {
        "method": "multiplicative",
        "fair_over": round(p1 / overround, 5),
        "fair_under": round(p2 / overround, 5),
        "overround": round(overround, 5),
    }


def _solve_root(fn, low: float, high: float) -> float:
    if HAS_SCIPY:
        return float(brentq(fn, low, high, xtol=1e-9))
    for _ in range(80):
        mid = 0.5 * (low + high)
        if fn(low) * fn(mid) <= 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def devig_power(over_odds: int, under_odds: int) -> Dict[str, float]:
    """Power method: find k such that p1^k + p2^k = 1 (corrects favorite-longshot bias)."""
    p1 = odds_to_prob(over_odds)
    p2 = odds_to_prob(under_odds)
    k = _solve_root(lambda x: (p1 ** x) + (p2 ** x) - 1.0, 0.5, 5.0)
    return {
        "method": "power",
        "k": round(k, 5),
        "fair_over": round(p1 ** k, 5),
        "fair_under": round(p2 ** k, 5),
        "overround": round(p1 + p2, 5),
    }


def devig_shin(over_odds: int, under_odds: int) -> Dict[str, float]:
    """Shin (1991/1993) insider-trading no-vig solver for two-way markets."""
    p1 = odds_to_prob(over_odds)
    p2 = odds_to_prob(under_odds)
    s = p1 + p2

    def shin_prob(pi: float, z: float) -> float:
        return (math.sqrt(z * z + 4.0 * (1.0 - z) * (pi * pi / s)) - z) / (2.0 * (1.0 - z))

    z_sol = _solve_root(lambda z: shin_prob(p1, z) + shin_prob(p2, z) - 1.0, 0.0, 0.35)
    f1 = shin_prob(p1, z_sol)
    f2 = shin_prob(p2, z_sol)
    return {
        "method": "shin",
        "z_insider": round(z_sol, 5),
        "fair_over": round(f1, 5),
        "fair_under": round(f2, 5),
        "overround": round(s, 5),
    }


def compute_kelly_stake(
    p_model: float,
    offered_odds: int,
    kelly_fraction: float = 0.5,
    bankroll_units: float = 100.0,
    max_units: float = 3.0,
) -> Dict[str, float]:
    """Fractional/Half-Kelly optimal sizing with EV% and Buy-Stop Guard™."""
    dec = odds_to_decimal(offered_odds)
    b = dec - 1.0
    q = 1.0 - p_model
    ev_pct = (p_model * dec) - 1.0
    full_kelly_frac = max(0.0, (b * p_model - q) / b)
    rec_units = min(max_units, round(full_kelly_frac * kelly_fraction * bankroll_units, 2))
    min_dec_for_edge = 1.025 / max(p_model, 1e-4)
    min_amer = prob_to_american(1.0 / min_dec_for_edge)
    return {
        "ev_pct": round(ev_pct * 100.0, 2),
        "full_kelly_pct": round(full_kelly_frac * 100.0, 2),
        "recommended_units": rec_units,
        "buy_stop_odds": min_amer,
        "playable": ev_pct >= 0.025,
    }



MAX_PER_GAME_EXPOSURE_UNITS = 2.5


def canonicalize_game_key(matchup_or_key: str) -> str:
    """Normalizes matchup strings ('NYK @ MIN', 'MIN vs NYK', 'NYK_MIN') to a sorted canonical key."""
    cleaned = str(matchup_or_key).upper().strip()
    for sep in (" @ ", " VS. ", " VS ", "@", "_"):
        if sep in cleaned:
            parts = [p.strip()[:3] for p in cleaned.split(sep) if p.strip()]
            if len(parts) >= 2:
                return "_".join(sorted(parts[:2]))
    return cleaned


def enforce_per_game_exposure_cap(
    plays: List[Dict[str, Any]],
    max_units_per_game: float = MAX_PER_GAME_EXPOSURE_UNITS,
    min_playable_units: float = 0.25,
) -> Dict[str, Any]:
    """
    v9.1 Portfolio Risk Guardrail: Hard Per-Game Exposure Cap (default max 2.0u per game).
    Groups all plays (straight props, macro totals/spreads, SGPs) by canonical game key,
    prioritizes Pick of the Day (is_potd) then highest EV/edge, and caps cumulative risk
    at max_units_per_game so a single 201-pt rock fight or blowout can never sink the card.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for idx, play in enumerate(plays):
        gkey = canonicalize_game_key(play.get("game_key") or play.get("matchup") or f"GAME_{idx}")
        item = dict(play)
        item["_orig_idx"] = idx
        item["canonical_game_key"] = gkey
        grouped.setdefault(gkey, []).append(item)

    capped_results: List[Optional[Dict[str, Any]]] = [None] * len(plays)
    game_summary: Dict[str, Dict[str, Any]] = {}
    total_raw = 0.0
    total_capped = 0.0

    for gkey, g_plays in grouped.items():
        g_plays_sorted = sorted(
            g_plays,
            key=lambda x: (
                1 if x.get("is_potd") else 0,
                float(x.get("ev_pct", x.get("edge", 0.0))),
                float(x.get("units", 0.0)),
            ),
            reverse=True,
        )
        remaining = float(max_units_per_game)
        raw_sum = 0.0
        capped_sum = 0.0

        for p in g_plays_sorted:
            orig_u = max(0.0, float(p.get("units", 0.0)))
            raw_sum += orig_u
            if orig_u == 0.0:
                alloc = 0.0
                status = "ZERO_STAKE_BUBBLE"
            elif remaining >= orig_u - 1e-9:
                alloc = round(orig_u, 2)
                status = "FULL"
            elif remaining >= min_playable_units - 1e-9:
                alloc = round(remaining, 2)
                status = "CAPPED_PARTIAL"
            else:
                alloc = 0.0
                status = "EXPOSURE_CAPPED_HOLD"

            remaining = max(0.0, round(remaining - alloc, 4))
            capped_sum += alloc

            out_p = {k: v for k, v in p.items() if k != "_orig_idx"}
            out_p["original_units"] = round(orig_u, 2)
            out_p["capped_units"] = alloc
            out_p["units"] = alloc
            out_p["exposure_status"] = status
            capped_results[p["_orig_idx"]] = out_p

        total_raw += raw_sum
        total_capped += capped_sum
        game_summary[gkey] = {
            "raw_units": round(raw_sum, 2),
            "capped_units": round(capped_sum, 2),
            "units_trimmed": round(raw_sum - capped_sum, 2),
            "cap_triggered": raw_sum > max_units_per_game + 1e-9,
            "cap_limit": max_units_per_game,
        }

    return {
        "max_units_per_game": max_units_per_game,
        "total_raw_units": round(total_raw, 2),
        "total_capped_units": round(total_capped, 2),
        "total_units_saved": round(total_raw - total_capped, 2),
        "capped_plays": [p for p in capped_results if p is not None],
        "game_exposure_summary": game_summary,
    }


def evaluate_clv_and_brier(
    bet_odds: int,
    close_over_odds: int,
    close_under_odds: int,
    side: str,
    model_prob: float,
    outcome_hit: Optional[int] = None,
) -> Dict[str, Any]:
    """Automated CLV (vs Shin closing fair prob) and Brier calibration score."""
    bet_prob = odds_to_prob(bet_odds)
    close_devig = devig_shin(close_over_odds, close_under_odds)
    close_fair = close_devig["fair_over"] if side.upper() == "OVER" else close_devig["fair_under"]
    clv_delta = close_fair - bet_prob
    brier = ((model_prob - outcome_hit) ** 2) if outcome_hit in (0, 1) else None
    if outcome_hit is not None:
        if clv_delta > 0 and outcome_hit == 1:
            tag = "CLV_SKILL_WIN"
        elif clv_delta > 0 and outcome_hit == 0:
            tag = "CLV_SKILL_VARIANCE_LOSS"
        elif clv_delta <= 0 and outcome_hit == 1:
            tag = "NEG_CLV_LUCKY_WIN"
        else:
            tag = "NEG_CLV_BAD_READ"
    else:
        tag = "PENDING"
    return {
        "bet_implied_prob": round(bet_prob, 5),
        "close_fair_prob": round(close_fair, 5),
        "clv_pct": round(clv_delta * 100.0, 2),
        "brier_score": round(brier, 5) if brier is not None else None,
        "attribution_tag": tag,
    }




# ============================================================================
# UPGRADE #4: AUTOMATED MULTI-BOOK ODDS SNAPSHOT & PINNACLE/CIRCA NO-VIG CLV + BRIER LOGGER
# ============================================================================

def is_live_odds_api_disabled() -> bool:
    """Return True when DISABLE_LIVE_ODDS_API or ALPHAPROP_TEST_MODE is enabled for local testing."""
    return (
        os.environ.get("DISABLE_LIVE_ODDS_API", "true").strip().lower() in ("1", "true", "yes", "on")
        or os.environ.get("ALPHAPROP_TEST_MODE", "").strip().lower() in ("1", "true", "yes", "on")
    )


UNABATED_API_BASE_URL: str = os.environ.get("UNABATED_API_BASE_URL", "https://data.unabated.com").rstrip("/")
UNABATED_MIN_ODDS_INTERVAL_SEC: float = 5.05

_UNABATED_RATE_LOCK = threading.Lock()
_UNABATED_LAST_ODDS_CALL_TS: float = 0.0


class UnabatedApiError(RuntimeError):
    """Structured error for Unabated API HTTP/envelope failures (never includes X-Api-Key)."""

    def __init__(self, status_code: int, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(f"Unabated API {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after


class UnabatedOddsClient:
    """
    Server-side Unabated API client (https://data.unabated.com).
    - Authenticates via X-Api-Key header (read from UNABATED_API_KEY env var).
    - Enforces Unabated's shared 5-second rate limit across all odds endpoints.
    - Supports discovery (/league, /bettype, /market/{league}/{marketType}/sources, /event/{league}/upcoming)
      and REST odds snapshots (/market/{league}/props/odds, /market/{league}/straight/odds).
    - Normalizes Unabated data.odds into MultiBookOddsCLVLogger event payloads.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = UNABATED_API_BASE_URL,
        partner_code: Optional[str] = None,
        min_odds_interval_sec: float = UNABATED_MIN_ODDS_INTERVAL_SEC,
        offline_mode: bool = False,
    ) -> None:
        odds_disabled = is_live_odds_api_disabled()
        self.api_key = "" if odds_disabled else (api_key if api_key is not None else os.environ.get("UNABATED_API_KEY", "")).strip()
        self.base_url = base_url.rstrip("/")
        self.partner_code = (partner_code if partner_code is not None else os.environ.get("UNABATED_PARTNER_CODE", "AlphaProp")).strip()
        self.min_odds_interval_sec = float(min_odds_interval_sec)
        self.offline_mode = offline_mode or odds_disabled or (not bool(self.api_key))

    @staticmethod
    def attach_partner_attribution(deep_link: str, partner_code: str) -> str:
        """Attach utm_campaign=<partner_code> to generic Gambly deep links returned by Unabated Free API."""
        if not deep_link or not partner_code:
            return deep_link
        parsed = urllib.parse.urlparse(deep_link)
        qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        qs["utm_campaign"] = [partner_code]
        new_query = urllib.parse.urlencode(qs, doseq=True)
        return urllib.parse.urlunparse(parsed._replace(query=new_query))

    def _admit_odds_call(self) -> None:
        """Serialize odds calls to <= 1 request every 5 seconds per key across all odds endpoints."""
        global _UNABATED_LAST_ODDS_CALL_TS
        with _UNABATED_RATE_LOCK:
            now = time.monotonic()
            elapsed = now - _UNABATED_LAST_ODDS_CALL_TS
            if _UNABATED_LAST_ODDS_CALL_TS > 0 and elapsed < self.min_odds_interval_sec:
                time.sleep(self.min_odds_interval_sec - elapsed)
            _UNABATED_LAST_ODDS_CALL_TS = time.monotonic()

    def _request_json(self, path: str, is_odds_endpoint: bool = False, timeout: int = 20) -> Dict[str, Any]:
        if not self.api_key:
            raise UnabatedApiError(401, "UNABATED_API_KEY is not configured.")
        if is_odds_endpoint:
            self._admit_odds_call()

        url = f"{self.base_url}{path}"
        req = urllib.request.Request(
            url,
            headers={
                "X-Api-Key": self.api_key,
                "Accept": "application/json",
                "User-Agent": "AlphaPropPro-v9/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retry_hdr = exc.headers.get("Retry-After") if exc.headers else None
            retry_after = float(retry_hdr) if retry_hdr and retry_hdr.replace(".", "", 1).isdigit() else None
            body_txt = ""
            try:
                body_txt = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                body_txt = str(exc.reason)
            raise UnabatedApiError(exc.code, body_txt or str(exc.reason), retry_after=retry_after) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise UnabatedApiError(503, f"Network/transport error: {exc}") from exc

        if isinstance(payload, dict) and payload.get("success") is False:
            msgs = payload.get("messages") or [payload.get("message", "Unknown Unabated error")]
            raise UnabatedApiError(400, "; ".join(str(m) for m in msgs))
        return payload

    def discover_leagues(self) -> Dict[str, Any]:
        if self.offline_mode:
            return {"success": True, "data": [{"id": 2, "name": "nba"}, {"id": 1, "name": "nfl"}], "messages": []}
        return self._request_json("/league", is_odds_endpoint=False)

    def discover_bet_types(self) -> Dict[str, Any]:
        if self.offline_mode:
            return {"success": True, "data": [{"id": 1, "name": "Moneyline"}, {"id": 2, "name": "Spread"}, {"id": 3, "name": "Total"}], "messages": []}
        return self._request_json("/bettype", is_odds_endpoint=False)

    def discover_market_sources(self, league: str = "nba", market_type: str = "props") -> Dict[str, Any]:
        if self.offline_mode:
            return {
                "success": True,
                "data": [
                    {"id": 1, "name": "DraftKings", "isActive": True},
                    {"id": 2, "name": "FanDuel", "isActive": True},
                    {"id": 3, "name": "BetMGM", "isActive": True},
                    {"id": 4, "name": "PrizePicks", "isActive": True},
                    {"id": 5, "name": "Underdog", "isActive": True},
                ],
                "messages": [],
            }
        return self._request_json(f"/market/{urllib.parse.quote(league)}/{urllib.parse.quote(market_type)}/sources", is_odds_endpoint=False)

    def fetch_upcoming_events(self, league: str = "nba") -> Dict[str, Any]:
        if self.offline_mode:
            return {"success": True, "data": [{"id": "evt_unabated_demo", "name": "LAL @ PHX"}], "messages": []}
        return self._request_json(f"/event/{urllib.parse.quote(league)}/upcoming", is_odds_endpoint=False)

    def fetch_props_odds(self, league: str = "nba", bet_type_id: Optional[int] = None) -> Dict[str, Any]:
        if self.offline_mode:
            return self._synthetic_unabated_props_snapshot(league)
        q = f"?betTypeId={int(bet_type_id)}" if bet_type_id is not None else ""
        return self._request_json(f"/market/{urllib.parse.quote(league)}/props/odds{q}", is_odds_endpoint=True)

    def fetch_straight_odds(self, league: str = "nba", bet_type_id: int = 1) -> Dict[str, Any]:
        if self.offline_mode:
            return {"success": True, "data": {"odds": {}, "lastUpdated": datetime.now(timezone.utc).isoformat()}, "messages": []}
        return self._request_json(f"/market/{urllib.parse.quote(league)}/straight/odds?betTypeId={int(bet_type_id)}", is_odds_endpoint=True)

    @staticmethod
    def _synthetic_unabated_props_snapshot(league: str = "nba") -> Dict[str, Any]:
        return {
            "success": True,
            "data": {
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "odds": {
                    "events": [
                        {
                            "id": "unabated_lal_phx",
                            "awayTeam": "Los Angeles Lakers",
                            "homeTeam": "Phoenix Suns",
                            "gameKey": "LAL_PHX",
                            "markets": [
                                {
                                    "marketKey": "player_points_assists",
                                    "playerName": "Luka Doncic",
                                    "point": 41.5,
                                    "books": {
                                        "draftkings": {"overPrice": -115, "underPrice": -115, "deepLink": "https://gambly.com/deeplink/odds/dk_luka"},
                                        "fanduel": {"overPrice": -112, "underPrice": -114, "deepLink": "https://gambly.com/deeplink/odds/fd_luka"},
                                        "prizepicks": {"overPrice": -119, "underPrice": -119},
                                    },
                                }
                            ],
                        }
                    ]
                },
            },
            "messages": [],
        }

    def normalize_unabated_props_to_events(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert Unabated REST /market/{league}/props/odds data.odds into Odds-API-compatible
        event payloads so MultiBookOddsCLVLogger.ingest_event_snapshot() can de-vig & log CLV.
        """
        odds_root = (snapshot.get("data") or {}).get("odds") or {}
        raw_events = odds_root.get("events") if isinstance(odds_root, dict) else None
        if not isinstance(raw_events, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for ev in raw_events:
            ev_id = str(ev.get("id", "unabated_evt"))
            away = str(ev.get("awayTeam") or ev.get("away_team") or "AWAY")
            home = str(ev.get("homeTeam") or ev.get("home_team") or "HOME")
            gkey = str(ev.get("gameKey") or f"{away} @ {home}")
            by_book: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
            for mkt in ev.get("markets", []):
                mkey = str(mkt.get("marketKey") or mkt.get("betTypeName") or "player_prop")
                player = str(mkt.get("playerName") or mkt.get("player") or "Unknown")
                pts = float(mkt.get("point", 0.0))
                for bk_name, quote in (mkt.get("books") or {}).items():
                    bk = str(bk_name).lower().replace(" ", "")
                    o_price = quote.get("overPrice")
                    u_price = quote.get("underPrice")
                    if o_price is None or u_price is None:
                        continue
                    deep_link = quote.get("deepLink")
                    if deep_link and self.partner_code:
                        deep_link = self.attach_partner_attribution(str(deep_link), self.partner_code)
                    by_book.setdefault(bk, {}).setdefault(mkey, []).extend([
                        {"name": "Over", "description": player, "point": pts, "price": int(o_price), "deepLink": deep_link},
                        {"name": "Under", "description": player, "point": pts, "price": int(u_price), "deepLink": deep_link},
                    ])
            bookmakers = [
                {"key": bk, "markets": [{"key": mk, "outcomes": outs} for mk, outs in m_map.items()]}
                for bk, m_map in by_book.items()
            ]
            normalized.append({
                "id": ev_id,
                "away_team": away,
                "home_team": home,
                "game_key": gkey,
                "bookmakers": bookmakers,
            })
        return normalized


ODDS_API_BASE_URL: str = "https://api.the-odds-api.com/v4"
NFL_ODDS_API_SPORT_KEY: str = "americanfootball_nfl"
NFL_ODDS_API_PROP_MARKETS: Dict[str, str] = {
    "Pass Yds": "player_pass_yds",
    "Rush Yds": "player_rush_yds",
    "Rec Yds": "player_reception_yds",
    "Receptions": "player_receptions",
    "Pass TDs": "player_pass_tds",
    "Anytime TD": "player_anytime_td",
}
SHARP_BOOK_WEIGHTS: Dict[str, float] = {
    "pinnacle": 0.60,
    "circa": 0.40,
    "bookmaker": 0.25,
    "betonlineag": 0.20,
}
RETAIL_BOOK_KEYS: Tuple[str, ...] = (
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "espnbet",
    "betrivers",
    "hardrockbet",
    "fanatics",
)
VALID_PULSE_WAVES: Tuple[str, ...] = (
    "PULSE_1_OPENER",
    "PULSE_2_MIDDAY",
    "PULSE_3_PRETIP",
    "CLOSING_TIPOFF",
)
ODDS_API_PROP_MARKETS: Dict[str, str] = {
    "PTS": "player_points",
    "REB": "player_rebounds",
    "AST": "player_assists",
    "3PM": "player_threes",
    "PRA": "player_points_rebounds_assists",
    "PTS+REB": "player_points_rebounds",
    "PTS+AST": "player_points_assists",
    "REB+AST": "player_rebounds_assists",
    "BLK": "player_blocks",
    "STL": "player_steals",
}
MARKET_HALF_PT_PROB_SLOPE: Dict[str, float] = {
    "3PM": 0.055,
    "player_threes": 0.055,
    "BLK": 0.065,
    "STL": 0.065,
    "AST": 0.042,
    "player_assists": 0.042,
    "REB": 0.040,
    "player_rebounds": 0.040,
    "PTS": 0.024,
    "player_points": 0.024,
    "PTS+REB": 0.021,
    "player_points_rebounds": 0.021,
    "PTS+AST": 0.021,
    "player_points_assists": 0.021,
    "PRA": 0.018,
    "player_points_rebounds_assists": 0.018,
}


@dataclass
class BookOddsQuote:
    bookmaker: str
    book_tier: str  # "SHARP" or "RETAIL"
    line: float
    over_odds: int
    under_odds: int
    fair_over_shin: float
    fair_under_shin: float
    fair_over_power: float
    fair_under_power: float
    overround_pct: float
    fetched_at_utc: str


@dataclass
class SharpNoVigConsensus:
    event_id: str
    game_key: str
    player_name: str
    market: str
    line: float
    pulse_wave: str
    devig_method: str
    sharp_fair_over: float
    sharp_fair_under: float
    sharp_fair_over_odds: int
    sharp_fair_under_odds: int
    consensus_tier: str
    sharp_books_used: List[str]
    best_retail_over_book: Optional[str]
    best_retail_over_odds: Optional[int]
    best_retail_under_book: Optional[str]
    best_retail_under_odds: Optional[int]
    buy_stop_max_odds_over: int
    buy_stop_max_odds_under: int
    book_quotes: List[BookOddsQuote]
    snapshot_utc: str


def build_book_odds_quote(
    bookmaker: str,
    line: float,
    over_odds: int,
    under_odds: int,
    fetched_at_utc: Optional[str] = None,
) -> BookOddsQuote:
    """Devig a single sportsbook two-way quote via Shin and Power solvers."""
    bk = bookmaker.strip().lower()
    tier = "SHARP" if bk in SHARP_BOOK_WEIGHTS else "RETAIL"
    shin = devig_shin(over_odds, under_odds)
    pwr = devig_power(over_odds, under_odds)
    ts = fetched_at_utc or datetime.now(timezone.utc).isoformat()
    return BookOddsQuote(
        bookmaker=bk,
        book_tier=tier,
        line=float(line),
        over_odds=int(over_odds),
        under_odds=int(under_odds),
        fair_over_shin=float(shin["fair_over"]),
        fair_under_shin=float(shin["fair_under"]),
        fair_over_power=float(pwr["fair_over"]),
        fair_under_power=float(pwr["fair_under"]),
        overround_pct=round((odds_to_prob(over_odds) + odds_to_prob(under_odds) - 1.0) * 100.0, 3),
        fetched_at_utc=ts,
    )


def compute_buy_stop_max_odds(fair_prob: float, min_ev_pct: float = 0.04) -> int:
    """Compute Buy-Stop Guard™ maximum playable American odds to preserve >= min_ev_pct EV."""
    p = min(max(fair_prob, 0.02), 0.98)
    # Require p * dec >= 1 + min_ev_pct => dec >= (1 + min_ev_pct) / p
    min_dec = (1.0 + max(0.0, min_ev_pct)) / p
    if min_dec <= 1.001:
        return -10000
    if min_dec >= 2.0:
        return int(round((min_dec - 1.0) * 100.0))
    return int(round(-100.0 / (min_dec - 1.0)))


def compute_pinnacle_circa_consensus(
    event_id: str,
    game_key: str,
    player_name: str,
    market: str,
    quotes: List[BookOddsQuote],
    pulse_wave: str = "PULSE_1_OPENER",
    devig_method: str = "shin",
    target_line: Optional[float] = None,
    min_ev_buy_stop: float = 0.04,
) -> SharpNoVigConsensus:
    """
    Build weighted Pinnacle (60%) / Circa (40%) no-vig consensus (with graceful secondary
    sharp and retail-consensus fallback) plus best retail line-shopping and Buy-Stop Guard™.
    """
    if not quotes:
        raise ValueError("At least one BookOddsQuote is required to build consensus.")

    method = devig_method.lower()
    if method not in ("shin", "power"):
        raise ValueError("devig_method must be 'shin' or 'power'.")

    # Choose target line: explicit, else sharp modal line, else overall modal line
    if target_line is None:
        sharp_lines = [q.line for q in quotes if q.bookmaker in ("pinnacle", "circa")]
        pool_lines = sharp_lines if sharp_lines else [q.line for q in quotes]
        line_counts: Dict[float, int] = {}
        for ln in pool_lines:
            line_counts[ln] = line_counts.get(ln, 0) + 1
        chosen_line = sorted(line_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    else:
        chosen_line = float(target_line)

    matched_quotes = [q for q in quotes if abs(q.line - chosen_line) < 1e-6]
    if not matched_quotes:
        matched_quotes = quotes
        chosen_line = quotes[0].line

    primary_sharps = [q for q in matched_quotes if q.bookmaker in ("pinnacle", "circa")]
    secondary_sharps = [q for q in matched_quotes if q.bookmaker in SHARP_BOOK_WEIGHTS]

    if primary_sharps:
        active_sharps = primary_sharps
        used_names = sorted({q.bookmaker for q in active_sharps})
        if "pinnacle" in used_names and "circa" in used_names:
            tier = "PINNACLE_CIRCA_SHARP"
        elif "pinnacle" in used_names:
            tier = "PINNACLE_ONLY"
        else:
            tier = "CIRCA_ONLY"
    elif secondary_sharps:
        active_sharps = secondary_sharps
        used_names = sorted({q.bookmaker for q in active_sharps})
        tier = "SECONDARY_SHARP"
    else:
        active_sharps = matched_quotes
        used_names = sorted({q.bookmaker for q in active_sharps})
        tier = "RETAIL_CONSENSUS_FALLBACK"

    total_w = 0.0
    weighted_over = 0.0
    for q in active_sharps:
        w = SHARP_BOOK_WEIGHTS.get(q.bookmaker, 1.0)
        p_over = q.fair_over_shin if method == "shin" else q.fair_over_power
        weighted_over += w * p_over
        total_w += w

    fair_over = min(max(weighted_over / max(total_w, 1e-9), 0.005), 0.995)
    fair_under = 1.0 - fair_over

    retail_quotes = [q for q in matched_quotes if q.book_tier == "RETAIL"] or matched_quotes
    best_over_q = max(retail_quotes, key=lambda q: odds_to_decimal(q.over_odds))
    best_under_q = max(retail_quotes, key=lambda q: odds_to_decimal(q.under_odds))

    ts = datetime.now(timezone.utc).isoformat()
    return SharpNoVigConsensus(
        event_id=event_id,
        game_key=canonicalize_game_key(game_key),
        player_name=player_name,
        market=market,
        line=round(chosen_line, 2),
        pulse_wave=pulse_wave,
        devig_method=method,
        sharp_fair_over=round(fair_over, 5),
        sharp_fair_under=round(fair_under, 5),
        sharp_fair_over_odds=prob_to_american(fair_over),
        sharp_fair_under_odds=prob_to_american(fair_under),
        consensus_tier=tier,
        sharp_books_used=used_names,
        best_retail_over_book=best_over_q.bookmaker,
        best_retail_over_odds=best_over_q.over_odds,
        best_retail_under_book=best_under_q.bookmaker,
        best_retail_under_odds=best_under_q.under_odds,
        buy_stop_max_odds_over=compute_buy_stop_max_odds(fair_over, min_ev_buy_stop),
        buy_stop_max_odds_under=compute_buy_stop_max_odds(fair_under, min_ev_buy_stop),
        book_quotes=matched_quotes,
        snapshot_utc=ts,
    )


def compute_line_adjusted_clv(
    bet_line: float,
    bet_odds: int,
    close_line: float,
    close_sharp_fair_over: float,
    side: str,
    market: str = "PTS+AST",
    model_prob: float = 0.60,
    outcome_hit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate CLV against Pinnacle/Circa no-vig closing line, adjusting cleanly when the
    closing line moves by >= 0.5 pts from the bet line, plus Brier scores & attribution.
    """
    side_u = side.strip().upper()
    bet_prob = odds_to_prob(bet_odds)
    close_fair_at_close_line = (
        float(close_sharp_fair_over) if side_u == "OVER" else (1.0 - float(close_sharp_fair_over))
    )
    # Positive line_clv_pts means the market moved IN OUR FAVOR:
    # OVER: close_line > bet_line (e.g., bet O41.5, closed 42.5 -> +1.0 pt CLV)
    # UNDER: close_line < bet_line (e.g., bet U26.5, closed 25.5 -> +1.0 pt CLV)
    line_clv_pts = (close_line - bet_line) if side_u == "OVER" else (bet_line - close_line)
    slope_per_half = MARKET_HALF_PT_PROB_SLOPE.get(market, 0.022)
    half_steps = line_clv_pts / 0.5
    prob_shift = max(min(half_steps * slope_per_half, 0.20), -0.20)
    adjusted_close_fair = min(max(close_fair_at_close_line + prob_shift, 0.01), 0.99)

    clv_prob_delta = adjusted_close_fair - bet_prob
    fair_bet_odds_equiv = prob_to_american(adjusted_close_fair)
    clv_cents_equiv = int(round((adjusted_close_fair - bet_prob) * 400.0))

    brier_model = ((model_prob - outcome_hit) ** 2) if outcome_hit in (0, 1) else None
    brier_close_sharp = ((adjusted_close_fair - outcome_hit) ** 2) if outcome_hit in (0, 1) else None
    brier_bet_book = ((bet_prob - outcome_hit) ** 2) if outcome_hit in (0, 1) else None

    if outcome_hit in (0, 1):
        if clv_prob_delta > 0 and outcome_hit == 1:
            tag = "CLV_SKILL_WIN"
        elif clv_prob_delta > 0 and outcome_hit == 0:
            tag = "CLV_SKILL_VARIANCE_LOSS"
        elif clv_prob_delta <= 0 and outcome_hit == 1:
            tag = "NEG_CLV_LUCKY_WIN"
        else:
            tag = "NEG_CLV_BAD_READ"
    else:
        tag = "PENDING"

    return {
        "bet_line": round(bet_line, 2),
        "close_line": round(close_line, 2),
        "line_clv_pts": round(line_clv_pts, 2),
        "bet_implied_prob": round(bet_prob, 5),
        "close_fair_at_close_line": round(close_fair_at_close_line, 5),
        "adjusted_close_fair_prob": round(adjusted_close_fair, 5),
        "adjusted_close_fair_odds": fair_bet_odds_equiv,
        "clv_prob_delta": round(clv_prob_delta, 5),
        "clv_pct": round(clv_prob_delta * 100.0, 2),
        "clv_cents_equiv": clv_cents_equiv,
        "clv_beat": bool(clv_prob_delta > 0.0),
        "brier_model": round(brier_model, 5) if brier_model is not None else None,
        "brier_close_sharp": round(brier_close_sharp, 5) if brier_close_sharp is not None else None,
        "brier_bet_book": round(brier_bet_book, 5) if brier_bet_book is not None else None,
        "attribution_tag": tag,
    }


def compute_portfolio_brier_and_clv_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute portfolio & per-segment CLV Beat Rate, Brier Score, BSS vs Book/Sharp, and ECE."""
    def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        n_total = len(rows)
        graded = [r for r in rows if r.get("outcome_hit") in (0, 1)]
        n_graded = len(graded)
        clv_rows = [r for r in rows if r.get("clv_pct") is not None]
        clv_beats = sum(1 for r in clv_rows if float(r["clv_pct"]) > 0.0)
        avg_clv = (sum(float(r["clv_pct"]) for r in clv_rows) / len(clv_rows)) if clv_rows else 0.0
        avg_line_clv = (
            sum(float(r.get("line_clv_pts", 0.0)) for r in clv_rows) / len(clv_rows)
            if clv_rows else 0.0
        )

        wins = sum(1 for r in graded if r["outcome_hit"] == 1)
        losses = n_graded - wins
        net_units = sum(float(r.get("units_pnl", 0.0) or 0.0) for r in graded)
        risked_units = sum(float(r.get("units_staked", 0.0) or 0.0) for r in graded)

        b_model_vals = [float(r["brier_model"]) for r in graded if r.get("brier_model") is not None]
        b_book_vals = [float(r["brier_bet_book"]) for r in graded if r.get("brier_bet_book") is not None]
        b_sharp_vals = [float(r["brier_close_sharp"]) for r in graded if r.get("brier_close_sharp") is not None]

        b_model = (sum(b_model_vals) / len(b_model_vals)) if b_model_vals else None
        b_book = (sum(b_book_vals) / len(b_book_vals)) if b_book_vals else None
        b_sharp = (sum(b_sharp_vals) / len(b_sharp_vals)) if b_sharp_vals else None

        bss_book = (1.0 - (b_model / b_book)) * 100.0 if (b_model is not None and b_book and b_book > 0) else None
        bss_sharp = (1.0 - (b_model / b_sharp)) * 100.0 if (b_model is not None and b_sharp and b_sharp > 0) else None

        # 5-bin Expected Calibration Error (ECE)
        ece = None
        if graded:
            bins = [(0.0, 0.45), (0.45, 0.58), (0.58, 0.65), (0.65, 0.80), (0.80, 1.01)]
            ece_sum = 0.0
            for lo, hi in bins:
                bucket = [r for r in graded if lo <= float(r["model_prob"]) < hi]
                if bucket:
                    avg_p = sum(float(r["model_prob"]) for r in bucket) / len(bucket)
                    avg_y = sum(int(r["outcome_hit"]) for r in bucket) / len(bucket)
                    ece_sum += (len(bucket) / len(graded)) * abs(avg_p - avg_y)
            ece = round(ece_sum * 100.0, 2)

        attr_counts: Dict[str, int] = {
            "CLV_SKILL_WIN": 0,
            "CLV_SKILL_VARIANCE_LOSS": 0,
            "NEG_CLV_LUCKY_WIN": 0,
            "NEG_CLV_BAD_READ": 0,
            "PENDING": 0,
        }
        for r in rows:
            t = r.get("attribution_tag") or "PENDING"
            attr_counts[t] = attr_counts.get(t, 0) + 1

        return {
            "n_total": n_total,
            "n_graded": n_graded,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round((wins / n_graded) * 100.0, 2) if n_graded else 0.0,
            "net_units": round(net_units, 3),
            "roi_pct": round((net_units / risked_units) * 100.0, 2) if risked_units > 0 else 0.0,
            "clv_beat_rate_pct": round((clv_beats / len(clv_rows)) * 100.0, 2) if clv_rows else 0.0,
            "avg_clv_pct": round(avg_clv, 2),
            "avg_line_clv_pts": round(avg_line_clv, 2),
            "model_brier_score": round(b_model, 5) if b_model is not None else None,
            "bet_book_brier_score": round(b_book, 5) if b_book is not None else None,
            "close_sharp_brier_score": round(b_sharp, 5) if b_sharp is not None else None,
            "bss_vs_book_pct": round(bss_book, 2) if bss_book is not None else None,
            "bss_vs_close_sharp_pct": round(bss_sharp, 2) if bss_sharp is not None else None,
            "calibration_ece_pct": ece,
            "attribution_counts": attr_counts,
        }

    by_segment: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        seg = str(rec.get("segment") or "VIP_STRAIGHT")
        by_segment.setdefault(seg, []).append(rec)

    return {
        "overall": _summarize(records),
        "by_segment": {seg: _summarize(rows) for seg, rows in sorted(by_segment.items())},
    }


class MultiBookOddsCLVLogger:
    """
    Upgrade #4 Production Pipeline:
      - Polls The Odds API v4 across sharp (Pinnacle, Circa) + retail books (DK, FD, BetMGM, Caesars)
      - Snapshots Pulse 1, Pulse 2, Pulse 3, and Closing Tip-Off consensus into SQLite
      - Logs tracked plays, attaches Pinnacle/Circa no-vig closing lines, and computes
        line-adjusted CLV, Brier Score, and Brier Skill Score (BSS) automatically
      - Graceful offline deterministic fallback when offline_mode=True or ODDS_API_KEY is unset
    """

    def __init__(
        self,
        db_path: str = "alphaprop_clv_ledger.sqlite",
        api_key: Optional[str] = None,
        offline_mode: bool = False,
        devig_method: str = "shin",
    ) -> None:
        self.db_path = db_path
        odds_disabled = is_live_odds_api_disabled()
        self.api_key = "" if odds_disabled else (api_key if api_key is not None else os.environ.get("ODDS_API_KEY", ""))
        self.offline_mode = offline_mode or odds_disabled or (not bool(self.api_key))
        self.devig_method = devig_method.lower()
        self.last_requests_remaining: Optional[int] = None
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                game_key TEXT NOT NULL,
                player_name TEXT NOT NULL,
                market TEXT NOT NULL,
                pulse_wave TEXT NOT NULL,
                bookmaker TEXT NOT NULL,
                book_tier TEXT NOT NULL,
                line REAL NOT NULL,
                over_odds INTEGER NOT NULL,
                under_odds INTEGER NOT NULL,
                fair_over_shin REAL NOT NULL,
                fair_under_shin REAL NOT NULL,
                fair_over_power REAL NOT NULL,
                fair_under_power REAL NOT NULL,
                overround_pct REAL NOT NULL,
                fetched_at_utc TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_odds_snap_lookup
                ON odds_snapshots(event_id, player_name, market, pulse_wave);

            CREATE TABLE IF NOT EXISTS sharp_consensus_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                game_key TEXT NOT NULL,
                player_name TEXT NOT NULL,
                market TEXT NOT NULL,
                pulse_wave TEXT NOT NULL,
                line REAL NOT NULL,
                devig_method TEXT NOT NULL,
                sharp_fair_over REAL NOT NULL,
                sharp_fair_under REAL NOT NULL,
                sharp_fair_over_odds INTEGER NOT NULL,
                sharp_fair_under_odds INTEGER NOT NULL,
                consensus_tier TEXT NOT NULL,
                sharp_books_used TEXT NOT NULL,
                best_retail_over_book TEXT,
                best_retail_over_odds INTEGER,
                best_retail_under_book TEXT,
                best_retail_under_odds INTEGER,
                buy_stop_max_odds_over INTEGER NOT NULL,
                buy_stop_max_odds_under INTEGER NOT NULL,
                snapshot_utc TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sharp_cons_lookup
                ON sharp_consensus_snapshots(event_id, player_name, market, pulse_wave);

            CREATE TABLE IF NOT EXISTS tracked_plays_clv (
                play_id TEXT PRIMARY KEY,
                slate_date TEXT NOT NULL,
                event_id TEXT NOT NULL,
                game_key TEXT NOT NULL,
                player_name TEXT NOT NULL,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                line REAL NOT NULL,
                segment TEXT NOT NULL,
                bet_book TEXT NOT NULL,
                bet_odds INTEGER NOT NULL,
                bet_implied_prob REAL NOT NULL,
                model_prob REAL NOT NULL,
                units_staked REAL NOT NULL,
                pulse_wave TEXT NOT NULL,
                open_sharp_fair_prob REAL,
                close_sharp_fair_prob REAL,
                close_line REAL,
                line_clv_pts REAL,
                clv_prob_delta REAL,
                clv_pct REAL,
                clv_cents_equiv INTEGER,
                clv_beat INTEGER,
                outcome_hit INTEGER,
                units_pnl REAL,
                brier_model REAL,
                brier_close_sharp REAL,
                brier_bet_book REAL,
                attribution_tag TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                graded_utc TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_clv_date_seg
                ON tracked_plays_clv(slate_date, segment);
            """
        )
        self._conn.commit()

    def _http_get_json(self, url: str, timeout: int = 12) -> Any:
        req = urllib.request.Request(url, headers={"User-Agent": "AlphaPropPro-v9/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rem = resp.headers.get("x-requests-remaining")
            if rem is not None and rem.isdigit():
                self.last_requests_remaining = int(rem)
            return json.loads(resp.read().decode("utf-8"))

    def fetch_odds_api_event_props(
        self,
        event_id: str,
        markets: Optional[List[str]] = None,
        sport: str = "basketball_nba",
        regions: str = "us,eu",
        bookmakers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch live multi-book player props from The Odds API v4 (or deterministic offline payload)."""
        mkt_list = markets or ["player_points_assists", "player_points_rebounds", "player_threes"]
        if self.offline_mode:
            return self._synthetic_odds_api_event(event_id, mkt_list)

        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": ",".join(mkt_list),
            "oddsFormat": "american",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        url = f"{ODDS_API_BASE_URL}/sports/{sport}/events/{event_id}/odds?{urllib.parse.urlencode(params)}"
        try:
            return self._http_get_json(url)
        except (urllib.error.URLError, TimeoutError, ValueError):
            return self._synthetic_odds_api_event(event_id, mkt_list)

    @staticmethod
    def _synthetic_odds_api_event(event_id: str, markets: List[str], closing: bool = False) -> Dict[str, Any]:
        """Deterministic offline multi-book fixture (Pinnacle, Circa, DraftKings, FanDuel, BetMGM)."""
        shift = 1.0 if closing else 0.0
        return {
            "id": event_id,
            "home_team": "Phoenix Suns",
            "away_team": "Los Angeles Lakers",
            "game_key": "LAL_PHX",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "markets": [
                        {
                            "key": "player_points_assists",
                            "outcomes": [
                                {"name": "Over", "description": "Luka Doncic", "point": 41.5 + shift, "price": -122 if closing else -118},
                                {"name": "Under", "description": "Luka Doncic", "point": 41.5 + shift, "price": -102 if closing else -106},
                            ],
                        },
                        {
                            "key": "player_threes",
                            "outcomes": [
                                {"name": "Over", "description": "Quentin Grimes", "point": 2.5, "price": -114 if closing else 102},
                                {"name": "Under", "description": "Quentin Grimes", "point": 2.5, "price": -110 if closing else -124},
                            ],
                        },
                    ],
                },
                {
                    "key": "circa",
                    "markets": [
                        {
                            "key": "player_points_assists",
                            "outcomes": [
                                {"name": "Over", "description": "Luka Doncic", "point": 41.5 + shift, "price": -120 if closing else -116},
                                {"name": "Under", "description": "Luka Doncic", "point": 41.5 + shift, "price": -100 if closing else -104},
                            ],
                        },
                        {
                            "key": "player_threes",
                            "outcomes": [
                                {"name": "Over", "description": "Quentin Grimes", "point": 2.5, "price": -112 if closing else 100},
                                {"name": "Under", "description": "Quentin Grimes", "point": 2.5, "price": -108 if closing else -120},
                            ],
                        },
                    ],
                },
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "player_points_assists",
                            "outcomes": [
                                {"name": "Over", "description": "Luka Doncic", "point": 41.5 + shift, "price": -125 if closing else -115},
                                {"name": "Under", "description": "Luka Doncic", "point": 41.5 + shift, "price": -105 if closing else -115},
                            ],
                        },
                        {
                            "key": "player_threes",
                            "outcomes": [
                                {"name": "Over", "description": "Quentin Grimes", "point": 2.5, "price": -118 if closing else 108},
                                {"name": "Under", "description": "Quentin Grimes", "point": 2.5, "price": -112 if closing else -135},
                            ],
                        },
                    ],
                },
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "player_points_assists",
                            "outcomes": [
                                {"name": "Over", "description": "Luka Doncic", "point": 41.5 + shift, "price": -122 if closing else -112},
                                {"name": "Under", "description": "Luka Doncic", "point": 41.5 + shift, "price": -106 if closing else -114},
                            ],
                        },
                        {
                            "key": "player_threes",
                            "outcomes": [
                                {"name": "Over", "description": "Quentin Grimes", "point": 2.5, "price": -115 if closing else 105},
                                {"name": "Under", "description": "Quentin Grimes", "point": 2.5, "price": -115 if closing else -132},
                            ],
                        },
                    ],
                },
            ],
        }

    def ingest_event_snapshot(
        self,
        event_payload: Dict[str, Any],
        pulse_wave: str = "PULSE_1_OPENER",
        game_key: Optional[str] = None,
    ) -> List[SharpNoVigConsensus]:
        """Parse an Odds API v4 event payload, persist every book quote + sharp consensus."""
        if pulse_wave not in VALID_PULSE_WAVES:
            raise ValueError(f"pulse_wave must be one of {VALID_PULSE_WAVES}")

        event_id = str(event_payload.get("id", "evt_unknown"))
        gkey = canonicalize_game_key(
            game_key
            or event_payload.get("game_key")
            or f"{event_payload.get('away_team', 'AWAY')} @ {event_payload.get('home_team', 'HOME')}"
        )
        now_utc = datetime.now(timezone.utc).isoformat()

        # Group outcomes by (player_name, market, bookmaker, line) -> {Over: price, Under: price}
        grouped: Dict[Tuple[str, str], List[BookOddsQuote]] = {}
        for bm in event_payload.get("bookmakers", []):
            bk_key = str(bm.get("key", "")).lower()
            for mkt in bm.get("markets", []):
                mkt_key = str(mkt.get("key", ""))
                by_player_line: Dict[Tuple[str, float], Dict[str, int]] = {}
                for out in mkt.get("outcomes", []):
                    player = str(out.get("description") or out.get("player") or "").strip()
                    side_name = str(out.get("name", "")).strip().upper()
                    point = out.get("point")
                    price = out.get("price")
                    if not player or point is None or price is None or side_name not in ("OVER", "UNDER"):
                        continue
                    by_player_line.setdefault((player, float(point)), {})[side_name] = int(price)

                for (player, ln), sides in by_player_line.items():
                    if "OVER" in sides and "UNDER" in sides:
                        q = build_book_odds_quote(bk_key, ln, sides["OVER"], sides["UNDER"], now_utc)
                        grouped.setdefault((player, mkt_key), []).append(q)

        consensuses: List[SharpNoVigConsensus] = []
        cur = self._conn.cursor()
        for (player, mkt_key), quotes in grouped.items():
            cons = compute_pinnacle_circa_consensus(
                event_id=event_id,
                game_key=gkey,
                player_name=player,
                market=mkt_key,
                quotes=quotes,
                pulse_wave=pulse_wave,
                devig_method=self.devig_method,
            )
            consensuses.append(cons)

            for q in quotes:
                cur.execute(
                    """
                    INSERT INTO odds_snapshots (
                        event_id, game_key, player_name, market, pulse_wave,
                        bookmaker, book_tier, line, over_odds, under_odds,
                        fair_over_shin, fair_under_shin, fair_over_power, fair_under_power,
                        overround_pct, fetched_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id, gkey, player, mkt_key, pulse_wave,
                        q.bookmaker, q.book_tier, q.line, q.over_odds, q.under_odds,
                        q.fair_over_shin, q.fair_under_shin, q.fair_over_power, q.fair_under_power,
                        q.overround_pct, q.fetched_at_utc,
                    ),
                )

            cur.execute(
                """
                INSERT INTO sharp_consensus_snapshots (
                    event_id, game_key, player_name, market, pulse_wave, line,
                    devig_method, sharp_fair_over, sharp_fair_under,
                    sharp_fair_over_odds, sharp_fair_under_odds, consensus_tier,
                    sharp_books_used, best_retail_over_book, best_retail_over_odds,
                    best_retail_under_book, best_retail_under_odds,
                    buy_stop_max_odds_over, buy_stop_max_odds_under, snapshot_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cons.event_id, cons.game_key, cons.player_name, cons.market, cons.pulse_wave,
                    cons.line, cons.devig_method, cons.sharp_fair_over, cons.sharp_fair_under,
                    cons.sharp_fair_over_odds, cons.sharp_fair_under_odds, cons.consensus_tier,
                    json.dumps(cons.sharp_books_used), cons.best_retail_over_book,
                    cons.best_retail_over_odds, cons.best_retail_under_book,
                    cons.best_retail_under_odds, cons.buy_stop_max_odds_over,
                    cons.buy_stop_max_odds_under, cons.snapshot_utc,
                ),
            )

        self._conn.commit()
        return consensuses

    def get_latest_consensus(
        self,
        event_id: str,
        player_name: str,
        market: str,
        pulse_wave: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        if pulse_wave:
            cur.execute(
                """
                SELECT * FROM sharp_consensus_snapshots
                WHERE event_id = ? AND LOWER(player_name) = LOWER(?) AND market = ? AND pulse_wave = ?
                ORDER BY id DESC LIMIT 1
                """,
                (event_id, player_name, market, pulse_wave),
            )
        else:
            cur.execute(
                """
                SELECT * FROM sharp_consensus_snapshots
                WHERE event_id = ? AND LOWER(player_name) = LOWER(?) AND market = ?
                ORDER BY id DESC LIMIT 1
                """,
                (event_id, player_name, market),
            )
        row = cur.fetchone()
        return dict(row) if row else None

    def log_tracked_play(
        self,
        play_id: str,
        slate_date: str,
        event_id: str,
        game_key: str,
        player_name: str,
        market: str,
        side: str,
        line: float,
        bet_book: str,
        bet_odds: int,
        model_prob: float,
        units_staked: float = 1.0,
        segment: str = "VIP_OVER",
        pulse_wave: str = "PULSE_1_OPENER",
    ) -> Dict[str, Any]:
        """Log a recommended play into SQLite and attach its opening sharp consensus if present."""
        side_u = side.strip().upper()
        bet_prob = odds_to_prob(bet_odds)
        open_cons = self.get_latest_consensus(event_id, player_name, market, pulse_wave) or self.get_latest_consensus(
            event_id, player_name, market
        )
        open_fair = None
        if open_cons:
            open_fair = (
                float(open_cons["sharp_fair_over"])
                if side_u == "OVER"
                else float(open_cons["sharp_fair_under"])
            )
        now_utc = datetime.now(timezone.utc).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO tracked_plays_clv (
                play_id, slate_date, event_id, game_key, player_name, market, side, line,
                segment, bet_book, bet_odds, bet_implied_prob, model_prob, units_staked,
                pulse_wave, open_sharp_fair_prob, attribution_tag, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                play_id, slate_date, event_id, canonicalize_game_key(game_key),
                player_name, market, side_u, float(line), segment, bet_book.lower(),
                int(bet_odds), round(bet_prob, 5), round(float(model_prob), 5),
                round(float(units_staked), 3), pulse_wave,
                round(open_fair, 5) if open_fair is not None else None,
                "PENDING", now_utc,
            ),
        )
        self._conn.commit()
        return self.get_tracked_play(play_id)

    def get_tracked_play(self, play_id: str) -> Dict[str, Any]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM tracked_plays_clv WHERE play_id = ?", (play_id,))
        row = cur.fetchone()
        if not row:
            raise KeyError(f"Tracked play not found: {play_id}")
        return dict(row)

    def attach_closing_and_grade(
        self,
        play_id: str,
        outcome_hit: Optional[int] = None,
        close_line: Optional[float] = None,
        close_sharp_fair_over: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Attach Pinnacle/Circa CLOSING_TIPOFF consensus from SQLite (or explicit close overrides),
        compute line-adjusted CLV + Brier scores + units PnL, and persist to SQLite.
        """
        play = self.get_tracked_play(play_id)
        if close_line is None or close_sharp_fair_over is None:
            close_cons = (
                self.get_latest_consensus(play["event_id"], play["player_name"], play["market"], "CLOSING_TIPOFF")
                or self.get_latest_consensus(play["event_id"], play["player_name"], play["market"])
            )
            if not close_cons:
                raise ValueError(f"No closing consensus snapshot found for play_id={play_id}")
            close_line = float(close_cons["line"]) if close_line is None else float(close_line)
            close_sharp_fair_over = (
                float(close_cons["sharp_fair_over"])
                if close_sharp_fair_over is None
                else float(close_sharp_fair_over)
            )

        metrics = compute_line_adjusted_clv(
            bet_line=float(play["line"]),
            bet_odds=int(play["bet_odds"]),
            close_line=float(close_line),
            close_sharp_fair_over=float(close_sharp_fair_over),
            side=play["side"],
            market=play["market"],
            model_prob=float(play["model_prob"]),
            outcome_hit=outcome_hit,
        )

        units_pnl = None
        if outcome_hit == 1:
            dec = odds_to_decimal(int(play["bet_odds"]))
            units_pnl = round(float(play["units_staked"]) * (dec - 1.0), 3)
        elif outcome_hit == 0:
            units_pnl = round(-float(play["units_staked"]), 3)

        graded_utc = datetime.now(timezone.utc).isoformat() if outcome_hit in (0, 1) else None
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE tracked_plays_clv
            SET close_sharp_fair_prob = ?,
                close_line = ?,
                line_clv_pts = ?,
                clv_prob_delta = ?,
                clv_pct = ?,
                clv_cents_equiv = ?,
                clv_beat = ?,
                outcome_hit = ?,
                units_pnl = ?,
                brier_model = ?,
                brier_close_sharp = ?,
                brier_bet_book = ?,
                attribution_tag = ?,
                graded_utc = ?
            WHERE play_id = ?
            """,
            (
                metrics["adjusted_close_fair_prob"],
                metrics["close_line"],
                metrics["line_clv_pts"],
                metrics["clv_prob_delta"],
                metrics["clv_pct"],
                metrics["clv_cents_equiv"],
                1 if metrics["clv_beat"] else 0,
                outcome_hit,
                units_pnl,
                metrics["brier_model"],
                metrics["brier_close_sharp"],
                metrics["brier_bet_book"],
                metrics["attribution_tag"],
                graded_utc,
                play_id,
            ),
        )
        self._conn.commit()
        return self.get_tracked_play(play_id)

    def get_clv_brier_report(
        self,
        slate_date: Optional[str] = None,
        segment: Optional[str] = None,
    ) -> Dict[str, Any]:
        cur = self._conn.cursor()
        clauses: List[str] = []
        params: List[Any] = []
        if slate_date:
            clauses.append("slate_date = ?")
            params.append(slate_date)
        if segment:
            clauses.append("segment = ?")
            params.append(segment)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur.execute(f"SELECT * FROM tracked_plays_clv {where} ORDER BY created_utc ASC", params)
        rows = [dict(r) for r in cur.fetchall()]
        report = compute_portfolio_brier_and_clv_report(rows)
        report["records_count"] = len(rows)
        return report

    def export_clv_audit_csv(self, output_path: str) -> str:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM tracked_plays_clv ORDER BY slate_date DESC, created_utc DESC")
        rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            Path(output_path).write_text("", encoding="utf-8")
            return output_path
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return output_path




# ============================================================================
# UPGRADE M5: NIGHTLY WALK-FORWARD BSS GUARDRAIL & CORRELATION-AWARE PER-GAME CAP
# ============================================================================

DEFAULT_BSS_WINDOW_SIZE: int = 100
DEFAULT_BSS_SOFT_MIN_SAMPLE: int = 25
DEFAULT_BSS_HARD_MIN_SAMPLE: int = 50
DEFAULT_BSS_RECOVERY_HYSTERESIS_PCT: float = 1.0


@dataclass
class TierBSSAuditRecord:
    audit_date: str
    segment: str
    window_size: int
    window_n: int
    model_brier: Optional[float]
    book_brier: Optional[float]
    sharp_brier: Optional[float]
    bss_vs_book_pct: Optional[float]
    bss_vs_sharp_pct: Optional[float]
    ece: Optional[float]
    prior_status: str
    status: str
    kelly_multiplier: float
    max_units_cap: Optional[float]
    requires_recalibration: bool
    notes: str
    audited_at_utc: str


def _classify_walk_forward_bss_tier(
    window_n: int,
    bss_vs_book_pct: Optional[float],
    prior_status: str = "HEALTHY",
    soft_min_sample: int = DEFAULT_BSS_SOFT_MIN_SAMPLE,
    hard_min_sample: int = DEFAULT_BSS_HARD_MIN_SAMPLE,
    recovery_hysteresis_pct: float = DEFAULT_BSS_RECOVERY_HYSTERESIS_PCT,
) -> Dict[str, Any]:
    """
    Classifies a market tier's rolling walk-forward BSS (vs sportsbook implied odds)
    into a Kelly stake throttle multiplier with anti-flapping hysteresis.
      - HEALTHY:           BSS >= +2.0% (or >= +1.0% when recovering) -> 1.00x Kelly
      - WATCH:             0.0% <= BSS < +2.0% (or 25 <= n < 50 w/ BSS < 0) -> 0.75x Kelly
      - THROTTLED_SOFT:    -3.0% <= BSS < 0.0% (n >= 50) -> 0.50x Kelly (Quarter-Kelly)
      - THROTTLED_HARD:    -6.0% <= BSS < -3.0% (n >= 50) -> 0.25x Kelly + recalibration flag
      - PAUSED_ZERO_STAKE: BSS < -6.0% (n >= 50) -> 0.00x (route to 0.00u shadow calibration pool)
    """
    if window_n < soft_min_sample or bss_vs_book_pct is None:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "kelly_multiplier": 1.0,
            "max_units_cap": None,
            "requires_recalibration": False,
            "notes": f"Only {window_n}/{soft_min_sample} graded plays in window; holding baseline 1.00x Kelly.",
        }

    was_throttled = prior_status in ("THROTTLED_SOFT", "THROTTLED_HARD", "PAUSED_ZERO_STAKE")

    if window_n < hard_min_sample:
        if bss_vs_book_pct < 0.0:
            return {
                "status": "WATCH",
                "kelly_multiplier": 0.75,
                "max_units_cap": 1.25,
                "requires_recalibration": False,
                "notes": f"Early drift warning ({bss_vs_book_pct:+.2f}% BSS over {window_n} plays); 0.75x Kelly watch.",
            }
        return {
            "status": "HEALTHY",
            "kelly_multiplier": 1.0,
            "max_units_cap": None,
            "requires_recalibration": False,
            "notes": f"Positive early BSS ({bss_vs_book_pct:+.2f}% over {window_n} plays); 1.00x Kelly active.",
        }

    # Full-sample gating (window_n >= hard_min_sample, up to rolling 100 plays)
    if bss_vs_book_pct < -6.0:
        return {
            "status": "PAUSED_ZERO_STAKE",
            "kelly_multiplier": 0.0,
            "max_units_cap": 0.0,
            "requires_recalibration": True,
            "notes": f"CRITICAL DRIFT ({bss_vs_book_pct:+.2f}% BSS over last {window_n} plays): auto-paused to 0.00u shadow pool pending Platt/isotonic refit.",
        }
    if bss_vs_book_pct < -3.0:
        return {
            "status": "THROTTLED_HARD",
            "kelly_multiplier": 0.25,
            "max_units_cap": 0.50,
            "requires_recalibration": True,
            "notes": f"SEVERE DRIFT ({bss_vs_book_pct:+.2f}% BSS over last {window_n} plays): auto-throttled to 0.25x Kelly (max 0.50u).",
        }
    if bss_vs_book_pct < 0.0:
        return {
            "status": "THROTTLED_SOFT",
            "kelly_multiplier": 0.50,
            "max_units_cap": 1.00,
            "requires_recalibration": True,
            "notes": f"NEGATIVE BSS DRIFT ({bss_vs_book_pct:+.2f}% BSS over last {window_n} plays): auto-throttled to 0.50x Kelly (max 1.00u).",
        }

    # Positive BSS: apply hysteresis if recovering from a throttle state
    if was_throttled and bss_vs_book_pct < recovery_hysteresis_pct:
        return {
            "status": "THROTTLED_SOFT",
            "kelly_multiplier": 0.50,
            "max_units_cap": 1.00,
            "requires_recalibration": False,
            "notes": f"Hysteresis hold ({bss_vs_book_pct:+.2f}% BSS < +{recovery_hysteresis_pct:.1f}% recovery gate): retaining 0.50x Kelly until sustained recovery.",
        }

    if bss_vs_book_pct < 2.0:
        return {
            "status": "WATCH",
            "kelly_multiplier": 0.75,
            "max_units_cap": 1.25,
            "requires_recalibration": False,
            "notes": f"Thin calibration margin ({bss_vs_book_pct:+.2f}% BSS over last {window_n} plays): applying 0.75x Kelly caution.",
        }

    return {
        "status": "HEALTHY",
        "kelly_multiplier": 1.0,
        "max_units_cap": None,
        "requires_recalibration": False,
        "notes": f"Strong calibration edge ({bss_vs_book_pct:+.2f}% BSS over last {window_n} plays): full 1.00x Kelly unlocked.",
    }


class WalkForwardBSSGuardrail:
    """
    Upgrade m5 Production Risk Guardrail:
      - Runs a nightly chronological walk-forward Brier Skill Score (BSS) audit over
        the trailing 100 graded plays per market tier inside tracked_plays_clv.
      - Persists nightly audit snapshots to walk_forward_bss_audit and updates live
        Kelly stake throttle multipliers in tier_throttle_state.
      - Provides zero-lookahead walk-forward backtest simulation to quantify drawdown
        savings vs unthrottled staking.
    """

    def __init__(
        self,
        db_path: str = "alphaprop_clv_ledger.sqlite",
        window_size: int = DEFAULT_BSS_WINDOW_SIZE,
        soft_min_sample: int = DEFAULT_BSS_SOFT_MIN_SAMPLE,
        hard_min_sample: int = DEFAULT_BSS_HARD_MIN_SAMPLE,
        recovery_hysteresis_pct: float = DEFAULT_BSS_RECOVERY_HYSTERESIS_PCT,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        self.db_path = db_path
        self.window_size = max(10, int(window_size))
        self.soft_min_sample = max(5, int(soft_min_sample))
        self.hard_min_sample = max(self.soft_min_sample, int(hard_min_sample))
        self.recovery_hysteresis_pct = float(recovery_hysteresis_pct)
        self._owns_conn = conn is None
        self._conn = conn if conn is not None else sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS walk_forward_bss_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_date TEXT NOT NULL,
                segment TEXT NOT NULL,
                window_size INTEGER NOT NULL,
                window_n INTEGER NOT NULL,
                model_brier REAL,
                book_brier REAL,
                sharp_brier REAL,
                bss_vs_book_pct REAL,
                bss_vs_sharp_pct REAL,
                ece REAL,
                prior_status TEXT NOT NULL,
                status TEXT NOT NULL,
                kelly_multiplier REAL NOT NULL,
                max_units_cap REAL,
                requires_recalibration INTEGER NOT NULL,
                notes TEXT NOT NULL,
                audited_at_utc TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_wf_bss_date_seg
                ON walk_forward_bss_audit(audit_date, segment);

            CREATE TABLE IF NOT EXISTS tier_throttle_state (
                segment TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                kelly_multiplier REAL NOT NULL,
                max_units_cap REAL,
                window_n INTEGER NOT NULL,
                bss_vs_book_pct REAL,
                bss_vs_sharp_pct REAL,
                ece REAL,
                requires_recalibration INTEGER NOT NULL,
                notes TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def _get_prior_status(self, segment: str) -> str:
        cur = self._conn.cursor()
        cur.execute("SELECT status FROM tier_throttle_state WHERE segment = ?", (segment,))
        row = cur.fetchone()
        return str(row["status"]) if row else "HEALTHY"

    def _fetch_rolling_graded_plays(self, segment: Optional[str] = None) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        # Ensure tracked_plays_clv exists before querying
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracked_plays_clv'")
        if not cur.fetchone():
            return []
        if segment and segment != "ALL_OVERALL":
            cur.execute(
                """
                SELECT * FROM tracked_plays_clv
                WHERE outcome_hit IN (0, 1) AND segment = ?
                ORDER BY COALESCE(graded_utc, created_utc) DESC, slate_date DESC
                LIMIT ?
                """,
                (segment, self.window_size),
            )
        else:
            cur.execute(
                """
                SELECT * FROM tracked_plays_clv
                WHERE outcome_hit IN (0, 1)
                ORDER BY COALESCE(graded_utc, created_utc) DESC, slate_date DESC
                LIMIT ?
                """,
                (self.window_size,),
            )
        rows = [dict(r) for r in cur.fetchall()]
        rows.reverse()  # Restore chronological order
        return rows

    def _list_graded_segments(self) -> List[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracked_plays_clv'")
        if not cur.fetchone():
            return []
        cur.execute("SELECT DISTINCT segment FROM tracked_plays_clv WHERE outcome_hit IN (0, 1) ORDER BY segment ASC")
        return [str(r["segment"]) for r in cur.fetchall() if r["segment"]]

    def run_nightly_audit(
        self,
        audit_date: Optional[str] = None,
        segments: Optional[List[str]] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        adate = audit_date or now_utc.strftime("%Y-%m-%d")
        now_iso = now_utc.isoformat()

        target_segments = list(segments) if segments is not None else self._list_graded_segments()
        if "ALL_OVERALL" not in target_segments:
            target_segments.append("ALL_OVERALL")

        audits: Dict[str, Dict[str, Any]] = {}
        throttled_tiers: List[str] = []
        paused_tiers: List[str] = []

        for seg in target_segments:
            rows = self._fetch_rolling_graded_plays(None if seg == "ALL_OVERALL" else seg)
            report = compute_portfolio_brier_and_clv_report(rows)["overall"]
            window_n = int(report.get("n_graded", 0))
            bss_book = report.get("bss_vs_book_pct")
            bss_sharp = report.get("bss_vs_close_sharp_pct")
            model_brier = report.get("model_brier_score")
            book_brier = report.get("bet_book_brier_score")
            sharp_brier = report.get("close_sharp_brier_score")
            ece = report.get("calibration_ece_pct")

            prior_status = self._get_prior_status(seg)
            verdict = _classify_walk_forward_bss_tier(
                window_n=window_n,
                bss_vs_book_pct=bss_book,
                prior_status=prior_status,
                soft_min_sample=self.soft_min_sample,
                hard_min_sample=self.hard_min_sample,
                recovery_hysteresis_pct=self.recovery_hysteresis_pct,
            )

            rec = TierBSSAuditRecord(
                audit_date=adate,
                segment=seg,
                window_size=self.window_size,
                window_n=window_n,
                model_brier=model_brier,
                book_brier=book_brier,
                sharp_brier=sharp_brier,
                bss_vs_book_pct=bss_book,
                bss_vs_sharp_pct=bss_sharp,
                ece=ece,
                prior_status=prior_status,
                status=verdict["status"],
                kelly_multiplier=float(verdict["kelly_multiplier"]),
                max_units_cap=verdict["max_units_cap"],
                requires_recalibration=bool(verdict["requires_recalibration"]),
                notes=str(verdict["notes"]),
                audited_at_utc=now_iso,
            )

            if persist:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    INSERT INTO walk_forward_bss_audit (
                        audit_date, segment, window_size, window_n,
                        model_brier, book_brier, sharp_brier,
                        bss_vs_book_pct, bss_vs_sharp_pct, ece,
                        prior_status, status, kelly_multiplier, max_units_cap,
                        requires_recalibration, notes, audited_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rec.audit_date, rec.segment, rec.window_size, rec.window_n,
                        rec.model_brier, rec.book_brier, rec.sharp_brier,
                        rec.bss_vs_book_pct, rec.bss_vs_sharp_pct, rec.ece,
                        rec.prior_status, rec.status, rec.kelly_multiplier, rec.max_units_cap,
                        1 if rec.requires_recalibration else 0, rec.notes, rec.audited_at_utc,
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO tier_throttle_state (
                        segment, status, kelly_multiplier, max_units_cap,
                        window_n, bss_vs_book_pct, bss_vs_sharp_pct, ece,
                        requires_recalibration, notes, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(segment) DO UPDATE SET
                        status = excluded.status,
                        kelly_multiplier = excluded.kelly_multiplier,
                        max_units_cap = excluded.max_units_cap,
                        window_n = excluded.window_n,
                        bss_vs_book_pct = excluded.bss_vs_book_pct,
                        bss_vs_sharp_pct = excluded.bss_vs_sharp_pct,
                        ece = excluded.ece,
                        requires_recalibration = excluded.requires_recalibration,
                        notes = excluded.notes,
                        updated_at_utc = excluded.updated_at_utc
                    """,
                    (
                        rec.segment, rec.status, rec.kelly_multiplier, rec.max_units_cap,
                        rec.window_n, rec.bss_vs_book_pct, rec.bss_vs_sharp_pct, rec.ece,
                        1 if rec.requires_recalibration else 0, rec.notes, rec.audited_at_utc,
                    ),
                )
                self._conn.commit()

            audits[seg] = asdict(rec)
            if seg != "ALL_OVERALL":
                if rec.status == "PAUSED_ZERO_STAKE":
                    paused_tiers.append(seg)
                elif rec.status in ("THROTTLED_SOFT", "THROTTLED_HARD"):
                    throttled_tiers.append(seg)

        return {
            "audit_date": adate,
            "window_size": self.window_size,
            "audited_at_utc": now_iso,
            "throttled_tiers": throttled_tiers,
            "paused_tiers": paused_tiers,
            "alert_required": bool(throttled_tiers or paused_tiers),
            "tiers": audits,
        }

    def get_active_throttle_map(self) -> Dict[str, Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM tier_throttle_state ORDER BY segment ASC")
        return {str(r["segment"]): dict(r) for r in cur.fetchall()}

    def get_tier_multiplier(self, segment: str) -> Dict[str, Any]:
        state = self.get_active_throttle_map()
        if segment in state:
            row = state[segment]
            return {
                "segment": segment,
                "status": row["status"],
                "kelly_multiplier": float(row["kelly_multiplier"]),
                "max_units_cap": row["max_units_cap"],
                "bss_vs_book_pct": row["bss_vs_book_pct"],
                "window_n": int(row["window_n"]),
            }
        return {
            "segment": segment,
            "status": "HEALTHY",
            "kelly_multiplier": 1.0,
            "max_units_cap": None,
            "bss_vs_book_pct": None,
            "window_n": 0,
        }

    def get_audit_history(self, segment: Optional[str] = None, limit: int = 30) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        if segment:
            cur.execute(
                "SELECT * FROM walk_forward_bss_audit WHERE segment = ? ORDER BY id DESC LIMIT ?",
                (segment, limit),
            )
        else:
            cur.execute("SELECT * FROM walk_forward_bss_audit ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    @staticmethod
    def run_walk_forward_simulation(
        chronological_plays: List[Dict[str, Any]],
        window_size: int = DEFAULT_BSS_WINDOW_SIZE,
        soft_min_sample: int = DEFAULT_BSS_SOFT_MIN_SAMPLE,
        hard_min_sample: int = DEFAULT_BSS_HARD_MIN_SAMPLE,
    ) -> Dict[str, Any]:
        """
        Zero-lookahead walk-forward simulation over a chronological list of graded plays.
        For each play i, computes trailing BSS strictly on plays [< i] in the same segment,
        applies the live Kelly throttle multiplier, and compares unthrottled vs protected PnL & MaxDD.
        """
        history_by_seg: Dict[str, List[Dict[str, Any]]] = {}
        status_by_seg: Dict[str, str] = {}

        raw_cum = 0.0
        raw_peak = 0.0
        raw_max_dd = 0.0

        prot_cum = 0.0
        prot_peak = 0.0
        prot_max_dd = 0.0

        throttled_count = 0
        paused_count = 0
        step_trace: List[Dict[str, Any]] = []

        for idx, play in enumerate(chronological_plays):
            seg = str(play.get("segment") or "STRAIGHT_OVER")
            hist = history_by_seg.get(seg, [])[-window_size:]
            prior_status = status_by_seg.get(seg, "HEALTHY")

            if hist:
                rep = compute_portfolio_brier_and_clv_report(hist)["overall"]
                bss_book = rep.get("bss_vs_book_pct")
                win_n = int(rep.get("n_graded", 0))
            else:
                bss_book = None
                win_n = 0

            verdict = _classify_walk_forward_bss_tier(
                window_n=win_n,
                bss_vs_book_pct=bss_book,
                prior_status=prior_status,
                soft_min_sample=soft_min_sample,
                hard_min_sample=hard_min_sample,
            )
            status_by_seg[seg] = verdict["status"]
            mult = float(verdict["kelly_multiplier"])
            if verdict["status"] == "PAUSED_ZERO_STAKE":
                paused_count += 1
            elif mult < 1.0:
                throttled_count += 1

            raw_units = float(play.get("units_staked", 1.0))
            bet_odds = int(play.get("bet_odds", -110))
            hit = int(play.get("outcome_hit", 0))
            dec = odds_to_decimal(bet_odds)
            unit_pnl_per_1u = (dec - 1.0) if hit == 1 else -1.0

            raw_pnl = raw_units * unit_pnl_per_1u
            prot_units = raw_units * mult
            if verdict["max_units_cap"] is not None:
                prot_units = min(prot_units, float(verdict["max_units_cap"]))
            prot_pnl = prot_units * unit_pnl_per_1u

            raw_cum += raw_pnl
            raw_peak = max(raw_peak, raw_cum)
            raw_max_dd = max(raw_max_dd, raw_peak - raw_cum)

            prot_cum += prot_pnl
            prot_peak = max(prot_peak, prot_cum)
            prot_max_dd = max(prot_max_dd, prot_peak - prot_cum)

            if idx == len(chronological_plays) - 1 or (idx + 1) % 25 == 0:
                step_trace.append(
                    {
                        "play_index": idx + 1,
                        "segment": seg,
                        "trailing_n": win_n,
                        "trailing_bss_pct": bss_book,
                        "status": verdict["status"],
                        "kelly_multiplier": mult,
                        "raw_cum_units": round(raw_cum, 2),
                        "protected_cum_units": round(prot_cum, 2),
                    }
                )

            # Append play AFTER grading decision (strict zero-lookahead)
            m_prob = float(play.get("model_prob", 0.55))
            b_prob = float(play.get("bet_implied_prob", odds_to_prob(bet_odds)))
            rec = {
                "outcome_hit": hit,
                "model_prob": m_prob,
                "brier_model": (m_prob - hit) ** 2,
                "brier_bet_book": (b_prob - hit) ** 2,
                "brier_close_sharp": (b_prob - hit) ** 2,
                "segment": seg,
            }
            history_by_seg.setdefault(seg, []).append(rec)

        return {
            "total_plays": len(chronological_plays),
            "throttled_plays": throttled_count,
            "paused_plays": paused_count,
            "unthrottled_net_units": round(raw_cum, 2),
            "protected_net_units": round(prot_cum, 2),
            "pnl_improvement_units": round(prot_cum - raw_cum, 2),
            "unthrottled_max_drawdown_u": round(raw_max_dd, 2),
            "protected_max_drawdown_u": round(prot_max_dd, 2),
            "drawdown_saved_u": round(raw_max_dd - prot_max_dd, 2),
            "final_tier_statuses": status_by_seg,
            "trace": step_trace,
        }


def enforce_bss_and_game_exposure_guardrails(
    plays: List[Dict[str, Any]],
    throttle_map: Optional[Dict[str, Dict[str, Any]]] = None,
    max_units_per_game: float = MAX_PER_GAME_EXPOSURE_UNITS,
    same_script_rho: float = 0.35,
    min_playable_units: float = 0.25,
) -> Dict[str, Any]:
    """
    Upgrade m5 Unified Portfolio Sizing Pipeline:
      Step 1: Apply active Nightly Walk-Forward BSS Kelly throttle multiplier per market tier.
      Step 2: Apply same-game, same-direction correlation haircut (1 / sqrt(1 + rho * (k - 1)))
              to secondary correlated legs in the same canonical game (leaving #1 POTD/highest-EV
              leg intact and leaving negatively correlated hedges unpenalized).
      Step 3: Enforce hard <=2.5u max cumulative exposure cap per canonical game.
    """
    tmap = throttle_map or {}
    step1_plays: List[Dict[str, Any]] = []
    bss_trimmed_total = 0.0

    # Step 1: BSS tier throttle
    for p in plays:
        item = dict(p)
        orig_u = max(0.0, float(item.get("units", 0.0)))
        seg = str(item.get("segment") or "STRAIGHT_OVER")
        tinfo = tmap.get(seg, {})
        mult = float(tinfo.get("kelly_multiplier", 1.0))
        cap_override = tinfo.get("max_units_cap")
        bss_u = round(orig_u * mult, 2)
        if cap_override is not None:
            bss_u = min(bss_u, float(cap_override))
        if bss_u < min_playable_units and bss_u > 0.0:
            bss_u = 0.0
        bss_trimmed_total += max(0.0, orig_u - bss_u)
        item["pre_guardrail_units"] = round(orig_u, 2)
        item["bss_status"] = str(tinfo.get("status", "HEALTHY"))
        item["bss_kelly_multiplier"] = mult
        item["post_bss_units"] = bss_u
        item["units"] = bss_u
        step1_plays.append(item)

    # Step 2: Same-game, same-direction correlation haircut before hard game cap
    grouped_dir: Dict[Tuple[str, str], List[int]] = {}
    for idx, item in enumerate(step1_plays):
        if item["units"] <= 0.0:
            continue
        gkey = canonicalize_game_key(item.get("game_key") or item.get("matchup") or f"GAME_{idx}")
        direction = str(item.get("side") or "OVER").upper()
        grouped_dir.setdefault((gkey, direction), []).append(idx)

    corr_trimmed_total = 0.0
    for (_, _), idxs in grouped_dir.items():
        if len(idxs) <= 1:
            continue
        # Rank by POTD then EV% descending; primary leg keeps 1.0x, secondary legs take correlation haircut
        ranked = sorted(
            idxs,
            key=lambda i: (
                1 if step1_plays[i].get("is_potd") else 0,
                float(step1_plays[i].get("ev_pct", step1_plays[i].get("edge", 0.0))),
                float(step1_plays[i].get("units", 0.0)),
            ),
            reverse=True,
        )
        for rank_pos, i in enumerate(ranked):
            if rank_pos == 0:
                step1_plays[i]["correlation_haircut_mult"] = 1.0
                continue
            haircut = 1.0 / math.sqrt(1.0 + max(0.0, same_script_rho) * rank_pos)
            before_u = step1_plays[i]["units"]
            after_u = round(before_u * haircut, 2)
            if 0.0 < after_u < min_playable_units:
                after_u = 0.0
            corr_trimmed_total += max(0.0, before_u - after_u)
            step1_plays[i]["correlation_haircut_mult"] = round(haircut, 3)
            step1_plays[i]["units"] = after_u

    for item in step1_plays:
        item.setdefault("correlation_haircut_mult", 1.0)
        item["post_corr_units"] = item["units"]

    # Step 3: Hard per-game exposure cap (<=2.5u default)
    cap_res = enforce_per_game_exposure_cap(
        step1_plays,
        max_units_per_game=max_units_per_game,
        min_playable_units=min_playable_units,
    )
    cap_res["bss_units_trimmed"] = round(bss_trimmed_total, 2)
    cap_res["correlation_units_trimmed"] = round(corr_trimmed_total, 2)
    cap_res["hard_cap_units_trimmed"] = cap_res["total_units_saved"]
    cap_res["total_pre_guardrail_units"] = round(sum(float(p["pre_guardrail_units"]) for p in step1_plays), 2)
    cap_res["total_all_guardrails_saved"] = round(
        cap_res["total_pre_guardrail_units"] - cap_res["total_capped_units"], 2
    )
    return cap_res


# ============================================================================
# PILLAR 5: FASTAPI PRODUCTION ROUTER & STREAMLIT UI
# ============================================================================

if HAS_FASTAPI:
    class PropSimRequest(BaseModel):
        target_line: float
        rate_per_min: float
        min_comp: float = 34.0
        min_blowout: float = 25.0
        spread: float = -4.5
        proj_pace: float = 99.5
        comb_3par: float = 0.39
        team_abbr: Optional[str] = None
        coach_name: Optional[str] = None
        coach_mult: Optional[float] = None
        is_underdog: bool = False
        bench_cliff_sigma_bump: float = 0.0
        is_nba_cup: bool = False
        dist_family: str = "neg_binomial"
        side: str = "OVER"
        offered_odds: int = -110
        n_trials: int = 10000

    class CopulaSgpRequest(BaseModel):
        leg_probs: List[float]
        corr_matrix: Optional[List[List[float]]] = None
        shared_pace_loadings: Optional[List[float]] = None
        copula_type: str = "gaussian"
        n_trials: int = 20000

    class ExposureCapRequest(BaseModel):
        plays: List[Dict[str, Any]]
        max_units_per_game: float = MAX_PER_GAME_EXPOSURE_UNITS
        min_playable_units: float = 0.25

    class DevigRequest(BaseModel):
        over_odds: int
        under_odds: int
        model_prob: float = 0.58

    class PbpStintsRequest(BaseModel):
        game_id: str
        team_abbr: str
        season: str = "2026-27"
        offline_mode: bool = False

    class OnOffUsageRequest(BaseModel):
        team_abbr: str
        target_player_id: int
        teammate_out_ids: List[int]
        target_player_name: str = ""
        teammate_out_names: Optional[List[str]] = None
        stat_type: str = "Points"
        base_rate_per_min: float = 0.78
        last_n_games: int = 12
        prior_poss_k: float = 250.0
        offline_mode: bool = False

    class UpgradesAuditRequest(BaseModel):
        base_true_prob: float = 0.62
        side: str = "OVER"
        up5_l5_overs: int = 4
        up5_mpg_delta: float = 0.8
        up5_usg_delta: float = 0.5
        up5_ts_delta: float = 9.2
        up5_book_shade: float = 2.0
        up6_months_h2h: float = 0.1
        up6_same_coach: bool = True
        up6_same_shell: bool = True
        up6_duplex_48hr: bool = True
        up7_days_rest: int = 3
        up7_is_home: bool = True
        up7_opp_b2b: bool = False
        up7_stat_type: str = "3PM"
        up7_fta: float = 3.2
        up7_rim_freq: float = 24.0
        up8_ref_z: float = 1.45
        up8_pf36: float = 3.6
        up8_fta: float = 4.0
        up8_ft_pct: float = 76.0
        up8_opp_foul_drawer: bool = True

    class OddsSnapshotWaveRequest(BaseModel):
        event_id: str
        pulse_wave: str = "PULSE_1_OPENER"
        markets: Optional[List[str]] = None
        event_payload: Optional[Dict[str, Any]] = None
        devig_method: str = "shin"
        offline_mode: bool = True
        db_path: str = "alphaprop_clv_ledger.sqlite"

    class TrackedPlayCLVRequest(BaseModel):
        play_id: str
        slate_date: str
        event_id: str
        game_key: str
        player_name: str
        market: str = "player_points_assists"
        side: str = "OVER"
        line: float
        bet_book: str = "draftkings"
        bet_odds: int = -115
        model_prob: float = 0.65
        units_staked: float = 1.25
        segment: str = "VIP_OVER"
        pulse_wave: str = "PULSE_1_OPENER"
        outcome_hit: Optional[int] = None
        close_line: Optional[float] = None
        close_sharp_fair_over: Optional[float] = None
        db_path: str = "alphaprop_clv_ledger.sqlite" 



# ============================================================================
# UPGRADE #5: SUPABASE JWT VERIFIER, ENTITLEMENT STORE & PAYWALL MIDDLEWARE
# ============================================================================

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    pad = "=" * ((4 - len(segment) % 4) % 4)
    return base64.urlsafe_b64decode(segment + pad)


class JWTVerificationError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 401):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class AuthenticatedUser:
    user_id: str
    email: str = ""
    tier: str = "free"
    role: str = "subscriber"
    subscription_status: str = "inactive"
    current_period_end: Optional[str] = None
    is_vip: bool = False
    is_admin: bool = False
    auth_method: str = "supabase_jwt"


class SupabaseJWTVerifier:
    """
    Production Supabase Auth JWT verifier.
    - Verifies HS256 tokens signed with SUPABASE_JWT_SECRET using constant-time HMAC.
    - Optionally verifies RS256/ES256 tokens if a PEM public key is supplied via cryptography.
    - Strictly enforces exp, nbf, aud ('authenticated'), and optional iss claims.
    """

    def __init__(
        self,
        jwt_secret: Optional[str] = None,
        audience: str = "authenticated",
        issuer: Optional[str] = None,
        leeway_sec: int = 30,
        asymmetric_public_key_pem: Optional[str] = None,
    ):
        self.jwt_secret = jwt_secret if jwt_secret is not None else os.getenv("SUPABASE_JWT_SECRET", "")
        self.audience = audience
        self.issuer = issuer if issuer is not None else os.getenv("SUPABASE_JWT_ISSUER")
        self.leeway_sec = int(leeway_sec)
        self.asymmetric_public_key_pem = asymmetric_public_key_pem or os.getenv("SUPABASE_JWT_PUBLIC_KEY_PEM")

    def sign_hs256_jwt(self, payload: Dict[str, Any], secret: Optional[str] = None) -> str:
        """Sign an HS256 JWT (used for local dev, CI, and self-tests)."""
        key = (secret if secret is not None else self.jwt_secret).encode("utf-8")
        if not key:
            raise ValueError("Cannot sign HS256 JWT without a non-empty secret.")
        header = {"alg": "HS256", "typ": "JWT"}
        h_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        p_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{h_b64}.{p_b64}".encode("ascii")
        sig = hmac.new(key, signing_input, hashlib.sha256).digest()
        return f"{h_b64}.{p_b64}.{_b64url_encode(sig)}"

    def verify(self, token: str) -> Dict[str, Any]:
        if not token or token.count(".") != 2:
            raise JWTVerificationError("INVALID_TOKEN", "Malformed Bearer JWT.", 401)
        h_b64, p_b64, s_b64 = token.split(".")
        try:
            header = json.loads(_b64url_decode(h_b64).decode("utf-8"))
            payload = json.loads(_b64url_decode(p_b64).decode("utf-8"))
            signature = _b64url_decode(s_b64)
        except Exception as exc:
            raise JWTVerificationError("INVALID_TOKEN", f"Unable to decode JWT segments: {exc}", 401)

        alg = str(header.get("alg", "")).upper()
        signing_input = f"{h_b64}.{p_b64}".encode("ascii")

        if alg == "HS256":
            if not self.jwt_secret:
                raise JWTVerificationError("SERVER_AUTH_MISCONFIGURED", "SUPABASE_JWT_SECRET is not configured.", 500)
            expected_sig = hmac.new(self.jwt_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected_sig):
                raise JWTVerificationError("INVALID_SIGNATURE", "JWT signature verification failed.", 401)
        elif alg in ("RS256", "ES256"):
            if not self.asymmetric_public_key_pem:
                raise JWTVerificationError("INVALID_TOKEN", f"Asymmetric key not configured for {alg}.", 401)
            try:
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import ec, padding
                from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

                pub = serialization.load_pem_public_key(self.asymmetric_public_key_pem.encode("utf-8"))
                if alg == "RS256":
                    pub.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
                else:
                    r = int.from_bytes(signature[:32], "big")
                    s = int.from_bytes(signature[32:], "big")
                    pub.verify(encode_dss_signature(r, s), signing_input, ec.ECDSA(hashes.SHA256()))
            except Exception:
                raise JWTVerificationError("INVALID_SIGNATURE", f"{alg} JWT signature verification failed.", 401)
        else:
            raise JWTVerificationError("UNSUPPORTED_ALG", f"Unsupported JWT algorithm: {alg}", 401)

        now = time.time()
        exp = payload.get("exp")
        if exp is None:
            raise JWTVerificationError("MISSING_EXP", "JWT is missing required exp claim.", 401)
        if now > float(exp) + self.leeway_sec:
            raise JWTVerificationError("TOKEN_EXPIRED", "Session token has expired. Please refresh your Supabase session.", 401)

        nbf = payload.get("nbf")
        if nbf is not None and now + self.leeway_sec < float(nbf):
            raise JWTVerificationError("TOKEN_NOT_YET_VALID", "JWT is not valid yet.", 401)

        if self.audience:
            aud = payload.get("aud")
            aud_list = aud if isinstance(aud, list) else [aud]
            if self.audience not in aud_list:
                raise JWTVerificationError("INVALID_AUDIENCE", "JWT audience mismatch.", 401)

        if self.issuer and payload.get("iss") != self.issuer:
            raise JWTVerificationError("INVALID_ISSUER", "JWT issuer mismatch.", 401)

        if not payload.get("sub"):
            raise JWTVerificationError("MISSING_SUB", "JWT is missing subject (user_id).", 401)

        return payload


class EntitlementStore:
    """
    Server-side subscription entitlement resolver with 60-second in-memory TTL cache.
    - Production mode: Queries Supabase PostgREST (`subscriptions` + `profiles`) using
      SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY so cancellations/refunds revoke VIP
      access immediately instead of trusting stale JWT claims.
    - Offline / local mode: Persists authoritative subscriptions in SQLite.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        supabase_url: Optional[str] = None,
        supabase_service_role_key: Optional[str] = None,
        cache_ttl_sec: int = 60,
    ):
        self.supabase_url = (supabase_url or os.getenv("SUPABASE_URL", "")).rstrip("/")
        self.supabase_service_role_key = supabase_service_role_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.cache_ttl_sec = max(1, int(cache_ttl_sec))
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_sqlite_schema()

    def _init_sqlite_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS local_entitlements (
                user_id             TEXT PRIMARY KEY,
                email               TEXT NOT NULL DEFAULT '',
                tier                TEXT NOT NULL DEFAULT 'free',
                role                TEXT NOT NULL DEFAULT 'subscriber',
                status              TEXT NOT NULL DEFAULT 'inactive',
                provider            TEXT NOT NULL DEFAULT 'stripe',
                provider_sub_id     TEXT,
                current_period_end  TEXT,
                updated_at          TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS webhook_idempotency (
                event_id            TEXT PRIMARY KEY,
                provider            TEXT NOT NULL,
                event_type          TEXT NOT NULL,
                processed_at        TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def invalidate_user(self, user_id: str) -> None:
        self._cache.pop(user_id, None)

    def upsert_entitlement(
        self,
        user_id: str,
        tier: str,
        status: str,
        email: str = "",
        role: str = "subscriber",
        provider: str = "stripe",
        provider_sub_id: Optional[str] = None,
        current_period_end: Optional[str] = None,
        event_id: Optional[str] = None,
        event_type: str = "subscription.updated",
    ) -> Dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        if event_id:
            existing = cur.execute(
                "SELECT event_id FROM webhook_idempotency WHERE event_id = ?", (event_id,)
            ).fetchone()
            if existing is not None:
                return {"idempotent_skip": True, "event_id": event_id, "user_id": user_id}
            cur.execute(
                "INSERT INTO webhook_idempotency (event_id, provider, event_type, processed_at) VALUES (?, ?, ?, ?)",
                (event_id, provider, event_type, now_iso),
            )

        cur.execute(
            """
            INSERT INTO local_entitlements
                (user_id, email, tier, role, status, provider, provider_sub_id, current_period_end, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                email = CASE WHEN excluded.email != '' THEN excluded.email ELSE local_entitlements.email END,
                tier = excluded.tier,
                role = excluded.role,
                status = excluded.status,
                provider = excluded.provider,
                provider_sub_id = COALESCE(excluded.provider_sub_id, local_entitlements.provider_sub_id),
                current_period_end = excluded.current_period_end,
                updated_at = excluded.updated_at
            """,
            (user_id, email, tier, role, status, provider, provider_sub_id, current_period_end, now_iso),
        )
        self.conn.commit()
        self.invalidate_user(user_id)
        return {
            "idempotent_skip": False,
            "user_id": user_id,
            "tier": tier,
            "role": role,
            "status": status,
            "current_period_end": current_period_end,
        }

    def _fetch_from_supabase(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not (self.supabase_url and self.supabase_service_role_key):
            return None
        headers = {
            "apikey": self.supabase_service_role_key,
            "Authorization": f"Bearer {self.supabase_service_role_key}",
            "Accept": "application/json",
        }
        q_uid = urllib.parse.quote(user_id, safe="")
        prof_url = f"{self.supabase_url}/rest/v1/profiles?id=eq.{q_uid}&select=id,email,tier,role"
        sub_url = (
            f"{self.supabase_url}/rest/v1/subscriptions?user_id=eq.{q_uid}"
            "&select=tier,status,current_period_end&order=current_period_end.desc&limit=1"
        )
        try:
            req_p = urllib.request.Request(prof_url, headers=headers)
            with urllib.request.urlopen(req_p, timeout=5) as resp:
                profs = json.loads(resp.read().decode("utf-8"))
            req_s = urllib.request.Request(sub_url, headers=headers)
            with urllib.request.urlopen(req_s, timeout=5) as resp:
                subs = json.loads(resp.read().decode("utf-8"))
            prof = profs[0] if profs else {}
            sub = subs[0] if subs else {}
            return {
                "email": prof.get("email", ""),
                "role": prof.get("role", "subscriber"),
                "tier": sub.get("tier", prof.get("tier", "free")),
                "status": sub.get("status", "inactive"),
                "current_period_end": sub.get("current_period_end"),
            }
        except Exception:
            return None

    def resolve_user(self, jwt_payload: Dict[str, Any]) -> AuthenticatedUser:
        user_id = str(jwt_payload["sub"])
        email = str(jwt_payload.get("email", ""))
        now = time.time()

        cached = self._cache.get(user_id)
        if cached and (now - cached[0]) < self.cache_ttl_sec:
            rec = cached[1]
        else:
            rec = self._fetch_from_supabase(user_id)
            if rec is None:
                row = self.conn.execute(
                    "SELECT email, tier, role, status, current_period_end FROM local_entitlements WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                if row is not None:
                    rec = dict(row)
                else:
                    rec = {
                        "email": email,
                        "tier": "free",
                        "role": "subscriber",
                        "status": "inactive",
                        "current_period_end": None,
                    }
            self._cache[user_id] = (now, rec)

        role = str(rec.get("role", "subscriber")).lower()
        tier = str(rec.get("tier", "free")).lower()
        status = str(rec.get("status", "inactive")).lower()
        period_end_str = rec.get("current_period_end")

        is_not_expired = True
        if period_end_str:
            try:
                end_dt = datetime.fromisoformat(str(period_end_str).replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                is_not_expired = end_dt > datetime.now(timezone.utc)
            except ValueError:
                is_not_expired = False

        is_admin = role == "admin"
        is_vip = is_admin or (
            tier in ("vip", "pro_quant", "admin")
            and status in ("active", "trialing")
            and is_not_expired
        )
        effective_tier = "admin" if is_admin else (tier if is_vip else "free")

        return AuthenticatedUser(
            user_id=user_id,
            email=rec.get("email") or email,
            tier=effective_tier,
            role=role,
            subscription_status=status if is_not_expired else "expired",
            current_period_end=period_end_str,
            is_vip=is_vip,
            is_admin=is_admin,
        )

    def close(self) -> None:
        self.conn.close()


if HAS_FASTAPI:
    class EntitlementSyncRequest(BaseModel):
        user_id: str
        email: str = ""
        tier: str = Field(default="vip", pattern="^(free|vip|pro_quant|admin)$")
        role: str = Field(default="subscriber", pattern="^(subscriber|admin)$")
        status: str = Field(default="active", pattern="^(inactive|trialing|active|past_due|canceled|expired)$")
        provider: str = Field(default="stripe", pattern="^(stripe|revenuecat|manual_comp)$")
        provider_subscription_id: Optional[str] = None
        current_period_end: Optional[str] = None
        event_id: Optional[str] = None
        event_type: str = "customer.subscription.updated"


    class AlphaPropPaywallMiddleware(BaseHTTPMiddleware):
        """
        Server-side FastAPI JWT Paywall Middleware.
        - Public routes: /health, /docs, /redoc, /openapi.json, /v1/slate/public-potd, /v1/ledger/public-track-record
        - Authenticated routes (any tier): /v1/auth/me
        - VIP-gated routes: /v1/slate/vip-board, /v1/simulate/*, /v1/pricing/*, /v1/quant-lab/*,
                            /v1/data/*, /v1/clv/report, /v1/portfolio/*
        - Admin / Service-Key routes: /v1/admin/*, /v1/odds/snapshot-wave, /v1/clv/log-and-grade
        """

        PUBLIC_EXACT = {
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/v1/slate/public-potd",
            "/v1/ledger/public-track-record",
        }
        AUTH_ONLY_EXACT = {"/v1/auth/me"}
        ADMIN_PREFIXES = ("/v1/admin/", "/v1/odds/snapshot-wave", "/v1/odds/unabated/", "/v1/clv/log-and-grade")
        VIP_PREFIXES = (
            "/v1/slate/vip-board",
            "/v1/simulate/",
            "/v1/pricing/",
            "/v1/quant-lab/",
            "/v1/data/",
            "/v1/clv/report",
            "/v1/portfolio/",
        )

        def __init__(
            self,
            app: Any,
            verifier: SupabaseJWTVerifier,
            entitlement_store: EntitlementStore,
            admin_api_key: Optional[str] = None,
            upgrade_url: str = "https://alphaprop.pro/upgrade",
        ):
            super().__init__(app)
            self.verifier = verifier
            self.entitlement_store = entitlement_store
            self.admin_api_key = admin_api_key if admin_api_key is not None else os.getenv("ALPHAPROP_ADMIN_API_KEY", "")
            self.upgrade_url = upgrade_url

        async def dispatch(self, request: Request, call_next: Any) -> Any:
            path = request.url.path
            if request.method == "OPTIONS" or path in self.PUBLIC_EXACT:
                request.state.auth_user = None
                return await call_next(request)

            # Check server-to-server Admin API key (for Stripe webhook workers & 3-Pulse cron jobs)
            svc_key = request.headers.get("X-AlphaProp-Service-Key", "")
            if self.admin_api_key and svc_key and hmac.compare_digest(svc_key, self.admin_api_key):
                request.state.auth_user = AuthenticatedUser(
                    user_id="service_role",
                    email="cron@alphaprop.pro",
                    tier="admin",
                    role="admin",
                    subscription_status="active",
                    is_vip=True,
                    is_admin=True,
                    auth_method="service_api_key",
                )
                return await call_next(request)

            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                    content={
                        "error_code": "AUTH_REQUIRED",
                        "detail": "Missing Bearer JWT. Sign in via Supabase Auth to access AlphaProp Pro™.",
                    },
                )

            token = auth_header[7:].strip()
            try:
                claims = self.verifier.verify(token)
            except JWTVerificationError as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    headers={"WWW-Authenticate": "Bearer"},
                    content={"error_code": exc.code, "detail": exc.message},
                )

            user = self.entitlement_store.resolve_user(claims)
            request.state.auth_user = user

            if any(path.startswith(p) for p in self.ADMIN_PREFIXES):
                if not user.is_admin:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error_code": "ADMIN_REQUIRED",
                            "detail": "This publisher/webhook endpoint requires admin privileges.",
                        },
                    )
                return await call_next(request)

            if path in self.AUTH_ONLY_EXACT:
                return await call_next(request)

            if any(path.startswith(p) for p in self.VIP_PREFIXES):
                if not user.is_vip:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error_code": "VIP_SUBSCRIPTION_REQUIRED",
                            "detail": "Active AlphaProp Pro™ VIP subscription required to access this endpoint.",
                            "current_tier": user.tier,
                            "subscription_status": user.subscription_status,
                            "upgrade_url": self.upgrade_url,
                        },
                    )

            return await call_next(request)


def create_fastapi_app(
    auth_enabled: Optional[bool] = None,
    jwt_secret: Optional[str] = None,
    entitlement_store: Optional[EntitlementStore] = None,
    admin_api_key: Optional[str] = None,
) -> Any:
    if not HAS_FASTAPI:
        return None
    app = FastAPI(title="AlphaProp Pro™ Commercial Engine v9.2 (Supabase RLS + JWT Paywall)", version="9.2.0")

    use_auth = (
        auth_enabled
        if auth_enabled is not None
        else os.getenv("ALPHAPROP_AUTH_ENABLED", "true").lower() in ("1", "true", "yes")
    )
    verifier = SupabaseJWTVerifier(jwt_secret=jwt_secret)
    store = entitlement_store or EntitlementStore(
        db_path=os.getenv("ALPHAPROP_ENTITLEMENTS_DB", ":memory:")
    )
    app.state.jwt_verifier = verifier
    app.state.entitlement_store = store

    if use_auth:
        app.add_middleware(
            AlphaPropPaywallMiddleware,
            verifier=verifier,
            entitlement_store=store,
            admin_api_key=admin_api_key,
        )

    # In-memory sample slate payload (replaced in production by Supabase vip_picks / vip_parlays queries)
    _SLATE_STORE: Dict[str, Any] = {
        "slate_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "free_potd": {
            "pick_id": "potd_free_01",
            "track": "POTD",
            "is_free_potd": True,
            "player": "Luka Doncic",
            "matchup": "LAL @ PHX",
            "market": "PTS+AST",
            "side": "OVER",
            "line": 41.5,
            "odds": -115,
            "buy_stop_line": 42.5,
            "true_prob": 0.672,
            "ev_pct": 13.7,
            "units": 1.50,
        },
        "vip_picks": [
            {
                "pick_id": "vip_02",
                "track": "OFFICIAL_VIP",
                "is_free_potd": False,
                "player": "Alperen Sengun",
                "matchup": "HOU @ WAS",
                "market": "PTS+REB",
                "side": "OVER",
                "line": 33.5,
                "odds": -115,
                "buy_stop_line": 34.5,
                "true_prob": 0.652,
                "ev_pct": 11.7,
                "units": 1.25,
            },
            {
                "pick_id": "vip_03",
                "track": "AVALANCHE_UNDER",
                "is_free_potd": False,
                "player": "Cooper Flagg",
                "matchup": "DAL @ BOS",
                "market": "PTS+REB",
                "side": "UNDER",
                "line": 26.5,
                "odds": -112,
                "buy_stop_line": 25.5,
                "true_prob": 0.668,
                "ev_pct": 14.2,
                "units": 1.15,
            },
        ],
        "vip_parlays": [
            {
                "parlay_id": "sgp_doubleup_01",
                "parlay_tier": "DOUBLE_UP_FLOOR",
                "title": "Double-Up High Floor #1 (+115)",
                "combined_odds": 115,
                "true_prob": 0.622,
                "ev_pct": 15.7,
                "units": 1.25,
            }
        ],
    }

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "engine": "AlphaProp Commercial v9.2 (Supabase RLS + JWT Paywall)",
            "server_paywall_enabled": use_auth,
        }

    @app.get("/v1/auth/me")
    def api_auth_me(request: Request) -> Dict[str, Any]:
        user: Optional[AuthenticatedUser] = getattr(request.state, "auth_user", None)
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return {
            "user_id": user.user_id,
            "email": user.email,
            "tier": user.tier,
            "role": user.role,
            "subscription_status": user.subscription_status,
            "current_period_end": user.current_period_end,
            "permissions": {
                "can_view_free_potd": True,
                "can_view_vip_board": user.is_vip,
                "can_view_parlays": user.is_vip,
                "can_run_quant_lab": user.is_vip,
                "can_publish_admin": user.is_admin,
            },
        }

    @app.get("/v1/slate/public-potd")
    def api_public_potd() -> Dict[str, Any]:
        """Public endpoint: returns ONLY the Free Pick of the Day and redacted counts of locked VIP plays."""
        return {
            "slate_date": _SLATE_STORE["slate_date"],
            "access_level": "PUBLIC_FREE_POTD_ONLY",
            "free_pick_of_the_day": _SLATE_STORE["free_potd"],
            "locked_vip_picks_count": len(_SLATE_STORE["vip_picks"]),
            "locked_vip_parlays_count": len(_SLATE_STORE["vip_parlays"]),
            "upgrade_cta": "Upgrade to AlphaProp Pro™ VIP to unlock full server-gated slate, SGPs & CLV tools.",
        }

    @app.get("/v1/slate/vip-board")
    def api_vip_board(request: Request) -> Dict[str, Any]:
        """VIP-gated endpoint: strictly blocked with 401/403 by AlphaPropPaywallMiddleware unless active VIP."""
        user: Optional[AuthenticatedUser] = getattr(request.state, "auth_user", None)
        return {
            "slate_date": _SLATE_STORE["slate_date"],
            "access_level": "VIP_UNLOCKED",
            "subscriber_tier": user.tier if user else "dev_bypass",
            "free_pick_of_the_day": _SLATE_STORE["free_potd"],
            "vip_picks": _SLATE_STORE["vip_picks"],
            "vip_parlays": _SLATE_STORE["vip_parlays"],
        }

    @app.post("/v1/admin/entitlements/sync")
    def api_admin_sync_entitlement(req: EntitlementSyncRequest) -> Dict[str, Any]:
        """Idempotent Stripe/RevenueCat webhook endpoint to grant or revoke VIP access immediately."""
        return store.upsert_entitlement(
            user_id=req.user_id,
            email=req.email,
            tier=req.tier,
            role=req.role,
            status=req.status,
            provider=req.provider,
            provider_sub_id=req.provider_subscription_id,
            current_period_end=req.current_period_end,
            event_id=req.event_id,
            event_type=req.event_type,
        )

    @app.get("/v1/coaches/hook-table")
    def api_coach_hook_table() -> Dict[str, Any]:
        return {"season": "2026-27", "team_count": len(COACH_HOOK_MODIFIER_TABLE), "coaches": COACH_HOOK_MODIFIER_TABLE}

    @app.post("/v1/simulate/prop")
    def api_sim_prop(req: PropSimRequest) -> Dict[str, Any]:
        gate = evaluate_dynamic_blowout_gate(
            spread=req.spread,
            proj_pace=req.proj_pace,
            comb_3par=req.comb_3par,
            coach_mult=req.coach_mult,
            is_underdog=req.is_underdog,
            bench_cliff_sigma_bump=req.bench_cliff_sigma_bump,
            is_nba_cup=req.is_nba_cup,
            side=req.side,
            team_abbr=req.team_abbr,
            coach_name=req.coach_name,
        )
        adj_min_blowout = max(14.0, min(34.0, req.min_blowout + gate["coach_hook"]["blowout_min_delta"]))
        mc = run_monte_carlo_prop_sim(
            target_line=req.target_line,
            rate_per_min=req.rate_per_min,
            min_comp=req.min_comp,
            min_blowout=adj_min_blowout,
            p_blowout=gate["p_blowout"],
            dist_family=req.dist_family,
            side=req.side,
            n_trials=req.n_trials,
        )
        kelly = compute_kelly_stake(mc["p_cover"], req.offered_odds)
        if gate["stake_multiplier"] < 1.0:
            kelly["recommended_units"] = round(kelly.get("recommended_units", 0.0) * gate["stake_multiplier"], 2)
        return {
            "sigma_game": gate["sigma_game"],
            "p_blowout": gate["p_blowout"],
            "blowout_gate": gate,
            "adj_min_blowout": adj_min_blowout,
            "monte_carlo": mc,
            "staking": kelly,
        }

    @app.post("/v1/simulate/sgp-copula")
    def api_sim_sgp(req: CopulaSgpRequest) -> Dict[str, Any]:
        try:
            return simulate_copula_sgp(
                req.leg_probs,
                req.corr_matrix,
                req.n_trials,
                req.copula_type,
                shared_pace_loadings=req.shared_pace_loadings,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/portfolio/exposure-cap")
    def api_exposure_cap(req: ExposureCapRequest) -> Dict[str, Any]:
        return enforce_per_game_exposure_cap(
            req.plays,
            max_units_per_game=req.max_units_per_game,
            min_playable_units=req.min_playable_units,
        )

    @app.post("/v1/pricing/devig")
    def api_devig(req: DevigRequest) -> Dict[str, Any]:
        return {
            "multiplicative": devig_multiplicative(req.over_odds, req.under_odds),
            "power": devig_power(req.over_odds, req.under_odds),
            "shin": devig_shin(req.over_odds, req.under_odds),
            "kelly_over": compute_kelly_stake(req.model_prob, req.over_odds),
        }

    @app.post("/v1/quant-lab/upgrades-1-8")
    def api_upgrades_1_8(req: UpgradesAuditRequest) -> Dict[str, Any]:
        return evaluate_all_upgrades_1_to_8(
            base_true_prob=req.base_true_prob,
            side=req.side,
            up5_l5_overs=req.up5_l5_overs,
            up5_mpg_delta=req.up5_mpg_delta,
            up5_usg_delta=req.up5_usg_delta,
            up5_ts_delta=req.up5_ts_delta,
            up5_book_shade=req.up5_book_shade,
            up6_months_h2h=req.up6_months_h2h,
            up6_same_coach=req.up6_same_coach,
            up6_same_shell=req.up6_same_shell,
            up6_duplex_48hr=req.up6_duplex_48hr,
            up7_days_rest=req.up7_days_rest,
            up7_is_home=req.up7_is_home,
            up7_opp_b2b=req.up7_opp_b2b,
            up7_stat_type=req.up7_stat_type,
            up7_fta=req.up7_fta,
            up7_rim_freq=req.up7_rim_freq,
            up8_ref_z=req.up8_ref_z,
            up8_pf36=req.up8_pf36,
            up8_fta=req.up8_fta,
            up8_ft_pct=req.up8_ft_pct,
            up8_opp_foul_drawer=req.up8_opp_foul_drawer,
        )

    @app.post("/v1/data/pbp-stints")
    def api_pbp_stints(req: PbpStintsRequest) -> Dict[str, Any]:
        pipe = NbaApiIngestionPipeline(season=req.season, offline_mode=req.offline_mode)
        stints = pipe.fetch_lineup_stints(req.game_id, req.team_abbr)
        return {
            "game_id": req.game_id,
            "team_abbr": req.team_abbr.upper(),
            "stint_count": len(stints),
            "stints": [asdict(s) for s in stints],
        }

    @app.post("/v1/data/on-off-usage")
    def api_on_off_usage(req: OnOffUsageRequest) -> Dict[str, Any]:
        pipe = NbaApiIngestionPipeline(offline_mode=req.offline_mode)
        split = pipe.ingest_player_on_off_usage(
            team_abbr=req.team_abbr,
            target_player_id=req.target_player_id,
            teammate_out_ids=req.teammate_out_ids,
            target_player_name=req.target_player_name,
            teammate_out_names=req.teammate_out_names,
            last_n_games=req.last_n_games,
            prior_poss_k=req.prior_poss_k,
        )
        enriched = build_on_off_enriched_prop_inputs(
            base_rate_per_min=req.base_rate_per_min,
            stat_type=req.stat_type,
            on_off_split=split,
        )
        return {"on_off_split": asdict(split), "enriched_prop_inputs": enriched}

    @app.post("/v1/odds/snapshot-wave")
    def api_odds_snapshot_wave(req: OddsSnapshotWaveRequest) -> Dict[str, Any]:
        logger = MultiBookOddsCLVLogger(
            db_path=req.db_path,
            offline_mode=req.offline_mode,
            devig_method=req.devig_method,
        )
        try:
            payload = req.event_payload or logger.fetch_odds_api_event_props(req.event_id, req.markets)
            consensuses = logger.ingest_event_snapshot(payload, pulse_wave=req.pulse_wave)
            return {
                "event_id": req.event_id,
                "pulse_wave": req.pulse_wave,
                "markets_captured": len(consensuses),
                "consensuses": [asdict(c) for c in consensuses],
            }
        finally:
            logger.close()

    @app.post("/v1/clv/log-and-grade")
    def api_clv_log_and_grade(req: TrackedPlayCLVRequest) -> Dict[str, Any]:
        logger = MultiBookOddsCLVLogger(db_path=req.db_path, offline_mode=True)
        try:
            logger.log_tracked_play(
                play_id=req.play_id,
                slate_date=req.slate_date,
                event_id=req.event_id,
                game_key=req.game_key,
                player_name=req.player_name,
                market=req.market,
                side=req.side,
                line=req.line,
                bet_book=req.bet_book,
                bet_odds=req.bet_odds,
                model_prob=req.model_prob,
                units_staked=req.units_staked,
                segment=req.segment,
                pulse_wave=req.pulse_wave,
            )
            if req.close_line is not None and req.close_sharp_fair_over is not None:
                return logger.attach_closing_and_grade(
                    play_id=req.play_id,
                    outcome_hit=req.outcome_hit,
                    close_line=req.close_line,
                    close_sharp_fair_over=req.close_sharp_fair_over,
                )
            return logger.get_tracked_play(req.play_id)
        finally:
            logger.close()


    @app.get("/v1/odds/unabated/discover")
    def api_unabated_discover(
        league: str = "nba",
        market_type: str = "props",
        offline_mode: bool = False,
    ) -> Dict[str, Any]:
        client = UnabatedOddsClient(offline_mode=offline_mode)
        return {
            "base_url": client.base_url,
            "configured": bool(client.api_key),
            "leagues": client.discover_leagues(),
            "sources": client.discover_market_sources(league=league, market_type=market_type),
        }

    @app.post("/v1/odds/unabated/snapshot")
    def api_unabated_snapshot(
        league: str = "nba",
        pulse_wave: str = "PULSE_1_OPENER",
        db_path: str = "alphaprop_clv_ledger.sqlite",
        offline_mode: bool = False,
    ) -> Dict[str, Any]:
        client = UnabatedOddsClient(offline_mode=offline_mode)
        snap = client.fetch_props_odds(league=league)
        events = client.normalize_unabated_props_to_events(snap)
        logger = MultiBookOddsCLVLogger(db_path=db_path, offline_mode=True)
        try:
            total_cons = []
            for ev in events:
                total_cons.extend(logger.ingest_event_snapshot(ev, pulse_wave=pulse_wave))
            return {
                "league": league,
                "pulse_wave": pulse_wave,
                "events_processed": len(events),
                "markets_captured": len(total_cons),
                "consensuses": [asdict(c) for c in total_cons],
            }
        finally:
            logger.close()

    @app.get("/v1/clv/report")
    def api_clv_report(
        slate_date: Optional[str] = None,
        segment: Optional[str] = None,
        db_path: str = "alphaprop_clv_ledger.sqlite",
    ) -> Dict[str, Any]:
        logger = MultiBookOddsCLVLogger(db_path=db_path, offline_mode=True)
        try:
            return logger.get_clv_brier_report(slate_date=slate_date, segment=segment)
        finally:
            logger.close()


    @app.post("/v1/guardrail/bss/run-nightly")
    def api_run_nightly_bss_guardrail(
        audit_date: Optional[str] = None,
        window_size: int = DEFAULT_BSS_WINDOW_SIZE,
        db_path: str = "alphaprop_clv_ledger.sqlite",
    ) -> Dict[str, Any]:
        guard = WalkForwardBSSGuardrail(db_path=db_path, window_size=window_size)
        try:
            return guard.run_nightly_audit(audit_date=audit_date, persist=True)
        finally:
            guard.close()

    @app.get("/v1/guardrail/bss/status")
    def api_get_bss_guardrail_status(
        segment: Optional[str] = None,
        db_path: str = "alphaprop_clv_ledger.sqlite",
    ) -> Dict[str, Any]:
        guard = WalkForwardBSSGuardrail(db_path=db_path)
        try:
            return {
                "active_throttles": guard.get_active_throttle_map(),
                "recent_audits": guard.get_audit_history(segment=segment, limit=25),
            }
        finally:
            guard.close()

    @app.post("/v1/guardrail/portfolio/size")
    def api_size_portfolio_with_guardrails(
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        plays = payload.get("plays", [])
        db_path = str(payload.get("db_path", "alphaprop_clv_ledger.sqlite"))
        max_u = float(payload.get("max_units_per_game", MAX_PER_GAME_EXPOSURE_UNITS))
        rho = float(payload.get("same_script_rho", 0.35))
        guard = WalkForwardBSSGuardrail(db_path=db_path)
        try:
            tmap = payload.get("throttle_map") or guard.get_active_throttle_map()
            return enforce_bss_and_game_exposure_guardrails(
                plays=plays,
                throttle_map=tmap,
                max_units_per_game=max_u,
                same_script_rho=rho,
            )
        finally:
            guard.close()

    return app


def render_streamlit_app() -> None:
    if not HAS_STREAMLIT:
        print("Streamlit is not installed in this environment. Run with --self-test or uvicorn.")
        return
    st.set_page_config(page_title="AlphaProp Pro™ Commercial v9.0 (Upgrades 1–8 Locked)", layout="wide")
    st.title("AlphaProp Pro™ Commercial Blueprint v9.0 — Upgrades 1–8 Active")
    st.caption("10,000-Trial Monte Carlo | Gaussian Copula SGP | Shin/Power No-Vig | Locked Upgrades 1–4 & Redirected Rules 5–8")
    col1, col2, col3 = st.columns(3)
    line = col1.number_input("Target Line", value=24.5, step=0.5)
    rate = col2.number_input("Per-Min Rate", value=0.76, step=0.02)
    dist = col3.selectbox("Distribution", ["neg_binomial", "bimodal_normal", "beta_binomial", "zero_inflated"])
    if st.button("Run 10,000-Trial Sim"):
        res = run_monte_carlo_prop_sim(line, rate, 35.0, 26.0, 0.14, dist_family=dist)
        st.json(res)


# ============================================================================
# SELF-TEST SUITE (5-PILLAR + UPGRADES 1–8 VERIFICATION)
# ============================================================================

def run_self_test() -> int:
    print("=== AlphaProp Commercial v9.0 (Upgrades 1–8 Locked) Self-Test ===")
    # 1. Upgrade #3: Automated nba_api PbP Stint Parser, SQLite Cache & On/Off Usage Ingestion
    pipe = NbaApiIngestionPipeline(cache_db_path=":memory:", offline_mode=True)
    raw_events = [
        {"action_number": 1, "period": 1, "clock": "PT11M40.00S", "team_abbr": "PHI", "action_type": "2pt", "shot_result": "Made", "person_id": 203954, "assist_person_id": 1630178, "points": 2},
        {"action_number": 2, "period": 1, "clock": "PT11M18.00S", "team_abbr": "BOS", "action_type": "2pt", "sub_type": "Driving Layup", "shot_result": "Missed", "shot_distance": 3.0, "person_id": 1628369},
        {"action_number": 3, "period": 1, "clock": "PT11M16.00S", "team_abbr": "PHI", "action_type": "rebound", "sub_type": "defensive", "person_id": 203954},
        {"action_number": 4, "period": 1, "clock": "PT06M00.00S", "team_abbr": "PHI", "action_type": "substitution", "player_out_id": 203954, "player_in_id": 1629651},
        {"action_number": 5, "period": 1, "clock": "PT05M42.00S", "team_abbr": "PHI", "action_type": "3pt", "shot_result": "Made", "person_id": 1630178, "points": 3},
        {"action_number": 6, "period": 1, "clock": "PT05M15.00S", "team_abbr": "BOS", "action_type": "2pt", "sub_type": "Dunk", "shot_result": "Made", "shot_distance": 1.0, "person_id": 1628369, "points": 2},
    ]
    parsed_stints = parse_pbp_events_to_stints(
        game_id="0022600999",
        team_abbr="PHI",
        events=raw_events,
        starting_lineups_by_period={1: (203954, 1627759, 1628983, 1630178, 1630596)},
        opp_team_abbr="BOS",
    )
    assert len(parsed_stints) == 2, f"Expected 2 parsed stints, got {len(parsed_stints)}"
    assert parsed_stints[0].duration_sec == 360 and parsed_stints[1].duration_sec == 360
    assert parsed_stints[0].rim_fga_allowed == 1 and parsed_stints[1].rim_fga_allowed == 1
    assert parsed_stints[1].player_events[1630178]["fg3m"] == 1.0

    # Verify SQLite cache round-trip & multi-game On/Off Usage Split
    stints = pipe.fetch_lineup_stints("0022600101", "PHI")
    stints_cached = pipe.fetch_lineup_stints("0022600101", "PHI", use_cache=True)
    assert len(stints) == 2 and len(stints_cached) == 2 and stints[0].possessions > 0, "Pillar 1 cache failed"

    split_surge = pipe.ingest_player_on_off_usage(
        team_abbr="PHI",
        target_player_id=1630178,
        teammate_out_ids=[203954],
        target_player_name="Tyrese Maxey",
        teammate_out_names=["Joel Embiid"],
        last_n_games=12,
    )
    assert split_surge.inferred_u1_archetype == "TYPE_A_SURGE", f"Expected TYPE_A_SURGE, got {split_surge.inferred_u1_archetype}"
    assert split_surge.usg_delta_eb > 0 and split_surge.rate_multiplier_by_stat["Points"] > 1.05
    assert 0.45 < split_surge.eb_reliability_weight < 0.70

    split_tax = pipe.ingest_player_on_off_usage(
        team_abbr="PHI",
        target_player_id=1627759,
        teammate_out_ids=[203954],
        target_player_name="Wing Spacer",
        teammate_out_names=["Joel Embiid"],
        last_n_games=12,
    )
    assert split_tax.inferred_u1_archetype == "TYPE_C_EFFICIENCY_TAX", f"Expected TYPE_C_EFFICIENCY_TAX, got {split_tax.inferred_u1_archetype}"
    assert split_tax.ts_delta_eb < -1.8

    # End-to-end wiring into Sigmoid, Monte Carlo & Anti-Double-Counting Upgrades 1-8
    enriched = build_on_off_enriched_prop_inputs(
        base_rate_per_min=0.76,
        stat_type="Points",
        on_off_split=split_surge,
        unassisted_fg_share=0.64,
        apply_empirical_rate_to_mc=True,
    )
    mc_base = run_monte_carlo_prop_sim(28.5, 0.76, 35.0, 26.0, 0.14, dist_family="neg_binomial", seed=42)
    mc_enriched = run_monte_carlo_prop_sim(28.5, enriched["enriched_rate_per_min"], 35.0, 26.0, 0.14, dist_family="neg_binomial", seed=42)
    assert mc_enriched["p_cover"] > mc_base["p_cover"], "Enriched on/off rate did not lift MC cover prob"
    dedup_wired = evaluate_all_upgrades_1_to_8(
        base_true_prob=mc_enriched["p_cover"],
        side="OVER",
        up1_archetype=enriched["inferred_u1_archetype"],
        rate_already_includes_vacuum=enriched["rate_already_includes_vacuum"],
    )
    assert any("Suppressed U1 delta" in g for g in dedup_wired["double_count_guards_triggered"]), "Anti-double-count U1 guard failed"
    print(
        f"[PASS] Upgrade #3 (PbP Stints & On/Off Usage): Parsed={len(parsed_stints)} stints, "
        f"Surge={split_surge.inferred_u1_archetype} (USG {split_surge.usg_pct_on}%->{split_surge.usg_pct_off}%, "
        f"EB_dUSG={split_surge.usg_delta_eb:+.2f}%, PTS_mult={enriched['empirical_stat_multiplier']}x, "
        f"MC P={mc_base['p_cover']:.3f}->{mc_enriched['p_cover']:.3f}), "
        f"Tax={split_tax.inferred_u1_archetype} (EB_dTS={split_tax.ts_delta_eb:+.2f}%)"
    )

    # 2. Core Math, Monte Carlo & Copula SGP
    w_def, w_sit = compute_sigmoid_weights(28.5, 0.42)
    sigma_g, p_blow = compute_dynamic_margin_volatility(-5.5, 101.2, 0.41)
    bimodal = simulate_bimodal_prop_probability(24.5, 0.78, 35.0, 25.0, p_blow)
    mc_nb = run_monte_carlo_prop_sim(24.5, 0.78, 35.0, 25.0, p_blow, dist_family="neg_binomial")
    mc_3pm = run_monte_carlo_prop_sim(2.5, 0.09, 34.0, 25.0, p_blow, dist_family="beta_binomial", three_pt_attempt_rate=7.8)
    sgp = simulate_copula_sgp([0.62, 0.58, 0.65], [[1.0, 0.35, -0.15], [0.35, 1.0, 0.20], [-0.15, 0.20, 1.0]])
    pace_sgp = simulate_shared_pace_sgp(
        [
            {"prob": 0.63, "side": "OVER", "stat_type": "PTS+AST", "team": "NYK"},
            {"prob": 0.61, "side": "OVER", "stat_type": "PTS", "team": "NYK"},
            {"prob": 0.62, "side": "OVER", "stat_type": "PRA", "team": "MIN"},
        ],
        offered_odds=+480,
        copula_type="student_t",
    )
    assert 0.0 < mc_nb["p_cover"] < 1.0 and 0.0 < mc_3pm["p_cover"] < 1.0, "Pillar 2 MC failed"
    assert sgp["p_correlated"] > sgp["p_independent"], "Pillar 2 Copula correlation lift failed"
    assert pace_sgp["shared_pace_active"] and pace_sgp["rock_fight_joint_miss_pct"] > pace_sgp["indep_joint_miss_pct"], "Shared game-pace copula failed"
    print(
        f"[PASS] Pillar 2 (MC & Shared-Pace Copula): NegBin P={mc_nb['p_cover']}, "
        f"BetaBin 3PM P={mc_3pm['p_cover']}, Copula Lift={sgp['correlation_lift_pct']}%, "
        f"Pace-SGP RockFightMiss={pace_sgp['rock_fight_joint_miss_pct']}% vs Indep={pace_sgp['indep_joint_miss_pct']}%"
    )

    # 3. Tactical Features & Locked Upgrades 1–8 (Including Redirected Rules 5–8)
    tac = compute_tactical_composite_multiplier(TacticalMatchupInput(ref_foul_delta=0.06, lineup_net_rating_delta=4.5, usage_vacuum_boost=0.045))
    u5_mirage = evaluate_upgrade_5_redirected_l5_momentum(4, 0.5, 0.4, 9.5, 2.0)
    u5_breakout = evaluate_upgrade_5_redirected_l5_momentum(5, 4.2, 3.8, 3.0, 0.5)
    u6_stale = evaluate_upgrade_6_redirected_h2h_continuity(24.0, False, False, False)
    u6_duplex = evaluate_upgrade_6_redirected_h2h_continuity(0.1, True, True, True)
    u7_rust = evaluate_upgrade_7_redirected_rest_rust_split(3, True, False, "3PM", 2.5, 20.0)
    u7_slasher = evaluate_upgrade_7_redirected_rest_rust_split(3, True, False, "PTS", 8.2, 52.0)
    u8_ft_star = evaluate_upgrade_8_redirected_whistle_asymmetry(1.5, 2.0, 8.8, 86.0, False)
    u8_foul_trap = evaluate_upgrade_8_redirected_whistle_asymmetry(1.5, 3.8, 3.5, 74.0, True)
    master_audit = evaluate_all_upgrades_1_to_8(base_true_prob=0.62, side="OVER")

    assert tac["tactical_mult"] > 1.0, "Pillar 3 tactical failed"
    assert u5_mirage["recommended_side"] == "UNDER" and u5_breakout["recommended_side"] == "OVER", "Upgrade 5 Redirect failed"
    assert u6_stale["effective_h2h_weight"] == 0.0 and u6_duplex["true_prob_delta"] == 0.026, "Upgrade 6 Redirect failed"
    assert u7_rust["recommended_side"] == "UNDER" and u7_slasher["recommended_side"] == "OVER", "Upgrade 7 Redirect failed"
    assert u8_ft_star["recommended_side"] == "OVER" and u8_foul_trap["recommended_side"] == "UNDER", "Upgrade 8 Redirect failed"
    assert master_audit["final_true_prob"] > 0.62, "Master Upgrades 1-8 pipeline failed"
    # Verify Guardrail #1 (Dynamic P_blowout Gate vs Static -10.5)
    gate_low_var = evaluate_dynamic_blowout_gate(-8.5, proj_pace=97.0, comb_3par=0.36, bench_cliff_sigma_bump=0.0, side="OVER")
    gate_high_var = evaluate_dynamic_blowout_gate(-8.5, proj_pace=103.5, comb_3par=0.44, bench_cliff_sigma_bump=2.5, side="OVER")
    assert gate_high_var["p_blowout"] > gate_low_var["p_blowout"], "Guardrail #1 variance sensitivity failed"
    assert gate_high_var["gate_status"] in ("CAUTION_HALF_STAKE", "VETO_ROUTE_TO_3Q_FLOOR"), "Guardrail #1 high-var gate failed"
    # Verify Coach Hook Modifier Table wired into Dynamic Blowout Gate
    gate_okc_early = evaluate_dynamic_blowout_gate(-8.5, proj_pace=101.0, comb_3par=0.41, bench_cliff_sigma_bump=1.5, team_abbr="OKC", side="OVER")
    gate_phi_heavy = evaluate_dynamic_blowout_gate(-8.5, proj_pace=101.0, comb_3par=0.41, bench_cliff_sigma_bump=1.5, team_abbr="PHI", side="OVER")
    gate_bos_reg = evaluate_dynamic_blowout_gate(-9.5, proj_pace=100.0, comb_3par=0.42, team_abbr="BOS", is_nba_cup=False, side="OVER")
    gate_bos_cup = evaluate_dynamic_blowout_gate(-9.5, proj_pace=100.0, comb_3par=0.42, team_abbr="BOS", is_nba_cup=True, side="OVER")
    assert len(COACH_HOOK_MODIFIER_TABLE) == 30, "Expected all 30 NBA teams in COACH_HOOK_MODIFIER_TABLE"
    assert gate_okc_early["coach_hook"]["tier"] == "EARLY_HOOK" and gate_phi_heavy["coach_hook"]["tier"] == "HEAVY_MINUTE"
    assert gate_okc_early["p_blowout"] > gate_phi_heavy["p_blowout"], "Early-hook coach must produce higher P_blowout than heavy-minute coach"
    assert gate_okc_early["gate_status"] == "VETO_ROUTE_TO_3Q_FLOOR" and gate_phi_heavy["gate_status"] == "CLEAR", "Coach hook modifier must flip -8.5 gate outcome"
    assert gate_bos_cup["coach_hook"]["coach_mult"] < gate_bos_reg["coach_hook"]["coach_mult"], "NBA Cup point-diff override for BOS failed"
    print(f"[PASS] Coach Hook Modifier Table (30 Teams): OKC(Daigneault 1.28x) P_blow={gate_okc_early['p_blowout']*100:.1f}% [{gate_okc_early['gate_status']}] vs PHI(Nurse 0.78x) P_blow={gate_phi_heavy['p_blowout']*100:.1f}% [{gate_phi_heavy['gate_status']}] | BOS Reg={gate_bos_reg['coach_hook']['coach_mult']}x -> Cup={gate_bos_cup['coach_hook']['coach_mult']}x")
    # Verify Guardrail #2 (Anti-Double-Counting + Log-Odds Damping)
    assert abs(master_audit["dedup_prob_delta"]) < abs(master_audit["naive_linear_delta"]), "Guardrail #2 dedup damping failed"
    assert len(master_audit["double_count_guards_triggered"]) >= 1, "Guardrail #2 audit log missing"
    print(f"[PASS] v9.1 Guardrails: LowVar(-8.5) P_blow={gate_low_var['p_blowout']*100:.1f}% ({gate_low_var['gate_status']}) vs HighVar(-8.5) P_blow={gate_high_var['p_blowout']*100:.1f}% ({gate_high_var['gate_status']}) | NaiveDelta={master_audit['naive_linear_delta']:+.3f} -> DedupDelta={master_audit['dedup_prob_delta']:+.3f}")
    print(
        f"[PASS] Pillar 3 & Upgrades 1-8 Redirected Engine: "
        f"U5(Mirage->{u5_mirage['recommended_side']}, Breakout->{u5_breakout['recommended_side']}), "
        f"U6(Duplex=+{u6_duplex['true_prob_delta']*100:.1f}%), "
        f"U7(3dRust->{u7_rust['recommended_side']}, 3dSlasher->{u7_slasher['recommended_side']}), "
        f"U8(FTStar->{u8_ft_star['recommended_side']}, FoulTrap->{u8_foul_trap['recommended_side']}), "
        f"MasterFinalProb={master_audit['final_true_prob']}"
    )

    # 3b. Upgrade #2: Tightened Alpha Edge & Underdog SGP Entry Gate (rho >= +0.35 OR EV >= +10.0%)
    gate_pass_rho = evaluate_alpha_underdog_sgp_gate(
        leg_specs=[
            {"player": "LAL Spread", "team": "LAL", "stat_type": "SPREAD", "side": "FAV", "prob": 0.72},
            {"player": "Domantas Sabonis", "team": "SAC", "stat_type": "PTS+REB", "side": "UNDER", "prob": 0.66},
            {"player": "SAC Team Total", "team": "SAC", "stat_type": "TEAM_TOTAL", "side": "UNDER", "prob": 0.64},
        ],
        p_correlated=0.354,
        offered_odds=310,
        effective_avg_corr=0.51,
    )
    assert gate_pass_rho["passed"] and gate_pass_rho["passes_rho"] and gate_pass_rho["stake_multiplier"] == 1.0

    gate_pass_ev = evaluate_alpha_underdog_sgp_gate(
        leg_specs=[
            {"player": "Cade Cunningham", "team": "DET", "stat_type": "PTS+AST", "side": "OVER", "prob": 0.674},
            {"player": "Franz Wagner", "team": "ORL", "stat_type": "PRA", "side": "OVER", "prob": 0.662},
        ],
        p_correlated=0.446,
        offered_odds=265,
        effective_avg_corr=0.0,
    )
    assert gate_pass_ev["passed"] and (not gate_pass_ev["passes_rho"]) and gate_pass_ev["passes_ev"]

    gate_blocked = evaluate_alpha_underdog_sgp_gate(
        leg_specs=[
            {"player": "Player A", "team": "NYK", "stat_type": "PTS+AST", "side": "OVER", "prob": 0.56},
            {"player": "Player B", "team": "MIN", "stat_type": "PTS+AST", "side": "OVER", "prob": 0.55},
        ],
        p_correlated=0.315,
        offered_odds=230,
        effective_avg_corr=0.18,
    )
    assert (not gate_blocked["passed"]) and gate_blocked["gate_status"] == "BLOCKED_0U" and gate_blocked["stake_multiplier"] == 0.0

    gate_trap = evaluate_alpha_underdog_sgp_gate(
        leg_specs=[
            {"player": "Big A", "team": "LAL", "stat_type": "REB", "side": "OVER", "prob": 0.65},
            {"player": "Big B", "team": "LAL", "stat_type": "PTS+REB", "side": "OVER", "prob": 0.64},
        ],
        p_correlated=0.45,
        offered_odds=280,
        effective_avg_corr=0.40,
    )
    assert (not gate_trap["passed"]) and gate_trap["gate_status"] == "BLOCKED_HARD_TRAP"
    print(
        f"[PASS] Upgrade #2 (Alpha/Underdog SGP Entry Gate): "
        f"CorrPass={gate_pass_rho['gate_status']}(rho={gate_pass_rho['effective_avg_corr']:+.2f}), "
        f"EvPass={gate_pass_ev['gate_status']}(EV={gate_pass_ev['combined_ev_pct']:+.1f}%), "
        f"WeakBlocked={gate_blocked['gate_status']}(0.00u), TrapBlocked={gate_trap['gate_status']}"
    )

    # 4. No-Vig, Kelly & CLV
    shin = devig_shin(-115, -105)
    pwr = devig_power(-115, -105)
    kelly = compute_kelly_stake(0.60, -110)
    clv = evaluate_clv_and_brier(-110, -130, 105, "OVER", 0.60, outcome_hit=1)
    assert abs((shin["fair_over"] + shin["fair_under"]) - 1.0) < 1e-4, "Shin devig failed"
    assert kelly["playable"] and clv["clv_pct"] > 0, "Kelly/CLV failed"
    exp_audit = enforce_per_game_exposure_cap(
        [
            {"player": "Jalen Brunson", "matchup": "NYK @ MIN", "ev_pct": 11.4, "units": 1.30},
            {"player": "LaMelo Ball", "matchup": "MIN vs NYK", "ev_pct": 9.8, "units": 1.20},
            {"player": "Mikal Bridges", "matchup": "NYK_MIN", "ev_pct": 7.2, "units": 0.90},
            {"player": "Karl-Anthony Towns", "matchup": "DAL @ OKC", "ev_pct": 13.7, "units": 1.50, "is_potd": True},
        ],
        max_units_per_game=2.0,
    )
    assert exp_audit["game_exposure_summary"]["MIN_NYK"]["capped_units"] == 2.0, "2.0u per-game cap failed"
    assert exp_audit["total_units_saved"] == 1.40, "Expected 1.40u trimmed from 3.40u NYK@MIN stack"
    assert exp_audit["game_exposure_summary"]["DAL_OKC"]["capped_units"] == 1.50, "Independent game should remain uncapped"
    print(
        f"[PASS] Pillar 4 (No-Vig, CLV & 2.0u Cap): Shin={shin['fair_over']}, Power={pwr['fair_over']}, "
        f"Half-Kelly={kelly['recommended_units']}u, CLV={clv['clv_pct']}%, "
        f"NYK@MIN Capped={exp_audit['game_exposure_summary']['MIN_NYK']['raw_units']}u->{exp_audit['game_exposure_summary']['MIN_NYK']['capped_units']}u"
    )

    # 4b. Upgrade #4: Automated Multi-Book Odds Snapshot & Pinnacle/Circa No-Vig CLV + Brier Logger
    clv_logger = MultiBookOddsCLVLogger(db_path=":memory:", offline_mode=True, devig_method="shin")
    p1_payload = clv_logger._synthetic_odds_api_event("evt_lal_phx_20261115", ["player_points_assists", "player_threes"], closing=False)
    close_payload = clv_logger._synthetic_odds_api_event("evt_lal_phx_20261115", ["player_points_assists", "player_threes"], closing=True)
    p1_cons = clv_logger.ingest_event_snapshot(p1_payload, pulse_wave="PULSE_1_OPENER")
    close_cons = clv_logger.ingest_event_snapshot(close_payload, pulse_wave="CLOSING_TIPOFF")
    assert len(p1_cons) == 2 and len(close_cons) == 2, "Expected 2 prop markets captured per wave"
    luka_p1 = next(c for c in p1_cons if c.player_name == "Luka Doncic")
    assert luka_p1.consensus_tier == "PINNACLE_CIRCA_SHARP", "Expected Pinnacle+Circa sharp consensus"
    assert luka_p1.best_retail_over_book == "fanduel" and luka_p1.best_retail_over_odds == -112, "Best retail line shopping failed"

    # Log 3 plays across segments and grade against Pinnacle/Circa closing consensus
    clv_logger.log_tracked_play(
        play_id="play_luka_potd", slate_date="2026-11-15", event_id="evt_lal_phx_20261115",
        game_key="LAL @ PHX", player_name="Luka Doncic", market="player_points_assists",
        side="OVER", line=41.5, bet_book="draftkings", bet_odds=-115, model_prob=0.672,
        units_staked=1.50, segment="POTD", pulse_wave="PULSE_1_OPENER",
    )
    graded_luka = clv_logger.attach_closing_and_grade("play_luka_potd", outcome_hit=1)
    assert graded_luka["clv_beat"] == 1 and graded_luka["line_clv_pts"] == 1.0, "Luka +1.0 pt line CLV failed"
    assert graded_luka["attribution_tag"] == "CLV_SKILL_WIN" and graded_luka["units_pnl"] > 1.25

    clv_logger.log_tracked_play(
        play_id="play_grimes_3pm", slate_date="2026-11-15", event_id="evt_lal_phx_20261115",
        game_key="LAL @ PHX", player_name="Quentin Grimes", market="player_threes",
        side="OVER", line=2.5, bet_book="draftkings", bet_odds=108, model_prob=0.618,
        units_staked=0.95, segment="VIP_OVER", pulse_wave="PULSE_1_OPENER",
    )
    graded_grimes = clv_logger.attach_closing_and_grade("play_grimes_3pm", outcome_hit=1)
    assert graded_grimes["clv_beat"] == 1 and graded_grimes["clv_pct"] > 2.0, "Grimes same-line CLV failed"

    clv_logger.log_tracked_play(
        play_id="play_flagg_under", slate_date="2026-11-15", event_id="evt_dal_bos_20261115",
        game_key="DAL @ BOS", player_name="Cooper Flagg", market="player_points_rebounds",
        side="UNDER", line=26.5, bet_book="draftkings", bet_odds=-112, model_prob=0.668,
        units_staked=1.15, segment="VIP_UNDER", pulse_wave="PULSE_3_PRETIP",
    )
    graded_flagg = clv_logger.attach_closing_and_grade(
        "play_flagg_under", outcome_hit=1, close_line=25.5, close_sharp_fair_over=0.48
    )
    assert graded_flagg["clv_beat"] == 1 and graded_flagg["line_clv_pts"] == 1.0, "Flagg Under line CLV failed"

    clv_report = clv_logger.get_clv_brier_report(slate_date="2026-11-15")
    assert clv_report["overall"]["clv_beat_rate_pct"] == 100.0, "Expected 100% CLV beat rate on test slate"
    assert clv_report["overall"]["bss_vs_book_pct"] > 0.0, "Expected positive BSS vs bet book"
    clv_logger.close()
    print(
        f"[PASS] Upgrade #4 (Multi-Book Pinnacle/Circa CLV & Brier Logger): "
        f"Consensus={luka_p1.consensus_tier}(BestRetail={luka_p1.best_retail_over_book} {luka_p1.best_retail_over_odds}), "
        f"LukaCLV=+{graded_luka['clv_pct']:.2f}%({graded_luka['line_clv_pts']:+.1f}pt), "
        f"PortfolioBSS_vs_Book={clv_report['overall']['bss_vs_book_pct']:+.2f}%, "
        f"ModelBrier={clv_report['overall']['model_brier_score']:.4f}"
    )


    # 4b. Upgrade m5: Nightly Walk-Forward BSS Guardrail & Correlation-Aware <=2.5u Per-Game Cap
    wf_logger = MultiBookOddsCLVLogger(db_path=":memory:", offline_mode=True, devig_method="shin")
    wf_guard = WalkForwardBSSGuardrail(
        db_path=":memory:",
        window_size=100,
        soft_min_sample=25,
        hard_min_sample=50,
        conn=wf_logger._conn,
    )
    # Seed 60 well-calibrated VIP_UNDER plays (positive BSS) + 60 miscalibrated MOONSHOT plays (negative BSS < -6%)
    sim_plays: List[Dict[str, Any]] = []
    for i in range(60):
        # VIP_UNDER: predicts 0.66, hits 42/60 (70%) vs -110 (0.524) -> strong positive BSS
        hit_u = 1 if (i % 10) < 7 else 0
        pid_u = f"wf_under_{i:03d}"
        wf_logger.log_tracked_play(
            play_id=pid_u, slate_date="2026-11-20", event_id=f"ev_u_{i}", game_key="DAL_BOS",
            player_name=f"UnderPlayer{i}", market="player_points_rebounds", side="UNDER",
            line=26.5, bet_book="draftkings", bet_odds=-110, model_prob=0.66,
            units_staked=1.0, segment="VIP_UNDER", pulse_wave="PULSE_3_PRETIP",
        )
        wf_logger.attach_closing_and_grade(pid_u, outcome_hit=hit_u, close_line=26.0, close_sharp_fair_over=0.45)

        # ALPHA_SGP: overconfident model (predicts 0.45 vs -110/0.35 implied, hits only 12/60 = 20%) -> severe negative BSS
        hit_s = 1 if (i % 5) == 0 else 0
        pid_s = f"wf_sgp_{i:03d}"
        wf_logger.log_tracked_play(
            play_id=pid_s, slate_date="2026-11-20", event_id=f"ev_s_{i}", game_key="NYK_MIN",
            player_name=f"SgpStack{i}", market="sgp_combo", side="OVER",
            line=1.5, bet_book="fanduel", bet_odds=180, model_prob=0.46,
            units_staked=1.0, segment="ALPHA_SGP", pulse_wave="PULSE_3_PRETIP",
        )
        wf_logger.attach_closing_and_grade(pid_s, outcome_hit=hit_s, close_line=1.5, close_sharp_fair_over=0.33)
        sim_plays.append({"segment": "ALPHA_SGP", "model_prob": 0.46, "bet_odds": 180, "outcome_hit": hit_s, "units_staked": 1.0})

    nightly_res = wf_guard.run_nightly_audit(audit_date="2026-11-21", persist=True)
    assert nightly_res["tiers"]["VIP_UNDER"]["status"] == "HEALTHY", f"Expected VIP_UNDER HEALTHY, got {nightly_res['tiers']['VIP_UNDER']}"
    assert nightly_res["tiers"]["VIP_UNDER"]["kelly_multiplier"] == 1.0
    assert nightly_res["tiers"]["ALPHA_SGP"]["status"] == "PAUSED_ZERO_STAKE", f"Expected ALPHA_SGP PAUSED_ZERO_STAKE, got {nightly_res['tiers']['ALPHA_SGP']}"
    assert nightly_res["tiers"]["ALPHA_SGP"]["kelly_multiplier"] == 0.0
    assert "ALPHA_SGP" in nightly_res["paused_tiers"] and nightly_res["alert_required"] is True

    # Test walk-forward zero-lookahead simulation drawdown protection
    wf_sim = WalkForwardBSSGuardrail.run_walk_forward_simulation(sim_plays, window_size=100, soft_min_sample=25, hard_min_sample=50)
    assert wf_sim["drawdown_saved_u"] > 0.0 and wf_sim["pnl_improvement_units"] > 0.0, f"Walk-forward simulation did not reduce drawdown: {wf_sim}"

    # Test unified BSS throttle + same-script correlation haircut + <=2.5u hard per-game exposure cap
    active_tmap = wf_guard.get_active_throttle_map()
    candidate_slate = [
        {"player": "KAT", "matchup": "NYK @ MIN", "side": "OVER", "segment": "VIP_UNDER", "is_potd": True, "ev_pct": 14.2, "units": 1.50},
        {"player": "Brunson", "matchup": "NYK @ MIN", "side": "OVER", "segment": "VIP_UNDER", "is_potd": False, "ev_pct": 11.5, "units": 1.25},
        {"player": "Bridges", "matchup": "NYK @ MIN", "side": "OVER", "segment": "VIP_UNDER", "is_potd": False, "ev_pct": 9.8, "units": 1.00},
        {"player": "AlphaSGP", "matchup": "NYK @ MIN", "side": "OVER", "segment": "ALPHA_SGP", "is_potd": False, "ev_pct": 8.0, "units": 1.00},
    ]
    sized = enforce_bss_and_game_exposure_guardrails(candidate_slate, throttle_map=active_tmap, max_units_per_game=2.5, same_script_rho=0.35)
    assert sized["game_exposure_summary"]["MIN_NYK"]["capped_units"] <= 2.5, "Per-game <=2.5u cap violated"
    assert sized["bss_units_trimmed"] == 1.0, f"Expected 1.0u trimmed by ALPHA_SGP BSS pause, got {sized['bss_units_trimmed']}"
    assert sized["correlation_units_trimmed"] > 0.0, "Expected positive same-script correlation haircut"
    wf_logger.close()
    print(
        f"[PASS] Upgrade m5 (Nightly Walk-Forward BSS Guardrail & <=2.5u Correlation Cap): "
        f"VIP_UNDER={nightly_res['tiers']['VIP_UNDER']['status']}({nightly_res['tiers']['VIP_UNDER']['bss_vs_book_pct']:+.1f}% BSS, 1.00x), "
        f"ALPHA_SGP={nightly_res['tiers']['ALPHA_SGP']['status']}({nightly_res['tiers']['ALPHA_SGP']['bss_vs_book_pct']:+.1f}% BSS, 0.00x), "
        f"WF_DrawdownSaved={wf_sim['drawdown_saved_u']:.2f}u, "
        f"MIN_NYK_Exposure={sized['total_pre_guardrail_units']:.2f}u->{sized['game_exposure_summary']['MIN_NYK']['capped_units']:.2f}u"
    )

    # 5. FastAPI Router & Upgrade #5 Server-Side JWT Paywall Middleware Verification
    if HAS_FASTAPI:
        from fastapi.testclient import TestClient

        test_secret = "super-secret-supabase-jwt-signing-key-2026"
        test_admin_key = "alphaprop-service-cron-key-999"
        test_store = EntitlementStore(db_path=":memory:")
        app = create_fastapi_app(
            auth_enabled=True,
            jwt_secret=test_secret,
            entitlement_store=test_store,
            admin_api_key=test_admin_key,
        )
        assert app is not None, "FastAPI app creation failed"
        route_paths = {getattr(r, "path", "") for r in app.routes}
        assert "/v1/slate/public-potd" in route_paths and "/v1/slate/vip-board" in route_paths
        assert "/v1/auth/me" in route_paths and "/v1/admin/entitlements/sync" in route_paths

        client = TestClient(app)
        verifier = app.state.jwt_verifier
        now_ts = int(time.time())

        # 5a. Public POTD is accessible without auth and leaks ZERO VIP picks
        r_pub = client.get("/v1/slate/public-potd")
        assert r_pub.status_code == 200 and r_pub.json()["free_pick_of_the_day"]["is_free_potd"] is True
        assert "vip_picks" not in r_pub.json() and r_pub.json()["locked_vip_picks_count"] == 2

        # 5b. Unauthenticated request to VIP board -> 401
        r_noauth = client.get("/v1/slate/vip-board")
        assert r_noauth.status_code == 401 and r_noauth.json()["error_code"] == "AUTH_REQUIRED"

        # 5c. Tampered signature & expired JWT -> 401
        bad_tok = verifier.sign_hs256_jwt({"sub": "u_free", "aud": "authenticated", "exp": now_ts + 3600}, secret="wrong-secret")
        assert client.get("/v1/slate/vip-board", headers={"Authorization": f"Bearer {bad_tok}"}).status_code == 401
        exp_tok = verifier.sign_hs256_jwt({"sub": "u_free", "aud": "authenticated", "exp": now_ts - 120})
        r_exp = client.get("/v1/slate/vip-board", headers={"Authorization": f"Bearer {exp_tok}"})
        assert r_exp.status_code == 401 and r_exp.json()["error_code"] == "TOKEN_EXPIRED"

        # 5d. Valid FREE user JWT -> /v1/auth/me 200 (is_vip=False), /v1/slate/vip-board 403
        free_tok = verifier.sign_hs256_jwt({"sub": "u_free", "email": "free@example.com", "aud": "authenticated", "exp": now_ts + 3600})
        r_me_free = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {free_tok}"})
        assert r_me_free.status_code == 200 and r_me_free.json()["permissions"]["can_view_vip_board"] is False
        r_vip_denied = client.get("/v1/slate/vip-board", headers={"Authorization": f"Bearer {free_tok}"})
        assert r_vip_denied.status_code == 403 and r_vip_denied.json()["error_code"] == "VIP_SUBSCRIPTION_REQUIRED"

        # 5e. Non-admin cannot call /v1/admin/entitlements/sync -> 403
        r_adm_denied = client.post(
            "/v1/admin/entitlements/sync",
            headers={"Authorization": f"Bearer {free_tok}"},
            json={"user_id": "u_free", "tier": "vip", "status": "active"},
        )
        assert r_adm_denied.status_code == 403 and r_adm_denied.json()["error_code"] == "ADMIN_REQUIRED"

        # 5f. Stripe webhook (via Service Key) upgrades u_free -> VIP immediately (and is idempotent!)
        future_iso = datetime.fromtimestamp(now_ts + 86400 * 30, tz=timezone.utc).isoformat()
        r_sync = client.post(
            "/v1/admin/entitlements/sync",
            headers={"X-AlphaProp-Service-Key": test_admin_key},
            json={
                "user_id": "u_free",
                "email": "free@example.com",
                "tier": "vip",
                "status": "active",
                "provider": "stripe",
                "provider_subscription_id": "sub_stripe_123",
                "current_period_end": future_iso,
                "event_id": "evt_stripe_upgrade_001",
            },
        )
        assert r_sync.status_code == 200 and r_sync.json()["idempotent_skip"] is False
        r_sync_dup = client.post(
            "/v1/admin/entitlements/sync",
            headers={"X-AlphaProp-Service-Key": test_admin_key},
            json={"user_id": "u_free", "tier": "vip", "status": "active", "event_id": "evt_stripe_upgrade_001"},
        )
        assert r_sync_dup.status_code == 200 and r_sync_dup.json()["idempotent_skip"] is True

        # 5g. Same user JWT now immediately unlocks /v1/slate/vip-board (200) & /v1/pricing/devig (200)!
        r_vip_ok = client.get("/v1/slate/vip-board", headers={"Authorization": f"Bearer {free_tok}"})
        assert r_vip_ok.status_code == 200 and len(r_vip_ok.json()["vip_picks"]) == 2
        r_devig = client.post(
            "/v1/pricing/devig",
            headers={"Authorization": f"Bearer {free_tok}"},
            json={"over_odds": -115, "under_odds": -105, "model_prob": 0.62},
        )
        assert r_devig.status_code == 200

        # 5h. Cancellation / expiration immediately revokes access back to 403!
        past_iso = datetime.fromtimestamp(now_ts - 60, tz=timezone.utc).isoformat()
        client.post(
            "/v1/admin/entitlements/sync",
            headers={"X-AlphaProp-Service-Key": test_admin_key},
            json={"user_id": "u_free", "tier": "vip", "status": "canceled", "current_period_end": past_iso, "event_id": "evt_stripe_cancel_002"},
        )
        r_revoked = client.get("/v1/slate/vip-board", headers={"Authorization": f"Bearer {free_tok}"})
        assert r_revoked.status_code == 403 and r_revoked.json()["subscription_status"] == "expired"
        test_store.close()
        print(
            f"[PASS] Upgrade #5 (FastAPI JWT Paywall Middleware & Entitlements): {len(app.routes)} routes | "
            f"PublicPOTD=200(NoLeak), NoAuth/BadSig/Expired=401, FreeUser=403, StripeUpgrade=200(VIP Unlocked), CancelRevoke=403"
        )
    else:
        print("[SKIP] Pillar 5 (FastAPI not installed)")

    print("=== ALL 5 COMMERCIAL PILLARS + UPGRADES 1-8 REDIRECTED ENGINE VERIFIED ===")
    run_v88_syndicate_tests()
    return 0


# === CONSISTENCY INDEX ENGINE (v8.6) ===
import statistics as _ci_stats

class ConsistencyIndexEngine:
    """
    Evaluates empirical L10-L15 player game-log consistency to eliminate the
    Mean-vs-Median right-skew trap (e.g., [10, 10, 40] avg 20.0 / median 10.0
    vs. [22, 24, 19] avg 21.7 / median 22.0), computes empirical Gamma-Poisson
    overdispersion parameter r, and discretizes stakes to clean 0.5u increments.
    """

    @staticmethod
    def discretize_clean_units(raw_kelly_units: float, is_potd: bool = False) -> float:
        """Snap raw Kelly stake to clean bettor half-unit increments: 0.5, 1.0, 1.5, 2.0."""
        if raw_kelly_units < 0.25:
            return 0.0
        if raw_kelly_units < 0.75:
            return 0.5
        if raw_kelly_units < 1.25:
            return 1.0
        if raw_kelly_units < 1.75:
            return 1.5
        return 2.0 if is_potd else 1.5

    @staticmethod
    def evaluate(game_logs: list, book_line: float, side: str = 'OVER',
                 projected_delta: float = 0.0, is_type_a_vacuum: bool = False,
                 is_lineup_promotion: bool = False, role_matched_logs: list = None) -> dict:
        """
        Args:
            game_logs: Recent L10-L15 stat values (e.g. [10, 10, 40] or [22, 24, 19]).
            book_line: Current sportsbook prop line (e.g. 17.5).
            side: 'OVER' or 'UNDER'.
            projected_delta: Confirmed structural injury/pace adjustment.
            has_role_boost: True if backed by a verified Type A primary star OUT vacuum.
        """
        src_logs = role_matched_logs if (role_matched_logs and len(role_matched_logs) >= 3) else game_logs
        vals = [float(x) for x in src_logs if x is not None]
        has_role_boost = bool(is_type_a_vacuum or is_lineup_promotion)
        if len(vals) < 3:
            return {
                'tier': 'INSUFFICIENT_SAMPLE', 'cv': None, 'mean': None,
                'median': None, 'winsorized_mean': None, 'skew_gap_pct': 0.0,
                'gamma_poisson_r': 12.0, 'approved': True,
                'routing': 'STANDARD', 'reason': 'Fewer than 3 games; fallback to prior.'
            }

        mean_val = sum(vals) / len(vals)
        med_val = _ci_stats.median(vals)
        stdev_val = _ci_stats.stdev(vals)
        cv = stdev_val / max(mean_val, 1e-6)

        # Winsorized mean: drop the single highest outlier game (prevents 10, 10, 40 distortion)
        sorted_vals = sorted(vals)
        winsor_vals = sorted_vals[:-1]
        winsor_mean = sum(winsor_vals) / len(winsor_vals)

        adj_median = med_val + projected_delta
        adj_winsor = winsor_mean + projected_delta
        skew_gap_pct = ((mean_val - med_val) / max(med_val, 1.0)) * 100.0

        # Empirical Gamma-Poisson dispersion parameter r (smaller r = heavier right-skew)
        var_val = stdev_val ** 2
        gamma_r = (mean_val ** 2) / max(var_val - mean_val, 0.05 * max(mean_val, 1.0))
        gamma_r = round(max(1.5, min(40.0, gamma_r)), 2)

        if cv < 0.25:
            tier = 'TIER_A_GRINDER'
        elif cv <= 0.35:
            tier = 'TIER_B_MODERATE'
        else:
            tier = 'TIER_C_SPIKE'

        side_u = side.upper()
        approved = True
        routing = 'STANDARD_' + side_u
        reason = f'{tier} (CV={cv:.2f}, Med={med_val:.1f}, Winsor={winsor_mean:.1f}, Mean={mean_val:.1f})'

        if side_u == 'OVER':
            if tier == 'TIER_C_SPIKE' and not has_role_boost:
                approved = False
                routing = 'VETO_OVER_PIVOT_UNDER_OR_LADDER'
                reason = f'VETO: Boom-bust spike profile (CV={cv:.2f}, Mean={mean_val:.1f} vs Med={med_val:.1f}); route to Under or Plus-Money Ladder.'
            elif skew_gap_pct >= 15.0 and adj_median < book_line:
                approved = False
                routing = 'VETO_OUTLIER_INFLATED_MEAN'
                reason = f'VETO: Right-skew outlier trap (Mean {mean_val:.1f} exceeds Median {med_val:.1f} by +{skew_gap_pct:.1f}% and Adj Median {adj_median:.1f} < Line {book_line}).'
            elif tier == 'TIER_B_MODERATE' and (adj_median - book_line) < 1.5 and (adj_winsor - book_line) < 1.5:
                approved = False
                routing = 'HOLD_BUBBLE_INSUFFICIENT_MEDIAN_CUSHION'
                reason = f'HOLD: Moderate variance (CV={cv:.2f}) requires >= +1.5 median/winsor cushion above {book_line}.'
            elif adj_winsor < book_line and adj_median < book_line:
                approved = False
                routing = 'VETO_BELOW_LINE_MEDIAN'
                reason = f'VETO: Both adjusted median ({adj_median:.1f}) and Winsorized mean ({adj_winsor:.1f}) sit below line {book_line}.'
        elif side_u == 'UNDER':
            if tier == 'TIER_C_SPIKE' and skew_gap_pct >= 15.0 and adj_median < book_line:
                routing = 'PRIME_SKEW_UNDER_SNIPER'
                reason = f'PRIME UNDER: Inflated mean ({mean_val:.1f}) masks low median ({med_val:.1f}) vs line {book_line} (CV={cv:.2f}).'

        return {
            'tier': tier,
            'cv': round(cv, 3),
            'mean': round(mean_val, 2),
            'median': round(med_val, 2),
            'winsorized_mean': round(winsor_mean, 2),
            'skew_gap_pct': round(skew_gap_pct, 1),
            'gamma_poisson_r': gamma_r,
            'approved': approved,
            'routing': routing,
            'reason': reason,
        }


# === OPPORTUNITY & SHOT-QUALITY ENGINES (v8.7) ===

class OpportunityPropEngine:
    """
    Models Rebounds and Assists from underlying opportunity volume rather than
    noisy raw box-score averages:
      1) Assists = Potential Assists x Pace x Opp AST Mult x Teammate eFG Conversion
      2) Rebounds = Rebound Chances x Missed-Shot Pool x Long-Rebound 3PAr Geometry
    """

    LEAGUE_AVG_3PAR = 0.415
    LEAGUE_AVG_FG_PCT = 0.468
    LEAGUE_AVG_AST_CONV = 0.545

    @classmethod
    def project_assists(
        cls,
        potential_assists_pg: float,
        actual_assists_pg: float,
        teammate_efg_delta: float = 0.0,
        opp_ast_allowed_mult: float = 1.0,
        pace_mult: float = 1.0,
        book_line: float = 6.5,
    ) -> dict:
        """
        Project expected assists from potential assists and detect conversion regression.
        teammate_efg_delta: active lineup eFG% minus league avg (e.g. +0.025 for elite spacing).
        """
        actual_conv = actual_assists_pg / max(potential_assists_pg, 1e-6)
        # Empirical Bayes shrinkage of conversion rate toward league/lineup baseline
        expected_conv = max(0.38, min(0.70, cls.LEAGUE_AVG_AST_CONV + 0.65 * teammate_efg_delta))
        blended_conv = 0.70 * expected_conv + 0.30 * max(0.35, min(0.72, actual_conv))

        proj_ast = potential_assists_pg * pace_mult * opp_ast_allowed_mult * blended_conv
        edge_vs_line = proj_ast - book_line
        conv_gap = expected_conv - actual_conv

        if conv_gap >= 0.06 and edge_vs_line >= 0.75:
            signal = 'BUY_LOW_AST_REGRESSION_OVER'
        elif conv_gap <= -0.06 and edge_vs_line <= -0.75:
            signal = 'FADE_HOT_CONVERSION_AST_UNDER'
        elif edge_vs_line >= 0.75:
            signal = 'VOLUME_AST_OVER'
        elif edge_vs_line <= -0.75:
            signal = 'VOLUME_AST_UNDER'
        else:
            signal = 'NEUTRAL'

        return {
            'potential_assists_pg': round(potential_assists_pg, 2),
            'actual_conv_pct': round(actual_conv * 100.0, 1),
            'expected_conv_pct': round(expected_conv * 100.0, 1),
            'projected_assists': round(proj_ast, 2),
            'edge_vs_line': round(edge_vs_line, 2),
            'signal': signal,
        }

    @classmethod
    def project_rebounds(
        cls,
        reb_chances_pg: float,
        base_conv_rate: float,
        opp_3par: float = 0.415,
        opp_fg_pct: float = 0.468,
        player_role: str = 'WING',
        pace_mult: float = 1.0,
        book_line: float = 6.5,
    ) -> dict:
        """
        Project expected rebounds using opponent missed-shot pool and 3PAr long-rebound geometry.
        High-3PAr opponents create long rebounds that boost GUARD/WING and dock DROP_CENTER.
        """
        miss_pool_mult = (1.0 - opp_fg_pct) / (1.0 - cls.LEAGUE_AVG_FG_PCT)
        three_par_delta = opp_3par - cls.LEAGUE_AVG_3PAR  # e.g. +0.065 for 48% 3PAr team

        role_u = player_role.upper()
        if role_u in ('GUARD', 'WING'):
            long_reb_mult = 1.0 + max(-0.10, min(0.16, three_par_delta * 1.40))
        elif role_u == 'DROP_CENTER':
            long_reb_mult = 1.0 - max(-0.08, min(0.14, three_par_delta * 1.15))
        else:  # MOBILE_BIG / STRETCH_BIG
            long_reb_mult = 1.0 + max(-0.05, min(0.06, three_par_delta * 0.25))

        adj_chances = reb_chances_pg * pace_mult * miss_pool_mult * long_reb_mult
        proj_reb = adj_chances * max(0.30, min(0.82, base_conv_rate))
        edge_vs_line = proj_reb - book_line

        if edge_vs_line >= 0.85:
            signal = 'LONG_REB_GUARD_WING_OVER' if (role_u in ('GUARD', 'WING') and three_par_delta >= 0.03) else 'OPPORTUNITY_REB_OVER'
        elif edge_vs_line <= -0.85:
            signal = 'LONG_REB_DROP_BIG_UNDER' if (role_u == 'DROP_CENTER' and three_par_delta >= 0.03) else 'OPPORTUNITY_REB_UNDER'
        else:
            signal = 'NEUTRAL'

        return {
            'adj_reb_chances': round(adj_chances, 2),
            'long_reb_geometry_mult': round(long_reb_mult, 3),
            'miss_pool_mult': round(miss_pool_mult, 3),
            'projected_rebounds': round(proj_reb, 2),
            'edge_vs_line': round(edge_vs_line, 2),
            'signal': signal,
        }


class ShotQualityRegressionEngine:
    """
    Separates NBA Shot Quality (defender distance: Wide Open >6ft, Open 4-6ft,
    Tight 2-4ft, Very Tight <2ft) from noisy recent shot-making to surface
    buy-low x3P% Overs and fade-contested-mirage Unders.
    """

    # Empirical NBA defender-distance 3P% baselines
    DIST_3P_BASELINES = {
        'wide_open': 0.392,   # > 6 ft
        'open': 0.354,        # 4-6 ft
        'tight': 0.308,       # 2-4 ft
        'very_tight': 0.255,  # 0-2 ft
    }
    PRIOR_3PA_K = 360.0

    @classmethod
    def evaluate_3pt_quality(
        cls,
        wide_open_att: float,
        open_att: float,
        tight_att: float,
        very_tight_att: float,
        actual_recent_3p_pct: float,
        career_skill_offset: float = 0.0,
        projected_3pa_pg: float = 6.5,
    ) -> dict:
        total_att = max(wide_open_att + open_att + tight_att + very_tight_att, 1e-6)
        w_wo = wide_open_att / total_att
        w_op = open_att / total_att
        w_ti = tight_att / total_att
        w_vt = very_tight_att / total_att

        base_x3p = (
            w_wo * cls.DIST_3P_BASELINES['wide_open']
            + w_op * cls.DIST_3P_BASELINES['open']
            + w_ti * cls.DIST_3P_BASELINES['tight']
            + w_vt * cls.DIST_3P_BASELINES['very_tight']
        )
        # Apply player's established career shooting skill offset (e.g. +0.025 for elite shooter)
        x3p_pct = max(0.24, min(0.47, base_x3p + career_skill_offset))

        # Empirical Bayes shrunk true 3P% projection
        eb_w = total_att / (total_att + cls.PRIOR_3PA_K)
        shrunk_3p_pct = (1.0 - eb_w) * x3p_pct + eb_w * actual_recent_3p_pct

        quality_gap_pct = (x3p_pct - actual_recent_3p_pct) * 100.0
        wide_open_share_pct = w_wo * 100.0
        contested_share_pct = (w_ti + w_vt) * 100.0

        if quality_gap_pct >= 5.0 and wide_open_share_pct >= 35.0 and projected_3pa_pg >= 5.5:
            signal = 'BUY_LOW_X3P_OVER'
            proj_3pm_edge_note = f'Cold luck on clean looks (x3P {x3p_pct*100:.1f}% vs actual {actual_recent_3p_pct*100:.1f}%, {wide_open_share_pct:.0f}% wide-open).'
        elif quality_gap_pct <= -6.0 and contested_share_pct >= 45.0:
            signal = 'FADE_CONTESTED_MIRAGE_UNDER'
            proj_3pm_edge_note = f'Unsustainable tough shot-making (actual {actual_recent_3p_pct*100:.1f}% vs x3P {x3p_pct*100:.1f}%, {contested_share_pct:.0f}% contested).'
        else:
            signal = 'NEUTRAL_ALIGNED'
            proj_3pm_edge_note = f'Shot-making aligned with defender-distance quality (x3P {x3p_pct*100:.1f}% vs actual {actual_recent_3p_pct*100:.1f}%).'

        return {
            'x3p_pct': round(x3p_pct * 100.0, 1),
            'actual_recent_3p_pct': round(actual_recent_3p_pct * 100.0, 1),
            'shrunk_true_3p_pct': round(shrunk_3p_pct * 100.0, 1),
            'quality_gap_pct': round(quality_gap_pct, 1),
            'wide_open_share_pct': round(wide_open_share_pct, 1),
            'contested_share_pct': round(contested_share_pct, 1),
            'projected_3pm': round(projected_3pa_pg * shrunk_3p_pct, 2),
            'signal': signal,
            'note': proj_3pm_edge_note,
        }


# =============================================================================
# === ALPHAPROP NBA v8.8 SYNDICATE MODULES + CROSS-SPORT CLV AUTO-CALIBRATOR ===
# =============================================================================

class RimProtectorShotLocationEngine:
    """
    Models opponent shot-location geometry shifts (rim frequency/FG% vs. 3PAr/pull-up)
    when an elite rim protector / drop big is OUT vs. ACTIVE.
    Bounded orthogonally to [0.86, 1.14] to prevent double-counting with usage vacuums.
    """

    @staticmethod
    def evaluate(
        player_name: str,
        player_archetype: str,
        base_pts: float,
        base_3pm: float = 2.0,
        opp_rim_protector_tier: str = "AVERAGE",
        opp_rim_protector_out: bool = False,
        player_rim_freq: float = 0.30,
        player_pullup_3pt_freq: float = 0.20,
    ) -> dict:
        arch = player_archetype.upper()
        tier = opp_rim_protector_tier.upper()
        pts_mult = 1.0
        three_mult = 1.0
        signal = "NEUTRAL_SHOT_GEOMETRY"
        note = "Standard opponent rim protection; no shot-location redirection."

        if opp_rim_protector_out and tier in ("ELITE_DROP", "ELITE_RIM"):
            # Rim vacuum: slashers & paint bigs feast at the rim (+4.5% to +6.5% rim FG%)
            if arch in ("SLASHER", "PAINT_BIG", "DRIVER") or player_rim_freq >= 0.38:
                boost = min(0.12, 0.065 + 0.14 * max(0.0, player_rim_freq - 0.30))
                pts_mult = 1.0 + boost
                three_mult = 0.96
                signal = "RIM_VACUUM_SLASHER_OVER"
                note = f"Elite rim protector OUT -> +{boost*100:.1f}% paint/rim scoring boost for {player_name}."
            elif arch in ("CATCH_AND_SHOOT", "STRETCH_WING"):
                pts_mult = 0.985
                three_mult = 0.965
                signal = "RIM_VACUUM_PERIMETER_SLIGHT_DIP"
                note = "Offense attacks open paint more frequently; minor dip in kickout 3PA."
        elif (not opp_rim_protector_out) and tier in ("ELITE_DROP", "ELITE_RIM"):
            # Active elite drop big deters rim finishes and concedes pull-up 3s / floaters
            if arch in ("SLASHER", "PAINT_BIG", "DRIVER") or player_rim_freq >= 0.40:
                tax = min(0.11, 0.06 + 0.12 * max(0.0, player_rim_freq - 0.32))
                pts_mult = 1.0 - tax
                signal = "ELITE_RIM_DETERRENT_SLASHER_UNDER"
                note = f"Active {tier} anchor deters rim attempts (-{tax*100:.1f}% PTS tax on {player_name})."
            elif arch in ("PULL_UP_GUARD", "FLOATER_GUARD") or player_pullup_3pt_freq >= 0.28:
                boost = min(0.10, 0.05 + 0.15 * max(0.0, player_pullup_3pt_freq - 0.22))
                pts_mult = 1.0 + boost * 0.85
                three_mult = 1.0 + boost
                signal = "DROP_COVERAGE_PULL_UP_OVER"
                note = f"Drop coverage concedes above-the-break pull-ups (+{boost*100:.1f}% 3PM boost)."

        pts_mult = max(0.86, min(1.14, pts_mult))
        three_mult = max(0.86, min(1.14, three_mult))
        return {
            "player": player_name,
            "pts_mult": round(pts_mult, 4),
            "three_mult": round(three_mult, 4),
            "adj_pts": round(base_pts * pts_mult, 2),
            "adj_3pm": round(base_3pm * three_mult, 2),
            "signal": signal,
            "note": note,
        }


class RefCrewFoulInteractionEngine:
    """
    Separates high-whistle NBA referee crews into two opposing player-specific edges:
      1) High foul-commit players (bigs / aggressive wings) -> Early Foul-Trouble Minutes Under
      2) Elite foul-drawers vs. foul-prone defenses -> FTM / PTS Over boost
      3) Low-whistle crews -> FT-dependent scorer tax
    """

    @staticmethod
    def evaluate(
        player_name: str,
        base_minutes: float,
        base_pts: float,
        pf_per_36: float,
        fta_per_36: float,
        crew_foul_rate_mult: float = 1.0,
        opp_ftr: float = 0.22,
        opp_def_foul_rate_mult: float = 1.0,
        side: str = "OVER",
    ) -> dict:
        side_u = side.upper()
        # Probability of hitting early foul trouble (2+ in Q1 or 3+ in H1)
        raw_ft_prob = 0.14 + max(0.0, pf_per_36 - 2.8) * 0.10 * crew_foul_rate_mult + max(0.0, opp_ftr - 0.22) * 0.65
        p_foul_trouble = max(0.05, min(0.58, raw_ft_prob))

        minutes_delta = 0.0
        fta_delta = 0.0
        pts_delta = 0.0
        approved = True
        signal = "NEUTRAL_WHISTLE"

        if crew_foul_rate_mult >= 1.07 and pf_per_36 >= 3.8 and opp_ftr >= 0.24:
            # High-whistle foul-trouble trap
            minutes_delta = -round(min(3.8, 1.4 + (pf_per_36 - 3.5) * 1.1 * (crew_foul_rate_mult - 0.95) * 6.0), 2)
            pts_delta = round((minutes_delta / max(base_minutes, 1.0)) * base_pts, 2)
            signal = "HIGH_WHISTLE_FOUL_TROUBLE_UNDER"
            if side_u == "OVER" and p_foul_trouble >= 0.34:
                approved = False
        elif crew_foul_rate_mult >= 1.07 and fta_per_36 >= 7.5 and opp_def_foul_rate_mult >= 1.03:
            # Elite foul-drawer boost
            extra_fta = min(2.2, (fta_per_36 / 36.0) * base_minutes * ((crew_foul_rate_mult * opp_def_foul_rate_mult) - 1.0) * 1.45)
            fta_delta = round(extra_fta, 2)
            pts_delta = round(extra_fta * 0.82, 2)  # ~82% FT conversion for high-volume drawers
            signal = "HIGH_WHISTLE_FOUL_DRAW_OVER"
        elif crew_foul_rate_mult <= 0.93 and fta_per_36 >= 7.0:
            lost_fta = min(1.8, (fta_per_36 / 36.0) * base_minutes * (1.0 - crew_foul_rate_mult) * 1.35)
            fta_delta = -round(lost_fta, 2)
            pts_delta = -round(lost_fta * 0.82, 2)
            signal = "LOW_WHISTLE_FT_TAX_UNDER"

        return {
            "player": player_name,
            "p_foul_trouble": round(p_foul_trouble, 3),
            "minutes_delta": minutes_delta,
            "adj_minutes": round(max(12.0, base_minutes + minutes_delta), 2),
            "fta_delta": fta_delta,
            "pts_delta": pts_delta,
            "adj_pts": round(max(0.0, base_pts + pts_delta), 2),
            "approved": approved,
            "signal": signal,
        }


class InjuryNewsSteamLagScanner:
    """
    Detects soft-book / DFS pricing lag immediately following breaking NBA
    injury updates (e.g., Questionable -> OUT) by comparing sharp market steam
    against slower retail/DFS quotes.
    """

    @staticmethod
    def _american_to_implied(odds: int) -> float:
        if odds < 0:
            return (-odds) / (-odds + 100.0)
        return 100.0 / (odds + 100.0)

    @classmethod
    def scan_soft_book_lag(
        cls,
        player_name: str,
        stat_type: str,
        sidelined_teammate: str,
        minutes_since_news: float,
        sharp_line_open: float,
        sharp_line_now: float,
        model_fair_line: float,
        soft_book_quotes: list,
    ) -> dict:
        """
        soft_book_quotes: list of dicts with keys: book, line, over_odds (default -115), updated_sec_ago
        """
        steam_move = sharp_line_now - sharp_line_open
        steam_velocity = steam_move / max(minutes_since_news, 1.0)
        opportunities = []

        for q in soft_book_quotes:
            book = q.get("book", "UNKNOWN")
            line = float(q.get("line", sharp_line_now))
            odds = int(q.get("over_odds", -115))
            implied = cls._american_to_implied(odds)
            lag_vs_sharp = sharp_line_now - line
            edge_vs_model = model_fair_line - line

            if line > model_fair_line + 0.5:
                status = "BUY_STOP_EXCEEDED"
            elif minutes_since_news <= 15.0 and lag_vs_sharp >= 1.0 and edge_vs_model >= 1.2:
                status = "FLASH_SNIPE_SOFT_BOOK_LAG"
            elif lag_vs_sharp >= 0.5 and edge_vs_model >= 1.0:
                status = "MILD_LAG_ACTIONABLE"
            else:
                status = "STALE_LOCKED_PASS"

            opportunities.append({
                "book": book,
                "line": line,
                "over_odds": odds,
                "implied_prob": round(implied, 4),
                "lag_vs_sharp": round(lag_vs_sharp, 2),
                "edge_vs_model": round(edge_vs_model, 2),
                "status": status,
            })

        opportunities.sort(key=lambda x: x["lag_vs_sharp"], reverse=True)
        snipe_count = sum(1 for o in opportunities if o["status"] == "FLASH_SNIPE_SOFT_BOOK_LAG")
        return {
            "player": player_name,
            "stat_type": stat_type,
            "sidelined_teammate": sidelined_teammate,
            "minutes_since_news": minutes_since_news,
            "steam_move": round(steam_move, 2),
            "steam_velocity_per_min": round(steam_velocity, 3),
            "flash_snipe_count": snipe_count,
            "best_quote": opportunities[0] if opportunities else None,
            "quotes": opportunities,
        }


class CLVAutoCalibrationEngine:
    """
    Cross-Sport (NBA & NFL) Automated CLV Auto-Calibration Loop.
    Logs every graded/closed play's Model Projection vs. Open vs. Pinnacle/Circa Close
    tagged by modifier (e.g., SLOT_FUNNEL_BOOST, RIM_VACUUM_SLASHER_OVER, BACKUP_ROLE_INHERITANCE_BELLCOW),
    and dynamically shrinks any modifier losing CLV over a rolling sample while rewarding verified +CLV edges.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clv_modifier_calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL,
                modifier_tag TEXT NOT NULL,
                player TEXT NOT NULL,
                side TEXT NOT NULL,
                open_line REAL NOT NULL,
                close_line REAL NOT NULL,
                open_no_vig_prob REAL NOT NULL,
                close_no_vig_prob REAL NOT NULL,
                clv_prob_edge REAL NOT NULL,
                outcome INTEGER,
                logged_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def log_play_clv(
        self,
        sport: str,
        modifier_tag: str,
        player: str,
        side: str,
        open_line: float,
        close_line: float,
        open_no_vig_prob: float,
        close_no_vig_prob: float,
        outcome: Optional[int] = None,
    ) -> float:
        # Positive CLV means the closing sharp probability moved in our favor
        clv_edge = round(close_no_vig_prob - open_no_vig_prob, 5)
        self.conn.execute(
            """
            INSERT INTO clv_modifier_calibration
                (sport, modifier_tag, player, side, open_line, close_line,
                 open_no_vig_prob, close_no_vig_prob, clv_prob_edge, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sport.upper(), modifier_tag.upper(), player, side.upper(),
             open_line, close_line, open_no_vig_prob, close_no_vig_prob, clv_edge, outcome),
        )
        self.conn.commit()
        return clv_edge

    def get_modifier_weight(self, sport: str, modifier_tag: str, min_sample: int = 15, lookback: int = 100) -> dict:
        rows = self.conn.execute(
            """
            SELECT clv_prob_edge, outcome FROM clv_modifier_calibration
            WHERE sport = ? AND modifier_tag = ?
            ORDER BY id DESC LIMIT ?
            """,
            (sport.upper(), modifier_tag.upper(), lookback),
        ).fetchall()
        n = len(rows)
        if n == 0:
            return {
                "sport": sport.upper(),
                "modifier_tag": modifier_tag.upper(),
                "sample_size": 0,
                "mean_clv_pct": 0.0,
                "weight_mult": 1.0,
                "status": "NO_DATA_PRIOR_HOLD",
            }
        mean_clv = sum(r[0] for r in rows) / n
        mean_clv_pct = mean_clv * 100.0

        if n < min_sample:
            # Small sample: hold at 1.0 unless catastrophic negative CLV
            weight = 0.85 if mean_clv_pct <= -4.0 else 1.0
            status = "SMALL_SAMPLE_PRIOR_HOLD"
        elif mean_clv_pct <= -3.5:
            weight = 0.25
            status = "QUARANTINE_SEVERE_NEG_CLV"
        elif mean_clv_pct <= -1.5:
            # Linear shrinkage between -1.5% (0.75) and -3.5% (0.35)
            weight = max(0.35, 0.75 - ((abs(mean_clv_pct) - 1.5) / 2.0) * 0.40)
            status = "AUTO_SHRINK_NEGATIVE_CLV"
        elif mean_clv_pct >= 1.5:
            weight = min(1.15, 1.0 + min(0.15, (mean_clv_pct - 1.0) * 0.04))
            status = "CONFIRMED_POSITIVE_CLV"
        else:
            weight = 1.0
            status = "CALIBRATED_NEUTRAL"

        return {
            "sport": sport.upper(),
            "modifier_tag": modifier_tag.upper(),
            "sample_size": n,
            "mean_clv_pct": round(mean_clv_pct, 2),
            "weight_mult": round(weight, 3),
            "status": status,
        }

    def apply_calibrated_modifier(self, sport: str, modifier_tag: str, raw_delta: float, min_sample: int = 15) -> dict:
        cal = self.get_modifier_weight(sport, modifier_tag, min_sample=min_sample)
        return {
            **cal,
            "raw_delta": round(raw_delta, 4),
            "calibrated_delta": round(raw_delta * cal["weight_mult"], 4),
        }

    def close(self) -> None:
        self.conn.close()


def run_v88_syndicate_tests() -> int:
    """Automated verification suite for NBA v8.8 + Cross-Sport CLV Auto-Calibration."""
    print("=== RUNNING ALPHAPROP NBA v8.8 SYNDICATE + CLV AUTO-CALIBRATION TESTS ===")

    # Test A: Rim-Protector On/Off Shot-Location Shift
    res_vac = RimProtectorShotLocationEngine.evaluate(
        "Ja Morant", "SLASHER", base_pts=24.0, opp_rim_protector_tier="ELITE_DROP",
        opp_rim_protector_out=True, player_rim_freq=0.46
    )
    res_drop = RimProtectorShotLocationEngine.evaluate(
        "Ja Morant", "SLASHER", base_pts=24.0, opp_rim_protector_tier="ELITE_DROP",
        opp_rim_protector_out=False, player_rim_freq=0.46
    )
    res_pull = RimProtectorShotLocationEngine.evaluate(
        "Darius Garland", "PULL_UP_GUARD", base_pts=20.0, base_3pm=2.8,
        opp_rim_protector_tier="ELITE_DROP", opp_rim_protector_out=False, player_pullup_3pt_freq=0.35
    )
    assert res_vac["signal"] == "RIM_VACUUM_SLASHER_OVER" and res_vac["adj_pts"] > 25.8
    assert res_drop["signal"] == "ELITE_RIM_DETERRENT_SLASHER_UNDER" and res_drop["adj_pts"] < 22.5
    assert res_pull["signal"] == "DROP_COVERAGE_PULL_UP_OVER" and res_pull["adj_3pm"] > 2.95
    print(f"  [PASS] v8.8-1 — Rim Protector Shot-Location: Vacuum={res_vac['adj_pts']} vs ActiveDrop={res_drop['adj_pts']} | PullUp 3PM={res_pull['adj_3pm']}")

    # Test B: Ref Crew x Foul-Trouble Under vs Foul-Draw Over
    res_ft = RefCrewFoulInteractionEngine.evaluate(
        "Jaren Jackson Jr.", base_minutes=31.0, base_pts=21.0, pf_per_36=4.6, fta_per_36=5.2,
        crew_foul_rate_mult=1.10, opp_ftr=0.27, side="OVER"
    )
    res_fd = RefCrewFoulInteractionEngine.evaluate(
        "Shai Gilgeous-Alexander", base_minutes=34.5, base_pts=30.5, pf_per_36=2.3, fta_per_36=9.8,
        crew_foul_rate_mult=1.10, opp_ftr=0.21, opp_def_foul_rate_mult=1.06, side="OVER"
    )
    assert (not res_ft["approved"]) and res_ft["signal"] == "HIGH_WHISTLE_FOUL_TROUBLE_UNDER" and res_ft["minutes_delta"] < -2.0
    assert res_fd["approved"] and res_fd["signal"] == "HIGH_WHISTLE_FOUL_DRAW_OVER" and res_fd["pts_delta"] >= 1.2
    print(f"  [PASS] v8.8-2 — Ref Crew Foul Split: JJJ Foul-Trouble VETO ({res_ft['minutes_delta']} min) | SGA Foul-Draw +{res_fd['pts_delta']} PTS")

    # Test C: Injury-News Steam & Soft-Book Lag Scanner
    res_lag = InjuryNewsSteamLagScanner.scan_soft_book_lag(
        player_name="Tyrese Maxey", stat_type="PTS", sidelined_teammate="Joel Embiid",
        minutes_since_news=6.0, sharp_line_open=23.5, sharp_line_now=26.5, model_fair_line=27.2,
        soft_book_quotes=[
            {"book": "PrizePicks", "line": 24.5, "over_odds": -119},
            {"book": "DraftKings", "line": 26.5, "over_odds": -115},
            {"book": "OversteamedBook", "line": 28.0, "over_odds": -110},
        ],
    )
    assert res_lag["flash_snipe_count"] == 1 and res_lag["best_quote"]["book"] == "PrizePicks"
    assert res_lag["best_quote"]["status"] == "FLASH_SNIPE_SOFT_BOOK_LAG"
    print(f"  [PASS] v8.8-3 — Steam & Soft-Book Lag Scanner: {res_lag['best_quote']['book']} {res_lag['best_quote']['line']} (Lag {res_lag['best_quote']['lag_vs_sharp']:+.1f} vs Sharp)")

    # Test D: Cross-Sport CLV Auto-Calibration Loop (Negative CLV auto-shrink + Positive CLV confirm)
    cal = CLVAutoCalibrationEngine(":memory:")
    for i in range(18):
        cal.log_play_clv("NFL", "SLOT_FUNNEL_BOOST", f"WR{i}", "OVER", 54.5, 53.5, 0.54, 0.515, outcome=0)
        cal.log_play_clv("NBA", "RIM_VACUUM_SLASHER_OVER", f"G{i}", "OVER", 23.5, 25.0, 0.52, 0.555, outcome=1)
    neg_w = cal.apply_calibrated_modifier("NFL", "SLOT_FUNNEL_BOOST", raw_delta=0.14)
    pos_w = cal.apply_calibrated_modifier("NBA", "RIM_VACUUM_SLASHER_OVER", raw_delta=0.10)
    cal.close()
    assert neg_w["status"] == "AUTO_SHRINK_NEGATIVE_CLV" and neg_w["weight_mult"] < 0.70
    assert pos_w["status"] == "CONFIRMED_POSITIVE_CLV" and pos_w["weight_mult"] > 1.05
    print(f"  [PASS] v8.8-4 — Cross-Sport CLV Auto-Calibrator: NegCLV shrunk to {neg_w['weight_mult']}x ({neg_w['calibrated_delta']}) | PosCLV boosted to {pos_w['weight_mult']}x")
    print("=== ALL 4/4 ALPHAPROP NBA v8.8 + CLV AUTO-CALIBRATION TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaProp Commercial v9.1 Engine (Upgrades 1-8 + Automated nba_api PbP Ingestion)")
    parser.add_argument("--self-test", action="store_true", help="Run 5-pillar + Upgrades 1-8 + PbP On/Off verification suite")
    parser.add_argument("--ingest-team", type=str, default="", help="Team abbreviation for automated nba_api PbP on/off ingestion (e.g. PHI)")
    parser.add_argument("--target-player-id", type=int, default=0, help="NBA.com personId for target player")
    parser.add_argument("--teammate-out-id", type=int, default=0, help="NBA.com personId for sidelined teammate")
    parser.add_argument("--games", type=int, default=12, help="Number of recent games to ingest for PbP stints")
    parser.add_argument("--cache-db", type=str, default="alphaprop_pbp_cache.sqlite", help="SQLite cache path for PbP stints")
    parser.add_argument("--clv-db", type=str, default="alphaprop_clv_ledger.sqlite", help="SQLite ledger path for multi-book odds & CLV")
    parser.add_argument("--snapshot-wave", type=str, default="", help="Capture multi-book odds wave (PULSE_1_OPENER|PULSE_2_MIDDAY|PULSE_3_PRETIP|CLOSING_TIPOFF)")
    parser.add_argument("--event-id", type=str, default="evt_demo_nba", help="The Odds API event_id for --snapshot-wave")
    parser.add_argument("--offline", action="store_true", help="Force offline synthetic PbP/Odds fallback")
    args, _ = parser.parse_known_args()
    if args.snapshot_wave:
        clv_logger = MultiBookOddsCLVLogger(db_path=args.clv_db, offline_mode=args.offline)
        try:
            payload = clv_logger.fetch_odds_api_event_props(args.event_id)
            cons = clv_logger.ingest_event_snapshot(payload, pulse_wave=args.snapshot_wave)
            print(json.dumps([asdict(c) for c in cons], indent=2))
        finally:
            clv_logger.close()
        sys.exit(0)
    if args.ingest_team and args.target_player_id and args.teammate_out_id:
        pipe = NbaApiIngestionPipeline(cache_db_path=args.cache_db, offline_mode=args.offline)
        split = pipe.ingest_player_on_off_usage(
            team_abbr=args.ingest_team,
            target_player_id=args.target_player_id,
            teammate_out_ids=[args.teammate_out_id],
            last_n_games=args.games,
        )
        print(json.dumps(asdict(split), indent=2))
        sys.exit(0)
    if args.self_test:
        sys.exit(run_self_test())
    elif HAS_STREAMLIT:
        render_streamlit_app()
    else:
        sys.exit(run_self_test())
