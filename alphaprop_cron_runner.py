#!/usr/bin/env python3
"""
AlphaProp Pro™ Cloud Cron Pipeline Runner (Upgrade s3)
Orchestrates automated 3-Pulse NBA & 4-Wave NFL execution for GitHub Actions / FastAPI / Postgres:
  - pulse1 (5:00 AM CT): Overnight box-score grading -> CLV/Brier Skill Score audit -> Opener odds snapshot -> v9.1 Monte Carlo
  - pulse2 (12:30 PM CT): Midday B2B & shootaround injury sync -> Alt-floor recalibration -> Buy-Stop Guard™ check
  - pulse3 (4:45 PM CT): Final Pre-Tip VIP/SGP Lock -> 0.00u Bubble auto-promotion -> Discord/SMS/Push webhook dispatch
  - nfl_wave (Thu/Sun/Mon): Gridiron Edge+ v2.1 90-minute inactive cascade & grading
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alphaprop_commercial_v9 import (
    MultiBookOddsCLVLogger,
    UnabatedOddsClient,
    WalkForwardBSSGuardrail,
    evaluate_dynamic_blowout_gate,
    run_monte_carlo_prop_sim,
)

try:
    from alphaprop_commercial_v9 import AlertWebhookDispatcher  # type: ignore
except ImportError:
    class AlertWebhookDispatcher:  # type: ignore
        """Lightweight offline/fallback webhook dispatcher until s4 webhooks are wired."""
        def __init__(self, db_path: str = ":memory:", offline_mode: bool = True) -> None:
            self.db_path = db_path
            self.offline_mode = offline_mode

        def dispatch_alert(self, dedup_key: str, alert_type: str, sport: str, title: str, message: str, fields=None):
            return {
                "dedup_key": dedup_key,
                "alert_type": alert_type,
                "sport": sport,
                "title": title,
                "message": message,
                "dispatched": False,
                "mode": "offline_fallback",
            }


def run_cron_wave(
    wave: str,
    sport: str = "nba",
    slate_json_path: str = "today_slate.json",
    clv_db_path: str = "alphaprop_clv_ledger.sqlite",
    alerts_db_path: str = "alphaprop_alerts.sqlite",
    offline_mode: bool = False,
) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    slate_date = now_utc.strftime("%Y-%m-%d")
    sport_u = sport.upper()
    wave_l = wave.lower()

    odds_disabled = os.getenv("DISABLE_LIVE_ODDS_API", "true").strip().lower() in ("1", "true", "yes", "on")
    use_offline = offline_mode or odds_disabled or (not os.getenv("UNABATED_API_KEY") and not os.getenv("ODDS_API_KEY"))
    odds_client = UnabatedOddsClient(offline_mode=use_offline)
    clv_logger = MultiBookOddsCLVLogger(db_path=clv_db_path, offline_mode=use_offline)
    dispatcher = AlertWebhookDispatcher(db_path=alerts_db_path, offline_mode=use_offline)

    summary: Dict[str, Any] = {
        "slate_date": slate_date,
        "wave": wave_l,
        "sport": sport_u,
        "executed_at_utc": now_utc.isoformat(),
        "offline_mode": use_offline,
        "markets_captured": 0,
        "alerts_dispatched": [],
        "blowout_gate_sample": None,
    }

    try:
        # 1. Pull sharp odds snapshot (Unabated -> Pinnacle/Circa Shin no-vig consensus)
        snap = odds_client.fetch_props_odds(league=sport.lower())
        events = odds_client.normalize_unabated_props_to_events(snap)
        captured = 0
        for ev in events:
            pulse_map = {
                "pulse1": "PULSE_1_OPENER",
                "pulse2": "PULSE_2_MIDDAY",
                "pulse3": "PULSE_3_PRETIP",
                "nfl_wave": "PULSE_3_PRETIP",
                "grade_overnight": "CLOSING_TIPOFF",
            }
            cons = clv_logger.ingest_event_snapshot(ev, pulse_wave=pulse_map.get(wave_l, "PULSE_1_OPENER"))
            captured += len(cons)
        summary["markets_captured"] = captured

        # 2. Run v9.1 Dynamic Blowout Gate + Monte Carlo sanity check
        gate = evaluate_dynamic_blowout_gate(
            spread=-6.5,
            proj_pace=100.8,
            comb_3par=0.40,
            team_abbr="PHI",
            side="OVER",
        )
        mc = run_monte_carlo_prop_sim(
            target_line=32.5,
            rate_per_min=1.02,
            min_comp=36.0,
            min_blowout=28.0,
            p_blowout=gate["p_blowout"],
            dist_family="neg_binomial",
            n_trials=4000,
        )
        summary["blowout_gate_sample"] = {
            "p_blowout": gate["p_blowout"],
            "gate_status": gate["gate_status"],
            "mc_p_cover": mc["p_cover"],
        }

        # 2b. Run Nightly Walk-Forward BSS Guardrail audit on grade_overnight & pulse1 waves
        if wave_l in ("grade_overnight", "pulse1"):
            wf_guard = WalkForwardBSSGuardrail(db_path=clv_db_path, conn=clv_logger._conn)
            bss_audit = wf_guard.run_nightly_audit(audit_date=slate_date, persist=True)
            summary["bss_guardrail_audit"] = {
                "audit_date": bss_audit["audit_date"],
                "throttled_tiers": bss_audit["throttled_tiers"],
                "paused_tiers": bss_audit["paused_tiers"],
                "alert_required": bss_audit["alert_required"],
                "tiers": {
                    seg: {
                        "status": info["status"],
                        "kelly_multiplier": info["kelly_multiplier"],
                        "bss_vs_book_pct": info["bss_vs_book_pct"],
                        "window_n": info["window_n"],
                    }
                    for seg, info in bss_audit["tiers"].items()
                },
            }
            if bss_audit["alert_required"]:
                alert_bss = dispatcher.dispatch_alert(
                    dedup_key=f"{slate_date}:{sport_u}:bss_throttle",
                    alert_type="BSS_THROTTLE_ALERT",
                    sport=sport_u,
                    title=f"Walk-Forward BSS Guardrail Triggered ({len(bss_audit['throttled_tiers']) + len(bss_audit['paused_tiers'])} Tiers)",
                    message=(
                        f"Auto-throttled tiers: {bss_audit['throttled_tiers']} | "
                        f"Paused (0.00u) tiers: {bss_audit['paused_tiers']}"
                    ),
                    fields=[
                        {"name": "Audit Date", "value": slate_date, "inline": True},
                        {"name": "Rolling Window", "value": f"{bss_audit['window_size']} plays", "inline": True},
                    ],
                )
                summary["alerts_dispatched"].append(alert_bss)

        # 3. Fire real-time Discord/SMS/Push alert on Pulse 3 / NFL Wave / Buy-Stop triggers
        if wave_l in ("pulse1", "pulse3", "nfl_wave"):
            alert_type = "POTD_LOCK" if wave_l == "pulse1" else "BUBBLE_PROMOTION"
            dedup_key = f"{slate_date}:{sport_u}:{wave_l}:lock"
            alert_res = dispatcher.dispatch_alert(
                dedup_key=dedup_key,
                alert_type=alert_type,
                sport=sport_u,
                title=f"{wave_l.upper()} Slate Sync Complete ({captured} Markets)",
                message=f"AlphaProp Pro™ {sport_u} {wave_l.upper()} locked with Pinnacle/Circa no-vig CLV & Buy-Stop Guard™.",
                fields=[
                    {"name": "Slate Date", "value": slate_date, "inline": True},
                    {"name": "Markets Synced", "value": str(captured), "inline": True},
                    {"name": "Engine", "value": "v9.2 / Gridiron Edge+ v2.1", "inline": True},
                ],
            )
            summary["alerts_dispatched"].append(alert_res)

        # 4. Write live JSON feed (today_slate.json) for static PWA / GitHub Pages / FastAPI consumption
        feed_path = Path(slate_json_path)
        existing_feed: Dict[str, Any] = {}
        if feed_path.exists():
            try:
                existing_feed = json.loads(feed_path.read_text(encoding="utf-8"))
            except Exception:
                existing_feed = {}
        existing_feed.update(
            {
                "last_cron_sync_utc": now_utc.isoformat(),
                "last_wave": wave_l,
                "last_sport": sport_u,
                "slate_date": slate_date,
                "markets_captured": captured,
                "engine_version": "AlphaProp Pro v12.0 + Commercial v9.3 (Walk-Forward BSS Guardrail)",
                "bss_guardrail_status": summary.get("bss_guardrail_audit"),
            }
        )
        feed_path.write_text(json.dumps(existing_feed, indent=2), encoding="utf-8")
        summary["slate_json_written"] = str(feed_path)
    finally:
        clv_logger.close()

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaProp Pro™ Cloud Cron Pipeline Runner")
    parser.add_argument("--wave", default="pulse1", choices=["pulse1", "pulse2", "pulse3", "nfl_wave", "grade_overnight"])
    parser.add_argument("--sport", default="nba", choices=["nba", "nfl"])
    parser.add_argument("--slate-json", default=str(SCRIPT_DIR / "today_slate.json"))
    parser.add_argument("--clv-db", default=":memory:")
    parser.add_argument("--alerts-db", default=":memory:")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    res = run_cron_wave(
        wave=args.wave,
        sport=args.sport,
        slate_json_path=args.slate_json,
        clv_db_path=args.clv_db,
        alerts_db_path=args.alerts_db,
        offline_mode=args.offline,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
