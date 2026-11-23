from pathlib import Path
#!/usr/bin/env python3
"""
AlphaProp Pro™ — 60-Second NBA Injury & Odds API Live Poller
==============================================================
Polls:
  1. ESPN NBA Injury Feed (live player status & transitions)
  2. The Odds API v4 (NBA player props across us, us2, us_dfs regions)
  3. Optional PrizePicks Public Projections fallback (league_id=7)

Flags:
  - 1.0+ point Line-Lag (cross-book / DFS vs. sharp sportsbook discrepancy, or vs. opener)
  - 🚦 Buy-Stop Guard™ breaches (when a steamed line exceeds maxPlayableLine)
  - 🫧 Bubble Watchlist Auto-Promotions (0.00u staged plays -> Official VIP when trigger player is OUT)

Usage:
  export ODDS_API_KEY="your_the_odds_api_key"
  python3 alphaprop_live_poller.py                # Poll every 60s continuously
  python3 alphaprop_live_poller.py --once         # Single poll cycle
  python3 alphaprop_live_poller.py --once --mock  # Offline self-test with sample data
"""

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

def _load_dotenv_if_present() -> None:
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    try:
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ and v:
                os.environ[k] = v
    except Exception:
        pass


_load_dotenv_if_present()

DISABLE_LIVE_ODDS_API = (
    os.environ.get("DISABLE_LIVE_ODDS_API", "true").strip().lower() in ("1", "true", "yes", "on")
    or os.environ.get("ALPHAPROP_TEST_MODE", "").strip().lower() in ("1", "true", "yes", "on")
)
ODDS_API_KEY = "" if DISABLE_LIVE_ODDS_API else os.environ.get("ODDS_API_KEY", "").strip()
UNABATED_API_KEY = "" if DISABLE_LIVE_ODDS_API else os.environ.get("UNABATED_API_KEY", "").strip()
UNABATED_API_BASE = os.environ.get("UNABATED_API_BASE_URL", "https://data.unabated.com").rstrip("/")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLATE_JSON_PATH = os.environ.get("SLATE_JSON_PATH", os.path.join(BASE_DIR, "today_slate.json"))
WATCHLIST_JSON_PATH = os.environ.get("WATCHLIST_JSON_PATH", os.path.join(BASE_DIR, "bubble_watchlist.json"))
ALERTS_LOG_PATH = os.environ.get("ALERTS_LOG_PATH", os.path.join(BASE_DIR, "poller_alerts.json"))
STATE_PATH = os.environ.get("POLLER_STATE_PATH", os.path.join(BASE_DIR, "poller_state.json"))

ESPN_INJURIES_URLS = {
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries",
}
ODDS_API_BASES = {
    "nba": "https://api.the-odds-api.com/v4/sports/basketball_nba",
    "nfl": "https://api.the-odds-api.com/v4/sports/americanfootball_nfl",
}
SPORT_PROP_MARKETS = {
    "nba": "player_points,player_rebounds,player_assists,player_threes,player_points_rebounds_assists,player_points_assists,player_points_rebounds",
    "nfl": "player_pass_yds,player_rush_yds,player_reception_yds,player_receptions,player_pass_tds,player_anytime_td",
}
ESPN_INJURIES_URL = ESPN_INJURIES_URLS["nba"]
ODDS_API_BASE = ODDS_API_BASES["nba"]

def _autoload_dotenv(env_path: str | None = None) -> dict[str, str]:
    loaded = {}
    p = Path(env_path) if env_path else Path(__file__).resolve().parent / ".env"
    if not p.exists():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        loaded[k] = v
        if k not in os.environ or not os.environ[k].strip():
            os.environ[k] = v
    return loaded

_autoload_dotenv()
PRIZEPICKS_URL = "https://api.prizepicks.com/projections?league_id=7&per_page=250&single_stat=true"

MARKET_TO_STAT = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "player_threes": "3PM",
    "player_points_rebounds_assists": "PRA",
    "player_points_assists": "PTS+AST",
    "player_points_rebounds": "PTS+REB",
    "player_rebounds_assists": "REB+AST",
}

PP_STAT_MAP = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "3-PT Made": "3PM",
    "Pts+Rebs+Asts": "PRA",
    "Pts+Asts": "PTS+AST",
    "Pts+Rebs": "PTS+REB",
    "Rebs+Asts": "REB+AST",
}

SHARP_BOOKS = {"fanduel", "draftkings", "betmgm", "caesars", "williamhill_us", "espnbet", "betrivers"}
DFS_BOOKS = {"prizepicks", "underdog", "sleeper", "betr_us_dfs", "pick6"}

RUNNING = True


def _handle_sigint(signum: int, frame: Any) -> None:
    global RUNNING
    RUNNING = False
    print("\n[INFO] Shutdown signal received. Finishing current cycle...")


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_name(name: str) -> str:
    return " ".join(name.lower().replace(".", "").replace("'", "").split())


def fetch_json(url: str, headers: dict[str, str] | None = None, timeout: int = 15) -> tuple[Any, dict[str, str]]:
    req = urllib.request.Request(
        url,
        headers=headers or {"User-Agent": "AlphaPropPro-Poller/11.2"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            data = json.loads(resp.read().decode("utf-8"))
            return data, resp_headers
    except urllib.error.HTTPError as exc:
        print(f"[WARN] HTTP {exc.code} fetching {url.split('?')[0]}: {exc.reason}")
        return None, {}
    except Exception as exc:
        print(f"[WARN] Failed fetching {url.split('?')[0]}: {exc}")
        return None, {}


def post_webhook(webhook_url: str, message: str) -> None:
    if not webhook_url:
        return
    payload = json.dumps({"content": message, "text": message}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "AlphaPropPro-Poller/11.2"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as exc:
        print(f"[WARN] Webhook delivery failed: {exc}")


def load_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json_file(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Data Fetchers (Live + Mock)
# ---------------------------------------------------------------------------

def fetch_espn_injuries(mock: bool = False) -> dict[str, dict[str, str]]:
    """
    Returns dict keyed by normalized player name:
      { "player": str, "team": str, "status": str, "detail": str, "updated": str }
    """
    if mock:
        raw = {
            "injuries": [
                {
                    "displayName": "Indiana Pacers",
                    "injuries": [
                        {
                            "athlete": {"displayName": "Andrew Nembhard"},
                            "status": "Out",
                            "shortComment": "Nembhard (knee) ruled out for tonight's game.",
                            "date": utc_now_iso(),
                        }
                    ],
                },
                {
                    "displayName": "Toronto Raptors",
                    "injuries": [
                        {
                            "athlete": {"displayName": "Scottie Barnes"},
                            "status": "Probable",
                            "shortComment": "Barnes (ankle) upgraded to probable.",
                            "date": utc_now_iso(),
                        }
                    ],
                },
            ]
        }
    else:
        raw, _ = fetch_json(ESPN_INJURIES_URL)
        if not isinstance(raw, dict):
            return {}

    parsed: dict[str, dict[str, str]] = {}
    for team_block in raw.get("injuries", []):
        team_name = team_block.get("displayName", "Unknown")
        for item in team_block.get("injuries", []):
            athlete = item.get("athlete") or {}
            player_name = (athlete.get("displayName") or "").strip()
            if not player_name:
                continue
            status = (item.get("status") or "Unknown").strip()
            detail = (item.get("shortComment") or item.get("longComment") or "").strip()
            parsed[normalize_name(player_name)] = {
                "player": player_name,
                "team": team_name,
                "status": status,
                "detail": detail,
                "updated": item.get("date") or utc_now_iso(),
            }
    return parsed


_LAST_UNABATED_ODDS_TS: float = 0.0


def fetch_unabated_props(
    api_key: str,
    sport: str = "nba",
    mock: bool = False,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, str]]:
    """
    Poll Unabated REST API (https://data.unabated.com/market/{sport}/props/odds)
    authenticating with the X-Api-Key header and honoring the 5-second shared odds rate limit.
    """
    global _LAST_UNABATED_ODDS_TS
    if mock:
        return fetch_odds_api_props(api_key="", mock=True)
    if not api_key:
        return {}, {"remaining": "NO_UNABATED_KEY", "used": "0"}

    # Enforce Unabated shared 5-second odds request spacing
    now = time.monotonic()
    elapsed = now - _LAST_UNABATED_ODDS_TS
    if _LAST_UNABATED_ODDS_TS > 0 and elapsed < 5.05:
        time.sleep(5.05 - elapsed)
    _LAST_UNABATED_ODDS_TS = time.monotonic()

    league = "nfl" if sport == "nfl" else "nba"
    url = f"{UNABATED_API_BASE}/market/{league}/props/odds"
    req = urllib.request.Request(
        url,
        headers={
            "X-Api-Key": api_key,
            "Accept": "application/json",
            "User-Agent": "AlphaPropPro-LivePoller/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"[WARN] Unabated API HTTP {exc.code} on /market/{league}/props/odds: {exc.reason}")
        return {}, {"remaining": f"HTTP_{exc.code}", "used": "1"}
    except Exception as exc:
        print(f"[WARN] Unabated API request failed: {exc}")
        return {}, {"remaining": "ERR", "used": "1"}

    if isinstance(payload, dict) and payload.get("success") is False:
        msgs = payload.get("messages") or [payload.get("message", "Unknown Unabated error")]
        print(f"[WARN] Unabated API envelope error: {'; '.join(str(m) for m in msgs)}")
        return {}, {"remaining": "ENVELOPE_ERR", "used": "1"}

    props_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    odds_root = (payload.get("data") or {}).get("odds") or {}
    events = odds_root.get("events") if isinstance(odds_root, dict) else []
    if not isinstance(events, list):
        events = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        matchup = f"{ev.get('awayTeam', '')} @ {ev.get('homeTeam', '')}".strip(" @")
        for m in (ev.get("markets") or []):
            if not isinstance(m, dict):
                continue
            stat = MARKET_TO_STAT.get(str(m.get("marketKey", "")), str(m.get("marketKey", "")).upper())
            player = str(m.get("playerName", "")).strip()
            point = m.get("point")
            if not player or point is None:
                continue
            key = (normalize_name(player), stat)
            books = m.get("books") or {}
            if isinstance(books, dict):
                for bkey, bval in books.items():
                    if not isinstance(bval, dict):
                        continue
                    over_p = bval.get("overPrice")
                    under_p = bval.get("underPrice")
                    props_by_key.setdefault(key, []).append({
                        "player": player,
                        "stat": stat,
                        "matchup": matchup,
                        "book_key": str(bkey).lower(),
                        "book_title": str(bkey).title(),
                        "line": float(point),
                        "over_odds": int(over_p) if over_p is not None else -110,
                        "under_odds": int(under_p) if under_p is not None else -110,
                    })

    return props_by_key, {"remaining": "UNABATED_OK", "used": "1"}


def fetch_odds_api_props(
    api_key: str,
    regions: str = "us,us2,us_dfs",
    max_events: int = 12,
    mock: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """
    Fetches NBA player props from The Odds API v4.
    Returns:
      - props_by_key: { "<norm_player>::<STAT>": [ {player, stat, book_key, book_title, line, over_price, under_price}, ... ] }
      - quota_info: { "remaining": str, "used": str }
    """
    if mock:
        sample = [
            {
                "id": "mock_ind_orl",
                "home_team": "Orlando Magic",
                "away_team": "Indiana Pacers",
                "commence_time": utc_now_iso(),
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "markets": [
                            {
                                "key": "player_points_rebounds_assists",
                                "outcomes": [
                                    {"name": "Over", "description": "Pascal Siakam", "point": 32.5, "price": -115},
                                    {"name": "Under", "description": "Pascal Siakam", "point": 32.5, "price": -115},
                                ],
                            },
                            {
                                "key": "player_points_assists",
                                "outcomes": [
                                    {"name": "Over", "description": "Tyrese Haliburton", "point": 29.5, "price": -112},
                                    {"name": "Under", "description": "Tyrese Haliburton", "point": 29.5, "price": -118},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "prizepicks",
                        "title": "PrizePicks",
                        "markets": [
                            {
                                "key": "player_points_rebounds_assists",
                                "outcomes": [
                                    {"name": "Over", "description": "Pascal Siakam", "point": 31.0, "price": -119},
                                ],
                            },
                            {
                                "key": "player_points_assists",
                                "outcomes": [
                                    {"name": "Over", "description": "Tyrese Haliburton", "point": 28.0, "price": -119},
                                ],
                            },
                        ],
                    },
                ],
            }
        ]
        return _parse_odds_events(sample), {"remaining": "MOCK", "used": "0"}

    if not api_key:
        return {}, {"remaining": "NO_API_KEY", "used": "0"}

    events_url = f"{ODDS_API_BASE}/events?apiKey={urllib.parse.quote(api_key)}"
    events, headers = fetch_json(events_url)
    quota = {
        "remaining": headers.get("x-requests-remaining", "?"),
        "used": headers.get("x-requests-used", "?"),
    }
    if not isinstance(events, list):
        return {}, quota

    # Filter out games that already started more than 3 hours ago
    now_ts = datetime.now(timezone.utc).timestamp()
    upcoming = []
    for ev in events:
        commence = ev.get("commence_time", "")
        try:
            ev_ts = datetime.fromisoformat(commence.replace("Z", "+00:00")).timestamp()
            if ev_ts >= now_ts - 10800:
                upcoming.append(ev)
        except Exception:
            upcoming.append(ev)

    markets_param = ",".join(MARKET_TO_STAT.keys())
    event_payloads = []
    for ev in upcoming[:max_events]:
        ev_id = ev.get("id")
        if not ev_id:
            continue
        odds_url = (
            f"{ODDS_API_BASE}/events/{ev_id}/odds"
            f"?apiKey={urllib.parse.quote(api_key)}"
            f"&regions={urllib.parse.quote(regions)}"
            f"&markets={urllib.parse.quote(markets_param)}"
            f"&oddsFormat=american"
        )
        data, ev_headers = fetch_json(odds_url)
        if ev_headers:
            quota["remaining"] = ev_headers.get("x-requests-remaining", quota["remaining"])
            quota["used"] = ev_headers.get("x-requests-used", quota["used"])
        if isinstance(data, dict):
            event_payloads.append(data)

    return _parse_odds_events(event_payloads), quota


def _parse_odds_events(event_payloads: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    props_by_key: dict[str, list[dict[str, Any]]] = {}
    for ev in event_payloads:
        matchup = f"{ev.get('away_team', '')} @ {ev.get('home_team', '')}".strip(" @")
        for bk in ev.get("bookmakers", []):
            bk_key = (bk.get("key") or "").lower()
            bk_title = bk.get("title") or bk_key
            for mkt in bk.get("markets", []):
                stat = MARKET_TO_STAT.get(mkt.get("key", ""))
                if not stat:
                    continue
                # Group Over/Under outcomes for the same player & point
                grouped: dict[tuple[str, float], dict[str, Any]] = {}
                for out in mkt.get("outcomes", []):
                    player = (out.get("description") or "").strip()
                    point = out.get("point")
                    side = (out.get("name") or "").lower()
                    price = out.get("price")
                    if not player or not isinstance(point, (int, float)):
                        continue
                    gk = (player, float(point))
                    if gk not in grouped:
                        grouped[gk] = {
                            "player": player,
                            "stat": stat,
                            "matchup": matchup,
                            "book_key": bk_key,
                            "book_title": bk_title,
                            "line": float(point),
                            "over_price": None,
                            "under_price": None,
                        }
                    if side == "over":
                        grouped[gk]["over_price"] = price
                    elif side == "under":
                        grouped[gk]["under_price"] = price

                for entry in grouped.values():
                    k = f"{normalize_name(entry['player'])}::{stat}"
                    props_by_key.setdefault(k, []).append(entry)
    return props_by_key


def fetch_prizepicks_lines(mock: bool = False) -> dict[str, dict[str, Any]]:
    if mock:
        return {}
    data, _ = fetch_json(
        PRIZEPICKS_URL,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    if not isinstance(data, dict):
        return {}

    players_by_id = {}
    for inc in data.get("included", []):
        if inc.get("type") == "new_player":
            players_by_id[inc.get("id")] = (inc.get("attributes") or {}).get("display_name", "")

    lines: dict[str, dict[str, Any]] = {}
    for item in data.get("data", []):
        attrs = item.get("attributes") or {}
        if attrs.get("odds_type", "standard") != "standard":
            continue
        stat = PP_STAT_MAP.get(attrs.get("stat_type", ""))
        if not stat:
            continue
        line_score = attrs.get("line_score")
        p_rel = ((item.get("relationships") or {}).get("new_player") or {}).get("data") or {}
        player = players_by_id.get(p_rel.get("id"), "").strip()
        if player and isinstance(line_score, (int, float)):
            key = f"{normalize_name(player)}::{stat}"
            lines[key] = {
                "player": player,
                "stat": stat,
                "book_key": "prizepicks",
                "book_title": "PrizePicks",
                "line": float(line_score),
                "over_price": -119,
                "under_price": -119,
            }
    return lines


# ---------------------------------------------------------------------------
# Detection Engines: Line-Lag, Buy-Stop, & Bubble Auto-Promotion
# ---------------------------------------------------------------------------

def detect_line_lags(
    props_by_key: dict[str, list[dict[str, Any]]],
    lag_threshold: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Compares lines across all books for each (player, stat) and flags >= 1.0 pt gaps.
    Prioritizes DFS vs. Sharp Sportsbook discrepancies when both exist.
    """
    alerts: list[dict[str, Any]] = []
    for key, entries in props_by_key.items():
        if len(entries) < 2:
            continue
        # Deduplicate by book_key keeping main line
        by_book: dict[str, dict[str, Any]] = {}
        for e in entries:
            if e["book_key"] not in by_book:
                by_book[e["book_key"]] = e
        if len(by_book) < 2:
            continue

        book_list = list(by_book.values())
        min_entry = min(book_list, key=lambda x: x["line"])
        max_entry = max(book_list, key=lambda x: x["line"])
        gap = round(max_entry["line"] - min_entry["line"], 2)

        if gap >= lag_threshold:
            is_dfs_vs_sharp = (
                (min_entry["book_key"] in DFS_BOOKS and max_entry["book_key"] in SHARP_BOOKS)
                or (max_entry["book_key"] in DFS_BOOKS and min_entry["book_key"] in SHARP_BOOKS)
            )
            alerts.append({
                "type": "LINE_LAG",
                "player": min_entry["player"],
                "stat": min_entry["stat"],
                "matchup": min_entry.get("matchup", ""),
                "gap": gap,
                "soft_side": f"OVER {min_entry['line']} ({min_entry['book_title']})",
                "sharp_side": f"UNDER {max_entry['line']} ({max_entry['book_title']})",
                "min_line": min_entry["line"],
                "min_book": min_entry["book_title"],
                "max_line": max_entry["line"],
                "max_book": max_entry["book_title"],
                "dfs_vs_sharp": is_dfs_vs_sharp,
                "timestamp": utc_now_iso(),
            })
    alerts.sort(key=lambda a: (a["dfs_vs_sharp"], a["gap"]), reverse=True)
    return alerts


def evaluate_board_and_watchlist(
    slate_data: dict[str, Any],
    watchlist_items: list[dict[str, Any]],
    injuries: dict[str, dict[str, str]],
    props_by_key: dict[str, list[dict[str, Any]]],
    lag_threshold: float = 1.0,
) -> list[dict[str, Any]]:
    """
    1. Checks tracked VIP board plays against live lines for 1.0+ pt lag or Buy-Stop Guard™ breaches.
    2. Checks 0.00u Bubble Watchlist plays for injury trigger auto-promotions.
    """
    alerts: list[dict[str, Any]] = []

    # --- A. Board Opener Lag & Buy-Stop Guard ---
    for play in slate_data.get("props", []):
        player = play.get("player", "")
        stat = play.get("stat", "")
        opener_line = play.get("line")
        max_playable = play.get("maxPlayableLine")
        side = (play.get("side") or "OVER").upper()
        if not player or not stat or not isinstance(opener_line, (int, float)):
            continue

        key = f"{normalize_name(player)}::{stat}"
        live_entries = props_by_key.get(key, [])
        if not live_entries:
            continue

        avg_live = round(sum(e["line"] for e in live_entries) / len(live_entries), 2)
        diff = round(abs(avg_live - float(opener_line)), 2)
        if diff >= lag_threshold:
            alerts.append({
                "type": "OPENER_LINE_LAG",
                "player": player,
                "stat": stat,
                "side": side,
                "opener_line": float(opener_line),
                "live_consensus": avg_live,
                "gap": diff,
                "timestamp": utc_now_iso(),
            })

        if isinstance(max_playable, (int, float)):
            best_live = min(e["line"] for e in live_entries) if side == "OVER" else max(e["line"] for e in live_entries)
            breached = (side == "OVER" and best_live > float(max_playable)) or (
                side == "UNDER" and best_live < float(max_playable)
            )
            if breached:
                alerts.append({
                    "type": "BUY_STOP_TRIGGERED",
                    "player": player,
                    "stat": stat,
                    "side": side,
                    "best_live_line": best_live,
                    "max_playable_line": float(max_playable),
                    "timestamp": utc_now_iso(),
                })

    # --- B. Bubble Watchlist Auto-Promotions ---
    for item in watchlist_items:
        if item.get("status") == "PROMOTED":
            continue
        trigger_player = item.get("trigger_player", "")
        if not trigger_player:
            continue
        allowed_triggers = {s.upper() for s in item.get("trigger_statuses", ["OUT", "DOUBTFUL"])}
        inj = injuries.get(normalize_name(trigger_player))
        if not inj:
            continue

        current_status = inj["status"].upper()
        if current_status in allowed_triggers:
            player = item.get("player", "")
            stat = item.get("stat", "")
            side = (item.get("side") or "OVER").upper()
            max_playable = item.get("maxPlayableLine")
            key = f"{normalize_name(player)}::{stat}"
            live_entries = props_by_key.get(key, [])

            best_line = item.get("line")
            best_book = "Watchlist Baseline"
            if live_entries:
                best_obj = (
                    min(live_entries, key=lambda x: x["line"])
                    if side == "OVER"
                    else max(live_entries, key=lambda x: x["line"])
                )
                best_line = best_obj["line"]
                best_book = best_obj["book_title"]

            buy_stop_ok = True
            if isinstance(max_playable, (int, float)) and isinstance(best_line, (int, float)):
                if (side == "OVER" and best_line > float(max_playable)) or (
                    side == "UNDER" and best_line < float(max_playable)
                ):
                    buy_stop_ok = False

            alerts.append({
                "type": "BUBBLE_PROMOTION",
                "player": player,
                "stat": stat,
                "side": side,
                "promoted_units": item.get("units_on_promote", 1.0),
                "trigger_player": inj["player"],
                "trigger_status": inj["status"],
                "injury_detail": inj["detail"],
                "live_best_line": best_line,
                "live_best_book": best_book,
                "max_playable_line": max_playable,
                "buy_stop_ok": buy_stop_ok,
                "timestamp": utc_now_iso(),
            })

    return alerts


def load_watchlist(slate_data: dict[str, Any], watchlist_path: str, mock: bool = False) -> list[dict[str, Any]]:
    # 1. Prefer dedicated watchlist file if present
    if os.path.exists(watchlist_path):
        data = load_json_file(watchlist_path, [])
        if isinstance(data, list):
            return data
    # 2. Fallback to bubbleWatchlist inside today_slate.json
    if isinstance(slate_data.get("bubbleWatchlist"), list):
        return slate_data["bubbleWatchlist"]
    # 3. In mock mode, supply a sample 0.00u Bubble Watchlist entry
    if mock:
        return [
            {
                "player": "Pascal Siakam",
                "stat": "PRA",
                "side": "OVER",
                "line": 31.0,
                "maxPlayableLine": 32.5,
                "units_on_promote": 1.0,
                "trigger_player": "Andrew Nembhard",
                "trigger_statuses": ["OUT"],
            },
            {
                "player": "Immanuel Quickley",
                "stat": "PTS+AST",
                "side": "OVER",
                "line": 23.5,
                "maxPlayableLine": 24.5,
                "units_on_promote": 1.0,
                "trigger_player": "Scottie Barnes",
                "trigger_statuses": ["OUT"],
            },
        ]
    return []


# ---------------------------------------------------------------------------
# Alert Formatting & Deduplication
# ---------------------------------------------------------------------------

def alert_dedupe_key(alert: dict[str, Any]) -> str:
    atype = alert["type"]
    if atype == "LINE_LAG":
        return f"LAG::{alert['player']}::{alert['stat']}::{alert['min_line']}::{alert['max_line']}"
    if atype == "BUBBLE_PROMOTION":
        return f"PROMO::{alert['player']}::{alert['stat']}::{alert['trigger_player']}::{alert['trigger_status']}"
    if atype == "INJURY_CHANGE":
        return f"INJ::{alert['player']}::{alert['new_status']}"
    if atype == "BUY_STOP_TRIGGERED":
        return f"STOP::{alert['player']}::{alert['stat']}::{alert['best_live_line']}"
    return f"{atype}::{alert.get('player', '')}::{alert.get('stat', '')}::{alert.get('gap', '')}"


def format_alert_line(alert: dict[str, Any]) -> str:
    atype = alert["type"]
    if atype == "LINE_LAG":
        tag = "[DFS vs SHARP]" if alert["dfs_vs_sharp"] else "[CROSS-BOOK]"
        return (
            f"⚡ LINE-LAG {tag} | {alert['player']} {alert['stat']} "
            f"({alert['gap']} pt gap): {alert['soft_side']} vs {alert['sharp_side']}"
        )
    if atype == "BUBBLE_PROMOTION":
        guard = "PASS" if alert["buy_stop_ok"] else f"HOLD (Past Buy-Stop {alert['max_playable_line']})"
        return (
            f"🫧 BUBBLE AUTO-PROMOTE | {alert['player']} {alert['side']} {alert['live_best_line']} {alert['stat']} "
            f"({alert['promoted_units']}u) @ {alert['live_best_book']} | "
            f"Trigger: {alert['trigger_player']} -> {alert['trigger_status']} | Buy-Stop: {guard}"
        )
    if atype == "INJURY_CHANGE":
        return (
            f"🏥 INJURY UPDATE | {alert['player']} ({alert['team']}): "
            f"{alert['old_status']} -> {alert['new_status']} ({alert['detail']})"
        )
    if atype == "BUY_STOP_TRIGGERED":
        return (
            f"🚦 BUY-STOP GUARD | {alert['player']} {alert['side']} {alert['stat']} "
            f"steamed to {alert['best_live_line']} (Ceiling: {alert['max_playable_line']})"
        )
    if atype == "OPENER_LINE_LAG":
        return (
            f"📈 OPENER LAG | {alert['player']} {alert['stat']} moved {alert['gap']} pts: "
            f"Opener {alert['opener_line']} -> Live {alert['live_consensus']}"
        )
    return json.dumps(alert)


# ---------------------------------------------------------------------------
# Poll Cycle
# ---------------------------------------------------------------------------

def run_poll_cycle(args: argparse.Namespace, state: dict[str, Any]) -> list[dict[str, Any]]:
    cycle_ts = utc_now_iso()
    injuries = fetch_espn_injuries(mock=args.mock)
    if args.unabated_key and not args.mock:
        props_by_key, quota = fetch_unabated_props(
            api_key=args.unabated_key,
            sport=args.sport,
            mock=args.mock,
        )
    else:
        props_by_key, quota = fetch_odds_api_props(
            api_key=args.api_key,
            regions=args.regions,
            max_events=args.max_events,
            mock=args.mock,
        )

    # Supplement with PrizePicks public projections if not already present from The Odds API
    pp_lines = fetch_prizepicks_lines(mock=args.mock)
    for k, entry in pp_lines.items():
        existing_books = {e["book_key"] for e in props_by_key.get(k, [])}
        if "prizepicks" not in existing_books:
            props_by_key.setdefault(k, []).append(entry)

    new_alerts: list[dict[str, Any]] = []

    # 1. Detect injury status transitions
    prev_injuries = state.get("injuries", {})
    for norm_name, info in injuries.items():
        old_status = (prev_injuries.get(norm_name) or {}).get("status")
        if old_status and old_status.upper() != info["status"].upper():
            new_alerts.append({
                "type": "INJURY_CHANGE",
                "player": info["player"],
                "team": info["team"],
                "old_status": old_status,
                "new_status": info["status"],
                "detail": info["detail"],
                "timestamp": cycle_ts,
            })
    state["injuries"] = injuries

    # 2. Detect 1.0+ pt cross-book / DFS vs. sharp line-lags
    new_alerts.extend(detect_line_lags(props_by_key, lag_threshold=args.lag_threshold))

    # 3. Evaluate tracked board & 0.00u Bubble Watchlist auto-promotions
    slate_data = load_json_file(args.slate_json, {})
    watchlist_items = load_watchlist(slate_data, args.watchlist, mock=args.mock)
    new_alerts.extend(
        evaluate_board_and_watchlist(
            slate_data=slate_data,
            watchlist_items=watchlist_items,
            injuries=injuries,
            props_by_key=props_by_key,
            lag_threshold=args.lag_threshold,
        )
    )

    # 4. Deduplicate & emit
    seen_keys = set(state.get("seen_alerts", []))
    emitted: list[dict[str, Any]] = []
    for alert in new_alerts:
        dkey = alert_dedupe_key(alert)
        if dkey in seen_keys:
            continue
        seen_keys.add(dkey)
        emitted.append(alert)

    state["seen_alerts"] = list(seen_keys)[-500:]
    state["last_poll"] = cycle_ts

    print(
        f"[{cycle_ts}] Injuries: {len(injuries)} | Prop Markets: {len(props_by_key)} | "
        f"Odds API Quota Remaining: {quota['remaining']} | New Alerts: {len(emitted)}"
    )
    for alert in emitted:
        line = format_alert_line(alert)
        print("  " + line)
        post_webhook(args.webhook_url, line)

    if emitted and not args.mock:
        existing_log = load_json_file(args.alerts_log, [])
        if not isinstance(existing_log, list):
            existing_log = []
        existing_log.extend(emitted)
        save_json_file(args.alerts_log, existing_log[-500:])

    if not args.mock:
        save_json_file(args.state_file, state)

    return emitted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaProp Pro™ 60s NBA Injury & Odds API Line-Lag / Bubble Promotion Poller"
    )
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds (default: 60)")
    parser.add_argument("--lag-threshold", type=float, default=1.0, help="Minimum point gap to flag (default: 1.0)")
    parser.add_argument("--regions", default="us,us2,us_dfs", help="The Odds API regions (default: us,us2,us_dfs)")
    parser.add_argument("--max-events", type=int, default=12, help="Max NBA games to poll per cycle (default: 12)")
    parser.add_argument("--api-key", default=ODDS_API_KEY, help="The Odds API key (or set ODDS_API_KEY env var)")
    parser.add_argument("--unabated-key", default=UNABATED_API_KEY, help="Unabated API key (or set UNABATED_API_KEY env var)")
    parser.add_argument("--webhook-url", default=WEBHOOK_URL, help="Optional Discord/Slack webhook URL")
    parser.add_argument("--slate-json", default=SLATE_JSON_PATH, help="Path to today_slate.json")
    parser.add_argument("--watchlist", default=WATCHLIST_JSON_PATH, help="Path to bubble_watchlist.json")
    parser.add_argument("--alerts-log", default=ALERTS_LOG_PATH, help="Path to output poller_alerts.json")
    parser.add_argument("--state-file", default=STATE_PATH, help="Path to poller_state.json")
    parser.add_argument("--sport", choices=["nba", "nfl"], default="nba", help="Sport to poll (nba or nfl)")
    parser.add_argument("--disable-odds-api", action="store_true", default=DISABLE_LIVE_ODDS_API, help="Disable Unabated & The Odds API calls for testing")
    parser.add_argument("--verify-keys", action="store_true", help="Verify .env / API key configuration & endpoint wiring")
    parser.add_argument("--once", action="store_true", help="Run a single poll cycle and exit")
    parser.add_argument("--mock", action="store_true", help="Run offline self-test with mock injury & odds data")
    return parser.parse_args()


def main() -> int:
    global ESPN_INJURIES_URL, ODDS_API_BASE
    args = parse_args()
    ESPN_INJURIES_URL = ESPN_INJURIES_URLS[args.sport]
    ODDS_API_BASE = ODDS_API_BASES[args.sport]

    if args.disable_odds_api:
        args.api_key = ""
        args.unabated_key = ""
    if args.verify_keys:
        odds_k = "" if args.disable_odds_api else (args.api_key or os.environ.get("ODDS_API_KEY", "")).strip()
        unab_k = "" if args.disable_odds_api else (args.unabated_key or os.environ.get("UNABATED_API_KEY", "")).strip()
        _mask = lambda k: (f"{k[:4]}...{k[-4:]} (len={len(k)})" if len(k) > 8 else f"EMPTY (len={len(k)})")
        active_provider = (
            f"Unabated API ({UNABATED_API_BASE}/market/{args.sport}/props/odds via X-Api-Key)"
            if unab_k else (f"The Odds API v4 ({ODDS_API_BASE})" if odds_k else "NONE (ESPN + PrizePicks public only)")
        )
        print(f"=== AlphaProp Multi-Sport API Verification ({args.sport.upper()}) ===")
        print(f"  Active Provider   : {active_provider}")
        print(f"  Unabated Endpoint : {UNABATED_API_BASE}/market/{args.sport}/props/odds")
        print(f"  ESPN Injuries URL : {ESPN_INJURIES_URL}")
        print(f"  Prop Markets      : {SPORT_PROP_MARKETS[args.sport]}")
        print(f"  UNABATED_API_KEY  : {_mask(unab_k)}")
        print(f"  ODDS_API_KEY      : {_mask(odds_k)} (optional fallback)")
        return 0

    if not args.mock and not args.unabated_key and not args.api_key:
        print("[WARN] Neither UNABATED_API_KEY nor ODDS_API_KEY is set. Only ESPN injuries and PrizePicks public lines will be polled.")
    elif not args.mock and args.unabated_key:
        print(f"[INFO] Active Live Odds Provider: Unabated ({UNABATED_API_BASE}/market/{args.sport}/props/odds via X-Api-Key)")

    state = {} if args.mock else load_json_file(args.state_file, {})
    if not isinstance(state, dict):
        state = {}

    while RUNNING:
        cycle_start = time.monotonic()
        run_poll_cycle(args, state)
        if args.once or not RUNNING:
            break
        elapsed = time.monotonic() - cycle_start
        sleep_for = max(1.0, args.interval - elapsed)
        for _ in range(int(sleep_for)):
            if not RUNNING:
                break
            time.sleep(1.0)
        rem = sleep_for - int(sleep_for)
        if RUNNING and rem > 0:
            time.sleep(rem)

    return 0


if __name__ == "__main__":
    sys.exit(main())
