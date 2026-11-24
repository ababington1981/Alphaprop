"""
AlphaProp Pro™ Production Backend Entrypoint
Wraps NBA v8.8, NFL Gridiron Edge+ v4.4, founder control settings (config.json),
and live odds endpoints for The Odds API & Unabated API behind FastAPI.
"""
import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = ROOT.parent / "config.json"

app = FastAPI(
    title="AlphaProp Pro™ API",
    version="12.5.0",
    description="NBA v8.8 + NFL Gridiron Edge+ v4.4 Syndicate Prop Analytics Backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ODDS_API_SPORTS = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
}
_CACHE: dict = {}
CACHE_TTL_SECONDS = 60


def _is_disabled() -> bool:
    return os.getenv("DISABLE_LIVE_ODDS_API", "false").strip().lower() in ("1", "true", "yes", "on")


def _get_keys() -> tuple[str, str, str]:
    odds_key = os.getenv("ODDS_API_KEY", "").strip()
    unabated_key = os.getenv("UNABATED_API_KEY", "").strip()
    unabated_base = os.getenv("UNABATED_API_BASE_URL", "https://data.unabated.com").rstrip("/")
    return odds_key, unabated_key, unabated_base


def _http_get_json(url: str, headers: dict | None = None, timeout: int = 12) -> tuple[int, dict, object]:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "AlphaPropPro/12.5"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            try:
                data = json.loads(raw)
            except Exception:
                data = {"raw": raw[:300]}
            return resp.status, hdrs, data
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        return e.code, {}, {"error": f"HTTP {e.code}", "detail": body}
    except Exception as e:
        return 0, {}, {"error": str(e)}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {"app_name": "AlphaProp Pro", "status": "default_config"}


@app.get("/health")
def health_check():
    cfg = load_config()
    odds_key, unabated_key, _ = _get_keys()
    return {
        "status": "ok",
        "app": cfg.get("app_name", "AlphaProp Pro"),
        "engines": cfg.get("engine_versions", {"nba": "v8.8", "nfl": "v4.4"}),
        "clv_auto_calibration": cfg.get("staking_rules", {}).get("clv_auto_calibration_enabled", True),
        "live_odds_enabled": not _is_disabled(),
        "odds_api_configured": bool(odds_key),
        "unabated_api_configured": bool(unabated_key),
    }


@app.get("/api/config")
def get_public_config():
    cfg = load_config()
    return {
        "app_name": cfg.get("app_name"),
        "engine_versions": cfg.get("engine_versions"),
        "feature_toggles": cfg.get("feature_toggles"),
        "promo_banner": cfg.get("promo_banner"),
    }


@app.get("/api/odds/status")
def test_odds_providers(sport: str = Query("nba", pattern="^(nba|nfl)$")):
    """Safely verifies The Odds API and Unabated API keys & live responses."""
    odds_key, unabated_key, unabated_base = _get_keys()
    disabled = _is_disabled()
    result = {
        "sport": sport,
        "live_odds_enabled": not disabled,
        "the_odds_api": {"configured": bool(odds_key), "ok": False},
        "unabated_api": {"configured": bool(unabated_key), "ok": False},
    }
    if disabled:
        result["note"] = "DISABLE_LIVE_ODDS_API is true. Set DISABLE_LIVE_ODDS_API=false in Render Environment."
        return result

    if odds_key:
        sport_key = ODDS_API_SPORTS[sport]
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events?apiKey={urllib.parse.quote(odds_key)}"
        status, hdrs, data = _http_get_json(url)
        result["the_odds_api"].update({
            "http_status": status,
            "ok": status == 200,
            "events_found": len(data) if isinstance(data, list) else 0,
            "requests_remaining": hdrs.get("x-requests-remaining"),
            "requests_used": hdrs.get("x-requests-used"),
            "error": None if status == 200 else data,
        })

    if unabated_key:
        url = f"{unabated_base}/market/{sport}/props/odds"
        status, _, data = _http_get_json(url, headers={"X-Api-Key": unabated_key, "Accept": "application/json"})
        result["unabated_api"].update({
            "endpoint": url,
            "http_status": status,
            "ok": status == 200,
            "error": None if status == 200 else data,
        })

    return result


@app.get("/api/odds/live")
def get_live_odds(
    sport: str = Query("nba", pattern="^(nba|nfl)$"),
    provider: str = Query("auto", pattern="^(auto|unabated|odds_api)$"),
):
    """Returns 60s-cached live odds from Unabated API and/or The Odds API."""
    if _is_disabled():
        return {"ok": False, "error": "DISABLE_LIVE_ODDS_API is enabled in Render environment."}

    cache_key = f"{sport}:{provider}"
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
        return {**cached["payload"], "cached": True}

    odds_key, unabated_key, unabated_base = _get_keys()
    payload = {"ok": False, "sport": sport, "provider": None, "data": None, "cached": False}

    if provider in ("auto", "unabated") and unabated_key:
        url = f"{unabated_base}/market/{sport}/props/odds"
        status, _, data = _http_get_json(url, headers={"X-Api-Key": unabated_key, "Accept": "application/json"})
        if status == 200:
            payload.update({"ok": True, "provider": "unabated", "data": data})
            _CACHE[cache_key] = {"ts": now, "payload": payload}
            return payload
        payload["unabated_error"] = {"http_status": status, "detail": data}

    if provider in ("auto", "odds_api") and odds_key:
        sport_key = ODDS_API_SPORTS[sport]
        url = (
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
            f"?apiKey={urllib.parse.quote(odds_key)}&regions=us&markets=h2h,spreads,totals&oddsFormat=american"
        )
        status, hdrs, data = _http_get_json(url)
        if status == 200:
            payload.update({
                "ok": True,
                "provider": "the_odds_api",
                "requests_remaining": hdrs.get("x-requests-remaining"),
                "events_count": len(data) if isinstance(data, list) else 0,
                "data": data,
            })
            _CACHE[cache_key] = {"ts": now, "payload": payload}
            return payload
        payload["odds_api_error"] = {"http_status": status, "detail": data}

    return payload
