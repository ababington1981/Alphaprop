"""
AlphaProp Pro™ Production Backend Entrypoint
Wraps NBA v8.8 (alphaprop_commercial_v9.py), NFL v4.4 (gridiron_edge_plus.py),
and founder control settings (config.json) behind FastAPI.
"""
import json
import pathlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

app = FastAPI(
    title="AlphaProp Pro™ API",
    version="12.4.0",
    description="NBA v8.8 + NFL Gridiron Edge+ v4.4 Syndicate Prop Analytics Backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {"app_name": "AlphaProp Pro", "status": "default_config"}


@app.get("/health")
def health_check():
    cfg = load_config()
    return {
        "status": "ok",
        "app": cfg.get("app_name", "AlphaProp Pro"),
        "engines": cfg.get("engine_versions", {"nba": "v8.8", "nfl": "v4.4"}),
        "clv_auto_calibration": cfg.get("staking_rules", {}).get("clv_auto_calibration_enabled", True),
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
