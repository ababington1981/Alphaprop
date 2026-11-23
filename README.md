# AlphaProp Pro™ (NBA v8.8 + NFL Gridiron Edge+ v4.4)

Owner: **ababington1981**

## What Is Inside This Repo
- `index.html`, `manifest.json`, `sw.js` — Your live installable AlphaProp Pro™ subscriber web app.
- `config.json` — Plain-English control panel for unit caps, feature toggles, and promo banners.
- `backend/alphaprop_commercial_v9.py` — NBA v8.8 Syndicate Engine + FastAPI Paywall + Cross-Sport CLV Auto-Calibrator.
- `backend/gridiron_edge_plus.py` — NFL Gridiron Edge+ v4.4 Syndicate Engine + Shared CLV Auto-Calibrator.
- `backend/server.py` — FastAPI production server entrypoint.
- `supabase/supabase_rls_schema.sql` — One-time database security setup for VIP subscriber logins.
- `.github/workflows/alphaprop_cloud_cron.yml` — Automated daily Pulse 1 / Pulse 2 / Pulse 3 scheduler.

Start by opening **`SETUP_GUIDE.md`** for the click-by-click, zero-coding setup checklist.
