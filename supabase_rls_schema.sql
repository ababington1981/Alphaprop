-- ============================================================================
-- AlphaProp Pro™ Upgrade #5 — Supabase Auth, Subscriptions & Row-Level Security
-- Target: Supabase PostgreSQL 15+ (auth.users + PostgREST + FastAPI Backend)
-- Purpose: Eliminate client-side HTML VIP pick leaks by enforcing server-side
--          entitlements at BOTH the database (RLS) and API (FastAPI JWT) layers.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ----------------------------------------------------------------------------
-- 1. ENUM TYPES (Idempotent Creation)
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'subscription_tier') THEN
        CREATE TYPE public.subscription_tier AS ENUM ('free', 'vip', 'pro_quant', 'admin');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'subscription_status') THEN
        CREATE TYPE public.subscription_status AS ENUM (
            'inactive', 'trialing', 'active', 'past_due', 'canceled', 'expired'
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'billing_provider') THEN
        CREATE TYPE public.billing_provider AS ENUM ('stripe', 'revenuecat', 'manual_comp');
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 2. USER PROFILES (1:1 with Supabase auth.users)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.profiles (
    id                      UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email                   TEXT NOT NULL,
    display_name            TEXT,
    tier                    public.subscription_tier NOT NULL DEFAULT 'free',
    role                    TEXT NOT NULL DEFAULT 'subscriber' CHECK (role IN ('subscriber', 'admin')),
    stripe_customer_id      TEXT UNIQUE,
    revenuecat_app_user_id  TEXT UNIQUE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_profiles_tier_role ON public.profiles (tier, role);

-- Auto-provision a free profile whenever a new user signs up via Supabase Auth
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.profiles (id, email, display_name, tier, role)
    VALUES (
        NEW.id,
        COALESCE(NEW.email, ''),
        COALESCE(NEW.raw_user_meta_data->>'full_name', split_part(COALESCE(NEW.email, ''), '@', 1)),
        'free',
        'subscriber'
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Prevent client-side privilege escalation via direct PostgREST UPDATE on profiles
CREATE OR REPLACE FUNCTION public.prevent_profile_privilege_escalation()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    -- Allow service_role or internal triggers (when auth.uid() IS NULL) to mutate billing/role columns
    IF current_setting('request.jwt.claim.role', true) = 'service_role'
       OR auth.uid() IS NULL THEN
        NEW.updated_at := NOW();
        RETURN NEW;
    END IF;

    IF NEW.tier IS DISTINCT FROM OLD.tier
       OR NEW.role IS DISTINCT FROM OLD.role
       OR NEW.stripe_customer_id IS DISTINCT FROM OLD.stripe_customer_id
       OR NEW.revenuecat_app_user_id IS DISTINCT FROM OLD.revenuecat_app_user_id THEN
        RAISE EXCEPTION 'Permission denied: billing tier, role, and customer IDs are managed server-side.'
            USING ERRCODE = '42501';
    END IF;

    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_profile_escalation ON public.profiles;
CREATE TRIGGER trg_prevent_profile_escalation
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.prevent_profile_privilege_escalation();

-- ----------------------------------------------------------------------------
-- 3. AUTHORITATIVE SUBSCRIPTIONS & WEBHOOK IDEMPOTENCY LOG
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.subscriptions (
    id                        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id                   UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    provider                  public.billing_provider NOT NULL DEFAULT 'stripe',
    provider_subscription_id  TEXT NOT NULL UNIQUE,
    plan_code                 TEXT NOT NULL, -- e.g. vip_monthly_39, vip_annual_349, pro_quant_79
    tier                      public.subscription_tier NOT NULL DEFAULT 'vip',
    status                    public.subscription_status NOT NULL DEFAULT 'inactive',
    current_period_start      TIMESTAMPTZ,
    current_period_end        TIMESTAMPTZ,
    cancel_at_period_end      BOOLEAN NOT NULL DEFAULT FALSE,
    raw_payload               JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hot-path index for sub-millisecond RLS entitlement checks
CREATE INDEX IF NOT EXISTS idx_subscriptions_active_lookup
    ON public.subscriptions (user_id, status, current_period_end DESC);

-- Idempotent Stripe / RevenueCat webhook log (strictly service_role only)
CREATE TABLE IF NOT EXISTS public.subscription_webhook_events (
    event_id        TEXT PRIMARY KEY,
    provider        public.billing_provider NOT NULL,
    event_type      TEXT NOT NULL,
    user_id         UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Automatically mirror the highest active subscription tier onto public.profiles.tier
CREATE OR REPLACE FUNCTION public.sync_profile_tier_from_subscription()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_target_user UUID;
    v_best_tier   public.subscription_tier;
BEGIN
    v_target_user := COALESCE(NEW.user_id, OLD.user_id);

    SELECT CASE
        WHEN BOOL_OR(s.tier = 'pro_quant') THEN 'pro_quant'::public.subscription_tier
        WHEN BOOL_OR(s.tier = 'vip')       THEN 'vip'::public.subscription_tier
        ELSE 'free'::public.subscription_tier
    END
    INTO v_best_tier
    FROM public.subscriptions s
    WHERE s.user_id = v_target_user
      AND s.status IN ('active', 'trialing')
      AND (s.current_period_end IS NULL OR s.current_period_end > NOW());

    UPDATE public.profiles
    SET tier = CASE WHEN role = 'admin' THEN 'admin'::public.subscription_tier ELSE COALESCE(v_best_tier, 'free') END,
        updated_at = NOW()
    WHERE id = v_target_user;

    RETURN COALESCE(NEW, OLD);
END;
$$;

DROP TRIGGER IF EXISTS trg_sync_profile_tier ON public.subscriptions;
CREATE TRIGGER trg_sync_profile_tier
    AFTER INSERT OR UPDATE OR DELETE ON public.subscriptions
    FOR EACH ROW EXECUTE FUNCTION public.sync_profile_tier_from_subscription();

-- ----------------------------------------------------------------------------
-- 4. SERVER-SIDE SECURITY DEFINER ENTITLEMENT HELPERS (Used by RLS Policies)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM public.profiles p
        WHERE p.id = auth.uid()
          AND p.role = 'admin'
    );
$$;

CREATE OR REPLACE FUNCTION public.has_active_vip()
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT
        public.is_admin()
        OR EXISTS (
            SELECT 1
            FROM public.subscriptions s
            WHERE s.user_id = auth.uid()
              AND s.tier IN ('vip', 'pro_quant', 'admin')
              AND s.status IN ('active', 'trialing')
              AND (s.current_period_end IS NULL OR s.current_period_end > NOW())
        );
$$;

REVOKE ALL ON FUNCTION public.is_admin() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.is_admin() TO authenticated, service_role;

REVOKE ALL ON FUNCTION public.has_active_vip() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.has_active_vip() TO authenticated, service_role;

-- ----------------------------------------------------------------------------
-- 5. GATED ALPHAPROP CONTENT TABLES (VIP Picks, Parlays, Ledger, User Slips)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.vip_picks (
    pick_id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_date        DATE NOT NULL,
    sport             VARCHAR(16) NOT NULL DEFAULT 'NBA' CHECK (sport IN ('NBA', 'NFL')),
    track             VARCHAR(24) NOT NULL CHECK (
        track IN ('POTD', 'OFFICIAL_VIP', 'BUBBLE_WATCHLIST', 'ALT_FLOOR_90', 'AVALANCHE_UNDER', 'DFS_PICKEM')
    ),
    is_free_potd      BOOLEAN NOT NULL DEFAULT FALSE,
    is_published      BOOLEAN NOT NULL DEFAULT FALSE,
    pulse_wave        VARCHAR(32) NOT NULL DEFAULT 'PULSE_1_OPENER',
    player_name       VARCHAR(96) NOT NULL,
    team_abbr         VARCHAR(8) NOT NULL,
    opp_abbr          VARCHAR(8) NOT NULL,
    matchup           TEXT NOT NULL,
    market_key        VARCHAR(32) NOT NULL,
    side              VARCHAR(16) NOT NULL CHECK (side IN ('OVER', 'UNDER', 'MORE', 'LESS')),
    line_value        NUMERIC(6,2) NOT NULL,
    open_odds         INTEGER NOT NULL,
    buy_stop_line     NUMERIC(6,2),
    buy_stop_odds     INTEGER NOT NULL DEFAULT -125,
    true_prob         NUMERIC(6,5) NOT NULL,
    ev_pct            NUMERIC(7,4) NOT NULL,
    stake_units       NUMERIC(5,2) NOT NULL DEFAULT 0.00,
    lens_breakdown    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vip_picks_slate_gate
    ON public.vip_picks (slate_date DESC, is_published, is_free_potd, track);

CREATE TABLE IF NOT EXISTS public.vip_parlays (
    parlay_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_date        DATE NOT NULL,
    sport             VARCHAR(16) NOT NULL DEFAULT 'NBA',
    parlay_tier       VARCHAR(32) NOT NULL CHECK (
        parlay_tier IN ('DOUBLE_UP_FLOOR', 'SPLASH_FLOOR', 'SAFE_SCRIPT_SGP', 'RISKY_SCRIPT_SGP', 'COPULA_MOONSHOT')
    ),
    title             TEXT NOT NULL,
    is_published      BOOLEAN NOT NULL DEFAULT FALSE,
    combined_odds     INTEGER NOT NULL,
    true_prob         NUMERIC(6,5) NOT NULL,
    ev_pct            NUMERIC(7,4) NOT NULL,
    effective_rho     NUMERIC(5,3) DEFAULT 0.000,
    stake_units       NUMERIC(5,2) NOT NULL DEFAULT 0.00,
    legs              JSONB NOT NULL DEFAULT '[]'::jsonb,
    synergy_notes     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vip_parlays_slate
    ON public.vip_parlays (slate_date DESC, is_published);

-- Ensure pick_ledger exists even if this migration is run standalone on Supabase
CREATE TABLE IF NOT EXISTS public.pick_ledger (
    pick_id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slate_date        DATE NOT NULL,
    sport             VARCHAR(16) NOT NULL DEFAULT 'NBA',
    track             VARCHAR(24) NOT NULL CHECK (
        track IN ('POTD', 'OFFICIAL_VIP', 'BUBBLE_WATCHLIST', 'SGP', 'MOONSHOT', 'DFS', 'MACRO', 'ALT_FLOOR')
    ),
    is_free_potd      BOOLEAN NOT NULL DEFAULT FALSE,
    player_or_title   VARCHAR(128) NOT NULL,
    market_key        VARCHAR(32) NOT NULL,
    line_value        NUMERIC(6,2),
    side              VARCHAR(16),
    sportsbook        VARCHAR(32),
    bet_odds          INTEGER NOT NULL,
    stake_units       NUMERIC(5,2) NOT NULL DEFAULT 0.00,
    status            VARCHAR(16) NOT NULL DEFAULT 'PENDING' CHECK (
        status IN ('PENDING', 'WIN', 'LOSS', 'PUSH', 'VOID', 'PROTECTED')
    ),
    actual_stat       NUMERIC(7,2),
    unit_pnl          NUMERIC(7,3) NOT NULL DEFAULT 0.000,
    clv_pct           NUMERIC(7,4),
    pulse_wave        VARCHAR(32),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    graded_at         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_pick_ledger_status_date
    ON public.pick_ledger (status, slate_date DESC, track);

-- Subscriber personal "My Bets" tracker (strictly scoped to auth.uid() = user_id)
CREATE TABLE IF NOT EXISTS public.user_bet_slips (
    slip_id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id           UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    pick_id           UUID REFERENCES public.vip_picks(pick_id) ON DELETE SET NULL,
    slate_date        DATE NOT NULL DEFAULT CURRENT_DATE,
    sportsbook        VARCHAR(32),
    stake_dollars     NUMERIC(10,2) NOT NULL CHECK (stake_dollars > 0),
    odds_taken        INTEGER NOT NULL,
    line_taken        NUMERIC(6,2),
    status            VARCHAR(16) NOT NULL DEFAULT 'PENDING' CHECK (
        status IN ('PENDING', 'WIN', 'LOSS', 'PUSH', 'VOID')
    ),
    notes             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_bet_slips_user
    ON public.user_bet_slips (user_id, slate_date DESC);

-- ----------------------------------------------------------------------------
-- 6. ENABLE & FORCE ROW-LEVEL SECURITY (RLS) ON ALL TABLES
-- ----------------------------------------------------------------------------
ALTER TABLE public.profiles                     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.profiles                     FORCE  ROW LEVEL SECURITY;

ALTER TABLE public.subscriptions                ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.subscriptions                FORCE  ROW LEVEL SECURITY;

ALTER TABLE public.subscription_webhook_events  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.subscription_webhook_events  FORCE  ROW LEVEL SECURITY;

ALTER TABLE public.vip_picks                    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vip_picks                    FORCE  ROW LEVEL SECURITY;

ALTER TABLE public.vip_parlays                  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vip_parlays                  FORCE  ROW LEVEL SECURITY;

ALTER TABLE public.pick_ledger                  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.pick_ledger                  FORCE  ROW LEVEL SECURITY;

ALTER TABLE public.user_bet_slips               ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_bet_slips               FORCE  ROW LEVEL SECURITY;

-- ----------------------------------------------------------------------------
-- 7. STRICT ROW-LEVEL SECURITY POLICIES
-- ----------------------------------------------------------------------------

-- 7A. public.profiles
DROP POLICY IF EXISTS "profiles_select_own_or_admin" ON public.profiles;
CREATE POLICY "profiles_select_own_or_admin"
    ON public.profiles FOR SELECT
    TO authenticated
    USING (id = auth.uid() OR public.is_admin());

DROP POLICY IF EXISTS "profiles_update_own_safe_fields" ON public.profiles;
CREATE POLICY "profiles_update_own_safe_fields"
    ON public.profiles FOR UPDATE
    TO authenticated
    USING (id = auth.uid())
    WITH CHECK (id = auth.uid());

-- 7B. public.subscriptions (Read-only for owner; writes strictly via service_role webhooks)
DROP POLICY IF EXISTS "subscriptions_select_own_or_admin" ON public.subscriptions;
CREATE POLICY "subscriptions_select_own_or_admin"
    ON public.subscriptions FOR SELECT
    TO authenticated
    USING (user_id = auth.uid() OR public.is_admin());

-- 7C. public.subscription_webhook_events (NO anon/authenticated policies -> service_role only)

-- 7D. public.vip_picks (Free users see ONLY published Free Pick of the Day; VIPs see full board)
DROP POLICY IF EXISTS "vip_picks_public_free_potd" ON public.vip_picks;
CREATE POLICY "vip_picks_public_free_potd"
    ON public.vip_picks FOR SELECT
    TO anon, authenticated
    USING (is_published = TRUE AND is_free_potd = TRUE);

DROP POLICY IF EXISTS "vip_picks_vip_full_access" ON public.vip_picks;
CREATE POLICY "vip_picks_vip_full_access"
    ON public.vip_picks FOR SELECT
    TO authenticated
    USING (is_published = TRUE AND public.has_active_vip());

DROP POLICY IF EXISTS "vip_picks_admin_manage" ON public.vip_picks;
CREATE POLICY "vip_picks_admin_manage"
    ON public.vip_picks FOR ALL
    TO authenticated
    USING (public.is_admin())
    WITH CHECK (public.is_admin());

-- 7E. public.vip_parlays (Strictly active VIP or Admin)
DROP POLICY IF EXISTS "vip_parlays_vip_select" ON public.vip_parlays;
CREATE POLICY "vip_parlays_vip_select"
    ON public.vip_parlays FOR SELECT
    TO authenticated
    USING (is_published = TRUE AND public.has_active_vip());

DROP POLICY IF EXISTS "vip_parlays_admin_manage" ON public.vip_parlays;
CREATE POLICY "vip_parlays_admin_manage"
    ON public.vip_parlays FOR ALL
    TO authenticated
    USING (public.is_admin())
    WITH CHECK (public.is_admin());

-- 7F. public.pick_ledger (Graded history is public for transparency; PENDING VIP picks are gated!)
DROP POLICY IF EXISTS "pick_ledger_public_settled_history" ON public.pick_ledger;
CREATE POLICY "pick_ledger_public_settled_history"
    ON public.pick_ledger FOR SELECT
    TO anon, authenticated
    USING (
        status IN ('WIN', 'LOSS', 'PUSH', 'VOID', 'PROTECTED')
        OR is_free_potd = TRUE
    );

DROP POLICY IF EXISTS "pick_ledger_vip_pending_access" ON public.pick_ledger;
CREATE POLICY "pick_ledger_vip_pending_access"
    ON public.pick_ledger FOR SELECT
    TO authenticated
    USING (public.has_active_vip());

DROP POLICY IF EXISTS "pick_ledger_admin_manage" ON public.pick_ledger;
CREATE POLICY "pick_ledger_admin_manage"
    ON public.pick_ledger FOR ALL
    TO authenticated
    USING (public.is_admin())
    WITH CHECK (public.is_admin());

-- 7G. public.user_bet_slips (Private per-user CRUD)
DROP POLICY IF EXISTS "user_bet_slips_owner_all" ON public.user_bet_slips;
CREATE POLICY "user_bet_slips_owner_all"
    ON public.user_bet_slips FOR ALL
    TO authenticated
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- ----------------------------------------------------------------------------
-- 8. SAFE PUBLIC SUMMARY VIEW (Marketing / Landing Page KPI Strip)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.v_public_track_record
WITH (security_invoker = true) AS
SELECT
    track,
    COUNT(*) FILTER (WHERE status = 'WIN')       AS wins,
    COUNT(*) FILTER (WHERE status = 'LOSS')      AS losses,
    COUNT(*) FILTER (WHERE status IN ('PUSH', 'VOID', 'PROTECTED')) AS pushes_voids,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE status = 'WIN')
        / NULLIF(COUNT(*) FILTER (WHERE status IN ('WIN', 'LOSS')), 0),
        1
    ) AS win_rate_pct,
    ROUND(SUM(unit_pnl)::numeric, 2)             AS net_units
FROM public.pick_ledger
WHERE status IN ('WIN', 'LOSS', 'PUSH', 'VOID', 'PROTECTED')
GROUP BY track;

GRANT SELECT ON public.v_public_track_record TO anon, authenticated;
