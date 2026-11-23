# AlphaProp Pro™ — Zero-Coding Setup Guide (for `ababington1981`)

You do **not** need to open a terminal or type code commands. Everything below is done with normal mouse clicks in your web browser.

---

## Step 1: Create Your Private Repo on GitHub (2 minutes)
1. Sign in to **github.com** as **`ababington1981`**.
2. Click the **`+`** icon in the top-right corner and select **New repository**.
3. Set **Repository name** to: `alphaprop-pro`
4. Select **Private** (so your proprietary NBA v8.8 & NFL v4.4 engines stay hidden from the public).
5. Click **Create repository**.

## Step 2: Drag-and-Drop Upload Your Files (2 minutes)
1. Unzip `alphaprop_pro_github_starter.zip` on your computer so you see the `alphaprop-pro` folder.
2. On your new empty GitHub repository page, click the blue link **"uploading an existing file"**.
3. Open the `alphaprop-pro` folder on your computer, select **everything inside it**, and drag-and-drop it into the GitHub browser box.
4. Wait for the file list to finish uploading, then click the green **Commit changes** button at the bottom.
   - *Tip:* If your browser hides `.github` or `.env.example`, that's totally fine—we can add the `.github/workflows` file in 30 seconds via GitHub's "Add file -> Create new file" button.

## Step 3: Put Your Frontend Live on Vercel (3 minutes — Free)
1. Go to **vercel.com** and click **Sign Up -> Continue with GitHub**.
2. Click **Add New -> Project** and select **`ababington1981/alphaprop-pro`**.
3. Leave all settings as default and click **Deploy**.
4. In ~45 seconds, Vercel gives you your live HTTPS website URL (and you can attach your custom domain later). Every time you upload a new `index.html` to GitHub, Vercel updates your live site automatically.

## Step 4: Turn On Your Backend API on Render (5 minutes)
1. Go to **render.com** and click **Get Started -> GitHub**.
2. Click **New -> Blueprint** and select **`ababington1981/alphaprop-pro`**.
3. Render automatically reads `render.yaml` and sets up your Python FastAPI server.
4. Paste your API keys (from Step 5 & Step 6 below) into Render's **Environment Variables** box and click **Apply**.
5. Once live, visit `https://your-render-url.onrender.com/health` — you should see `{"status": "ok"}`.

## Step 5: Connect Subscriber Logins & Database on Supabase (5 minutes — Free Tier)
1. Go to **supabase.com**, sign in with GitHub, and click **New Project** (name it `alphaprop-pro`).
2. In the left menu, click **SQL Editor -> New query**.
3. Open `supabase/supabase_rls_schema.sql` from your repo, copy-paste the entire text into Supabase, and click **Run**.
4. Go to **Project Settings -> API** and copy these 3 values into Render (Step 4):
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `SUPABASE_JWT_SECRET`

## Step 6: Connect Live Odds & Stripe Payments (5 minutes)
1. **The Odds API** (`the-odds-api.com`): Create an account, copy your API key, and paste it as `ODDS_API_KEY` in both **Render -> Environment** and **GitHub -> Settings -> Secrets and variables -> Actions**.
2. **Stripe** (`stripe.com`): Create your VIP subscription product, copy your `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET`, and paste them into Render.

## How to Edit Your App Anytime Later
- **Daily tweaks (unit caps, promo banner, feature switches):** Click `config.json` on GitHub -> click the **Pencil icon** -> edit -> click **Commit changes**.
- **Design or model upgrades:** Tell Gemini what you want changed -> download the updated file -> click **Add file -> Upload files** on GitHub -> **Commit changes**.
- **Undo anything:** Click **Commits** on GitHub -> click the previous working snapshot -> **Revert**.
