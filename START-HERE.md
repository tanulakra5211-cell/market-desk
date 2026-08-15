# Start here

Three steps. No terminal, no Python install, no cost.

---

### 1. Put these files on GitHub

Sign up at **github.com** if you don't have an account, and verify your email.

Click the **+** at top right → **New repository**. Name it `market-desk`.
Leave everything else as-is and click **Create repository**.

On the next screen click **uploading an existing file**. Unzip the folder you
downloaded, open it, select **everything inside** (not the folder itself —
its contents), and drag it into the browser. Scroll down, click
**Commit changes**.

You should now see `app.py`, `requirements.txt`, and folders called `sources`
and `data`. If `sources` and `data` are missing and their files are sitting
loose at the top level instead, delete the repo and re-drag — the structure
matters.

---

### 2. Deploy it

Go to **share.streamlit.io** and click **Sign in with GitHub**. Approve the
permission prompt.

Click **Create app** → **Deploy a public app from GitHub**.

- Repository: `your-username/market-desk`
- Branch: `main`
- Main file path: `app.py`

Click **Deploy**. First build takes three to five minutes — a log will scroll
past. Yellow warnings are fine. Only red errors matter.

---

### 3. Make it yours

You'll land on a URL like `market-desk.streamlit.app`. That's permanent.
Open it on your phone and use *Add to Home Screen*.

To change the stocks: in GitHub, open `data/watchlist.csv`, click the pencil
icon, edit, and commit. The live app rebuilds itself in about a minute.

Columns are: `ticker, company, bse_scrip, sector`

- NSE tickers end in `.NS` — e.g. `RELIANCE.NS`
- `bse_scrip` is the numeric BSE code, findable by searching the company on
  bseindia.com. It's used to match corporate filings to the company.

---

## Things that are meant to look broken but aren't

**FII/DII tab shows a yellow notice.** NSE blocks requests from cloud servers
fairly often, and the daily figures only publish around 7pm IST. Every other
tab works regardless. Check back in the evening.

**Depth tab says "not configured".** Correct — it needs a broker API key.
Angel One, Dhan and Fyers all give this away free with an account. Add the
credentials under your app's **Settings → Secrets** in Streamlit Cloud, never
in GitHub.

**Ratios tab is slow the first time.** It's pulling fundamentals company by
company, then caching for six hours. Subsequent loads are instant.

**App takes 30 seconds to open after a few days away.** Free tier apps sleep
when idle. Normal.

---

## Two honest limitations

**The order book figures shipped in `data/order_backlog.csv` are made up.**
They're placeholders so the tab renders. Company order backlogs are disclosed
only in quarterly investor presentation PDFs, in prose — no data feed carries
them at any price. Replace them with real figures from the presentations. The
News tab flags order-win filings so you know when a number has changed.

**Everything here is delayed data.** It's a research and monitoring screen,
not an execution tool, and nothing in it is a recommendation to buy or sell.

---

Full technical detail is in `README.md`.
