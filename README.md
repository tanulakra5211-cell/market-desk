# Market Desk

An Indian equities dashboard that pulls news, global markets, institutional
flows, fundamental ratios, order backlog and market depth onto one screen.

Runs as a web app you can open from any browser or phone.

---

## Setup

```bash
git clone <your-repo>
cd market-desk
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy free

1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   point it at `app.py`.
3. You get a permanent public URL that works on mobile. Free tier, no card.

If you want it private, set the app to private in Streamlit Cloud settings, or
self-host on a small VPS with `nginx` in front.

---

## What comes from where

| Panel | Source | Reliability |
|---|---|---|
| Watchlist prices, returns | Yahoo Finance via `yfinance` | Solid |
| Global markets, commodities, USD/INR | Yahoo Finance | Solid |
| Financial ratios | Yahoo Finance `.info` + computed ROCE | Good, occasionally gappy |
| Market news | Publisher RSS (Moneycontrol, ET, BS, Mint, BL) | Solid |
| Corporate filings, order-win alerts | BSE announcements API | Good |
| FII/DII cash flows | NSE `fiidiiTradeReact` | Flaky, see below |
| F&O participant OI | NSE daily CSV | Published after close only |
| Order backlog | You, via `data/order_backlog.csv` | Manual by necessity |
| Live market depth | Your broker's API | Needs credentials |

---

## The three things that aren't free or automatic

**1. Live market depth needs a broker.** No public source exists. NSE doesn't
expose the order book, and the depth widget on their website can't be scraped
at any useful rate. Free broker APIs that do provide it: Angel One SmartAPI,
DhanHQ, Fyers, Shoonya. Each needs an account with that broker but costs
nothing for market data. Add credentials to `.streamlit/secrets.toml`
(template in `secrets.toml.example`) and the Depth tab activates.

*Note: since April 2026, exchange rules require registered static IPs for
**order placement** through broker APIs. Market-data reads are unaffected — but
if you extend this into execution, check your broker's current requirements.*

**2. Order backlog is manual.** Company order books live in quarterly investor
presentation PDFs and earnings calls, in prose. No API anywhere carries them —
not free ones, not paid ones. The workflow this app supports: the News tab
scans BSE filings for order-win language and flags them, you open the
presentation, and log the figure in the Order Book tab. Two minutes a quarter
per company. Book-to-bill and revenue visibility are computed for you.

**3. NSE blocks scrapers.** The FII/DII endpoint is behind cookie checks and
rate limits, and it refuses datacenter IPs more aggressively than home ones.
`sources/nse_session.py` handles cookie priming and uses HTTP/2 via `httpx`,
which is what makes it work at all from cloud hosts. Expect it to fail
sometimes anyway — the app shows a clear notice rather than a stack trace.
Keep requests under ~3/second. The daily figures publish around 7pm IST, so
after-hours is the reliable window.

If you want FII/DII to be rock solid, run a scheduled job on your own machine
that writes the numbers to a CSV in the repo, and have the cloud app read that
instead of hitting NSE directly.

---

## Customising

**Watchlist** — edit `data/watchlist.csv`. NSE tickers take `.NS`, BSE takes
`.BO`. The `bse_scrip` column is the numeric BSE code, used to filter filings.

**News sources** — add or remove feeds in `RSS_FEEDS` at the top of
`sources/news.py`.

**Order-win detection** — the keyword list is in `find_order_wins()` in
`sources/news.py`. Tune it for your sectors.

**Global markets** — `GLOBAL_INDICES` in `sources/prices.py`.

---

## Where to go next

- **Better fundamentals.** Yahoo's Indian coverage has gaps. If you outgrow it,
  Financial Modeling Prep and EODHD both have paid India coverage with proper
  quarterly statements. Screener.in is the best data but scraping it breaks
  their terms of service, so don't build on it.
- **Alerts.** A scheduled job that runs the order-win scan and pushes to
  Telegram would catch filings without you opening the dashboard.
- **Historical flows.** Store each day's FII/DII pull into a local SQLite file
  so you can chart trends instead of seeing one day at a time.

---

This is a research and monitoring tool. Nothing in it is a recommendation to
buy or sell, and delayed data should never be used for execution decisions.
