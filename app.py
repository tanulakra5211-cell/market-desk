"""
Market Desk -- an Indian equities dashboard.

Single-file version. Nothing else is required except requirements.txt.
Watchlist and order-book data are built in, and are overridden by
data/watchlist.csv and data/order_backlog.csv if those files exist.

Run locally:  streamlit run app.py
"""

import io
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import httpx
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Market Desk",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"


# ============================================================ DEFAULT DATA ===
# Used when data/*.csv are absent, so the app runs from app.py alone.

DEFAULT_WATCHLIST = """ticker,company,bse_scrip,sector
RELIANCE.NS,Reliance Industries,500325,Energy
HDFCBANK.NS,HDFC Bank,500180,Banking
TCS.NS,Tata Consultancy Services,532540,IT
INFY.NS,Infosys,500209,IT
ICICIBANK.NS,ICICI Bank,532174,Banking
LT.NS,Larsen and Toubro,500510,Capital Goods
ITC.NS,ITC,500875,FMCG
SBIN.NS,State Bank of India,500112,Banking
BHARTIARTL.NS,Bharti Airtel,532454,Telecom
RVNL.NS,Rail Vikas Nigam,542649,Infrastructure
IRCON.NS,Ircon International,541956,Infrastructure
BEL.NS,Bharat Electronics,500049,Defence
HAL.NS,Hindustan Aeronautics,541154,Defence
NBCC.NS,NBCC India,534309,Infrastructure
"""

# These order-book figures are PLACEHOLDERS, not disclosed numbers.
# Replace them with figures from each company's investor presentation.
DEFAULT_BACKLOG = """ticker,company,as_of,order_book_cr,notes
RVNL.NS,Rail Vikas Nigam,2025-09-30,97000,PLACEHOLDER - replace with disclosed figure
IRCON.NS,Ircon International,2025-09-30,22000,PLACEHOLDER - replace with disclosed figure
LT.NS,Larsen and Toubro,2025-09-30,610000,PLACEHOLDER - replace with disclosed figure
BEL.NS,Bharat Electronics,2025-09-30,74000,PLACEHOLDER - replace with disclosed figure
NBCC.NS,NBCC India,2025-09-30,105000,PLACEHOLDER - replace with disclosed figure
"""


# ========================================================== NSE SESSION ======
# NSE blocks naive requests. Cookies must be primed from the homepage, and
# cloud hosts need HTTP/2 via httpx -- plain requests gets fingerprinted.

NSE_BASE = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{NSE_BASE}/",
    "Connection": "keep-alive",
}


class NSESession:
    def __init__(self, timeout: float = 15.0):
        self._client = httpx.Client(
            http2=True, headers=NSE_HEADERS, timeout=timeout, follow_redirects=True
        )
        self._primed = False

    def _prime(self):
        self._client.get(NSE_BASE)
        time.sleep(0.4)
        self._client.get(f"{NSE_BASE}/market-data/live-equity-market")
        time.sleep(0.4)
        self._primed = True

    def get_json(self, path: str, retries: int = 2):
        if not self._primed:
            try:
                self._prime()
            except Exception:
                return None
        url = path if path.startswith("http") else f"{NSE_BASE}{path}"
        for attempt in range(retries + 1):
            try:
                resp = self._client.get(url)
                if resp.status_code in (401, 403) and attempt < retries:
                    self._primed = False
                    self._prime()
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception:
                if attempt >= retries:
                    return None
                time.sleep(1.0 + attempt)
        return None


@st.cache_resource
def get_nse_session() -> NSESession:
    return NSESession()


# ============================================================== PRICES =======

GLOBAL_INDICES = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Nifty Bank": "^NSEBANK",
    "India VIX": "^INDIAVIX",
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "Shanghai": "000001.SS",
    "Brent Crude": "BZ=F",
    "Gold": "GC=F",
    "USD/INR": "INR=X",
    "US 10Y": "^TNX",
}


def _pct_change(hist: pd.DataFrame):
    closes = hist["Close"].dropna()
    if len(closes) < 2:
        return (float(closes.iloc[-1]), 0.0) if len(closes) else (float("nan"), 0.0)
    last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
    return last, ((last - prev) / prev) * 100 if prev else 0.0


def fetch_global_markets() -> pd.DataFrame:
    symbols = list(GLOBAL_INDICES.values())
    try:
        raw = yf.download(
            symbols, period="7d", interval="1d", group_by="ticker",
            progress=False, auto_adjust=False, threads=True,
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for name, sym in GLOBAL_INDICES.items():
        try:
            hist = raw[sym] if len(symbols) > 1 else raw
            last, chg = _pct_change(hist)
        except Exception:
            last, chg = float("nan"), 0.0
        rows.append({"Market": name, "Symbol": sym, "Last": last, "Change %": chg})
    return pd.DataFrame(rows)


def fetch_quotes(tickers: list) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            tickers, period="1y", interval="1d", group_by="ticker",
            progress=False, auto_adjust=False, threads=True,
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for t in tickers:
        try:
            hist = raw[t] if len(tickers) > 1 else raw
            closes = hist["Close"].dropna()
            if closes.empty:
                continue
            last, day = _pct_change(hist)

            def ret(days, _c=closes, _l=last):
                if len(_c) <= days:
                    return float("nan")
                base = float(_c.iloc[-days - 1])
                return ((_l - base) / base) * 100 if base else float("nan")

            hi = float(closes.max())
            rows.append({
                "Ticker": t, "Last": last, "Day %": day,
                "1M %": ret(21), "6M %": ret(126), "1Y %": ret(250),
                "52W High": hi, "52W Low": float(closes.min()),
                "Off 52W High %": ((last - hi) / hi) * 100 if hi else float("nan"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def _safe(d: dict, key: str, scale: float = 1.0):
    v = d.get(key)
    if v is None or isinstance(v, str):
        return float("nan")
    try:
        return float(v) * scale
    except (TypeError, ValueError):
        return float("nan")


def fetch_ratios(tickers: list) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}

            roce = float("nan")
            try:
                fin, bs = tk.financials, tk.balance_sheet
                ebit = fin.loc["EBIT"].iloc[0]
                capital = bs.loc["Total Assets"].iloc[0] - bs.loc["Current Liabilities"].iloc[0]
                if capital:
                    roce = (ebit / capital) * 100
            except Exception:
                pass

            rows.append({
                "Ticker": t,
                "Name": info.get("shortName", t),
                "Sector": info.get("sector", "—"),
                "Mkt Cap (Cr)": _safe(info, "marketCap", 1 / 1e7),
                "P/E": _safe(info, "trailingPE"),
                "Fwd P/E": _safe(info, "forwardPE"),
                "P/B": _safe(info, "priceToBook"),
                "EV/EBITDA": _safe(info, "enterpriseToEbitda"),
                "ROE %": _safe(info, "returnOnEquity", 100),
                "ROCE %": roce,
                "Op Margin %": _safe(info, "operatingMargins", 100),
                "Net Margin %": _safe(info, "profitMargins", 100),
                "D/E": _safe(info, "debtToEquity"),
                "Current Ratio": _safe(info, "currentRatio"),
                "Rev Growth %": _safe(info, "revenueGrowth", 100),
                "Profit Growth %": _safe(info, "earningsGrowth", 100),
                "Div Yield %": _safe(info, "dividendYield", 100),
                "EPS": _safe(info, "trailingEps"),
                "Beta": _safe(info, "beta"),
            })
        except Exception:
            rows.append({"Ticker": t, "Name": t})
    return pd.DataFrame(rows)


# =============================================================== NEWS ========

RSS_FEEDS = {
    "Moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "MC Business": "https://www.moneycontrol.com/rss/business.xml",
    "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET Stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
    "Mint Markets": "https://www.livemint.com/rss/markets",
    "Hindu BusinessLine": "https://www.thehindubusinessline.com/markets/feeder/default.rss",
}

BSE_ANN_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
BSE_HEADERS = {
    "User-Agent": NSE_HEADERS["User-Agent"],
    "Referer": "https://www.bseindia.com/",
    "Origin": "https://www.bseindia.com",
    "Accept": "application/json, text/plain, */*",
}


def fetch_market_news(limit_per_feed: int = 15) -> pd.DataFrame:
    rows = []
    for source, url in RSS_FEEDS.items():
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:limit_per_feed]:
                published = None
                if getattr(entry, "published_parsed", None):
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                rows.append({
                    "Time": published,
                    "Source": source,
                    "Headline": entry.get("title", "").strip(),
                    "Link": entry.get("link", ""),
                    "Summary": entry.get("summary", "")[:280],
                })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["Headline"])
    return df.sort_values("Time", ascending=False, na_position="last").reset_index(drop=True)


def filter_news_for(df: pd.DataFrame, keywords: list) -> pd.DataFrame:
    if df.empty or not keywords:
        return df
    pattern = "|".join(k.strip() for k in keywords if k.strip())
    mask = df["Headline"].str.contains(pattern, case=False, na=False) | df[
        "Summary"
    ].str.contains(pattern, case=False, na=False)
    return df[mask].reset_index(drop=True)


def fetch_bse_announcements(days_back: int = 3, scrip_code: str = "") -> pd.DataFrame:
    today = datetime.now()
    params = {
        "pageno": 1, "strCat": "-1",
        "strPrevDate": (today - timedelta(days=days_back)).strftime("%Y%m%d"),
        "strScrip": scrip_code, "strSearch": "P",
        "strToDate": today.strftime("%Y%m%d"),
        "strType": "C", "subcategory": "-1",
    }
    try:
        resp = httpx.get(BSE_ANN_URL, params=params, headers=BSE_HEADERS, timeout=20.0)
        resp.raise_for_status()
        table = resp.json().get("Table", []) or []
    except Exception:
        return pd.DataFrame()

    rows = []
    for item in table:
        attachment = item.get("ATTACHMENTNAME") or ""
        rows.append({
            "Time": item.get("News_submission_dt") or item.get("DT_TM"),
            "Company": item.get("SLONGNAME", ""),
            "Scrip": item.get("SCRIP_CD", ""),
            "Category": item.get("CATEGORYNAME", ""),
            "Headline": item.get("NEWSSUB", ""),
            "PDF": (
                f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attachment}"
                if attachment else ""
            ),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.sort_values("Time", ascending=False).reset_index(drop=True)
    return df


def find_order_wins(announcements: pd.DataFrame) -> pd.DataFrame:
    if announcements.empty:
        return announcements
    terms = ["order", "contract", "LOA", "letter of award", "work order",
             "bagged", "awarded", "L1", "tender", "project win"]
    mask = announcements["Headline"].str.contains("|".join(terms), case=False, na=False)
    return announcements[mask].reset_index(drop=True)


# =============================================================== FLOWS =======

def fetch_fii_dii_cash() -> pd.DataFrame:
    data = get_nse_session().get_json("/api/fiidiiTradeReact")
    if not data:
        return pd.DataFrame()
    rows = []
    for item in data:
        try:
            buy = float(item.get("buyValue", 0) or 0)
            sell = float(item.get("sellValue", 0) or 0)
            rows.append({
                "Date": item.get("date", ""),
                "Participant": item.get("category", "").strip(),
                "Buy (Cr)": buy, "Sell (Cr)": sell, "Net (Cr)": buy - sell,
            })
        except (TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def fetch_fno_participant_oi() -> pd.DataFrame:
    date_str = datetime.now().strftime("%d%m%Y")
    url = (
        "https://nsearchives.nseindia.com/content/nsccl/"
        f"fao_participant_oi_{date_str}.csv"
    )
    try:
        df = pd.read_csv(url, skiprows=1)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def summarise_flows(cash: pd.DataFrame) -> dict:
    if cash.empty:
        return {}
    out = {}
    for _, row in cash.iterrows():
        upper = row["Participant"].upper()
        label = "FII" if ("FII" in upper or "FPI" in upper) else "DII"
        out[label] = {
            "buy": row["Buy (Cr)"], "sell": row["Sell (Cr)"],
            "net": row["Net (Cr)"], "date": row["Date"],
        }
    return out


# =============================================================== DEPTH =======
# No free public source exists for the bid/ask book. It must come from a
# broker. Angel One SmartAPI, DhanHQ, Fyers and Shoonya all provide it free
# with an account. Credentials go in Streamlit's Secrets, never in the repo.

class DepthLevel:
    def __init__(self, price, quantity, orders):
        self.price, self.quantity, self.orders = price, quantity, orders

    def as_dict(self):
        return {"Price": self.price, "Qty": self.quantity, "Orders": self.orders}


class MarketDepth:
    def __init__(self, symbol, ltp=0.0, bids=None, asks=None):
        self.symbol, self.ltp = symbol, ltp
        self.bids, self.asks = bids or [], asks or []

    @property
    def total_bid_qty(self):
        return sum(b.quantity for b in self.bids)

    @property
    def total_ask_qty(self):
        return sum(a.quantity for a in self.asks)

    @property
    def imbalance(self):
        total = self.total_bid_qty + self.total_ask_qty
        return (self.total_bid_qty - self.total_ask_qty) / total if total else 0.0

    @property
    def spread(self):
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0].price - self.bids[0].price


class AngelOneDepth:
    name = "Angel One SmartAPI"

    def __init__(self, api_key, client_id, password, totp_secret):
        self.configured = False
        self.error = None
        try:
            import pyotp
            from SmartApi import SmartConnect

            self._client = SmartConnect(api_key=api_key)
            self._client.generateSession(client_id, password, pyotp.TOTP(totp_secret).now())
            self.configured = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)

    def fetch(self, symbol_token, exchange="NSE"):
        if not self.configured:
            return None
        try:
            resp = self._client.getMarketData(
                mode="FULL", exchangeTokens={exchange: [symbol_token]}
            )
            fetched = resp.get("data", {}).get("fetched", [])
            if not fetched:
                return None
            row = fetched[0]
            d = row.get("depth", {})

            def levels(side):
                return [
                    DepthLevel(float(x.get("price", 0)), int(x.get("quantity", 0)),
                               int(x.get("orders", 0)))
                    for x in d.get(side, [])
                ]

            return MarketDepth(
                row.get("tradingSymbol", symbol_token), float(row.get("ltp", 0)),
                levels("buy"), levels("sell"),
            )
        except Exception:
            return None


class DhanDepth:
    name = "DhanHQ"

    def __init__(self, client_id, access_token):
        self.configured = False
        self.error = None
        try:
            from dhanhq import dhanhq

            self._client = dhanhq(client_id, access_token)
            self.configured = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)

    def fetch(self, symbol_token, exchange="NSE_EQ"):
        if not self.configured:
            return None
        try:
            resp = self._client.quote_data({exchange: [int(symbol_token)]})
            data = resp.get("data", {}).get("data", {}).get(exchange, {})
            row = next(iter(data.values()), None)
            if not row:
                return None
            d = row.get("depth", {})
            return MarketDepth(
                symbol_token, float(row.get("last_price", 0)),
                [DepthLevel(float(x["price"]), int(x["quantity"]), int(x.get("orders", 0)))
                 for x in d.get("buy", [])],
                [DepthLevel(float(x["price"]), int(x["quantity"]), int(x.get("orders", 0)))
                 for x in d.get("sell", [])],
            )
        except Exception:
            return None


def build_depth_provider():
    try:
        secrets = st.secrets
    except Exception:
        return None
    try:
        if "angelone" in secrets:
            a = secrets["angelone"]
            return AngelOneDepth(a["api_key"], a["client_id"], a["password"], a["totp_secret"])
        if "dhan" in secrets:
            d = secrets["dhan"]
            return DhanDepth(d["client_id"], d["access_token"])
    except Exception:
        return None
    return None


# ============================================================= BACKLOG =======
# Company order backlogs appear only in quarterly investor presentations,
# in prose, in a PDF. No data feed carries them at any price.

BACKLOG_PATH = DATA_DIR / "order_backlog.csv"


def load_backlog() -> pd.DataFrame:
    if BACKLOG_PATH.exists():
        df = pd.read_csv(BACKLOG_PATH)
    else:
        df = pd.read_csv(io.StringIO(DEFAULT_BACKLOG))
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    return df.sort_values(["ticker", "as_of"])


def _ttm_revenue_cr(ticker: str) -> float:
    try:
        rev = (yf.Ticker(ticker).info or {}).get("totalRevenue")
        return float(rev) / 1e7 if rev else float("nan")
    except Exception:
        return float("nan")


def analyse_backlog(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("as_of")
        latest = group.iloc[-1]
        prev = group.iloc[-2] if len(group) > 1 else None
        revenue = _ttm_revenue_cr(ticker)
        book = float(latest["order_book_cr"])
        ratio = book / revenue if revenue and revenue == revenue else float("nan")
        rows.append({
            "Ticker": ticker,
            "Company": latest.get("company", ticker),
            "As Of": latest["as_of"].date() if pd.notna(latest["as_of"]) else None,
            "Order Book (Cr)": book,
            "TTM Revenue (Cr)": revenue,
            "Book-to-Bill": ratio,
            "Visibility (yrs)": ratio,
            "QoQ Change %": (
                ((book - float(prev["order_book_cr"])) / float(prev["order_book_cr"])) * 100
                if prev is not None and float(prev["order_book_cr"]) else float("nan")
            ),
            "Notes": latest.get("notes", ""),
        })
    return pd.DataFrame(rows).sort_values("Book-to-Bill", ascending=False, na_position="last")


def load_watchlist() -> pd.DataFrame:
    path = DATA_DIR / "watchlist.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"bse_scrip": str})
    return pd.read_csv(io.StringIO(DEFAULT_WATCHLIST), dtype={"bse_scrip": str})


# ============================================================== STYLING ======

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-feature-settings: "tnum" 1, "lnum" 1; }
      .stApp { background: #0f1216; }
      h1, h2, h3 { letter-spacing: -0.02em; font-weight: 650; }
      .pulse-strip { display: flex; gap: 0; flex-wrap: wrap;
        border-top: 1px solid #232a33; border-bottom: 1px solid #232a33;
        margin-bottom: 1.4rem; }
      .pulse-cell { flex: 1 1 140px; padding: 0.7rem 1rem;
        border-right: 1px solid #1a2028; }
      .pulse-cell:last-child { border-right: none; }
      .pulse-label { font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.09em; color: #6b7684; margin-bottom: 0.25rem; }
      .pulse-value { font-family: ui-monospace, "SF Mono", monospace;
        font-size: 1.05rem; font-weight: 600; color: #e6eaef; }
      .pulse-delta { font-family: ui-monospace, monospace; font-size: 0.78rem;
        margin-left: 0.4rem; }
      .up { color: #2fbf71; } .down { color: #e5484d; } .flat { color: #6b7684; }
      .stale { font-size: 0.72rem; color: #c9a227; background: #1f1a10;
        border-left: 2px solid #b8860b; padding: 0.6rem 0.9rem; margin: 0.5rem 0; }
      .headline-row { padding: 0.55rem 0; border-bottom: 1px solid #1a2028; }
      .headline-meta { font-size: 0.7rem; color: #6b7684;
        text-transform: uppercase; letter-spacing: 0.06em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ========================================================= CACHED LOADERS ====

@st.cache_data(ttl=300)
def load_global():
    return fetch_global_markets()


@st.cache_data(ttl=300)
def load_quotes(tickers: tuple):
    return fetch_quotes(list(tickers))


@st.cache_data(ttl=21600)
def load_ratios(tickers: tuple):
    return fetch_ratios(list(tickers))


@st.cache_data(ttl=900)
def load_news():
    return fetch_market_news()


@st.cache_data(ttl=1800)
def load_announcements(days: int, scrip: str):
    return fetch_bse_announcements(days_back=days, scrip_code=scrip)


@st.cache_data(ttl=1800)
def load_flows():
    return fetch_fii_dii_cash()


@st.cache_data(ttl=3600)
def load_fno_oi():
    return fetch_fno_participant_oi()


# ============================================================== HELPERS ======

def fmt(value, decimals: int = 2, dash: str = "—") -> str:
    if value is None or value != value:
        return dash
    return f"{value:,.{decimals}f}"


def delta_class(value) -> str:
    if value is None or value != value:
        return "flat"
    return "up" if value > 0 else "down" if value < 0 else "flat"


def pulse_cell(label: str, value: str, delta: str = "", cls: str = "flat") -> str:
    d = f'<span class="pulse-delta {cls}">{delta}</span>' if delta else ""
    return (f'<div class="pulse-cell"><div class="pulse-label">{label}</div>'
            f'<div class="pulse-value">{value}{d}</div></div>')


def colour_frame(df: pd.DataFrame, pct_cols: list):
    existing = [c for c in pct_cols if c in df.columns]

    def shade(v):
        if v is None or v != v:
            return "color: #6b7684"
        return "color: #2fbf71" if v > 0 else "color: #e5484d" if v < 0 else ""

    return df.style.map(shade, subset=existing).format(precision=2, na_rep="—")


# ============================================================== SIDEBAR ======

watchlist = load_watchlist()

with st.sidebar:
    st.markdown("### Market Desk")
    st.caption("Indian equities, one screen")

    name_by_ticker = watchlist.set_index("ticker")["company"].to_dict()
    selected = st.multiselect(
        "Watchlist",
        options=watchlist["ticker"].tolist(),
        default=watchlist["ticker"].tolist(),
        format_func=lambda t: name_by_ticker.get(t, t),
    )

    st.divider()
    news_days = st.slider("Filing lookback (days)", 1, 15, 3)
    only_watchlist_news = st.checkbox("Filter news to watchlist", value=False)

    st.divider()
    if st.button("Refresh all data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Prices via Yahoo · flows via NSE · news via publisher RSS and BSE filings. "
        "All figures delayed. Research tool, not an execution tool."
    )

wl = watchlist[watchlist["ticker"].isin(selected)] if selected else watchlist
tickers = tuple(wl["ticker"].tolist())


# =========================================================== PULSE STRIP =====

global_df = load_global()
flows_df = load_flows()
flow_summary = summarise_flows(flows_df)

cells = []
if not global_df.empty:
    for name in ["Nifty 50", "Sensex", "Nifty Bank", "India VIX", "USD/INR"]:
        row = global_df[global_df["Market"] == name]
        if row.empty:
            continue
        cells.append(pulse_cell(
            name, fmt(row.iloc[0]["Last"]),
            f'{row.iloc[0]["Change %"]:+.2f}%', delta_class(row.iloc[0]["Change %"]),
        ))

for label in ["FII", "DII"]:
    if label in flow_summary:
        net = flow_summary[label]["net"]
        cells.append(pulse_cell(f"{label} net", f"{net:+,.0f} Cr", "", delta_class(net)))
    else:
        cells.append(pulse_cell(f"{label} net", "—", "", "flat"))

st.markdown(f'<div class="pulse-strip">{"".join(cells)}</div>', unsafe_allow_html=True)


# ================================================================= TABS ======

tab_market, tab_news, tab_flows, tab_ratios, tab_book, tab_depth = st.tabs(
    ["Markets", "News", "FII / DII", "Ratios", "Order Book", "Depth"]
)

with tab_market:
    left, right = st.columns(2)

    with left:
        st.markdown("#### Your watchlist")
        quotes = load_quotes(tickers)
        if quotes.empty:
            st.info("No price data returned. Hit Refresh, or check your tickers.")
        else:
            merged = quotes.merge(
                wl[["ticker", "company", "sector"]],
                left_on="Ticker", right_on="ticker", how="left",
            ).drop(columns=["ticker"])
            merged = merged[["company", "sector", "Last", "Day %", "1M %", "6M %",
                             "1Y %", "Off 52W High %"]].rename(
                columns={"company": "Company", "sector": "Sector"})
            st.dataframe(
                colour_frame(merged, ["Day %", "1M %", "6M %", "1Y %", "Off 52W High %"]),
                use_container_width=True, height=520, hide_index=True,
            )

    with right:
        st.markdown("#### Global markets")
        if global_df.empty:
            st.info("Global data unavailable. Hit Refresh.")
        else:
            st.dataframe(
                colour_frame(global_df.drop(columns=["Symbol"]), ["Change %"]),
                use_container_width=True, height=520, hide_index=True,
            )


with tab_news:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("#### Market headlines")
        headlines = load_news()
        if only_watchlist_news and not wl.empty:
            headlines = filter_news_for(headlines, wl["company"].str.split().str[0].tolist())

        if headlines.empty:
            st.info("No headlines available right now.")
        else:
            for _, row in headlines.head(40).iterrows():
                stamp = row["Time"].strftime("%d %b, %H:%M") if pd.notna(row["Time"]) else "—"
                st.markdown(
                    f'<div class="headline-row">'
                    f'<div class="headline-meta">{row["Source"]} · {stamp}</div>'
                    f'<a href="{row["Link"]}" target="_blank">{row["Headline"]}</a></div>',
                    unsafe_allow_html=True,
                )

    with col_b:
        st.markdown("#### Corporate filings")
        st.caption("From BSE — results, board meetings, order wins.")
        anns = load_announcements(news_days, "")

        if anns.empty:
            st.info("BSE announcements unavailable. Try Refresh.")
        else:
            if only_watchlist_news and not wl.empty:
                anns = anns[anns["Scrip"].astype(str).isin(wl["bse_scrip"].astype(str))]

            wins = find_order_wins(anns)
            if not wins.empty:
                st.markdown("**Possible order wins**")
                for _, row in wins.head(10).iterrows():
                    link = f' [PDF]({row["PDF"]})' if row["PDF"] else ""
                    st.markdown(f'- **{row["Company"]}** — {row["Headline"]}{link}')
                st.divider()

            st.markdown("**All filings**")
            st.dataframe(anns[["Time", "Company", "Category", "Headline"]].head(60),
                         use_container_width=True, height=380, hide_index=True)


with tab_flows:
    st.markdown("#### Institutional flows")

    if flows_df.empty:
        st.markdown(
            '<div class="stale">NSE returned no flow data. This endpoint is scraped '
            'and refuses requests intermittently, especially from cloud hosts. Daily '
            'figures publish around 7pm IST — try again after that.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"Cash market, Rs crore · session dated {flows_df.iloc[0]['Date']}")
        st.dataframe(colour_frame(flows_df, ["Net (Cr)"]),
                     use_container_width=True, hide_index=True)
        st.bar_chart(flows_df.set_index("Participant")[["Buy (Cr)", "Sell (Cr)"]], height=280)

    st.divider()
    st.markdown("#### F&O participant positioning")
    st.caption("Whether FIIs are net long or short index futures — the signal cash "
               "flows alone don't give you.")

    oi = load_fno_oi()
    if oi.empty:
        st.info("Not published yet for today. NSE releases this after close on trading days.")
    else:
        st.dataframe(oi, use_container_width=True, hide_index=True)


with tab_ratios:
    st.markdown("#### Fundamentals screener")
    st.caption("Cached six hours — the source is rate-limited, so repeated refreshes "
               "will get you throttled.")

    with st.spinner("Pulling fundamentals…"):
        ratios = load_ratios(tickers)

    if ratios.empty:
        st.info("No fundamental data returned.")
    else:
        c1, c2, c3 = st.columns(3)
        max_pe = c1.number_input("Max P/E", value=0.0, help="0 disables this filter")
        min_roe = c2.number_input("Min ROE %", value=0.0)
        max_de = c3.number_input("Max D/E", value=0.0, help="0 disables this filter")

        view = ratios.copy()
        if max_pe > 0:
            view = view[view["P/E"].fillna(1e9) <= max_pe]
        if min_roe > 0:
            view = view[view["ROE %"].fillna(-1e9) >= min_roe]
        if max_de > 0:
            view = view[view["D/E"].fillna(1e9) <= max_de]

        st.dataframe(view.style.format(precision=2, na_rep="—"),
                     use_container_width=True, height=520, hide_index=True)
        st.caption(f"{len(view)} of {len(ratios)} companies pass the filters. "
                   "Blanks mean missing data, not zero — Yahoo's coverage thins out "
                   "on mid and small caps.")


with tab_book:
    st.markdown("#### Order backlog")
    st.markdown(
        "Company order books are disclosed in quarterly investor presentations and "
        "nowhere else — no data feed carries them. Log the number here after each "
        "result and this tab does the analysis. The News tab flags order-win filings "
        "so you know when there's something to log."
    )

    raw_backlog = load_backlog()
    analysis = analyse_backlog(raw_backlog)

    if analysis.empty:
        st.info("No backlog entries yet.")
    else:
        st.dataframe(analysis.style.format(precision=2, na_rep="—"),
                     use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="stale">The figures shipped with this app are PLACEHOLDERS, '
            'not disclosed numbers. Replace them before reading anything into '
            'book-to-bill.</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Edit entries**")
    st.caption("Edit like a spreadsheet, then download and commit the file to your "
               "repo — hosted apps reset local disk on every rebuild.")

    editable = raw_backlog.copy()
    if not editable.empty:
        editable["as_of"] = editable["as_of"].dt.strftime("%Y-%m-%d")

    edited = st.data_editor(
        editable, num_rows="dynamic", use_container_width=True,
        column_config={
            "ticker": st.column_config.SelectboxColumn(
                "Ticker", options=watchlist["ticker"].tolist(), required=True),
            "company": st.column_config.TextColumn("Company"),
            "as_of": st.column_config.TextColumn("As of", help="Quarter end, YYYY-MM-DD"),
            "order_book_cr": st.column_config.NumberColumn(
                "Order book (Rs Cr)", min_value=0.0, step=100.0, format="%.0f"),
            "notes": st.column_config.TextColumn("Source / notes", width="large"),
        },
        key="backlog_editor",
    )

    st.download_button(
        "Download order_backlog.csv",
        data=edited.to_csv(index=False).encode("utf-8"),
        file_name="order_backlog.csv", mime="text/csv",
        help="Commit this to data/ in your repo to make changes permanent.",
    )


with tab_depth:
    st.markdown("#### Live market depth")
    provider = build_depth_provider()

    if provider is None or not getattr(provider, "configured", False):
        st.markdown(
            "Depth needs a broker connection. There is no free public source for the "
            "bid/ask book — NSE doesn't expose one, and the depth widget on their site "
            "can't be scraped at any useful rate."
        )
        st.markdown(
            """
            **Free broker APIs that provide depth** (each needs an account with them):

            | Broker | Python package |
            |---|---|
            | Angel One SmartAPI | `smartapi-python` |
            | DhanHQ | `dhanhq` |
            | Fyers | `fyers-apiv3` |
            | Shoonya (Finvasia) | `NorenRestApiPy` |

            Add credentials under **Settings → Secrets** in Streamlit Cloud — never in
            your repo — then add the package to `requirements.txt`:

            ```toml
            [angelone]
            api_key = "..."
            client_id = "..."
            password = "..."      # your MPIN
            totp_secret = "..."   # from the TOTP setup QR
            ```
            """
        )
        if provider is not None and getattr(provider, "error", None):
            st.error(f"Connection failed: {provider.error}")
    else:
        st.success(f"Connected via {provider.name}")
        token = st.text_input(
            "Symbol token",
            help="From your broker's instrument master file — not the ticker symbol.",
        )
        if token:
            book = provider.fetch(token)
            if book is None:
                st.warning("No depth returned for that token.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("LTP", fmt(book.ltp))
                m2.metric("Spread", fmt(book.spread))
                m3.metric("Imbalance", f"{book.imbalance:+.1%}",
                          help="Positive means more size resting on the bid.")

                bid_col, ask_col = st.columns(2)
                bid_col.markdown("**Bids**")
                bid_col.dataframe(pd.DataFrame([b.as_dict() for b in book.bids]),
                                  use_container_width=True, hide_index=True)
                ask_col.markdown("**Asks**")
                ask_col.dataframe(pd.DataFrame([a.as_dict() for a in book.asks]),
                                  use_container_width=True, hide_index=True)
