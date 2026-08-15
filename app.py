"""
Market Desk -- an Indian equities dashboard.

Single-file version. Nothing else is required except requirements.txt.
Watchlist and order-book data are built in, and are overridden by
data/watchlist.csv and data/order_backlog.csv if those files exist.

Run locally:  streamlit run app.py
"""

import io
import time
import traceback
import zipfile
from contextlib import contextmanager
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
APP_VERSION = "v14 — disclosures & journal"


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
        df = pd.read_csv(url, skiprows=1, skipinitialspace=True)
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


# ====================================================== FULL NSE UNIVERSE ====
# Fetching 2,000+ companies one at a time is not viable. Instead NSE publishes
# a single daily file covering every security that traded -- the bhavcopy.
# One request gives you the whole market. Screen on that, then pull expensive
# per-company fundamentals only for the handful that survive your filters.

EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

# Full security-wise file: includes delivery quantity and delivery percentage,
# which the plain UDiFF bhavcopy does not carry.
SEC_BHAV_URL = (
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
)

# UDiFF common bhavcopy -- fallback. The pre-July-2024 URL is discontinued.
UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)


def _archive_get(url: str):
    """
    GET a file from nsearchives. These need a Referer pointing at the reports
    page rather than the API referer used for /api/ endpoints, otherwise NSE
    returns 403. Returns (response | None, status_note).
    """
    sess = get_nse_session()
    try:
        if not sess._primed:
            sess._prime()
    except Exception as exc:  # noqa: BLE001
        return None, f"session priming failed: {exc}"

    headers = {
        "Referer": "https://www.nseindia.com/all-reports",
        "Accept": "text/csv,application/zip,application/octet-stream,*/*",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }
    try:
        resp = sess._client.get(url, headers=headers)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        if len(resp.content) < 200:
            return None, f"empty response ({len(resp.content)} bytes)"
        head = resp.content[:300].lstrip().lower()
        if head.startswith(b"<!doctype") or head.startswith(b"<html"):
            return None, f"HTML page, not data ({len(resp.content)} bytes)"
        return resp, f"ok ({len(resp.content) // 1024} KB)"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def fetch_nse_universe() -> pd.DataFrame:
    """Every equity listed on NSE: symbol, name, series, listing date, ISIN."""
    resp, _ = _archive_get(EQUITY_LIST_URL)
    if resp is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(resp.text), skipinitialspace=True)
    except Exception:
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    rename = {
        "SYMBOL": "Symbol", "NAME OF COMPANY": "Company", "SERIES": "Series",
        "DATE OF LISTING": "Listed", "ISIN NUMBER": "ISIN", "FACE VALUE": "Face Value",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ["Symbol", "Company", "Series", "Listed", "ISIN"] if c in df.columns]
    return df[keep]


def _normalise_udiff(df: pd.DataFrame) -> pd.DataFrame:
    """Map UDiFF column names onto the sec_bhavdata_full shape."""
    df = df[df.get("FinInstrmTp", "STK") == "STK"].copy()
    out = pd.DataFrame({
        "SYMBOL": df["TckrSymb"],
        "SERIES": df.get("SctySrs", "EQ"),
        "CLOSE_PRICE": pd.to_numeric(df["ClsPric"], errors="coerce"),
        "PREV_CLOSE": pd.to_numeric(df["PrvsClsgPric"], errors="coerce"),
        "TTL_TRD_QNTY": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
        "TURNOVER_LACS": pd.to_numeric(df["TtlTrfVal"], errors="coerce") / 1e5,
        "NO_OF_TRADES": pd.to_numeric(df.get("TtlNbOfTxsExctd"), errors="coerce"),
        "DELIV_PER": float("nan"),  # not carried in UDiFF
    })
    return out


def _fetch_bhav_for(date: datetime, attempts: list | None = None) -> pd.DataFrame:
    """
    One trading day's full-market data.

    Tries the security-wise file first (it carries delivery %), then falls back
    to the UDiFF common bhavcopy zip. Every attempt is recorded in `attempts`
    so the UI can show what actually happened.
    """
    if attempts is None:
        attempts = []

    # 1. Security-wise full bhavcopy -- includes DELIV_PER.
    url = SEC_BHAV_URL.format(ddmmyyyy=date.strftime("%d%m%Y"))
    resp, note = _archive_get(url)
    attempts.append((date.strftime("%d-%b-%Y"), "sec_bhavdata_full", note))
    if resp is not None:
        try:
            df = pd.read_csv(io.StringIO(resp.text), skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype(str).str.strip()
            for col in ["PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
                        "LAST_PRICE", "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY",
                        "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["TRADE_DATE"] = date.date()
            return df
        except Exception as exc:  # noqa: BLE001
            attempts.append((date.strftime("%d-%b-%Y"), "sec parse", str(exc)[:80]))

    # 2. UDiFF common bhavcopy zip -- no delivery data, but reliable.
    url = UDIFF_URL.format(yyyymmdd=date.strftime("%Y%m%d"))
    resp, note = _archive_get(url)
    attempts.append((date.strftime("%d-%b-%Y"), "UDiFF zip", note))
    if resp is not None:
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                name = zf.namelist()[0]
                raw = pd.read_csv(zf.open(name), skipinitialspace=True)
            raw.columns = [c.strip() for c in raw.columns]
            df = _normalise_udiff(raw)
            df["TRADE_DATE"] = date.date()
            return df
        except Exception as exc:  # noqa: BLE001
            attempts.append((date.strftime("%d-%b-%Y"), "UDiFF parse", str(exc)[:80]))

    return pd.DataFrame()


def fetch_latest_bhavcopy(max_lookback: int = 7):
    """
    Walk backwards until a trading day's file is found.
    Returns (dataframe, date, attempts).
    """
    attempts: list = []
    for offset in range(max_lookback):
        day = datetime.now() - timedelta(days=offset)
        if day.weekday() >= 5:
            attempts.append((day.strftime("%d-%b-%Y"), "skipped", "weekend"))
            continue
        df = _fetch_bhav_for(day, attempts)
        if not df.empty:
            return df, day.date(), attempts
        time.sleep(0.4)
    return pd.DataFrame(), None, attempts


def build_market_screen(latest: pd.DataFrame, prior: pd.DataFrame,
                       attempts: list | None = None) -> pd.DataFrame:
    """
    Whole-market screening table. Day change comes from the file's own
    PREV_CLOSE; the 1M column is computed against an older bhavcopy if one
    was retrieved.

    Each stage records its row count in `attempts` so a silent drop to zero
    is visible rather than mysterious.
    """
    def log(stage, detail):
        if attempts is not None:
            attempts.append(("build", stage, str(detail)))

    if latest.empty:
        log("input", "empty frame")
        return pd.DataFrame()

    log("rows received", len(latest))
    log("columns", ", ".join(list(latest.columns)[:14]))

    if "SERIES" in latest.columns:
        series_vals = latest["SERIES"].astype(str).str.strip().str.upper()
        log("series found", ", ".join(sorted(series_vals.unique())[:12]))
        df = latest[series_vals == "EQ"].copy()
    else:
        log("series column", "absent - keeping all rows")
        df = latest.copy()

    log("rows after EQ filter", len(df))

    required = ["SYMBOL", "CLOSE_PRICE", "PREV_CLOSE", "TTL_TRD_QNTY", "TURNOVER_LACS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log("MISSING COLUMNS", ", ".join(missing))
        return pd.DataFrame()

    out = pd.DataFrame({
        "Symbol": df["SYMBOL"],
        "Close": df["CLOSE_PRICE"],
        "Day %": ((df["CLOSE_PRICE"] - df["PREV_CLOSE"]) / df["PREV_CLOSE"]) * 100,
        "Volume": df["TTL_TRD_QNTY"],
        "Turnover (Cr)": df["TURNOVER_LACS"] / 100,
        "Delivery %": df.get("DELIV_PER"),
        "Trades": df.get("NO_OF_TRADES"),
    })
    log("rows built", len(out))

    if not prior.empty and "CLOSE_PRICE" in prior.columns:
        base = prior[prior.get("SERIES", "EQ") == "EQ"][["SYMBOL", "CLOSE_PRICE"]]
        base = base.rename(columns={"CLOSE_PRICE": "_base"})
        out = out.merge(base, left_on="Symbol", right_on="SYMBOL", how="left")
        out["1M %"] = ((out["Close"] - out["_base"]) / out["_base"]) * 100
        out = out.drop(columns=["_base", "SYMBOL"], errors="ignore")

    log("rows returned", len(out))
    return out.reset_index(drop=True)


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
        df = pd.read_csv(BACKLOG_PATH, skipinitialspace=True)
    else:
        df = pd.read_csv(io.StringIO(DEFAULT_BACKLOG), skipinitialspace=True)
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


@st.cache_data(ttl=21600)
def cached_backlog_analysis() -> pd.DataFrame:
    """Cached — this hits yfinance per ticker for the revenue denominator."""
    return analyse_backlog(load_backlog())


def load_watchlist() -> pd.DataFrame:
    path = DATA_DIR / "watchlist.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"bse_scrip": str}, skipinitialspace=True)
    return pd.read_csv(io.StringIO(DEFAULT_WATCHLIST), dtype={"bse_scrip": str}, skipinitialspace=True)


# ========================================================== TECHNICALS =======
# Standard indicators computed from price history. These describe what price
# and volume have ALREADY done. They are inputs to your judgement, not signals
# to act on -- every one of them can and does persist while a stock falls.

def _rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    if pd.isna(loss.iloc[-1]) or pd.isna(gain.iloc[-1]):
        return float("nan")
    if loss.iloc[-1] == 0:
        return 100.0 if gain.iloc[-1] > 0 else float("nan")
    rs = gain.iloc[-1] / loss.iloc[-1]
    return 100 - (100 / (1 + rs))


def fetch_technicals(tickers: list) -> pd.DataFrame:
    """RSI, moving-average position, 52-week position and volume surge."""
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                          progress=False, auto_adjust=False, threads=True)
    except Exception:
        return pd.DataFrame()

    rows = []
    for t in tickers:
        try:
            hist = raw[t] if len(tickers) > 1 else raw
            closes = hist["Close"].dropna()
            vols = hist["Volume"].dropna()
            if len(closes) < 60:
                continue

            last = float(closes.iloc[-1])
            sma20 = float(closes.rolling(20).mean().iloc[-1])
            sma50 = float(closes.rolling(50).mean().iloc[-1])
            sma200 = (float(closes.rolling(200).mean().iloc[-1])
                      if len(closes) >= 200 else float("nan"))
            hi52, lo52 = float(closes.max()), float(closes.min())

            ema12 = closes.ewm(span=12).mean()
            ema26 = closes.ewm(span=26).mean()
            macd = ema12 - ema26
            macd_hist = float((macd - macd.ewm(span=9).mean()).iloc[-1])

            avg_vol = float(vols.rolling(20).mean().iloc[-1]) if len(vols) >= 20 else float("nan")
            vol_surge = float(vols.iloc[-1]) / avg_vol if avg_vol else float("nan")

            structure = "—"
            if sma200 == sma200:
                if last > sma50 > sma200:
                    structure = "Above 50 & 200"
                elif last < sma50 < sma200:
                    structure = "Below 50 & 200"
                else:
                    structure = "Mixed"

            rows.append({
                "Ticker": t,
                "Close": last,
                "RSI (14)": _rsi(closes),
                "vs 20DMA %": ((last - sma20) / sma20) * 100,
                "vs 50DMA %": ((last - sma50) / sma50) * 100,
                "vs 200DMA %": ((last - sma200) / sma200) * 100 if sma200 == sma200 else float("nan"),
                "MA structure": structure,
                "52W position %": ((last - lo52) / (hi52 - lo52)) * 100 if hi52 > lo52 else float("nan"),
                "MACD hist": macd_hist,
                "Vol vs 20d avg": vol_surge,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def market_breadth(screen: pd.DataFrame) -> dict:
    """
    Advance/decline breadth from the whole market. This is a measurement of
    what the market did, unlike headline-counting, which measures what
    journalists wrote.
    """
    if screen.empty or "Day %" not in screen.columns:
        return {}
    day = screen["Day %"].dropna()
    if day.empty:
        return {}

    liquid = screen[screen["Turnover (Cr)"].fillna(0) >= 1]
    liquid_day = liquid["Day %"].dropna()

    return {
        "advances": int((day > 0).sum()),
        "declines": int((day < 0).sum()),
        "unchanged": int((day == 0).sum()),
        "ad_ratio": (day > 0).sum() / max((day < 0).sum(), 1),
        "median_move": float(day.median()),
        "liquid_advances": int((liquid_day > 0).sum()),
        "liquid_declines": int((liquid_day < 0).sum()),
        "up_5pct": int((day >= 5).sum()),
        "down_5pct": int((day <= -5).sum()),
    }


# ================================================ WHOLE-MARKET TECHNICALS ====
# yfinance cannot do 2,000 companies -- it fetches one at a time. But a stack
# of bhavcopies gives the entire market's price history in N requests, and
# every indicator is then a vectorised pandas operation across the whole matrix.

def fetch_bhav_history(calendar_days: int = 120, progress=None) -> pd.DataFrame:
    """
    Stack N days of bhavcopy into one long frame:
    SYMBOL, TRADE_DATE, CLOSE_PRICE, TTL_TRD_QNTY, DELIV_PER, TURNOVER_LACS.
    """
    frames, checked = [], 0
    for offset in range(calendar_days):
        day = datetime.now() - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        checked += 1
        if progress:
            progress(offset / calendar_days, f"{day:%d %b} — {len(frames)} days loaded")
        df = _fetch_bhav_for(day)
        if df.empty:
            continue

        if "SERIES" in df.columns:
            df = df[df["SERIES"].astype(str).str.strip().str.upper() == "EQ"]
        keep = [c for c in ["SYMBOL", "CLOSE_PRICE", "TTL_TRD_QNTY", "DELIV_PER",
                            "TURNOVER_LACS", "TRADE_DATE"] if c in df.columns]
        frames.append(df[keep])
        time.sleep(0.35)  # stay under NSE's ~3 req/sec limit

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_market_technicals(hist: pd.DataFrame) -> pd.DataFrame:
    """
    RSI, moving averages, momentum, volume expansion and delivery trend for
    every stock in the market at once.
    """
    if hist.empty:
        return pd.DataFrame()

    closes = hist.pivot_table(index="TRADE_DATE", columns="SYMBOL",
                              values="CLOSE_PRICE", aggfunc="last").sort_index()
    vols = hist.pivot_table(index="TRADE_DATE", columns="SYMBOL",
                            values="TTL_TRD_QNTY", aggfunc="last").sort_index()
    deliv = hist.pivot_table(index="TRADE_DATE", columns="SYMBOL",
                             values="DELIV_PER", aggfunc="last").sort_index()

    n = len(closes)
    if n < 20:
        return pd.DataFrame()

    last = closes.iloc[-1]

    # Wilder-style RSI, vectorised across all columns
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(loss != 0, 100.0)

    def sma(w):
        return closes.rolling(w).mean().iloc[-1] if n >= w else pd.Series(dtype=float)

    def ret(days):
        if n <= days:
            return pd.Series(dtype=float)
        return ((last - closes.iloc[-days - 1]) / closes.iloc[-days - 1]) * 100

    sma20, sma50, sma200 = sma(20), sma(50), sma(200)
    avg_vol20 = vols.rolling(20).mean().iloc[-1] if n >= 20 else pd.Series(dtype=float)

    out = pd.DataFrame({
        "Symbol": last.index,
        "Close": last.values,
        "RSI (14)": rsi.reindex(last.index).values,
        "vs 20DMA %": (((last - sma20) / sma20) * 100).reindex(last.index).values,
        "vs 50DMA %": (((last - sma50) / sma50) * 100).reindex(last.index).values,
    })

    if len(sma200):
        out["vs 200DMA %"] = (((last - sma200) / sma200) * 100).reindex(last.index).values
    for label, days in [("1M %", 21), ("3M %", 63), ("6M %", 126)]:
        r = ret(days)
        if len(r):
            out[label] = r.reindex(last.index).values

    out["Range position %"] = (
        ((last - closes.min()) / (closes.max() - closes.min())) * 100
    ).reindex(last.index).values

    if len(avg_vol20):
        out["Vol vs 20d"] = (vols.iloc[-1] / avg_vol20).reindex(last.index).values

    if not deliv.empty and n >= 20:
        recent = deliv.tail(5).mean()
        base = deliv.tail(20).mean()
        out["Delivery %"] = recent.reindex(last.index).values
        out["Delivery trend"] = (recent - base).reindex(last.index).values

    out["Days of history"] = n
    return out.reset_index(drop=True)


# ============================================== STORED HISTORY & BACKTEST ====
# collect.py writes one gzipped CSV per month into data/history/. Reading from
# disk costs milliseconds instead of hundreds of NSE requests, and it makes
# testing a screen against a past date possible at all.

HISTORY_DIR = Path(__file__).parent / "data" / "history"


def load_stored_history() -> pd.DataFrame:
    """Everything the collector has gathered so far."""
    if not HISTORY_DIR.exists():
        return pd.DataFrame()
    files = sorted(HISTORY_DIR.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, skipinitialspace=True))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
    return df.dropna(subset=["DATE"]).sort_values(["DATE", "SYMBOL"])


def snapshot_at(hist: pd.DataFrame, as_of, lookback: int = 60) -> pd.DataFrame:
    """
    What every stock looked like on `as_of`, using only data available then.
    No forward-looking fields -- that is the whole point.
    """
    window = hist[hist["DATE"] <= pd.Timestamp(as_of)]
    if window.empty:
        return pd.DataFrame()

    dates = sorted(window["DATE"].unique())[-lookback:]
    window = window[window["DATE"].isin(dates)]

    closes = window.pivot_table(index="DATE", columns="SYMBOL",
                                values="CLOSE", aggfunc="last").sort_index()
    vols = window.pivot_table(index="DATE", columns="SYMBOL",
                              values="VOLUME", aggfunc="last").sort_index()
    deliv = window.pivot_table(index="DATE", columns="SYMBOL",
                               values="DELIV_PER", aggfunc="last").sort_index()

    n = len(closes)
    last = closes.iloc[-1]

    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rsi = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))
    rsi = rsi.where(loss != 0, 100.0)

    turnover = window[window["DATE"] == dates[-1]].set_index("SYMBOL")["TURNOVER_LACS"]

    snap = pd.DataFrame({
        "Symbol": last.index,
        "Close": last.values,
        "RSI (14)": rsi.reindex(last.index).values,
        "Turnover (Cr)": (turnover.reindex(last.index) / 100).values,
        "Delivery %": deliv.tail(5).mean().reindex(last.index).values,
    })

    if n >= 20:
        sma20 = closes.rolling(20).mean().iloc[-1]
        snap["vs 20DMA %"] = (((last - sma20) / sma20) * 100).reindex(last.index).values
        avg_vol = vols.rolling(20).mean().iloc[-1]
        snap["Vol vs 20d"] = (vols.iloc[-1] / avg_vol).reindex(last.index).values
        snap["Delivery trend"] = (
            deliv.tail(5).mean() - deliv.tail(20).mean()
        ).reindex(last.index).values
    if n >= 50:
        sma50 = closes.rolling(50).mean().iloc[-1]
        snap["vs 50DMA %"] = (((last - sma50) / sma50) * 100).reindex(last.index).values
    if n > 21:
        base = closes.iloc[-22]
        snap["1M %"] = (((last - base) / base) * 100).reindex(last.index).values

    return snap.reset_index(drop=True)


def forward_returns(hist: pd.DataFrame, symbols: list, start, end) -> pd.DataFrame:
    """
    What happened next.

    Stocks that stopped trading are reported as such rather than dropped.
    Dropping them is the mechanism by which backtests flatter themselves.
    """
    window = hist[(hist["DATE"] >= pd.Timestamp(start)) & (hist["DATE"] <= pd.Timestamp(end))]
    if window.empty:
        return pd.DataFrame()

    first = window.sort_values("DATE").groupby("SYMBOL").first()["CLOSE"]
    last = window.sort_values("DATE").groupby("SYMBOL").last()["CLOSE"]
    peak = window.groupby("SYMBOL")["CLOSE"].max()
    trough = window.groupby("SYMBOL")["CLOSE"].min()
    last_seen = window.groupby("SYMBOL")["DATE"].max()

    final_date = window["DATE"].max()
    rows = []
    for s in symbols:
        if s not in first.index:
            rows.append({"Symbol": s, "Return %": float("nan"),
                         "Status": "no data at entry"})
            continue
        still = last_seen[s] >= final_date - pd.Timedelta(days=7)
        rows.append({
            "Symbol": s,
            "Entry": float(first[s]),
            "Exit": float(last[s]),
            "Return %": ((last[s] - first[s]) / first[s]) * 100,
            "Peak gain %": ((peak[s] - first[s]) / first[s]) * 100,
            "Max drawdown %": ((trough[s] - first[s]) / first[s]) * 100,
            "Status": "trading" if still else f"stopped {last_seen[s]:%d %b %Y}",
        })
    return pd.DataFrame(rows)


def benchmark_return(hist: pd.DataFrame, start, end) -> float:
    """
    Median return of every stock that traded over the window.

    This is the honest bar: not whether your screen made money, but whether it
    beat picking at random from the same universe.
    """
    window = hist[(hist["DATE"] >= pd.Timestamp(start)) & (hist["DATE"] <= pd.Timestamp(end))]
    if window.empty:
        return float("nan")
    first = window.sort_values("DATE").groupby("SYMBOL").first()["CLOSE"]
    last = window.sort_values("DATE").groupby("SYMBOL").last()["CLOSE"]
    return float((((last - first) / first) * 100).median())


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


# ======================================================== CHART ANALYSIS =====
# Support/resistance from actual swing points, and range projection from
# realised volatility. Note what this is NOT: the range is a statement about
# how much this stock typically moves, not a claim about direction. A wide
# band means "size your position for this", not "it will reach the top".

def analyse_chart(ticker: str) -> dict | None:
    """Trend, momentum, levels and a volatility-based range for one stock."""
    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d")
    except Exception:
        return None
    if hist is None or len(hist) < 60:
        return None

    close = hist["Close"].dropna()
    high, low = hist["High"], hist["Low"]
    last = float(close.iloc[-1])

    # Average True Range -- realised daily movement, the basis for the range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    atr_pct = (atr / last) * 100

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    # Swing points: local extremes over a 5-day window
    win = 5
    highs = high.rolling(win * 2 + 1, center=True).max()
    lows = low.rolling(win * 2 + 1, center=True).min()
    swing_highs = sorted({round(float(v), 2) for v in high[(high == highs)].tail(40)})
    swing_lows = sorted({round(float(v), 2) for v in low[(low == lows)].tail(40)})

    resistance = [p for p in swing_highs if p > last][:3]
    support = [p for p in swing_lows if p < last][-3:][::-1]

    if sma200 and last > sma50 > sma200:
        trend = "Uptrend — above both the 50 and 200 day averages"
    elif sma200 and last < sma50 < sma200:
        trend = "Downtrend — below both the 50 and 200 day averages"
    elif last > sma20 and last > sma50:
        trend = "Rising — above short and medium term averages"
    elif last < sma20 and last < sma50:
        trend = "Falling — below short and medium term averages"
    else:
        trend = "Sideways — averages are tangled, no clear direction"

    # Range projection: ATR scaled by root-time. This is a volatility band,
    # symmetric by construction, and deliberately makes no directional call.
    def band(days: int) -> tuple:
        move = atr * (days ** 0.5)
        return round(last - move, 2), round(last + move, 2)

    return {
        "ticker": ticker,
        "last": last,
        "trend": trend,
        "rsi": _rsi(close),
        "atr": atr,
        "atr_pct": atr_pct,
        "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "support": support,
        "resistance": resistance,
        "range_1w": band(5),
        "range_1m": band(21),
        "hi52": float(close.max()), "lo52": float(close.min()),
        "series": close.tail(120),
        "ohlc": hist,
    }


# ====================================================== PATTERN DETECTION ====
# Real detection on real bars: trend structure, MA crossovers, range breakouts,
# double tops/bottoms, RSI divergence and the standard candlestick formations.
#
# On reliability: these are descriptive patterns with weak and unstable
# predictive power. Published tests of candlestick patterns on liquid equities
# generally find edges close to zero after costs. Treat the read below as a
# structured description of what the chart shows, with the odds barely tilted.

def detect_patterns(hist: pd.DataFrame) -> list:
    """Return a list of (pattern, direction, note) found in the price data."""
    found = []
    c, h, l, o = hist["Close"], hist["High"], hist["Low"], hist["Open"]
    v = hist["Volume"]
    n = len(c)
    if n < 60:
        return found

    last = float(c.iloc[-1])

    # --- Trend structure from swing points
    win = 5
    sh = h[(h == h.rolling(win * 2 + 1, center=True).max())].dropna()
    sl = l[(l == l.rolling(win * 2 + 1, center=True).min())].dropna()
    if len(sh) >= 2 and len(sl) >= 2:
        hh = sh.iloc[-1] > sh.iloc[-2]
        hl = sl.iloc[-1] > sl.iloc[-2]
        if hh and hl:
            found.append(("Higher highs and higher lows", "bullish",
                          "Textbook uptrend structure."))
        elif not hh and not hl:
            found.append(("Lower highs and lower lows", "bearish",
                          "Textbook downtrend structure."))
        else:
            found.append(("Mixed swing structure", "neutral",
                          "Highs and lows disagree — no clean trend."))

    # --- Double top / bottom
    if len(sh) >= 2:
        a, b = float(sh.iloc[-2]), float(sh.iloc[-1])
        if abs(a - b) / a < 0.03 and last < min(a, b) * 0.97:
            found.append(("Possible double top", "bearish",
                          f"Two highs near Rs {b:,.0f}, price has rolled over."))
    if len(sl) >= 2:
        a, b = float(sl.iloc[-2]), float(sl.iloc[-1])
        if abs(a - b) / a < 0.03 and last > max(a, b) * 1.03:
            found.append(("Possible double bottom", "bullish",
                          f"Two lows near Rs {b:,.0f}, price has lifted off."))

    # --- Moving average crossover, recent only
    if n >= 200:
        s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
        above = s50 > s200
        recent = above.tail(25)
        if recent.iloc[-1] and not recent.iloc[0]:
            found.append(("Golden cross (50 above 200)", "bullish",
                          "50-day crossed above the 200-day this month."))
        elif not recent.iloc[-1] and recent.iloc[0]:
            found.append(("Death cross (50 below 200)", "bearish",
                          "50-day crossed below the 200-day this month."))

    # --- Consolidation and breakout
    recent_range = (h.tail(20).max() - l.tail(20).min()) / last
    prior_range = (h.tail(60).max() - l.tail(60).min()) / last
    if recent_range < prior_range * 0.45:
        found.append(("Consolidation / tightening range", "neutral",
                      f"Last 20 days span only {recent_range * 100:.1f}%."))

    hi20 = float(h.iloc[-21:-1].max())
    lo20 = float(l.iloc[-21:-1].min())
    vol_ok = float(v.iloc[-1]) > float(v.tail(20).mean()) * 1.4
    if last > hi20:
        found.append(("Breakout above 20-day high", "bullish",
                      "With volume confirmation." if vol_ok
                      else "But volume is not confirming — weaker signal."))
    elif last < lo20:
        found.append(("Breakdown below 20-day low", "bearish",
                      "With volume confirmation." if vol_ok
                      else "But volume is not confirming — weaker signal."))

    # --- RSI divergence
    if n >= 40:
        delta = c.diff()
        g = delta.clip(lower=0).rolling(14).mean()
        ls = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_s = 100 - (100 / (1 + g / ls.replace(0, float("nan"))))
        p_now, p_prev = float(c.iloc[-1]), float(c.iloc[-21])
        r_now, r_prev = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-21])
        if p_now > p_prev * 1.02 and r_now < r_prev - 5:
            found.append(("Bearish RSI divergence", "bearish",
                          "Price made a higher high, momentum did not."))
        elif p_now < p_prev * 0.98 and r_now > r_prev + 5:
            found.append(("Bullish RSI divergence", "bullish",
                          "Price made a lower low, momentum did not."))

    # --- Candlestick formations on the latest bars
    body = (c - o).abs()
    rng = (h - l).replace(0, float("nan"))
    for i in [-1, -2]:
        oi, ci, hi_, li = float(o.iloc[i]), float(c.iloc[i]), float(h.iloc[i]), float(l.iloc[i])
        bi, ri = float(body.iloc[i]), float(rng.iloc[i])
        if ri != ri or ri == 0:
            continue
        when = "today" if i == -1 else "yesterday"

        if bi / ri < 0.1:
            found.append((f"Doji ({when})", "neutral", "Open and close nearly equal — indecision."))
        lower_wick = min(oi, ci) - li
        upper_wick = hi_ - max(oi, ci)
        if lower_wick > bi * 2 and upper_wick < bi:
            found.append((f"Hammer ({when})", "bullish", "Long lower wick — sellers rejected."))
        if upper_wick > bi * 2 and lower_wick < bi:
            found.append((f"Shooting star ({when})", "bearish", "Long upper wick — buyers rejected."))

    if n >= 2:
        o1, c1 = float(o.iloc[-2]), float(c.iloc[-2])
        o2, c2 = float(o.iloc[-1]), float(c.iloc[-1])
        if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:
            found.append(("Bullish engulfing", "bullish", "Today's up bar swallows yesterday's down bar."))
        if c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1:
            found.append(("Bearish engulfing", "bearish", "Today's down bar swallows yesterday's up bar."))

    return found


def chart_verdict(patterns: list, ca: dict) -> dict:
    """Weigh the detected patterns into a single positional read."""
    weights = {"bullish": 0, "bearish": 0}
    for _, direction, _ in patterns:
        if direction in weights:
            weights[direction] += 1

    # Trend and momentum carry more than any single candle
    if ca["sma200"] and ca["last"] > ca["sma50"] > ca["sma200"]:
        weights["bullish"] += 2
    elif ca["sma200"] and ca["last"] < ca["sma50"] < ca["sma200"]:
        weights["bearish"] += 2
    if ca["rsi"] > 70:
        weights["bearish"] += 1
    elif ca["rsi"] < 30:
        weights["bullish"] += 1

    net = weights["bullish"] - weights["bearish"]
    total = weights["bullish"] + weights["bearish"]

    if net >= 3:
        stance, tone = "Constructive", "up"
    elif net >= 1:
        stance, tone = "Mildly constructive", "up"
    elif net <= -3:
        stance, tone = "Negative", "down"
    elif net <= -1:
        stance, tone = "Mildly negative", "down"
    else:
        stance, tone = "Neutral / no edge", "flat"

    agreement = abs(net) / total if total else 0
    if agreement > 0.6 and total >= 4:
        conviction = "Signals mostly agree"
    elif total >= 3:
        conviction = "Signals are mixed"
    else:
        conviction = "Few signals — little to read"

    return {"stance": stance, "tone": tone, "net": net,
            "bull": weights["bullish"], "bear": weights["bearish"],
            "conviction": conviction}


# ================================================ MARKET-WIDE PATTERN SCAN ===
# Same patterns as the single-stock read, but computed columnwise across every
# stock at once. Looping 2,000 symbols would take minutes; this takes seconds.

def scan_market_patterns(hist: pd.DataFrame) -> pd.DataFrame:
    """Detect patterns for every stock in the stored history."""
    if hist.empty:
        return pd.DataFrame()

    # Only the last year matters for these patterns, and float32 halves the
    # memory. Without both, the pivots below exhaust a 1GB container.
    keep_dates = sorted(hist["DATE"].unique())[-260:]
    hist = hist[hist["DATE"].isin(keep_dates)]

    def wide(col):
        if col not in hist.columns:
            return None
        m = hist.pivot_table(index="DATE", columns="SYMBOL",
                             values=col, aggfunc="last").sort_index()
        return m.astype("float32")

    C, O, H, L, V = (wide(x) for x in ["CLOSE", "OPEN", "HIGH", "LOW", "VOLUME"])
    if C is None or len(C) < 60:
        return pd.DataFrame()

    have_ohlc = all(x is not None and x.notna().any().any() for x in (O, H, L))
    if not have_ohlc:
        O, H, L = C, C, C   # degrade gracefully; candle patterns will be inert

    n = len(C)
    last = C.iloc[-1]
    out = pd.DataFrame(index=C.columns)
    out["Close"] = last

    # --- momentum
    delta = C.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    ls = (-delta.clip(upper=0)).rolling(14).mean()
    rsi_full = 100 - (100 / (1 + g / ls.replace(0, float("nan"))))
    out["RSI"] = rsi_full.iloc[-1]

    # --- moving averages
    s50 = C.rolling(50).mean().iloc[-1] if n >= 50 else None
    if s50 is not None:
        out["Above 50DMA"] = last > s50
    if n >= 200:
        s200_series = C.rolling(200).mean()
        s200 = s200_series.iloc[-1]
        out["Above 200DMA"] = last > s200
        cross = (C.rolling(50).mean() > s200_series)
        out["Golden cross"] = cross.iloc[-1] & ~cross.iloc[-25]
        out["Death cross"] = ~cross.iloc[-1] & cross.iloc[-25]

    # --- breakouts against the prior 20 sessions
    hi20 = H.iloc[-21:-1].max()
    lo20 = L.iloc[-21:-1].min()
    out["Breakout 20d"] = last > hi20
    out["Breakdown 20d"] = last < lo20

    # --- volume confirmation
    avg_v = V.rolling(20).mean().iloc[-1]
    out["Vol vs 20d"] = V.iloc[-1] / avg_v
    out["Volume surge"] = out["Vol vs 20d"] >= 1.5

    # --- consolidation: recent range tight against the longer one
    r20 = (H.tail(20).max() - L.tail(20).min()) / last
    r60 = (H.tail(60).max() - L.tail(60).min()) / last
    out["Consolidating"] = r20 < r60 * 0.45

    # --- trend structure proxy: rolling extremes stepping up or down
    hi_now, hi_prev = H.tail(20).max(), H.iloc[-40:-20].max()
    lo_now, lo_prev = L.tail(20).min(), L.iloc[-40:-20].min()
    out["Higher highs & lows"] = (hi_now > hi_prev) & (lo_now > lo_prev)
    out["Lower highs & lows"] = (hi_now < hi_prev) & (lo_now < lo_prev)

    # --- RSI divergence over the last month
    if n >= 40:
        p_now, p_prev = C.iloc[-1], C.iloc[-21]
        r_now, r_prev = rsi_full.iloc[-1], rsi_full.iloc[-21]
        out["Bearish divergence"] = (p_now > p_prev * 1.02) & (r_now < r_prev - 5)
        out["Bullish divergence"] = (p_now < p_prev * 0.98) & (r_now > r_prev + 5)

    # --- candlestick formations on the last two bars
    if have_ohlc:
        o1, c1 = O.iloc[-2], C.iloc[-2]
        o2, c2 = O.iloc[-1], C.iloc[-1]
        out["Bullish engulfing"] = (c1 < o1) & (c2 > o2) & (c2 > o1) & (o2 < c1)
        out["Bearish engulfing"] = (c1 > o1) & (c2 < o2) & (c2 < o1) & (o2 > c1)

        body = (c2 - o2).abs()
        rng = (H.iloc[-1] - L.iloc[-1]).replace(0, float("nan"))
        upper = H.iloc[-1] - pd.concat([o2, c2], axis=1).max(axis=1)
        lower = pd.concat([o2, c2], axis=1).min(axis=1) - L.iloc[-1]
        out["Hammer"] = (lower > body * 2) & (upper < body)
        out["Shooting star"] = (upper > body * 2) & (lower < body)
        out["Doji"] = (body / rng) < 0.1

    # --- weighted stance, mirroring the single-stock logic
    bull_cols = ["Golden cross", "Breakout 20d", "Higher highs & lows",
                 "Bullish divergence", "Bullish engulfing", "Hammer", "Above 50DMA"]
    bear_cols = ["Death cross", "Breakdown 20d", "Lower highs & lows",
                 "Bearish divergence", "Bearish engulfing", "Shooting star"]

    bull = sum(out[c].fillna(False).astype(int) for c in bull_cols if c in out)
    bear = sum(out[c].fillna(False).astype(int) for c in bear_cols if c in out)
    if "Above 200DMA" in out and s50 is not None:
        bull = bull + ((last > s50) & (out["Above 200DMA"].fillna(False))).astype(int)
        bear = bear + ((last < s50) & (~out["Above 200DMA"].fillna(True))).astype(int)
    bear = bear + (out["RSI"] > 70).fillna(False).astype(int)
    bull = bull + (out["RSI"] < 30).fillna(False).astype(int)

    out["Bull signals"] = bull
    out["Bear signals"] = bear
    out["Score"] = bull - bear
    out["Stance"] = pd.cut(
        out["Score"], bins=[-99, -3, -1, 0, 2, 99],
        labels=["Negative", "Mildly negative", "Neutral",
                "Mildly constructive", "Constructive"],
    ).astype(str)

    out = out.reset_index().rename(columns={"SYMBOL": "Symbol", "index": "Symbol"})
    out["Has OHLC"] = have_ohlc
    return out


# ============================================ IMPLIED VOLATILITY (BSM) =======
# The F&O bhavcopy carries prices and open interest but no implied volatility,
# so it is solved from the closing price. Bisection rather than Newton: slower,
# but it cannot diverge on the deep-ITM and near-expiry contracts where Newton
# routinely blows up.

def _norm_cdf(x):
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot, strike, t_years, vol, rate=0.065, is_call=True):
    """Black-Scholes-Merton price. No dividend yield -- Indian options are on
    futures-style underlyings often enough that adding one would be guesswork."""
    import math
    if t_years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        intrinsic = (spot - strike) if is_call else (strike - spot)
        return max(intrinsic, 0.0)

    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * math.sqrt(t_years))
    d2 = d1 - vol * math.sqrt(t_years)
    disc = math.exp(-rate * t_years)
    if is_call:
        return spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    return strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def implied_vol(price, spot, strike, t_years, rate=0.065, is_call=True):
    """
    Solve for the volatility that reproduces the observed price.

    Returns NaN where no solution exists — a price below intrinsic value, a
    contract that did not trade, or an expiry too close to be meaningful.
    Returning a number there would be inventing one.
    """
    if not all(map(lambda v: v == v, [price, spot, strike, t_years])):
        return float("nan")
    if price <= 0 or spot <= 0 or strike <= 0 or t_years <= 1 / 365:
        return float("nan")

    intrinsic = max((spot - strike) if is_call else (strike - spot), 0.0)
    if price < intrinsic - 1e-6:
        return float("nan")          # price below intrinsic: stale or bad print

    lo, hi = 1e-4, 5.0
    if bs_price(spot, strike, t_years, hi, rate, is_call) < price:
        return float("nan")          # beyond 500% vol: not a real quote

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(spot, strike, t_years, mid, rate, is_call) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    vol = 0.5 * (lo + hi)
    return vol * 100 if 0.001 < vol < 4.99 else float("nan")


def add_implied_vols(chain: pd.DataFrame, as_of) -> pd.DataFrame:
    """Attach an IV column to an F&O chain slice."""
    if chain.empty:
        return chain
    out = chain.copy()
    ref = pd.Timestamp(as_of)
    exp = pd.to_datetime(out["EXPIRY"], errors="coerce")
    t = (exp - ref).dt.days.clip(lower=0) / 365.0

    ivs = []
    for price, spot, strike, tt, opt in zip(
            out["CLOSE"], out["UNDERLYING"], out["STRIKE"], t, out["OPT"]):
        ivs.append(implied_vol(price, spot, strike, tt,
                               is_call=str(opt).upper().startswith("C")))
    out["IV %"] = ivs
    out["Days to expiry"] = (exp - ref).dt.days
    return out


# ================================================================ GREEKS =====
# Sensitivities from the same Black-Scholes model used for implied volatility.
# Theta is the one option buyers underestimate: it is reported per calendar
# day here, because that is how it is actually felt.

def _norm_pdf(x):
    import math
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def greeks(spot, strike, t_years, vol, rate=0.065, is_call=True):
    """vol as a decimal (0.15 = 15%). Returns delta, gamma, theta/day, vega/1%."""
    import math
    if t_years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return {"delta": float("nan"), "gamma": float("nan"),
                "theta": float("nan"), "vega": float("nan")}

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc = math.exp(-rate * t_years)
    pdf = _norm_pdf(d1)

    delta = _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0
    gamma = pdf / (spot * vol * sqrt_t)
    vega = spot * pdf * sqrt_t / 100.0          # per 1 volatility point

    theta_year = -(spot * pdf * vol) / (2 * sqrt_t)
    if is_call:
        theta_year -= rate * strike * disc * _norm_cdf(d2)
    else:
        theta_year += rate * strike * disc * _norm_cdf(-d2)

    return {"delta": delta, "gamma": gamma,
            "theta": theta_year / 365.0, "vega": vega}


def scenario_table(spot, strike, premium, expiry_days, hold_days, vol_pct,
                   is_call, direction, qty, atr_pct, iv_shift=0.0):
    """
    What the option is worth at a range of prices, after `hold_days`.

    The scenarios come from the underlying's own realised volatility (ATR
    scaled by root-time), so the spread is calibrated to how much this stock
    actually moves rather than to round numbers.
    """
    import numpy as np
    move = atr_pct / 100.0 * spot * (hold_days ** 0.5)
    if move <= 0 or not np.isfinite(move):
        move = spot * 0.03

    t_left = max((expiry_days - hold_days), 0) / 365.0
    vol = max((vol_pct + iv_shift), 0.1) / 100.0
    sign = 1 if direction == "long" else -1

    rows = []
    for mult, label in [(-2, "Sharp fall"), (-1, "Fall"), (-0.5, "Drift down"),
                        (0, "Unchanged"), (0.5, "Drift up"), (1, "Rise"),
                        (2, "Sharp rise")]:
        s = max(spot + mult * move, 0.01)
        value = bs_price(s, strike, t_left, vol, is_call=is_call)
        pnl = sign * (value - premium) * qty
        rows.append({
            "Scenario": label,
            "Underlying": s,
            "Move %": ((s - spot) / spot) * 100,
            "Option value": value,
            "P&L": pnl,
            "Return %": (pnl / (premium * qty)) * 100 if premium else float("nan"),
        })
    return pd.DataFrame(rows)


# =============================================================== OPTIONS =====
# Payoff maths for arbitrary multi-leg positions. Everything here is exact
# arithmetic on expiry values -- no model, no assumptions, no forecast. What
# it cannot tell you is the probability of any outcome; it tells you what each
# outcome pays.

LOT_SIZES = {
    "NIFTY": 75, "BANKNIFTY": 35, "FINNIFTY": 65, "MIDCPNIFTY": 140,
    "RELIANCE": 500, "TCS": 175, "INFY": 400, "HDFCBANK": 550,
    "ICICIBANK": 700, "SBIN": 750, "ITC": 1600, "LT": 150,
    "BHARTIARTL": 475, "BEL": 2850, "HAL": 150,
}


def leg_payoff(spot, kind: str, strike: float, premium: float,
               direction: str, qty: int):
    """
    Value of one leg at expiry, per the whole position (qty = lots x lot size).
    kind: call | put | future.  direction: long | short.
    """
    import numpy as np
    s = np.asarray(spot, dtype=float)

    if kind == "call":
        intrinsic = np.maximum(s - strike, 0.0)
    elif kind == "put":
        intrinsic = np.maximum(strike - s, 0.0)
    else:  # future or stock -- premium field carries the entry price
        intrinsic = s - premium
        return (intrinsic if direction == "long" else -intrinsic) * qty

    if direction == "long":
        return (intrinsic - premium) * qty
    return (premium - intrinsic) * qty


def position_payoff(legs: list, spot_range):
    """Total payoff across all legs over a range of expiry prices."""
    import numpy as np
    total = np.zeros(len(spot_range), dtype=float)
    for lg in legs:
        total = total + leg_payoff(
            spot_range, lg["kind"], lg["strike"], lg["premium"],
            lg["direction"], lg["qty"])
    return total


def analyse_position(legs: list, underlying: float):
    """
    Breakevens, max profit and max loss for a multi-leg position.

    Unbounded outcomes are reported as unlimited rather than as the edge of
    whatever price range happened to be sampled -- a naked short call does not
    have a maximum loss, and rounding one in would be a dangerous fiction.
    """
    import numpy as np
    if not legs:
        return None

    strikes = [lg["strike"] for lg in legs if lg["kind"] in ("call", "put")]
    ref = max(strikes + [underlying])

    # The grid starts at zero deliberately. A share cannot go below zero, so
    # the downside is ALWAYS bounded -- its worst case is the value at S=0.
    # Only the upside can be genuinely unlimited. Sampling from a non-zero
    # floor would silently understate the loss on anything holding the
    # underlying or short puts.
    grid = np.linspace(0.0, ref * 2.0, 6000)
    pay = position_payoff(legs, grid)

    # Slope in the far right tail decides whether the upside is open, and in
    # which direction.
    right_slope = (pay[-1] - pay[-2]) / (grid[-1] - grid[-2])
    unlimited_up = right_slope > 1e-6      # profit grows without bound
    unlimited_loss = right_slope < -1e-6   # loss grows without bound

    max_profit = float("inf") if unlimited_up else float(pay.max())
    max_loss = float("-inf") if unlimited_loss else float(pay.min())
    unlimited_down = unlimited_loss

    # Breakevens: sign changes on the grid, refined by linear interpolation
    breakevens = []
    sign = np.sign(pay)
    for i in range(len(grid) - 1):
        if sign[i] == 0:
            breakevens.append(float(grid[i]))
        elif sign[i] * sign[i + 1] < 0:
            x0, x1, y0, y1 = grid[i], grid[i + 1], pay[i], pay[i + 1]
            breakevens.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    breakevens = sorted({round(b, 2) for b in breakevens})

    net_premium = sum(
        (-lg["premium"] if lg["direction"] == "long" else lg["premium"]) * lg["qty"]
        for lg in legs if lg["kind"] in ("call", "put")
    )

    return {
        "grid": grid, "payoff": pay,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "unlimited_up": unlimited_up, "unlimited_down": unlimited_down,
        "worst_at_zero": float(pay[0]),
        "breakevens": breakevens,
        "net_premium": net_premium,
        "payoff_at_spot": float(np.interp(underlying, grid, pay)),
    }


# ============================================ DISK-FIRST MARKET DATA =========
# NSE rate-limits this app's host, but the GitHub Action collector reaches it
# fine. So stored history is the PRIMARY source and live NSE is the fallback,
# not the other way round. Everything below reads from disk in milliseconds.

def history_to_bhav_schema(hist: pd.DataFrame) -> pd.DataFrame:
    """Rename stored columns to the bhavcopy names the analytics expect."""
    if hist.empty:
        return hist
    return hist.rename(columns={
        "DATE": "TRADE_DATE", "CLOSE": "CLOSE_PRICE",
        "VOLUME": "TTL_TRD_QNTY",
    })


def screen_from_history(hist: pd.DataFrame):
    """
    The whole-market screener table, built from the newest stored day.
    Returns (screen, date) or (empty, None).
    """
    if hist.empty:
        return pd.DataFrame(), None

    latest_date = hist["DATE"].max()
    day = hist[hist["DATE"] == latest_date]
    if day.empty:
        return pd.DataFrame(), None

    out = pd.DataFrame({
        "Symbol": day["SYMBOL"].values,
        "Close": day["CLOSE"].values,
        "Day %": (((day["CLOSE"] - day["PREV_CLOSE"]) / day["PREV_CLOSE"]) * 100).values
        if "PREV_CLOSE" in day.columns else float("nan"),
        "Volume": day["VOLUME"].values if "VOLUME" in day.columns else float("nan"),
        "Turnover (Cr)": (day["TURNOVER_LACS"] / 100).values
        if "TURNOVER_LACS" in day.columns else float("nan"),
        "Delivery %": day["DELIV_PER"].values if "DELIV_PER" in day.columns else float("nan"),
    })

    dates = sorted(hist["DATE"].unique())

    # Returns against earlier stored dates
    for label, back in [("1M %", 21), ("3M %", 63), ("6M %", 126), ("1Y %", 250)]:
        if len(dates) > back:
            base = hist[hist["DATE"] == dates[-back - 1]].set_index("SYMBOL")["CLOSE"]
            b = base.reindex(out["Symbol"]).values
            out[label] = ((out["Close"].values - b) / b) * 100

    # 52-week high and low, from the stored window. If fewer than ~250 trading
    # days are on disk this is the high and low of whatever was collected, not
    # a true year -- so the actual span is reported alongside it rather than
    # letting the label imply more than the data supports.
    window = hist[hist["DATE"] >= pd.Timestamp(latest_date) - pd.Timedelta(days=365)]
    if not window.empty:
        agg = window.groupby("SYMBOL")["CLOSE"].agg(["max", "min"])
        hi = agg["max"].reindex(out["Symbol"]).values
        lo = agg["min"].reindex(out["Symbol"]).values
        out["52W High"] = hi
        out["52W Low"] = lo
        out["Off 52W High %"] = ((out["Close"].values - hi) / hi) * 100
        out["Above 52W Low %"] = ((out["Close"].values - lo) / lo) * 100
        # Guard the divide: a stock that never moved has zero span.
        import numpy as np
        span = np.where((hi - lo) == 0, np.nan, hi - lo)
        out["52W position %"] = ((out["Close"].values - lo) / span) * 100

        out.attrs["hl_days"] = int(window["DATE"].nunique())

    return out.reset_index(drop=True), pd.Timestamp(latest_date).date()


def technicals_from_history(hist: pd.DataFrame):
    """Whole-market technicals off stored history — no network at all."""
    if hist.empty:
        return pd.DataFrame(), 0
    bhav = history_to_bhav_schema(hist)
    return compute_market_technicals(bhav), bhav["TRADE_DATE"].nunique()


FLOWS_PATH = Path(__file__).parent / "data" / "flows.csv"


def load_stored_flows() -> pd.DataFrame:
    """FII/DII history written by the collector."""
    if not FLOWS_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(FLOWS_PATH, skipinitialspace=True)
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        return df.dropna(subset=["DATE"]).sort_values("DATE")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def symbol_universe() -> pd.DataFrame:
    """
    Every tradable NSE symbol, from whichever source answers.

    NSE's equity-list file refuses requests often enough that it cannot be the
    only source. The stored history is the better fallback -- it needs no
    network at all and contains every symbol that actually traded.
    """
    # 1. NSE's official list (has company names)
    uni = load_universe()
    if not uni.empty and "Symbol" in uni.columns:
        uni = uni.copy()
        uni["Source"] = "NSE list"
        return uni

    # 2. Stored history on disk -- no network needed
    try:
        if HISTORY_DIR.exists():
            syms = set()
            for f in sorted(HISTORY_DIR.glob("*.csv.gz"))[-3:]:
                syms.update(pd.read_csv(f, usecols=["SYMBOL"],
                                        skipinitialspace=True)["SYMBOL"].unique())
            if syms:
                return pd.DataFrame({"Symbol": sorted(syms),
                                     "Company": sorted(syms),
                                     "Source": "stored history"})
    except Exception:
        pass

    # 3. Today's bhavcopy, if the screener already fetched it
    try:
        screen, _, _ = load_market_screen()
        if not screen.empty and "Symbol" in screen.columns:
            out = screen[["Symbol"]].copy()
            out["Company"] = (screen["Company"] if "Company" in screen.columns
                              else screen["Symbol"])
            out["Source"] = "today's bhavcopy"
            return out
    except Exception:
        pass

    return pd.DataFrame(columns=["Symbol", "Company", "Source"])



@contextmanager
def safe_tab(name: str):
    """
    Isolate a tab. Without this, an exception anywhere stops the script and
    every tab further down silently renders blank -- which looks identical to
    "no data" and is why the last three tabs appeared empty.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        st.error(f"The {name} tab hit an error: {type(exc).__name__}: {exc}")
        with st.expander("Details"):
            st.code(traceback.format_exc())
        st.caption("Other tabs are unaffected. Send me this message and I'll fix it.")



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
    """Live NSE if it answers, otherwise the collector's stored history."""
    live = fetch_fii_dii_cash()
    if not live.empty:
        return live

    stored = load_stored_flows()
    if stored.empty:
        return pd.DataFrame()
    latest = stored[stored["DATE"] == stored["DATE"].max()]
    return pd.DataFrame({
        "Date": latest["DATE"].dt.strftime("%d-%b-%Y"),
        "Participant": latest["PARTICIPANT"],
        "Buy (Cr)": latest["BUY_CR"],
        "Sell (Cr)": latest["SELL_CR"],
        "Net (Cr)": latest["BUY_CR"] - latest["SELL_CR"],
    }).reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_fno_oi():
    return fetch_fno_participant_oi()


@st.cache_data(ttl=21600, show_spinner=False)
def cached_setups(min_turnover: float, min_rr: float):
    inputs = compute_setup_inputs(load_stored_history())
    if inputs.empty:
        return pd.DataFrame()
    return weekly_setups(inputs, min_turnover, min_rr)


@st.cache_data(ttl=1800)
def cached_pattern_scan():
    return scan_market_patterns(load_stored_history())


OPTIONS_DIR = Path(__file__).parent / "data" / "options"
IV_PATH = Path(__file__).parent / "data" / "iv_history.csv"


@st.cache_data(ttl=1800)
def load_iv_history() -> pd.DataFrame:
    if not IV_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(IV_PATH, skipinitialspace=True)
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        return df.dropna(subset=["DATE"]).sort_values("DATE")
    except Exception:
        return pd.DataFrame()


FNO_DIR = Path(__file__).parent / "data" / "fno"
LOTS_PATH = Path(__file__).parent / "data" / "lot_sizes.csv"


@st.cache_data(ttl=1800)
def load_fno_latest() -> pd.DataFrame:
    """Newest stored F&O bhavcopy — every underlying, strike and expiry."""
    if not FNO_DIR.exists():
        return pd.DataFrame()
    files = sorted(FNO_DIR.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[-1], skipinitialspace=True)
        return df[df["DATE"] == df["DATE"].max()].copy()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def load_lot_sizes() -> dict:
    """Exchange lot sizes if collected, else the small built-in fallback."""
    if LOTS_PATH.exists():
        try:
            df = pd.read_csv(LOTS_PATH, skipinitialspace=True)
            return {str(r["SYMBOL"]): int(r["LOT_SIZE"])
                    for _, r in df.iterrows() if pd.notna(r["LOT_SIZE"])}
        except Exception:
            pass
    return dict(LOT_SIZES)


@st.cache_data(ttl=1800)
def load_latest_chain() -> pd.DataFrame:
    """Most recent option chain snapshot the collector stored."""
    if not OPTIONS_DIR.exists():
        return pd.DataFrame()
    files = sorted(OPTIONS_DIR.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[-1], skipinitialspace=True)
        return df[df["DATE"] == df["DATE"].max()]
    except Exception:
        return pd.DataFrame()


def iv_rank(series: pd.Series, current: float):
    """
    IV rank and percentile against the stock's own history.

    Rank places today between the period low and high. Percentile is the share
    of days that were lower. They disagree when the distribution is skewed,
    which is exactly when it matters.
    """
    s = series.dropna()
    if len(s) < 20 or pd.isna(current):
        return None, None, len(s)
    lo, hi = float(s.min()), float(s.max())
    rank = ((current - lo) / (hi - lo)) * 100 if hi > lo else float("nan")
    pctile = (s < current).mean() * 100
    return rank, pctile, len(s)


def classify_oi(price_chg, oi_chg):
    """The standard four-way read of price against open interest."""
    if pd.isna(price_chg) or pd.isna(oi_chg):
        return "—"
    if price_chg > 0 and oi_chg > 0:
        return "Long buildup"
    if price_chg < 0 and oi_chg > 0:
        return "Short buildup"
    if price_chg > 0 and oi_chg < 0:
        return "Short covering"
    if price_chg < 0 and oi_chg < 0:
        return "Long unwinding"
    return "—"


@st.cache_data(ttl=1800)
def cached_history():
    return load_stored_history()


@st.cache_data(ttl=21600, show_spinner=False)
def load_market_technicals(calendar_days: int):
    """Stored history first — it needs no network and is far faster."""
    stored = load_stored_history()
    if not stored.empty:
        tech, days = technicals_from_history(stored)
        if not tech.empty:
            return tech, days

    hist = fetch_bhav_history(calendar_days)
    if hist.empty:
        return pd.DataFrame(), 0
    return compute_market_technicals(hist), hist["TRADE_DATE"].nunique()


@st.cache_data(ttl=900)
def load_chart_analysis(ticker: str):
    return analyse_chart(ticker)


@st.cache_data(ttl=900)
def load_technicals(tickers: tuple):
    return fetch_technicals(list(tickers))


@st.cache_data(ttl=86400)
def load_universe():
    return fetch_nse_universe()


@st.cache_data(ttl=3600)
def load_market_screen():
    """
    Whole-market table, its date, and a log of what was tried.

    Stored history first: it is instant and does not depend on NSE answering.
    Only if the store is empty do we ask NSE directly.
    """
    stored = load_stored_history()
    if not stored.empty:
        screen, date = screen_from_history(stored)
        if not screen.empty:
            log = [("stored history", f"{stored['DATE'].nunique()} days on disk",
                    f"latest {date}")]
            uni = load_universe()
            if not uni.empty:
                screen = screen.merge(uni[["Symbol", "Company"]],
                                      on="Symbol", how="left")
                cols = ["Symbol", "Company"] + [c for c in screen.columns
                                                if c not in ("Symbol", "Company")]
                screen = screen[cols]
            return screen, date, log

    latest, latest_date, attempts = fetch_latest_bhavcopy()
    attempts.insert(0, ("stored history", "empty", "falling back to live NSE"))
    if latest.empty:
        return pd.DataFrame(), None, attempts

    prior = pd.DataFrame()
    for offset in range(30, 38):
        day = datetime.now() - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        prior = _fetch_bhav_for(day)
        if not prior.empty:
            break

    screen = build_market_screen(latest, prior, attempts)

    universe = load_universe()
    attempts.append(("build", "universe rows", str(len(universe))))
    if not universe.empty and not screen.empty:
        screen = screen.merge(
            universe[["Symbol", "Company"]], on="Symbol", how="left"
        )
        cols = ["Symbol", "Company"] + [
            c for c in screen.columns if c not in ("Symbol", "Company")
        ]
        screen = screen[cols]

    attempts.append(("build", "final rows", str(len(screen))))
    return screen, latest_date, attempts


# ========================================================= WEEKLY SETUPS =====
# Named chart configurations across the whole market, each with levels taken
# from actual price structure rather than round numbers: stops sit below the
# swing that would invalidate the idea, targets at the next structural level.
#
# Risk/reward here is arithmetic on those levels. It is not a probability. A
# 3:1 setup that works one time in five loses money. The Backtest tab is the
# only way to find out which of these has actually paid on your names.

def compute_setup_inputs(hist: pd.DataFrame) -> pd.DataFrame:
    """Everything the setup rules need, computed across all stocks at once."""
    if hist.empty:
        return pd.DataFrame()

    keep = sorted(hist["DATE"].unique())[-260:]
    h = hist[hist["DATE"].isin(keep)]

    def wide(col):
        if col not in h.columns:
            return None
        return h.pivot_table(index="DATE", columns="SYMBOL", values=col,
                             aggfunc="last").sort_index().astype("float32")

    C, H, L, V, D = (wide(x) for x in ["CLOSE", "HIGH", "LOW", "VOLUME", "DELIV_PER"])
    if C is None or len(C) < 60:
        return pd.DataFrame()
    if H is None or not H.notna().any().any():
        H = C
    if L is None or not L.notna().any().any():
        L = C

    n = len(C)
    last = C.iloc[-1]
    out = pd.DataFrame({"Symbol": C.columns, "Close": last.values})

    # True range -> ATR. Falls back to close-to-close where OHLC is absent.
    tr = pd.concat([(H - L).stack(), (H - C.shift()).abs().stack(),
                    (L - C.shift()).abs().stack()], axis=1).max(axis=1).unstack()
    atr = tr.rolling(14).mean().iloc[-1]
    out["ATR"] = atr.reindex(C.columns).values
    out["ATR %"] = (atr / last * 100).reindex(C.columns).values

    for w in (20, 50, 200):
        if n >= w:
            out[f"SMA{w}"] = C.rolling(w).mean().iloc[-1].reindex(C.columns).values

    delta = C.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    ls = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + g / ls.replace(0, float("nan"))))
    out["RSI"] = rsi.iloc[-1].reindex(C.columns).values

    out["High20"] = H.iloc[-21:-1].max().reindex(C.columns).values
    out["Low20"] = L.iloc[-21:-1].min().reindex(C.columns).values
    out["High52"] = C.max().reindex(C.columns).values
    out["Low52"] = C.min().reindex(C.columns).values

    # Most recent swing low: the lowest point of the last 10 sessions, which is
    # where a long idea stops being right.
    out["SwingLow"] = L.tail(10).min().reindex(C.columns).values
    out["SwingHigh"] = H.tail(10).max().reindex(C.columns).values

    r20 = (H.tail(20).max() - L.tail(20).min()) / last
    r60 = (H.tail(60).max() - L.tail(60).min()) / last
    out["Tight"] = (r20 < r60 * 0.45).reindex(C.columns).values
    out["RangeHeight"] = (H.tail(20).max() - L.tail(20).min()).reindex(C.columns).values

    if V is not None:
        avgv = V.rolling(20).mean().iloc[-1]
        out["VolX"] = (V.iloc[-1] / avgv).reindex(C.columns).values
        out["Turnover proxy"] = (V.iloc[-1] * last).reindex(C.columns).values

    if D is not None and D.notna().any().any():
        out["Delivery %"] = D.tail(5).mean().reindex(C.columns).values
        out["Delivery trend"] = (D.tail(5).mean() - D.tail(20).mean()
                                 ).reindex(C.columns).values

    for label, back in [("1M %", 21), ("3M %", 63)]:
        if n > back:
            base = C.iloc[-back - 1]
            out[label] = (((last - base) / base) * 100).reindex(C.columns).values

    return out


def weekly_setups(inp: pd.DataFrame, min_turnover_cr: float = 1.0,
                  min_rr: float = 1.5) -> pd.DataFrame:
    """
    Classify each stock into a named setup, with levels.

    A stock can only appear once — the first rule that matches wins, ordered
    from most specific to least. Otherwise the same name shows up under four
    headings and the list stops meaning anything.
    """
    if inp.empty:
        return pd.DataFrame()

    df = inp.copy()
    if "Turnover proxy" in df.columns:
        df = df[df["Turnover proxy"].fillna(0) >= min_turnover_cr * 1e7]

    rows = []
    for _, r in df.iterrows():
        close, atr = r.get("Close"), r.get("ATR")
        if not (close == close) or not (atr == atr) or atr <= 0:
            continue

        s50, s200 = r.get("SMA50", float("nan")), r.get("SMA200", float("nan"))
        rsi, volx = r.get("RSI", float("nan")), r.get("VolX", 1.0)
        above200 = s200 == s200 and close > s200
        above50 = s50 == s50 and close > s50
        dtrend = r.get("Delivery trend", 0) or 0

        setup = stop = target = note = None

        # 1. Breakout from a tight range, confirmed by volume
        if r.get("Tight") and close > r.get("High20", 1e18) and volx >= 1.5:
            setup = "Range breakout"
            stop = min(r["Low20"], close - 1.5 * atr)
            target = close + max(r.get("RangeHeight", 0), 2 * atr)
            note = "Broke a tightening range on above-average volume."

        # 2. New 52-week high in an established uptrend
        elif above200 and close >= r.get("High52", 1e18) * 0.995 and volx >= 1.2:
            setup = "52-week high breakout"
            stop = close - 2 * atr
            target = close + 4 * atr
            note = "At a yearly high with the long-term trend already up."

        # 3. Pullback to the 50-day inside a genuine uptrend.
        # "Above the 200DMA" alone is far too loose -- roughly half the market
        # qualifies on any given day. This also demands the 50 above the 200
        # and real three-month progress, so a drifting stock does not read as
        # a pullback in an uptrend it never had.
        elif (above200 and s50 == s50 and s50 > s200
              and abs(close - s50) / s50 < 0.03
              and 38 <= rsi <= 58
              and (r.get("3M %", -1e9) or -1e9) > 5):
            setup = "Pullback to 50DMA"
            stop = min(r.get("SwingLow", close - 2 * atr), s50 - 1.5 * atr)
            target = max(r.get("SwingHigh", close + 3 * atr), close + 2.5 * atr)
            note = "Uptrend intact, price has come back to the 50-day line."

        # 4. Oversold while the long-term trend is still up
        elif (above200 and rsi < 35 and s50 == s50 and s50 > s200
              and (r.get("3M %", -1e9) or -1e9) > -15):
            setup = "Oversold in uptrend"
            stop = r.get("SwingLow", close - 2 * atr)
            target = s50 if s50 == s50 and s50 > close else close + 3 * atr
            note = "Above the 200-day but short-term momentum is washed out."

        # 5. Quiet accumulation: delivery rising without a price spike
        elif dtrend > 5 and volx >= 1.3 and above50:
            setup = "Delivery accumulation"
            stop = close - 2 * atr
            target = close + 3 * atr
            note = ("Rising share of volume taken as delivery — buyers holding "
                    "rather than trading.")

        # 6. Breaking down
        elif close < r.get("Low20", -1e18) and not above50:
            setup = "Breakdown"
            stop = close + 1.5 * atr
            target = close - 3 * atr
            note = "Below the 20-day low with the medium trend already down."

        if setup is None:
            continue

        direction = "short" if setup == "Breakdown" else "long"
        if direction == "long":
            risk, reward = close - stop, target - close
        else:
            risk, reward = stop - close, close - target
        if risk <= 0:
            continue

        rr = reward / risk
        if rr < min_rr:
            continue

        # Confirmation score: how many independent things agree. Not a
        # probability -- a count of corroborating signals, nothing more.
        score = 0
        if volx == volx and volx >= 1.5:
            score += 1
        if dtrend == dtrend and dtrend > 3:
            score += 1
        if direction == "long" and above200:
            score += 1
        if direction == "short" and not above200:
            score += 1
        if rr >= 2.5:
            score += 1
        risk_pct = (risk / close) * 100
        if 1.0 <= risk_pct <= 8.0:      # a stop that is neither tight nor absurd
            score += 1

        rows.append({
            "Symbol": r["Symbol"], "Setup": setup, "Bias": direction,
            "Confirmations": score,
            "Entry": close, "Stop": stop, "Target": target,
            "Risk %": risk_pct,
            "Reward %": (reward / close) * 100,
            "R:R": rr,
            "RSI": rsi, "Vol x": volx,
            "Delivery trend": r.get("Delivery trend", float("nan")),
            "ATR %": r.get("ATR %", float("nan")),
            "1M %": r.get("1M %", float("nan")),
            "Note": note,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Confirmations", "R:R"], ascending=[False, False])


def position_size(capital: float, risk_pct: float, entry: float, stop: float):
    """
    Shares such that being stopped out costs a fixed share of capital.

    This is the part that decides outcomes far more than setup selection, and
    it is arithmetic rather than opinion.
    """
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or entry <= 0:
        return 0, 0.0
    budget = capital * (risk_pct / 100.0)
    shares = int(budget // risk_per_share)
    return shares, shares * entry


# ================================================ DISCLOSURES & CALENDAR =====

DEALS_PATH = DATA_DIR / "deals.csv"
INSIDER_PATH = DATA_DIR / "insider.csv"
ACTIONS_PATH = DATA_DIR / "corporate_actions.csv"
MEETINGS_PATH = DATA_DIR / "board_meetings.csv"
JOURNAL_PATH = DATA_DIR / "journal.csv"


def _load_csv(path: Path, date_cols=()) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, skipinitialspace=True)
    except Exception:
        return pd.DataFrame()
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    return df


@st.cache_data(ttl=1800)
def load_deals():
    return _load_csv(DEALS_PATH, ["DATE"])


@st.cache_data(ttl=1800)
def load_insider():
    return _load_csv(INSIDER_PATH, ["DATE"])


@st.cache_data(ttl=1800)
def load_actions():
    return _load_csv(ACTIONS_PATH, ["EX_DATE", "RECORD_DATE"])


@st.cache_data(ttl=1800)
def load_meetings():
    return _load_csv(MEETINGS_PATH, ["MEETING_DATE"])


def events_before(symbol: str, days: int) -> pd.DataFrame:
    """
    Scheduled events for a symbol within the next N days.

    Used by the options calculator: an earnings date inside your holding
    period is precisely when implied volatility collapses, and a scenario
    priced without knowing that is misleading.
    """
    base = str(symbol).replace(".NS", "").upper()
    today = pd.Timestamp(datetime.now().date())
    cutoff = today + pd.Timedelta(days=days)
    out = []

    mtg = load_meetings()
    if not mtg.empty and "SYMBOL" in mtg.columns:
        m = mtg[(mtg["SYMBOL"].astype(str).str.upper() == base)
                & (mtg["MEETING_DATE"] >= today)
                & (mtg["MEETING_DATE"] <= cutoff)]
        for _, r in m.iterrows():
            out.append({"Date": r["MEETING_DATE"], "Event": "Board meeting",
                        "Detail": str(r.get("PURPOSE", ""))[:120]})

    act = load_actions()
    if not act.empty and "SYMBOL" in act.columns:
        a = act[(act["SYMBOL"].astype(str).str.upper() == base)
                & (act["EX_DATE"] >= today) & (act["EX_DATE"] <= cutoff)]
        for _, r in a.iterrows():
            out.append({"Date": r["EX_DATE"], "Event": "Ex-date",
                        "Detail": str(r.get("PURPOSE", ""))[:120]})

    df = pd.DataFrame(out)
    return df.sort_values("Date") if not df.empty else df


# =============================================================== JOURNAL =====
# The only thing here that produces information nobody else has. Memory
# reliably rewrites why you took a trade; a written thesis does not.

JOURNAL_COLUMNS = ["date", "symbol", "setup", "bias", "entry", "stop", "target",
                   "qty", "thesis", "invalidation", "status", "exit_price",
                   "exit_date", "notes"]


def load_journal() -> pd.DataFrame:
    if JOURNAL_PATH.exists():
        try:
            df = pd.read_csv(JOURNAL_PATH, skipinitialspace=True)
            for c in JOURNAL_COLUMNS:
                if c not in df.columns:
                    df[c] = ""
            return df[JOURNAL_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=JOURNAL_COLUMNS)


def score_journal(journal: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """
    Mark each logged idea against what the price actually did.

    Open positions are marked to the last stored close; closed ones use the
    exit you recorded. Both are shown, because a run of open winners is not
    the same as a record of realised ones.
    """
    if journal.empty:
        return pd.DataFrame()

    latest = {}
    if not hist.empty:
        last_day = hist[hist["DATE"] == hist["DATE"].max()]
        latest = dict(zip(last_day["SYMBOL"], last_day["CLOSE"]))

    rows = []
    for _, r in journal.iterrows():
        sym = str(r.get("symbol", "")).replace(".NS", "").upper()
        try:
            entry = float(r.get("entry") or 0)
            stop = float(r.get("stop") or 0)
            qty = float(r.get("qty") or 0)
        except (TypeError, ValueError):
            continue
        if entry <= 0:
            continue

        status = str(r.get("status", "open")).lower()
        closed = status.startswith("clos")
        try:
            exit_px = float(r.get("exit_price") or 0)
        except (TypeError, ValueError):
            exit_px = 0.0
        mark = exit_px if (closed and exit_px > 0) else latest.get(sym, float("nan"))

        long_side = str(r.get("bias", "long")).lower() != "short"
        if mark == mark:
            move = (mark - entry) if long_side else (entry - mark)
            ret_pct = (move / entry) * 100
            pnl = move * qty
        else:
            ret_pct = pnl = float("nan")

        planned_risk = abs(entry - stop) if stop else float("nan")
        r_multiple = (move / planned_risk) if (mark == mark and planned_risk
                                               and planned_risk == planned_risk
                                               and planned_risk > 0) else float("nan")

        rows.append({
            "Date": r.get("date", ""), "Symbol": sym, "Setup": r.get("setup", ""),
            "Bias": r.get("bias", ""), "Status": "closed" if closed else "open",
            "Entry": entry, "Mark": mark, "Return %": ret_pct, "P&L": pnl,
            "R multiple": r_multiple,
            "Thesis": str(r.get("thesis", ""))[:160],
            "Invalidation": str(r.get("invalidation", ""))[:160],
        })
    return pd.DataFrame(rows)


def journal_stats(scored: pd.DataFrame) -> dict:
    """Headline record. Closed trades only — open ones are not results yet."""
    if scored.empty:
        return {}
    closed = scored[scored["Status"] == "closed"]
    live = scored[scored["Status"] == "open"]
    rets = closed["Return %"].dropna()
    rmul = closed["R multiple"].dropna()
    return {
        "closed": len(closed), "open": len(live),
        "hit_rate": (rets > 0).mean() * 100 if len(rets) else float("nan"),
        "median_return": rets.median() if len(rets) else float("nan"),
        "total_pnl": closed["P&L"].dropna().sum() if len(closed) else 0.0,
        "avg_r": rmul.mean() if len(rmul) else float("nan"),
        "best": rets.max() if len(rets) else float("nan"),
        "worst": rets.min() if len(rets) else float("nan"),
    }


# ============================================================== GLOSSARY =====
# Every term the app shows, in plain English. Definitions say what a number
# measures AND what it does not tell you -- the second half is usually the
# part that costs money.

GLOSSARY = {
    "Board meeting": "When results are approved. For options, a date inside your holding period is when implied volatility collapses.",
    "Ex-date": "The date from which a buyer no longer receives the dividend. Price drops by roughly the dividend — not a loss, but it looks like one on a chart.",
    "R multiple": "Profit as a multiple of the risk you planned. Above 0.3 over many trades is a real result; a single high R is luck.",
    "Insider filing": "A SEBI Regulation 7(2) disclosure — promoters and designated persons must report their own dealings.",
    "Block deal": "A large pre-negotiated trade in a separate window, both sides agreed in advance.",
    "Bulk deal": "A trade above 0.5% of a company's equity, reported with the client named.",
    "Reward %": "Distance from entry to target, as a percentage.",
    "Risk %": "Distance from entry to stop, as a percentage. A wider stop means a smaller position.",
    "Target": "The next structural level, or a multiple of typical daily range.",
    "Stop": "Where the idea stops being right, placed below the structure that would invalidate it.",
    "Entry": "The current price. Levels are computed from it.",
    "R:R": "Reward divided by risk, from the levels shown. Arithmetic, not odds — a 3:1 setup that works one time in five loses money.",
    "Confirmations": "How many independent things agree, out of six. A count of corroborating signals, not a probability.",
    "Breakdown": "Below the 20-day low with the medium-term trend already down.",
    "Delivery accumulation": "A rising share of volume taken as delivery — buyers holding rather than trading.",
    "Oversold in uptrend": "Above the 200-day average but short-term momentum washed out.",
    "Pullback to 50DMA": "An established uptrend where price has come back to its 50-day average.",
    "52-week high breakout": "At a yearly high with the long-term trend already up.",
    "Range breakout": "Price has broken above a tightening range on above-average volume.",
    # --- price and market
    "Day %": "Change from yesterday's close to today's, in percent.",
    "1M %": "Return over about 21 trading days. Not annualised.",
    "3M %": "Return over about 63 trading days.",
    "6M %": "Return over about 126 trading days.",
    "1Y %": "Return over about 250 trading days, roughly one year.",
    "Off 52W High %": "How far below the past year's highest close it now sits. "
        "Always zero or negative. A large gap means it has fallen a long way "
        "from its peak — which is neither a bargain nor a warning by itself.",
    "52W High": "Highest closing price in the past year.",
    "52W Low": "Lowest closing price in the past year.",
    "Turnover (Cr)": "Rupee value traded today, in crore. The practical measure "
        "of whether you can get in and out. Below about 1 crore, your own order "
        "moves the price.",
    "Volume": "Number of shares traded today.",
    "Trades": "Number of separate transactions. High volume across few trades "
        "means a handful of large orders rather than broad participation.",
    "Delivery %": "The share of today's volume that actually settled into "
        "demat accounts instead of being squared off the same day. High "
        "delivery means buyers intend to hold; low delivery means churn. "
        "This is the most underused free number in Indian markets.",
    "Delivery trend": "Recent delivery percentage against its 20-day average. "
        "Positive means a rising share of buyers are holding.",
    "Range position %": "Where the price sits in its own recent range. 0 is the "
        "period low, 100 the high.",

    # --- technicals
    "RSI": "Relative Strength Index, 0-100. Compares recent gains to recent "
        "losses. Above 70 is called overbought, below 30 oversold — but both "
        "persist for months in a trending stock, so neither is a signal on its own.",
    "RSI (14)": "Relative Strength Index over 14 days. Above 70 is conventionally "
        "overbought, below 30 oversold. Both can persist for months.",
    "vs 20DMA %": "Distance from the 20-day moving average, in percent. A short-"
        "term trend gauge.",
    "vs 50DMA %": "Distance from the 50-day moving average. The medium-term trend.",
    "vs 200DMA %": "Distance from the 200-day moving average. The long-term trend, "
        "and the line most institutional mandates watch.",
    "MACD hist": "The gap between the MACD line and its signal line. Positive and "
        "widening means momentum is building; shrinking means it is fading.",
    "Vol vs 20d": "Today's volume divided by the 20-day average. Above 2 means "
        "double the usual activity. Pair it with delivery percentage — a volume "
        "spike with low delivery is day-trading noise.",
    "Vol vs 20d avg": "Today's volume against the 20-day average.",
    "Above 50DMA": "Price is above its 50-day moving average.",
    "Above 200DMA": "Price is above its 200-day moving average.",
    "Golden cross": "The 50-day average has crossed above the 200-day. Widely "
        "watched, weakly predictive, and it fires after the move has begun.",
    "Death cross": "The 50-day average has crossed below the 200-day.",
    "Breakout 20d": "Today's price is above the highest of the previous 20 days.",
    "Breakdown 20d": "Today's price is below the lowest of the previous 20 days.",
    "Consolidating": "The recent range has tightened well inside the longer range. "
        "Says nothing about which way it breaks.",
    "Higher highs & lows": "Both recent peaks and recent troughs are rising — "
        "textbook uptrend structure.",
    "Lower highs & lows": "Both peaks and troughs are falling — downtrend structure.",
    "Bullish divergence": "Price made a lower low but momentum did not follow. "
        "Sometimes precedes a turn; often does not.",
    "Bearish divergence": "Price made a higher high but momentum did not follow.",
    "Bullish engulfing": "Today's up candle fully covers yesterday's down candle.",
    "Bearish engulfing": "Today's down candle fully covers yesterday's up candle.",
    "Hammer": "A candle with a long lower wick — sellers pushed down and were "
        "rejected.",
    "Shooting star": "A candle with a long upper wick — buyers pushed up and were "
        "rejected.",
    "Doji": "Open and close almost equal. Indecision, nothing more.",
    "Stance": "A weighted tally of the patterns found. It summarises what the "
        "chart shows; it is not a probability and not a recommendation.",
    "Score": "Bullish signals minus bearish signals.",
    "ATR": "Average True Range — how much this stock typically moves in a day, "
        "in rupees. The basis for position sizing and stop distance.",

    # --- fundamentals
    "P/E": "Price divided by earnings per share. How many rupees you pay per "
        "rupee of annual profit. Only comparable within an industry.",
    "Fwd P/E": "Same, using forecast earnings instead of past ones. Depends "
        "entirely on whose forecast.",
    "P/B": "Price divided by book value per share. Most meaningful for banks "
        "and asset-heavy businesses.",
    "EV/EBITDA": "Enterprise value against operating earnings. Includes debt, so "
        "it compares leveraged and unleveraged companies more fairly than P/E.",
    "ROE %": "Return on equity — profit as a percentage of shareholders' funds. "
        "Can be flattered by heavy borrowing.",
    "ROCE %": "Return on capital employed — profit before interest against all "
        "capital used, debt included. Harder to flatter than ROE, which is why "
        "it is the better quality test.",
    "Op Margin %": "Operating profit as a share of revenue. Pricing power.",
    "Net Margin %": "Net profit as a share of revenue, after everything.",
    "D/E": "Debt divided by equity. How much of the business is borrowed. "
        "Note this source reports it as a percentage, so 50 means 0.5x.",
    "Current Ratio": "Current assets over current liabilities. Below 1 means "
        "short-term obligations exceed short-term assets.",
    "Rev Growth %": "Year-on-year revenue growth.",
    "Profit Growth %": "Year-on-year earnings growth.",
    "Div Yield %": "Annual dividend as a percentage of price.",
    "EPS": "Earnings per share over the trailing twelve months.",
    "Beta": "How much the stock moves relative to the market. Above 1 is more "
        "volatile than the index, below 1 less.",
    "Mkt Cap (Cr)": "Market capitalisation in crore — share price times shares "
        "outstanding. The size of the company as the market prices it.",

    # --- order book
    "Order Book (Cr)": "Total value of contracts a company has won but not yet "
        "executed. Disclosed in quarterly presentations only — no data feed "
        "carries it.",
    "Book-to-Bill": "Order book divided by trailing annual revenue. Above 3x "
        "means roughly three years of work already contracted. Revenue "
        "visibility, not profitability — a large book at thin margins is not good news.",
    "Visibility (yrs)": "Years of revenue already contracted, at current run rate.",
    "QoQ Change %": "Change in the order book against the previous quarter.",

    # --- flows
    "Net (Cr)": "Buy value minus sell value, in crore. Positive means net buying.",
    "FII": "Foreign Institutional Investors — overseas funds. Their flows move "
        "Indian large caps more than any other single factor.",
    "DII": "Domestic Institutional Investors — Indian mutual funds, insurers, "
        "pension funds. Often take the opposite side to FIIs.",
    "A/D ratio": "Advancing stocks divided by declining ones. Above 1 means more "
        "rose than fell. Measures breadth — how broad a move is, not how large.",

    # --- options
    "Delta": "How much the option's price moves for a 1 rupee move in the "
        "underlying. A 0.5 delta call gains 50 paise per rupee. Also a rough "
        "proxy for the chance of finishing in the money.",
    "Gamma": "How fast delta itself changes. High gamma means the position's "
        "behaviour shifts quickly — highest near the strike and near expiry.",
    "Theta / day": "Rupees the position loses to time each day if nothing moves. "
        "The reason most long option positions lose: being right on direction "
        "but slow still loses money.",
    "Vega / 1% IV": "Rupees gained or lost per one point change in implied "
        "volatility. Why options can fall after good news — the event passes "
        "and volatility collapses.",
    "IV": "Implied volatility — the volatility the current price implies. Not a "
        "forecast; it is what the market is charging.",
    "IV %": "Implied volatility, solved from the option's closing price.",
    "ATM IV %": "Implied volatility at the strike nearest the current price.",
    "IV rank": "Where today's implied volatility sits between this underlying's "
        "own low and high over the period. High rank means options are expensive "
        "relative to their own history — an argument about price, not direction.",
    "IV percentile": "The share of past days with lower implied volatility. "
        "Differs from IV rank when the distribution is skewed.",
    "PCR": "Put-Call Ratio — put open interest divided by call open interest. "
        "Read as sentiment, though it mostly reflects hedging by large holders.",
    "PCR (OI)": "Put open interest divided by call open interest.",
    "OI": "Open Interest — contracts currently outstanding. Tells you what "
        "positions exist, not who is right.",
    "CHG_OI": "Change in open interest today. Rising OI means new positions; "
        "falling OI means positions being closed.",
    "Max pain": "The strike at which option writers would pay out least. Often "
        "described as a magnet for price on evidence that is thin.",
    "Long buildup": "Price up and open interest up — new buyers entering.",
    "Short buildup": "Price down and open interest up — new sellers entering.",
    "Short covering": "Price up and open interest down — sellers closing out.",
    "Long unwinding": "Price down and open interest down — buyers exiting.",
    "Breakeven": "The underlying price at which the position makes nothing and "
        "loses nothing at expiry.",
    "Lot size": "The fixed number of units in one contract. You trade whole lots.",
    "Net premium": "Total paid or received. Positive is a credit, negative a debit.",

    # --- backtest
    "Hit rate": "Share of selected stocks that finished positive.",
    "Beat market": "Share that beat the median stock over the same window. The "
        "number that matters — a screen returning 18% when the median stock did "
        "22% has cost you money.",
    "Median return": "The middle outcome. More honest than the mean, which one "
        "outlier can carry.",
    "Peak gain %": "The best it got to at any point, not where it ended.",
    "Max drawdown %": "The worst it got to at any point. A name that ended +40% "
        "after being -60% is not one most people would have held.",
}


def explain(term: str) -> str:
    """Definition for a column or label, or empty if there isn't one."""
    if term in GLOSSARY:
        return GLOSSARY[term]
    for key, val in GLOSSARY.items():          # tolerate suffixes and prefixes
        if key.lower() in str(term).lower():
            return val
    return ""


def help_config(df) -> dict:
    """
    Attach a hover explanation to every column that has one.

    Streamlit shows a small marker on the header; hovering gives the meaning.
    Applied automatically wherever a table is rendered.
    """
    cfg = {}
    for col in getattr(df, "columns", []):
        text = explain(col)
        if text:
            cfg[col] = st.column_config.Column(str(col), help=text)
    return cfg


def glossary_panel(label: str = "What do these terms mean?") -> None:
    """Searchable glossary, droppable on any tab."""
    with st.expander(label):
        q = st.text_input("Search", key=f"gloss_{abs(hash(label)) % 99999}",
                          placeholder="e.g. theta, delivery, ROCE")
        items = sorted(GLOSSARY.items())
        if q.strip():
            ql = q.strip().lower()
            items = [(k, v) for k, v in items
                     if ql in k.lower() or ql in v.lower()]
        if not items:
            st.caption("Nothing matches. Try a shorter word.")
        for term, meaning in items[:40]:
            st.markdown(f"**{term}** — {meaning}")
        if len(items) > 40:
            st.caption(f"{len(items) - 40} more — narrow the search.")


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
    st.caption(f"Indian equities, one screen · {APP_VERSION}")

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

    st.divider()
    glossary_panel("Full glossary")

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

(tab_market, tab_setups, tab_screen, tab_tech, tab_opt, tab_disc, tab_journal,
 tab_hunt, tab_test, tab_news, tab_flows, tab_ratios, tab_book,
 tab_depth) = st.tabs(
    ["Markets", "Weekly setups", "Screener (all NSE)", "Technicals", "Options",
     "Disclosures", "Journal", "Small-cap hunt", "Backtest", "News", "FII / DII",
     "Ratios", "Order Book", "Depth"]
)

with tab_market, safe_tab("Markets"):
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
            st.dataframe(colour_frame(merged, ["Day %", "1M %", "6M %", "1Y %", "Off 52W High %"]),
                     column_config=help_config(merged),
                use_container_width=True, height=520, hide_index=True,
            )

    with right:
        st.markdown("#### Global markets")
        if global_df.empty:
            st.info("Global data unavailable. Hit Refresh.")
        else:
            st.dataframe(colour_frame(global_df.drop(columns=["Symbol"]), ["Change %"]),
                     column_config=help_config(global_df.drop(columns=["Symbol"])),
                use_container_width=True, height=520, hide_index=True,
            )


with tab_setups, safe_tab("Weekly setups"):
    st.markdown("#### Weekly setups")
    st.markdown(
        "Named chart configurations across the whole market, each with levels "
        "taken from price structure — the stop sits where the idea stops being "
        "right, the target at the next structural level."
    )
    glossary_panel("What do these setups mean?")

    if not HISTORY_DIR.exists() or not any(HISTORY_DIR.glob("*.csv.gz")):
        st.info("Needs stored history. Run the collector — see the Backtest tab.")
    else:
        s1, s2, s3 = st.columns(3)
        su_turn = s1.number_input("Min turnover (Cr)", value=2.0, step=0.5,
                                  key="su_turn",
                                  help="Liquidity floor. Below this you cannot "
                                       "exit at the price you see.")
        su_rr = s2.number_input("Min reward:risk", value=1.5, step=0.5, key="su_rr")
        su_conf = s3.number_input("Min confirmations", value=2, min_value=0,
                                  max_value=6, step=1, key="su_conf",
                                  help="How many independent things agree, out of six.")

        if st.button("Scan for setups", type="primary", key="su_go"):
            st.session_state["_run_setups"] = True

        if st.session_state.get("_run_setups"):
            with st.spinner("Scanning the market…"):
                setups = cached_setups(su_turn, su_rr)

            if setups.empty:
                st.warning("Nothing matched. Loosen the filters, or the store "
                           "may not have 60+ trading days yet.")
            else:
                view = setups[setups["Confirmations"] >= su_conf]
                uni = symbol_universe()
                if not uni.empty:
                    view = view.merge(uni[["Symbol", "Company"]], on="Symbol",
                                      how="left")

                by_setup = setups["Setup"].value_counts()
                st.caption(f"{len(view)} setups pass · from {len(setups)} found · "
                           + " · ".join(f"{k} {v}" for k, v in by_setup.items()))

                lead = [c for c in ["Symbol", "Company", "Setup", "Bias",
                                    "Confirmations", "Entry", "Stop", "Target",
                                    "Risk %", "Reward %", "R:R"] if c in view.columns]
                rest = [c for c in view.columns if c not in lead + ["Note"]]
                st.dataframe(
                    colour_frame(view[lead + rest], ["Risk %", "Reward %", "1M %",
                                                     "Delivery trend"]),
                    column_config=help_config(view),
                    use_container_width=True, height=460, hide_index=True)

                st.download_button("Download setups as CSV",
                                   data=view.to_csv(index=False).encode("utf-8"),
                                   file_name="weekly_setups.csv", mime="text/csv",
                                   key="su_dl")

                st.markdown(
                    '<div class="stale"><b>I tested these rules against random '
                    'walks, and roughly half the setups came from pure noise.</b> '
                    'That is not a flaw in the code — random price data genuinely '
                    'produces breakouts, pullbacks and consolidations, because '
                    'those shapes occur in any random series. It means a setup '
                    'appearing here is evidence of a shape, not of an edge. The '
                    'Backtest tab exists precisely to tell you which of these has '
                    'actually paid on your names, and until you run it you have no '
                    'basis for believing any of them.</div>',
                    unsafe_allow_html=True)

                st.divider()
                st.markdown("**Position sizing**")
                st.caption("This decides outcomes more than setup selection, and "
                           "unlike setup selection it is arithmetic.")

                z1, z2, z3 = st.columns(3)
                capital = z1.number_input("Capital (Rs)", value=1000000.0,
                                          step=50000.0, key="su_cap")
                risk_pct = z2.number_input("Risk per trade %", value=1.0, step=0.25,
                                           key="su_risk",
                                           help="Share of capital lost if stopped out.")
                pick_sym = z3.selectbox("Size which setup",
                                        view["Symbol"].tolist(), key="su_pick")

                row = view[view["Symbol"] == pick_sym].iloc[0]
                shares, deployed = position_size(capital, risk_pct,
                                                 row["Entry"], row["Stop"])
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Shares", f"{shares:,}")
                q2.metric("Capital deployed", f"Rs {deployed:,.0f}")
                q3.metric("Loss if stopped",
                          f"Rs {shares * abs(row['Entry'] - row['Stop']):,.0f}")
                q4.metric("Gain if target hit",
                          f"Rs {shares * abs(row['Target'] - row['Entry']):,.0f}")

                st.caption(
                    f"**{row['Setup']}** — {row['Note']} Entry Rs {row['Entry']:,.2f}, "
                    f"stop Rs {row['Stop']:,.2f}, target Rs {row['Target']:,.2f}. "
                    f"Deploying Rs {deployed:,.0f} risks {risk_pct}% of capital "
                    f"because the stop is {row['Risk %']:.1f}% away — a wider stop "
                    "means a smaller position, not a bigger loss."
                )
        else:
            st.info("Click Scan to run. Reads from disk, so it takes seconds.")


with tab_screen, safe_tab("Screener"):
    st.markdown("#### Whole-market screener")
    glossary_panel("What do these columns mean?")
    st.caption(
        "Every EQ-series stock on NSE, from the daily bhavcopy — one file, one "
        "request. Screen here on price and volume, then send a shortlist to the "
        "Ratios tab for fundamentals."
    )

    with st.spinner("Loading full market…"):
        screen, screen_date, attempts = load_market_screen()

    if screen.empty:
        st.markdown(
            '<div class="stale">Could not retrieve the bhavcopy. It publishes after '
            'market close on trading days, and NSE refuses cloud-host requests '
            'intermittently. The log below shows exactly what was tried.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Fetch log", expanded=True):
            if attempts:
                st.dataframe(
                    pd.DataFrame(attempts, columns=["Date", "File", "Result"]),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "HTTP 403 means NSE blocked this server. HTTP 404 means no file "
                    "for that date, which is normal on holidays. A session error "
                    "means NSE refused the initial handshake."
                )
            else:
                st.write("No attempts recorded.")
    else:
        hl_days = screen.attrs.get("hl_days")
        st.caption(
            f"{len(screen):,} stocks · trading day {screen_date}"
            + (f" · 52-week columns computed from {hl_days} stored trading days"
               if hl_days else "")
        )
        if hl_days and hl_days < 240:
            st.markdown(
                f'<div class="stale">Only {hl_days} trading days are on disk, so '
                'the "52 week" high and low are really the high and low of that '
                'shorter window. They become true 52-week figures once the '
                'collector has banked a full year.</div>',
                unsafe_allow_html=True)

        breadth = market_breadth(screen)
        if breadth:
            st.markdown("**Market breadth**")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Advances", f"{breadth['advances']:,}")
            b2.metric("Declines", f"{breadth['declines']:,}")
            b3.metric("A/D ratio", f"{breadth['ad_ratio']:.2f}")
            b4.metric("Median move", f"{breadth['median_move']:+.2f}%")
            st.caption(
                f"{breadth['up_5pct']:,} stocks up 5%+ · "
                f"{breadth['down_5pct']:,} down 5%+ · "
                f"among liquid names (>1 Cr turnover): "
                f"{breadth['liquid_advances']:,} up, {breadth['liquid_declines']:,} down. "
                "Breadth measures what the market actually did — more informative "
                "than counting positive headlines."
            )
            st.divider()

        with st.expander("Fetch log"):
            st.dataframe(pd.DataFrame(attempts, columns=["Date", "File", "Result"]),
                         use_container_width=True, hide_index=True)

        f1, f2, f3, f4 = st.columns(4)
        min_turnover = f1.number_input("Min turnover (Cr)", value=5.0, step=1.0,
                                       key="scr_turnover")
        min_delivery = f2.number_input("Min delivery %", value=0.0, step=5.0,
                                       key="scr_delivery")
        min_price = f3.number_input("Min price", value=0.0, step=10.0, key="scr_minpx")
        max_price = f4.number_input("Max price (0 = no cap)", value=0.0, step=100.0,
                                    key="scr_maxpx")

        g1, g2, g3, g4 = st.columns(4)
        move_min = g1.number_input("Min day % move", value=-100.0, step=1.0, key="scr_movemin")
        move_max = g2.number_input("Max day % move", value=100.0, step=1.0, key="scr_movemax")
        near_high = g3.number_input("Within % of 52W high", value=0.0, step=1.0,
                                    key="scr_nearhigh",
                                    help="0 disables. Enter 5 to see stocks within "
                                         "5% of their yearly high.")
        near_low = g4.number_input("Within % of 52W low", value=0.0, step=1.0,
                                   key="scr_nearlow", help="0 disables.")

        view = screen.copy()
        view = view[view["Turnover (Cr)"].fillna(0) >= min_turnover]
        if min_delivery > 0 and "Delivery %" in view.columns:
            view = view[view["Delivery %"].fillna(0) >= min_delivery]
        if min_price > 0:
            view = view[view["Close"].fillna(0) >= min_price]
        if max_price > 0:
            view = view[view["Close"].fillna(1e12) <= max_price]
        view = view[view["Day %"].between(move_min, move_max, inclusive="both")]
        if near_high > 0 and "Off 52W High %" in view.columns:
            view = view[view["Off 52W High %"].fillna(-1e9) >= -near_high]
        if near_low > 0 and "Above 52W Low %" in view.columns:
            view = view[view["Above 52W Low %"].fillna(1e9) <= near_low]

        sort_options = [c for c in ["Turnover (Cr)", "Day %", "1M %", "3M %",
                                    "6M %", "1Y %", "Delivery %", "Volume", "Close",
                                    "Off 52W High %", "Above 52W Low %",
                                    "52W position %"] if c in view.columns]
        s1, s2 = st.columns([3, 1])
        sort_by = s1.selectbox("Sort by", options=sort_options, key="scr_sort")
        descending = s2.checkbox("Descending", value=True)
        view = view.sort_values(sort_by, ascending=not descending, na_position="last")

        st.caption(f"{len(view):,} stocks pass the filters.")
        st.dataframe(colour_frame(view.head(500), ["Day %", "1M %"]),
                     column_config=help_config(view.head(500)),
            use_container_width=True, height=520, hide_index=True,
        )
        if len(view) > 500:
            st.caption("Showing the top 500. Tighten the filters to narrow further.")

        st.download_button(
            "Download these results as CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name=f"nse_screen_{screen_date}.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown("**Send a shortlist to the Ratios tab**")
        st.caption(
            "Fundamentals are fetched company by company, roughly 4 seconds each, "
            "so this is capped at 40. Screen down first, then drill in."
        )

        shortlist = st.multiselect(
            "Pick up to 40 symbols",
            options=view["Symbol"].head(500).tolist(),
            max_selections=40,
        )
        if shortlist:
            st.session_state["shortlist"] = [f"{s}.NS" for s in shortlist]
            st.success(
                f"{len(shortlist)} symbols queued. Open the Ratios tab and switch "
                "the source to 'Screener shortlist'."
            )


with tab_tech, safe_tab("Technicals"):
    st.markdown("#### Technical position")
    glossary_panel("What do RSI, DMA and delivery trend mean?")
    st.markdown(
        "Where each stock sits relative to its own moving averages, range and "
        "average volume. These describe what price has **already** done. None of "
        "them forecasts what it will do next — a stock can sit above its 200DMA "
        "with RSI at 65 all the way down."
    )

    tech_source = st.radio(
        "Universe",
        ["All NSE stocks", "Watchlist", "Screener shortlist"],
        horizontal=True, key="tech_source",
    )

    if tech_source == "All NSE stocks":
        st.caption(
            "Computed from stacked bhavcopies — every EQ stock at once. The first "
            "load takes a few minutes because each trading day is a separate file "
            "from NSE, then it's cached for six hours."
        )
        hist_days = st.select_slider(
            "History depth",
            options=[40, 60, 90, 120, 200, 300],
            value=90,
            help="Calendar days. 60+ needed for the 50DMA, 300 for the 200DMA.",
        )

        if st.button("Load whole-market technicals", type="primary"):
            st.session_state["_mkt_tech_days"] = hist_days

        if st.session_state.get("_mkt_tech_days"):
            with st.spinner("Fetching bhavcopies and computing indicators…"):
                mkt_tech, days_loaded = load_market_technicals(
                    st.session_state["_mkt_tech_days"])

            if mkt_tech.empty:
                st.warning("No history retrieved. Check the Screener tab's fetch log.")
            else:
                universe = load_universe()
                if not universe.empty:
                    mkt_tech = mkt_tech.merge(universe[["Symbol", "Company"]],
                                              on="Symbol", how="left")
                st.caption(f"{len(mkt_tech):,} stocks · {days_loaded} trading days of history")

                t1, t2, t3 = st.columns(3)
                rsi_lo = t1.number_input("RSI min", value=0.0, step=5.0, key="tech_rsilo")
                rsi_hi = t2.number_input("RSI max", value=100.0, step=5.0, key="tech_rsihi")
                min_vol_surge = t3.number_input("Min vol vs 20d avg", value=0.0, step=0.5,
                                                key="tech_volsurge")

                u1, u2 = st.columns(2)
                above_50 = u1.checkbox("Only above 50DMA", key="tech_above50")
                rising_deliv = u2.checkbox("Only rising delivery %", key="tech_deliv")

                v = mkt_tech.copy()
                v = v[v["RSI (14)"].fillna(-1).between(rsi_lo, rsi_hi)]
                if min_vol_surge > 0 and "Vol vs 20d" in v.columns:
                    v = v[v["Vol vs 20d"].fillna(0) >= min_vol_surge]
                if above_50 and "vs 50DMA %" in v.columns:
                    v = v[v["vs 50DMA %"].fillna(-1e9) > 0]
                if rising_deliv and "Delivery trend" in v.columns:
                    v = v[v["Delivery trend"].fillna(-1e9) > 0]

                sortable = [c for c in v.columns if v[c].dtype.kind in "fi"]
                sort_col = st.selectbox("Sort by", sortable, key="tech_sort",
                                        index=sortable.index("1M %") if "1M %" in sortable else 0)
                v = v.sort_values(sort_col, ascending=False, na_position="last")

                lead = [c for c in ["Symbol", "Company"] if c in v.columns]
                v = v[lead + [c for c in v.columns if c not in lead]]

                st.caption(f"{len(v):,} stocks pass.")
                st.dataframe(colour_frame(v.head(400), ["vs 20DMA %", "vs 50DMA %", "vs 200DMA %",
                                               "1M %", "3M %", "6M %", "Delivery trend"]),
                     column_config=help_config(v.head(400)),
                    use_container_width=True, height=520, hide_index=True,
                )
                st.download_button("Download as CSV",
                                   data=v.to_csv(index=False).encode("utf-8"),
                                   file_name="nse_technicals.csv", mime="text/csv")
        else:
            st.info("Click the button above to load. This runs once, then caches.")

        tech_target = ()
    else:
        tech_target = (tuple(st.session_state.get("shortlist", []))
                       if tech_source == "Screener shortlist" else tickers)

    if tech_source != "All NSE stocks" and not tech_target:
        st.info(
            "Nothing selected. Switch the Universe above to **All NSE stocks**, "
            "or build a shortlist on the Screener tab."
        )
    elif tech_source != "All NSE stocks":
        with st.spinner("Computing indicators…"):
            tech = load_technicals(tech_target)

        if tech.empty:
            st.info("No price history returned for these symbols.")
        else:
            merged = tech.merge(wl[["ticker", "company"]], left_on="Ticker",
                                right_on="ticker", how="left")
            merged["Company"] = merged["company"].fillna(merged["Ticker"])
            merged = merged.drop(columns=["ticker", "company"], errors="ignore")
            cols = ["Company"] + [c for c in merged.columns
                                  if c not in ("Company", "Ticker")]
            st.dataframe(
                colour_frame(merged[cols],
                             ["vs 20DMA %", "vs 50DMA %", "vs 200DMA %", "MACD hist"]),
                use_container_width=True, height=460, hide_index=True,
            )

            st.caption(
                "**RSI (14)** — above 70 is conventionally 'overbought', below 30 "
                "'oversold'; both persist for months in trending stocks. "
                "**52W position %** — 0 is the year's low, 100 the high. "
                "**Vol vs 20d avg** — above 2 means today's volume is double the "
                "recent norm, which is worth pairing with the delivery % on the "
                "Screener tab to tell accumulation from churn."
            )


    st.divider()
    st.markdown("#### Pattern scan across all stocks")
    st.caption(
        "Runs the same pattern detection over every stock in your stored "
        "history — instant, because it reads from disk rather than NSE."
    )

    if not HISTORY_DIR.exists() or not any(HISTORY_DIR.glob("*.csv.gz")):
        st.info(
            "Needs stored history. Run the collector from the Actions tab "
            "in GitHub first — see the Backtest tab."
        )
    elif not st.session_state.get("_run_scan"):
        if st.button("Run pattern scan", type="primary", key="scan_go"):
            st.session_state["_run_scan"] = True
            st.rerun()
        st.caption(
            "Runs on demand rather than automatically — scanning the whole "
            "market is memory-heavy and would otherwise run on every click."
        )
    else:
        if st.button("Hide scan", key="scan_hide"):
            st.session_state["_run_scan"] = False
            st.rerun()
        with st.spinner("Scanning the whole market…"):
            scan = cached_pattern_scan()

        if scan.empty:
            st.warning("Not enough history to detect patterns. Needs 60+ trading days.")
        else:
            if not bool(scan["Has OHLC"].iloc[0]):
                st.markdown(
                    '<div class="stale">Your stored data has no open/high/low '
                    'prices, so candlestick patterns (engulfing, hammer, doji) are '
                    'switched off. Trend, breakout, crossover and divergence '
                    'patterns still work. To enable the rest, update '
                    '<code>collect.py</code> and re-run the workflow with the '
                    '<b>refresh</b> option — the newer collector stores OHLC.</div>',
                    unsafe_allow_html=True,
                )

            uni = symbol_universe()
            if not uni.empty:
                scan = scan.merge(uni[["Symbol", "Company"]], on="Symbol", how="left")

            st.caption(f"{len(scan):,} stocks scanned.")

            p1, p2 = st.columns([2, 2])
            stance_pick = p1.multiselect(
                "Stance", options=["Constructive", "Mildly constructive", "Neutral",
                                   "Mildly negative", "Negative"],
                default=["Constructive"], key="scan_stance")
            flag_opts = [c for c in scan.columns if scan[c].dtype == bool
                         and c not in ("Has OHLC",)]
            must_have = p2.multiselect("Must show these patterns", options=flag_opts,
                                       key="scan_flags")

            sv = scan.copy()
            if stance_pick:
                sv = sv[sv["Stance"].isin(stance_pick)]
            for flag in must_have:
                sv = sv[sv[flag].fillna(False)]

            sv = sv.sort_values("Score", ascending=False)
            lead = [c for c in ["Symbol", "Company", "Stance", "Score",
                                "Bull signals", "Bear signals", "Close", "RSI",
                                "Vol vs 20d"] if c in sv.columns]
            rest = [c for c in sv.columns if c not in lead + ["Has OHLC"]]

            st.caption(f"{len(sv):,} stocks match.")
            st.dataframe(sv[lead + rest].head(400),
                         use_container_width=True, height=520, hide_index=True)
            st.download_button("Download scan as CSV",
                               data=sv.to_csv(index=False).encode("utf-8"),
                               file_name="pattern_scan.csv", mime="text/csv",
                               key="dl_scan")

            counts = {c: int(scan[c].fillna(False).sum()) for c in flag_opts}
            st.caption(
                "Frequency today: "
                + " · ".join(f"{k} {v}" for k, v in
                             sorted(counts.items(), key=lambda x: -x[1])[:8])
            )
            st.markdown(
                '<div class="stale">A pattern appearing in 400 of 2,000 stocks on '
                'the same day is telling you about the market, not about those '
                'stocks. Check the frequency line above before reading anything '
                'into a signal — and test it on the Backtest tab, which is the '
                'only way to know whether a setup has paid on your names.</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### Single stock chart read")
    st.caption("Pick any listed stock — this does not depend on your watchlist.")

    chart_univ = symbol_universe()

    if not chart_univ.empty:
        chart_names = dict(zip(chart_univ["Symbol"], chart_univ["Company"]))
        all_opts = [f"{s}.NS" for s in sorted(chart_univ["Symbol"])]
        watch_first = [t for t in tickers if t in all_opts]
        all_opts = watch_first + [t for t in all_opts if t not in watch_first]
        src_label = chart_univ["Source"].iloc[0]
        st.caption(
            f"{len(all_opts):,} stocks (via {src_label}) — start typing to "
            "search. Your watchlist is at the top."
        )
    else:
        chart_names = {}
        all_opts = sorted(set(list(tickers) + list(st.session_state.get("shortlist", []))))
        st.caption(
            "No symbol source reachable. Showing your watchlist — collect some "
            "history from the Backtest tab and the full list works offline."
        )

    st.caption(
        "Any NSE symbol works even if it is not listed above: type it with a "
        "`.NS` suffix in the box below if the dropdown is short."
    )

    def chart_label(t):
        if t in name_by_ticker:
            return name_by_ticker[t]
        base = t.replace(".NS", "")
        name = chart_names.get(base)
        return f"{name} ({base})" if name else base

    if not all_opts:
        st.caption("No stocks available.")
    else:
        pick = st.selectbox("Stock", all_opts, key="chart_pick",
                            format_func=chart_label)
        with st.spinner("Reading the chart…"):
            ca = load_chart_analysis(pick)

        if ca is None:
            st.info("Not enough price history for this symbol.")
        else:
            st.markdown(f"**{name_by_ticker.get(pick, pick)}** — Rs {ca['last']:,.2f}")
            st.markdown(f"*{ca['trend']}.* RSI {ca['rsi']:.0f}. "
                        f"Typical daily move {ca['atr_pct']:.1f}% "
                        f"(ATR {ca['atr']:.2f}).")

            k1, k2 = st.columns(2)
            with k1:
                st.markdown("**Support below**")
                if ca["support"]:
                    for lvl in ca["support"]:
                        st.markdown(f"- Rs {lvl:,.2f}  ({((lvl / ca['last']) - 1) * 100:+.1f}%)")
                else:
                    st.caption("No prior swing low below — at or near 52-week lows.")
            with k2:
                st.markdown("**Resistance above**")
                if ca["resistance"]:
                    for lvl in ca["resistance"]:
                        st.markdown(f"- Rs {lvl:,.2f}  ({((lvl / ca['last']) - 1) * 100:+.1f}%)")
                else:
                    st.caption("No prior swing high above — at or near 52-week highs.")

            st.markdown("**Expected range from realised volatility**")
            r1, r2 = st.columns(2)
            r1.metric("Next week",
                      f"{ca['range_1w'][0]:,.0f} – {ca['range_1w'][1]:,.0f}")
            r2.metric("Next month",
                      f"{ca['range_1m'][0]:,.0f} – {ca['range_1m'][1]:,.0f}")

            st.markdown(
                '<div class="stale"><b>Read these bands correctly.</b> They are '
                'ATR scaled by the square root of time — a statement about how '
                'much this stock typically moves, not where it is going. They are '
                'symmetric by construction, so the upper figure is not a target '
                'and the lower is not a prediction. Roughly two-thirds of one-month '
                'outcomes land inside a band like this; the other third is what '
                'position sizing is for. A single earnings release or order '
                'announcement voids the whole calculation.</div>',
                unsafe_allow_html=True,
            )


            st.divider()
            st.markdown("**What the chart is showing**")

            pats = detect_patterns(ca["ohlc"])
            verdict = chart_verdict(pats, ca)

            colour = {"up": "#2fbf71", "down": "#e5484d", "flat": "#6b7684"}[verdict["tone"]]
            st.markdown(
                f'<div style="border-left:3px solid {colour};padding:0.7rem 1rem;'
                f'background:#141a21;margin:0.5rem 0;">'
                f'<div style="font-size:1.15rem;font-weight:650;color:{colour};">'
                f'{verdict["stance"]}</div>'
                f'<div style="color:#8b95a1;font-size:0.85rem;">'
                f'{verdict["bull"]} bullish vs {verdict["bear"]} bearish signals · '
                f'{verdict["conviction"]}</div></div>',
                unsafe_allow_html=True,
            )

            if pats:
                for name, direction, note in pats:
                    mark = {"bullish": "▲", "bearish": "▼", "neutral": "■"}[direction]
                    col = {"bullish": "#2fbf71", "bearish": "#e5484d",
                           "neutral": "#6b7684"}[direction]
                    st.markdown(
                        f'<div style="padding:0.3rem 0;border-bottom:1px solid #1a2028;">'
                        f'<span style="color:{col};">{mark}</span> '
                        f'<b>{name}</b> — <span style="color:#8b95a1;">{note}</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No recognised patterns in the recent bars.")

            lo1m, hi1m = ca["range_1m"]

            res_txt = (", ".join(f"Rs {x:,.0f}" for x in ca["resistance"][:2])
                       or f"the 52-week high at Rs {ca['hi52']:,.0f}")
            sup_txt = (", ".join(f"Rs {x:,.0f}" for x in ca["support"][:2])
                       or f"the 52-week low at Rs {ca['lo52']:,.0f}")
            invalidate_down = (f"Rs {ca['support'][0]:,.0f}" if ca["support"]
                               else f"the 52-week low at Rs {ca['lo52']:,.0f}")
            invalidate_up = (f"Rs {ca['resistance'][0]:,.0f}" if ca["resistance"]
                             else f"the 52-week high at Rs {ca['hi52']:,.0f}")

            if verdict["tone"] == "up":
                narrative = (
                    f"Structure leans higher. The levels that matter above are "
                    f"{res_txt}. The read is wrong if it loses {invalidate_down}"
                )
            elif verdict["tone"] == "down":
                narrative = (
                    f"Structure leans lower. Support to watch: {sup_txt}. "
                    f"The read is wrong if it reclaims {invalidate_up}"
                )
            else:
                narrative = (
                    f"No directional edge in the chart. It has been ranging "
                    f"roughly between {sup_txt} and {res_txt}"
                )

            st.markdown(f"**One-month read.** {narrative}. "
                        f"Volatility puts the likely band at "
                        f"Rs {lo1m:,.0f}–{hi1m:,.0f}.")

            st.markdown(
                '<div class="stale"><b>How much to trust this.</b> Every pattern '
                'above is real and detected from the actual bars — but tested on '
                'liquid equities, candlestick and chart patterns show edges close '
                'to zero once costs are counted, and they fail more often than the '
                'textbooks suggest. The stance is a weighted tally of what the '
                'chart shows, not a probability. It knows nothing about earnings, '
                'orders, or promoter activity, any one of which overrides the whole '
                'read overnight. Use the Backtest tab to check whether this kind of '
                'setup has actually worked on your names.</div>',
                unsafe_allow_html=True,
            )

            st.line_chart(ca["series"], height=260)
            st.caption(
                f"120 days. 20DMA {ca['sma20']:,.0f} · 50DMA {ca['sma50']:,.0f}"
                + (f" · 200DMA {ca['sma200']:,.0f}" if ca["sma200"] else "")
                + f" · 52W range {ca['lo52']:,.0f}–{ca['hi52']:,.0f}"
            )


with tab_opt, safe_tab("Options"):
    st.markdown("#### Options")
    glossary_panel("What do delta, theta, IV rank and PCR mean?")

    opt_view = st.radio(
        "View",
        ["Scenario calculator", "Payoff calculator", "Option chain", "IV rank"],
        horizontal=True, key="opt_view")

    # -------------------------------------------------- scenario calculator ---
    if opt_view == "Scenario calculator":
        st.caption(
            "Reads the chart, then prices the option across a range of outcomes "
            "sized by that stock's own volatility. It answers *if this happens, "
            "what is it worth* — not *what will happen*."
        )

        univ = symbol_universe()
        opts_list = ([f"{s}.NS" for s in sorted(univ["Symbol"])]
                     if not univ.empty else list(tickers))
        watch_first = [t for t in tickers if t in opts_list]
        opts_list = watch_first + [t for t in opts_list if t not in watch_first]

        if not opts_list:
            st.info("No symbols available. Check the Screener tab.")
        else:
            sc_pick = st.selectbox("Underlying", opts_list, key="sc_pick",
                                   format_func=lambda t: name_by_ticker.get(
                                       t, t.replace(".NS", "")))
            with st.spinner("Reading the chart…"):
                ca = load_chart_analysis(sc_pick)

            if ca is None:
                st.info("Not enough price history for this symbol.")
            else:
                pats = detect_patterns(ca["ohlc"])
                verdict = chart_verdict(pats, ca)
                colour = {"up": "#2fbf71", "down": "#e5484d",
                          "flat": "#6b7684"}[verdict["tone"]]

                st.markdown(
                    f'<div style="border-left:3px solid {colour};padding:0.7rem 1rem;'
                    f'background:#141a21;margin:0.4rem 0;">'
                    f'<div style="font-size:1.1rem;font-weight:650;color:{colour};">'
                    f'Chart reads: {verdict["stance"]}</div>'
                    f'<div style="color:#8b95a1;font-size:0.85rem;">'
                    f'Spot Rs {ca["last"]:,.2f} · RSI {ca["rsi"]:.0f} · '
                    f'typical daily move {ca["atr_pct"]:.1f}% · '
                    f'{verdict["bull"]} bullish vs {verdict["bear"]} bearish signals'
                    f'</div></div>', unsafe_allow_html=True)

                suggested = "call" if verdict["tone"] == "up" else (
                    "put" if verdict["tone"] == "down" else "call")

                st.markdown("**Your option**")
                o1, o2, o3, o4 = st.columns(4)
                o_dir = o1.selectbox("Direction", ["long", "short"], key="sc_dir")
                o_kind = o2.selectbox("Type", ["call", "put"],
                                      index=0 if suggested == "call" else 1,
                                      key="sc_kind",
                                      help="Pre-set from the chart read — override freely.")
                o_strike = o3.number_input("Strike", value=float(round(ca["last"], -1)),
                                           step=50.0, key="sc_strike")
                o_prem = o4.number_input("Premium paid/received", value=100.0,
                                         step=5.0, key="sc_prem")

                p1, p2, p3 = st.columns(3)
                days_exp = p1.number_input("Days to expiry", value=30, min_value=1,
                                           step=1, key="sc_exp")
                hold = p2.number_input("Days you'll hold", value=10, min_value=0,
                                       step=1, key="sc_hold",
                                       help="Scenario is priced this many days from now.")
                lots = p3.number_input("Lots", value=1, min_value=1, step=1, key="sc_lots")

                base = sc_pick.replace(".NS", "")
                lot_size = load_lot_sizes().get(base, LOT_SIZES.get(base, 1))
                qty = int(lots) * int(lot_size)

                solved_iv = implied_vol(o_prem, ca["last"], o_strike,
                                        days_exp / 365.0,
                                        is_call=(o_kind == "call"))
                iv_default = solved_iv if solved_iv == solved_iv else 25.0

                v1, v2 = st.columns(2)
                iv_use = v1.number_input(
                    "Implied volatility %", value=float(round(iv_default, 1)),
                    step=1.0, key="sc_iv",
                    help="Solved from your premium where possible.")
                iv_shift = v2.number_input(
                    "IV change in the scenario", value=0.0, step=1.0, key="sc_ivsh",
                    help="Volatility usually rises when price falls. Try -5 for a "
                         "post-event crush.")

                if solved_iv == solved_iv:
                    st.caption(f"Your premium of Rs {o_prem:,.2f} implies "
                               f"{solved_iv:.1f}% volatility at that strike. "
                               f"Lot size {lot_size}, so {qty:,} units.")
                else:
                    st.caption(
                        "That premium has no Black-Scholes solution — usually it is "
                        "below intrinsic value, or the expiry is too near. The "
                        "default volatility is being used instead."
                    )

                ev = events_before(sc_pick, int(days_exp))
                if not ev.empty:
                    inside = ev[ev["Date"] <= pd.Timestamp(datetime.now().date())
                                + pd.Timedelta(days=int(hold))]
                    lines = " · ".join(
                        f"{r['Event']} on {r['Date']:%d %b}" for _, r in ev.iterrows())
                    if not inside.empty:
                        st.markdown(
                            f'<div class="stale"><b>There is a scheduled event inside '
                            f'your holding period.</b> {lines}. This is exactly when '
                            f'implied volatility collapses: the uncertainty the '
                            f'premium was pricing resolves, and the option can lose '
                            f'value even if the stock moves your way. Set the IV '
                            f'change box to -5 or -10 and look at the difference '
                            f'before taking this.</div>', unsafe_allow_html=True)
                    else:
                        st.caption(f"Scheduled before expiry: {lines}. Outside your "
                                   "holding window, but it will affect the premium.")

                tbl = scenario_table(
                    ca["last"], o_strike, o_prem, days_exp, hold, iv_use,
                    o_kind == "call", o_dir, qty, ca["atr_pct"], iv_shift)

                st.markdown("**If this happens, this is what you have**")
                st.dataframe(colour_frame(tbl, ["Move %", "P&L", "Return %"]),
                     column_config=help_config(tbl),
                    use_container_width=True, hide_index=True)

                g = greeks(ca["last"], o_strike, days_exp / 365.0, iv_use / 100.0,
                           is_call=(o_kind == "call"))
                sign = 1 if o_dir == "long" else -1
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Delta", f"{g['delta'] * sign:+.3f}",
                          help="Change in option value per 1 rupee of underlying.")
                g2.metric("Theta / day", f"Rs {g['theta'] * sign * qty:,.0f}",
                          help="Value lost to time each day, whole position.")
                g3.metric("Vega / 1% IV", f"Rs {g['vega'] * sign * qty:,.0f}")
                g4.metric("Gamma", f"{g['gamma'] * sign:.5f}")

                theta_total = g["theta"] * sign * qty * hold
                if o_dir == "long" and theta_total < 0:
                    cost = abs(theta_total)
                    st.markdown(
                        f'<div class="stale"><b>Time decay over {int(hold)} days: '
                        f'Rs {cost:,.0f}.</b> That is what the position loses if the '
                        f'underlying does not move at all. Against a premium of '
                        f'Rs {o_prem * qty:,.0f}, decay alone is '
                        f'{cost / (o_prem * qty) * 100:.0f}% of what you paid. '
                        f'Being right on direction but slow is how most long option '
                        f'positions lose.</div>', unsafe_allow_html=True)

                st.markdown(
                    '<div class="stale"><b>What this does and does not do.</b> The '
                    'scenarios are sized by this stock\'s realised volatility, so '
                    'the spread is calibrated to how much it actually moves — but '
                    'they carry no probabilities, and the chart stance is a tally '
                    'of patterns whose predictive power is weak. Repricing also '
                    'holds implied volatility fixed unless you shift it, which is '
                    'wrong in the direction that hurts buyers: IV typically rises '
                    'when price falls and collapses after events. Use the IV change '
                    'box to test that rather than assuming it away.</div>',
                    unsafe_allow_html=True)

    # ------------------------------------------------------------ payoff ---
    elif opt_view == "Payoff calculator":
        st.caption(
            "Exact arithmetic on expiry values — no model and no assumptions. "
            "It tells you what each outcome pays, not how likely any of them is."
        )

        pc1, pc2 = st.columns([2, 1])
        underlying_sym = pc1.selectbox(
            "Underlying", options=list(LOT_SIZES.keys()), key="opt_sym")
        lot = LOT_SIZES.get(underlying_sym, 1)
        spot = pc2.number_input("Spot price", value=24366.0, step=50.0, key="opt_spot")
        st.caption(f"Lot size {lot}. Quantities below are in lots.")

        if "opt_legs" not in st.session_state:
            st.session_state["opt_legs"] = pd.DataFrame([
                {"Direction": "long", "Type": "call", "Strike": 24400.0,
                 "Premium": 150.0, "Lots": 1},
            ])

        legs_df = st.data_editor(
            st.session_state["opt_legs"], num_rows="dynamic",
            use_container_width=True, key="opt_editor",
            column_config={
                "Direction": st.column_config.SelectboxColumn(
                    options=["long", "short"], required=True),
                "Type": st.column_config.SelectboxColumn(
                    options=["call", "put", "future"], required=True,
                    help="For a future leg, put your entry price in Premium."),
                "Strike": st.column_config.NumberColumn(format="%.2f"),
                "Premium": st.column_config.NumberColumn(format="%.2f"),
                "Lots": st.column_config.NumberColumn(min_value=1, step=1),
            },
        )

        legs = []
        for _, r in legs_df.iterrows():
            try:
                legs.append({
                    "kind": str(r["Type"]), "strike": float(r["Strike"] or 0),
                    "premium": float(r["Premium"] or 0),
                    "direction": str(r["Direction"]),
                    "qty": int(r["Lots"] or 1) * lot,
                })
            except Exception:
                continue

        if not legs:
            st.info("Add at least one leg above.")
        else:
            res = analyse_position(legs, spot)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max profit",
                      "Unlimited" if res["unlimited_up"] else f"Rs {res['max_profit']:,.0f}")
            m2.metric("Max loss",
                      "Unlimited" if res["unlimited_down"] else f"Rs {res['max_loss']:,.0f}")
            m3.metric("Net premium",
                      f"Rs {res['net_premium']:,.0f}",
                      help="Positive is a credit received, negative a debit paid.")
            m4.metric("P&L at spot", f"Rs {res['payoff_at_spot']:,.0f}")

            st.markdown(
                "**Breakeven"
                + ("s" if len(res["breakevens"]) != 1 else "")
                + ":** "
                + (", ".join(f"{b:,.2f}" for b in res["breakevens"])
                   or "none — the position never crosses zero")
            )

            chart = pd.DataFrame({"Payoff": res["payoff"]},
                                 index=pd.Index(res["grid"], name="Price at expiry"))
            window = chart[(chart.index > spot * 0.8) & (chart.index < spot * 1.2)]
            st.line_chart(window if len(window) > 50 else chart, height=320)

            if res["unlimited_down"]:
                st.markdown(
                    '<div class="stale"><b>This position has unlimited loss.</b> '
                    'A short call or short future has no cap on the upside, so no '
                    'maximum loss exists. Size it on what you can afford to lose, '
                    'not on the premium received.</div>',
                    unsafe_allow_html=True,
                )
            elif not res["unlimited_up"]:
                st.caption(
                    f"Worst case if the underlying went to zero: "
                    f"Rs {res['worst_at_zero']:,.0f}."
                )

            st.caption(
                "Payoff is at expiry only. Before expiry the position is worth "
                "something different because of time value and volatility — a "
                "spread showing max profit here can be well short of it a week "
                "early. Brokerage, STT and slippage are not included."
            )

    # ------------------------------------------------------------- chain ---
    elif opt_view == "Option chain":
        fno = load_fno_latest()

        if not fno.empty:
            opts = fno[fno["OPT"].isin(["CE", "PE"])] if "OPT" in fno.columns else fno
            unders = sorted(opts["SYMBOL"].dropna().unique())
            snap_date = fno["DATE"].iloc[0]

            st.caption(
                f"{len(unders)} F&O underlyings · {len(opts):,} option contracts · "
                f"snapshot {snap_date}. NSE permits derivatives on roughly 190 "
                "stocks, not the whole cash market — that is exchange eligibility, "
                "not a gap in this data."
            )

            c1, c2 = st.columns(2)
            sym = c1.selectbox("Underlying", unders, key="fno_sym")
            sub = opts[opts["SYMBOL"] == sym].copy()
            exps = sorted(pd.to_datetime(sub["EXPIRY"], errors="coerce").dropna().unique())
            exp_labels = [pd.Timestamp(e).strftime("%d-%b-%Y") for e in exps]
            exp_pick = c2.selectbox("Expiry", exp_labels, key="fno_exp")
            chosen = exps[exp_labels.index(exp_pick)]

            view = sub[pd.to_datetime(sub["EXPIRY"], errors="coerce") == chosen].copy()
            sv = pd.to_numeric(view["UNDERLYING"], errors="coerce").dropna()
            spot_val = float(sv.iloc[0]) if len(sv) else float("nan")

            with st.spinner("Solving implied volatility…"):
                view = add_implied_vols(view, snap_date)

            ce = view[view["OPT"] == "CE"].set_index("STRIKE")
            pe = view[view["OPT"] == "PE"].set_index("STRIKE")
            merged = pd.DataFrame({"STRIKE": sorted(set(ce.index) | set(pe.index))})
            for pfx, frame in (("CE", ce), ("PE", pe)):
                for col, out in [("OI","OI"),("CHG_OI","CHG_OI"),("IV %","IV"),
                                 ("CLOSE","LTP"),("VOLUME","VOL"),("PREV_CLOSE","PREV")]:
                    if col in frame.columns:
                        merged[f"{pfx}_{out}"] = frame[col].reindex(merged["STRIKE"]).values
                if f"{pfx}_LTP" in merged and f"{pfx}_PREV" in merged:
                    chg = merged[f"{pfx}_LTP"] - merged[f"{pfx}_PREV"]
                    merged[f"{pfx} view"] = [classify_oi(p, o) for p, o in
                                             zip(chg, merged.get(f"{pfx}_CHG_OI", chg * 0))]

            lots = load_lot_sizes()
            st.caption(f"Spot {spot_val:,.2f} · lot size {lots.get(sym, '—')} · "
                       "IV solved from closing prices via Black-Scholes")

            if spot_val == spot_val:
                near = merged.iloc[(merged["STRIKE"] - spot_val).abs()
                                   .argsort()[:21]].sort_values("STRIKE")
            else:
                near = merged.head(21)

            order = ["CE_OI","CE_CHG_OI","CE_IV","CE_LTP","CE view","STRIKE",
                     "PE view","PE_LTP","PE_IV","PE_CHG_OI","PE_OI"]
            st.dataframe(near[[c for c in order if c in near.columns]]
                         .style.format(precision=2, na_rep="—"),
                         use_container_width=True, height=520, hide_index=True)

            ce_oi = float(merged["CE_OI"].sum()) if "CE_OI" in merged else 0.0
            pe_oi = float(merged["PE_OI"].sum()) if "PE_OI" in merged else 0.0
            k1, k2, k3 = st.columns(3)
            k1.metric("PCR (OI)", f"{pe_oi / ce_oi:.2f}" if ce_oi else "—")
            if ce_oi:
                k2.metric("Max call OI",
                          f"{merged.loc[merged['CE_OI'].idxmax(), 'STRIKE']:,.0f}")
            if pe_oi:
                k3.metric("Max put OI",
                          f"{merged.loc[merged['PE_OI'].idxmax(), 'STRIKE']:,.0f}")

            st.markdown(
                '<div class="stale">Open interest tells you what positions exist, '
                'not who is right. The strike with the most call OI gets called '
                'resistance — it is equally consistent with writers about to be run '
                'over. This is end-of-day, so intraday shifts are invisible, and '
                'the IV column is solved from closing prices, which are unreliable '
                'on contracts that barely traded.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No F&O data stored yet. Run the *Collect bhavcopy* workflow — it "
                "now pulls the entire F&O bhavcopy in a single request, covering "
                "every underlying NSE permits options on."
            )

    # ---------------------------------------------------------- IV rank ----
    else:
        iv = load_iv_history()
        if iv.empty:
            st.info(
                "No IV history yet. The collector starts building it on its next "
                "run and one row per underlying per day accumulates from there."
            )
            st.caption(
                "This is the tab worth waiting for. Today's implied volatility "
                "in isolation says nothing — 28% is high for a large cap and low "
                "for a smallcap. Its rank against that stock's own past year is "
                "the useful number, and no free source sells it to you "
                "retroactively. It exists only if you collected it."
            )
        else:
            days = iv["DATE"].nunique()
            st.caption(f"{days} day(s) collected · "
                       f"{iv['DATE'].min():%d %b %Y} to {iv['DATE'].max():%d %b %Y}")

            if days < 60:
                st.markdown(
                    f'<div class="stale">Only {days} days collected. IV rank needs '
                    'months before it means anything — a rank computed against two '
                    'weeks of history just says whether today beat a fortnight.</div>',
                    unsafe_allow_html=True,
                )

            latest = iv[iv["DATE"] == iv["DATE"].max()]
            rows = []
            for _, r in latest.iterrows():
                hist = iv[iv["SYMBOL"] == r["SYMBOL"]]["ATM_IV"]
                rank, pctile, n = iv_rank(hist, r["ATM_IV"])
                rows.append({
                    "Symbol": r["SYMBOL"], "Spot": r.get("UNDERLYING"),
                    "ATM IV %": r["ATM_IV"], "IV rank": rank,
                    "IV percentile": pctile, "Days": n,
                    "PCR": r.get("PCR"), "Max pain": r.get("MAX_PAIN"),
                    "Expiry": r.get("EXPIRY"),
                })
            table = pd.DataFrame(rows).sort_values("IV rank", ascending=False,
                                                   na_position="last")
            st.dataframe(table.style.format(precision=2, na_rep="—"),
                         use_container_width=True, hide_index=True)

            st.caption(
                "**IV rank** places today between the period's low and high. "
                "**IV percentile** is the share of days that were lower. They "
                "diverge when the distribution is skewed, which is precisely "
                "when the difference matters. High rank means options are "
                "expensive relative to this underlying's own history — which is "
                "an argument about pricing, not about direction."
            )

            if len(iv) > 20:
                pick = st.selectbox("Chart IV history for",
                                    sorted(iv["SYMBOL"].unique()), key="iv_pick")
                series = iv[iv["SYMBOL"] == pick].set_index("DATE")["ATM_IV"]
                st.line_chart(series, height=260)


with tab_disc, safe_tab("Disclosures"):
    st.markdown("#### Deals, insiders and the corporate calendar")

    deals, insider = load_deals(), load_insider()
    actions, meetings = load_actions(), load_meetings()

    if all(d.empty for d in (deals, insider, actions, meetings)):
        st.info(
            "Nothing collected yet. Run the *Collect bhavcopy* workflow — it now "
            "also pulls bulk and block deals, insider filings, corporate actions "
            "and board meeting dates."
        )
    else:
        view = st.radio("View", ["Bulk & block deals", "Insider filings",
                                 "Results calendar", "Corporate actions"],
                        horizontal=True, key="disc_view")

        only_mine = st.checkbox("Only my watchlist", value=False, key="disc_mine")
        watch = {t.replace(".NS", "").upper() for t in tickers}

        def narrow(df):
            if only_mine and not df.empty and "SYMBOL" in df.columns:
                return df[df["SYMBOL"].astype(str).str.upper().isin(watch)]
            return df

        if view == "Bulk & block deals":
            d = narrow(deals)
            if d.empty:
                st.info("No deals stored for this selection.")
            else:
                side = st.multiselect("Side", sorted(d["SIDE"].dropna().unique()),
                                      key="disc_side")
                if side:
                    d = d[d["SIDE"].isin(side)]
                d = d.sort_values("DATE", ascending=False)
                st.caption(f"{len(d):,} deals · {d['SYMBOL'].nunique()} companies")
                st.dataframe(d.head(400), use_container_width=True, height=460,
                             hide_index=True)
                st.markdown(
                    '<div class="stale">Every trade above half a percent of equity '
                    'is reported here with the buyer named. That makes it a '
                    'disclosure rather than a rumour — but a named fund buying is '
                    'not a recommendation, and bulk deals are frequently one side '
                    'of a block that was pre-negotiated at an agreed price.</div>',
                    unsafe_allow_html=True)

        elif view == "Insider filings":
            i = narrow(insider)
            if i.empty:
                st.info("No insider filings stored for this selection.")
            else:
                i = i.sort_values("DATE", ascending=False)
                st.caption(f"{len(i):,} filings · {i['SYMBOL'].nunique()} companies")
                st.dataframe(i.head(400), use_container_width=True, height=460,
                             hide_index=True)
                st.markdown(
                    '<div class="stale">SEBI Regulation 7(2) disclosures. Promoter '
                    'buying in the open market is one of the few genuinely '
                    'informative signals in a small cap. Read the <b>MODE</b> '
                    'column carefully — shares received under an ESOP or a gift '
                    'are not someone spending their own money, and <b>pledge</b> '
                    'entries are the risk the fundamentals screen cannot see.</div>',
                    unsafe_allow_html=True)

        elif view == "Results calendar":
            m = narrow(meetings)
            if m.empty:
                st.info("No board meetings stored for this selection.")
            else:
                today = pd.Timestamp(datetime.now().date())
                m = m[m["MEETING_DATE"] >= today].sort_values("MEETING_DATE")
                m = m.assign(**{"Days away": (m["MEETING_DATE"] - today).dt.days})
                st.caption(f"{len(m):,} upcoming board meetings")
                st.dataframe(m, use_container_width=True, height=460, hide_index=True)
                st.caption(
                    "Board meetings are when results are approved. For options, a "
                    "date inside your holding period is exactly when implied "
                    "volatility collapses — the Options tab now warns about this."
                )

        else:
            a = narrow(actions)
            if a.empty:
                st.info("No corporate actions stored for this selection.")
            else:
                today = pd.Timestamp(datetime.now().date())
                a = a[a["EX_DATE"] >= today].sort_values("EX_DATE")
                st.caption(f"{len(a):,} upcoming ex-dates")
                st.dataframe(a, use_container_width=True, height=460, hide_index=True)
                st.caption(
                    "Price drops by roughly the dividend on the ex-date. That is "
                    "not a loss, but it will show up as one on any chart."
                )


with tab_journal, safe_tab("Journal"):
    st.markdown("#### Trade journal")
    st.markdown(
        "Log what you took and why, before you know the outcome. Six months of "
        "this tells you which setups work **for you**, which no generic backtest "
        "can — and it catches the thing memory always hides: that you took the "
        "trade for a different reason than you now remember."
    )

    journal = load_journal()
    scored = score_journal(journal, cached_history())
    stats = journal_stats(scored)

    if stats:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Closed trades", stats["closed"])
        m2.metric("Hit rate",
                  f"{stats['hit_rate']:.0f}%" if stats["hit_rate"] == stats["hit_rate"] else "—")
        m3.metric("Median return",
                  f"{stats['median_return']:+.1f}%" if stats["median_return"] == stats["median_return"] else "—")
        m4.metric("Average R",
                  f"{stats['avg_r']:+.2f}" if stats["avg_r"] == stats["avg_r"] else "—",
                  help="Profit as a multiple of the risk you planned. Above 0.3 "
                       "over many trades is a real result.")

        if stats["closed"] < 20:
            st.markdown(
                f'<div class="stale">Only {stats["closed"]} closed trades. Nothing '
                'here is a result yet — hit rate over a handful of trades is noise, '
                'and the temptation to conclude something from it is the main way '
                'journals mislead. Thirty or more before you read anything into '
                'the numbers.</div>', unsafe_allow_html=True)

    if not scored.empty:
        by_setup = scored[scored["Status"] == "closed"]
        if len(by_setup) >= 3:
            st.markdown("**By setup**")
            agg = by_setup.groupby("Setup").agg(
                Trades=("Return %", "size"),
                **{"Hit rate %": ("Return %", lambda s: (s > 0).mean() * 100)},
                **{"Median %": ("Return %", "median")},
                **{"Avg R": ("R multiple", "mean")}).reset_index()
            st.dataframe(agg.style.format(precision=2, na_rep="—"),
                         use_container_width=True, hide_index=True)

        st.markdown("**All entries**")
        st.dataframe(colour_frame(scored, ["Return %", "P&L", "R multiple"]),
                     column_config=help_config(scored),
                     use_container_width=True, height=360, hide_index=True)

    st.divider()
    st.markdown("**Log a trade**")
    st.caption(
        "Fill in the thesis and invalidation before you know the outcome — those "
        "two columns are the entire value of this. Download and commit the file "
        "to your repo to keep it."
    )

    edited_j = st.data_editor(
        journal if not journal.empty else pd.DataFrame(
            [{c: "" for c in JOURNAL_COLUMNS}]),
        num_rows="dynamic", use_container_width=True, key="journal_editor",
        column_config={
            "date": st.column_config.TextColumn("Date", help="YYYY-MM-DD"),
            "symbol": st.column_config.TextColumn("Symbol"),
            "setup": st.column_config.SelectboxColumn(
                "Setup", options=["Range breakout", "52-week high breakout",
                                  "Pullback to 50DMA", "Oversold in uptrend",
                                  "Delivery accumulation", "Breakdown",
                                  "Fundamental", "Options", "Other"]),
            "bias": st.column_config.SelectboxColumn("Bias", options=["long", "short"]),
            "entry": st.column_config.NumberColumn("Entry", format="%.2f"),
            "stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            "target": st.column_config.NumberColumn("Target", format="%.2f"),
            "qty": st.column_config.NumberColumn("Qty", format="%d"),
            "thesis": st.column_config.TextColumn("Why I took it", width="large"),
            "invalidation": st.column_config.TextColumn(
                "What would prove me wrong", width="large"),
            "status": st.column_config.SelectboxColumn("Status", options=["open", "closed"]),
            "exit_price": st.column_config.NumberColumn("Exit", format="%.2f"),
            "exit_date": st.column_config.TextColumn("Exit date"),
            "notes": st.column_config.TextColumn("Notes after the fact", width="large"),
        })

    st.download_button("Download journal.csv",
                       data=edited_j.to_csv(index=False).encode("utf-8"),
                       file_name="journal.csv", mime="text/csv", key="jr_dl",
                       help="Upload to data/ in your repo to make it permanent.")


with tab_hunt, safe_tab("Small-cap hunt"):
    st.markdown("#### Small-cap factor screen")

    st.markdown(
        "Cupid before its run had a specific profile: **ROCE 33.5%, ROE 27.3%, "
        "debt-to-equity 0.13x**, with revenue and profit compounding at roughly "
        "**35% and 50%** over three years, in a company small enough that modest "
        "absolute growth moved the percentages hard."
    )

    st.markdown(
        '<div class="stale"><b>Read this before using the output.</b> Hundreds of '
        'microcaps shared that profile at the same time. Almost none became Cupid; '
        'a good number went to zero. Screening for the profile returns the whole '
        'cohort, not the winner — the survivors are only visible in hindsight, which '
        'is precisely the information this screen does not have. Treat the output as '
        'a research queue to investigate one by one, not a list of candidates to buy. '
        'The two things that separated Cupid from its cohort — an export order cycle '
        'and execution — appear in neither the price data nor the ratios.</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Stage 1 — narrow the market on price and volume**")
    st.caption("Runs off the bhavcopy, so it covers every listed stock at no cost.")

    screen_hunt, hunt_date, _ = load_market_screen()

    if screen_hunt.empty:
        st.info("Load the Screener tab first — this reuses the same market data.")
    else:
        h1, h2, h3 = st.columns(3)
        px_max = h1.number_input("Max price", value=200.0, step=25.0, key="hunt_px",
                                 help="A low price is not a low valuation. This is "
                                      "only a crude proxy for a small company.")
        turn_min = h2.number_input("Min turnover (Cr)", value=1.0, step=0.5,
                                   key="hunt_turnover",
                                   help="Below ~1 Cr you may not be able to exit.")
        deliv_min = h3.number_input("Min delivery %", value=40.0, step=5.0,
                                    key="hunt_delivery",
                                    help="High delivery means buyers are holding, "
                                         "not day-trading.")

        cand = screen_hunt.copy()
        cand = cand[cand["Close"].fillna(1e9) <= px_max]
        cand = cand[cand["Turnover (Cr)"].fillna(0) >= turn_min]
        if "Delivery %" in cand.columns:
            cand = cand[cand["Delivery %"].fillna(0) >= deliv_min]

        st.caption(f"{len(cand):,} stocks pass stage 1.")
        st.dataframe(
            colour_frame(cand.sort_values("Turnover (Cr)", ascending=False).head(300),
                         ["Day %", "1M %"]),
            use_container_width=True, height=320, hide_index=True,
        )

        st.divider()
        st.markdown("**Stage 2 — check fundamentals against the profile**")
        st.caption(
            "Fundamentals come one company at a time, so pick up to 30. "
            "Thresholds default to Cupid's actual figures; loosen them, since "
            "matching all four exactly will usually return nothing."
        )

        picks = st.multiselect("Companies to check",
                               options=cand["Symbol"].head(300).tolist(),
                               max_selections=30, key="hunt_picks")

        c1, c2, c3, c4 = st.columns(4)
        min_roce = c1.number_input("Min ROCE %", value=20.0, step=5.0, key="hunt_roce")
        max_de_h = c2.number_input("Max D/E", value=0.5, step=0.1, key="hunt_de")
        min_growth = c3.number_input("Min rev growth %", value=20.0, step=5.0, key="hunt_growth")
        max_mcap = c4.number_input("Max mkt cap (Cr)", value=5000.0, step=500.0, key="hunt_mcap")

        if picks and st.button("Check fundamentals", type="primary", key="hunt_go"):
            bar = st.progress(0.0, text="Starting…")
            rows = []
            for i, s in enumerate(picks):
                rows.append(fetch_ratios([f"{s}.NS"]))
                bar.progress((i + 1) / len(picks), text=f"{s} — {i + 1} of {len(picks)}")
            bar.empty()
            st.session_state["_hunt_data"] = (
                pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())

        hunt_data = st.session_state.get("_hunt_data", pd.DataFrame())
        if not hunt_data.empty:
            hd = hunt_data.copy()
            hd["Matches"] = (
                (hd["ROCE %"].fillna(-1e9) >= min_roce).astype(int)
                + (hd["D/E"].fillna(1e9) <= max_de_h * 100).astype(int)
                + (hd["Rev Growth %"].fillna(-1e9) >= min_growth).astype(int)
                + (hd["Mkt Cap (Cr)"].fillna(1e9) <= max_mcap).astype(int)
            )
            cols = ["Ticker", "Name", "Sector", "Matches", "Mkt Cap (Cr)", "ROCE %",
                    "ROE %", "D/E", "Rev Growth %", "Profit Growth %", "P/E", "P/B"]
            hd = hd[[c for c in cols if c in hd.columns]].sort_values(
                "Matches", ascending=False)
            st.dataframe(hd.style.format(precision=2, na_rep="—"),
                         use_container_width=True, hide_index=True)
            st.caption(
                "**Matches** counts how many of the four thresholds each company "
                "clears. Blanks mean Yahoo has no data for that field — common in "
                "exactly this size band, so a blank is a reason to go read the "
                "annual report, not to exclude the company."
            )
            st.markdown(
                '<div class="stale">Two checks this screen cannot make, both of '
                'which sank companies that looked like this: <b>promoter pledging</b> '
                '(Cupid itself has run around 25% of promoter holding pledged) and '
                '<b>auditor or governance flags</b>. Neither is in any free feed. '
                'Check the shareholding pattern on the BSE filing page before '
                'putting money behind anything here.</div>',
                unsafe_allow_html=True,
            )


with tab_test, safe_tab("Backtest"):
    st.markdown("#### Backtest a screen")
    glossary_panel("What do hit rate, drawdown and beat market mean?")
    st.markdown(
        "Apply your filters to the market **as it looked on a past date**, then "
        "show what every selected name actually did afterwards. Losers included, "
        "delisted names flagged rather than dropped."
    )

    hist = cached_history()

    if hist.empty:
        st.markdown(
            '<div class="stale">No stored history yet. This tab needs '
            '<code>collect.py</code> to have run at least once — see the setup note '
            'below. Nothing else in the app depends on it.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "**The normal route is the GitHub Action.** In your repo, open the "
            "**Actions** tab, choose *Collect bhavcopy*, click **Run workflow** and "
            "enter a backfill of `400`. It commits the data straight to "
            "`data/history/` and then runs itself every weekday at 19:30 IST.\n\n"
            "If that has already run and this message is still showing, Streamlit "
            "has not picked up the new files — use **Manage app → Reboot**.\n\n"
            "The collector below is a manual fallback for when the Action fails."
        )

        st.divider()
        st.markdown("**Build the history file**")

        bf_days = st.select_slider(
            "How far back", options=[90, 180, 270, 400, 550, 730], value=400,
            help="Calendar days. 400 is about 18 months and takes roughly "
                 "5-8 minutes. Start smaller if you want to check it works first.",
        )
        st.caption(
            f"Roughly {int(bf_days * 5 / 7)} trading days. Requests are throttled "
            "to stay under NSE's rate limit, so this is slow by design. Leave the "
            "tab open while it runs."
        )

        if st.button("Collect history now", type="primary"):
            bar = st.progress(0.0, text="Starting…")
            status = st.empty()
            frames, ok, miss = [], 0, 0

            for i in range(bf_days):
                day = datetime.now() - timedelta(days=i)
                if day.weekday() >= 5:
                    continue
                bar.progress(min(i / bf_days, 1.0),
                             text=f"{day:%d %b %Y} — {ok} days collected, {miss} skipped")
                df = _fetch_bhav_for(day)
                if df.empty:
                    miss += 1
                else:
                    if "SERIES" in df.columns:
                        df = df[df["SERIES"].astype(str).str.strip().str.upper() == "EQ"]
                    frames.append(pd.DataFrame({
                        "DATE": day.strftime("%Y-%m-%d"),
                        "SYMBOL": df["SYMBOL"],
                        "CLOSE": pd.to_numeric(df.get("CLOSE_PRICE"), errors="coerce"),
                        "PREV_CLOSE": pd.to_numeric(df.get("PREV_CLOSE"), errors="coerce"),
                        "VOLUME": pd.to_numeric(df.get("TTL_TRD_QNTY"), errors="coerce"),
                        "TURNOVER_LACS": pd.to_numeric(df.get("TURNOVER_LACS"), errors="coerce"),
                        "DELIV_QTY": pd.to_numeric(df.get("DELIV_QTY"), errors="coerce"),
                        "DELIV_PER": pd.to_numeric(df.get("DELIV_PER"), errors="coerce"),
                    }).dropna(subset=["CLOSE"]))
                    ok += 1
                time.sleep(0.35)

            bar.empty()
            if not frames:
                status.error(
                    "Nothing collected. Check the Screener tab's fetch log — if "
                    "that is also failing, NSE is refusing this server too."
                )
            else:
                collected = pd.concat(frames, ignore_index=True)
                st.session_state["_collected"] = collected
                status.success(
                    f"{ok} trading days collected, {miss} unavailable "
                    f"(holidays and refusals). {len(collected):,} rows."
                )

        collected = st.session_state.get("_collected", pd.DataFrame())
        if not collected.empty:
            st.divider()
            st.markdown("**Save it to your repo**")

            months = sorted(collected["DATE"].str[:7].unique())
            st.caption(
                f"{len(months)} monthly files, {collected['SYMBOL'].nunique():,} "
                "distinct symbols. Download each and upload them into a "
                "`data/history/` folder in your repo."
            )

            for m in months:
                chunk = collected[collected["DATE"].str[:7] == m].sort_values(
                    ["DATE", "SYMBOL"])
                buf = io.BytesIO()
                chunk.to_csv(buf, index=False, compression="gzip")
                st.download_button(
                    f"{m}.csv.gz  ·  {len(chunk):,} rows",
                    data=buf.getvalue(), file_name=f"{m}.csv.gz",
                    mime="application/gzip", key=f"dl_{m}",
                )

            st.markdown(
                "In GitHub: **Add file → Upload files**, drag all of these in, and "
                "set the path to `data/history/` before committing. Refresh this "
                "app afterwards and the backtest becomes available."
            )
    else:
        dates = sorted(hist["DATE"].unique())
        span_days = (dates[-1] - dates[0]).days
        st.caption(
            f"{len(dates)} trading days stored · "
            f"{dates[0]:%d %b %Y} to {dates[-1]:%d %b %Y} · "
            f"{hist['SYMBOL'].nunique():,} distinct symbols"
        )

        if span_days < 90:
            st.warning(
                f"Only {span_days} days of history. Results below will be noise "
                "until you have at least a year — run the backfill."
            )

        d1, d2 = st.columns(2)
        as_of = d1.date_input(
            "Screen as of", value=dates[max(0, len(dates) - 130)].date(),
            min_value=dates[0].date(), max_value=dates[-1].date(),
        )
        end_on = d2.date_input(
            "Measure returns to", value=dates[-1].date(),
            min_value=dates[0].date(), max_value=dates[-1].date(),
        )

        if pd.Timestamp(end_on) <= pd.Timestamp(as_of):
            st.error("The end date must be after the screening date.")
        else:
            with st.spinner("Rebuilding the market as of that date…"):
                snap = snapshot_at(hist, as_of)

            if snap.empty:
                st.warning("Not enough history before that date. Pick a later one.")
            else:
                st.markdown("**Screen criteria** — as they would have been applied then")
                c1, c2, c3 = st.columns(3)
                bt_px = c1.number_input("Max price", value=200.0, step=25.0, key="bt_px")
                bt_turn = c2.number_input("Min turnover (Cr)", value=1.0, step=0.5, key="bt_turn")
                bt_deliv = c3.number_input("Min delivery %", value=40.0, step=5.0, key="bt_deliv")

                c4, c5, c6 = st.columns(3)
                bt_rsi_lo = c4.number_input("RSI min", value=0.0, step=5.0, key="bt_rlo")
                bt_rsi_hi = c5.number_input("RSI max", value=100.0, step=5.0, key="bt_rhi")
                bt_vol = c6.number_input("Min vol vs 20d", value=0.0, step=0.5, key="bt_vol")

                bt_above50 = st.checkbox("Only above 50DMA", key="bt_50")

                sel = snap.copy()
                sel = sel[sel["Close"].fillna(1e9) <= bt_px]
                sel = sel[sel["Turnover (Cr)"].fillna(0) >= bt_turn]
                if bt_deliv > 0 and "Delivery %" in sel.columns:
                    sel = sel[sel["Delivery %"].fillna(0) >= bt_deliv]
                sel = sel[sel["RSI (14)"].fillna(-1).between(bt_rsi_lo, bt_rsi_hi)]
                if bt_vol > 0 and "Vol vs 20d" in sel.columns:
                    sel = sel[sel["Vol vs 20d"].fillna(0) >= bt_vol]
                if bt_above50 and "vs 50DMA %" in sel.columns:
                    sel = sel[sel["vs 50DMA %"].fillna(-1e9) > 0]

                st.caption(f"{len(sel):,} stocks would have been selected on {as_of:%d %b %Y}.")

                if sel.empty:
                    st.info("No stocks matched. Loosen the filters.")
                elif len(sel) > 400:
                    st.warning(
                        f"{len(sel):,} names is too broad to be a screen — you are "
                        "holding most of the market. Tighten the filters."
                    )
                else:
                    fwd = forward_returns(hist, sel["Symbol"].tolist(), as_of, end_on)
                    bench = benchmark_return(hist, as_of, end_on)
                    valid = fwd["Return %"].dropna()

                    if valid.empty:
                        st.warning("No forward data for these names.")
                    else:
                        st.divider()
                        st.markdown("**What actually happened**")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Median return", f"{valid.median():+.1f}%",
                                  delta=f"{valid.median() - bench:+.1f}% vs market",
                                  help="The market bar is the median return of every "
                                       "stock that traded over the same window.")
                        m2.metric("Hit rate", f"{(valid > 0).mean():.0%}",
                                  help="Share that finished positive.")
                        m3.metric("Beat market", f"{(valid > bench).mean():.0%}")
                        m4.metric("Mean return", f"{valid.mean():+.1f}%",
                                  help="Skewed by outliers — compare against the median.")

                        n1, n2, n3, n4 = st.columns(4)
                        n1.metric("Doubled or better", f"{int((valid >= 100).sum())}")
                        n2.metric("Lost half or more", f"{int((valid <= -50).sum())}")
                        n3.metric("Best", f"{valid.max():+.0f}%")
                        n4.metric("Worst", f"{valid.min():+.0f}%")

                        stopped = fwd[fwd["Status"].astype(str).str.startswith("stopped")]
                        if len(stopped):
                            st.markdown(
                                f'<div class="stale">{len(stopped)} of these stopped '
                                'trading before the end date — suspension, delisting or '
                                'illiquidity. Their last printed price is used, which '
                                '<b>overstates</b> the result, since a suspended stock '
                                'is rarely sellable at its final quote.</div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            f"Median stock in the whole market over the same period: "
                            f"**{bench:+.1f}%**. If your screen's median is not "
                            "meaningfully above that, the filters are not adding "
                            "anything beyond market direction."
                        )

                        st.dataframe(
                            colour_frame(
                                fwd.sort_values("Return %", ascending=False),
                                ["Return %", "Peak gain %", "Max drawdown %"]),
                            use_container_width=True, height=400, hide_index=True,
                        )
                        st.download_button(
                            "Download results",
                            data=fwd.to_csv(index=False).encode("utf-8"),
                            file_name=f"backtest_{as_of}_{end_on}.csv", mime="text/csv")

                        st.caption(
                            "**Peak gain** and **max drawdown** matter as much as the "
                            "final return: a name that ended +40% after being -60% "
                            "along the way is not one most people would have held. "
                            "One caveat this cannot fix — the store only contains "
                            "dates you have collected, so a screen tested over a "
                            "single market regime tells you about that regime, not "
                            "about the screen."
                        )


with tab_news, safe_tab("News"):
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


with tab_flows, safe_tab("FII/DII"):
    st.markdown("#### Institutional flows")
    glossary_panel("What are FII, DII and A/D ratio?")

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
                     column_config=help_config(flows_df),
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
        st.dataframe(oi,
                     column_config=help_config(oi), use_container_width=True, hide_index=True)


with tab_ratios, safe_tab("Ratios"):
    st.markdown("#### Fundamentals")
    glossary_panel("What do these ratios mean?")
    st.caption("Cached six hours — the source is rate-limited, so repeated refreshes "
               "will get you throttled.")

    shortlist = st.session_state.get("shortlist", [])
    source = st.radio(
        "Source", key="ratio_source",
        options=["Watchlist", "Screener shortlist"],
        horizontal=True,
        index=0,
        help="Build a shortlist on the Screener tab to analyse screened names.",
    )

    if source == "Screener shortlist":
        if not shortlist:
            st.info("No shortlist yet. Pick symbols on the Screener tab first.")
            target = ()
        else:
            target = tuple(shortlist)
            st.caption(f"{len(target)} symbols from your screen.")
    else:
        target = tickers

    # Fetching fundamentals is slow and Streamlit re-runs every tab on every
    # interaction, so this is explicitly triggered rather than automatic.
    # Otherwise it blocks the whole app on each click.
    cache_key = str(sorted(target))
    have_cached = st.session_state.get("_ratio_key") == cache_key
    ratios = st.session_state.get("_ratio_data", pd.DataFrame()) if have_cached else pd.DataFrame()

    if target:
        label = ("Reload fundamentals" if have_cached
                 else f"Load fundamentals for {len(target)} companies")
        est = int(len(target) * 4)
        if st.button(label, type="primary"):
            bar = st.progress(0.0, text="Starting…")
            rows = []
            for i, t in enumerate(target):
                rows.append(fetch_ratios([t]))
                bar.progress((i + 1) / len(target),
                             text=f"{t} — {i + 1} of {len(target)}")
            bar.empty()
            ratios = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            st.session_state["_ratio_key"] = cache_key
            st.session_state["_ratio_data"] = ratios

        if not have_cached and ratios.empty:
            st.info(
                f"Click above to fetch. Roughly {est // 60}m {est % 60}s for "
                f"{len(target)} companies — the source allows one at a time. "
                "Results stay loaded until you change the selection."
            )

    if ratios.empty:
        if not target:
            st.info("Nothing selected — pick a source above.")
        # otherwise the "click above to fetch" prompt already explains things
    else:
        c1, c2, c3 = st.columns(3)
        max_pe = c1.number_input("Max P/E", value=0.0, key="ratio_pe",
                                 help="0 disables this filter")
        min_roe = c2.number_input("Min ROE %", value=0.0, key="ratio_roe")
        max_de = c3.number_input("Max D/E", value=0.0, key="ratio_de",
                                 help="0 disables this filter")

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


with tab_book, safe_tab("Order Book"):
    st.markdown("#### Order backlog")
    glossary_panel("What is book-to-bill?")
    st.markdown(
        "Company order books are disclosed in quarterly investor presentations and "
        "nowhere else — no data feed carries them. Log the number here after each "
        "result and this tab does the analysis. The News tab flags order-win filings "
        "so you know when there's something to log."
    )

    raw_backlog = load_backlog()
    analysis = cached_backlog_analysis()

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


with tab_depth, safe_tab("Depth"):
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
