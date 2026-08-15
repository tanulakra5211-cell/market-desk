"""
Market Desk -- an Indian equities dashboard.

Single-file version. Nothing else is required except requirements.txt.
Watchlist and order-book data are built in, and are overridden by
data/watchlist.csv and data/order_backlog.csv if those files exist.

Run locally:  streamlit run app.py
"""

import io
import time
import zipfile
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
        df = pd.read_csv(io.StringIO(resp.text))
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
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]
            for col in df.columns:
                if df[col].dtype == object:
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
                raw = pd.read_csv(zf.open(name))
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


@st.cache_data(ttl=21600)
def cached_backlog_analysis() -> pd.DataFrame:
    """Cached — this hits yfinance per ticker for the revenue denominator."""
    return analyse_backlog(load_backlog())


def load_watchlist() -> pd.DataFrame:
    path = DATA_DIR / "watchlist.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"bse_scrip": str})
    return pd.read_csv(io.StringIO(DEFAULT_WATCHLIST), dtype={"bse_scrip": str})


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


@st.cache_data(ttl=900)
def load_technicals(tickers: tuple):
    return fetch_technicals(list(tickers))


@st.cache_data(ttl=86400)
def load_universe():
    return fetch_nse_universe()


@st.cache_data(ttl=3600)
def load_market_screen():
    """Whole-market table, the date it represents, and the fetch attempt log."""
    latest, latest_date, attempts = fetch_latest_bhavcopy()
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

(tab_market, tab_screen, tab_tech, tab_news, tab_flows,
 tab_ratios, tab_book, tab_depth) = st.tabs(
    ["Markets", "Screener (all NSE)", "Technicals", "News", "FII / DII",
     "Ratios", "Order Book", "Depth"]
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


with tab_screen:
    st.markdown("#### Whole-market screener")
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
        st.caption(f"{len(screen):,} stocks · trading day {screen_date}")

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
        min_turnover = f1.number_input("Min turnover (Cr)", value=5.0, step=1.0)
        min_delivery = f2.number_input("Min delivery %", value=0.0, step=5.0)
        min_price = f3.number_input("Min price", value=0.0, step=10.0)
        max_price = f4.number_input("Max price (0 = no cap)", value=0.0, step=100.0)

        g1, g2 = st.columns(2)
        move_min = g1.number_input("Min day % move", value=-100.0, step=1.0)
        move_max = g2.number_input("Max day % move", value=100.0, step=1.0)

        view = screen.copy()
        view = view[view["Turnover (Cr)"].fillna(0) >= min_turnover]
        if min_delivery > 0 and "Delivery %" in view.columns:
            view = view[view["Delivery %"].fillna(0) >= min_delivery]
        if min_price > 0:
            view = view[view["Close"].fillna(0) >= min_price]
        if max_price > 0:
            view = view[view["Close"].fillna(1e12) <= max_price]
        view = view[view["Day %"].between(move_min, move_max, inclusive="both")]

        sort_options = [c for c in ["Turnover (Cr)", "Day %", "1M %", "Delivery %",
                                    "Volume", "Close"] if c in view.columns]
        s1, s2 = st.columns([3, 1])
        sort_by = s1.selectbox("Sort by", options=sort_options)
        descending = s2.checkbox("Descending", value=True)
        view = view.sort_values(sort_by, ascending=not descending, na_position="last")

        st.caption(f"{len(view):,} stocks pass the filters.")
        st.dataframe(
            colour_frame(view.head(500), ["Day %", "1M %"]),
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


with tab_tech:
    st.markdown("#### Technical position")
    st.markdown(
        "Where each stock sits relative to its own moving averages, range and "
        "average volume. These describe what price has **already** done. None of "
        "them forecasts what it will do next — a stock can sit above its 200DMA "
        "with RSI at 65 all the way down."
    )

    tech_source = st.radio("Source", ["Watchlist", "Screener shortlist"],
                           horizontal=True, key="tech_source")
    tech_target = (tuple(st.session_state.get("shortlist", []))
                   if tech_source == "Screener shortlist" else tickers)

    if not tech_target:
        st.info("Nothing selected. Build a shortlist on the Screener tab first.")
    else:
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
    st.markdown("#### Fundamentals")
    st.caption("Cached six hours — the source is rate-limited, so repeated refreshes "
               "will get you throttled.")

    shortlist = st.session_state.get("shortlist", [])
    source = st.radio(
        "Source",
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
