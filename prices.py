"""
Prices, fundamentals and global index performance.

Everything here runs off yfinance, which works fine from cloud hosts and
covers NSE (SYMBOL.NS) and BSE (SYMBOL.BO) tickers.
"""

import pandas as pd
import yfinance as yf

# Global markets. GIFT Nifty has no reliable Yahoo symbol, so the Indian
# indices are the cash indices; overnight cues come from the rest.
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


def _pct_change(hist: pd.DataFrame) -> tuple[float, float]:
    """Return (last close, % change vs previous close)."""
    closes = hist["Close"].dropna()
    if len(closes) < 2:
        return (float(closes.iloc[-1]), 0.0) if len(closes) else (float("nan"), 0.0)
    last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
    return last, ((last - prev) / prev) * 100 if prev else 0.0


def fetch_global_markets() -> pd.DataFrame:
    """One row per global index with last level and day change."""
    symbols = list(GLOBAL_INDICES.values())
    raw = yf.download(
        symbols, period="7d", interval="1d",
        group_by="ticker", progress=False, auto_adjust=False, threads=True,
    )

    rows = []
    for name, sym in GLOBAL_INDICES.items():
        try:
            hist = raw[sym] if len(symbols) > 1 else raw
            last, chg = _pct_change(hist)
        except Exception:
            last, chg = float("nan"), 0.0
        rows.append({"Market": name, "Symbol": sym, "Last": last, "Change %": chg})

    return pd.DataFrame(rows)


def fetch_quotes(tickers: list[str]) -> pd.DataFrame:
    """Last price, day change and 1M/6M/1Y returns for a watchlist."""
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers, period="1y", interval="1d",
        group_by="ticker", progress=False, auto_adjust=False, threads=True,
    )

    rows = []
    for t in tickers:
        try:
            hist = raw[t] if len(tickers) > 1 else raw
            closes = hist["Close"].dropna()
            if closes.empty:
                continue
            last, day = _pct_change(hist)

            def ret(days: int) -> float:
                if len(closes) <= days:
                    return float("nan")
                base = float(closes.iloc[-days - 1])
                return ((last - base) / base) * 100 if base else float("nan")

            rows.append({
                "Ticker": t,
                "Last": last,
                "Day %": day,
                "1M %": ret(21),
                "6M %": ret(126),
                "1Y %": ret(250),
                "52W High": float(closes.max()),
                "52W Low": float(closes.min()),
                "Off 52W High %": ((last - closes.max()) / closes.max()) * 100,
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


def fetch_ratios(tickers: list[str]) -> pd.DataFrame:
    """
    Valuation, profitability, leverage and growth ratios.

    Note: yfinance's `.info` is slow and rate-limited -- cache this hard.
    ROCE isn't provided, so it's computed from the financial statements
    where they're available and left blank where they aren't.
    """
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}

            roce = float("nan")
            try:
                fin, bs = tk.financials, tk.balance_sheet
                ebit = fin.loc["EBIT"].iloc[0]
                assets = bs.loc["Total Assets"].iloc[0]
                cur_liab = bs.loc["Current Liabilities"].iloc[0]
                capital = assets - cur_liab
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
