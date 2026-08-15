"""
Bhavcopy collector.

Builds and maintains a local history of NSE end-of-day data so the dashboard
stops re-fetching the same files, and so screens can be tested against what
the market actually looked like on a past date.

Storage: one gzipped CSV per month under data/history/. Small monthly files
rather than one big file, so git commits stay tiny and nothing is rewritten.

Usage:
    python collect.py                  # yesterday and today (daily run)
    python collect.py --backfill 400   # seed ~400 calendar days of history
    python collect.py --from 2026-01-01 --to 2026-03-31

Delisted companies simply stop appearing in later files. That is deliberate:
the store keeps whatever traded on each date, so a stock that later went to
zero is still present on the dates it existed. Removing those rows is exactly
the survivorship bias that makes backtests lie.
"""

import argparse
import io
import re
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

HISTORY_DIR = Path(__file__).parent / "data" / "history"

NSE_BASE = "https://www.nseindia.com"
SEC_BHAV_URL = (
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
)
UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{NSE_BASE}/",
}

KEEP = ["DATE", "SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "PREV_CLOSE",
        "VOLUME", "TURNOVER_LACS", "DELIV_QTY", "DELIV_PER"]


def make_client() -> httpx.Client:
    """Cookie-primed client. NSE rejects requests without a prior homepage visit."""
    client = httpx.Client(http2=True, headers=HEADERS, timeout=30.0,
                          follow_redirects=True)
    client.get(NSE_BASE)
    time.sleep(0.5)
    client.get(f"{NSE_BASE}/market-data/live-equity-market")
    time.sleep(0.5)
    return client


def fetch_day(client: httpx.Client, date: datetime, diag: list) -> pd.DataFrame:
    """One trading day, normalised. Empty frame on a holiday or failure."""
    archive_headers = {
        "Referer": "https://www.nseindia.com/all-reports",
        "Accept": "text/csv,application/zip,*/*",
    }

    # Preferred: security-wise file, which carries delivery data.
    url = SEC_BHAV_URL.format(ddmmyyyy=date.strftime("%d%m%Y"))
    try:
        resp = client.get(url, headers=archive_headers)
        diag.append(resp.status_code)

        if resp.status_code == 200:
            body = resp.text
            # Always sample on a 200 -- if parsing fails we need to see why,
            # and guessing has already cost several rounds.
            sample = body[:220].replace("\n", " | ").replace("\r", "")
            diag.append(f"{len(body)}B head={sample!r}")

            df = pd.read_csv(io.StringIO(body), skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            diag.append(f"parsed {len(df)}r cols={list(df.columns)[:16]}")

            for c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].astype(str).str.strip()

            if "SERIES" in df.columns:
                before = len(df)
                df = df[df["SERIES"].str.upper() == "EQ"]
                diag.append(f"EQ filter {before}->{len(df)}")

            if "SYMBOL" not in df.columns or "CLOSE_PRICE" not in df.columns:
                diag.append(f"MISSING cols, have={list(df.columns)}")
                return pd.DataFrame()

            out = pd.DataFrame({
                "DATE": date.strftime("%Y-%m-%d"),
                "SYMBOL": df["SYMBOL"],
                "OPEN": pd.to_numeric(df.get("OPEN_PRICE"), errors="coerce"),
                "HIGH": pd.to_numeric(df.get("HIGH_PRICE"), errors="coerce"),
                "LOW": pd.to_numeric(df.get("LOW_PRICE"), errors="coerce"),
                "CLOSE": pd.to_numeric(df["CLOSE_PRICE"], errors="coerce"),
                "PREV_CLOSE": pd.to_numeric(df.get("PREV_CLOSE"), errors="coerce"),
                "VOLUME": pd.to_numeric(df.get("TTL_TRD_QNTY"), errors="coerce"),
                "TURNOVER_LACS": pd.to_numeric(df.get("TURNOVER_LACS"), errors="coerce"),
                "DELIV_QTY": pd.to_numeric(df.get("DELIV_QTY"), errors="coerce"),
                "DELIV_PER": pd.to_numeric(df.get("DELIV_PER"), errors="coerce"),
            })
            before = len(out)
            out = out.dropna(subset=["CLOSE"])
            diag.append(f"dropna CLOSE {before}->{len(out)}")
            if not out.empty:
                return out
    except Exception as exc:  # noqa: BLE001
        diag.append(f"EXC {type(exc).__name__}: {str(exc)[:120]}")

    # Fallback: UDiFF zip. No delivery data, but better than a gap.
    url = UDIFF_URL.format(yyyymmdd=date.strftime("%Y%m%d"))
    try:
        resp = client.get(url, headers=archive_headers)
        diag.append(resp.status_code)
        if resp.status_code != 200:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            raw = pd.read_csv(zf.open(zf.namelist()[0]), skipinitialspace=True)
        raw.columns = [c.strip() for c in raw.columns]
        raw = raw[raw.get("FinInstrmTp", "STK") == "STK"]
        if "SctySrs" in raw.columns:
            raw = raw[raw["SctySrs"].astype(str).str.strip().str.upper() == "EQ"]
        out = pd.DataFrame({
            "DATE": date.strftime("%Y-%m-%d"),
            "SYMBOL": raw["TckrSymb"],
            "OPEN": pd.to_numeric(raw.get("OpnPric"), errors="coerce"),
            "HIGH": pd.to_numeric(raw.get("HghPric"), errors="coerce"),
            "LOW": pd.to_numeric(raw.get("LwPric"), errors="coerce"),
            "CLOSE": pd.to_numeric(raw["ClsPric"], errors="coerce"),
            "PREV_CLOSE": pd.to_numeric(raw["PrvsClsgPric"], errors="coerce"),
            "VOLUME": pd.to_numeric(raw["TtlTradgVol"], errors="coerce"),
            "TURNOVER_LACS": pd.to_numeric(raw["TtlTrfVal"], errors="coerce") / 1e5,
            "DELIV_QTY": float("nan"),
            "DELIV_PER": float("nan"),
        })
        return out.dropna(subset=["CLOSE"])
    except Exception as exc:  # noqa: BLE001
        print(f"    udiff failed: {exc}", file=sys.stderr)

    return pd.DataFrame()



FLOWS_PATH = Path(__file__).parent / "data" / "flows.csv"


def collect_flows(client: httpx.Client) -> bool:
    """
    Daily FII/DII cash-market activity.

    The app's host is rate-limited by NSE, so this has to be gathered here and
    committed, same as the bhavcopy. Appends to data/flows.csv, skipping dates
    already present.
    """
    try:
        resp = client.get(f"{NSE_BASE}/api/fiidiiTradeReact")
        if resp.status_code != 200:
            print(f"  flows: HTTP {resp.status_code}")
            return False
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  flows: {type(exc).__name__}: {str(exc)[:80]}")
        return False

    rows = []
    for item in data or []:
        try:
            rows.append({
                "DATE": pd.to_datetime(item.get("date"), dayfirst=True,
                                       errors="coerce").strftime("%Y-%m-%d"),
                "PARTICIPANT": str(item.get("category", "")).strip(),
                "BUY_CR": float(item.get("buyValue", 0) or 0),
                "SELL_CR": float(item.get("sellValue", 0) or 0),
            })
        except Exception:
            continue

    if not rows:
        print("  flows: nothing parseable returned")
        return False

    new = pd.DataFrame(rows).dropna(subset=["DATE"])
    FLOWS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if FLOWS_PATH.exists():
        prev = pd.read_csv(FLOWS_PATH, skipinitialspace=True)
        combined = pd.concat([prev, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["DATE", "PARTICIPANT"],
                                            keep="last")
    else:
        combined = new

    combined = combined.sort_values(["DATE", "PARTICIPANT"])
    combined.to_csv(FLOWS_PATH, index=False)
    print(f"  flows: {len(new)} row(s) for {new['DATE'].iloc[0]} "
          f"({len(combined)} total on file)")
    return True






SHP_PATH = Path(__file__).parent / "data" / "shareholding.csv"
MF_PATH = Path(__file__).parent / "data" / "mf_holdings.csv"

# Symbols to track shareholding for. Per-company requests, so this is a list
# rather than the whole market. Quarterly data, so a weekly run is plenty.
SHP_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "TCS", "INFY", "ICICIBANK", "LT", "ITC", "SBIN",
    "BHARTIARTL", "RVNL", "IRCON", "BEL", "HAL", "NBCC", "PIIND", "CUPID",
]


def _num(v):
    """NSE mixes numbers, strings and dashes in the same field."""
    try:
        s = str(v).replace(",", "").replace("%", "").strip()
        return float(s) if s and s not in ("-", "NA", "None") else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def fetch_shareholding(client: httpx.Client, symbol: str):
    """
    One company's quarterly shareholding pattern.

    NSE has moved this endpoint before, so several known paths are tried and
    the one that answers is reported. Parsing is deliberately loose because
    the field names differ between them.
    """
    paths = [
        f"/api/corporate-share-holdings-master?index=equities&symbol={symbol}",
        f"/api/corporate-shareHoldings?index=equities&symbol={symbol}",
        f"/api/quote-equity?symbol={symbol}&section=corp_info",
    ]
    for path in paths:
        payload, note = nse_api(client, path,
                                f"/get-quotes/equity?symbol={symbol}")
        if payload is None:
            continue

        data = payload
        for key in ("data", "shareholdingPatterns", "corpInfo"):
            if isinstance(data, dict) and key in data:
                data = data[key]
        if isinstance(data, dict):
            data = data.get("data", list(data.values()))
        if not isinstance(data, list) or not data:
            continue

        rows = []
        for item in data:
            if not isinstance(item, dict):
                continue
            period = _pick(item, "date", "period", "asOnDate", "quarter", "name")
            if not period:
                continue
            rows.append({
                "SYMBOL": symbol,
                "PERIOD": period,
                "PROMOTER_%": _num(_pick(item, "promoter", "promoterAndPromoterGroup",
                                         "pr_and_prgrp", default="")),
                "PLEDGED_%": _num(_pick(item, "pledged", "promoterPledge",
                                        "sharesPledged", default="")),
                "FII_%": _num(_pick(item, "fii", "foreignPortfolioInvestors",
                                    "fpi", default="")),
                "DII_%": _num(_pick(item, "dii", "domesticInstitutional",
                                    default="")),
                "MF_%": _num(_pick(item, "mutualFunds", "mf", default="")),
                "PUBLIC_%": _num(_pick(item, "public", "publicShareholding",
                                       default="")),
                "SOURCE": path.split("?")[0],
            })
        if rows:
            return pd.DataFrame(rows), f"ok via {path.split('?')[0]}"

    return None, "no endpoint returned parseable shareholding data"


def collect_shareholding(client: httpx.Client) -> int:
    """
    Quarterly ownership, per company.

    The single filing is not the point -- the history is. A promoter stake
    creeping up over three quarters, or institutional holding going from 2% to
    9% in a smallcap, is a slow signal that survives being public precisely
    because it is slow. That history only exists if you collect it.
    """
    total, failures = 0, []
    for symbol in SHP_SYMBOLS:
        df, note = fetch_shareholding(client, symbol)
        if df is None:
            failures.append(f"{symbol}: {note}")
        else:
            n = _append_csv(SHP_PATH, df, ["SYMBOL", "PERIOD"])
            total += n
            print(f"    {symbol}: {len(df)} periods, {n} new")
        time.sleep(0.6)

    if failures:
        print(f"    {len(failures)} symbol(s) returned nothing:")
        for f in failures[:4]:
            print(f"      {f}")
    return total


AMFI_PORTFOLIO_INDEX = "https://www.amfiindia.com/research-information/other-data/monthly-portfolio-disclosures"


def collect_mf_disclosure_links(client: httpx.Client) -> int:
    """
    Monthly mutual fund portfolio disclosures.

    Honest warning: AMFI publishes an index page of links, and each AMC then
    posts its own spreadsheet in its own layout. There is no common schema, so
    this collects the LINKS and leaves the parsing to you rather than
    pretending to a standardisation that does not exist. Scheme-level holdings
    genuinely require per-AMC work.

    The aggregate picture -- how much of a company mutual funds own -- comes
    from the shareholding pattern above and is far more reliable.
    """
    try:
        resp = client.get(AMFI_PORTFOLIO_INDEX,
                          headers={"Referer": "https://www.amfiindia.com/"})
        if resp.status_code != 200:
            print(f"    AMFI index: HTTP {resp.status_code}")
            return 0
        html = resp.text
    except Exception as exc:  # noqa: BLE001
        print(f"    AMFI index: {type(exc).__name__}: {str(exc)[:60]}")
        return 0

    links = re.findall(r'href="([^"]+\.(?:xls|xlsx|csv|zip))"[^>]*>([^<]{3,120})<',
                       html, flags=re.I)
    if not links:
        print("    AMFI index: no disclosure links found — page layout may have changed")
        return 0

    rows = [{"COLLECTED": datetime.now().strftime("%Y-%m-%d"),
             "AMC": label.strip()[:120],
             "URL": url if url.startswith("http") else f"https://www.amfiindia.com{url}"}
            for url, label in links]
    df = pd.DataFrame(rows)
    n = _append_csv(MF_PATH, df, ["URL"])
    print(f"    AMFI disclosures: {len(df)} links, {n} new")
    return n


DEALS_PATH = Path(__file__).parent / "data" / "deals.csv"
INSIDER_PATH = Path(__file__).parent / "data" / "insider.csv"
ACTIONS_PATH = Path(__file__).parent / "data" / "corporate_actions.csv"
MEETINGS_PATH = Path(__file__).parent / "data" / "board_meetings.csv"


def nse_api(client: httpx.Client, path: str, referer: str = "/"):
    """GET an NSE /api/ endpoint with the right referer. Returns JSON or None."""
    try:
        resp = client.get(f"{NSE_BASE}{path}",
                          headers={"Referer": f"{NSE_BASE}{referer}"})
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        return resp.json(), "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:60]}"


def _append_csv(path: Path, new: pd.DataFrame, keys: list) -> int:
    """Append rows, dropping ones already stored. Returns how many were new."""
    if new.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prev = pd.read_csv(path, skipinitialspace=True, dtype=str)
        before = len(prev)
        combined = pd.concat([prev, new.astype(str)], ignore_index=True)
        combined = combined.drop_duplicates(subset=[k for k in keys
                                                    if k in combined.columns])
        added = len(combined) - before
    else:
        combined = new.astype(str).drop_duplicates(
            subset=[k for k in keys if k in new.columns])
        added = len(combined)
    combined.to_csv(path, index=False)
    return added


def _pick(row: dict, *names, default=""):
    """NSE renames response fields between endpoints; take the first that exists."""
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return default


def collect_deals(client: httpx.Client, days_back: int = 7) -> int:
    """
    Bulk and block deals — every trade above half a percent of equity, named.

    This is the closest thing to seeing who is actually buying. A promoter or
    a known fund appearing on the buy side is a disclosure, not a rumour, and
    it lands here before it reaches any headline.
    """
    end = datetime.now()
    start = end - timedelta(days=days_back)
    span = f"from={start:%d-%m-%Y}&to={end:%d-%m-%Y}"
    total = 0

    for kind, path in [("bulk", f"/api/historical/bulk-deals?{span}"),
                       ("block", f"/api/historical/block-deals?{span}")]:
        payload, note = nse_api(client, path, "/report-detail/display-bulk-and-block-deals")
        if payload is None:
            print(f"    {kind} deals: {note}")
            time.sleep(0.5)
            continue

        data = payload.get("data", []) if isinstance(payload, dict) else payload
        rows = []
        for item in data or []:
            rows.append({
                "DATE": _pick(item, "BD_DT_DATE", "date", "mDate"),
                "TYPE": kind,
                "SYMBOL": _pick(item, "BD_SYMBOL", "symbol"),
                "COMPANY": _pick(item, "BD_SCRIP_NAME", "scripName"),
                "CLIENT": _pick(item, "BD_CLIENT_NAME", "clientName"),
                "SIDE": _pick(item, "BD_BUY_SELL", "buySell"),
                "QTY": _pick(item, "BD_QTY_TRD", "quantityTraded"),
                "PRICE": _pick(item, "BD_TP_WATP", "tradePrice", "watp"),
                "REMARKS": _pick(item, "BD_REMARKS", "remarks"),
            })
        df = pd.DataFrame(rows)
        n = _append_csv(DEALS_PATH, df, ["DATE", "SYMBOL", "CLIENT", "SIDE", "QTY"])
        print(f"    {kind} deals: {len(df)} fetched, {n} new")
        total += n
        time.sleep(0.5)

    return total


def collect_insider(client: httpx.Client, days_back: int = 14) -> int:
    """
    Insider trading disclosures under SEBI Regulation 7(2).

    Promoters and designated persons must report their own dealings. Promoter
    buying is one of the few genuinely informative signals in a small cap --
    and pledging shows up here too, which the fundamentals screen cannot see.
    """
    end = datetime.now()
    start = end - timedelta(days=days_back)
    path = (f"/api/corporates-pit?index=equities"
            f"&from_date={start:%d-%m-%Y}&to_date={end:%d-%m-%Y}")
    payload, note = nse_api(client, path, "/companies-listing/corporate-filings-insider-trading")
    if payload is None:
        print(f"    insider filings: {note}")
        return 0

    data = payload.get("data", []) if isinstance(payload, dict) else payload
    rows = []
    for item in data or []:
        rows.append({
            "DATE": _pick(item, "date", "acqfromDt", "intimDt"),
            "SYMBOL": _pick(item, "symbol"),
            "COMPANY": _pick(item, "company"),
            "PERSON": _pick(item, "acqName"),
            "CATEGORY": _pick(item, "personCategory", "anex"),
            "TRANSACTION": _pick(item, "tdpTransactionType"),
            "SECURITY": _pick(item, "secType"),
            "QTY": _pick(item, "secAcq"),
            "VALUE": _pick(item, "secVal"),
            "MODE": _pick(item, "acqMode"),
            "HOLDING_BEFORE": _pick(item, "befAcqSharesNo"),
            "HOLDING_AFTER": _pick(item, "afterAcqSharesNo"),
        })
    df = pd.DataFrame(rows)
    n = _append_csv(INSIDER_PATH, df, ["DATE", "SYMBOL", "PERSON", "QTY", "TRANSACTION"])
    print(f"    insider filings: {len(df)} fetched, {n} new")
    return n


def collect_corporate_actions(client: httpx.Client, days_ahead: int = 60) -> int:
    """Ex-dates for dividends, splits, bonuses and rights."""
    start = datetime.now() - timedelta(days=7)
    end = datetime.now() + timedelta(days=days_ahead)
    path = (f"/api/corporates-corporateActions?index=equities"
            f"&from_date={start:%d-%m-%Y}&to_date={end:%d-%m-%Y}")
    payload, note = nse_api(client, path, "/companies-listing/corporate-filings-actions")
    if payload is None:
        print(f"    corporate actions: {note}")
        return 0

    data = payload.get("data", []) if isinstance(payload, dict) else payload
    rows = []
    for item in data or []:
        rows.append({
            "SYMBOL": _pick(item, "symbol"),
            "COMPANY": _pick(item, "comp", "company"),
            "PURPOSE": _pick(item, "subject", "purpose"),
            "EX_DATE": _pick(item, "exDate", "ex_date"),
            "RECORD_DATE": _pick(item, "recDate", "recordDate"),
            "FACE_VALUE": _pick(item, "faceVal"),
        })
    df = pd.DataFrame(rows)
    n = _append_csv(ACTIONS_PATH, df, ["SYMBOL", "PURPOSE", "EX_DATE"])
    print(f"    corporate actions: {len(df)} fetched, {n} new")
    return n


def collect_board_meetings(client: httpx.Client, days_ahead: int = 45) -> int:
    """
    Board meeting dates — which is to say, when each company reports results.

    This matters most for options: an earnings date inside your holding period
    is exactly when implied volatility collapses.
    """
    start = datetime.now() - timedelta(days=3)
    end = datetime.now() + timedelta(days=days_ahead)
    path = (f"/api/corporate-board-meetings?index=equities"
            f"&from_date={start:%d-%m-%Y}&to_date={end:%d-%m-%Y}")
    payload, note = nse_api(client, path, "/companies-listing/corporate-filings-board-meetings")
    if payload is None:
        print(f"    board meetings: {note}")
        return 0

    data = payload.get("data", []) if isinstance(payload, dict) else payload
    rows = []
    for item in data or []:
        rows.append({
            "SYMBOL": _pick(item, "bm_symbol", "symbol"),
            "COMPANY": _pick(item, "sm_name", "company"),
            "MEETING_DATE": _pick(item, "bm_date"),
            "PURPOSE": _pick(item, "bm_purpose"),
            "DETAILS": str(_pick(item, "bm_desc"))[:300],
        })
    df = pd.DataFrame(rows)
    n = _append_csv(MEETINGS_PATH, df, ["SYMBOL", "MEETING_DATE", "PURPOSE"])
    print(f"    board meetings: {len(df)} fetched, {n} new")
    return n


FNO_DIR = Path(__file__).parent / "data" / "fno"
LOTS_PATH = Path(__file__).parent / "data" / "lot_sizes.csv"

# The F&O bhavcopy: every contract on every underlying in one file. Far better
# than the per-symbol option-chain API, which needs one request per stock and
# gets throttled long before covering the whole F&O universe.
FNO_BHAV_URL = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)


def fetch_fno_bhav(client: httpx.Client, date: datetime):
    """Every F&O contract for one trading day. Returns (df, note)."""
    url = FNO_BHAV_URL.format(yyyymmdd=date.strftime("%Y%m%d"))
    headers = {"Referer": "https://www.nseindia.com/all-reports-derivatives",
               "Accept": "application/zip,text/csv,*/*"}
    try:
        resp = client.get(url, headers=headers)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            raw = pd.read_csv(zf.open(zf.namelist()[0]), skipinitialspace=True)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:70]}"

    raw.columns = [c.strip() for c in raw.columns]
    keep = {
        "TckrSymb": "SYMBOL", "FinInstrmTp": "TYPE", "XpryDt": "EXPIRY",
        "StrkPric": "STRIKE", "OptnTp": "OPT", "ClsPric": "CLOSE",
        "PrvsClsgPric": "PREV_CLOSE", "UndrlygPric": "UNDERLYING",
        "OpnIntrst": "OI", "ChngInOpnIntrst": "CHG_OI",
        "TtlTradgVol": "VOLUME", "NewBrdLotQty": "LOT_SIZE",
    }
    have = {k: v for k, v in keep.items() if k in raw.columns}
    if "TckrSymb" not in have:
        return None, f"unexpected columns: {list(raw.columns)[:10]}"

    df = raw[list(have)].rename(columns=have)
    for c in ["STRIKE", "CLOSE", "PREV_CLOSE", "UNDERLYING", "OI", "CHG_OI",
              "VOLUME", "LOT_SIZE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["DATE"] = date.strftime("%Y-%m-%d")
    return df, "ok"


def store_lot_sizes(df: pd.DataFrame) -> None:
    """Lot sizes straight from the exchange file — no hardcoded table to rot."""
    if "LOT_SIZE" not in df.columns:
        return
    lots = (df.dropna(subset=["LOT_SIZE"])
              .groupby("SYMBOL")["LOT_SIZE"].max().reset_index())
    lots.columns = ["SYMBOL", "LOT_SIZE"]
    LOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lots.to_csv(LOTS_PATH, index=False)
    print(f"    lot sizes: {len(lots)} underlyings")


def collect_fno(client: httpx.Client, max_lookback: int = 6) -> bool:
    """
    One request gives the entire F&O market — every underlying, every strike,
    every expiry. Stored monthly like the equity bhavcopy.
    """
    FNO_DIR.mkdir(parents=True, exist_ok=True)

    for offset in range(max_lookback):
        day = datetime.now() - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        df, note = fetch_fno_bhav(client, day)
        if df is None:
            print(f"    {day:%Y-%m-%d}: {note}")
            time.sleep(0.4)
            continue

        stamp = day.strftime("%Y-%m-%d")
        opts = df[df["TYPE"].isin(["STO", "IDO"])] if "TYPE" in df.columns else df
        print(f"    {stamp}: {len(df):,} contracts "
              f"({df['SYMBOL'].nunique()} underlyings, {len(opts):,} options)")

        path = FNO_DIR / f"{stamp[:7]}.csv.gz"
        if path.exists():
            prev = pd.read_csv(path, skipinitialspace=True)
            prev = prev[prev["DATE"] != stamp]
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(path, index=False, compression="gzip")

        store_lot_sizes(df[df["DATE"] == stamp])
        return True

    print("    no F&O bhavcopy available in the lookback window")
    return False


OPTIONS_DIR = Path(__file__).parent / "data" / "options"
IV_PATH = Path(__file__).parent / "data" / "iv_history.csv"

# Underlyings to snapshot daily. Indices use a different endpoint to stocks.
CHAIN_INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
CHAIN_STOCKS = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
                "SBIN", "ITC", "LT", "BHARTIARTL", "BEL", "HAL"]


def fetch_chain(client: httpx.Client, symbol: str, is_index: bool):
    """One underlying's full option chain. Returns (rows, underlying) or (None, None)."""
    endpoint = "option-chain-indices" if is_index else "option-chain-equities"
    url = f"{NSE_BASE}/api/{endpoint}?symbol={symbol}"
    try:
        resp = client.get(url, headers={"Referer": f"{NSE_BASE}/option-chain"})
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:60]}"

    records = (payload or {}).get("records", {})
    data = records.get("data", []) or []
    underlying = records.get("underlyingValue")
    if not data:
        return None, "no data in payload"

    rows = []
    for item in data:
        strike = item.get("strikePrice")
        expiry = item.get("expiryDate")
        row = {"STRIKE": strike, "EXPIRY": expiry}
        for side, prefix in (("CE", "CE"), ("PE", "PE")):
            leg = item.get(side) or {}
            row[f"{prefix}_OI"] = leg.get("openInterest")
            row[f"{prefix}_CHG_OI"] = leg.get("changeinOpenInterest")
            row[f"{prefix}_IV"] = leg.get("impliedVolatility")
            row[f"{prefix}_LTP"] = leg.get("lastPrice")
            row[f"{prefix}_CHG"] = leg.get("change")
            row[f"{prefix}_VOL"] = leg.get("totalTradedVolume")
        rows.append(row)

    df = pd.DataFrame(rows)
    df["SYMBOL"] = symbol
    df["UNDERLYING"] = underlying
    return df, "ok"


def summarise_chain(df: pd.DataFrame, underlying: float) -> dict:
    """
    ATM implied volatility, put-call ratio and max pain for the nearest expiry.

    These are descriptions of current positioning, not forecasts. Max pain in
    particular is widely treated as a price magnet on evidence that is thin.
    """
    if df.empty or underlying is None:
        return {}

    expiries = sorted(pd.to_datetime(df["EXPIRY"], format="%d-%b-%Y",
                                     errors="coerce").dropna().unique())
    if not expiries:
        return {}
    near = pd.Timestamp(expiries[0]).strftime("%d-%b-%Y")
    chain = df[df["EXPIRY"] == near].copy()
    if chain.empty:
        return {}

    for c in ["STRIKE", "CE_OI", "PE_OI", "CE_IV", "PE_IV"]:
        chain[c] = pd.to_numeric(chain[c], errors="coerce")

    # ATM = strike closest to spot; IV = mean of the two sides where present
    atm_row = chain.iloc[(chain["STRIKE"] - underlying).abs().argsort()[:1]]
    ivs = [v for v in [atm_row["CE_IV"].iloc[0], atm_row["PE_IV"].iloc[0]]
           if pd.notna(v) and v > 0]
    atm_iv = sum(ivs) / len(ivs) if ivs else float("nan")

    ce_oi, pe_oi = chain["CE_OI"].sum(), chain["PE_OI"].sum()
    pcr = pe_oi / ce_oi if ce_oi else float("nan")

    # Max pain: the strike at which option writers pay out the least in total
    pains = []
    for s in chain["STRIKE"].dropna().unique():
        ce_pay = (chain["CE_OI"] * (s - chain["STRIKE"]).clip(lower=0)).sum()
        pe_pay = (chain["PE_OI"] * (chain["STRIKE"] - s).clip(lower=0)).sum()
        pains.append((ce_pay + pe_pay, s))
    max_pain = min(pains)[1] if pains else float("nan")

    return {
        "EXPIRY": near, "UNDERLYING": underlying, "ATM_STRIKE": float(atm_row["STRIKE"].iloc[0]),
        "ATM_IV": atm_iv, "PCR": pcr, "MAX_PAIN": float(max_pain),
        "CE_OI_TOTAL": float(ce_oi), "PE_OI_TOTAL": float(pe_oi),
    }


def collect_options(client: httpx.Client) -> int:
    """
    Snapshot every tracked chain and append the daily IV/PCR/max-pain summary.

    The IV history is the point of this: a single day's implied volatility says
    nothing, but its rank against the past year is the most useful single
    number in options, and nobody sells it to you retroactively. It only exists
    if you started collecting.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    OPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    chains, summaries, ok = [], [], 0
    targets = ([(s, True) for s in CHAIN_INDICES]
               + [(s, False) for s in CHAIN_STOCKS])

    for symbol, is_index in targets:
        df, note = fetch_chain(client, symbol, is_index)
        if df is None:
            print(f"    {symbol}: {note}")
            time.sleep(0.5)
            continue

        underlying = df["UNDERLYING"].iloc[0]
        df["DATE"] = today
        chains.append(df)

        summ = summarise_chain(df, underlying)
        if summ:
            summ.update({"DATE": today, "SYMBOL": symbol})
            summaries.append(summ)
        ok += 1
        print(f"    {symbol}: {len(df)} strikes, spot {underlying}")
        time.sleep(0.5)  # stay under NSE's rate limit

    if chains:
        snap = pd.concat(chains, ignore_index=True)
        path = OPTIONS_DIR / f"{today[:7]}.csv.gz"
        if path.exists():
            prev = pd.read_csv(path, skipinitialspace=True)
            prev = prev[prev["DATE"] != today]
            snap = pd.concat([prev, snap], ignore_index=True)
        snap.to_csv(path, index=False, compression="gzip")

    if summaries:
        new = pd.DataFrame(summaries)
        if IV_PATH.exists():
            prev = pd.read_csv(IV_PATH, skipinitialspace=True)
            combined = pd.concat([prev, new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["DATE", "SYMBOL"], keep="last")
        else:
            combined = new
        combined.sort_values(["DATE", "SYMBOL"]).to_csv(IV_PATH, index=False)
        print(f"    IV history: {len(combined)} rows on file")

    return ok


def month_path(date_str: str) -> Path:
    return HISTORY_DIR / f"{date_str[:7]}.csv.gz"


def existing_dates() -> set:
    """Dates already stored, so re-runs don't refetch."""
    dates = set()
    if not HISTORY_DIR.exists():
        return dates
    for f in HISTORY_DIR.glob("*.csv.gz"):
        try:
            dates.update(pd.read_csv(f, usecols=["DATE"], skipinitialspace=True)["DATE"].unique())
        except Exception:
            continue
    return dates


def store(df: pd.DataFrame) -> None:
    """Append a day into its month file, replacing that date if already present."""
    if df.empty:
        return
    date_str = df["DATE"].iloc[0]
    path = month_path(date_str)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    if path.exists():
        prev = pd.read_csv(path, skipinitialspace=True)
        prev = prev[prev["DATE"] != date_str]
        df = pd.concat([prev, df], ignore_index=True)

    for col in KEEP:
        if col not in df.columns:
            df[col] = float("nan")
    df = df.sort_values(["DATE", "SYMBOL"])[KEEP]
    df.to_csv(path, index=False, compression="gzip")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0,
                    help="Seed this many calendar days back from today")
    ap.add_argument("--from", dest="start", type=str, default="")
    ap.add_argument("--to", dest="end", type=str, default="")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-fetch dates already stored (needed after a schema change)")
    args = ap.parse_args()

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        days = [end - timedelta(days=i) for i in range((end - start).days + 1)]
    elif args.backfill:
        days = [datetime.now() - timedelta(days=i) for i in range(args.backfill)]
    else:
        days = [datetime.now() - timedelta(days=i) for i in range(4)]

    days = [d for d in days if d.weekday() < 5]
    if args.refresh:
        print("Refresh mode: re-fetching dates already on disk.")
        have = set()
    else:
        have = existing_dates()
    days = [d for d in days if d.strftime("%Y-%m-%d") not in have]

    if not days:
        print(f"Already up to date — {len(have)} trading days already stored.")
        print("\nCollecting FII/DII flows…")
        client = make_client()
        collect_flows(client)

        print("\nCollecting deals and disclosures…")
        collect_deals(client)
        collect_insider(client)

        print("\nCollecting shareholding patterns…")
        collect_shareholding(client)
        collect_mf_disclosure_links(client)

        print("\nCollecting corporate calendar…")
        collect_corporate_actions(client)
        collect_board_meetings(client)

        print("\nCollecting F&O bhavcopy (all underlyings)…")
        collect_fno(client)

        print("\nCollecting option chains for implied volatility…")
        collect_options(client)
        client.close()
        return 0

    if have:
        print(f"{len(have)} trading day(s) already stored.")
    print(f"Fetching {len(days)} trading day(s) not yet on disk…")
    client = make_client()
    stored = failed = 0
    consecutive_misses = 0
    oldest_stored = "—"
    statuses: dict = {}

    # Newest first. Recent files are the ones that exist and the ones you need;
    # NSE drops older security-wise files out of the archive entirely.
    for n, day in enumerate(sorted(days, reverse=True)):
        label = day.strftime("%Y-%m-%d")
        diag: list = []
        df = fetch_day(client, day, diag)
        if failed >= 3 and df.empty:
            diag = [d for d in diag if isinstance(d, int)]
        for code in diag:
            key = code if isinstance(code, int) else str(code)[:40]
            statuses[key] = statuses.get(key, 0) + 1

        if df.empty:
            print(f"  {label}: no data")
            for step in diag:
                print(f"      {step}")
            failed += 1
            consecutive_misses += 1
            # Re-prime the session periodically -- cookies expire mid-run.
            if failed % 25 == 0:
                print("    re-priming session…")
                try:
                    client.close()
                    client = make_client()
                except Exception as exc:  # noqa: BLE001
                    print(f"    re-prime failed: {exc}", file=sys.stderr)
        else:
            store(df)
            print(f"  {label}: {len(df):,} stocks stored")
            stored += 1
            consecutive_misses = 0
            oldest_stored = label

        # If the ten most recent trading days all fail, something is genuinely
        # wrong. Failures further back just mean the archive doesn't go that
        # far, which is expected and not a reason to stop.
        if n == 9 and stored == 0:
            print("\nAborting: the 10 most recent trading days all failed.")
            break

        # Stop once we're clearly past the archive's retention window.
        if stored > 0 and consecutive_misses >= 30:
            print(f"\nStopping: {consecutive_misses} consecutive days unavailable "
                  f"— past the end of NSE's archive. Collected back to "
                  f"{oldest_stored}.")
            break

        time.sleep(0.4)  # stay under NSE's ~3 req/sec limit

    # FII/DII is a single request and worth doing on every run, including runs
    # where every bhavcopy date was already on disk.
    print("\nCollecting FII/DII flows…")
    collect_flows(client)

    print("\nCollecting deals and disclosures…")
    collect_deals(client)
    collect_insider(client)

    print("\nCollecting shareholding patterns…")
    collect_shareholding(client)
    collect_mf_disclosure_links(client)

    print("\nCollecting corporate calendar…")
    collect_corporate_actions(client)
    collect_board_meetings(client)

    print("\nCollecting F&O bhavcopy (all underlyings)…")
    collect_fno(client)

    print("\nCollecting option chains for implied volatility…")
    collect_options(client)

    client.close()

    print(f"\nDone. {stored} day(s) stored, {failed} unavailable.")
    print(f"Oldest date collected: {oldest_stored}")
    print("HTTP status counts:", statuses or "none recorded")

    if stored == 0:
        codes = [k for k in statuses if isinstance(k, int)]
        # A 404 means the file does not exist for that date -- which is exactly
        # what a market holiday looks like. That is a correct outcome, not a
        # failure, so it must not turn the scheduled run red.
        only_missing = bool(codes) and all(c == 404 for c in codes)
        if only_missing:
            print(
                "Every requested date returned 404, i.e. no file exists for them. "
                "On an up-to-date store the only dates left unfetched are market "
                "holidays, so this is the expected result and not an error."
            )
            return 0

        print(
            "\nNothing was stored and not everything was a 404. 403 means the IP "
            "range is blocked. A 200 with a tiny body means NSE served a "
            "placeholder -- the sample above shows what."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
