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

KEEP = ["DATE", "SYMBOL", "CLOSE", "PREV_CLOSE", "VOLUME",
        "TURNOVER_LACS", "DELIV_QTY", "DELIV_PER"]


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
        if resp.status_code == 200 and not resp.content[:20].lstrip().lower().startswith(b"<"):
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].astype(str).str.strip()
            df = df[df["SERIES"].str.upper() == "EQ"] if "SERIES" in df.columns else df
            out = pd.DataFrame({
                "DATE": date.strftime("%Y-%m-%d"),
                "SYMBOL": df["SYMBOL"],
                "CLOSE": pd.to_numeric(df.get("CLOSE_PRICE"), errors="coerce"),
                "PREV_CLOSE": pd.to_numeric(df.get("PREV_CLOSE"), errors="coerce"),
                "VOLUME": pd.to_numeric(df.get("TTL_TRD_QNTY"), errors="coerce"),
                "TURNOVER_LACS": pd.to_numeric(df.get("TURNOVER_LACS"), errors="coerce"),
                "DELIV_QTY": pd.to_numeric(df.get("DELIV_QTY"), errors="coerce"),
                "DELIV_PER": pd.to_numeric(df.get("DELIV_PER"), errors="coerce"),
            })
            return out.dropna(subset=["CLOSE"])
    except Exception as exc:  # noqa: BLE001
        print(f"    sec file failed: {exc}", file=sys.stderr)

    # Fallback: UDiFF zip. No delivery data, but better than a gap.
    url = UDIFF_URL.format(yyyymmdd=date.strftime("%Y%m%d"))
    try:
        resp = client.get(url, headers=archive_headers)
        diag.append(resp.status_code)
        if resp.status_code != 200:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            raw = pd.read_csv(zf.open(zf.namelist()[0]))
        raw.columns = [c.strip() for c in raw.columns]
        raw = raw[raw.get("FinInstrmTp", "STK") == "STK"]
        if "SctySrs" in raw.columns:
            raw = raw[raw["SctySrs"].astype(str).str.strip().str.upper() == "EQ"]
        out = pd.DataFrame({
            "DATE": date.strftime("%Y-%m-%d"),
            "SYMBOL": raw["TckrSymb"],
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


def month_path(date_str: str) -> Path:
    return HISTORY_DIR / f"{date_str[:7]}.csv.gz"


def existing_dates() -> set:
    """Dates already stored, so re-runs don't refetch."""
    dates = set()
    if not HISTORY_DIR.exists():
        return dates
    for f in HISTORY_DIR.glob("*.csv.gz"):
        try:
            dates.update(pd.read_csv(f, usecols=["DATE"])["DATE"].unique())
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
        prev = pd.read_csv(path)
        prev = prev[prev["DATE"] != date_str]
        df = pd.concat([prev, df], ignore_index=True)

    df = df.sort_values(["DATE", "SYMBOL"])[KEEP]
    df.to_csv(path, index=False, compression="gzip")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0,
                    help="Seed this many calendar days back from today")
    ap.add_argument("--from", dest="start", type=str, default="")
    ap.add_argument("--to", dest="end", type=str, default="")
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
    have = existing_dates()
    days = [d for d in days if d.strftime("%Y-%m-%d") not in have]

    if not days:
        print("Already up to date, nothing to fetch.")
        return 0

    print(f"Fetching {len(days)} trading day(s)…")
    client = make_client()
    stored = failed = 0
    statuses: dict = {}

    for n, day in enumerate(sorted(days)):
        label = day.strftime("%Y-%m-%d")
        diag: list = []
        df = fetch_day(client, day, diag)
        for code in diag:
            statuses[code] = statuses.get(code, 0) + 1

        if df.empty:
            print(f"  {label}: no data (HTTP {diag or 'error'})")
            failed += 1
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

        # Bail out early if NSE is clearly refusing everything, rather than
        # spending ten minutes proving it.
        if n >= 14 and stored == 0:
            print("\nAborting: 15 consecutive days returned nothing.")
            break

        time.sleep(0.4)  # stay under NSE's ~3 req/sec limit

    client.close()

    print(f"\nDone. {stored} day(s) stored, {failed} unavailable.")
    print("HTTP status counts:", dict(sorted(statuses.items())) or "none recorded")

    if stored == 0:
        print(
            "\nNothing was stored. If the statuses above are mostly 403, NSE is "
            "blocking this server's IP range and the collector needs to run "
            "somewhere else -- see README. If they are 404, the URL pattern is "
            "wrong. If no statuses appear at all, the initial handshake failed."
        )
    # Exit 0 unless nothing at all was stored, so one bad day doesn't turn the
    # whole schedule red.
    return 1 if stored == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
