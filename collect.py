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

    client.close()

    print(f"\nDone. {stored} day(s) stored, {failed} unavailable.")
    print(f"Oldest date collected: {oldest_stored}")
    print("HTTP status counts:", statuses or "none recorded")

    if stored == 0:
        print(
            "\nNothing was stored. 403 means the IP range is blocked. 404 means "
            "the file does not exist for those dates. A 200 with a tiny body "
            "means NSE served a placeholder -- the sample above shows what."
        )
    # Exit 0 unless nothing at all was stored, so one bad day doesn't turn the
    # whole schedule red.
    return 1 if stored == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
