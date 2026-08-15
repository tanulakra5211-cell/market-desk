"""
Telegram alerts.

A dashboard only helps when you remember to open it. This runs after the
collector and pushes what changed — new setups, promoter buying, results
dates approaching, unusual delivery.

Setup:
  1. Message @BotFather on Telegram, send /newbot, copy the token.
  2. Message your new bot once, then open
     https://api.telegram.org/bot<TOKEN>/getUpdates and copy your chat id.
  3. In GitHub: Settings -> Secrets and variables -> Actions -> New secret.
     Add TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.

If the secrets are absent the script exits quietly, so the workflow stays
green whether or not you use it.

Deliberately conservative: only high-signal events, and a cap on message
length. An alert feed you learn to ignore is worse than none.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

DATA = Path(__file__).parent / "data"
TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# Only these get alerted on. Edit freely.
WATCHLIST = {
    "RELIANCE", "HDFCBANK", "TCS", "INFY", "ICICIBANK", "LT", "ITC", "SBIN",
    "BHARTIARTL", "RVNL", "IRCON", "BEL", "HAL", "NBCC", "PIIND",
}


def send(text: str) -> bool:
    if not TOKEN or not CHAT_ID:
        print("No Telegram credentials set — skipping (this is not an error).")
        return False
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=20.0)
        if resp.status_code != 200:
            print(f"Telegram returned {resp.status_code}: {resp.text[:200]}",
                  file=sys.stderr)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Telegram send failed: {exc}", file=sys.stderr)
        return False


def read(name: str, date_cols=()) -> pd.DataFrame:
    path = DATA / name
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


def insider_alerts() -> list:
    """
    Promoter and insider buying on watchlist names in the last two days.

    Acquisitions only. Disposals are common and rarely informative on their
    own — people sell for reasons that have nothing to do with the company.
    """
    df = read("insider.csv", ["DATE"])
    if df.empty or "SYMBOL" not in df.columns:
        return []

    cutoff = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=2)
    df = df[(df["DATE"] >= cutoff)
            & df["SYMBOL"].astype(str).str.upper().isin(WATCHLIST)]
    if df.empty:
        return []

    out = []
    for _, r in df.iterrows():
        txn = str(r.get("TRANSACTION", "")).lower()
        if "acqu" not in txn and "buy" not in txn:
            continue
        mode = str(r.get("MODE", ""))
        out.append(
            f"🟢 <b>{r['SYMBOL']}</b> — insider acquisition\n"
            f"    {str(r.get('PERSON',''))[:40]} · {r.get('QTY','')} shares · {mode}")
    return out[:8]


def deal_alerts() -> list:
    """Bulk and block deals on watchlist names."""
    df = read("deals.csv", ["DATE"])
    if df.empty or "SYMBOL" not in df.columns:
        return []
    cutoff = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=2)
    df = df[(df["DATE"] >= cutoff)
            & df["SYMBOL"].astype(str).str.upper().isin(WATCHLIST)]
    return [f"📊 <b>{r['SYMBOL']}</b> — {r.get('SIDE','')} {r.get('QTY','')} @ "
            f"{r.get('PRICE','')}\n    {str(r.get('CLIENT',''))[:45]}"
            for _, r in df.head(8).iterrows()]


def results_alerts() -> list:
    """Watchlist results landing in the next five days."""
    df = read("board_meetings.csv", ["MEETING_DATE"])
    if df.empty or "SYMBOL" not in df.columns:
        return []
    today = pd.Timestamp(datetime.now().date())
    df = df[(df["MEETING_DATE"] >= today)
            & (df["MEETING_DATE"] <= today + pd.Timedelta(days=5))
            & df["SYMBOL"].astype(str).str.upper().isin(WATCHLIST)]
    return [f"📅 <b>{r['SYMBOL']}</b> — {r['MEETING_DATE']:%d %b} · "
            f"{str(r.get('PURPOSE',''))[:60]}" for _, r in df.head(8).iterrows()]


def delivery_alerts() -> list:
    """
    Watchlist names where delivery percentage jumped well above its own norm.

    Delivery rising sharply means a larger share of volume is being taken as
    delivery rather than squared off — the closest free proxy for someone
    building a position rather than trading one.
    """
    hist_dir = DATA / "history"
    if not hist_dir.exists():
        return []
    files = sorted(hist_dir.glob("*.csv.gz"))[-2:]
    if not files:
        return []
    try:
        hist = pd.concat([pd.read_csv(f, skipinitialspace=True) for f in files],
                         ignore_index=True)
    except Exception:
        return []
    if "DELIV_PER" not in hist.columns:
        return []

    hist["DATE"] = pd.to_datetime(hist["DATE"], errors="coerce")
    hist = hist[hist["SYMBOL"].astype(str).str.upper().isin(WATCHLIST)]
    if hist.empty:
        return []

    out = []
    for sym, g in hist.groupby("SYMBOL"):
        g = g.sort_values("DATE")
        d = g["DELIV_PER"].dropna()
        if len(d) < 20:
            continue
        today_d, avg = float(d.iloc[-1]), float(d.iloc[-21:-1].mean())
        if avg > 0 and today_d > avg * 1.5 and today_d > 50:
            out.append(f"📦 <b>{sym}</b> — delivery {today_d:.0f}% "
                       f"vs {avg:.0f}% average")
    return out[:6]


def main() -> int:
    if not TOKEN or not CHAT_ID:
        print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set — nothing to do.")
        return 0

    sections = [
        ("Insider activity", insider_alerts()),
        ("Bulk & block deals", deal_alerts()),
        ("Results coming up", results_alerts()),
        ("Unusual delivery", delivery_alerts()),
    ]
    sections = [(title, items) for title, items in sections if items]

    if not sections:
        print("Nothing worth alerting on today.")
        return 0

    parts = [f"<b>Market Desk — {datetime.now():%d %b %Y}</b>"]
    for title, items in sections:
        parts.append(f"\n<b>{title}</b>")
        parts.extend(items)
    message = "\n".join(parts)[:3800]      # Telegram caps at 4096

    ok = send(message)
    print(f"Alert {'sent' if ok else 'not sent'} — "
          f"{sum(len(i) for _, i in sections)} items across "
          f"{len(sections)} sections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
