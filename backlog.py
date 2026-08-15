"""
Company order backlog (the order book EPC/infra investors care about).

This number is not available from any API, free or paid. Companies disclose
it in quarterly investor presentations and earnings calls, in prose, in a PDF.
There is no structured feed for it anywhere.

So the honest design is: you maintain data/order_backlog.csv (one line per
company per quarter, takes two minutes after results), and this module does
the analysis on top -- book-to-bill, revenue visibility, QoQ growth. The news
module flags order-win announcements so you know when to update it.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "order_backlog.csv"


def load_backlog() -> pd.DataFrame:
    """Read the manually maintained backlog file."""
    if not CSV_PATH.exists():
        return pd.DataFrame(
            columns=["ticker", "company", "as_of", "order_book_cr", "notes"]
        )

    df = pd.read_csv(CSV_PATH)
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    return df.sort_values(["ticker", "as_of"])


def _ttm_revenue_cr(ticker: str) -> float:
    """Trailing twelve month revenue in Rs crore, for book-to-bill."""
    try:
        info = yf.Ticker(ticker).info or {}
        rev = info.get("totalRevenue")
        return float(rev) / 1e7 if rev else float("nan")
    except Exception:
        return float("nan")


def analyse_backlog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Latest backlog per company, with book-to-bill and QoQ change.

    Book-to-bill = order book / TTM revenue. Above ~3x means roughly three
    years of revenue already contracted; that's the visibility metric.
    """
    if df.empty:
        return df

    rows = []
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("as_of")
        latest = group.iloc[-1]
        prev = group.iloc[-2] if len(group) > 1 else None

        revenue = _ttm_revenue_cr(ticker)
        book = float(latest["order_book_cr"])

        rows.append({
            "Ticker": ticker,
            "Company": latest.get("company", ticker),
            "As Of": latest["as_of"].date() if pd.notna(latest["as_of"]) else None,
            "Order Book (Cr)": book,
            "TTM Revenue (Cr)": revenue,
            "Book-to-Bill": book / revenue if revenue and revenue == revenue else float("nan"),
            "Revenue Visibility (yrs)": book / revenue if revenue and revenue == revenue else float("nan"),
            "QoQ Change %": (
                ((book - float(prev["order_book_cr"])) / float(prev["order_book_cr"])) * 100
                if prev is not None and float(prev["order_book_cr"]) else float("nan")
            ),
            "Notes": latest.get("notes", ""),
        })

    out = pd.DataFrame(rows)
    return out.sort_values("Book-to-Bill", ascending=False, na_position="last")


def append_entry(ticker: str, company: str, as_of: str, order_book_cr: float, notes: str = "") -> None:
    """Add a new quarterly reading. Called from the dashboard's entry form."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([{
        "ticker": ticker, "company": company, "as_of": as_of,
        "order_book_cr": order_book_cr, "notes": notes,
    }])

    if CSV_PATH.exists():
        existing = pd.read_csv(CSV_PATH)
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    combined.to_csv(CSV_PATH, index=False)
