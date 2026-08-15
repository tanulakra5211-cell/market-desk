"""
FII / DII activity.

Primary source is NSE's own fiidiiTradeReact endpoint (cash market, updated
after close ~7pm IST). Derivatives positioning comes from the F&O participant
-wise OI report. Both are scraped, so both have a real failure rate -- the
dashboard shows staleness rather than pretending.
"""

from datetime import datetime

import pandas as pd

from .nse_session import get_session


def fetch_fii_dii_cash() -> pd.DataFrame:
    """
    Latest session's FII and DII cash-market buy/sell/net, in Rs crore.

    Returns an empty frame if NSE refuses -- the caller should show the
    last cached value rather than a hard error.
    """
    data = get_session().get_json("/api/fiidiiTradeReact")
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
                "Buy (Cr)": buy,
                "Sell (Cr)": sell,
                "Net (Cr)": buy - sell,
            })
        except (TypeError, ValueError):
            continue

    return pd.DataFrame(rows)


def fetch_fno_participant_oi() -> pd.DataFrame:
    """
    Participant-wise open interest in F&O -- shows whether FIIs are net
    long or short index futures, which is the positioning signal the cash
    numbers alone don't give you.

    NSE publishes this as a daily CSV named by date.
    """
    date_str = datetime.now().strftime("%d%m%Y")
    url = (
        "https://nsearchives.nseindia.com/content/nsccl/"
        f"fao_participant_oi_{date_str}.csv"
    )

    try:
        # Header row is the second line; the first is a date banner.
        df = pd.read_csv(url, skiprows=1)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def summarise_flows(cash: pd.DataFrame) -> dict:
    """Condense the cash table into headline numbers for the top strip."""
    if cash.empty:
        return {}

    out = {}
    for _, row in cash.iterrows():
        label = "FII" if "FII" in row["Participant"].upper() or "FPI" in row["Participant"].upper() else "DII"
        out[label] = {
            "buy": row["Buy (Cr)"],
            "sell": row["Sell (Cr)"],
            "net": row["Net (Cr)"],
            "date": row["Date"],
        }
    return out
