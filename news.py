"""
News: broad market RSS plus company-specific corporate announcements.

RSS is the reliable half -- publishers want you to read it, so no blocking.
BSE's announcements endpoint is the useful half for stock-specific filings
(orders won, results, board meetings) and is more tolerant than NSE's.
"""

from datetime import datetime, timedelta, timezone

import feedparser
import httpx
import pandas as pd

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
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bseindia.com/",
    "Origin": "https://www.bseindia.com",
    "Accept": "application/json, text/plain, */*",
}


def fetch_market_news(limit_per_feed: int = 15) -> pd.DataFrame:
    """Merge all RSS feeds into one time-sorted table."""
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


def filter_news_for(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """Keep only headlines mentioning any of the given company keywords."""
    if df.empty or not keywords:
        return df
    pattern = "|".join(k.strip() for k in keywords if k.strip())
    mask = df["Headline"].str.contains(pattern, case=False, na=False) | df[
        "Summary"
    ].str.contains(pattern, case=False, na=False)
    return df[mask].reset_index(drop=True)


def fetch_bse_announcements(days_back: int = 3, scrip_code: str = "") -> pd.DataFrame:
    """
    Corporate announcements from BSE.

    Pass a scrip_code (e.g. '500325' for Reliance) to narrow to one company,
    or leave blank for the full exchange-wide feed. This is where order-win
    announcements and results actually land first.
    """
    today = datetime.now()
    params = {
        "pageno": 1,
        "strCat": "-1",
        "strPrevDate": (today - timedelta(days=days_back)).strftime("%Y%m%d"),
        "strScrip": scrip_code,
        "strSearch": "P",
        "strToDate": today.strftime("%Y%m%d"),
        "strType": "C",
        "subcategory": "-1",
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
    """
    Surface announcements that look like order wins.

    This is the automated half of order-backlog tracking: it flags the
    filing, you read it and update data/order_backlog.csv with the number.
    """
    if announcements.empty:
        return announcements

    terms = [
        "order", "contract", "LOA", "letter of award", "work order",
        "bagged", "awarded", "L1", "tender", "project win",
    ]
    pattern = "|".join(terms)
    mask = announcements["Headline"].str.contains(pattern, case=False, na=False)
    return announcements[mask].reset_index(drop=True)
