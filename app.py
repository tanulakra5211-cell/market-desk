"""
Market Desk -- an Indian equities dashboard.

Run locally:   streamlit run app.py
Deploy free:   push to GitHub, connect at share.streamlit.io
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from sources import backlog, depth, flows, news, prices

st.set_page_config(
    page_title="Market Desk",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------- styling ---

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-feature-settings: "tnum" 1, "lnum" 1; }
      .stApp { background: #0f1216; }
      h1, h2, h3 { letter-spacing: -0.02em; font-weight: 650; }

      .pulse-strip {
        display: flex; gap: 0; flex-wrap: wrap;
        border-top: 1px solid #232a33; border-bottom: 1px solid #232a33;
        margin-bottom: 1.4rem;
      }
      .pulse-cell {
        flex: 1 1 140px; padding: 0.7rem 1rem;
        border-right: 1px solid #1a2028;
      }
      .pulse-cell:last-child { border-right: none; }
      .pulse-label {
        font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.09em;
        color: #6b7684; margin-bottom: 0.25rem;
      }
      .pulse-value {
        font-family: ui-monospace, "SF Mono", "JetBrains Mono", monospace;
        font-size: 1.05rem; font-weight: 600; color: #e6eaef;
      }
      .pulse-delta {
        font-family: ui-monospace, monospace; font-size: 0.78rem; margin-left: 0.4rem;
      }
      .up { color: #2fbf71; } .down { color: #e5484d; } .flat { color: #6b7684; }

      .stale {
        font-size: 0.72rem; color: #8b6d3f; background: #1f1a10;
        border-left: 2px solid #b8860b; padding: 0.5rem 0.8rem; margin: 0.5rem 0;
      }
      .headline-row {
        padding: 0.55rem 0; border-bottom: 1px solid #1a2028;
      }
      .headline-meta {
        font-size: 0.7rem; color: #6b7684; text-transform: uppercase;
        letter-spacing: 0.06em;
      }
      div[data-testid="stMetricValue"] { font-family: ui-monospace, monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------ data loading ---

@st.cache_data(ttl=60 * 5)
def load_global():
    return prices.fetch_global_markets()


@st.cache_data(ttl=60 * 5)
def load_quotes(tickers: tuple):
    return prices.fetch_quotes(list(tickers))


@st.cache_data(ttl=60 * 60 * 6)
def load_ratios(tickers: tuple):
    return prices.fetch_ratios(list(tickers))


@st.cache_data(ttl=60 * 15)
def load_news():
    return news.fetch_market_news()


@st.cache_data(ttl=60 * 30)
def load_announcements(days: int, scrip: str):
    return news.fetch_bse_announcements(days_back=days, scrip_code=scrip)


@st.cache_data(ttl=60 * 30)
def load_flows():
    return flows.fetch_fii_dii_cash()


@st.cache_data(ttl=60 * 60)
def load_fno_oi():
    return flows.fetch_fno_participant_oi()


def load_watchlist() -> pd.DataFrame:
    path = DATA_DIR / "watchlist.csv"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "company", "bse_scrip", "sector"])
    return pd.read_csv(path, dtype={"bse_scrip": str})


# ---------------------------------------------------------------- helpers ---

def fmt(value, decimals: int = 2, dash: str = "—") -> str:
    if value is None or value != value:
        return dash
    return f"{value:,.{decimals}f}"


def delta_class(value) -> str:
    if value is None or value != value:
        return "flat"
    return "up" if value > 0 else "down" if value < 0 else "flat"


def pulse_cell(label: str, value: str, delta: str = "", cls: str = "flat") -> str:
    delta_html = f'<span class="pulse-delta {cls}">{delta}</span>' if delta else ""
    return (
        f'<div class="pulse-cell"><div class="pulse-label">{label}</div>'
        f'<div class="pulse-value">{value}{delta_html}</div></div>'
    )


def colour_frame(df: pd.DataFrame, pct_cols: list[str]):
    """Red/green tint on percentage columns."""
    existing = [c for c in pct_cols if c in df.columns]

    def shade(v):
        if v is None or v != v:
            return "color: #6b7684"
        return "color: #2fbf71" if v > 0 else "color: #e5484d" if v < 0 else ""

    return df.style.map(shade, subset=existing).format(precision=2, na_rep="—")


# ---------------------------------------------------------------- sidebar ---

watchlist = load_watchlist()

with st.sidebar:
    st.markdown("### Market Desk")
    st.caption("Indian equities, one screen")

    selected = st.multiselect(
        "Watchlist",
        options=watchlist["ticker"].tolist(),
        default=watchlist["ticker"].tolist(),
        format_func=lambda t: watchlist.set_index("ticker").loc[t, "company"]
        if t in watchlist["ticker"].values else t,
    )

    st.divider()
    news_days = st.slider("Announcement lookback (days)", 1, 15, 3)
    only_watchlist_news = st.checkbox("Filter news to watchlist", value=False)

    st.divider()
    if st.button("Refresh all data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Prices via Yahoo. Flows via NSE. News via publisher RSS and BSE filings. "
        "All figures are delayed — this is a research tool, not an execution tool."
    )

wl = watchlist[watchlist["ticker"].isin(selected)] if selected else watchlist
tickers = tuple(wl["ticker"].tolist())


# ------------------------------------------------------------ pulse strip ---

global_df = load_global()
flows_df = load_flows()
flow_summary = flows.summarise_flows(flows_df)

cells = []
for name in ["Nifty 50", "Sensex", "Nifty Bank", "India VIX", "USD/INR"]:
    row = global_df[global_df["Market"] == name]
    if row.empty:
        continue
    last = row.iloc[0]["Last"]
    chg = row.iloc[0]["Change %"]
    cells.append(
        pulse_cell(name, fmt(last), f"{chg:+.2f}%", delta_class(chg))
    )

for label in ["FII", "DII"]:
    if label in flow_summary:
        net = flow_summary[label]["net"]
        cells.append(
            pulse_cell(f"{label} net (cash)", f"{net:+,.0f} Cr", "", delta_class(net))
        )
    else:
        cells.append(pulse_cell(f"{label} net (cash)", "—", "", "flat"))

st.markdown(f'<div class="pulse-strip">{"".join(cells)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------- tabs ---

tab_market, tab_news, tab_flows, tab_ratios, tab_book, tab_depth = st.tabs(
    ["Markets", "News", "FII / DII", "Ratios", "Order Book", "Depth"]
)


with tab_market:
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Your watchlist")
        quotes = load_quotes(tickers)
        if quotes.empty:
            st.info("No price data returned. Try Refresh, or check your tickers.")
        else:
            merged = quotes.merge(
                wl[["ticker", "company", "sector"]],
                left_on="Ticker", right_on="ticker", how="left",
            ).drop(columns=["ticker"])
            merged = merged[[
                "company", "sector", "Last", "Day %", "1M %", "6M %", "1Y %",
                "Off 52W High %",
            ]].rename(columns={"company": "Company", "sector": "Sector"})
            st.dataframe(
                colour_frame(merged, ["Day %", "1M %", "6M %", "1Y %", "Off 52W High %"]),
                use_container_width=True, height=520, hide_index=True,
            )

    with right:
        st.markdown("#### Global markets")
        gd = global_df.drop(columns=["Symbol"])
        st.dataframe(
            colour_frame(gd, ["Change %"]),
            use_container_width=True, height=520, hide_index=True,
        )


with tab_news:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("#### Market headlines")
        headlines = load_news()

        if only_watchlist_news and not wl.empty:
            keywords = wl["company"].str.split().str[0].tolist()
            headlines = news.filter_news_for(headlines, keywords)

        if headlines.empty:
            st.info("No headlines available right now.")
        else:
            for _, row in headlines.head(40).iterrows():
                stamp = row["Time"].strftime("%d %b, %H:%M") if pd.notna(row["Time"]) else "—"
                st.markdown(
                    f'<div class="headline-row">'
                    f'<div class="headline-meta">{row["Source"]} · {stamp}</div>'
                    f'<a href="{row["Link"]}" target="_blank">{row["Headline"]}</a>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with col_b:
        st.markdown("#### Corporate filings")
        st.caption("Straight from BSE — results, board meetings, order wins.")

        anns = load_announcements(news_days, "")
        if anns.empty:
            st.info("BSE announcements unavailable. Retry with Refresh.")
        else:
            if only_watchlist_news and not wl.empty:
                anns = anns[anns["Scrip"].astype(str).isin(wl["bse_scrip"].astype(str))]

            order_wins = news.find_order_wins(anns)
            if not order_wins.empty:
                st.markdown("**Possible order wins**")
                for _, row in order_wins.head(10).iterrows():
                    link = f' [PDF]({row["PDF"]})' if row["PDF"] else ""
                    st.markdown(f'- **{row["Company"]}** — {row["Headline"]}{link}')
                st.divider()

            st.markdown("**All filings**")
            st.dataframe(
                anns[["Time", "Company", "Category", "Headline"]].head(60),
                use_container_width=True, height=380, hide_index=True,
            )


with tab_flows:
    st.markdown("#### Institutional flows")

    if flows_df.empty:
        st.markdown(
            '<div class="stale">NSE did not return flow data. This endpoint is '
            'scraped and refuses requests intermittently, especially from cloud '
            'hosts during market hours. Try again after 7pm IST when the daily '
            'figures publish.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"Cash market, Rs crore · session dated {flows_df.iloc[0]['Date']}")
        st.dataframe(
            colour_frame(flows_df, ["Net (Cr)"]),
            use_container_width=True, hide_index=True,
        )

        chart_df = flows_df.set_index("Participant")[["Buy (Cr)", "Sell (Cr)"]]
        st.bar_chart(chart_df, height=280)

    st.divider()
    st.markdown("#### F&O participant positioning")
    st.caption(
        "Whether FIIs are net long or short index futures — the positioning "
        "signal cash flows alone don't give you."
    )

    oi = load_fno_oi()
    if oi.empty:
        st.info(
            "Participant-wise OI not published yet for today. NSE releases it "
            "after market close on trading days."
        )
    else:
        st.dataframe(oi, use_container_width=True, hide_index=True)


with tab_ratios:
    st.markdown("#### Fundamentals screener")
    st.caption(
        "Cached for six hours — the underlying source is rate-limited, so "
        "hammering Refresh will get you throttled."
    )

    with st.spinner("Pulling fundamentals…"):
        ratios = load_ratios(tickers)

    if ratios.empty:
        st.info("No fundamental data returned.")
    else:
        c1, c2, c3 = st.columns(3)
        max_pe = c1.number_input("Max P/E", value=0.0, help="0 disables the filter")
        min_roe = c2.number_input("Min ROE %", value=0.0)
        max_de = c3.number_input("Max D/E", value=0.0, help="0 disables the filter")

        view = ratios.copy()
        if max_pe > 0:
            view = view[view["P/E"].fillna(1e9) <= max_pe]
        if min_roe > 0:
            view = view[view["ROE %"].fillna(-1e9) >= min_roe]
        if max_de > 0:
            view = view[view["D/E"].fillna(1e9) <= max_de]

        st.dataframe(
            view.style.format(precision=2, na_rep="—"),
            use_container_width=True, height=520, hide_index=True,
        )
        st.caption(f"{len(view)} of {len(ratios)} companies pass the filters.")


with tab_book:
    st.markdown("#### Order backlog")
    st.markdown(
        "Company order books are disclosed in quarterly investor presentations "
        "and nowhere else — no API carries them. Log the number here after each "
        "result and this tab does the analysis. The News tab flags order-win "
        "filings so you know when there's something to log."
    )

    raw_backlog = backlog.load_backlog()
    analysis = backlog.analyse_backlog(raw_backlog)

    if analysis.empty:
        st.info("No backlog entries yet. Add one below.")
    else:
        st.dataframe(
            analysis.style.format(precision=2, na_rep="—"),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Book-to-bill above ~3x means roughly three years of revenue already "
            "contracted. Sample rows ship with the repo — replace them with real "
            "disclosed figures."
        )

    st.divider()
    st.markdown("**Edit entries**")
    st.caption(
        "Add or change rows directly, like a spreadsheet. Then download the file "
        "and replace data/order_backlog.csv in your repo — hosted apps reset their "
        "local disk on every rebuild, so the download is what makes it stick."
    )

    editable = raw_backlog.copy()
    if not editable.empty:
        editable["as_of"] = editable["as_of"].dt.strftime("%Y-%m-%d")

    edited = st.data_editor(
        editable,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "ticker": st.column_config.SelectboxColumn(
                "Ticker", options=watchlist["ticker"].tolist(), required=True
            ),
            "company": st.column_config.TextColumn("Company"),
            "as_of": st.column_config.TextColumn("As of", help="Quarter end, YYYY-MM-DD"),
            "order_book_cr": st.column_config.NumberColumn(
                "Order book (Rs Cr)", min_value=0.0, step=100.0, format="%.0f"
            ),
            "notes": st.column_config.TextColumn("Source / notes", width="large"),
        },
        key="backlog_editor",
    )

    save_col, dl_col = st.columns([1, 1])

    if save_col.button("Apply to this session", use_container_width=True):
        edited.to_csv(backlog.CSV_PATH, index=False)
        st.cache_data.clear()
        st.rerun()

    dl_col.download_button(
        "Download order_backlog.csv",
        data=edited.to_csv(index=False).encode("utf-8"),
        file_name="order_backlog.csv",
        mime="text/csv",
        use_container_width=True,
        help="Upload this to your repo's data/ folder to make changes permanent.",
    )


with tab_depth:
    st.markdown("#### Live market depth")

    provider = depth.build_provider(dict(st.secrets)) if hasattr(st, "secrets") else None

    if provider is None or not provider.configured:
        st.markdown(
            "Depth needs a broker connection — there is no free public source "
            "for the bid/ask book, and NSE doesn't expose one."
        )
        st.markdown(
            """
            **Free broker APIs that provide depth:**

            | Broker | Package | Notes |
            |---|---|---|
            | Angel One SmartAPI | `smartapi-python` | Free, generous rate limits |
            | DhanHQ | `dhanhq` | Free, modern SDK, TradingView integration |
            | Fyers | `fyers-apiv3` | Free, deep historical data |
            | Shoonya (Finvasia) | `NorenRestApiPy` | Free, zero-brokerage model |

            Each needs an account with that broker. Once you have credentials,
            add them to `.streamlit/secrets.toml` and this tab activates:

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
                m3.metric(
                    "Imbalance", f"{book.imbalance:+.1%}",
                    help="Positive means more size resting on the bid.",
                )

                bid_col, ask_col = st.columns(2)
                bid_col.markdown("**Bids**")
                bid_col.dataframe(
                    pd.DataFrame([vars(b) for b in book.bids]),
                    use_container_width=True, hide_index=True,
                )
                ask_col.markdown("**Asks**")
                ask_col.dataframe(
                    pd.DataFrame([vars(a) for a in book.asks]),
                    use_container_width=True, hide_index=True,
                )
