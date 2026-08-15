"""
Live market depth (the bid/ask order book).

There is no free public source for this. NSE does not expose depth over any
open API, and the depth widget on their site is not scrapeable at any useful
rate. It has to come from a broker.

Free broker APIs that provide depth (all require an account with them):
  - Angel One SmartAPI   pip install smartapi-python
  - DhanHQ               pip install dhanhq
  - Fyers                pip install fyers-apiv3
  - Shoonya (Finvasia)

This module defines one interface and ships an Angel One implementation.
Fill in credentials in .streamlit/secrets.toml to switch it on; until then
the dashboard shows a "not configured" panel instead of failing.

Note: from 1 April 2026, SEBI/exchange rules require registered static IPs
for *order placement* via broker APIs. Market-data reads are unaffected, but
if you later add order placement, check your broker's current requirements.
"""

from dataclasses import dataclass, field


@dataclass
class DepthLevel:
    price: float
    quantity: int
    orders: int


@dataclass
class MarketDepth:
    symbol: str
    ltp: float = 0.0
    bids: list[DepthLevel] = field(default_factory=list)
    asks: list[DepthLevel] = field(default_factory=list)

    @property
    def total_bid_qty(self) -> int:
        return sum(b.quantity for b in self.bids)

    @property
    def total_ask_qty(self) -> int:
        return sum(a.quantity for a in self.asks)

    @property
    def imbalance(self) -> float:
        """>0 means buy-side pressure, <0 sell-side. Range roughly -1 to 1."""
        total = self.total_bid_qty + self.total_ask_qty
        if not total:
            return 0.0
        return (self.total_bid_qty - self.total_ask_qty) / total

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0].price - self.bids[0].price


class DepthProvider:
    """Base interface. Implement fetch() for any broker."""

    name = "none"
    configured = False

    def fetch(self, symbol_token: str, exchange: str = "NSE") -> MarketDepth | None:
        raise NotImplementedError


class AngelOneDepth(DepthProvider):
    """
    Angel One SmartAPI depth via getMarketData(mode='FULL').

    Needs: api_key, client_id, password (MPIN), totp_secret.
    Symbol tokens come from Angel's instrument master:
      https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json
    """

    name = "Angel One SmartAPI"

    def __init__(self, api_key: str, client_id: str, password: str, totp_secret: str):
        self.configured = False
        self._client = None
        try:
            import pyotp
            from SmartApi import SmartConnect

            self._client = SmartConnect(api_key=api_key)
            self._client.generateSession(
                client_id, password, pyotp.TOTP(totp_secret).now()
            )
            self.configured = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)

    def fetch(self, symbol_token: str, exchange: str = "NSE") -> MarketDepth | None:
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
            depth = row.get("depth", {})

            def levels(side: str) -> list[DepthLevel]:
                return [
                    DepthLevel(
                        price=float(lvl.get("price", 0)),
                        quantity=int(lvl.get("quantity", 0)),
                        orders=int(lvl.get("orders", 0)),
                    )
                    for lvl in depth.get(side, [])
                ]

            return MarketDepth(
                symbol=row.get("tradingSymbol", symbol_token),
                ltp=float(row.get("ltp", 0)),
                bids=levels("buy"),
                asks=levels("sell"),
            )
        except Exception:
            return None


class DhanDepth(DepthProvider):
    """DhanHQ depth. Needs client_id and access_token from web.dhan.co."""

    name = "DhanHQ"

    def __init__(self, client_id: str, access_token: str):
        self.configured = False
        try:
            from dhanhq import dhanhq

            self._client = dhanhq(client_id, access_token)
            self.configured = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)

    def fetch(self, symbol_token: str, exchange: str = "NSE_EQ") -> MarketDepth | None:
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
                symbol=symbol_token,
                ltp=float(row.get("last_price", 0)),
                bids=[
                    DepthLevel(float(x["price"]), int(x["quantity"]), int(x.get("orders", 0)))
                    for x in d.get("buy", [])
                ],
                asks=[
                    DepthLevel(float(x["price"]), int(x["quantity"]), int(x.get("orders", 0)))
                    for x in d.get("sell", [])
                ],
            )
        except Exception:
            return None


def build_provider(secrets: dict) -> DepthProvider | None:
    """
    Pick a provider from whatever credentials exist in secrets.
    Returns None if nothing is configured -- caller shows the setup panel.
    """
    angel = secrets.get("angelone", {})
    if angel.get("api_key"):
        return AngelOneDepth(
            angel["api_key"], angel["client_id"],
            angel["password"], angel["totp_secret"],
        )

    dhan = secrets.get("dhan", {})
    if dhan.get("access_token"):
        return DhanDepth(dhan["client_id"], dhan["access_token"])

    return None
