"""
Shared NSE session.

NSE blocks naive requests. Two things are required:
  1. Prime cookies by visiting the homepage before hitting /api/ endpoints.
  2. On cloud hosts (Streamlit Cloud, Render, EC2) use HTTP/2 via httpx --
     plain `requests` is fingerprinted and 403'd from datacenter IPs.

Rate limit: keep under ~3 requests/second. Everything here is cached upstream.
"""

import time
import httpx

BASE = "https://www.nseindia.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": f"{BASE}/",
    "Connection": "keep-alive",
}


class NSESession:
    """Cookie-primed NSE client. Re-primes automatically on 401/403."""

    def __init__(self, timeout: float = 15.0):
        self._client = httpx.Client(
            http2=True,
            headers=HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )
        self._primed = False

    def _prime(self) -> None:
        # Homepage sets the cookies the /api/ routes check for.
        self._client.get(BASE)
        time.sleep(0.4)
        self._client.get(f"{BASE}/market-data/live-equity-market")
        time.sleep(0.4)
        self._primed = True

    def get_json(self, path: str, retries: int = 2):
        """GET an NSE /api/ path and return parsed JSON, or None on failure."""
        if not self._primed:
            self._prime()

        url = path if path.startswith("http") else f"{BASE}{path}"

        for attempt in range(retries + 1):
            try:
                resp = self._client.get(url)
                if resp.status_code in (401, 403) and attempt < retries:
                    self._primed = False
                    self._prime()
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception:
                if attempt >= retries:
                    return None
                time.sleep(1.0 + attempt)
        return None

    def close(self) -> None:
        self._client.close()


_session: NSESession | None = None


def get_session() -> NSESession:
    """Process-wide singleton so cookies are reused across reruns."""
    global _session
    if _session is None:
        _session = NSESession()
    return _session
