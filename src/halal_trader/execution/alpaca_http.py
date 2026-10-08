"""What every Alpaca REST client shares: endpoints, auth, timestamps.

Three clients talk to Alpaca for different jobs -- the broker that trades
(execution/alpaca_broker.py), the read-only ledger client
(execution/alpaca_rest.py) and the research market-data client
(data/alpaca_market.py) -- and each chooses its own timeouts and retries
(core/http.py). Nothing else about Alpaca is written twice.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

# The free plan serves SIP (consolidated) bars only once they are this old.
SIP_EMBARGO = timedelta(minutes=16)


def trading_url(paper: bool) -> str:
    """The trading API's base URL for the account's environment."""
    return PAPER_URL if paper else LIVE_URL


def auth_headers(api_key: str, secret_key: str) -> dict[str, str]:
    """Alpaca's two auth headers; refuses missing keys rather than send empty ones."""
    if not api_key or not secret_key:
        raise ValueError("Alpaca API key and secret are required")
    return {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}


def iso_z(when: datetime) -> str:
    """``when`` in UTC as Alpaca writes timestamps (``...Z``)."""
    return when.astimezone(UTC).isoformat().replace("+00:00", "Z")
