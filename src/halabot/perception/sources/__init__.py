"""Concrete perception sources — adapters over real feeds (venue/data adapters).

These wrap the existing ``halal_trader`` clients (Alpaca MCP, Finnhub) behind
the :class:`Source`/`PollingSource` contracts, emitting ``observation.*`` events.
This is the one place ``halabot`` couples to the legacy package during
migration; everything upstream stays venue-agnostic.
"""
