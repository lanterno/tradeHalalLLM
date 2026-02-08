"""Zoya GraphQL API client for Shariah compliance screening."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Zoya compliance status -> our internal format
_STATUS_MAP: dict[str, str] = {
    "COMPLIANT": "halal",
    "NON_COMPLIANT": "not_halal",
    "DOUBTFUL": "doubtful",
}

_LIVE_URL = "https://api.zoya.finance/graphql"
_SANDBOX_URL = "https://sandbox-api.zoya.finance/graphql"

_REPORT_QUERY = """
query GetReport($symbol: String!) {
  basicCompliance {
    report(symbol: $symbol) {
      symbol
      name
      status
      reportDate
    }
  }
}
"""

_REPORTS_QUERY = """
query ListReports($input: BasicReportsInput) {
  basicCompliance {
    reports(input: $input) {
      items {
        symbol
        name
        status
        reportDate
      }
      nextToken
    }
  }
}
"""


class ZoyaClient:
    """Client for the Zoya Shariah-compliance GraphQL API.

    Authentication uses a simple API key passed in the Authorization header.
    """

    def __init__(self, api_key: str, *, use_sandbox: bool = False) -> None:
        self.api_key = api_key
        self._url = _SANDBOX_URL if use_sandbox else _LIVE_URL
        self._http = httpx.AsyncClient(timeout=30.0)

    async def _query(self, query: str, variables: dict[str, Any] | None = None) -> Any:
        """Execute a GraphQL query against the Zoya API."""
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        response = await self._http.post(self._url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()

        # GraphQL errors are returned in the "errors" key with HTTP 200
        if "errors" in body:
            error_msgs = "; ".join(e.get("message", str(e)) for e in body["errors"])
            raise RuntimeError(f"Zoya GraphQL error: {error_msgs}")

        return body.get("data")

    @staticmethod
    def _normalize_status(status: str) -> str:
        """Map Zoya compliance status to our internal format."""
        return _STATUS_MAP.get(status, "doubtful")

    async def screen_stock(self, symbol: str) -> dict[str, Any]:
        """Get the Shariah compliance report for a single stock.

        Returns a dict with at least:
            - compliance: "halal" | "not_halal" | "doubtful"
            - detail: human-readable explanation
        """
        try:
            data = await self._query(_REPORT_QUERY, {"symbol": symbol})
            report = (data or {}).get("basicCompliance", {}).get("report")

            if report is None:
                return {
                    "symbol": symbol,
                    "compliance": "doubtful",
                    "detail": "No report found",
                }

            return {
                "symbol": symbol,
                "compliance": self._normalize_status(report.get("status", "")),
                "detail": f"{report.get('name', symbol)} — {report.get('status', 'UNKNOWN')}",
                "raw": report,
            }
        except httpx.HTTPStatusError as e:
            logger.warning("Zoya API error for %s: %s", symbol, e)
            return {
                "symbol": symbol,
                "compliance": "doubtful",
                "detail": f"API error: {e.response.status_code}",
            }
        except Exception as e:
            logger.warning("Zoya API error for %s: %s", symbol, e)
            return {
                "symbol": symbol,
                "compliance": "doubtful",
                "detail": str(e),
            }

    async def screen_bulk(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Screen multiple stocks for Shariah compliance.

        Screens each symbol individually via the single-stock query since Zoya's
        bulk endpoint lists all US stocks rather than a specific subset.
        """
        results = []
        for symbol in symbols:
            result = await self.screen_stock(symbol)
            results.append(result)
        return results

    async def close(self) -> None:
        await self._http.aclose()
