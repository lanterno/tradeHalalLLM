"""JSON-friendly recursive serializer used by all route modules."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def serialize(obj: Any) -> Any:
    """Convert datetime values in dicts/lists to ISO strings for JSON."""
    if isinstance(obj, list):
        return [serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
