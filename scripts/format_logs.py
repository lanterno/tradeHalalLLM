#!/usr/bin/env python3
"""Pretty-print JSON log lines from halal_trader on stdin.

Used by the `just logs`, `just logs-tail`, and `just logs-errors` recipes.
Filters to halal_trader.* records by default; passes everything through if
--errors is given (the error log is already pre-filtered).

Output: HH:MM:SS.fff LEVEL [cycle-xxxxxxxx] [event=…] message
"""

from __future__ import annotations

import json
import sys


def _format(record: dict) -> str:
    ts = (record.get("timestamp") or "")[-12:]
    level = (record.get("level") or "?")[:7].ljust(7)
    cycle = record.get("cycle_id") or record.get("monitor_id") or record.get("request_id")
    event = record.get("event")
    message = (record.get("message") or "")[:140]

    parts = [ts, level]
    if cycle:
        parts.append(f"[{cycle}]")
    if event:
        parts.append(f"[{event}]")
    parts.append(message)
    return " ".join(parts)


def main() -> int:
    errors_only = "--errors" in sys.argv
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not errors_only:
            name = record.get("name") or ""
            if not name.startswith("halal_trader"):
                continue
        print(_format(record), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
