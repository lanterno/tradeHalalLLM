"""Container healthcheck: is a component's heartbeat younger than a limit?

    python -m halal_trader.core.healthcheck stock.process 180

Exit 0 when the ``heartbeats`` row for the component is younger than the
limit (seconds), 1 when it is older or missing, 2 when the database cannot
be asked. Compose runs it for trader-stocks and trader-shadow so ``docker
ps`` says "unhealthy" for a process that is running but no longer beating.

Kept deliberately small: one synchronous query with the ``psycopg`` driver
the image already has, no settings, no logging setup -- docker runs it every
minute.
"""

from __future__ import annotations

import os
import sys


def heartbeat_age(database_url: str, component: str) -> float | None:
    """Seconds since ``component`` last beat, or None if it never did."""
    import psycopg

    url = database_url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )
    with psycopg.connect(url, connect_timeout=5, autocommit=True) as conn:
        row = conn.execute(
            "SELECT extract(epoch FROM now() - beat_at) FROM heartbeats WHERE component = %s",
            (component,),
        ).fetchone()
    return None if row is None else float(row[0])


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m halal_trader.core.healthcheck <component> <max-age-seconds>")
        return 2
    component, limit = args[0], float(args[1])
    try:
        age = heartbeat_age(os.environ["DATABASE_URL"], component)
    except Exception as exc:  # noqa: BLE001 -- any failure is "cannot tell"
        print(f"{component}: database unavailable ({exc!r})")
        return 2
    if age is None:
        print(f"{component}: no heartbeat on record")
        return 1
    print(f"{component}: last beat {age:.0f}s ago (limit {limit:.0f}s)")
    return 0 if age <= limit else 1


if __name__ == "__main__":
    sys.exit(main())
