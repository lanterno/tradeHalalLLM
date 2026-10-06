# Rotating the Postgres password

A playbook, not an alert. The repo's default password (`trader-dev-only`)
is public; a deployment should set its own.

The compose files read `POSTGRES_PASSWORD` from the repo's `.env` (the
justfile passes `--env-file .env`) and build every container's
`DATABASE_URL` from it, so the containers need one value. Postgres itself
only reads `POSTGRES_PASSWORD` when it creates an empty data directory: an
existing volume keeps its old password until it is changed in SQL.

Outside market hours:

```bash
# 1. Generate a password (no characters that need URL-escaping in a DSN).
openssl rand -hex 24

# 2. Change it in the database (local socket, no password needed).
docker exec -it halal-trader-pg psql -U trader -d halal_trader \
    -c "ALTER USER trader PASSWORD '<new>'"

# 3. In .env: set POSTGRES_PASSWORD=<new>, and put the same password in
#    DATABASE_URL (used by `uv run` on the host, on 127.0.0.1:5433).

# 4. Recreate the app containers so they get the new DATABASE_URL.
just home-up

# 5. Check.
just home-health
```

Test runs are unaffected: the suite reads `TEST_PG_*` (default
`trader-dev-only`), so set `TEST_PG_PASS=<new>` in the shell that runs
`just test`.

---

_Last reviewed: 2026-10-06_
