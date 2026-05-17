# `db.connection_lost` — Postgres unreachable

**Severity:** PAGE
**Triggers when:** repeated `OperationalError` from asyncpg /
psycopg in the cycle / monitor / repository layer.
**Acknowledgement window:** 5 minutes — without the database
the bot can't write trades, can't read halal-cache, and the
monitor can't enforce SL/TP.

## Likely causes

1. **Postgres container stopped.** Most common cause —
   `docker compose down` / system reboot / OOM kill.
2. **Network partition.** Bot host can't reach the Postgres
   host (when running on separate machines).
3. **Postgres ran out of connections.** `max_connections`
   hit; new requests rejected.
4. **Postgres ran out of disk.** WAL backup failed and
   the data directory filled the volume.
5. **Auth broken.** Password or `pg_hba.conf` rotated
   without updating `DATABASE_URL`.

## Diagnose

```bash
# Container status
docker compose ps postgres

# Reachability
nc -vz localhost 5433

# Postgres responding to queries
docker exec -it $(docker compose ps -q postgres) \
  psql -U trader -d halal_trader -c "SELECT version()"

# Connection count + max
docker exec -it $(docker compose ps -q postgres) \
  psql -U trader -d halal_trader \
  -c "SELECT count(*), (SELECT setting FROM pg_settings
                        WHERE name='max_connections')
      FROM pg_stat_activity"

# Disk usage on the data volume
docker exec -it $(docker compose ps -q postgres) df -h
```

## Mitigate

1. **If container stopped** — engage halt + restart Postgres:
   ```bash
   uv run halal-trader halt --reason "DB down"
   just pg-up
   ```
   Wait for healthy status, then resume:
   ```bash
   uv run halal-trader resume
   ```
2. **If connection-count exhausted** — most likely a connection
   leak in a recent code change. Restart the bot to release its
   pool; if the count comes back high, investigate the leak.
   Temporary mitigation:
   ```bash
   docker exec -it $(docker compose ps -q postgres) \
     psql -U trader -d halal_trader \
     -c "SELECT pg_terminate_backend(pid)
         FROM pg_stat_activity
         WHERE state = 'idle' AND state_change < now() - interval '10 minutes'"
   ```
3. **If disk full** — engage halt; clean up old WAL / log
   archives:
   ```bash
   docker exec -it $(docker compose ps -q postgres) \
     bash -c 'find /var/lib/postgresql -name "*.log.*" -mtime +7 -delete'
   ```
   Increase the volume size for a permanent fix (Wave 8.F backup
   architecture covers retention policy).
4. **If auth broken** — update `DATABASE_URL` in `.env`,
   restart the bot. Verify by running `uv run halal-trader db
   current` (should print the migration revision without
   error).
5. **If network partition** — the bot must run on the same
   network as Postgres; investigate the host-side connectivity
   (firewall rules, VPN). The bot's halt status persists across
   restarts so resuming is safe once the partition heals.

## Escalate

If Postgres is up + reachable + has connection headroom + has
disk + auth works, but the bot still reports
`OperationalError`, this is a client-side issue. Check the
bot's connection pool config in `Settings.database_url` and
inspect with verbose logging:

```bash
SQLALCHEMY_LOG_LEVEL=DEBUG uv run halal-trader crypto start
```

## Postmortem

PAGE alert — file `docs/postmortems/<date>-db-down.md` within
48h. Include: time-to-detection, time-to-mitigation, root
cause, what to add to monitoring (Wave 8.E alert routing) so
this fires earlier next time.

---

_Last reviewed: 2026-05-01_
