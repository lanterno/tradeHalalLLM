# Backups + point-in-time recovery

This is the project's backup architecture and the on-call's
restore-drill procedure. Targets: **RPO ≤ 5 minutes**, **RTO ≤ 30
minutes**. Treat this document as a runbook — when the operator
needs to restore, they should be able to follow it without
referencing other docs.

> **Related runbooks.** A live database failure is the
> [`db-connection-lost`](db-connection-lost.md) PAGE alert; a
> trade-row gap is the [`snapshot-store-failure`](snapshot-store-failure.md)
> WARN alert. This document is what the operator runs after
> they've stabilised the immediate failure and need to recover
> data.

## What's backed up

The bot's persistent state is in **one** Postgres database
(`halal_trader`). The `infra/docker-compose.yml` Postgres service
is the source of truth for every audit-trail table:

| Concern | Tables |
|---|---|
| Trade history | `trades`, `crypto_trades`, `crypto_daily_pnl` |
| LLM decisions | `llm_decisions` (with prompt / response / cost) |
| Halal compliance audit | `halal_screenings`, `crypto_halal_cache`, `stock_halal_cache`, `halal_exception_queue`, `purification_entries` |
| Indicator snapshots (replay) | `indicator_snapshots` |
| ML artefacts | `ml_artefacts` |
| Strategy adjustments | `strategy_adjustments` |
| Web actions audit | `web_actions` |
| Pair pauses, runtime config | `pair_pause`, `runtime_config` |
| Per-asset metadata | `prompt_genomes`, `thesis_tags`, `regret_records`, `shadow_ledger`, `replay_snapshots`, `research_jobs` |

The bot's own filesystem state (`models/`, `logs/`, `dashboard/dist/`)
is rebuildable from source + the database; no separate backup needed.

The HuggingFace model cache (`~/.cache/huggingface/`) is
deliberately **not** backed up — it's third-party model weights,
re-downloadable on demand.

## What's NOT backed up

* **`.env` secrets** — these are operator-local and should be
  backed up by the operator's own secret-management process
  (1Password / Bitwarden / encrypted USB), not by the bot.
  Including them in the bot's backup pipeline would make every
  backup a credential disclosure.
* **Trading-account positions** — the broker (Alpaca / Binance)
  is authoritative for actual position state. The bot's view is
  reconstructable via `core/reconcile.py:reconcile_positions`
  on resume.
* **Live LLM API keys** — same reasoning as `.env`.

## Architecture

```
┌─────────────┐     ┌─────────────────────────┐     ┌─────────────────┐
│ Postgres    │────→│ /var/lib/postgres/wal/  │────→│ pgBackRest      │
│ (live)      │ WAL │   archive_command       │ ssh │ remote (S3 /    │
│             │     │                         │     │  separate disk) │
│             │     │ /var/lib/postgres/base/ │────→│                 │
│             │ base│   pgBackRest base backup│     │                 │
└─────────────┘     └─────────────────────────┘     └─────────────────┘
```

Two layers compose:

* **WAL archiving** — Postgres streams every committed
  transaction's WAL segment to an archive directory.
  `archive_command` ships each segment to remote storage as
  it closes (default 16 MB / segment, so a busy bot rotates
  every few minutes).
* **Base backups** — daily `pg_basebackup` (or `pgBackRest
  backup --type=full` weekly + `--type=diff` daily) captures a
  full data-directory snapshot. Base backups + WAL stream
  together let the operator restore to any point in time
  between the oldest retained base backup and the most-recent
  archived WAL.

### Why pgBackRest over plain `pg_basebackup` + cron

`pg_basebackup` works for a small deployment but doesn't
deduplicate, doesn't parallelise, and doesn't have a clean
"restore to time X" command. pgBackRest:

* Deduplicates blocks across full / differential / incremental
  backups (so 7 days of dailies + a weekly full take ~1.5×
  the database size, not 8×).
* Parallelises compression + upload (8-way default).
* Encrypts at rest with a passphrase the operator owns.
* Has a single `pgbackrest restore --type=time --target=...`
  command that handles the WAL replay.

For a one-machine paper-trading deployment, `pg_basebackup` +
cron is acceptable. For anything operator considers production,
ship pgBackRest.

## Setup (one-time)

### 1. Enable WAL archiving

In `infra/docker-compose.yml`, add to the Postgres service:

```yaml
postgres:
  command:
    - postgres
    - -c
    - wal_level=replica
    - -c
    - archive_mode=on
    - -c
    - archive_command=/scripts/archive-wal.sh %p %f
    - -c
    - max_wal_senders=3
  volumes:
    - postgres-data:/var/lib/postgresql/data
    - ./scripts/archive-wal.sh:/scripts/archive-wal.sh:ro
    - postgres-wal-archive:/var/lib/postgresql/wal-archive
```

Where `scripts/archive-wal.sh` is:

```bash
#!/bin/bash
# Copy a WAL segment to the archive volume; pgBackRest replaces
# this in production. Pin: errors must propagate (set -e); a
# silently-failed archive is the worst-case backup outcome.
set -euo pipefail
cp "$1" "/var/lib/postgresql/wal-archive/$2"
```

Restart the container after the config change.

### 2. Take the first base backup

```bash
docker exec -it $(docker compose ps -q postgres) \
  pg_basebackup \
    -U trader \
    -D /var/lib/postgresql/base-backups/$(date -u +%Y%m%d-%H%M%S) \
    -F tar -z -P -v
```

Verify the backup landed:

```bash
docker exec -it $(docker compose ps -q postgres) \
  ls -lh /var/lib/postgresql/base-backups/
```

### 3. Schedule daily base backups

`infra/cron-backups.sh` (host-side cron, not container-side):

```bash
#!/bin/bash
set -euo pipefail
docker exec $(docker compose ps -q postgres) \
  pg_basebackup \
    -U trader \
    -D /var/lib/postgresql/base-backups/$(date -u +%Y%m%d-%H%M%S) \
    -F tar -z -P
# Retain 14 days of base backups; rotate older.
find /var/lib/postgresql/base-backups -maxdepth 1 -mindepth 1 \
  -type d -mtime +14 -exec rm -rf {} \;
```

Crontab line on the host:

```
0 4 * * * /path/to/halal-trader/infra/cron-backups.sh \
  >> /var/log/halal-trader-backups.log 2>&1
```

### 4. Off-machine sync

The on-machine archive volume is **not enough** — a single-host
disk failure loses base backups + WAL together. Ship to remote
storage:

```bash
# rclone to S3 / GCS / Backblaze B2 — operator picks
rclone sync /var/lib/postgresql/wal-archive remote:wal-archive
rclone sync /var/lib/postgresql/base-backups remote:base-backups
```

Run nightly via cron, after the base backup completes.

## Restore

### Restore to "right now" (full disaster recovery)

The most common scenario: data volume corrupted; restore the
database in-place from the latest base backup + replay all WAL
since.

```bash
# 1. Stop the bot.
uv run halal-trader halt --reason "DB restore in progress"
docker compose stop bot  # or whatever wraps the bot service

# 2. Stop Postgres and wipe its data directory.
docker compose stop postgres
docker volume rm halal-trader_postgres-data

# 3. Recreate the volume + restore the most recent base backup.
docker compose up -d postgres
docker exec -i $(docker compose ps -q postgres) bash <<'EOF'
set -euo pipefail
cd /var/lib/postgresql/data
LATEST=$(ls -t /var/lib/postgresql/base-backups/ | head -1)
tar -xzf "/var/lib/postgresql/base-backups/$LATEST/base.tar.gz" -C .
EOF

# 4. Recovery config: replay WAL.
docker exec -i $(docker compose ps -q postgres) bash <<'EOF'
cat > /var/lib/postgresql/data/recovery.signal << 'END'
END
cat > /var/lib/postgresql/data/postgresql.auto.conf << 'END'
restore_command = 'cp /var/lib/postgresql/wal-archive/%f %p'
recovery_target_timeline = 'latest'
END
EOF

# 5. Restart Postgres; it'll replay WAL until exhaustion.
docker compose restart postgres

# 6. Verify the latest data is present.
docker exec $(docker compose ps -q postgres) \
  psql -U trader -d halal_trader \
  -c "SELECT max(timestamp) FROM crypto_trades"

# 7. Run reconcile to sync the bot's position view.
uv run halal-trader db migrate    # confirm at head
uv run halal-trader resume
```

**Expected RTO**: under 30 minutes for a database under ~20 GB.
Larger databases scale linearly with the base backup tar
extraction time.

### Point-in-time recovery (PITR)

Used when the operator needs to roll back to a specific moment
— e.g., a corrupt migration committed at 14:20 UTC and the
operator wants to restore to 14:15 UTC, before the corruption.

Same procedure as full recovery but with an explicit target:

```bash
# In step 4, replace the recovery_target_timeline line with:
cat > /var/lib/postgresql/data/postgresql.auto.conf << 'END'
restore_command = 'cp /var/lib/postgresql/wal-archive/%f %p'
recovery_target_time = '2026-05-01 14:15:00 UTC'
recovery_target_action = 'promote'
END
```

Postgres replays WAL up to (and including) the target time, then
promotes. Anything after 14:15 is **not** in the restored
database — make sure the operator has captured anything they
care about from the live state before starting.

### Restore drill (quarterly)

The on-call runs this drill every quarter to verify the backup
chain works. **Do not** run on the production volume; spin up a
disposable Postgres container side-by-side:

```bash
# 1. Spin up a disposable side container with a fresh volume.
docker run --rm -d --name pg-restore-drill \
  -e POSTGRES_PASSWORD=drill \
  -v $(pwd)/var/lib/postgresql/wal-archive:/wal-archive:ro \
  -v $(pwd)/var/lib/postgresql/base-backups:/base-backups:ro \
  -v pg-restore-drill-data:/var/lib/postgresql/data \
  postgres:16

# 2. Wait for it to start.
sleep 10

# 3. Pick yesterday's base backup and a target time at midnight UTC.
LATEST=$(ls -t /var/lib/postgresql/base-backups/ | head -2 | tail -1)
docker exec -i pg-restore-drill bash <<EOF
set -euo pipefail
cd /var/lib/postgresql/data
tar -xzf "/base-backups/$LATEST/base.tar.gz" -C .
cat > recovery.signal << 'END'
END
cat > postgresql.auto.conf << 'END'
restore_command = 'cp /wal-archive/%f %p'
recovery_target_time = '$(date -u -d "yesterday 23:59:00" +%Y-%m-%d\ %H:%M:%S) UTC'
recovery_target_action = 'promote'
END
EOF
docker restart pg-restore-drill

# 4. Wait for promotion + verify.
sleep 30
docker exec pg-restore-drill \
  psql -U trader -d halal_trader \
  -c "SELECT count(*), max(timestamp) FROM crypto_trades"

# 5. Tear down the drill container + volume.
docker stop pg-restore-drill
docker volume rm pg-restore-drill-data
```

Record the drill result in `docs/postmortems/drills/<date>-restore.md`
even when it succeeds — the value is the verifiable evidence
that the backup chain works on a known cadence.

### Drill failure modes

If the drill produces:

* **Empty `crypto_trades`** — base backup tar didn't extract
  cleanly, or the WAL chain was broken. Check
  `archive_status/` directory in the data dir for
  `<segment>.partial` files (incomplete archive).
* **Postgres won't start after restore** — usually a missing
  WAL segment. The archive needs the *unbroken* sequence from
  the base backup's checkpoint through the target time.
  Compare `pg_controldata` output against the WAL files
  present.
* **`max(timestamp)` earlier than expected** — the target
  time was earlier than the operator wanted, or WAL archiving
  was lagging behind the live database. Tighten the
  archive_command timing or increase the WAL archive retention.

## Targets

| Metric | Target | Notes |
|---|---|---|
| RPO (Recovery Point Objective) | ≤ 5 min | WAL ships per segment (~16 MB); a busy bot rotates segments every 1-5 minutes. |
| RTO (Recovery Time Objective) | ≤ 30 min | DB ≤ 20 GB. Larger DBs scale linearly with extraction time. |
| Base backup retention | 14 days on-disk, 30 days off-machine | Deduplication makes this cheap with pgBackRest. |
| WAL archive retention | 14 days | Matches base-backup retention so any base + WAL replay works. |
| Drill cadence | Quarterly | Recorded in `docs/postmortems/drills/`. |

## Halal-specific considerations

Two edge cases the operator should know about:

* **Restoring past a halal screening update.** A symbol
  approved as `halal` 3 weeks ago and re-screened to
  `not_halal` last week — restoring to "3 weeks ago" makes
  the bot believe the symbol is halal again. The operator
  must re-run the screener (`halal-trader crypto screen`)
  after every restore that crosses a screening update.
* **Restoring past a purification disbursement.** A
  purification entry marked `paid_at` last week — restoring
  to before that mark makes the entry look outstanding
  again. The post-restore checklist below covers this.

## Post-restore checklist

After **any** restore (full DR or PITR):

* [ ] `halal-trader db current` reports the migration head
      revision.
* [ ] `halal-trader crypto screen` re-refreshes the halal
      cache (in case the restore reverted a `not_halal`
      update).
* [ ] `halal-trader status` shows the position view matches
      the broker (`reconcile_positions` runs on bot startup).
* [ ] `halal-trader halt-status` shows the previous halt
      reason or the resumed state.
* [ ] The Wave 6.E model fingerprint of the active model
      matches what was active before the restore — if not,
      a fresh promotion run is needed (Wave 6.A registry).
* [ ] Notify the operator's audit-record-keeper that a
      restore happened and over what time window
      (per-trade audit rows now correspond to the restored
      timeline, not the original).

---

_Last reviewed: 2026-05-01. Review quarterly._
