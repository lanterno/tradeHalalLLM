#!/usr/bin/env bash
# Nightly backup: `just backup` dumps and verifies the database into a dated
# local directory, then restic ships it, encrypted, to the off-site
# repository in /etc/halabot/server.env. The previous machine kept its dumps
# on the same hardware and lost them with it; a backup only counts once it
# is off this server, so the `backup.offsite` heartbeat is written only after
# the upload succeeds, and the evening run alerts when it goes stale.
#
#   backup.sh            nightly run (halabot-backup.timer)
#   backup.sh --init     create the restic repository, once
#   backup.sh --list     list the off-site snapshots
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
SERVER_ENV=${SERVER_ENV:-/etc/halabot/server.env}
BACKUP_DIR=${BACKUP_DIR:-/var/backups/halabot}
RESTIC=${RESTIC:-restic}

set -a
# shellcheck disable=SC1090
. "$SERVER_ENV"
set +a
: "${RESTIC_REPOSITORY:?set RESTIC_REPOSITORY in $SERVER_ENV}"
: "${RESTIC_PASSWORD:?set RESTIC_PASSWORD in $SERVER_ENV}"

case "${1:-}" in
    --init)
        if $RESTIC cat config >/dev/null 2>&1; then
            echo "restic repository already initialised"
        else
            $RESTIC init
        fi
        exit 0
        ;;
    --list)
        exec $RESTIC snapshots --tag halabot
        ;;
    "") ;;
    *)
        echo "usage: $0 [--init|--list]" >&2
        exit 2
        ;;
esac

dest="$BACKUP_DIR/$(date -u +%F)"
mkdir -p "$dest"
just backup "$dest"

snapshot=$($RESTIC backup --json --tag halabot --host halabot "$dest" \
    | jq -r 'select(.message_type == "summary") | .snapshot_id')
[ -n "$snapshot" ] || { echo "restic backup reported no snapshot" >&2; exit 1; }
$RESTIC forget --tag halabot --keep-daily 14 --keep-weekly 8 --keep-monthly 12 --prune
# Sundays: read back a slice of the stored data, not just the index.
if [ "$(date -u +%u)" = 7 ]; then
    $RESTIC check --read-data-subset=5%
fi

# The upload is done: old local copies can go whatever the heartbeat does.
find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d -mtime "+${KEEP_LOCAL_DAYS:-7}" -exec rm -rf {} +

# Fails only if the database went away since `just backup` read it. The
# snapshot is safe; the unit still fails (OnFailure alerts) because the
# evening run would otherwise report backup.offsite stale with no reason.
docker exec halal-trader-pg psql -U trader -d halal_trader -tAq -c \
    "INSERT INTO heartbeats (component, beat_at, detail) VALUES ('backup.offsite', now(), '{\"snapshot\": \"${snapshot:0:8}\"}') ON CONFLICT (component) DO UPDATE SET beat_at = EXCLUDED.beat_at, detail = EXCLUDED.detail" \
    || { echo "off-site snapshot ${snapshot:0:8} stored, but the backup.offsite heartbeat could not be written" >&2; exit 1; }
echo "off-site snapshot ${snapshot:0:8} of $dest"
