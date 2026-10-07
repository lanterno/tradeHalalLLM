#!/usr/bin/env bash
# Every 5 minutes (halabot-health.timer): is the fleet alive? `just health`
# passes only when every long-running container runs and the bot's database
# heartbeat is fresh. After two failures in a row (so a rebuild or a reboot
# does not page), Telegram; then at most hourly while it lasts; and once more
# when it recovers. While healthy it pings
# HEALTHCHECK_PING_URL, the outside dead-man switch for a server that is
# down too hard to alert about itself.
set -uo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo" || exit 1
SERVER_ENV=${SERVER_ENV:-/etc/halabot/server.env}
STATE_DIR=${STATE_DIR:-/var/lib/halabot}
ALERTED="$STATE_DIR/health-alerted"   # epoch of the last alert while failing
STRIKES="$STATE_DIR/health-strikes"   # consecutive failed checks
REALERT_S=3600

ping_url=$(grep -E '^HEALTHCHECK_PING_URL=' "$SERVER_ENV" 2>/dev/null | tail -1 | cut -d= -f2-)

if out=$(just health 2>&1); then
    rm -f "$STRIKES"
    if [ -f "$ALERTED" ]; then
        rm -f "$ALERTED"
        infra/server/alert.sh "recovered: the fleet is healthy again"
    fi
    [ -n "$ping_url" ] && curl -fsS --max-time 10 -o /dev/null "$ping_url"
    exit 0
fi

strikes=$(( $(cat "$STRIKES" 2>/dev/null || echo 0) + 1 ))
echo "$strikes" > "$STRIKES"
now=$(date +%s)
last=$(cat "$ALERTED" 2>/dev/null || echo 0)
if [ "$strikes" -ge 2 ] && [ $((now - last)) -ge "$REALERT_S" ]; then
    echo "$now" > "$ALERTED"
    infra/server/alert.sh "UNHEALTHY: ${out:-just health failed} (see: just docker-status)"
fi
exit 1
