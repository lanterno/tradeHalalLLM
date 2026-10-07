#!/usr/bin/env bash
# Send one operator alert to Telegram, with the bot's own credentials
# (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from the repo's .env). For the host
# side of the fleet, where the bot's AlertSink is not running: a failed
# backup, a dead container. Never fails its caller; no token, no alert.
#
#   infra/server/alert.sh "backup failed on $(hostname)"
set -uo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
env_value() { grep -E "^$1=" "$repo/.env" 2>/dev/null | tail -1 | cut -d= -f2-; }

token=$(env_value TELEGRAM_BOT_TOKEN)
chat=$(env_value TELEGRAM_CHAT_ID)
text="halabot $(hostname): $*"
echo "$text" >&2
[ -n "$token" ] && [ -n "$chat" ] || exit 0
curl -fsS --max-time 10 -o /dev/null "https://api.telegram.org/bot${token}/sendMessage" \
    --data-urlencode "chat_id=${chat}" --data-urlencode "text=${text}" \
    || echo "alert.sh: Telegram send failed" >&2
exit 0
