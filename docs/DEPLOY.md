# Deploying to a server (Hetzner, from scratch)

The fleet runs on one Linux server: Postgres, a one-shot migrate, the stock
bot, the shadow engine and the dashboard, all in Docker (`infra/docker-compose.yml`),
plus three systemd units on the host for what Docker cannot do itself: the
nightly off-site backup, a health check that alerts, and a failure alert.

The previous machine (WSL2, 2026-07 to 2026-10) was lost with its database
and its `.env`, because both, and the nightly dumps, lived only on that one
computer. Two rules come from that:

- **Every secret also lives in a password manager.** `.env` and
  `/etc/halabot/server.env` (above all `RESTIC_PASSWORD`) are on the server
  only, and the server can go the same way.
- **A backup counts once it is off the server.** The evening run alerts
  when the off-site copy is missing or more than 36 h old.

## 1. Create the server

Hetzner Cloud → Add server:

- **Image:** Ubuntu 24.04.
- **Type:** 4 vCPU / 8 GB or more, 80 GB disk or more. ARM (CAX21) and x86
  (CPX31) both work: every locked wheel, torch included, exists for both.
  The image build needs the memory (bootstrap adds 4 GB of swap); the
  research store grows past 1 GB once seeded.
- **Location:** any. Alpaca is in US-East, so Ashburn is nearest, but at
  15-minute cycles latency does not matter. Check the location offers the
  type you picked.
- **SSH key:** add yours. Bootstrap turns off password logins only once a
  key is in place.
- **Firewall (optional, a second layer):** a Hetzner Cloud Firewall that
  allows inbound TCP 22 only. Tailscale needs no inbound port.

## 2. Bootstrap

```bash
ssh root@<server-ip>
curl -fsSL https://raw.githubusercontent.com/lanterno/tradeHalalLLM/main/infra/server/bootstrap.sh -o bootstrap.sh
less bootstrap.sh       # read it: it runs as root
bash bootstrap.sh       # TAILSCALE_AUTHKEY=tskey-... bash bootstrap.sh joins the tailnet unattended
```

What it does, idempotently: updates the system and enables security
updates; installs Docker (with log rotation), `just`, `restic`, `jq`;
adds 4 GB of swap; sets ufw to SSH only; makes SSH key-only
(`/etc/ssh/sshd_config.d/10-halabot.conf`, only once root has a key);
creates the `halabot` user (in the `docker` group, with root's SSH key);
clones the repo to `/opt/halabot`; writes `.env` from `.env.example` with a
fresh `POSTGRES_PASSWORD` (also in `DATABASE_URL`) and `WEB_API_TOKEN`;
writes `/etc/halabot/server.env` from `infra/server/server.env.example`;
installs and starts the systemd timers; installs Tailscale. Without
`TAILSCALE_AUTHKEY`, join the tailnet now with
`tailscale up --hostname halabot` (it prints a login URL).

Before you log out, check from a second terminal that
`ssh halabot@<server-ip>` and `ssh root@<server-ip>` still let you in.

The timers run from now on, before anything is configured, so expect
failures until the steps below are done: the health check fails every 5
minutes until the fleet is up (step 5), and once Telegram is filled in
(step 3) it alerts after two failures, then hourly; the 07:00 UTC backup
fails, and alerts, until step 4 is done. Both say why in
`journalctl -u halabot-health` / `-u halabot-backup`.

## 3. Fill in `.env`

```bash
sudo -u halabot nano /opt/halabot/.env
```

Fill in the Required section; the file says where each key comes from,
and everything below it has working defaults. Add the Recommended keys
(Telegram above all: the host's alerts use it too), and for the core
portfolio a **second** Alpaca paper account's keys (`CORE_ALPACA_*`; the
core stays off while they are empty).

**Starting from an empty database against an Alpaca paper account that the
old bot traded:** reset the paper account (Alpaca → Paper Trading → Reset)
before the first start. Otherwise its open positions have no rows here and
show up as reconcile drift. Resetting issues new keys.

Then save `.env` in your password manager.

## 4. Off-site backups

Pick a repository and fill in `/etc/halabot/server.env` (`sudo nano`). The
file has the forms for Cloudflare R2, Hetzner Object Storage and a Hetzner
Storage Box. Write the values bare (no quotes). Generate the password with
`openssl rand -base64 32` and **save it in your password manager now**:
without it no snapshot can be read.

A Storage Box is reached over SSH by the `halabot` user, which has no key
of its own yet. Give it one and put it on the box; `ssh-copy-id` asks for
the box's password once and records its host key (an unknown host key
makes the unattended run fail):

```bash
sudo -u halabot ssh-keygen -t ed25519 -N '' -f /home/halabot/.ssh/id_ed25519
sudo -u halabot ssh-copy-id -p 23 -s -i /home/halabot/.ssh/id_ed25519 u123456@u123456.your-storagebox.de
sudo -u halabot sftp -P 23 u123456@u123456.your-storagebox.de <<< 'ls'   # must list, with no password asked
```

Then, for any repository:

```bash
sudo -u halabot /opt/halabot/infra/server/backup.sh --init   # once
```

The first real run needs the fleet up (step 5). After it:

```bash
sudo systemctl start halabot-backup.service       # a run now, instead of waiting for 07:00 UTC
journalctl -u halabot-backup.service -n 30
sudo -u halabot /opt/halabot/infra/server/backup.sh --list
```

`just backup` runs inside it every night. It writes the dump plus
`live_events.jsonl.gz` and verifies the dump. On the 1st of the month it
also runs the restore drill into a scratch database. Then restic uploads it
(14 daily, 8 weekly and 12 monthly snapshots kept; a 5% read-back check on
Sundays), and local copies older than `KEEP_LOCAL_DAYS` are deleted. A
failure sends a Telegram alert (`halabot-alert@`), and the evening run
reports a stale `backup.nightly` or `backup.offsite` heartbeat.

## 5. Start the fleet

Outside US market hours (before 09:00 or after 16:00 ET):

```bash
sudo -iu halabot
cd /opt/halabot
just build          # ~10 min the first time
just up             # postgres → migrate (to head, on the empty DB) → bot, shadow, web
just health         # exit 0: every container runs and the bot's heartbeat is fresh
just docker-logs    # the bot; `just docker-logs trader-shadow` etc.
```

Watch the first market-hours cycle (`just docker-logs`). The failure mode
is a bot that silently does nothing, not a crash.

Inside the fleet, the web container's watchdog (`web/watchdog.py`) alerts on
Telegram when a heartbeat or a daily job goes stale. From the host,
`halabot-health.timer` runs `just health` every 5 minutes. After two
failures in a row it alerts on Telegram, then at most hourly, and once more
on recovery. For a server that dies outright, set `HEALTHCHECK_PING_URL`
in `server.env` to a dead-man service (e.g. healthchecks.io), which alerts
when the pings stop.

## 6. The dashboard

The dashboard listens on `127.0.0.1:8082` only. Reach it over:

- **Tailscale (recommended):** `sudo tailscale serve --bg 8082`, then
  `https://halabot.<tailnet>.ts.net` from any device on the tailnet,
  phone included. The first time, it prints a link to turn on Serve and
  HTTPS certificates for the tailnet in the admin console.
- **An SSH tunnel:** `ssh -L 8082:127.0.0.1:8082 halabot@<server>`, then
  http://localhost:8082.

State-changing endpoints need `WEB_API_TOKEN` (generated in `.env`).

## 7. Seed the research store (fresh database)

Both strategies trade only what today's in-house screen holds halal, and
that screen starts empty: until `compliance screen` has run, the
day-trader's universe is empty and the core refuses to trade. Run the
first four steps below (assets, backfill, etf-history, screen) before the
first market open. The rest (the event store, forward books, history)
can follow. The evening run keeps every store current but does not fill
ten years of history. Every step is resumable: if one is interrupted,
run it again.

Run them in `tmux` (the long ones take hours), as the `halabot` user:

```bash
run() { docker exec trader-stocks halal-trader "$@"; }
run data assets
run data backfill                # the liquid universe's daily bars
run compliance etf-history       # SPUS/HLAL holdings (the index veto)
run compliance screen            # today's verdicts: both strategies need these
run data fundamentals
run data pit-universe            # monthly + daily bars for the point-in-time universe (long)
run compliance screen-history    # every past quarter end (needs pit-universe)
run events backfill all          # news, filings, insiders, EPS (long; shares the Alpaca key's rate)
run events extract
run books create core --strategy core-strict-cap --top 100 --cost-bps 5   # the core's book: its gate measures the account against it
run books create s1               # the day-trader's research book (momentum, low vol)
```

`halal-trader core readiness` then says what the core still needs before
it may hold real money.

## Day to day

| Task | Command |
|---|---|
| Deploy new code | `git pull && just build && just up` outside market hours; watch the next cycle |
| Stop new entries now | `docker exec trader-stocks halal-trader halt --reason "..."` |
| Status of every service | `just docker-status` |
| A shell in the database | `just docker-psql` |
| Restore the latest off-site backup | see below |

### Restore from off-site

As the `halabot` user:

```bash
(set -a; . /etc/halabot/server.env; set +a; restic restore latest --tag halabot --target /tmp/restore)
cd /opt/halabot && just down && just pg-up
just db-restore /tmp/restore/var/backups/halabot/<date>/halal_trader.dump
just up
docker exec trader-stocks halal-trader ledger sync   # Alpaca's record fills the gap since the dump
```

The dump leaves out rows that can be rebuilt (bars, the event store,
fundamentals). Step 7 refills them. `live_events.jsonl.gz` beside the
dump holds the live reactor's headlines and scores, which cannot be rebuilt.

On a new server, install the same `server.env` (from your password
manager) after bootstrap and run the same commands.
