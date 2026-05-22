# launchd agents for the halabot bots

Run the stocks and crypto bots under macOS `launchd` so the OS handles
auto-restart on crash, throttles respawn storms, and keeps a captured
stdout/stderr per agent. The agents wrap each bot in `caffeinate -i`
to prevent idle-sleep and App-Nap throttling of the asyncio loop.

## One-time install

```bash
just launchd-install          # stocks + watchdog only (safer default)
# or
just launchd-install-all      # stocks + crypto + watchdog
```

Both copy plists into `~/Library/LaunchAgents/`, `launchctl bootstrap`
them under your user GUI session, and start them immediately. The bots
auto-start at every login from then on.

The default `launchd-install` **does not enable crypto** — the bot
would loop on `API Secret required for private endpoints` until
`BINANCE_API_KEY` + `BINANCE_SECRET_KEY` are in `.env`. Once creds
are set, enable it with:

```bash
just launchd-enable-crypto    # bootstrap crypto from the on-disk plist
just launchd-disable-crypto   # bootout crypto (plist stays on disk)
```

## Operate

```bash
just launchd-status                   # ps + launchctl print for all three
just launchd-restart-stocks           # bounce just the stocks bot
just launchd-restart-crypto           # bounce just the crypto bot
just launchd-uninstall                # bootout + delete all plists
```

Log files:
- `logs/launchd-stocks.out.log` / `logs/launchd-stocks.err.log`
- `logs/launchd-crypto.out.log` / `logs/launchd-crypto.err.log`

The bots still write their own structured JSON logs to
`logs/halal_trader.log` — the launchd-* files only capture anything
that escapes the Python logger (uncaught exceptions, segfaults,
caffeinate complaints, etc.).

## Why launchd vs `just stocks` in a terminal

A bot started from `just stocks` is bound to its terminal. If the
terminal closes (SIGHUP) or the shell dies, the bot dies with it. macOS
also applies background QoS to terminal processes once the terminal
loses focus, which is what caused the 75-min APScheduler silence on
2026-05-20. `launchd` adopts the bot as an OS-managed background
process with full priority, captures crashes, and respawns
automatically.

## Paths

The plists hardcode:
- `/Users/nourataha/.local/bin/uv` — `uv` binary
- `/Users/nourataha/lab/halabot` — repo root

If either moves, edit the plists and re-run `just launchd-install`.
