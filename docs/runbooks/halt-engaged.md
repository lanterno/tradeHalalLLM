# Kill-switch engaged

**What it does:** with the halt on, the stock bot's cycle checks
`core/halt.is_halted` before anything else and places no new entries;
the news reactor and the core portfolio refuse new orders too. The
position monitor keeps running, so stops and take-profits still exit.

**Who engages it:** only the operator, from the CLI
(`halal-trader halt --reason "..."`) or the dashboard's kill-switch
button. Nothing in the code engages it automatically.

## Diagnose

```bash
docker exec trader-stocks halal-trader halt-status   # reason, who, when
```

The dashboard's Risk & System page shows the same.

## Resume

Only when the reason for the halt is resolved:

```bash
docker exec trader-stocks halal-trader resume
```

To liquidate rather than wait (the core's own account):
`halal-trader halt --close-all core`.

---

_Last reviewed: 2026-10-08_
