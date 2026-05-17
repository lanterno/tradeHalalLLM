# `halt.engaged` — kill-switch trip

**Severity:** PAGE
**Triggers when:** the bot's halt flag is engaged
(`halal-trader halt --reason "..."`, `LLM_DAILY_USD_CAP` exceeded,
LLM circuit-breaker tripped, automated drift-breach gate). The
cycle's `BaseCycleService.run_cycle` checks `core/halt.is_halted(engine)`
*before* any other logic and refuses to enter new positions.
**Acknowledgement window:** 15 minutes.

## Likely causes

1. **Operator-engaged halt** — someone ran
   `halal-trader halt --reason "..."` for a planned maintenance
   window or a manual de-risk.
2. **LLM daily-spend cap tripped** — `LLM_DAILY_USD_CAP` was hit;
   the budget enforcer engages halt.
3. **LLM circuit-breaker tripped** — repeated provider failures
   chained-backoff to the kill switch.
4. **Drift-breach auto-halt** — `Wave 4.I` equity-curve detector
   landed an `alert` and the wiring (when it lands) engaged halt.

## Diagnose

```bash
uv run halal-trader halt-status
```

The output lists the active reason and the timestamp. Cross-check
with:

```bash
just logs-tail | grep halt.engaged
```

…to find the cycle that engaged it.

## Mitigate

1. **If operator-engaged + planned** — confirm the maintenance
   window is over, then:
   ```bash
   uv run halal-trader resume
   ```
2. **If LLM spend cap** — the cap resets at UTC midnight. Either
   wait, or raise `LLM_DAILY_USD_CAP` in `.env` if budget allows
   and restart the bot.
3. **If LLM circuit-breaker** — check
   `logs/error.log | grep llm.circuit` for the underlying
   provider failure. Once the provider is healthy:
   ```bash
   uv run halal-trader resume
   ```
4. **If drift breach** — read `logs/halal_trader.log | grep drift`
   and decide whether to re-train (Wave 6.A registry promotion)
   or temporarily lower the drift threshold. Don't blindly resume.

## Escalate

If the halt cause is unclear after 15 minutes, post in
`#halal-trader-oncall` with the `halt-status` output + the last
50 lines of `logs/error.log`.

## Postmortem

PAGE alert — file `docs/postmortems/<date>-halt-engaged.md`
within 48h. Include: trigger cause, time-to-detection,
time-to-mitigation, what would have prevented it.

---

_Last reviewed: 2026-05-01_
