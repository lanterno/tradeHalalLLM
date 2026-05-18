# `broker.api.error_rate` — broker error rate over threshold

**Severity:** PAGE
**Triggers when:** the broker (Binance / Alpaca) returns a non-2xx
status on >5% of calls in a 5-minute window.
**Acknowledgement window:** 10 minutes — broker outages produce
missed fills and stale prices fast.

## Likely causes

1. **Broker outage** — Binance / Alpaca status page reports an
   incident.
2. **Rate-limit hit** — Binance `-1003` or HTTP 429; the bot's
   per-pair circuit breaker should already be slowing requests.
3. **API key revoked / rotated** — auth-error 401s.
4. **Network partition** — DNS / TLS handshake failures.

## Diagnose

```bash
just logs-tail | grep "broker.error"
```

Look at the status codes:

* 401 / 403 → auth issue
* 429 / -1003 → rate limit
* 500 / 502 / 503 / 504 → upstream
* connection refused / TLS errors → network

```bash
curl -I https://www.binance.com/   # quick reachability check
```

## Mitigate

1. **If outage** — confirm with the broker's status page. Engage
   halt:
   ```bash
   uv run halal-trader halt --reason "broker outage"
   ```
   Resume after the broker reports recovery.
2. **If rate-limit** — the bot's circuit breaker should
   self-recover within 30s. If it doesn't, tighten cycle cadence
   in `.env` (raise `CRYPTO_CYCLE_INTERVAL_SECONDS`).
3. **If auth** — re-export keys, restart the bot.
4. **If network** — wait one minute, re-test. If sustained,
   engage halt and investigate from the host.

## Escalate

If the broker is up + keys are valid + network is fine and the
error rate persists, file a postmortem-blocking issue in
`#halal-trader-oncall`. Don't blindly resume.

## Postmortem

PAGE alert — file `docs/postmortems/<date>-broker-error-rate.md`
within 48h.

---

_Last reviewed: 2026-05-01_
