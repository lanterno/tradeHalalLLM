# `llm.circuit_breaker` — LLM circuit-breaker tripped

**Severity:** PAGE
**Triggers when:** the `FallbackLLM` chain has cycled through
every configured GLM endpoint without success and the
extended-backoff window (60s → 30min after all fail) has tripped
the bot's halt — the bot refuses new entries until an endpoint
recovers.
**Acknowledgement window:** 15 minutes.

## Likely causes

1. **All GLM endpoints down** — the OpenRouter primary and the
   `GLM_FALLBACK_BASE_URL` endpoint (if configured) are out at
   the same time (rare but possible during major cloud-provider
   incidents). Cycles degrade to no-action plans; the position
   monitor keeps enforcing SL/TP.
2. **Fallback endpoint mis-configured** — the endpoint set via
   `GLM_FALLBACK_BASE_URL` / `GLM_FALLBACK_MODEL` /
   `GLM_FALLBACK_API_KEY` is itself down or auth-broken, so the
   chain has no working option when the primary fails.
3. **Sustained cost-cap budget** — the daily LLM USD cap
   (`LLM_DAILY_USD_CAP`) has been exhausted; the budget enforcer
   has refused to start new cycles. Distinct from circuit-
   breaker but presents similarly.
4. **Quota throttling on every endpoint** — every endpoint
   simultaneously returned 429 / per-minute-quota error. Bot
   correctly waits, but if it persists across the full backoff
   window, the circuit breaker engages.

## Diagnose

```bash
just logs-tail | grep -E "llm\.fallback|llm\.circuit|llm\.budget"
```

Check the daily LLM cost:

```bash
uv run halal-trader llm cost-today  # if implemented
# else: query the LlmDecision table for sum(cost_usd) where
# timestamp >= today UTC midnight
```

Endpoint status:

```bash
# OpenRouter (primary GLM host)
# https://status.openrouter.ai
curl -fsS https://status.openrouter.ai/api/v2/summary.json | jq .
# Z.ai (if a Z.ai-direct fallback endpoint is configured)
# https://status.z.ai
curl -fsS https://status.z.ai/api/v2/summary.json | jq .
```

## Mitigate

1. **If all GLM endpoints down** — engage halt with reason "all
   GLM endpoints down; awaiting recovery". Resume when at least
   one endpoint's status page reports green:
   ```bash
   uv run halal-trader resume
   ```
2. **If fallback endpoint mis-configured** — confirm
   `GLM_FALLBACK_BASE_URL` / `GLM_FALLBACK_MODEL` /
   `GLM_FALLBACK_API_KEY` in `.env` point at a working host
   (e.g. Z.ai direct: base URL
   `https://api.z.ai/api/paas/v4`, model `glm-5.2`) and that
   the key is valid.
3. **If cost cap hit** — either wait for UTC midnight reset or
   raise `LLM_DAILY_USD_CAP` if budget allows. The cap engages
   the kill switch (halt) so the bot will refuse new entries
   even after the LLM recovers — `halal-trader resume` after
   confirming the cap is no longer the issue.
4. **If quota throttling** — slow the cycle cadence
   (`CRYPTO_CYCLE_INTERVAL_SECONDS=120` halves the call rate);
   restart the bot.
5. **If only the primary endpoint is out** — the fallback
   endpoint takes over automatically. If no fallback is
   configured, point `GLM_BASE_URL` / `LLM_MODEL` at a working
   host (or set the `GLM_FALLBACK_*` trio) in `.env` and
   restart the bot.

## Escalate

If the circuit stays tripped for >30 min and the endpoints'
status pages report green, this is a sign of the bot's local
HTTP client being misconfigured (proxy, TLS, DNS). Restart the
bot with verbose logging:

```bash
HTTPX_LOG_LEVEL=DEBUG uv run halal-trader crypto start
```

…and inspect the raw HTTP errors.

## Postmortem

PAGE alert — file `docs/postmortems/<date>-llm-circuit.md`
within 48h.

---

_Last reviewed: 2026-05-01_
