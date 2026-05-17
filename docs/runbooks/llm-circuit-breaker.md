# `llm.circuit_breaker` — LLM circuit-breaker tripped

**Severity:** PAGE
**Triggers when:** the `FallbackLLM` chain has cycled through
every configured provider without success and the
extended-backoff window (60s → 30min after all fail) has tripped
the bot's halt — the bot refuses new entries until a provider
recovers.
**Acknowledgement window:** 15 minutes.

## Likely causes

1. **All providers down** — coincidental outages across
   primary + fallback (rare but possible during major cloud-
   provider incidents).
2. **Single provider mis-configured** — fallbacks listed in
   `LLM_FALLBACK_PROVIDERS` are themselves down or auth-broken
   so the chain has no working option.
3. **Sustained cost-cap budget** — the daily LLM USD cap
   (`LLM_DAILY_USD_CAP`) has been exhausted; the budget enforcer
   has refused to start new cycles. Distinct from circuit-
   breaker but presents similarly.
4. **Quota throttling on every provider** — every provider
   simultaneously returned 429 / per-minute-quota error. Bot
   correctly waits, but if it persists across the full backoff
   window, the circuit breaker engages.
5. **Local Ollama died** — primary `LLM_PROVIDER=ollama`
   but the local server is down.

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

Provider status:

```bash
# Primary: hit each provider's status page
curl -fsS https://status.openai.com/api/v2/summary.json | jq .
curl -fsS https://status.anthropic.com/api/v2/summary.json | jq .
# Local Ollama
curl -fsS http://localhost:11434/api/tags
```

## Mitigate

1. **If all providers down** — engage halt with reason "all
   LLMs down; awaiting recovery". Resume when at least one
   provider's status page reports green:
   ```bash
   uv run halal-trader resume
   ```
2. **If fallback chain mis-configured** — confirm
   `LLM_FALLBACK_PROVIDERS` in `.env` has at least 2 working
   alternates. Test each:
   ```bash
   uv run halal-trader llm test --provider openai --model gpt-4o-mini
   ```
3. **If cost cap hit** — either wait for UTC midnight reset or
   raise `LLM_DAILY_USD_CAP` if budget allows. The cap engages
   the kill switch (halt) so the bot will refuse new entries
   even after the LLM recovers — `halal-trader resume` after
   confirming the cap is no longer the issue.
4. **If quota throttling** — slow the cycle cadence
   (`CRYPTO_CYCLE_INTERVAL_SECONDS=120` halves the call rate);
   restart the bot.
5. **If local Ollama** — `ollama serve` in another terminal,
   or switch to OpenAI/Anthropic via `LLM_PROVIDER` in `.env`.

## Escalate

If the circuit stays tripped for >30 min and the providers'
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
