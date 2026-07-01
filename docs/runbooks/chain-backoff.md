# `chain.backoff` — LLM endpoint chain in backoff

**Severity:** WARN
**Triggers when:** the `FallbackLLM` chain has tried every
configured GLM endpoint and is now sleeping the
exponential-backoff window (60s → 30min after all fail).
**Acknowledgement window:** 30 minutes.

## Likely causes

1. **Primary GLM endpoint out** — the OpenRouter endpoint is
   failing, and a fallback endpoint hasn't been configured or is
   also down.
2. **All endpoints throttling** — same minute many cycles, every
   endpoint responded with 429.
3. **API-key rotation in progress** — keys rotated, env not
   reloaded.

## Diagnose

```bash
just logs-tail | grep llm.fallback
```

Each entry shows which endpoint failed with what error.

## Mitigate

1. **Primary endpoint out** — confirm with the status page
   (<https://status.openrouter.ai>, or <https://status.z.ai> for
   a Z.ai endpoint); if it's a known outage, set
   `GLM_FALLBACK_BASE_URL` (+ `GLM_FALLBACK_MODEL` /
   `GLM_FALLBACK_API_KEY`) in `.env`, restart the bot.
2. **All throttled** — wait one cycle. If sustained, lower
   `LLM_DAILY_USD_CAP` to slow the cycle's call rate, or pause
   the agentic-tool loop (`LLM_AGENT_ENABLED=false`).
3. **Key rotation** — re-export the new keys, restart the bot.

## Escalate

If the chain stays in backoff for >30 min and no endpoint is
responding, this likely indicates a billing problem (OpenRouter
credits exhausted / account suspended / hard limit hit).
Operator with billing access escalates.

---

_Last reviewed: 2026-05-01_
