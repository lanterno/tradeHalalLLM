# `chain.backoff` — LLM provider chain in backoff

**Severity:** WARN
**Triggers when:** the `FallbackLLM` chain has tried every
configured provider and is now sleeping the exponential-backoff
window (60s → 30min after all fail).
**Acknowledgement window:** 30 minutes.

## Likely causes

1. **Single provider out** — primary OpenAI / Anthropic / Ollama
   is failing, fallbacks haven't been configured or are also down.
2. **All providers throttling** — same minute many cycles, all
   providers responded with 429.
3. **API-key rotation in progress** — keys rotated, env not
   reloaded.

## Diagnose

```bash
just logs-tail | grep llm.fallback
```

Each entry shows which provider failed with what error.

## Mitigate

1. **Single provider** — confirm with the provider's status page;
   if it's a known outage, add the alternates to
   `LLM_FALLBACK_PROVIDERS` in `.env`, restart the bot.
2. **All throttled** — wait one cycle. If sustained, lower
   `LLM_DAILY_USD_CAP` to slow the cycle's call rate, or pause
   the agentic-tool loop (`LLM_AGENT_ENABLED=false`).
3. **Key rotation** — re-export the new keys, restart the bot.

## Escalate

If the chain stays in backoff for >30 min and no provider is
responding, this likely indicates a billing problem (account
suspended / hard limit hit). Operator with billing access
escalates.

---

_Last reviewed: 2026-05-01_
