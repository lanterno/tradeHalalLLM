# Session handoff — updated 2026-07-02 ~02:15 CEST

**UPDATE**: GLM_API_KEY is set (OpenRouter, $10 credit). Live smoke test
PASSED all three hot-path compat points (json_object 1.5s, forced
tool_choice honoring schema 2.5s, classifier@temp0 0.8s, ~$0.0005
total). Stocks bot restarted on the GLM code at 02:06 CEST — startup
clean (`LLM: GLM z-ai/glm-5.2 via https://openrouter.ai/api/v1`),
idle until Thu 2026-07-02 09:30 ET open. First LIVE market cycle on GLM
still needs eyeballing at open. NEW FINDING: the crypto launchd daemon
(com.halabot.crypto, running since Jun 12 on pre-cutover code) fails
every 60s cycle since 07-01 18:34 CEST — Binance API keys are EMPTY in
.env (empty before the cutover too; not caused by it). Operator
decision pending: supply testnet keys + restart crypto, or stop the
daemon. Do not disable unilaterally.

# Original handoff — 2026-07-01 (evening ET)

Briefing for continuing the working session on another machine. Give this
file to Claude Code at the start of the new session ("read
docs/SESSION_HANDOFF.md and resume"). Delete or update it when stale.

## What just happened (this session)

1. **EOD trading review (Wed 2026-07-01)** — first full day of restored
   trading after the MCP-envelope outage fix (`a499daf`). Clean day:
   ~15 cycles, equity $98,056, holding AAPL 50 @ $294.33 and INTU 18 @
   $270.66. Midday OpenAI 429 self-recovered via backoff. Concentration
   caps verified working (rejected over-buying AAPL at 15% equity).

2. **GLM-5.2 research** (fact-checked multi-agent sweep) — verdict:
   GPT-5.5-parity quality at ~1/5 price via OpenRouter; Z.ai direct API
   and the GLM Coding Plan are NOT suitable (outages / ToS). Full
   verdict in the memory file `glm-52-provider-evaluation.md` (local to
   the original machine) and summarized in commit `d42c5aa`'s context.

3. **GLM-5.2 hard cutover (user-directed)** — commit **`d42c5aa`**,
   pushed to main. GLM-5.2 is now the SOLE LLM provider; OpenAI,
   Anthropic, and Ollama were removed entirely (code, settings, deps,
   docs, tests). Gates at commit time: ruff clean, mypy strict clean,
   3,362 tests passed.

## ⚠ Blocking operator action

**The bot cannot start until `GLM_API_KEY` is set in `.env`.**

1. Create a key at https://openrouter.ai/keys and add credits.
2. In `.env` set:
   ```
   GLM_API_KEY=<openrouter key>
   GLM_BASE_URL=https://openrouter.ai/api/v1
   LLM_MODEL=z-ai/glm-5.2
   ```
   (remove any old `LLM_PROVIDER`, `OPENAI_*`, `ANTHROPIC_*`, `OLLAMA_*`,
   `LLM_FALLBACK_PROVIDERS` lines — see `.env.example` for the new
   block, including optional `GLM_FALLBACK_*` second-endpoint chain,
   `GLM_TIMEOUT_SECONDS=60`, `GLM_THINKING=false`,
   `GLM_REQUIRE_PARAMETERS=true`).
3. Restart: `just launchd-restart-stocks` (on the machine that runs the
   bot — the original machine's launchd process is still running the
   pre-cutover code from memory and keeps trading until restarted).

**First-cycle smoke test after restart**: watch one full cycle in
`logs/halal_trader.log`. The load-bearing compat points are forced
`tool_choice` (`submit_decisions`) and JSON mode; the failure mode is a
silent no-action plan (`LLM returned no tool calls`), not a crash. If
that appears, suspect the OpenRouter host routing — `GLM_REQUIRE_PARAMETERS=true`
should prevent it.

Note: on the ORIGINAL machine, the pre-cutover `.env` (with the old
OpenAI key) is backed up at `.env.pre-glm.bak` — gitignored, never
pushed. Delete it once confident.

## Standing loops (session-local — re-arm on the new machine if wanted)

- **Trading-review loop** (`/loop`, dynamic): after each stocks cycle /
  scored reactor event, review logs, write a 2-5 bullet improvement
  plan, implement actionable items, lint+typecheck+test+commit, restart
  via `just launchd-restart-stocks`. Pace ~15 min during 9:30–15:45 ET,
  once ~16:00 ET EOD, sleep otherwise.
- **Roadmap-build cron** (was `13,43 * * * *`): pick the next unchecked
  item in `docs/ROADMAP.md`, build a tested+committed slice, check it
  off. Off-market = safe build window.

## Standing constraints (operator-set, always in force)

- Halal non-negotiable: long-only, no short/interest/leverage/derivatives.
- Paper/testnet only. Never real money.
- Do NOT run `reconcile fix-drift --apply` or modify fix-drift/reconcile
  coupling unilaterally (destructive, operator-gated).
- Never change live trading/sizing paths without offline validation;
  advisory + infra + validation work preferred for autonomous slices.
- Keep stock/crypto cycle logic separate. `src/halabot/` shadow engine
  stays execution-DORMANT.
- Dev mode: autonomous sustained progress, commit/push to main directly.

## Open operator items (not code-fixable)

1. **GLM_API_KEY** (above) — new, blocking.
2. **Reconcile drift ~225/day** from phantom DB positions left by the
   outage — needs operator-run `halal-trader reconcile fix-drift`.
3. **Zoya sandbox universe** (3/20 halal, randomized) causes symbol
   fixation (AAPL/ADBE/INTU) — needs paid Zoya prod key +
   `ZOYA_USE_SANDBOX=false`.
4. ~~OpenAI quota~~ — obsolete after the GLM cutover; OpenRouter credits
   are the new equivalent (watch for 402 "Insufficient credits").

## Where the roadmap stands

See `docs/ROADMAP.md` (authoritative, checkboxes current). Recently done:
what-if simulator (Phase 4), FinBERT classifier (Phase 2), factor core +
IC/ICIR harness (Phase 2), sizing primitives (Phase 1, live wiring
deferred pending backtest evidence). Next candidates: halal-screening
freshness gate (Phase 0 leftover), SEC EDGAR Form-4 clustering,
sentence-transformer embeddings (friction: pgvector dim 512 vs MiniLM
384), dashboard equity chart for the what-if curve.

Deferred telemetry follow-up: strategy-LLM quota-exhaustion AlertSink
wiring (FallbackLLM logs "All LLM providers failed" but fires no
Telegram alert, unlike the classifier's quota breaker).

## Suggested first message on the new machine

> Read docs/SESSION_HANDOFF.md. I've set GLM_API_KEY — restart the bot,
> verify the first GLM cycle end to end, then resume the trading-review
> loop.
