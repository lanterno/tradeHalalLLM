# Security policy

This document is the project's security contract — what the bot
defends against, what's out of scope, and how to disclose a
vulnerability responsibly.

## TL;DR

* The bot is **paper-trade and testnet only by design**. No live
  money flows; the strongest guarantee is mechanical (the brokers
  can't reach mainnet with the default config).
* Send security reports to the address in [Disclosure](#disclosure)
  below — please don't open public GitHub issues for
  vulnerabilities.
* In-scope: anything that could leak operator secrets, bypass the
  halal screener, or escalate from a paper account to a live
  account. Out of scope: rate-limit shenanigans, social
  engineering, third-party dependency CVEs we haven't responded
  to within 90 days.

## Threat model

### What we're defending

The bot is a **single-operator paper-trading research tool** that
the operator runs on their own machine or VPS. The valuable
assets in scope:

| Asset | Defended via |
|---|---|
| Operator's broker API keys (Alpaca, Binance) | `.env` is gitignored; secrets never logged; read-only by default in WS adapters |
| LLM provider API keys (OpenAI, Anthropic, Ollama) | Same as above |
| Operator's Postgres credentials | `.env` only; `pg_hba.conf` localhost-only by default |
| Halal screening cache integrity | Append-only `HalalScreening` rows; consensus aggregator's STRICT-default rejects on any provider's `not_halal` |
| Audit trail integrity | `LlmDecision`, `Trade`, `HalalScreening` rows are append-only; Wave 6.E model fingerprint detects post-hoc tampering |
| Operator's local trading config | `Settings` is a singleton; reload requires restart |

### What we're explicitly NOT defending

* **Multi-tenant isolation.** The bot is single-operator. Wave 3
  scopes the multi-user platform; until that lands, treat each
  deployment as one trust boundary.
* **Side-channel attacks on a colocated cloud VM.** If the
  operator's neighbour on the same physical host can read CPU
  caches, the bot's process memory is no harder to read than any
  other.
* **Confidentiality of trade rationales.** The bot logs full LLM
  reasoning to disk for the audit trail (Wave 5.C makes them
  searchable). Operators who consider their rationales sensitive
  must encrypt the disk.
* **Denial of service from a determined external actor.** The
  bot has rate-limit handling, circuit-breakers, and the kill
  switch. None of these are anti-DDoS.

### Trust boundaries

Diagram:

```
              ┌─────────────────────────────────────────┐
              │  Operator's machine                      │
              │  ┌───────────┐   ┌──────────────┐       │
              │  │  Bot      │←─→│  Postgres    │       │
              │  │  process  │   │  (localhost) │       │
              │  └─────┬─────┘   └──────────────┘       │
              │        │                                 │
              │        ↓ HTTPS                          │
              └────────┼─────────────────────────────────┘
                       │
              ┌────────┴────────┬───────────────┐
              ↓                 ↓               ↓
         Alpaca API       Binance API      LLM provider
         (paper)         (testnet)        (OpenAI / etc.)
```

Each external service is a separate trust boundary; the bot
**must not** assume any of them are trustworthy. Defences:

* Every broker response is type-checked + range-validated before
  acting on it (`crypto/exchange.py`).
* LLM responses are JSON-schema-validated; an invalid response
  retries once with a "your previous response was invalid" repair
  prompt, then falls back to an empty plan rather than executing
  on garbage (`core/strategy.py`).
* Halal screener responses go through `halal/consensus.py` which
  applies the conservative-wins-on-ties rule.

## Secure defaults

Every default value in the codebase is the **safer** of any
two reasonable choices.

| Setting | Default | Why |
|---|---|---|
| `ALPACA_PAPER_TRADE` | `true` | Mainnet trading requires explicit operator opt-in |
| `BINANCE_TESTNET` | `true` | Same |
| `WEB_API_TOKEN` | empty | Empty token disables mutation endpoints (read-only) |
| `WEB_REQUIRE_CONFIRMATION` | `true` | Destructive ops require `X-Trader-Confirm: true` |
| `LLM_DAILY_USD_CAP` | `0.0` | Operator must explicitly cap to avoid runaway spend |
| `MIN_BUY_NOTIONAL_USD` | `50` | Below-min orders refused; documented in `tests/test_crypto_executor.py` |
| Halal `ConsensusPolicy` | `STRICT` | "Any rejection rejects" is the safest interpretation |
| Bot kill-switch | engaged on first error chain | `core/halt.py` engages before retry storm |

## Reporting a vulnerability

### Disclosure

Send security reports to:

```
ahmed.elghareeb@proton.me
```

Encrypt with the maintainer's PGP key (fingerprint published in
the repo's GitHub profile). Plain-text is acceptable for
low-severity findings.

Please include:

* The version / commit hash you tested against.
* A minimal reproducer (a script or a captured request).
* Your assessment of the impact.
* Whether you'd like credit in the disclosure.

### Response timeline

* **Within 72 hours** — acknowledgement of receipt.
* **Within 7 days** — initial triage with a severity classification.
* **Within 30 days for high-severity findings** — a fix landed
  in `main` with a coordinated disclosure date.
* **Within 90 days for any other finding** — a fix or a
  documented decision not to fix (with rationale published).

If a fix takes longer than 90 days, the maintainer commits to
publishing a status update and proposing a coordinated
disclosure window with the reporter.

### Bug bounty

The project does not currently run a paid bug bounty program.
Researchers are credited in the changelog (with consent) and in
a `SECURITY-HALL-OF-FAME.md` file.

## In-scope vs out-of-scope

### In scope

* **Secret extraction.** Anything that could read the operator's
  API keys, Postgres credentials, or LLM tokens from the running
  bot's memory, logs, or database.
* **Halal screener bypass.** A path that lets a `not_halal` symbol
  reach the executor — including consensus-aggregator subversion
  (e.g. a way to make `STRICT` policy approve a not_halal vote).
* **Halt-switch bypass.** A path that lets new entries proceed
  while the kill-switch is engaged.
* **Audit-trail tampering.** A way to modify or delete
  `LlmDecision` / `Trade` / `HalalScreening` rows that doesn't
  break the Wave 6.E fingerprint detection.
* **Cross-broker secret leakage.** Any path where an Alpaca-only
  cycle could send data to Binance or vice-versa.
* **Paper-to-live escalation.** Any path that lets the bot reach
  a *live* trading endpoint when the operator has the testnet
  defaults set.
* **Pre-auth code execution** on any process the bot starts
  (Postgres container, dashboard, MCP subprocess).

### Out of scope

* **Brute-force or credential-stuffing of operator-chosen
  passwords.** Operators are expected to set strong passwords.
* **Social engineering of the maintainer or reporters.**
* **Dependency CVEs published less than 30 days ago that we
  haven't yet responded to.** We track them via Dependabot but
  reasonable response time is the policy, not "instant".
* **Findings in third-party services we depend on** (Postgres,
  Ollama, Alpaca SDK, Binance SDK). Report those upstream;
  we'll ship the upgrade once a fix is available.
* **Rate-limit shenanigans** that don't lead to data loss or
  secret extraction. The bot's circuit-breakers are
  rate-limit-aware; degraded performance under hostile traffic
  is by design.
* **Issues that require the attacker to already be on the
  operator's machine** with root or equivalent privileges.
* **Theoretical timing or side-channel attacks on a shared
  cloud host.** Operators concerned about side-channels should
  run on dedicated hardware.

## Security-relevant features by wave

The Round-4 roadmap landed several pieces that compose into the
bot's security posture:

| Feature | Module | Wave |
|---|---|---|
| Per-trade signed receipt (Ed25519) | `halal/signing.py` | 2.A |
| Multi-source halal consensus | `halal/consensus.py` | 2.B |
| Per-scholar policy profiles | `halal/scholar_profiles.py` | 2.C |
| Equity-curve anomaly detector | `ml/equity_anomaly.py` | 4.I |
| Promotion gate (no regression) | `core/promotion_gate.py` | 4.F |
| Chaos engineering harness | `core/chaos.py` | 8.B |
| Multi-channel alert router | `core/alert_router.py` | 8.E |
| Model fingerprint (replay integrity) | `ml/fingerprint.py` | 6.E |
| Feature schema migration | `ml/feature_store.py` | 6.B |
| Operator runbooks | `docs/runbooks/` | 8.E + 9.D |

## Operator hardening checklist

Before deploying the bot beyond a single laptop:

* [ ] `.env` is on disk-encrypted storage (LUKS / FileVault).
* [ ] `.env` permissions are 0600 (`chmod 600 .env`).
* [ ] Postgres is bound to localhost or a private VPC subnet.
* [ ] The bot's host firewall allows only outbound HTTPS to the
      configured broker / LLM endpoints.
* [ ] `WEB_API_TOKEN` is set to a long random value if the
      dashboard is exposed beyond localhost.
* [ ] `LLM_DAILY_USD_CAP` is set to a non-zero ceiling.
* [ ] The kill-switch and halt status alerts route to a channel
      the operator monitors (Wave 8.E `core/alert_router.py`).
* [ ] The operator's signing keypair (Wave 2.A `halal/signing.py`)
      is backed up offline.
* [ ] If running on a public cloud, the VM doesn't run other
      workloads that could share memory pages.

## Halal-specific security considerations

The bot's compliance posture depends on the screener path.
Threats specific to that path:

* **Spoofed compliance verdicts.** A man-in-the-middle on the
  Zoya / IFG / Musaffa API could downgrade a `not_halal` to
  `halal`. Defences: HTTPS-only in `_is_placeholder` (Wave 5.G);
  multi-source consensus in STRICT mode (Wave 2.B); the audit
  row records the source, so a downstream review can detect a
  consistently-spoofed provider.
* **Cache poisoning.** A `CryptoHalalCache` row written with the
  wrong decision would propagate. Defences: cache TTL
  (`HALAL_CACHE_MAX_AGE_HOURS`); refresh on each cycle's
  ensure_cache; the screening's audit row preserves the original
  source so the scholar reviewer can spot a flipped row vs the
  upstream.
* **Operator override misuse.** An override that doesn't go
  through the exception queue (Section 7 of
  `docs/halal_jurisprudence.md`) could let a `doubtful` symbol
  silently trade. Defences: every override is audit-logged with
  the operator identifier and reason; the dashboard surfaces
  the override count per period.

## Deprecated / not implemented

For transparency, these security features are **not yet** in the
codebase:

* **Per-secret encryption at rest.** `.env` is plaintext;
  Wave 3.B (per-user encrypted secrets vault) is the planned
  replacement.
* **CSRF protection on mutation endpoints.** The current
  `WEB_REQUIRE_CONFIRMATION` header is a coarse defence;
  Wave 3.A (user accounts + auth) lands proper session-based
  CSRF.
* **Two-factor on dashboard mutation.** Single-token model today;
  Wave 3.A.

These limitations should be considered when deciding whether to
expose the dashboard beyond localhost.

---

_Last reviewed: 2026-05-01. Reviewed quarterly._
