# `halal.screener.stale` — halal cache age past threshold

**Severity:** WARN
**Triggers when:** the halal cache (`halal/cache.py` for
crypto, the Zoya layer for stocks) hasn't refreshed in
`HALAL_CACHE_MAX_AGE_HOURS` (default 6h). Cycles continue but
the symbols they screen against may be stale.
**Acknowledgement window:** 60 minutes — a stale cache rarely
flips a decision, but operators must know it's out of date.

## Likely causes

1. **Screener API down.** Zoya / IFG / Musaffa / CoinGecko
   returning errors; the cache refresher swallows and waits.
2. **Auth broken.** Zoya sandbox key expired; `ZOYA_API_KEY`
   needs rotating.
3. **Rate limit hit.** CoinGecko free tier (10-30 req/min)
   doesn't keep up with the cache refresh cadence; the cache
   refresher backs off and the freshness check trips before
   it catches up.
4. **Refresh task crashed silently.** Background coroutine
   raised an exception that wasn't caught by the alert sink.
   Rare but possible.

## Diagnose

```bash
just logs-tail | grep -E "halal\.cache|halal\.refresh|zoya"
```

For crypto:

```bash
# When did the cache last refresh?
docker exec -it $(docker compose ps -q postgres) \
  psql -U trader -d halal_trader \
  -c "SELECT symbol, compliance, last_updated
      FROM crypto_halal_cache
      ORDER BY last_updated DESC LIMIT 5"
```

For stocks:

```bash
# Same shape on the Zoya cache table.
docker exec -it $(docker compose ps -q postgres) \
  psql -U trader -d halal_trader \
  -c "SELECT symbol, decision, last_updated
      FROM stock_halal_cache
      ORDER BY last_updated DESC LIMIT 5"
```

Check the screener's reachability:

```bash
# CoinGecko
curl -fsS "https://api.coingecko.com/api/v3/ping"

# Zoya sandbox
curl -fsS "https://api.zoya.finance/sandbox/health" \
  -H "X-Api-Key: $ZOYA_API_KEY"
```

## Mitigate

1. **If screener API down** — wait one cycle; the bot's
   STRICT consensus (Wave 2.B) and the conservative-default
   decision (`doubtful` on no-data) keep the operator safe.
   If the API is down for >2 hours and the cache age exceeds
   24h, engage halt with reason "halal screener extended
   outage" until the upstream recovers.
2. **If auth broken** — rotate the key in `.env`, restart
   the bot. The next cycle will re-attempt the refresh and
   the staleness alert clears once the cache lands.
3. **If rate-limited (CoinGecko)** — set `COINGECKO_API_KEY`
   in `.env` for the higher-tier rate limit. Or raise
   `HALAL_CACHE_MAX_AGE_HOURS` to a value the free tier can
   sustain (12h or 24h is reasonable for a small universe).
4. **If refresher crashed silently** — restart the bot.
   Check the logs for the cycle that raised the unhandled
   exception (`grep "halal.cache" logs/error.log`).

## Pin

The bot's defence-in-depth assumes a stale cache is **safer
than no cache**: the `STRICT` consensus default (Wave 2.B)
already rejects on any provider's `not_halal` vote and any
unknown symbol stays `doubtful` until refreshed. So a stale
cache mostly affects *coverage* of the universe (newly-listed
tokens won't appear), not *correctness* of decisions on cached
symbols.

## Escalate

If the cache stays stale for > 24h despite the screener API
being up, this is a code-side issue with the refresher loop.
Open an issue with the cache-age query output + the relevant
log lines.

---

_Last reviewed: 2026-05-01_
