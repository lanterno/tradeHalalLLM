# Strategy author's guide

This guide is for operators or contributors who want to **write a
custom trading strategy** for halal-trader — either as a one-off
research experiment, an A/B prompt test, or a long-lived production
strategy. It covers the contract every strategy must satisfy, the
prompt-version registry, the testing patterns the project uses, and
ends with a fully-worked example that runs end-to-end against the
existing backtest harness.

If you just want to *run* the bot, read [`QUICKSTART.md`](QUICKSTART.md)
first — this guide assumes you can already start the bot and watch a
cycle.

## What is a "strategy" in this codebase?

A strategy is a function (in practice, a method on a class) that
turns the per-cycle data the bot collects — klines, indicators,
sentiment, regime label, open positions — into a `CryptoTradingPlan`
or `TradingPlan` (`pydantic` models defined under
`domain/models.py`). Every per-cycle plan carries a list of
per-symbol decisions: `BUY` / `SELL` / `HOLD`, plus a confidence
score and a written rationale.

**The strategy chooses; the executor enforces.** The strategy never
calls a broker, never signs a halal screening, never tweaks risk
parameters mid-cycle. It produces a plan; downstream layers
(`core/safeguards.py`, `core/sizing.py`, `crypto/executor.py`) check
it against the kill switch, the per-pair circuit breaker, the
sector cap, the halal screen, the LLM-cost cap, and so on.

## The base contract — `BaseStrategy`

Every strategy ultimately extends
[`core/strategy.py:BaseStrategy`](../src/halal_trader/core/strategy.py).
The base class owns the orchestration:

* Building the system prompt and the per-cycle user prompt.
* Calling the LLM with the right backend and budget.
* Parsing the response as JSON, validating against your schema,
  and on failure retrying once with a "your previous response was
  invalid because X — fix it" repair instruction.
* Recording the decision (prompt summary, raw response, parsed
  action, token counts, cost, prompt version) to the
  `llm_decisions` Postgres table.

You override two things:

| Method | Purpose |
|---|---|
| `_build_prompts(...)` | Return `(system_prompt, user_prompt, prompt_summary)`. |
| `_validate_plan(raw_dict)` → `Plan` | Parse + validate the LLM's JSON response into a typed plan. |

For most cases, you'll subclass `CryptoTradingStrategy` or
`StockTradingStrategy` (which already implement `_validate_plan`)
and only override the prompt builder. That's the path the worked
example takes below.

### Schema — the plan you must return

The crypto strategy returns a `CryptoTradingPlan`:

```python
class CryptoTradingPlan(BaseModel):
    decisions: list[CryptoDecision]     # one entry per pair
    market_outlook: str = ""

class CryptoDecision(BaseModel):
    pair: str                            # e.g. "BTCUSDT"
    action: TradeAction                  # BUY | SELL | HOLD
    confidence: float                    # 0..1
    quantity: float                      # in base-currency
    reasoning: str                       # operator-readable
```

The LLM **must** return JSON that parses into this. The base class
will retry once on validation failure; if the retry also fails, it
returns an empty plan and logs the failure — your strategy doesn't
have to handle the error path itself.

## Prompt versioning — `core/llm/prompts/`

Every prompt template you write **must register a `PromptVersion`**
at import time. The registry computes a stable short SHA over the
template body, so every `LlmDecision` row records exactly which
prompt version produced it. Editing a template mid-week and
forgetting to bump the version is impossible: the hash changes the
moment the bytes change.

```python
from halal_trader.core.llm.prompts import register

MY_PROMPT = register(
    name="crypto.strategy.my_custom",
    template="""You are a halal crypto trader.
Available pairs: {pairs}
Latest prices: {prices}
…
""",
)
# At call time:
user_prompt = MY_PROMPT.template.format(pairs=…, prices=…)
prompt_version = MY_PROMPT.version  # "crypto.strategy.my_custom@7f3a"
```

Why this matters in production:

* The `LlmDecision.prompt_version` column lets the **A/B comparator**
  (`core/ab_compare.py`) split a window of trades by prompt version
  and Welch's-t-test the difference. You can't do this if you
  silently rewrite the prompt without bumping versions.
* The **prompt-evolution GA** (`core/llm/prompt_evo_runner.py`)
  stores measured fitness against the prompt version — same reason.

## Per-symbol filtering — halal compliance

**Your strategy must NOT trade non-halal symbols, full stop.** The
codebase enforces this by passing only the *already-screened*
symbols into the strategy:

```python
# In crypto/cycle.py:
halal_pairs = await self._screener.get_halal_pairs()
plan = await self._strategy.analyze(halal_pairs, …)
```

So the strategy receives a pre-filtered list. Don't call a broker
or screen an arbitrary symbol from inside the strategy — that path
bypasses the audit trail and the cache.

If you want a stricter cohort than what the default screener
allows, layer your own filter on top *before* calling the LLM, but
let the screener's halal output remain the upstream boundary.

If you need to apply a *scholar profile* (Wave 2.C — see
[`halal/scholar_profiles.py`](../src/halal_trader/halal/scholar_profiles.py)),
do it as a downstream check on a `BUY` decision, not as a
substitute for the upstream screener.

## Testing your strategy

The project ships four test harnesses. Use all four during
development; promote to live only after all four pass.

### 1. Unit tests on the strategy class

`tests/test_crypto_strategy.py` is the canonical example. The
pattern: mock the `LLMBackend` to return a canned JSON response,
call `analyze`, assert the plan's decisions match what you expect.

```python
from unittest.mock import AsyncMock
mock_llm = AsyncMock()
mock_llm.generate.return_value = '{"decisions": [{...}], …}'
strategy = MyStrategy(mock_llm, repo, …)
plan = await strategy.analyze(["BTCUSDT"], indicators, …)
assert plan.decisions[0].action == TradeAction.BUY
```

### 2. Stress harness

`crypto/stress.py` ships a battery of synthetic kline scenarios
your strategy **must** pass before promote-to-live. The eight
canned scenarios (Round-4 wave 7.B added the last three):

| Scenario | What it tests |
|---|---|
| `flash_crash` | Don't buy a 15% drop in 3 bars. |
| `blow_off_pump` | Don't size up at the parabolic top. |
| `gap_down` | Don't blindly buy the gap. |
| `illiquid_drift` | Mostly hold; no edge. |
| `sustained_downtrend` | No counter-trend buys. |
| `regime_shift` | De-risk after a vol jump. |
| `volatility_explosion` | Don't mistake range for trend. |
| `liquidity_crunch` | Refuse to trade in wide-bar regime. |

```python
from halal_trader.crypto.stress import (
    standard_scenarios, evaluate_scenarios, render_report,
)
async def my_strategy_call(klines):
    return await strategy.analyze(["TEST"], {…}, …)
verdicts = await evaluate_scenarios(my_strategy_call, standard_scenarios())
print(render_report(verdicts))
# All 8 must report severity < 0.5.
```

### 3. Scenario simulator (Wave 5.F)

`core/scenario_sim.py` projects a list of *currently-open
positions* through a kline path. Use this to ask "if my strategy
opens these three positions and then a flash-crash hits, what
happens?". The simulator runs SL/TP enforcement bar-by-bar so you
see exactly which positions trip and which survive.

### 4. A/B comparator (Wave 5.B)

After running the strategy live for ≥100 trades, use
`core/ab_compare.compare(returns_old, returns_new)` to verify the
new strategy's per-trade returns are statistically distinguishable
from the previous prompt version. The Welch's-t-test p-value is
the gate — don't promote on a p > 0.05.

## A worked example — RSI mean-reversion

Goal: a strategy that buys when RSI < 30 and sells when RSI > 70,
otherwise holds. Pure-deterministic (no LLM) — useful as a
research baseline against which the LLM strategy must outperform.

This is the file structure:

```
src/halal_trader/strategies/
  rsi_mean_reversion.py    # the strategy class
tests/
  test_rsi_mean_reversion.py
```

### The strategy class

```python
"""Pure-deterministic RSI mean-reversion strategy.

No LLM call. Useful as a research baseline:
* Sets a floor on what the LLM strategy must beat to justify cost.
* Trivially reproducible on any historical kline window.
"""
from __future__ import annotations

from halal_trader.domain.models import (
    CryptoDecision, CryptoTradingPlan, TradeAction,
)


class RsiMeanReversionStrategy:
    """Pure-Python; no LLM, no DB call. Just RSI thresholds."""

    def __init__(
        self,
        *,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        default_qty: float = 0.001,
    ) -> None:
        if oversold_threshold >= overbought_threshold:
            raise ValueError("oversold must be below overbought")
        self._oversold = oversold_threshold
        self._overbought = overbought_threshold
        self._qty = default_qty

    async def analyze(
        self,
        pairs: list[str],
        indicators: dict[str, dict[str, float]],
    ) -> CryptoTradingPlan:
        decisions: list[CryptoDecision] = []
        for pair in pairs:
            ind = indicators.get(pair, {})
            rsi = ind.get("rsi_14")
            if rsi is None:
                action = TradeAction.HOLD
                reasoning = "RSI unavailable; skip."
                confidence = 0.0
            elif rsi < self._oversold:
                action = TradeAction.BUY
                reasoning = f"RSI {rsi:.1f} < {self._oversold}; oversold."
                confidence = 0.7
            elif rsi > self._overbought:
                action = TradeAction.SELL
                reasoning = f"RSI {rsi:.1f} > {self._overbought}; overbought."
                confidence = 0.7
            else:
                action = TradeAction.HOLD
                reasoning = f"RSI {rsi:.1f} in neutral zone."
                confidence = 0.0
            decisions.append(CryptoDecision(
                pair=pair, action=action, confidence=confidence,
                quantity=self._qty, reasoning=reasoning,
            ))
        return CryptoTradingPlan(
            decisions=decisions,
            market_outlook="Mechanical RSI mean-reversion — no narrative.",
        )
```

### The unit tests

```python
import pytest
from halal_trader.strategies.rsi_mean_reversion import RsiMeanReversionStrategy
from halal_trader.domain.models import TradeAction


@pytest.mark.asyncio
async def test_buys_when_rsi_oversold():
    s = RsiMeanReversionStrategy()
    plan = await s.analyze(["BTCUSDT"], {"BTCUSDT": {"rsi_14": 25.0}})
    assert plan.decisions[0].action == TradeAction.BUY


@pytest.mark.asyncio
async def test_sells_when_rsi_overbought():
    s = RsiMeanReversionStrategy()
    plan = await s.analyze(["BTCUSDT"], {"BTCUSDT": {"rsi_14": 75.0}})
    assert plan.decisions[0].action == TradeAction.SELL


@pytest.mark.asyncio
async def test_holds_when_rsi_neutral():
    s = RsiMeanReversionStrategy()
    plan = await s.analyze(["BTCUSDT"], {"BTCUSDT": {"rsi_14": 50.0}})
    assert plan.decisions[0].action == TradeAction.HOLD


@pytest.mark.asyncio
async def test_holds_when_rsi_unavailable():
    """Don't trade on missing data — pin the safety default."""
    s = RsiMeanReversionStrategy()
    plan = await s.analyze(["BTCUSDT"], {"BTCUSDT": {}})
    assert plan.decisions[0].action == TradeAction.HOLD


def test_constructor_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        RsiMeanReversionStrategy(oversold_threshold=70, overbought_threshold=30)
```

### Running it through the stress harness

A non-LLM strategy doesn't have klines as a direct input — it
operates on indicators. So wrap it:

```python
from halal_trader.crypto.indicators import compute_indicators

async def my_strategy_call(klines):
    indicators = {"TEST": compute_indicators(klines)}
    plan = await strategy.analyze(["TEST"], indicators)
    return plan
```

…then feed `my_strategy_call` to `evaluate_scenarios`. A
mechanical RSI strategy will fail the `flash_crash` scenario
(RSI dives below 30 right at the bottom) — that's the value of
the harness: it proves the strategy has a known weakness, which
the operator can decide whether to accept or guard against
upstream.

### Backtesting

```bash
uv run halal-trader crypto backtest \
  --pair BTCUSDT --candles 1000 --strategy rsi_mean_reversion
```

(See `crypto/backtest.py` for how the bot wires strategies into
the backtest CLI. Adding a new strategy to the CLI registry is one
line per strategy.)

## Promotion checklist

Before promoting a strategy to live testnet trading:

* [ ] Unit tests pass (`uv run pytest tests/test_my_strategy.py -q`).
* [ ] All 8 stress scenarios pass with severity < 0.5.
* [ ] Backtest on ≥1000 candles produces a Sharpe > 0.5 and a
      profit factor > 1.2.
* [ ] Lint + format clean (`uv run ruff check src/halal_trader/strategies/...`).
* [ ] Type check clean for any new code under `domain/` or `core/`
      (`uv run mypy src/halal_trader/strategies/...`).
* [ ] Halal compliance: the strategy must not call any broker,
      screener, or external service directly — it operates only on
      the inputs the cycle hands in.
* [ ] Prompt registered with `PromptVersion` if it uses an LLM, and
      every change to the template bumps the version hash.

## Common pitfalls

**Don't open new files mid-strategy.** Reading from disk inside
`analyze` blocks the cycle's event loop; use `BaseStrategy`'s
constructor to load any static data once.

**Don't catch exceptions from the LLM call.** The base class
already wraps them in the validate-then-retry-then-fail-soft
loop. Catching them yourself just hides failures from the audit
trail.

**Don't mutate the input lists.** `analyze(pairs, indicators)`
gets shared references — your strategy must not reorder or
extend them.

**Don't introduce new sources of randomness without a seed.**
Reproducibility is a hard requirement for the A/B comparator and
the regression tests. If you need randomness, take a seed in the
constructor.

**Don't use real money.** The bot is paper-trade only by design.
`ALPACA_PAPER_TRADE=true` and `BINANCE_TESTNET=true` are pinned in
`.env.example`; do not flip them.

## Where to read next

* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — the full system map.
* [`docs/halal_jurisprudence.md`](halal_jurisprudence.md) — the
  rulings every strategy must respect.
* [`src/halal_trader/crypto/strategy.py`](../src/halal_trader/crypto/strategy.py) —
  the production crypto strategy. The cleanest reference
  implementation.
* [`src/halal_trader/trading/strategy.py`](../src/halal_trader/trading/strategy.py) —
  the production stocks strategy.
* [`src/halal_trader/core/llm/prompts/registry.py`](../src/halal_trader/core/llm/prompts/registry.py) —
  the prompt registry's full API.
* [`src/halal_trader/crypto/stress.py`](../src/halal_trader/crypto/stress.py) —
  the stress harness; the eight canned scenarios + their graders.
* [`src/halal_trader/core/scenario_sim.py`](../src/halal_trader/core/scenario_sim.py) —
  the scenario simulator (Wave 5.F).
* [`src/halal_trader/core/ab_compare.py`](../src/halal_trader/core/ab_compare.py) —
  the A/B comparator (Wave 5.B).
