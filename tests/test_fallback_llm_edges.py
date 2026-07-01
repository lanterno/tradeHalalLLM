"""Edge-case tests for :class:`FallbackLLM`.

`test_fallback_llm.py` covers the happy paths (primary success,
fallback rotation, recovery, chain-backoff growth/clear, no-fallbacks).
This file pins the remaining branches: quota-error fast-track to
backoff (1 failure vs 3 for normal errors), exponential primary
backoff with the 30-minute cap, primary-in-backoff window skip,
``last_usage`` / ``last_thinking`` propagation from the chosen
provider, and the parallel ``generate_json`` method.
"""

from __future__ import annotations

import time

import pytest

from halal_trader.core.llm import BaseLLM, FallbackLLM


class _StubLLM(BaseLLM):
    """Controllable BaseLLM with usage/thinking propagation."""

    def __init__(
        self,
        name: str,
        *,
        fail_n: int = 0,
        fail_with: type[Exception] = RuntimeError,
        fail_msg: str = "stub failure",
        usage: object | None = None,
        thinking: str | None = None,
        json_response: dict | None = None,
    ) -> None:
        super().__init__(model=name)
        self.calls = 0
        self.json_calls = 0
        self._fail_n = fail_n
        self._fail_with = fail_with
        self._fail_msg = fail_msg
        self.last_usage = usage  # type: ignore[assignment]
        self.last_thinking = thinking
        self._json_response = json_response or {}

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        if self._fail_n > 0:
            self._fail_n -= 1
            raise self._fail_with(self._fail_msg)
        return f"{self.model}:{prompt}"

    async def generate_json(self, prompt: str, system: str | None = None) -> dict:
        self.json_calls += 1
        if self._fail_n > 0:
            self._fail_n -= 1
            raise self._fail_with(self._fail_msg)
        return self._json_response


# ── quota-error fast track ─────────────────────────────────


@pytest.mark.asyncio
async def test_quota_error_arms_backoff_after_one_failure():
    """A 429 / insufficient_quota error from the primary should arm the
    backoff *immediately* (threshold=1) — keep retrying a quota-exhausted
    key just burns time. Normal errors take 3 failures."""
    primary = _StubLLM("primary", fail_n=1, fail_msg="HTTP 429: rate limit hit")
    fb = _StubLLM("fb")
    chain = FallbackLLM(primary, fallbacks=[fb])

    await chain.generate("x")  # primary fails → fb serves
    # Quota error → backoff armed even though only one failure.
    assert chain._consecutive_failures == 1
    assert chain._backoff_until > 0


@pytest.mark.asyncio
async def test_insufficient_quota_message_also_fast_tracks():
    """The matcher accepts either "429" or "insufficient_quota" anywhere
    in the error string — GLM endpoints (OpenRouter vs Z.ai direct)
    phrase quota errors differently."""
    primary = _StubLLM("primary", fail_n=1, fail_msg="error: insufficient_quota")
    fb = _StubLLM("fb")
    chain = FallbackLLM(primary, fallbacks=[fb])
    await chain.generate("x")
    assert chain._backoff_until > 0


@pytest.mark.asyncio
async def test_normal_error_does_not_arm_backoff_until_third_fail():
    """Generic RuntimeErrors take 3 consecutive failures before backoff
    arms — transient blips shouldn't disable the primary."""
    primary = _StubLLM("primary", fail_n=10, fail_msg="connection reset")
    fb = _StubLLM("fb")
    chain = FallbackLLM(primary, fallbacks=[fb])

    # Fail 1: counter increments, no backoff.
    await chain.generate("a")
    assert chain._consecutive_failures == 1
    assert chain._backoff_until == 0
    # Fail 2: still no backoff.
    await chain.generate("b")
    assert chain._consecutive_failures == 2
    assert chain._backoff_until == 0
    # Fail 3: backoff arms.
    await chain.generate("c")
    assert chain._consecutive_failures == 3
    assert chain._backoff_until > 0


# ── primary backoff exponent ────────────────────────────────


@pytest.mark.asyncio
async def test_primary_backoff_grows_exponentially_with_cap(monkeypatch):
    """Beyond the threshold, backoff doubles per failure but caps at
    `_max_backoff_minutes` (30 min). Verify the doubling and the cap."""
    primary = _StubLLM("p", fail_n=20, fail_msg="generic error")
    fb = _StubLLM("f")
    chain = FallbackLLM(primary, fallbacks=[fb])

    fake_now = [time.monotonic()]
    monkeypatch.setattr("halal_trader.core.llm.fallback.time.monotonic", lambda: fake_now[0])

    backoffs: list[float] = []
    for _ in range(8):
        await chain.generate("x")
        backoffs.append(chain._backoff_until - fake_now[0])
        # Skip past the primary backoff window so the next call hits primary again.
        fake_now[0] = chain._backoff_until + 1

    # First two failures don't arm — backoff stays 0 (negative once we
    # advance fake_now past the prior window).
    # First armed backoff (3rd fail) is 2**0 = 1 min = 60s.
    armed = [b for b in backoffs if b > 0]
    assert armed[0] == pytest.approx(60.0)
    # Each subsequent backoff should be ≥ previous.
    for i in range(1, len(armed)):
        assert armed[i] >= armed[i - 1]
    # All ≤ 30 minutes (1800s).
    assert all(b <= 1800.0 for b in armed)


# ── primary in backoff → skipped ────────────────────────────


@pytest.mark.asyncio
async def test_primary_in_backoff_window_is_skipped(monkeypatch):
    """When the primary is in its backoff window, the chain skips it
    entirely and only tries fallbacks — saves the wasted round-trip."""
    primary = _StubLLM("p", fail_n=10, fail_msg="generic")
    fb = _StubLLM("f")
    chain = FallbackLLM(primary, fallbacks=[fb])

    fake_now = [time.monotonic()]
    monkeypatch.setattr("halal_trader.core.llm.fallback.time.monotonic", lambda: fake_now[0])

    # Burn three failures to arm the primary backoff.
    for _ in range(3):
        await chain.generate("x")
    primary_calls_after_arm = primary.calls
    assert chain._backoff_until > fake_now[0]

    # Next call within window: primary must NOT be invoked.
    await chain.generate("y")
    assert primary.calls == primary_calls_after_arm  # unchanged
    assert fb.calls >= 1


@pytest.mark.asyncio
async def test_primary_eligible_again_after_backoff_window(monkeypatch):
    """Once the backoff timer expires, the primary is re-tried."""
    primary = _StubLLM("p", fail_n=3, fail_msg="generic")
    fb = _StubLLM("f")
    chain = FallbackLLM(primary, fallbacks=[fb])

    fake_now = [time.monotonic()]
    monkeypatch.setattr("halal_trader.core.llm.fallback.time.monotonic", lambda: fake_now[0])

    for _ in range(3):
        await chain.generate("x")
    fake_now[0] = chain._backoff_until + 1

    primary_calls_before = primary.calls
    await chain.generate("after")
    # Primary attempted again; recovery resets counter.
    assert primary.calls == primary_calls_before + 1
    assert chain._consecutive_failures == 0


# ── usage / thinking propagation ────────────────────────────


@pytest.mark.asyncio
async def test_active_provider_usage_propagates_to_chain():
    """When a fallback serves the call, its `last_usage` (with the real
    cost paid for *this* call) must surface on the chain — otherwise
    the LlmDecision row records the stale primary usage from a prior
    call. Same for `last_thinking`."""
    sentinel_usage = object()
    primary = _StubLLM("primary", fail_n=1, fail_msg="generic")
    fb = _StubLLM("fb", usage=sentinel_usage, thinking="reasoning trace")
    chain = FallbackLLM(primary, fallbacks=[fb])

    await chain.generate("x")
    assert chain.last_usage is sentinel_usage
    assert chain.last_thinking == "reasoning trace"


@pytest.mark.asyncio
async def test_primary_success_propagates_its_usage():
    primary_usage = object()
    primary = _StubLLM("primary", usage=primary_usage, thinking="primary trace")
    fb = _StubLLM("fb")
    chain = FallbackLLM(primary, fallbacks=[fb])

    await chain.generate("x")
    assert chain.last_usage is primary_usage
    assert chain.last_thinking == "primary trace"


# ── generate_json parallel path ─────────────────────────────


@pytest.mark.asyncio
async def test_generate_json_primary_success():
    primary = _StubLLM("primary", json_response={"ok": True})
    fb = _StubLLM("fb")
    chain = FallbackLLM(primary, fallbacks=[fb])

    result = await chain.generate_json("hi")
    assert result == {"ok": True}
    assert primary.json_calls == 1
    assert fb.json_calls == 0


@pytest.mark.asyncio
async def test_generate_json_falls_through_to_fallback():
    """generate_json mirrors generate() — failures rotate through the
    chain in the same order with the same backoff plumbing."""
    primary = _StubLLM("primary", fail_n=1, fail_msg="generic")
    fb = _StubLLM("fb", json_response={"served_by": "fb"})
    chain = FallbackLLM(primary, fallbacks=[fb])

    result = await chain.generate_json("hi")
    assert result == {"served_by": "fb"}
    assert chain._consecutive_failures == 1


@pytest.mark.asyncio
async def test_generate_json_chain_backoff_blocks(monkeypatch):
    """Chain backoff must apply to generate_json the same way as generate."""
    primary = _StubLLM("p", fail_n=10)
    fb = _StubLLM("f", fail_n=10)
    chain = FallbackLLM(primary, fallbacks=[fb])

    # First call: both fail → chain backoff arms.
    with pytest.raises(Exception):
        await chain.generate_json("first")
    assert chain._chain_backoff_until > 0

    primary_before = primary.json_calls
    fb_before = fb.json_calls
    with pytest.raises(RuntimeError, match="All LLM providers in backoff"):
        await chain.generate_json("blocked")
    # No providers invoked during the backoff window.
    assert primary.json_calls == primary_before
    assert fb.json_calls == fb_before


@pytest.mark.asyncio
async def test_generate_json_propagates_last_error_when_all_fail():
    """All providers raised — the last_error from the chain bubbles up
    rather than a generic 'No LLM providers available' (we have a real
    error, surface it)."""
    primary = _StubLLM("primary", fail_n=1, fail_msg="primary boom")
    fb = _StubLLM("fb", fail_n=1, fail_msg="fb boom")
    chain = FallbackLLM(primary, fallbacks=[fb])

    with pytest.raises(RuntimeError, match="fb boom"):
        await chain.generate_json("x")


# ── chain backoff blocks generate too (parity check) ────────


@pytest.mark.asyncio
async def test_chain_backoff_message_includes_remaining_seconds():
    """The error message tells the operator how long until the chain is
    eligible again — important for ops triage."""
    primary = _StubLLM("p", fail_n=10)
    fb = _StubLLM("f", fail_n=10)
    chain = FallbackLLM(primary, fallbacks=[fb])

    with pytest.raises(Exception):
        await chain.generate("first")

    with pytest.raises(RuntimeError, match=r"backoff for \d+s more"):
        await chain.generate("blocked")
