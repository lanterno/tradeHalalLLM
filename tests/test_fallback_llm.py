"""Tests for FallbackLLM — provider rotation, primary backoff, chain backoff."""

from __future__ import annotations

import pytest

from halal_trader.core.llm import BaseLLM, FallbackLLM


class _StubLLM(BaseLLM):
    """A controllable BaseLLM that records calls and can fail on demand."""

    def __init__(self, name: str, fail_n: int = 0, fail_with: type[Exception] = RuntimeError):
        super().__init__(model=name)
        self.calls = 0
        self._fail_n = fail_n
        self._fail_with = fail_with

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        if self._fail_n > 0:
            self._fail_n -= 1
            raise self._fail_with(f"{self.model} failure")
        return f"{self.model}:{prompt}"


@pytest.mark.asyncio
async def test_primary_success_no_fallback():
    primary = _StubLLM("primary")
    fb = _StubLLM("fallback")
    chain = FallbackLLM(primary, fallbacks=[fb])

    result = await chain.generate("hi")
    assert result == "primary:hi"
    assert primary.calls == 1
    assert fb.calls == 0
    assert chain.model == "primary"


@pytest.mark.asyncio
async def test_primary_fails_then_fallback_serves():
    primary = _StubLLM("primary", fail_n=1)
    fb = _StubLLM("fallback")
    chain = FallbackLLM(primary, fallbacks=[fb])

    result = await chain.generate("hi")
    assert result == "fallback:hi"
    assert chain.model == "fallback"  # tracks active provider


@pytest.mark.asyncio
async def test_empty_error_message_logs_exception_type(caplog):
    """A provider failure whose ``str(e)`` is empty (bare timeout /
    connection reset) must still log the exception TYPE — otherwise the
    log was a useless 'LLM provider X failed: ' with no cause."""

    class _BlankError(RuntimeError):
        def __str__(self) -> str:  # mimics bare ConnectError/ReadTimeout
            return ""

    primary = _StubLLM("primary", fail_n=1, fail_with=_BlankError)
    fb = _StubLLM("fallback")
    chain = FallbackLLM(primary, fallbacks=[fb])

    import logging

    with caplog.at_level(logging.WARNING):
        result = await chain.generate("hi")
    assert result == "fallback:hi"
    # The type name appears even though str(error) was empty.
    assert "_BlankError" in caplog.text


@pytest.mark.asyncio
async def test_primary_recovers_resets_failure_counter():
    primary = _StubLLM("primary", fail_n=2)
    fb = _StubLLM("fallback")
    chain = FallbackLLM(primary, fallbacks=[fb])

    # First two raise → fallback serves both times.
    await chain.generate("a")
    await chain.generate("b")
    # Now primary recovers.
    result = await chain.generate("c")
    assert result == "primary:c"
    assert chain._consecutive_failures == 0


@pytest.mark.asyncio
async def test_chain_backoff_blocks_after_all_fail(monkeypatch):
    primary = _StubLLM("primary", fail_n=10)
    fb = _StubLLM("fb", fail_n=10)
    chain = FallbackLLM(primary, fallbacks=[fb])

    # First call: both fail → chain_failures becomes 1, backoff is set.
    with pytest.raises(Exception):
        await chain.generate("first")
    assert chain._chain_failures == 1
    assert chain._chain_backoff_until > 0

    # Second call within backoff window → blocked without invoking providers.
    primary_calls_before = primary.calls
    fb_calls_before = fb.calls
    with pytest.raises(RuntimeError, match="All LLM providers in backoff"):
        await chain.generate("blocked")
    assert primary.calls == primary_calls_before
    assert fb.calls == fb_calls_before


@pytest.mark.asyncio
async def test_chain_backoff_grows_exponentially(monkeypatch):
    """Each consecutive full-chain failure doubles the backoff up to 30 min."""
    primary = _StubLLM("p", fail_n=10)
    fb = _StubLLM("f", fail_n=10)
    chain = FallbackLLM(primary, fallbacks=[fb])

    # Force the backoff window to be already elapsed so we can fail repeatedly.
    import time

    real_monotonic = time.monotonic
    fake_now = [real_monotonic()]
    monkeypatch.setattr("halal_trader.core.llm.fallback.time.monotonic", lambda: fake_now[0])

    backoffs: list[float] = []
    for _ in range(4):
        try:
            await chain.generate("x")
        except Exception:
            pass
        backoffs.append(chain._chain_backoff_until - fake_now[0])
        # Skip past the backoff window.
        fake_now[0] = chain._chain_backoff_until + 1

    # Each successive backoff should be at least as long as the previous.
    assert backoffs[0] < backoffs[1]
    assert backoffs[1] <= backoffs[2]
    # 30-minute cap = 1800s.
    assert all(b <= 1800 for b in backoffs)


@pytest.mark.asyncio
async def test_chain_backoff_clears_on_success(monkeypatch):
    primary = _StubLLM("primary", fail_n=10)
    fb = _StubLLM("fallback")
    chain = FallbackLLM(primary, fallbacks=[fb])

    # Trigger chain_failures = 1 by failing primary; fallback succeeds, so
    # chain_failures should NOT increment (we got a result).
    result = await chain.generate("ok")
    assert result == "fallback:ok"
    assert chain._chain_failures == 0
    assert chain._chain_backoff_until == 0


@pytest.mark.asyncio
async def test_with_no_fallbacks_propagates_primary_error():
    primary = _StubLLM("primary", fail_n=1)
    chain = FallbackLLM(primary, fallbacks=[])
    with pytest.raises(RuntimeError):
        await chain.generate("x")


@pytest.mark.asyncio
async def test_active_model_property_tracks_provider():
    primary = _StubLLM("primary", fail_n=1)
    fb = _StubLLM("fallback")
    chain = FallbackLLM(primary, fallbacks=[fb])

    assert chain.model == "primary"
    await chain.generate("first")  # primary fails → fallback served
    assert chain.model == "fallback"
