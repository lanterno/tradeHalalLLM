"""Tests for BaseStrategy's schema-repair retry pass."""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.core.llm.base import BaseLLM, CallUsage
from halal_trader.core.strategy import BaseStrategy
from halal_trader.db import admin
from halal_trader.db.repository import Repository


class _Plan(BaseModel):
    action: str
    confidence: float = Field(ge=0.0, le=1.0)


class _StubLLM(BaseLLM):
    """LLM stub that returns scripted JSON payloads in order."""

    def __init__(self, responses: list[Any]) -> None:
        super().__init__(model="stub")
        self._responses = list(responses)
        self.calls: list[tuple[str, str | None]] = []
        self.last_usage = CallUsage(model="stub")

    async def generate(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError

    async def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        self.calls.append((prompt, system))
        if not self._responses:
            raise RuntimeError("stub LLM out of scripted responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _NoopStrategy(BaseStrategy):
    """Concrete BaseStrategy so we can drive ``_run_llm_analysis`` directly."""

    pass


async def _engine_repo(tmp_path):
    db_path = tmp_path / "repair.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    head = admin.head()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')"))
    return engine, Repository(engine)


def _strategy(llm: BaseLLM, repo: Repository) -> _NoopStrategy:
    return _NoopStrategy(
        llm=llm,
        repo=repo,
        llm_provider_name="stub",
        max_position_pct=0.25,
        daily_loss_limit=0.03,
        daily_return_target=0.01,
        max_simultaneous_positions=3,
    )


def _validate(raw: dict[str, Any]) -> _Plan:
    return _Plan.model_validate(raw)


def _make_empty(msg: str) -> _Plan:
    return _Plan(action="hold", confidence=0.0)


def _extract_symbols(_p: _Plan) -> list[str]:
    return []


def _count_actions(p: _Plan) -> dict:
    return {p.action: 1}


async def test_first_response_validates_skips_repair(tmp_path):
    engine, repo = await _engine_repo(tmp_path)
    try:
        llm = _StubLLM([{"action": "buy", "confidence": 0.7}])
        strat = _strategy(llm, repo)
        plan = await strat._run_llm_analysis(
            "system",
            "user",
            prompt_summary="t",
            validate=_validate,
            make_empty=_make_empty,
            extract_symbols=_extract_symbols,
            count_actions=_count_actions,
        )
        assert plan.action == "buy"
        assert len(llm.calls) == 1  # no repair pass
    finally:
        await engine.dispose()


async def test_invalid_json_triggers_repair_pass(tmp_path):
    """Schema-invalid initial response → one repair call → valid plan."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        bad = {"action": "buy", "confidence": 1.7}  # confidence > 1 → ValidationError
        good = {"action": "sell", "confidence": 0.5}
        llm = _StubLLM([bad, good])
        strat = _strategy(llm, repo)
        plan = await strat._run_llm_analysis(
            "system",
            "user",
            prompt_summary="t",
            validate=_validate,
            make_empty=_make_empty,
            extract_symbols=_extract_symbols,
            count_actions=_count_actions,
        )
        assert plan.action == "sell"  # repaired version was used
        assert len(llm.calls) == 2
        # The repair call's user prompt mentions both the validation error
        # and the previous bad response.
        repair_user, repair_system = llm.calls[1]
        assert "validation" in repair_user.lower() or "fix" in repair_user.lower()
        assert "1.7" in repair_user  # echoed previous bad value
        assert repair_system == "system"
    finally:
        await engine.dispose()


async def test_repair_pass_also_invalid_falls_back_to_empty(tmp_path):
    """Two consecutive failures → empty plan (no infinite retry)."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        bad1 = {"action": "buy", "confidence": 1.7}
        bad2 = {"action": "buy", "confidence": -0.5}
        llm = _StubLLM([bad1, bad2])
        strat = _strategy(llm, repo)
        plan = await strat._run_llm_analysis(
            "system",
            "user",
            prompt_summary="t",
            validate=_validate,
            make_empty=_make_empty,
            extract_symbols=_extract_symbols,
            count_actions=_count_actions,
        )
        # _make_empty fallback fires.
        assert plan.action == "hold"
        assert len(llm.calls) == 2
    finally:
        await engine.dispose()


async def test_network_error_does_not_trigger_repair(tmp_path):
    """Transport-layer errors should NOT consume a repair attempt."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        llm = _StubLLM([RuntimeError("boom: connection reset")])
        strat = _strategy(llm, repo)
        plan = await strat._run_llm_analysis(
            "system",
            "user",
            prompt_summary="t",
            validate=_validate,
            make_empty=_make_empty,
            extract_symbols=_extract_symbols,
            count_actions=_count_actions,
        )
        assert plan.action == "hold"
        assert len(llm.calls) == 1  # repair not attempted
    finally:
        await engine.dispose()


async def test_repair_call_itself_failing_falls_back_cleanly(tmp_path):
    """If the repair call raises, we should still return an empty plan."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        bad = {"action": "buy", "confidence": 1.7}
        llm = _StubLLM([bad, RuntimeError("repair call timed out")])
        strat = _strategy(llm, repo)
        plan = await strat._run_llm_analysis(
            "system",
            "user",
            prompt_summary="t",
            validate=_validate,
            make_empty=_make_empty,
            extract_symbols=_extract_symbols,
            count_actions=_count_actions,
        )
        assert plan.action == "hold"
        assert len(llm.calls) == 2
    finally:
        await engine.dispose()


async def test_validation_error_class_used(tmp_path):
    """Sanity check that pydantic raises ValidationError so our repair clause sees it."""
    with pytest.raises(ValidationError):
        _Plan.model_validate({"action": "buy", "confidence": 5})
