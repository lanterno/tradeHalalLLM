"""Tests for core/committee_transcript.py — Round-5 Wave 8.E."""

from __future__ import annotations

from datetime import datetime

import pytest

from halal_trader.core.committee_transcript import (
    InMemoryTranscriptStore,
    TranscriptEntry,
    render_entry,
    supersede,
    verify_chain,
)
from halal_trader.core.llm_committee import AgentRole, AgentVote, Stance


def _vote(
    role: AgentRole = AgentRole.BULL,
    stance: Stance = Stance.BUY,
    confidence: float = 0.7,
    rationale: str = "",
) -> AgentVote:
    return AgentVote(role=role, stance=stance, confidence=confidence, rationale=rationale)


def _entry(
    transcript_id: str = "T1",
    ticker: str = "AAPL",
    decision_at: datetime = datetime(2026, 5, 1, 10, 0, 0),
    debate_round: int = 1,
    final_stance: Stance = Stance.BUY,
    final_confidence: float = 0.7,
    veto_invoked: bool = False,
    votes: tuple[AgentVote, ...] | None = None,
    prev_hash: str = "",
    supersedes_id: str = "",
) -> TranscriptEntry:
    if votes is None:
        votes = (_vote(),)
    return TranscriptEntry(
        transcript_id=transcript_id,
        ticker=ticker,
        decision_at=decision_at,
        debate_round=debate_round,
        final_stance=final_stance,
        final_confidence=final_confidence,
        veto_invoked=veto_invoked,
        votes=votes,
        prev_hash=prev_hash,
        supersedes_id=supersedes_id,
    )


# --- TranscriptEntry validation -----------------------------------------


def test_entry_valid():
    e = _entry()
    assert e.ticker == "AAPL"


def test_entry_empty_id_rejected():
    with pytest.raises(ValueError):
        _entry(transcript_id="")


def test_entry_empty_ticker_rejected():
    with pytest.raises(ValueError):
        _entry(ticker=" ")


def test_entry_zero_round_rejected():
    with pytest.raises(ValueError):
        _entry(debate_round=0)


def test_entry_invalid_confidence_rejected():
    with pytest.raises(ValueError):
        _entry(final_confidence=1.5)


def test_entry_no_votes_rejected():
    with pytest.raises(ValueError):
        _entry(votes=tuple())


def test_entry_immutable():
    e = _entry()
    with pytest.raises(AttributeError):
        e.ticker = "X"  # type: ignore[misc]


# --- entry_hash determinism ---------------------------------------------


def test_entry_hash_stable():
    e1 = _entry()
    e2 = _entry()
    assert e1.entry_hash() == e2.entry_hash()


def test_entry_hash_changes_with_field():
    e1 = _entry(ticker="AAPL")
    e2 = _entry(ticker="MSFT")
    assert e1.entry_hash() != e2.entry_hash()


def test_entry_hash_changes_with_prev_hash():
    e1 = _entry(prev_hash="")
    e2 = _entry(prev_hash="abc")
    assert e1.entry_hash() != e2.entry_hash()


def test_entry_hash_changes_with_rationale():
    e1 = _entry(votes=(_vote(rationale="strong fundamentals"),))
    e2 = _entry(votes=(_vote(rationale="momentum break"),))
    assert e1.entry_hash() != e2.entry_hash()


# --- InMemoryTranscriptStore — append + chain ----------------------------


def test_append_first_entry_with_empty_prev_hash():
    store = InMemoryTranscriptStore()
    e = _entry(prev_hash="")
    store.append(e)
    assert len(store.all()) == 1


def test_append_first_entry_with_nonempty_prev_hash_rejected():
    store = InMemoryTranscriptStore()
    with pytest.raises(ValueError):
        store.append(_entry(prev_hash="abc"))


def test_append_chain_links_correctly():
    store = InMemoryTranscriptStore()
    e1 = _entry(transcript_id="T1", prev_hash="")
    store.append(e1)
    e2 = _entry(transcript_id="T2", prev_hash=e1.entry_hash())
    store.append(e2)
    assert len(store.all()) == 2


def test_append_with_wrong_prev_hash_rejected():
    store = InMemoryTranscriptStore()
    e1 = _entry(transcript_id="T1", prev_hash="")
    store.append(e1)
    with pytest.raises(ValueError):
        store.append(_entry(transcript_id="T2", prev_hash="wrong"))


def test_append_duplicate_id_rejected():
    store = InMemoryTranscriptStore()
    e = _entry(prev_hash="")
    store.append(e)
    e2 = _entry(transcript_id=e.transcript_id, prev_hash=e.entry_hash())
    with pytest.raises(ValueError):
        store.append(e2)


# --- by_id ----------------------------------------------------------------


def test_by_id_returns_match():
    store = InMemoryTranscriptStore()
    e = _entry(prev_hash="")
    store.append(e)
    found = store.by_id("T1")
    assert found is not None
    assert found.transcript_id == "T1"


def test_by_id_returns_none_when_missing():
    store = InMemoryTranscriptStore()
    assert store.by_id("nonexistent") is None


# --- search --------------------------------------------------------------


def test_search_by_ticker():
    store = InMemoryTranscriptStore()
    e1 = _entry(transcript_id="T1", ticker="AAPL", prev_hash="")
    store.append(e1)
    e2 = _entry(transcript_id="T2", ticker="MSFT", prev_hash=e1.entry_hash())
    store.append(e2)
    out = store.search(ticker="AAPL")
    assert len(out) == 1
    assert out[0].ticker == "AAPL"


def test_search_by_role():
    store = InMemoryTranscriptStore()
    bull_vote = _vote(role=AgentRole.BULL)
    quant_vote = _vote(role=AgentRole.QUANT)
    e1 = _entry(transcript_id="T1", votes=(bull_vote,), prev_hash="")
    store.append(e1)
    e2 = _entry(transcript_id="T2", votes=(quant_vote,), prev_hash=e1.entry_hash())
    store.append(e2)
    out_bull = store.search(role=AgentRole.BULL)
    out_quant = store.search(role=AgentRole.QUANT)
    assert len(out_bull) == 1
    assert len(out_quant) == 1


def test_search_by_stance():
    store = InMemoryTranscriptStore()
    e1 = _entry(transcript_id="T1", final_stance=Stance.BUY, prev_hash="")
    store.append(e1)
    e2 = _entry(
        transcript_id="T2",
        final_stance=Stance.SELL,
        prev_hash=e1.entry_hash(),
    )
    store.append(e2)
    out_buy = store.search(stance=Stance.BUY)
    assert len(out_buy) == 1
    assert out_buy[0].final_stance is Stance.BUY


def test_search_by_date_range():
    store = InMemoryTranscriptStore()
    e1 = _entry(
        transcript_id="T1",
        decision_at=datetime(2026, 1, 1),
        prev_hash="",
    )
    store.append(e1)
    e2 = _entry(
        transcript_id="T2",
        decision_at=datetime(2026, 6, 1),
        prev_hash=e1.entry_hash(),
    )
    store.append(e2)
    out = store.search(date_from=datetime(2026, 5, 1))
    assert len(out) == 1
    assert out[0].transcript_id == "T2"


def test_search_combined_filters():
    store = InMemoryTranscriptStore()
    e1 = _entry(
        transcript_id="T1",
        ticker="AAPL",
        final_stance=Stance.BUY,
        prev_hash="",
    )
    store.append(e1)
    e2 = _entry(
        transcript_id="T2",
        ticker="AAPL",
        final_stance=Stance.SELL,
        prev_hash=e1.entry_hash(),
    )
    store.append(e2)
    out = store.search(ticker="AAPL", stance=Stance.BUY)
    assert len(out) == 1
    assert out[0].transcript_id == "T1"


# --- supersede helper ----------------------------------------------------


def test_supersede_appends_with_link():
    store = InMemoryTranscriptStore()
    e1 = _entry(prev_hash="")
    store.append(e1)
    new_entry = supersede(
        store,
        new_id="T2",
        superseded_id="T1",
        decision_at=datetime(2026, 5, 1, 11, 0),
        debate_round=2,
        final_stance=Stance.SELL,
        final_confidence=0.9,
        veto_invoked=False,
        votes=(_vote(stance=Stance.SELL),),
    )
    assert new_entry.supersedes_id == "T1"
    # Same ticker preserved.
    assert new_entry.ticker == "AAPL"


def test_supersede_unknown_id_rejected():
    store = InMemoryTranscriptStore()
    with pytest.raises(ValueError):
        supersede(
            store,
            new_id="T2",
            superseded_id="nonexistent",
            decision_at=datetime(2026, 5, 1),
            debate_round=1,
            final_stance=Stance.SELL,
            final_confidence=0.9,
            veto_invoked=False,
            votes=(_vote(),),
        )


def test_double_supersede_rejected():
    """Pin: a transcript can only be superseded once."""
    store = InMemoryTranscriptStore()
    e1 = _entry(prev_hash="")
    store.append(e1)
    supersede(
        store,
        new_id="T2",
        superseded_id="T1",
        decision_at=datetime(2026, 5, 1, 11, 0),
        debate_round=2,
        final_stance=Stance.SELL,
        final_confidence=0.9,
        veto_invoked=False,
        votes=(_vote(stance=Stance.SELL),),
    )
    with pytest.raises(ValueError):
        supersede(
            store,
            new_id="T3",
            superseded_id="T1",
            decision_at=datetime(2026, 5, 1, 12, 0),
            debate_round=3,
            final_stance=Stance.HOLD,
            final_confidence=0.5,
            veto_invoked=False,
            votes=(_vote(stance=Stance.HOLD),),
        )


# --- verify_chain --------------------------------------------------------


def test_verify_chain_clean():
    store = InMemoryTranscriptStore()
    e1 = _entry(transcript_id="T1", prev_hash="")
    store.append(e1)
    e2 = _entry(transcript_id="T2", prev_hash=e1.entry_hash())
    store.append(e2)
    assert verify_chain(store.all())


def test_verify_chain_empty_passes():
    assert verify_chain([])


def test_verify_chain_detects_tamper():
    """If we splice in an out-of-chain entry, verify_chain returns False."""
    store = InMemoryTranscriptStore()
    e1 = _entry(transcript_id="T1", prev_hash="")
    store.append(e1)
    e2 = _entry(transcript_id="T2", prev_hash=e1.entry_hash())
    store.append(e2)
    # Bad entry with wrong prev_hash inserted via internal access.
    bad = _entry(transcript_id="T3", prev_hash="wrong")
    chain_with_tamper = (e1, e2, bad)
    assert not verify_chain(chain_with_tamper)


# --- Render --------------------------------------------------------------


def test_render_truncates_long_rationale():
    long_rationale = "x" * 200
    e = _entry(votes=(_vote(rationale=long_rationale),))
    out = render_entry(e, rationale_chars=80)
    assert "…" in out


def test_render_keeps_short_rationale():
    e = _entry(votes=(_vote(rationale="momentum is strong"),))
    out = render_entry(e)
    assert "momentum is strong" in out


def test_render_marks_veto():
    e = _entry(veto_invoked=True)
    out = render_entry(e)
    assert "[VETO]" in out


def test_render_marks_supersedes():
    e = _entry(supersedes_id="T0")
    out = render_entry(e)
    assert "supersedes=T0" in out


def test_render_no_rationale_omits_dash():
    e = _entry(votes=(_vote(rationale=""),))
    out = render_entry(e)
    assert " — " not in out
