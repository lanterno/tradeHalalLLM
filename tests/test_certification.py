"""Tests for education/certification.py — Round-5 Wave 20.B."""

from __future__ import annotations

from datetime import datetime

import pytest

from halal_trader.education.certification import (
    Answer,
    AttemptRecord,
    CandidateHistory,
    Certificate,
    CertTier,
    Exam,
    Question,
    QuestionKind,
    can_take,
    grade,
    issue_certificate,
    render_attempt,
    render_certificate,
    verify_certificate,
)


def _q(
    question_id: str = "Q1",
    kind: QuestionKind = QuestionKind.SINGLE_CHOICE,
    prompt: str = "Is interest haram?",
    option_keys: tuple[str, ...] = ("A", "B", "C", "D"),
    correct_keys: tuple[str, ...] = ("A",),
    weight: float = 1.0,
) -> Question:
    return Question(
        question_id=question_id,
        kind=kind,
        prompt=prompt,
        option_keys=option_keys,
        correct_keys=correct_keys,
        weight=weight,
    )


def _exam(
    exam_id: str = "E1",
    tier: CertTier = CertTier.HT_1,
    n_q: int = 10,
    pass_threshold: float | None = None,
) -> Exam:
    qs = tuple(_q(question_id=f"Q{i}") for i in range(n_q))
    return Exam(
        exam_id=exam_id,
        tier=tier,
        questions=qs,
        pass_threshold=pass_threshold,
    )


# --- Question validation ----------------------------------------------


def test_question_valid():
    q = _q()
    assert q.kind is QuestionKind.SINGLE_CHOICE


def test_question_empty_prompt_rejected():
    with pytest.raises(ValueError):
        _q(prompt="")


def test_question_no_options_rejected():
    with pytest.raises(ValueError):
        _q(option_keys=())


def test_question_duplicate_option_rejected():
    with pytest.raises(ValueError):
        _q(option_keys=("A", "A", "B"))


def test_question_correct_not_in_options_rejected():
    with pytest.raises(ValueError):
        _q(option_keys=("A", "B"), correct_keys=("C",))


def test_question_single_choice_multi_correct_rejected():
    with pytest.raises(ValueError):
        _q(
            kind=QuestionKind.SINGLE_CHOICE,
            correct_keys=("A", "B"),
        )


def test_question_multi_choice_multi_correct_allowed():
    q = _q(
        kind=QuestionKind.MULTI_CHOICE,
        correct_keys=("A", "B"),
    )
    assert q.correct_keys == ("A", "B")


def test_question_true_false_two_options_required():
    with pytest.raises(ValueError):
        _q(
            kind=QuestionKind.TRUE_FALSE,
            option_keys=("A", "B", "C"),
            correct_keys=("A",),
        )


def test_question_zero_weight_rejected():
    with pytest.raises(ValueError):
        _q(weight=0)


def test_question_immutable():
    q = _q()
    with pytest.raises(AttributeError):
        q.weight = 2.0  # type: ignore[misc]


# --- Exam validation -------------------------------------------------


def test_exam_valid():
    e = _exam()
    assert e.effective_threshold() == pytest.approx(0.70)


def test_exam_empty_questions_rejected():
    with pytest.raises(ValueError):
        Exam(exam_id="E", tier=CertTier.HT_1, questions=())


def test_exam_duplicate_question_id_rejected():
    bad = (_q(question_id="Q1"), _q(question_id="Q1"))
    with pytest.raises(ValueError):
        Exam(exam_id="E", tier=CertTier.HT_1, questions=bad)


def test_exam_pass_threshold_out_of_range_rejected():
    with pytest.raises(ValueError):
        _exam(pass_threshold=0.0)
    with pytest.raises(ValueError):
        _exam(pass_threshold=1.0)


def test_exam_default_threshold_per_tier():
    """Pin: HT-1=0.70, HT-2=0.75, HT-3=0.80."""
    assert _exam(tier=CertTier.HT_1).effective_threshold() == 0.70
    assert _exam(tier=CertTier.HT_2).effective_threshold() == 0.75
    assert _exam(tier=CertTier.HT_3).effective_threshold() == 0.80


def test_exam_custom_threshold_used():
    e = _exam(pass_threshold=0.60)
    assert e.effective_threshold() == 0.60


# --- Answer validation -----------------------------------------------


def test_answer_duplicate_selected_rejected():
    with pytest.raises(ValueError):
        Answer(question_id="Q1", selected_keys=("A", "A"))


def test_answer_empty_id_rejected():
    with pytest.raises(ValueError):
        Answer(question_id="", selected_keys=("A",))


# --- grade -----------------------------------------------------------


def _grade(
    answers: list[Answer],
    exam: Exam | None = None,
) -> AttemptRecord:
    exam = exam or _exam(n_q=10)
    return grade(
        exam,
        answers,
        attempt_id="AT1",
        candidate_id="alice",
        started_at=datetime(2026, 5, 11, 9, 0),
        finished_at=datetime(2026, 5, 11, 10, 0),
    )


def test_grade_all_correct_passes():
    exam = _exam(n_q=10)
    answers = [Answer(question_id=f"Q{i}", selected_keys=("A",)) for i in range(10)]
    record = _grade(answers, exam)
    assert record.percent_score() == 1.0
    assert record.passed


def test_grade_above_threshold_passes():
    exam = _exam(n_q=10)  # threshold 0.70 default
    # 7 correct + 3 wrong.
    answers = [Answer(question_id=f"Q{i}", selected_keys=("A",)) for i in range(7)] + [
        Answer(question_id=f"Q{i}", selected_keys=("B",)) for i in range(7, 10)
    ]
    record = _grade(answers, exam)
    assert record.percent_score() == pytest.approx(0.70)
    assert record.passed


def test_grade_below_threshold_fails():
    exam = _exam(n_q=10)
    answers = [Answer(question_id=f"Q{i}", selected_keys=("A",)) for i in range(6)] + [
        Answer(question_id=f"Q{i}", selected_keys=("B",)) for i in range(6, 10)
    ]
    record = _grade(answers, exam)
    assert not record.passed


def test_grade_missing_answers_count_as_zero():
    exam = _exam(n_q=10)
    answers = [Answer(question_id="Q0", selected_keys=("A",))]
    record = _grade(answers, exam)
    assert record.percent_score() == pytest.approx(0.10)


def test_grade_unknown_answers_ignored():
    exam = _exam(n_q=2)
    answers = [
        Answer(question_id="Q0", selected_keys=("A",)),
        Answer(question_id="NONEXISTENT", selected_keys=("A",)),
    ]
    record = _grade(answers, exam)
    # 1/2 correct = 50%; below default 70% → fail.
    assert record.percent_score() == 0.5


def test_grade_multi_choice_all_or_nothing():
    """Pin: MULTI_CHOICE requires exact key-set match."""
    exam = Exam(
        exam_id="E",
        tier=CertTier.HT_1,
        questions=(
            _q(
                question_id="Q0",
                kind=QuestionKind.MULTI_CHOICE,
                option_keys=("A", "B", "C", "D"),
                correct_keys=("A", "B"),
            ),
        ),
        pass_threshold=0.5,
    )
    # Partial match → 0 points.
    rec_partial = grade(
        exam,
        [Answer(question_id="Q0", selected_keys=("A",))],
        attempt_id="AT1",
        candidate_id="alice",
        started_at=datetime(2026, 5, 11, 9, 0),
        finished_at=datetime(2026, 5, 11, 10, 0),
    )
    assert rec_partial.percent_score() == 0.0
    # Exact match → full points.
    rec_full = grade(
        exam,
        [Answer(question_id="Q0", selected_keys=("A", "B"))],
        attempt_id="AT2",
        candidate_id="alice",
        started_at=datetime(2026, 5, 11, 9, 0),
        finished_at=datetime(2026, 5, 11, 10, 0),
    )
    assert rec_full.percent_score() == 1.0


def test_grade_weights_respected():
    """Pin: question weights bias the score."""
    exam = Exam(
        exam_id="E",
        tier=CertTier.HT_1,
        questions=(
            _q(question_id="Q0", weight=1.0),
            _q(question_id="Q1", weight=4.0),
        ),
        pass_threshold=0.5,
    )
    # Only the heavyweight question right.
    record = grade(
        exam,
        [Answer(question_id="Q1", selected_keys=("A",))],
        attempt_id="AT1",
        candidate_id="alice",
        started_at=datetime(2026, 5, 11, 9, 0),
        finished_at=datetime(2026, 5, 11, 10, 0),
    )
    # Earned 4 / 5 = 80%.
    assert record.percent_score() == pytest.approx(0.80)


# --- CandidateHistory ------------------------------------------------


def test_history_helpers():
    pass_rec = AttemptRecord(
        attempt_id="AT1",
        candidate_id="alice",
        exam_id="E1",
        tier=CertTier.HT_1,
        started_at=datetime(2026, 5, 11, 9, 0),
        finished_at=datetime(2026, 5, 11, 10, 0),
        raw_score=8.0,
        total_weight=10.0,
        passed=True,
    )
    fail_rec = AttemptRecord(
        attempt_id="AT2",
        candidate_id="alice",
        exam_id="E2",
        tier=CertTier.HT_2,
        started_at=datetime(2026, 5, 12, 9, 0),
        finished_at=datetime(2026, 5, 12, 10, 0),
        raw_score=5.0,
        total_weight=10.0,
        passed=False,
    )
    hist = CandidateHistory(candidate_id="alice", attempts=(pass_rec, fail_rec))
    assert hist.has_passed(CertTier.HT_1)
    assert not hist.has_passed(CertTier.HT_2)
    assert hist.highest_passed_tier() is CertTier.HT_1


def test_history_no_passes_highest_is_none():
    hist = CandidateHistory(candidate_id="alice")
    assert hist.highest_passed_tier() is None


def test_history_mismatched_attempt_rejected():
    bad = AttemptRecord(
        attempt_id="AT1",
        candidate_id="bob",
        exam_id="E1",
        tier=CertTier.HT_1,
        started_at=datetime(2026, 5, 11, 9, 0),
        finished_at=datetime(2026, 5, 11, 10, 0),
        raw_score=8.0,
        total_weight=10.0,
        passed=True,
    )
    with pytest.raises(ValueError):
        CandidateHistory(candidate_id="alice", attempts=(bad,))


# --- can_take --------------------------------------------------------


def test_can_take_ht1_with_no_history():
    hist = CandidateHistory(candidate_id="alice")
    ok, reason = can_take(hist, CertTier.HT_1, now=datetime(2026, 5, 11, 12, 0))
    assert ok
    assert reason == "ok"


def test_can_take_ht2_without_ht1_blocked():
    hist = CandidateHistory(candidate_id="alice")
    ok, reason = can_take(hist, CertTier.HT_2, now=datetime(2026, 5, 11, 12, 0))
    assert not ok
    assert "prerequisite" in reason


def test_can_take_already_certified_blocked():
    pass_rec = AttemptRecord(
        attempt_id="AT1",
        candidate_id="alice",
        exam_id="E1",
        tier=CertTier.HT_1,
        started_at=datetime(2026, 5, 11, 9, 0),
        finished_at=datetime(2026, 5, 11, 10, 0),
        raw_score=8.0,
        total_weight=10.0,
        passed=True,
    )
    hist = CandidateHistory(candidate_id="alice", attempts=(pass_rec,))
    ok, reason = can_take(hist, CertTier.HT_1, now=datetime(2026, 5, 11, 12, 0))
    assert not ok
    assert "already certified" in reason


def test_can_take_cooldown_active():
    fail_rec = AttemptRecord(
        attempt_id="AT1",
        candidate_id="alice",
        exam_id="E1",
        tier=CertTier.HT_1,
        started_at=datetime(2026, 5, 11, 9, 0),
        finished_at=datetime(2026, 5, 11, 10, 0),
        raw_score=5.0,
        total_weight=10.0,
        passed=False,
    )
    hist = CandidateHistory(candidate_id="alice", attempts=(fail_rec,))
    # 1 hour later → cooldown still active (default 24h).
    ok, reason = can_take(hist, CertTier.HT_1, now=datetime(2026, 5, 11, 11, 0))
    assert not ok
    assert "cooldown" in reason


def test_can_take_cooldown_expired():
    fail_rec = AttemptRecord(
        attempt_id="AT1",
        candidate_id="alice",
        exam_id="E1",
        tier=CertTier.HT_1,
        started_at=datetime(2026, 5, 11, 9, 0),
        finished_at=datetime(2026, 5, 11, 10, 0),
        raw_score=5.0,
        total_weight=10.0,
        passed=False,
    )
    hist = CandidateHistory(candidate_id="alice", attempts=(fail_rec,))
    # 25 hours later → cooldown expired.
    ok, reason = can_take(hist, CertTier.HT_1, now=datetime(2026, 5, 12, 11, 0))
    assert ok


def test_can_take_invalid_cooldown_rejected():
    hist = CandidateHistory(candidate_id="alice")
    with pytest.raises(ValueError):
        can_take(
            hist,
            CertTier.HT_1,
            now=datetime(2026, 5, 11, 12, 0),
            cooldown_hours=0,
        )


# --- Certificates ----------------------------------------------------


def _make_passing_history(tier: CertTier = CertTier.HT_1) -> CandidateHistory:
    rec = AttemptRecord(
        attempt_id="AT1",
        candidate_id="alice",
        exam_id="E1",
        tier=tier,
        started_at=datetime(2026, 5, 11, 9, 0),
        finished_at=datetime(2026, 5, 11, 10, 0),
        raw_score=9.0,
        total_weight=10.0,
        passed=True,
    )
    return CandidateHistory(candidate_id="alice", attempts=(rec,))


def test_issue_certificate_basic():
    hist = _make_passing_history()
    cert = issue_certificate(
        hist,
        tier=CertTier.HT_1,
        certificate_id="CERT-001",
        issued_on=datetime(2026, 5, 12),
    )
    assert cert.candidate_id == "alice"
    assert verify_certificate(cert)


def test_issue_certificate_no_pass_rejected():
    hist = CandidateHistory(candidate_id="alice")
    with pytest.raises(ValueError):
        issue_certificate(
            hist,
            tier=CertTier.HT_1,
            certificate_id="CERT-001",
            issued_on=datetime(2026, 5, 12),
        )


def test_verify_certificate_detects_tamper():
    hist = _make_passing_history()
    cert = issue_certificate(
        hist,
        tier=CertTier.HT_1,
        certificate_id="CERT-001",
        issued_on=datetime(2026, 5, 12),
    )
    tampered = Certificate(
        certificate_id=cert.certificate_id,
        candidate_id="mallory",  # changed
        tier=cert.tier,
        issued_on=cert.issued_on,
        anchor_hash=cert.anchor_hash,
        attempt_id=cert.attempt_id,
    )
    assert not verify_certificate(tampered)


# --- Render ----------------------------------------------------------


def test_render_attempt_no_answer_leak():
    rec = AttemptRecord(
        attempt_id="AT1",
        candidate_id="alice",
        exam_id="E1",
        tier=CertTier.HT_1,
        started_at=datetime(2026, 5, 11, 9, 0),
        finished_at=datetime(2026, 5, 11, 10, 0),
        raw_score=8.0,
        total_weight=10.0,
        passed=True,
    )
    out = render_attempt(rec)
    assert "80.00%" in out
    assert "alice" not in out  # masked


def test_render_certificate_format():
    hist = _make_passing_history()
    cert = issue_certificate(
        hist,
        tier=CertTier.HT_1,
        certificate_id="CERT-001",
        issued_on=datetime(2026, 5, 12),
    )
    out = render_certificate(cert)
    assert "🎓" in out
    assert "CERT-001" in out
    assert "alice" not in out
