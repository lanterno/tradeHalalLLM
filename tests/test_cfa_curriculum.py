"""Tests for education/cfa_curriculum.py — Round-5 Wave 20.I."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.education.cfa_curriculum import (
    CertLevel,
    Curriculum,
    EnrolmentStatus,
    LearnerEnrolment,
    Level,
    Module,
    Topic,
    certify,
    complete_topic,
    enrol,
    highest_certified_level,
    is_exam_eligible,
    percent_complete,
    render_enrolment,
    withdraw,
)


def _topic(topic_id: str = "T1", title: str = "Topic", hours: float = 2.0) -> Topic:
    return Topic(topic_id=topic_id, title=title, estimated_hours=hours)


def _module(
    module_id: str = "M1",
    title: str = "Module",
    n_topics: int = 2,
) -> Module:
    topics = tuple(_topic(topic_id=f"{module_id}-T{i}") for i in range(n_topics))
    return Module(module_id=module_id, title=title, topics=topics)


def _level(
    level: CertLevel = CertLevel.HCA_I,
    n_modules: int = 2,
    pass_threshold: float = 0.70,
) -> Level:
    modules = tuple(_module(module_id=f"{level.value}-M{i}", n_topics=2) for i in range(n_modules))
    return Level(level=level, modules=modules, pass_threshold_pct=pass_threshold)


def _curriculum() -> Curriculum:
    return Curriculum(
        levels=(
            _level(level=CertLevel.HCA_I),
            _level(level=CertLevel.HCA_II),
            _level(level=CertLevel.HCA_III),
        )
    )


# --- Topic validation -----------------------


def test_topic_valid():
    t = _topic()
    assert t.estimated_hours == 2.0


def test_topic_empty_id_rejected():
    with pytest.raises(ValueError):
        _topic(topic_id="")


def test_topic_empty_title_rejected():
    with pytest.raises(ValueError):
        _topic(title=" ")


def test_topic_long_title_rejected():
    with pytest.raises(ValueError):
        _topic(title="x" * 300)


def test_topic_zero_hours_rejected():
    with pytest.raises(ValueError):
        _topic(hours=0)


def test_topic_excessive_hours_rejected():
    with pytest.raises(ValueError):
        _topic(hours=200)


# --- Module validation ----------------------


def test_module_valid():
    m = _module(n_topics=3)
    assert m.total_hours() == 6.0


def test_module_empty_topics_rejected():
    with pytest.raises(ValueError):
        Module(module_id="M1", title="x", topics=())


def test_module_duplicate_topic_rejected():
    bad = (_topic(topic_id="T1"), _topic(topic_id="T1"))
    with pytest.raises(ValueError):
        Module(module_id="M1", title="x", topics=bad)


# --- Level validation -----------------------


def test_level_valid():
    L = _level()
    assert L.pass_threshold_pct == 0.70


def test_level_empty_modules_rejected():
    with pytest.raises(ValueError):
        Level(level=CertLevel.HCA_I, modules=())


def test_level_duplicate_module_rejected():
    bad = (_module(module_id="M1"), _module(module_id="M1"))
    with pytest.raises(ValueError):
        Level(level=CertLevel.HCA_I, modules=bad)


def test_level_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        _level(pass_threshold=0.0)
    with pytest.raises(ValueError):
        _level(pass_threshold=1.0)


def test_level_total_hours():
    L = _level(n_modules=3)
    # 3 modules × 2 topics × 2 hours = 12.
    assert L.total_hours() == 12.0


def test_level_all_topic_ids():
    L = _level(n_modules=2)
    assert len(L.all_topic_ids()) == 4


# --- Curriculum validation ------------------


def test_curriculum_valid():
    c = _curriculum()
    assert len(c.levels) == 3


def test_curriculum_wrong_level_count_rejected():
    with pytest.raises(ValueError):
        Curriculum(levels=(_level(level=CertLevel.HCA_I),))


def test_curriculum_duplicate_level_rejected():
    with pytest.raises(ValueError):
        Curriculum(
            levels=(
                _level(level=CertLevel.HCA_I),
                _level(level=CertLevel.HCA_I),
                _level(level=CertLevel.HCA_III),
            )
        )


def test_curriculum_duplicate_topic_id_across_levels_rejected():
    # Two levels both contain topic_id "shared".
    t_shared = _topic(topic_id="shared")
    m_a = Module(module_id="MA", title="x", topics=(t_shared,))
    m_b = Module(module_id="MB", title="y", topics=(t_shared,))
    with pytest.raises(ValueError):
        Curriculum(
            levels=(
                Level(level=CertLevel.HCA_I, modules=(m_a,)),
                Level(level=CertLevel.HCA_II, modules=(m_b,)),
                _level(level=CertLevel.HCA_III),
            )
        )


def test_curriculum_by_level():
    c = _curriculum()
    L = c.by_level(CertLevel.HCA_II)
    assert L.level is CertLevel.HCA_II


def test_curriculum_topic_to_level():
    c = _curriculum()
    # HCA_I module IDs start with "hca_i-M..."; topic IDs are "hca_i-M0-T0" etc.
    found = c.topic_to_level("hca_i-M0-T0")
    assert found is CertLevel.HCA_I


def test_curriculum_topic_to_level_unknown_none():
    c = _curriculum()
    assert c.topic_to_level("missing") is None


# --- LearnerEnrolment validation -------------


def test_enrolment_valid():
    e = LearnerEnrolment(
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    assert e.status is EnrolmentStatus.ENROLLED


def test_enrolment_not_enrolled_status_rejected():
    with pytest.raises(ValueError):
        LearnerEnrolment(
            enrolment_id="E1",
            learner_id="bob",
            level=CertLevel.HCA_I,
            enrolled_on=date(2026, 5, 1),
            status=EnrolmentStatus.NOT_ENROLLED,
        )


def test_enrolment_certified_without_date_rejected():
    with pytest.raises(ValueError):
        LearnerEnrolment(
            enrolment_id="E1",
            learner_id="bob",
            level=CertLevel.HCA_I,
            enrolled_on=date(2026, 5, 1),
            status=EnrolmentStatus.CERTIFIED,
            certified_on=None,
        )


def test_enrolment_certified_before_enrolled_rejected():
    with pytest.raises(ValueError):
        LearnerEnrolment(
            enrolment_id="E1",
            learner_id="bob",
            level=CertLevel.HCA_I,
            enrolled_on=date(2026, 5, 1),
            status=EnrolmentStatus.CERTIFIED,
            certified_on=date(2026, 4, 1),
        )


def test_enrolment_duplicate_topic_rejected():
    with pytest.raises(ValueError):
        LearnerEnrolment(
            enrolment_id="E1",
            learner_id="bob",
            level=CertLevel.HCA_I,
            enrolled_on=date(2026, 5, 1),
            completed_topic_ids=("t1", "t1"),
        )


# --- enrol -----------------------------------


def test_enrol_hca_i_no_prereq():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    assert e.status is EnrolmentStatus.ENROLLED


def test_enrol_hca_ii_without_hca_i_rejected():
    c = _curriculum()
    with pytest.raises(ValueError):
        enrol(
            c,
            enrolment_id="E1",
            learner_id="bob",
            level=CertLevel.HCA_II,
            enrolled_on=date(2026, 5, 1),
        )


def test_enrol_hca_ii_after_hca_i_certified():
    c = _curriculum()
    prior = LearnerEnrolment(
        enrolment_id="EH1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 1, 1),
        status=EnrolmentStatus.CERTIFIED,
        certified_on=date(2026, 4, 1),
    )
    e = enrol(
        c,
        enrolment_id="E2",
        learner_id="bob",
        level=CertLevel.HCA_II,
        enrolled_on=date(2026, 5, 1),
        prior_enrolments=[prior],
    )
    assert e.level is CertLevel.HCA_II


def test_enrol_hca_iii_without_hca_ii_rejected():
    c = _curriculum()
    hca_i = LearnerEnrolment(
        enrolment_id="EH1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 1, 1),
        status=EnrolmentStatus.CERTIFIED,
        certified_on=date(2026, 4, 1),
    )
    with pytest.raises(ValueError):
        enrol(
            c,
            enrolment_id="E3",
            learner_id="bob",
            level=CertLevel.HCA_III,
            enrolled_on=date(2026, 5, 1),
            prior_enrolments=[hca_i],
        )


def test_enrol_duplicate_rejected():
    c = _curriculum()
    existing = LearnerEnrolment(
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 1, 1),
    )
    with pytest.raises(ValueError):
        enrol(
            c,
            enrolment_id="E2",
            learner_id="bob",
            level=CertLevel.HCA_I,
            enrolled_on=date(2026, 5, 1),
            prior_enrolments=[existing],
        )


# --- complete_topic --------------------------


def test_complete_topic_basic():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    topic_id = c.by_level(CertLevel.HCA_I).all_topic_ids()[0]
    e2 = complete_topic(e, c, topic_id=topic_id)
    assert topic_id in e2.completed_topic_ids


def test_complete_topic_idempotent():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    topic_id = c.by_level(CertLevel.HCA_I).all_topic_ids()[0]
    e2 = complete_topic(e, c, topic_id=topic_id)
    e3 = complete_topic(e2, c, topic_id=topic_id)
    assert len(e3.completed_topic_ids) == 1


def test_complete_topic_wrong_level_rejected():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    other_topic = c.by_level(CertLevel.HCA_II).all_topic_ids()[0]
    with pytest.raises(ValueError):
        complete_topic(e, c, topic_id=other_topic)


def test_complete_all_topics_promotes_to_eligible():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    for t in c.by_level(CertLevel.HCA_I).all_topic_ids():
        e = complete_topic(e, c, topic_id=t)
    assert e.status is EnrolmentStatus.ELIGIBLE


def test_complete_topic_in_withdrawn_rejected():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    e = withdraw(e, on=date(2026, 6, 1))
    with pytest.raises(ValueError):
        complete_topic(e, c, topic_id=c.by_level(CertLevel.HCA_I).all_topic_ids()[0])


# --- percent_complete ------------------------


def test_percent_complete_zero_initial():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    assert percent_complete(e, c) == 0.0


def test_percent_complete_half():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    topics = c.by_level(CertLevel.HCA_I).all_topic_ids()
    # Complete half.
    for t in topics[: len(topics) // 2]:
        e = complete_topic(e, c, topic_id=t)
    assert percent_complete(e, c) == pytest.approx(0.5)


# --- certify ---------------------------------


def _eligible_enrolment() -> tuple[LearnerEnrolment, Curriculum]:
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    for t in c.by_level(CertLevel.HCA_I).all_topic_ids():
        e = complete_topic(e, c, topic_id=t)
    return e, c


def test_certify_passing_score():
    e, c = _eligible_enrolment()
    cert = certify(e, on=date(2026, 6, 1), exam_score_pct=0.80, curriculum=c)
    assert cert.status is EnrolmentStatus.CERTIFIED


def test_certify_failing_score_rejected():
    e, c = _eligible_enrolment()
    with pytest.raises(ValueError):
        certify(e, on=date(2026, 6, 1), exam_score_pct=0.50, curriculum=c)


def test_certify_invalid_score_rejected():
    e, c = _eligible_enrolment()
    with pytest.raises(ValueError):
        certify(e, on=date(2026, 6, 1), exam_score_pct=1.5, curriculum=c)


def test_certify_non_eligible_rejected():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    with pytest.raises(ValueError):
        certify(e, on=date(2026, 6, 1), exam_score_pct=0.80, curriculum=c)


def test_certify_before_enrolled_rejected():
    e, c = _eligible_enrolment()
    with pytest.raises(ValueError):
        certify(e, on=date(2026, 1, 1), exam_score_pct=0.80, curriculum=c)


# --- withdraw -------------------------------


def test_withdraw_enrolled():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    w = withdraw(e, on=date(2026, 6, 1))
    assert w.status is EnrolmentStatus.WITHDRAWN


def test_withdraw_certified_rejected():
    e, c = _eligible_enrolment()
    cert = certify(e, on=date(2026, 6, 1), exam_score_pct=0.80, curriculum=c)
    with pytest.raises(ValueError):
        withdraw(cert, on=date(2026, 7, 1))


def test_withdraw_before_enrolled_rejected():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    with pytest.raises(ValueError):
        withdraw(e, on=date(2026, 4, 1))


# --- Helpers --------------------------------


def test_is_exam_eligible():
    e, c = _eligible_enrolment()
    assert is_exam_eligible(e, c)


def test_is_exam_eligible_not_yet():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    assert not is_exam_eligible(e, c)


def test_highest_certified_level():
    e_i = LearnerEnrolment(
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 1, 1),
        status=EnrolmentStatus.CERTIFIED,
        certified_on=date(2026, 4, 1),
    )
    e_ii = LearnerEnrolment(
        enrolment_id="E2",
        learner_id="bob",
        level=CertLevel.HCA_II,
        enrolled_on=date(2026, 5, 1),
        status=EnrolmentStatus.CERTIFIED,
        certified_on=date(2026, 9, 1),
    )
    assert highest_certified_level([e_i, e_ii]) is CertLevel.HCA_II


def test_highest_certified_none_when_only_enrolled():
    e = LearnerEnrolment(
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    assert highest_certified_level([e]) is None


# --- Render --------------------------------


def test_render_enrolment_no_secret_leak():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob@example.com",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    out = render_enrolment(e, c)
    assert "bob@example.com" not in out


def test_render_enrolment_status_emoji():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    out = render_enrolment(e, c)
    assert "📖" in out


def test_render_enrolment_level_emoji():
    c = _curriculum()
    e = enrol(
        c,
        enrolment_id="E1",
        learner_id="bob",
        level=CertLevel.HCA_I,
        enrolled_on=date(2026, 5, 1),
    )
    out = render_enrolment(e, c)
    assert "1️⃣" in out
