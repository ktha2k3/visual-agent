from education.mastery import evidence_for_outcome, update_mastery
from education.models import MasteryRecord, Subject
from education.learning_profile import summarize_learning_profile
from education.recommendations import rank_recommendations


def test_mastery_formula_and_evidence_levels():
    assert evidence_for_outcome(answer_correct=True, hint_level=0, used_full_solution=False) == 1.0
    assert evidence_for_outcome(answer_correct=True, hint_level=1, used_full_solution=False) == 0.75
    assert evidence_for_outcome(answer_correct=True, hint_level=2, used_full_solution=False) == 0.5
    assert evidence_for_outcome(answer_correct=True, hint_level=3, used_full_solution=True) == 0.25
    assert evidence_for_outcome(answer_correct=False, hint_level=0, used_full_solution=False) == 0.0

    record, change = update_mastery(
        user_id="student",
        skill_id="math.derivative",
        subject=Subject.MATH,
        old_mastery=0.4,
        evidence=1.0,
    )
    assert record.mastery == 0.58
    assert change.old_mastery == 0.4


def test_recommendation_uses_required_weighting_and_sorts_descending():
    mastery = [
        MasteryRecord(
            user_id="student",
            skill_id="math.derivative",
            subject=Subject.MATH,
            mastery=0.2,
            attempts=4,
            recent_errors=3,
        )
    ]
    result = rank_recommendations(mastery)
    assert len(result) == 1
    assert result[0].skill_id == "math.derivative"
    assert result[0].score == 0.785
    assert "tỷ lệ lỗi" in result[0].reason


def test_empty_profile_never_creates_template_recommendations():
    assert rank_recommendations([]) == []
    summary = summarize_learning_profile([])
    assert summary.status == "insufficient_data"
    assert summary.overall_mastery is None
    assert summary.evidence_count == 0


def test_profile_summary_uses_only_observed_attempts():
    records = [
        MasteryRecord(
            user_id="student",
            skill_id="math.derivatives",
            subject=Subject.MATH,
            mastery=0.8,
            attempts=4,
        ),
        MasteryRecord(
            user_id="student",
            skill_id="math.integrals",
            subject=Subject.MATH,
            mastery=0.2,
            attempts=0,
        ),
    ]
    summary = summarize_learning_profile(records)
    assert summary.overall_mastery == 0.8
    assert summary.evidence_count == 4
    assert summary.observed_skills == 1
    assert "Đạo hàm" in summary.strengths[0]
