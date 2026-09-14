from education.learning_supervisor import build_learning_plan
from education.models import LearningMode, MasteryRecord, ProblemExtraction, Subject


def test_supervisor_uses_only_current_subject_and_skill_evidence():
    problem = ProblemExtraction(
        subject=Subject.MATH,
        problem_markdown="Đạo hàm là gì?",
        skill_ids=["math.derivatives"],
        confidence=0.95,
    )
    plan = build_learning_plan(
        problem=problem,
        message="Giúp mình hiểu sâu bản chất của đạo hàm",
        mastery_records=[
            MasteryRecord(
                user_id="student",
                skill_id="math.derivatives",
                subject=Subject.MATH,
                mastery=0.74,
                attempts=3,
            ),
            MasteryRecord(
                user_id="student",
                skill_id="physics.nuclear",
                subject=Subject.PHYSICS,
                mastery=0.1,
                attempts=10,
            ),
        ],
    )

    assert plan.intent == "understand_concept"
    assert plan.understanding_level == "proficient"
    assert plan.evidence_count == 3
    assert [item.level for item in plan.approaches] == [
        "foundation",
        "intermediate",
        "advanced",
    ]
    assert "cơ bản, trung bình, nâng cao" in plan.research_task
    assert plan.needs_research is True


def test_supervisor_does_not_invent_a_level_without_evidence():
    plan = build_learning_plan(
        problem=ProblemExtraction(
            subject=Subject.PHYSICS,
            problem_markdown="Giải thích định luật II Newton",
            skill_ids=["physics.electromagnetism"],
            confidence=0.9,
        ),
        message="Giải thích định luật II Newton",
        mastery_records=[],
    )

    assert plan.understanding_level == "insufficient_data"
    assert plan.understanding_label == "Chưa đủ dữ liệu"
    assert plan.evidence_count == 0
    assert plan.confidence == 0.1


def test_explicit_session_mode_overrides_message_classification():
    problem = ProblemExtraction(
        subject=Subject.MATH,
        problem_markdown="Đạo hàm",
        skill_ids=["math.derivatives"],
        confidence=0.9,
    )
    guided = build_learning_plan(
        problem=problem,
        message="Giải thích thật sâu khái niệm đạo hàm",
        mastery_records=[],
        learning_mode=LearningMode.GUIDED_PROBLEM,
    )
    concept = build_learning_plan(
        problem=problem,
        message="Giải bài đạo hàm này giúp em",
        mastery_records=[],
        learning_mode=LearningMode.CONCEPT,
    )

    assert guided.intent == "solve_problem"
    assert concept.intent == "understand_concept"


def test_concept_mode_researches_new_context_but_not_ambiguous_followup():
    problem = ProblemExtraction(
        subject=Subject.MATH,
        problem_markdown="Đạo hàm",
        skill_ids=["math.derivatives"],
        confidence=0.9,
    )
    first = build_learning_plan(
        problem=problem,
        message="Đạo hàm",
        mastery_records=[],
        learning_mode=LearningMode.CONCEPT,
    )
    followup = build_learning_plan(
        problem=problem,
        message="Em hiểu đoạn này rồi",
        mastery_records=[],
        learning_mode=LearningMode.CONCEPT,
        previous_plan=first.model_dump(mode="json"),
    )

    assert first.needs_research is True
    assert followup.needs_research is False
