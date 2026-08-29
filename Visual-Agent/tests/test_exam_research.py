from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from education.exam_research import ExamResearchAgent, LearningPathSupervisor
from education.models import (
    CognitiveLevel,
    ExamQuestion,
    ExamSource,
    MasteryRecord,
    ReviewStatus,
    Subject,
)
from services.repository import MemoryRepository
from tools.research.search_tools import current_datetime


async def _seed_two_exam_years(repository: MemoryRepository) -> None:
    source_2025 = await repository.save_exam_source(
        ExamSource(
            url="https://vqa.moet.gov.vn/math-2025",
            publisher="MOET",
            exam_year=2025,
            subject=Subject.MATH,
        )
    )
    source_2026 = await repository.save_exam_source(
        ExamSource(
            url="https://vqa.moet.gov.vn/math-2026",
            publisher="MOET",
            exam_year=2026,
            subject=Subject.MATH,
        )
    )
    questions = [
        ExamQuestion(
            source_id=source_2026.id,
            exam_year=2026,
            subject=Subject.MATH,
            question_number="1",
            stem_markdown="Đạo hàm",
            skill_ids=["math.derivatives"],
            cognitive_level=CognitiveLevel.UNDERSTANDING,
            difficulty_estimate=2,
            review_status=ReviewStatus.APPROVED,
        ),
        ExamQuestion(
            source_id=source_2026.id,
            exam_year=2026,
            subject=Subject.MATH,
            question_number="2",
            stem_markdown="Tích phân vận dụng",
            skill_ids=["math.integrals"],
            cognitive_level=CognitiveLevel.APPLICATION,
            difficulty_estimate=4,
            review_status=ReviewStatus.APPROVED,
        ),
        ExamQuestion(
            source_id=source_2025.id,
            exam_year=2025,
            subject=Subject.MATH,
            question_number="1",
            stem_markdown="Đạo hàm cơ bản",
            skill_ids=["math.derivatives"],
            cognitive_level=CognitiveLevel.RECOGNITION,
            difficulty_estimate=2,
            review_status=ReviewStatus.APPROVED,
        ),
        ExamQuestion(
            exam_year=2027,
            subject=Subject.MATH,
            question_number="future",
            stem_markdown="Không được tính",
            skill_ids=["math.functions"],
            review_status=ReviewStatus.APPROVED,
        ),
        ExamQuestion(
            exam_year=2026,
            subject=Subject.MATH,
            question_number="pending",
            stem_markdown="Chưa duyệt",
            skill_ids=["math.functions"],
            review_status=ReviewStatus.PENDING,
        ),
    ]
    await repository.save_exam_questions(questions)


@pytest.mark.asyncio
async def test_research_uses_current_time_and_only_two_completed_approved_years():
    repository = MemoryRepository()
    await _seed_two_exam_years(repository)
    now = datetime(2026, 8, 25, 9, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    report = await ExamResearchAgent(
        repository,
        now_provider=lambda: now,
        timezone_name="Asia/Ho_Chi_Minh",
    ).research(Subject.MATH)

    assert report.generated_at == now
    assert report.latest_completed_year == 2026
    assert report.years == [2026, 2025]
    assert report.status == "ready"
    assert report.approved_question_count == 3
    assert report.coverage_by_year == {2026: 2, 2025: 1}
    by_skill = {item.skill_id: item for item in report.skills}
    assert by_skill["math.derivatives"].weighted_prevalence == 0.7
    assert by_skill["math.integrals"].weighted_prevalence == 0.3
    assert "math.functions" not in by_skill
    assert {item.exam_year for item in report.sources} == {2025, 2026}


@pytest.mark.asyncio
async def test_supervisor_personalizes_the_requested_score_band():
    repository = MemoryRepository()
    await _seed_two_exam_years(repository)
    await repository.upsert_mastery(
        MasteryRecord(
            user_id="student",
            skill_id="math.derivatives",
            subject=Subject.MATH,
            mastery=0.95,
            attempts=5,
        )
    )
    await repository.upsert_mastery(
        MasteryRecord(
            user_id="student",
            skill_id="math.integrals",
            subject=Subject.MATH,
            mastery=0.2,
            attempts=5,
            recent_errors=4,
        )
    )
    research = ExamResearchAgent(
        repository,
        now_provider=lambda: datetime(2026, 8, 25),
    )
    plan = await LearningPathSupervisor(repository, research).build_plan(
        "student", Subject.MATH, target_score=9.0
    )

    assert [item.band for item in plan.score_bands] == ["8.5–10"]
    assert plan.score_bands[0].priorities[0].skill_id == "math.integrals"
    assert plan.score_bands[0].personalization_confidence > 0


@pytest.mark.asyncio
async def test_research_does_not_invent_a_plan_without_reviewed_questions():
    repository = MemoryRepository()
    plan = await LearningPathSupervisor(repository).build_plan(
        "new-student", Subject.BIOLOGY
    )
    assert plan.research.status == "insufficient_data"
    assert plan.score_bands == []


def test_research_time_tool_exposes_timezone_and_offset():
    value = current_datetime.invoke({})
    assert "Asia/Ho_Chi_Minh" in value
    assert "+07:00" in value
