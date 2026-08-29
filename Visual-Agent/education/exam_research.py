from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from education.curriculum import SKILL_BY_ID
from education.models import (
    ExamResearchReport,
    MasteryRecord,
    PersonalizedExamPlan,
    ScoreBandStudyPlan,
    SkillExamDistribution,
    StudyPlanPriority,
    Subject,
)
from exam.sources import latest_completed_exam_year, rolling_completed_years
from services.repository import TutorRepository
from utils.config import get_settings


class ExamResearchAgent:
    """Analyze only approved exam questions from the two latest completed exams."""

    def __init__(
        self,
        repository: TutorRepository,
        *,
        now_provider: Callable[[], datetime] | None = None,
        timezone_name: str | None = None,
    ):
        self.repository = repository
        self.timezone_name = timezone_name or get_settings().app_timezone
        self.timezone = ZoneInfo(self.timezone_name)
        self.now_provider = now_provider or (lambda: datetime.now(self.timezone))

    def _now(self) -> datetime:
        current = self.now_provider()
        if current.tzinfo is None:
            return current.replace(tzinfo=self.timezone)
        return current.astimezone(self.timezone)

    async def research(self, subject: Subject) -> ExamResearchReport:
        current = self._now()
        cutoff = latest_completed_exam_year(current.date())
        approved = await self.repository.list_approved_exam_questions([subject])
        eligible = [item for item in approved if item.exam_year <= cutoff]
        years = rolling_completed_years(cutoff, [item.exam_year for item in eligible])
        selected = [item for item in eligible if item.exam_year in years]
        coverage = Counter(item.exam_year for item in selected)

        if not selected:
            return ExamResearchReport(
                subject=subject,
                generated_at=current,
                timezone=self.timezone_name,
                latest_completed_year=cutoff,
                status="insufficient_data",
                approved_question_count=0,
                notes=[
                    "Chưa có câu hỏi đã duyệt thuộc các kỳ thi đã hoàn tất; không tạo phân bổ hoặc lộ trình giả định."
                ],
            )

        cognitive_counts = Counter(item.cognitive_level.value for item in selected)
        difficulty_counts = Counter(item.difficulty_estimate for item in selected)
        skill_year_weights: dict[str, dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        skill_questions: dict[str, list] = defaultdict(list)
        uncategorized = 0

        for question in selected:
            valid_skills = list(
                dict.fromkeys(
                    skill_id
                    for skill_id in question.skill_ids
                    if skill_id in SKILL_BY_ID
                    and SKILL_BY_ID[skill_id].subject == subject
                )
            )
            if not valid_skills:
                uncategorized += 1
                continue
            fractional_weight = 1.0 / len(valid_skills)
            for skill_id in valid_skills:
                skill_year_weights[skill_id][question.exam_year] += fractional_weight
                skill_questions[skill_id].append(question)

        year_weights = {year: (0.6 if index == 0 else 0.4) for index, year in enumerate(years)}
        if len(years) == 1:
            year_weights[years[0]] = 1.0

        distributions: list[SkillExamDistribution] = []
        for skill_id, by_year in skill_year_weights.items():
            year_prevalence = {
                year: round(by_year.get(year, 0.0) / max(coverage[year], 1), 4)
                for year in years
            }
            weighted = sum(year_weights[year] * year_prevalence[year] for year in years)
            related = skill_questions[skill_id]
            skill = SKILL_BY_ID[skill_id]
            distributions.append(
                SkillExamDistribution(
                    skill_id=skill_id,
                    title=skill.title,
                    question_count=len(related),
                    weighted_prevalence=round(weighted, 4),
                    year_prevalence=year_prevalence,
                    cognitive_counts=dict(
                        Counter(item.cognitive_level.value for item in related)
                    ),
                    average_difficulty=round(
                        sum(item.difficulty_estimate for item in related) / len(related), 2
                    ),
                )
            )
        distributions.sort(
            key=lambda item: (item.weighted_prevalence, item.question_count),
            reverse=True,
        )

        source_ids = list(
            dict.fromkeys(item.source_id for item in selected if item.source_id)
        )
        sources = await self.repository.list_exam_sources(source_ids)
        notes = []
        if len(years) < 2:
            notes.append(
                "Mới có dữ liệu đã duyệt của một năm; kết quả là tạm thời và chưa đủ để so sánh hai kỳ thi."
            )
        if uncategorized:
            notes.append(
                f"Có {uncategorized} câu chưa gắn kỹ năng CTGDPT 2018 và không được phân bổ vào chủ đề."
            )
        if not sources:
            notes.append("Các câu đã duyệt chưa liên kết được metadata nguồn đề.")

        total = len(selected)
        return ExamResearchReport(
            subject=subject,
            generated_at=current,
            timezone=self.timezone_name,
            latest_completed_year=cutoff,
            years=years,
            status="ready" if len(years) == 2 else "partial",
            approved_question_count=total,
            coverage_by_year=dict(coverage),
            cognitive_distribution={
                key: round(value / total, 4) for key, value in cognitive_counts.items()
            },
            difficulty_distribution={
                key: round(value / total, 4) for key, value in difficulty_counts.items()
            },
            skills=distributions,
            sources=sources,
            notes=notes,
        )


@dataclass(frozen=True)
class _ScoreBand:
    label: str
    minimum: float
    maximum: float
    target_mastery: float
    target_difficulty: float
    cognitive_focus: tuple[str, ...]


_SCORE_BANDS = (
    _ScoreBand("0–5", 0.0, 5.0, 0.65, 1.5, ("recognition",)),
    _ScoreBand("5–7", 5.0, 7.0, 0.75, 2.3, ("recognition", "understanding")),
    _ScoreBand("7–8.5", 7.0, 8.5, 0.85, 3.2, ("understanding", "application")),
    _ScoreBand("8.5–10", 8.5, 10.0, 0.92, 4.2, ("application",)),
)


class LearningPathSupervisor:
    """Turn the research document and observed mastery into score-band plans."""

    def __init__(self, repository: TutorRepository, research_agent: ExamResearchAgent | None = None):
        self.repository = repository
        self.research_agent = research_agent or ExamResearchAgent(repository)

    async def build_plan(
        self,
        user_id: str,
        subject: Subject,
        *,
        target_score: float | None = None,
    ) -> PersonalizedExamPlan:
        report = await self.research_agent.research(subject)
        if report.status == "insufficient_data" or not report.skills:
            return PersonalizedExamPlan(research=report)

        mastery = {
            item.skill_id: item
            for item in await self.repository.get_mastery(user_id)
            if item.subject == subject and item.attempts > 0
        }
        selected_bands = [
            band
            for band in _SCORE_BANDS
            if target_score is None
            or (
                target_score <= band.maximum
                and (band.minimum == 0 or target_score > band.minimum)
            )
        ]
        if target_score is not None and selected_bands:
            selected_bands = selected_bands[:1]

        plans = [self._build_band(band, report, mastery) for band in selected_bands]
        return PersonalizedExamPlan(research=report, score_bands=plans)

    def _build_band(
        self,
        band: _ScoreBand,
        report: ExamResearchReport,
        mastery: dict[str, MasteryRecord],
    ) -> ScoreBandStudyPlan:
        priorities: list[StudyPlanPriority] = []
        for distribution in report.skills:
            record = mastery.get(distribution.skill_id)
            current_mastery = record.mastery if record else None
            mastery_gap = (
                max(0.0, band.target_mastery - current_mastery)
                if current_mastery is not None
                else 0.5
            )
            difficulty_fit = max(
                0.0,
                1.0 - abs(distribution.average_difficulty - band.target_difficulty) / 4.0,
            )
            cognitive_total = max(sum(distribution.cognitive_counts.values()), 1)
            cognitive_fit = sum(
                distribution.cognitive_counts.get(level, 0)
                for level in band.cognitive_focus
            ) / cognitive_total
            score = min(
                1.0,
                0.5 * distribution.weighted_prevalence
                + 0.3 * mastery_gap
                + 0.1 * difficulty_fit
                + 0.1 * cognitive_fit,
            )
            mastery_text = (
                f"năng lực quan sát {current_mastery:.0%}"
                if current_mastery is not None
                else "chưa có đủ bằng chứng năng lực"
            )
            priorities.append(
                StudyPlanPriority(
                    skill_id=distribution.skill_id,
                    title=distribution.title,
                    priority_score=round(score, 4),
                    exam_prevalence=distribution.weighted_prevalence,
                    current_mastery=current_mastery,
                    target_mastery=band.target_mastery,
                    reason=(
                        f"Tỷ trọng đề hai năm {distribution.weighted_prevalence:.0%}; "
                        f"{mastery_text}; độ khó trung bình {distribution.average_difficulty:.1f}/5."
                    ),
                )
            )
        priorities.sort(key=lambda item: item.priority_score, reverse=True)
        priorities = priorities[:6]
        lead = ", ".join(item.title for item in priorities[:3])
        observed = sum(item.current_mastery is not None for item in priorities)
        research_factor = 1.0 if report.status == "ready" else 0.6
        confidence = research_factor * min(1.0, observed / max(len(priorities), 1))
        return ScoreBandStudyPlan(
            band=band.label,
            min_score=band.minimum,
            max_score=band.maximum,
            cognitive_focus=list(band.cognitive_focus),
            target_mastery=band.target_mastery,
            priorities=priorities,
            steps=[
                f"Xác định lỗ hổng qua hội thoại và củng cố các chủ đề ưu tiên: {lead}.",
                "Luyện theo từng mức nhận thức, chỉ chuyển mức khi đạt ngưỡng mastery mục tiêu.",
                "Làm đề có giới hạn thời gian, ghi lại lỗi sai theo kỹ năng và cập nhật hồ sơ năng lực.",
                "Thi thử lại sau một chu kỳ ôn và điều chỉnh thứ tự chủ đề theo bằng chứng mới.",
            ],
            personalization_confidence=round(confidence, 4),
        )
