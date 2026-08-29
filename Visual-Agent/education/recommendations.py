from __future__ import annotations

from education.curriculum import SKILL_BY_ID
from education.models import ExamTrend, MasteryRecord, Recommendation


def rank_recommendations(
    mastery_records: list[MasteryRecord],
    exam_trends: list[ExamTrend],
    *,
    limit: int = 8,
) -> list[Recommendation]:
    observed = [item for item in mastery_records if item.attempts > 0]
    if not observed:
        return []
    trend_by_skill = {item.skill_id: item for item in exam_trends}
    recommendations: list[Recommendation] = []

    for record in observed:
        trend = trend_by_skill.get(record.skill_id)
        mastery = record.mastery
        attempts = record.attempts
        error_rate = min(1.0, record.recent_errors / max(1, attempts))
        gap = 1.0 - mastery
        prevalence = trend.weighted_prevalence if trend else 0.0
        score = round(0.5 * gap + 0.3 * prevalence + 0.2 * error_rate, 4)
        skill = SKILL_BY_ID.get(record.skill_id)
        title = skill.title if skill else record.skill_id
        recommendations.append(
            Recommendation(
                skill_id=record.skill_id,
                subject=record.subject,
                title=title,
                score=score,
                mastery_gap=round(gap, 4),
                exam_prevalence=round(prevalence, 4),
                recent_error_rate=round(error_rate, 4),
                reason=(
                    f"Khoảng trống năng lực {gap:.0%}; xuất hiện trong đề gần đây "
                    f"{prevalence:.0%}; tỷ lệ lỗi gần đây {error_rate:.0%}."
                ),
            )
        )

    return sorted(recommendations, key=lambda item: item.score, reverse=True)[:limit]
