from __future__ import annotations

from education.curriculum import SKILL_BY_ID
from education.models import MasteryRecord, Recommendation


def rank_recommendations(
    mastery_records: list[MasteryRecord],
    *,
    limit: int = 8,
) -> list[Recommendation]:
    observed = [item for item in mastery_records if item.attempts > 0]
    if not observed:
        return []
    recommendations: list[Recommendation] = []

    for record in observed:
        mastery = record.mastery
        attempts = record.attempts
        error_rate = min(1.0, record.recent_errors / max(1, attempts))
        gap = 1.0 - mastery
        score = round(0.7 * gap + 0.3 * error_rate, 4)
        skill = SKILL_BY_ID.get(record.skill_id)
        title = skill.title if skill else record.skill_id
        recommendations.append(
            Recommendation(
                skill_id=record.skill_id,
                subject=record.subject,
                title=title,
                score=score,
                mastery_gap=round(gap, 4),
                recent_error_rate=round(error_rate, 4),
                reason=(
                    f"Khoảng trống năng lực {gap:.0%}; tỷ lệ lỗi quan sát được "
                    f"{error_rate:.0%}."
                ),
            )
        )

    return sorted(recommendations, key=lambda item: item.score, reverse=True)[:limit]
