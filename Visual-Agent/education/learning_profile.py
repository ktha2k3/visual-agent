from __future__ import annotations

from education.curriculum import SKILL_BY_ID
from education.models import LearningProfileSummary, MasteryRecord


def summarize_learning_profile(
    records: list[MasteryRecord],
) -> LearningProfileSummary:
    """Build a transparent profile using only observed assessment evidence."""
    observed = [record for record in records if record.attempts > 0]
    evidence_count = sum(record.attempts for record in observed)
    if not observed or evidence_count == 0:
        return LearningProfileSummary(
            status="insufficient_data",
            level="Chưa đủ dữ liệu",
            confidence=0.0,
            evidence_count=0,
            observed_skills=0,
            narrative=(
                "Hồ sơ chưa có câu trả lời nào đủ điều kiện đánh giá. "
                "Hãy trình bày cách làm trong các phiên chat để hệ thống thu thập bằng chứng thực tế."
            ),
        )

    weighted_total = sum(record.mastery * record.attempts for record in observed)
    overall = round(weighted_total / evidence_count, 4)
    breadth = min(1.0, len(observed) / 4)
    volume = min(1.0, evidence_count / 10)
    confidence = round(0.7 * volume + 0.3 * breadth, 4)

    if evidence_count < 3:
        status = "collecting"
        level = "Đang thu thập dữ liệu"
    elif overall < 0.4:
        status = "assessed"
        level = "Cần củng cố"
    elif overall < 0.6:
        status = "assessed"
        level = "Đang hình thành"
    elif overall < 0.8:
        status = "assessed"
        level = "Khá"
    else:
        status = "assessed"
        level = "Vững"

    ranked = sorted(observed, key=lambda item: item.mastery, reverse=True)
    strengths = [
        SKILL_BY_ID[item.skill_id].title
        for item in ranked
        if item.mastery >= 0.7 and item.skill_id in SKILL_BY_ID
    ][:3]
    focus_areas = [
        SKILL_BY_ID[item.skill_id].title
        for item in reversed(ranked)
        if (item.mastery < 0.6 or item.recent_errors > 0)
        and item.skill_id in SKILL_BY_ID
    ][:3]
    qualifier = (
        "Kết quả còn sơ bộ" if status == "collecting" else "Kết quả được tổng hợp"
    )
    return LearningProfileSummary(
        status=status,
        level=level,
        overall_mastery=overall,
        confidence=confidence,
        evidence_count=evidence_count,
        observed_skills=len(observed),
        strengths=strengths,
        focus_areas=focus_areas,
        narrative=(
            f"{qualifier} từ {evidence_count} câu trả lời có thể đánh giá "
            f"trên {len(observed)} kỹ năng trong các phiên học."
        ),
    )
