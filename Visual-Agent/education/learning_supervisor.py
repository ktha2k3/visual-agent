from __future__ import annotations

from education.curriculum import SKILL_BY_ID
from education.models import (
    LearningApproach,
    LearningMode,
    LearningPlan,
    MasteryRecord,
    ProblemExtraction,
    Subject,
)
from tools.research.routing import (
    is_concept_learning_request,
    is_problem_solving_request,
    should_research_problem,
)


LEVEL_LABELS = {
    "insufficient_data": "Chưa đủ dữ liệu",
    "foundation": "Đang xây nền tảng",
    "developing": "Đang phát triển",
    "proficient": "Khá vững",
    "advanced": "Nâng cao",
}


def _understanding_level(score: float) -> str:
    if score >= 0.85:
        return "advanced"
    if score >= 0.7:
        return "proficient"
    if score >= 0.45:
        return "developing"
    return "foundation"


def _concept_name(problem: ProblemExtraction) -> str:
    for skill_id in problem.skill_ids:
        skill = SKILL_BY_ID.get(skill_id)
        if skill:
            return skill.title
    concise = " ".join(problem.problem_markdown.split())
    return concise[:120] or ("Toán học" if problem.subject == Subject.MATH else "Vật lí")


def build_learning_plan(
    *,
    problem: ProblemExtraction,
    message: str,
    mastery_records: list[MasteryRecord],
    learning_mode: LearningMode | None = None,
    previous_plan: dict | None = None,
) -> LearningPlan:
    """Create a grounded supervisor plan from persisted learning evidence."""
    previous_intent = (previous_plan or {}).get("intent")
    if learning_mode == LearningMode.CONCEPT:
        intent = "understand_concept"
    elif learning_mode == LearningMode.GUIDED_PROBLEM:
        intent = "solve_problem"
    elif is_concept_learning_request(message):
        intent = "understand_concept"
    elif is_problem_solving_request(message):
        intent = "solve_problem"
    elif previous_intent in {"understand_concept", "solve_problem"}:
        intent = previous_intent
    else:
        intent = "solve_problem"

    subject_records = [
        item
        for item in mastery_records
        if item.subject == problem.subject and item.attempts > 0
    ]
    exact_records = [
        item for item in subject_records if item.skill_id in problem.skill_ids
    ]
    evidence = exact_records or subject_records
    evidence_count = sum(item.attempts for item in evidence)
    if evidence_count:
        score = sum(item.mastery * item.attempts for item in evidence) / evidence_count
        level = _understanding_level(score)
        confidence = min(0.95, 0.35 + 0.08 * evidence_count)
        if not exact_records:
            confidence *= 0.7
            scope = "hồ sơ chung của môn"
        else:
            scope = "các lần trao đổi và làm bài về kỹ năng liên quan"
        assessment = (
            f"Ước lượng từ {evidence_count} bằng chứng trong {scope}; "
            f"mức làm chủ hiện tại khoảng {round(score * 100)}%."
        )
    else:
        level = "insufficient_data"
        confidence = 0.1
        assessment = (
            "Chưa có câu trả lời đủ rõ để đánh giá. Trợ giảng sẽ bắt đầu bằng một "
            "câu kiểm tra ngắn và cập nhật sau các lượt học."
        )

    concept = _concept_name(problem)
    subject_label = "Toán" if problem.subject == Subject.MATH else "Vật lí"
    approaches = [
        LearningApproach(
            level="foundation",
            title="Nền tảng",
            focus=f"Làm rõ trực giác, định nghĩa và kiến thức tiên quyết của {concept}.",
        ),
        LearningApproach(
            level="intermediate",
            title="Vận dụng chuẩn",
            focus=(
                f"Nhận diện dạng điển hình, chọn công thức/phương pháp và giải thích từng bước trong {concept}."
            ),
        ),
        LearningApproach(
            level="advanced",
            title="Mở rộng",
            focus=(
                f"So sánh biến thể, xử lý bài nhiều bước và chuyển kiến thức {concept} sang tình huống mới."
            ),
        ),
    ]
    research_task = (
        f"Tra cứu cho môn {subject_label}, khái niệm/chủ đề '{concept}'. Xác minh kiến thức nền "
        "và tìm các dạng bài tập tiêu biểu theo ba mức: cơ bản, trung bình, nâng cao. Với mỗi "
        "mức, nêu mục tiêu học, dấu hiệu nhận dạng, cách tiếp cận, một đề bài mẫu ngắn và URL "
        "nguồn đáng tin cậy; không lấy đáp án từ ngân hàng đề thi."
    )
    previous_concept = (previous_plan or {}).get("concept")
    concept_context_changed = (
        not previous_plan
        or previous_intent != intent
        or previous_concept != concept
    )
    needs_research = should_research_problem(message) or (
        intent == "understand_concept" and concept_context_changed
    )
    return LearningPlan(
        subject=problem.subject,
        concept=concept,
        intent=intent,
        understanding_level=level,
        understanding_label=LEVEL_LABELS[level],
        confidence=round(confidence, 3),
        evidence_count=evidence_count,
        assessment_summary=assessment,
        approaches=approaches,
        research_task=research_task,
        needs_research=needs_research,
    )
