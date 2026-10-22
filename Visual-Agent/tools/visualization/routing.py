from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from education.models import Subject


@dataclass(frozen=True)
class VisualizationDecision:
    should_visualize: bool
    subject: str | None
    reason: str


_CONCEPT_INTENT = re.compile(
    r"\b(giải thích|là gì|tại sao|vì sao|ý nghĩa|định luật|định lý|quy tắc|"
    r"công thức|hiện tượng|trực quan|minh họa|minh hoạ|mô phỏng|vẽ|ve do thi|mo phong|minh hoa|visuali[sz]e|explain|"
    r"what is|why)\b",
    re.IGNORECASE,
)
_CALCULATION_ONLY = re.compile(
    r"^\s*(môn\s+(toán|vật\s*l[ií])\s*[.:,-]?\s*)?"
    r"(tính|giải|rút gọn|chứng minh|tìm)\b",
    re.IGNORECASE,
)
_PICTURE_REQUEST = re.compile(
    r"\b(?:đưa ra|cung cấp|hiển thị|tạo|cho (?:em|mình|tôi|tớ|chúng tôi))\s+"
    r"(?:(?:một|cái|bức|các)\s+)*(?:hình|ảnh|đồ thị|sơ đồ)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def decide_visualization(question: str, subject: Subject | str) -> VisualizationDecision:
    """Route concept/formula/law questions without spending another model call."""
    value = subject.value if isinstance(subject, Subject) else str(subject)
    if value not in {Subject.MATH.value, Subject.PHYSICS.value}:
        return VisualizationDecision(False, None, "subject_not_supported")
    normalized = _normalize(question)
    if re.search(r"(?:không|đừng|khỏi)\s+(?:(?:cần|đưa ra|cung cấp|hiển thị|tạo)\s+)?(?:vẽ|minh họa|minh hoạ|mô phỏng|giải thích|hình|ảnh|đồ thị|sơ đồ)|(?:no|without)\s+(?:animation|illustration)", normalized):
        return VisualizationDecision(False, value, "visualization_declined")
    if _PICTURE_REQUEST.search(normalized):
        return VisualizationDecision(True, value, "explicit_picture_request")
    if _CONCEPT_INTENT.search(normalized):
        return VisualizationDecision(True, value, "concept_or_formula_explanation")
    if _CALCULATION_ONLY.search(normalized):
        return VisualizationDecision(False, value, "calculation_without_visual_intent")
    return VisualizationDecision(False, value, "no_explanation_intent")

