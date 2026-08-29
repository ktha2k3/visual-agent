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
    r"công thức|hiện tượng|trực quan|minh họa|mô phỏng|visuali[sz]e|explain|"
    r"what is|why)\b",
    re.IGNORECASE,
)
_CALCULATION_ONLY = re.compile(
    r"^\s*(môn\s+(toán|vật\s*l[ií])\s*[.:,-]?\s*)?"
    r"(tính|giải|rút gọn|chứng minh|tìm)\b",
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
    if _CONCEPT_INTENT.search(normalized):
        return VisualizationDecision(True, value, "concept_or_formula_explanation")
    if _CALCULATION_ONLY.search(normalized):
        return VisualizationDecision(False, value, "calculation_without_visual_intent")
    return VisualizationDecision(False, value, "no_explanation_intent")

