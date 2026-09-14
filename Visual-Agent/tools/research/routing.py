from __future__ import annotations

import re
import unicodedata


_RESEARCH_INTENT = re.compile(
    r"\b(tra cứu|tìm nguồn|bài (toán )?tương tự|cách giải|cách làm|phương pháp|"
    r"giúp\s+(mình|em|tôi)?\s*giải|giải (bài|câu)|chứng minh|solve|how to solve)\b",
    re.IGNORECASE,
)
_CALCULATE_REQUEST = re.compile(
    r"^\s*(môn\s+[^.:,-]+[.:,-]?\s*)?(hãy\s+)?(tính|tìm|xác định)\b",
    re.IGNORECASE,
)

_CONCEPT_INTENT = re.compile(
    r"(giải thích|hiểu sâu|bản chất|trực giác|tại sao|vì sao|khái niệm|"
    r"ý nghĩa (vật lí|hình học|của)|là gì|từ cơ bản đến nâng cao|"
    r"explain|intuition|concept)",
    re.IGNORECASE,
)


def _normalize(message: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", message).casefold().split())


def is_concept_learning_request(message: str) -> bool:
    """Identify requests whose primary goal is understanding, not an answer."""
    return bool(_CONCEPT_INTENT.search(_normalize(message)))


def is_problem_solving_request(message: str) -> bool:
    normalized = _normalize(message)
    return bool(
        _RESEARCH_INTENT.search(normalized)
        or _CALCULATE_REQUEST.search(normalized)
    )


def should_research_problem(message: str) -> bool:
    """Research explicit solving, concept-learning, or exercise-progression requests."""
    return is_problem_solving_request(message) or is_concept_learning_request(message)
