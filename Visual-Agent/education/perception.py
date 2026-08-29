from __future__ import annotations

import base64
import re

from langchain_core.messages import HumanMessage, SystemMessage

from education.curriculum import skills_for_subject
from education.models import ProblemExtraction, Subject
from utils.model_factory import get_chat_model


SUBJECT_KEYWORDS = {
    Subject.MATH: ("toán", "hàm", "đạo hàm", "tích phân", "xác suất", "phương trình", "tọa độ"),
    Subject.PHYSICS: ("vật lí", "vận tốc", "điện", "từ trường", "nhiệt", "hạt nhân", "áp suất"),
    Subject.CHEMISTRY: ("hóa", "phản ứng", "mol", "ester", "polymer", "điện phân", "dung dịch"),
    Subject.BIOLOGY: ("sinh học", "di truyền", "gen", "nhiễm sắc thể", "tiến hóa", "sinh thái", "tế bào"),
}

SKILL_KEYWORDS = {
    "math.functions": ("hàm", "đồ thị"),
    "math.derivatives": ("đạo hàm", "cực trị", "đồng biến"),
    "math.integrals": ("tích phân", "nguyên hàm", "diện tích"),
    "math.spatial_geometry": ("không gian", "mặt phẳng", "thể tích"),
    "math.probability": ("xác suất", "thống kê"),
    "physics.thermal": ("nhiệt", "nhiệt độ"),
    "physics.gases": ("khí", "áp suất", "thể tích"),
    "physics.electromagnetism": ("từ trường", "cảm ứng", "điện từ"),
    "physics.ac": ("xoay chiều", "rms", "hiệu dụng"),
    "physics.nuclear": ("hạt nhân", "phóng xạ"),
    "chemistry.esters": ("ester", "lipid", "chất béo"),
    "chemistry.carbohydrates": ("glucose", "saccharose", "tinh bột", "carbohydrate"),
    "chemistry.amines": ("amine", "amino", "nitrogen", "nitơ"),
    "chemistry.polymers": ("polymer", "trùng hợp", "vật liệu"),
    "chemistry.electrochemistry": ("điện phân", "pin điện", "điện cực"),
    "biology.genetics": ("di truyền", "gen", "nhiễm sắc thể"),
    "biology.evolution": ("tiến hóa", "chọn lọc"),
    "biology.ecology": ("sinh thái", "quần thể", "hệ sinh thái"),
    "biology.cells": ("tế bào", "enzyme", "màng"),
    "biology.biotechnology": ("công nghệ sinh học", "dna tái tổ hợp", "pcr"),
}


def infer_subject(text: str) -> Subject:
    lowered = text.lower()
    scores = {
        subject: sum(keyword in lowered for keyword in keywords)
        for subject, keywords in SUBJECT_KEYWORDS.items()
    }
    subject, score = max(scores.items(), key=lambda item: item[1])
    return subject if score else Subject.UNKNOWN


def infer_skills(text: str, subject: Subject) -> list[str]:
    lowered = text.lower()
    found = [
        skill_id
        for skill_id, keywords in SKILL_KEYWORDS.items()
        if skill_id.startswith(f"{subject.value}.") and any(word in lowered for word in keywords)
    ]
    if found:
        return found[:3]
    candidates = skills_for_subject(subject)
    return [candidates[0].id] if candidates else []


class ProblemPerception:
    def __init__(self):
        self.model = get_chat_model("vision")

    async def extract(
        self,
        *,
        message: str,
        images: list[tuple[bytes, str]],
    ) -> ProblemExtraction:
        if not images:
            subject = infer_subject(message)
            return ProblemExtraction(
                subject=subject,
                problem_markdown=message.strip(),
                skill_ids=infer_skills(message, subject),
                confidence=0.78 if subject != Subject.UNKNOWN else 0.45,
            )

        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Đọc đề bài và phần bài làm trong ảnh. Giữ công thức bằng LaTeX, "
                    "không giải bài. Kết hợp yêu cầu của học sinh: " + message
                ),
            }
        ]
        for data, mime in images:
            encoded = base64.b64encode(data).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            )
        messages = [
            SystemMessage(
                content=(
                    "Bạn là bộ đọc đề STEM lớp 12 Việt Nam. Chỉ trích xuất nội dung, "
                    "không đưa lời giải. subject phải là math, physics, chemistry, "
                    "biology hoặc unknown. confidence phản ánh độ chắc chắn của OCR."
                )
            ),
            HumanMessage(content=content),
        ]
        try:
            structured = self.model.with_structured_output(
                ProblemExtraction, method="function_calling"
            )
            result = await structured.ainvoke(messages)
            if not result.skill_ids:
                result.skill_ids = infer_skills(result.problem_markdown, result.subject)
            return result
        except Exception:
            # Never fabricate a transcription when vision fails. The low-confidence
            # result sends the request to the clarification node.
            return ProblemExtraction(
                subject=infer_subject(message),
                problem_markdown=message.strip(),
                confidence=0.2,
            )


FULL_SOLUTION_PATTERN = re.compile(
    r"\b(lời giải|giải chi tiết|đáp án|solution|full answer)\b", re.IGNORECASE
)


def requests_full_solution(text: str) -> bool:
    return bool(FULL_SOLUTION_PATTERN.search(text))
