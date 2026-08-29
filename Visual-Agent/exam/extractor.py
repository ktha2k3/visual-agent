from __future__ import annotations

import base64

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from education.curriculum import skills_for_subject
from education.models import ExamQuestion, ExamSource
from utils.model_factory import get_chat_model


class ExamQuestionBatch(BaseModel):
    questions: list[ExamQuestion] = Field(default_factory=list)


class ExamExtractor:
    def __init__(self):
        self.model = get_chat_model("vision")

    async def extract_image(
        self, source: ExamSource, image: bytes, mime_type: str
    ) -> list[ExamQuestion]:
        allowed_skills = [skill.id for skill in skills_for_subject(source.subject)]
        encoded = base64.b64encode(image).decode("ascii")
        messages = [
            SystemMessage(
                content=(
                    "Bạn trích xuất câu hỏi từ ảnh đề thi, không tự giải và không bịa phần "
                    "bị mờ. Giữ công thức bằng LaTeX. Chỉ dùng skill_ids trong danh sách: "
                    f"{allowed_skills}. difficulty_basis luôn là expert_estimate, "
                    "review_status luôn pending. confidence dưới 0.65 nếu có phần không rõ."
                )
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": f"Môn {source.subject.value}, năm {source.exam_year}, mã {source.exam_code}."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                ]
            ),
        ]
        structured = self.model.with_structured_output(
            ExamQuestionBatch, method="function_calling"
        )
        batch = await structured.ainvoke(messages)
        normalized: list[ExamQuestion] = []
        for item in batch.questions:
            item.source_id = source.id
            item.exam_year = source.exam_year
            item.subject = source.subject
            item.exam_code = source.exam_code
            normalized.append(item)
        return normalized
