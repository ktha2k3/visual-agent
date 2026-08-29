from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from education.curriculum import SKILL_BY_ID, skills_for_subject
from education.mastery import evidence_for_outcome, update_mastery
from education.models import (
    Citation,
    MasteryUpdate,
    ProblemExtraction,
    Recommendation,
    TutorResponse,
)
from education.perception import ProblemPerception, requests_full_solution
from education.recommendations import rank_recommendations
from services.repository import TutorRepository
from services.storage import AssetStorage
from tools.visualization.routing import decide_visualization
from tools.visualization.service import StemVisualizationService
from utils.model_factory import get_chat_model, get_embedding_model


MAX_CONVERSATION_MESSAGES = 24


def merge_conversation(
    current: list[dict[str, str]] | None,
    incoming: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Append serializable chat messages and retain a bounded per-thread window."""
    combined = list(current or [])
    for item in incoming or []:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            combined.append({"role": role, "content": content})
    return combined[-MAX_CONVERSATION_MESSAGES:]


class TutorAssessment(BaseModel):
    assessable: bool = False
    answer_correct: bool | None = None
    misconception: str | None = None
    feedback_type: str = "guidance"
    skill_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TutorState(TypedDict, total=False):
    user_id: str
    session_id: str
    message: str
    attachment_ids: list[str]
    knowledge_document_ids: list[str]
    problem: dict[str, Any]
    problem_id: str
    needs_clarification: bool
    citations: list[dict[str, Any]]
    hint_level: int
    tutor_response: dict[str, Any]
    mastery_updates: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    conversation: Annotated[list[dict[str, str]], merge_conversation]
    chat_assessment: dict[str, Any] | None
    assessed_skill_ids: list[str]
    visualization_decision: dict[str, Any]
    visualization: dict[str, Any]


SUBJECT_RULES = {
    "math": "Kiểm tra biến đổi bằng đại số; ưu tiên biểu diễn LaTeX rõ ràng.",
    "physics": "Kiểm tra đơn vị, chiều và ý nghĩa vật lí trước khi thay số.",
    "chemistry": "Kiểm tra cân bằng phản ứng và bảo toàn trước khi tính toán.",
    "biology": "Phân biệt dữ kiện với suy luận; ưu tiên giải thích đúng kiến thức phổ thông.",
}


def build_tutor_graph(
    repository: TutorRepository,
    storage: AssetStorage,
    *,
    checkpointer=None,
    visualization_service: StemVisualizationService | None = None,
):
    perception_service = ProblemPerception()
    tutor_model = get_chat_model("tutor")
    embedding_model = get_embedding_model()
    visualizer = visualization_service or StemVisualizationService()

    async def perception(state: TutorState) -> dict:
        attachment_ids = state.get("attachment_ids", [])
        previous = state.get("problem")
        if not attachment_ids and previous:
            if state.get("needs_clarification") and state.get("message", "").strip():
                corrected = await perception_service.extract(
                    message=state["message"], images=[]
                )
                corrected_data = corrected.model_dump(mode="json")
                problem_id = await repository.save_problem(
                    state["user_id"], state["session_id"], corrected_data, None
                )
                return {
                    "problem": corrected_data,
                    "problem_id": problem_id,
                    "needs_clarification": corrected.confidence < 0.65,
                    "hint_level": 0,
                }
            return {"needs_clarification": False}
        images: list[tuple[bytes, str]] = []
        for asset_id in attachment_ids:
            record = await repository.get_asset(asset_id, state["user_id"])
            if not record:
                continue
            images.append((await storage.read(record), record.mime_type))
        problem = await perception_service.extract(
            message=state.get("message", ""), images=images
        )
        problem_data = problem.model_dump(mode="json")
        problem_id = await repository.save_problem(
            state["user_id"], state["session_id"], problem_data,
            attachment_ids[0] if attachment_ids else None,
        )
        return {
            "problem": problem_data,
            "problem_id": problem_id,
            "needs_clarification": problem.confidence < 0.65,
            "hint_level": 0,
        }

    def route_after_perception(state: TutorState) -> str:
        return (
            "clarification"
            if state.get("needs_clarification")
            else "visualization_router"
        )

    async def visualization_decision(state: TutorState) -> dict:
        problem = ProblemExtraction.model_validate(state["problem"])
        decision = decide_visualization(state.get("message", ""), problem.subject)
        return {
            "visualization_decision": {
                "should_visualize": decision.should_visualize,
                "subject": decision.subject,
                "reason": decision.reason,
            },
            "visualization": {},
        }

    def route_after_visualization_decision(state: TutorState) -> str:
        decision = state.get("visualization_decision", {})
        return "visualization_tool" if decision.get("should_visualize") else "retrieval"

    async def visualization(state: TutorState) -> dict:
        decision = state.get("visualization_decision", {})
        writer = get_stream_writer()

        def report(stage: str) -> None:
            writer({"type": "visualization_progress", "stage": stage})

        result = await visualizer.visualize(
            question=state.get("message", ""),
            subject=str(decision["subject"]),
            user_id=state["user_id"],
            level="THPT lớp 12",
            progress=report,
        )
        return {"visualization": result.model_dump(mode="json")}

    async def clarification(state: TutorState) -> dict:
        problem = ProblemExtraction.model_validate(state.get("problem", {}))
        transcription = problem.problem_markdown or "(chưa đọc được nội dung)"
        response = TutorResponse(
            message=(
                "Mình chưa đọc đề đủ chắc chắn. Bạn kiểm tra và gửi lại phần đề dưới "
                f"đây, hoặc chụp ảnh rõ hơn:\n\n> {transcription}"
            ),
            hint_level=0,
            feedback_type="clarification",
        )
        if state.get("problem_id"):
            await repository.save_hint_event(
                state["user_id"], state["session_id"], state["problem_id"],
                response.model_dump(mode="json"),
            )
        return {
            "tutor_response": response.model_dump(mode="json"),
            "conversation": [{"role": "assistant", "content": response.message}],
        }

    async def retrieval(state: TutorState) -> dict:
        problem = ProblemExtraction.model_validate(state["problem"])
        try:
            query_embedding = await embedding_model.aembed_query(problem.problem_markdown)
        except Exception:
            query_embedding = None
        citations = await repository.retrieve_context(
            problem.subject,
            problem.problem_markdown,
            problem.skill_ids,
            query_embedding=query_embedding,
            user_id=state["user_id"],
            document_ids=state.get("knowledge_document_ids"),
        )
        return {"citations": [item.model_dump(mode="json") for item in citations]}

    async def tutor(state: TutorState) -> dict:
        problem = ProblemExtraction.model_validate(state["problem"])
        citations = [Citation.model_validate(item) for item in state.get("citations", [])]
        previous_hint = int(state.get("hint_level", 0))
        message = state.get("message", "")
        allow_full = previous_hint >= 3 and requests_full_solution(message)
        first_tutor_turn = not bool(state.get("tutor_response"))
        target_hint = 4 if allow_full else min(3, previous_hint + 1)
        if first_tutor_turn:
            target_hint = 0

        context_text = "\n\n".join(
            f"[{index + 1}] {item.title}: {item.excerpt or ''} ({item.url})"
            for index, item in enumerate(citations)
        )
        conversation = state.get("conversation", [])
        # The current user message has already been appended by the graph input.
        prior_conversation = conversation[:-1] if conversation else []
        history_text = "\n".join(
            f"{'Học sinh' if item['role'] == 'user' else 'Trợ giảng'}: {item['content']}"
            for item in prior_conversation
        )
        system = (
            "Bạn là trợ giảng STEM lớp 12 theo phương pháp Socrates. Không tiết lộ "
            "chain-of-thought. Trả lời tiếng Việt, ngắn và hỏi đúng một câu dẫn dắt. "
            f"Mức gợi ý cho phép: {target_hint}/4. "
            + ("Được phép đưa lời giải đầy đủ." if allow_full else "Không đưa lời giải đầy đủ hoặc đáp án cuối.")
            + "\n"
            + SUBJECT_RULES.get(problem.subject.value, "")
            + " Bạn có thể dùng kiến thức chuẩn của chương trình THPT để hướng dẫn. "
            "Học liệu truy xuất là ngữ cảnh tham khảo ưu tiên khi có. Không tự động mở đầu "
            "hoặc lặp lại rằng học liệu thiếu hay không chứa nội dung. Chỉ nêu giới hạn nguồn "
            "khi học sinh yêu cầu dẫn nguồn, câu hỏi phụ thuộc một tài liệu cụ thể, hoặc độ chắc "
            "chắn thấp. Khi dùng học liệu, trích dẫn ngắn gọn và đúng chỗ."
        )
        learning_context = (
            f"\n\nHọc liệu tham khảo có liên quan:\n{context_text}"
            if context_text
            else ""
        )
        visualization = state.get("visualization") or {}
        if visualization.get("success"):
            visualization_context = (
                "\n\nMột animation đã được tạo và sẽ hiển thị cạnh câu trả lời. "
                "Hãy tổ chức câu trả lời theo thứ tự: Khái niệm, Giải thích ngắn, "
                "Công thức, Điều cần quan sát trong animation, Ý chính cần nhớ. "
                "Không nói rằng bạn không thể tạo hình ảnh.\n"
                + json.dumps(
                    {
                        "concept": visualization.get("concept"),
                        "formulas": visualization.get("formulas", []),
                        "key_takeaway": visualization.get("key_takeaway"),
                    },
                    ensure_ascii=False,
                )
            )
        elif state.get("visualization_decision", {}).get("should_visualize"):
            visualization_context = (
                "\n\nAnimation không khả dụng trong lượt này. Tiếp tục giải thích đầy "
                "đủ bằng văn bản và công thức, không nêu lỗi kỹ thuật nội bộ."
            )
        else:
            visualization_context = ""
        human = (
            f"Đề bài:\n{problem.problem_markdown}\n\n"
            f"Bài làm đã đọc được:\n{problem.student_work or '(chưa có)'}\n\n"
            f"Lịch sử hội thoại gần nhất:\n{history_text or '(chưa có)'}\n\n"
            f"Tin nhắn mới:\n{message}"
            f"{learning_context}"
            f"{visualization_context}"
        )
        streamed = False
        assessment_result: TutorAssessment | None = None
        try:
            writer = get_stream_writer()
            parts: list[str] = []
            async for chunk in tutor_model.astream(
                [SystemMessage(content=system), HumanMessage(content=human)]
            ):
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    text = "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                else:
                    text = content if isinstance(content, str) else ""
                if text:
                    parts.append(text)
                    writer({"type": "token", "content": text})
            answer = "".join(parts).strip()
            if not answer:
                raise RuntimeError("Tutor model returned an empty stream")
            streamed = True
            response = TutorResponse(
                message=answer,
                hint_level=target_hint,
                feedback_type="guidance",
                used_full_solution=allow_full,
                citations=citations,
            )
        except Exception:
            try:
                structured = tutor_model.with_structured_output(
                    TutorResponse, method="function_calling"
                )
                response = await structured.ainvoke(
                    [SystemMessage(content=system), HumanMessage(content=human)]
                )
                response.hint_level = target_hint
                response.used_full_solution = allow_full
                response.citations = citations
            except Exception:
                response = TutorResponse(
                    message=(
                        "Mình đã nhận đề. Trước tiên, bạn hãy cho biết mình đã thử bước nào "
                        "và đang vướng ở dữ kiện hay công thức nào?"
                    ),
                    hint_level=target_hint,
                    feedback_type="guidance",
                    used_full_solution=False,
                    citations=citations,
                )

        # Keep evaluation structured while the student-facing answer is streamed.
        if streamed:
            try:
                allowed_skills = skills_for_subject(problem.subject)
                assessor = tutor_model.with_structured_output(
                    TutorAssessment, method="function_calling"
                )
                assessed = await assessor.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "Phân tích riêng câu trả lời mới của học sinh so với đề bài. "
                                "Chỉ đặt assessable=true khi học sinh thực sự trình bày đáp án, "
                                "phép tính hoặc lập luận có thể kiểm tra. Câu hỏi xin gợi ý không "
                                "phải là câu trả lời sai. Chỉ chọn skill_ids trong danh sách cho phép. "
                                "confidence phản ánh độ chắc chắn của việc chấm từ 0 đến 1."
                            )
                        ),
                        HumanMessage(
                            content=(
                                "Kỹ năng cho phép: "
                                + ", ".join(skill.id for skill in allowed_skills)
                                + "\n"
                                f"Đề bài: {problem.problem_markdown}\n"
                                f"Câu trả lời học sinh: {message}\n"
                                f"Phản hồi trợ giảng: {response.message}"
                            )
                        ),
                    ]
                )
                valid_skill_ids = {
                    skill.id for skill in allowed_skills
                }
                assessed.skill_ids = [
                    skill_id
                    for skill_id in assessed.skill_ids
                    if skill_id in valid_skill_ids and skill_id in SKILL_BY_ID
                ]
                if not assessed.skill_ids and assessed.assessable:
                    assessed.skill_ids = [
                        skill_id
                        for skill_id in problem.skill_ids
                        if skill_id in valid_skill_ids and skill_id in SKILL_BY_ID
                    ]
                assessment_result = assessed
                response.answer_correct = assessed.answer_correct
                response.misconception = assessed.misconception
                response.feedback_type = assessed.feedback_type
            except Exception:
                pass
        elif response.answer_correct is not None:
            assessment_result = TutorAssessment(
                assessable=True,
                answer_correct=response.answer_correct,
                misconception=response.misconception,
                feedback_type=response.feedback_type,
                skill_ids=[
                    skill_id for skill_id in problem.skill_ids if skill_id in SKILL_BY_ID
                ],
                confidence=0.65,
            )
        if state.get("problem_id"):
            await repository.save_hint_event(
                state["user_id"], state["session_id"], state["problem_id"],
                response.model_dump(mode="json"),
            )
        return {
            "hint_level": response.hint_level,
            "tutor_response": response.model_dump(mode="json"),
            "conversation": [{"role": "assistant", "content": response.message}],
            "chat_assessment": (
                assessment_result.model_dump(mode="json")
                if assessment_result is not None
                else None
            ),
        }

    async def assessment(state: TutorState) -> dict:
        response = TutorResponse.model_validate(state["tutor_response"])
        raw_assessment = state.get("chat_assessment")
        if not raw_assessment:
            return {"mastery_updates": [], "assessed_skill_ids": []}
        chat_assessment = TutorAssessment.model_validate(raw_assessment)
        if (
            not chat_assessment.assessable
            or chat_assessment.answer_correct is None
            or chat_assessment.confidence < 0.65
            or not chat_assessment.skill_ids
        ):
            return {"mastery_updates": [], "assessed_skill_ids": []}
        response.answer_correct = chat_assessment.answer_correct
        response.misconception = chat_assessment.misconception
        evidence = response.evidence
        if evidence is None:
            evidence = evidence_for_outcome(
                answer_correct=response.answer_correct,
                hint_level=response.hint_level,
                used_full_solution=response.used_full_solution,
            )
        if evidence is None:
            return {"mastery_updates": [], "assessed_skill_ids": []}
        response.evidence = evidence
        if response.answer_correct is not None and state.get("problem_id"):
            await repository.save_attempt(
                state["user_id"], state["session_id"], state["problem_id"],
                state.get("message", ""), response.answer_correct, evidence,
                response.misconception,
            )
        return {
            "tutor_response": response.model_dump(mode="json"),
            "assessed_skill_ids": chat_assessment.skill_ids,
        }

    async def mastery(state: TutorState) -> dict:
        response = TutorResponse.model_validate(state["tutor_response"])
        problem = ProblemExtraction.model_validate(state["problem"])
        assessed_skill_ids = state.get("assessed_skill_ids", [])
        if response.evidence is None or not assessed_skill_ids:
            return {"mastery_updates": [], "recommendations": []}
        current = {item.skill_id: item for item in await repository.get_mastery(state["user_id"])}
        updates: list[MasteryUpdate] = []
        for skill_id in assessed_skill_ids:
            record, change = update_mastery(
                user_id=state["user_id"],
                skill_id=skill_id,
                subject=problem.subject,
                old_mastery=current.get(skill_id).mastery if skill_id in current else None,
                evidence=response.evidence,
            )
            previous = current.get(skill_id)
            if previous:
                record.attempts = previous.attempts + 1
                record.recent_errors = min(
                    10, previous.recent_errors + (1 if response.evidence == 0 else 0)
                )
            await repository.upsert_mastery(record)
            updates.append(change)
        all_mastery = await repository.get_mastery(state["user_id"])
        trends = await repository.get_exam_trends([problem.subject])
        recommendations = rank_recommendations(all_mastery, trends)
        return {
            "mastery_updates": [item.model_dump(mode="json") for item in updates],
            "recommendations": [item.model_dump(mode="json") for item in recommendations],
        }

    graph = StateGraph(TutorState)
    graph.add_node("perception", perception)
    graph.add_node("clarification", clarification)
    graph.add_node("visualization_router", visualization_decision)
    graph.add_node("visualization_tool", visualization)
    graph.add_node("retrieval", retrieval)
    graph.add_node("tutor", tutor)
    graph.add_node("assessment", assessment)
    graph.add_node("mastery", mastery)
    graph.set_entry_point("perception")
    graph.add_conditional_edges(
        "perception",
        route_after_perception,
        {
            "clarification": "clarification",
            "visualization_router": "visualization_router",
        },
    )
    graph.add_edge("clarification", END)
    graph.add_conditional_edges(
        "visualization_router",
        route_after_visualization_decision,
        {"visualization_tool": "visualization_tool", "retrieval": "retrieval"},
    )
    graph.add_edge("visualization_tool", "retrieval")
    graph.add_edge("retrieval", "tutor")
    graph.add_edge("tutor", "assessment")
    graph.add_edge("assessment", "mastery")
    graph.add_edge("mastery", END)
    return graph.compile(checkpointer=checkpointer)


def state_json(value: Any) -> Any:
    """Convert graph/Pydantic values to SSE-safe JSON values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
