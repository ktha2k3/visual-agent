import pytest
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver

from education.models import ProblemExtraction, Subject, TutorResponse
from education.tutor_graph import (
    MAX_CONVERSATION_MESSAGES,
    TutorAssessment,
    build_tutor_graph,
    merge_conversation,
)
from services.repository import MemoryRepository


class _Storage:
    async def read(self, record):  # pragma: no cover - not used in this branch
        return b""


def test_conversation_buffer_is_bounded_and_filters_invalid_messages():
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
        for index in range(MAX_CONVERSATION_MESSAGES + 5)
    ]
    result = merge_conversation([], messages + [{"role": "system", "content": "ignore"}])
    assert len(result) == MAX_CONVERSATION_MESSAGES
    assert result[0]["content"] == "5"
    assert all(item["role"] in {"user", "assistant"} for item in result)


@pytest.mark.asyncio
async def test_low_confidence_extraction_stops_for_student_confirmation(monkeypatch):
    async def fake_extract(self, *, message, images):
        return ProblemExtraction(
            subject=Subject.MATH,
            problem_markdown="Tính đạo hàm của ...",
            confidence=0.4,
        )

    monkeypatch.setattr("education.tutor_graph.ProblemPerception.extract", fake_extract)
    graph = build_tutor_graph(MemoryRepository(), _Storage(), checkpointer=MemorySaver())
    result = await graph.ainvoke(
        {"user_id": "student", "session_id": "session", "message": "đọc đề", "attachment_ids": []},
        {"configurable": {"thread_id": "student:session"}},
    )
    assert result["needs_clarification"] is True
    assert result["tutor_response"]["feedback_type"] == "clarification"
    assert "chưa đọc đề đủ chắc chắn" in result["tutor_response"]["message"]


@pytest.mark.asyncio
async def test_first_tutor_turn_asks_before_revealing_hint(monkeypatch):
    async def fake_extract(self, *, message, images):
        return ProblemExtraction(
            subject=Subject.MATH,
            problem_markdown="Tính đạo hàm của x^2",
            skill_ids=["math.derivatives"],
            confidence=0.95,
        )

    class FakeStructured:
        async def ainvoke(self, messages):
            return TutorResponse(message="Bạn đã thử quy tắc đạo hàm nào?")

    class FakeModel:
        def with_structured_output(self, schema, **kwargs):
            return FakeStructured()

    class FakeEmbedding:
        async def aembed_query(self, query):
            return [0.0] * 1024

    monkeypatch.setattr("education.tutor_graph.ProblemPerception.extract", fake_extract)
    monkeypatch.setattr("education.tutor_graph.get_chat_model", lambda role: FakeModel())
    monkeypatch.setattr("education.tutor_graph.get_embedding_model", lambda: FakeEmbedding())
    graph = build_tutor_graph(MemoryRepository(), _Storage(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "student:socratic"}}
    first = await graph.ainvoke(
        {"user_id": "student", "session_id": "socratic", "message": "Giúp em", "attachment_ids": []},
        config,
    )
    assert first["tutor_response"]["hint_level"] == 0
    second = await graph.ainvoke(
        {"user_id": "student", "session_id": "socratic", "message": "Em đã thử quy tắc lũy thừa", "attachment_ids": []},
        config,
    )
    assert second["tutor_response"]["hint_level"] == 1


@pytest.mark.asyncio
async def test_session_memory_reaches_prompt_and_emits_custom_tokens(monkeypatch):
    async def fake_extract(self, *, message, images):
        return ProblemExtraction(
            subject=Subject.MATH,
            problem_markdown="Tính đạo hàm của x^2",
            skill_ids=["math.derivatives"],
            confidence=0.95,
        )

    prompts = []

    class FakeAssessment:
        async def ainvoke(self, messages):
            return TutorAssessment()

    class StreamingModel:
        async def astream(self, messages):
            prompts.append(messages)
            yield AIMessageChunk(content="Gợi ý ")
            yield AIMessageChunk(content="từng bước")

        def with_structured_output(self, schema, **kwargs):
            return FakeAssessment()

    class FakeEmbedding:
        async def aembed_query(self, query):
            return [0.0] * 1024

    monkeypatch.setattr("education.tutor_graph.ProblemPerception.extract", fake_extract)
    monkeypatch.setattr("education.tutor_graph.get_chat_model", lambda role: StreamingModel())
    monkeypatch.setattr("education.tutor_graph.get_embedding_model", lambda: FakeEmbedding())
    graph = build_tutor_graph(MemoryRepository(), _Storage(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "student:memory"}}

    async def run_turn(message):
        chunks = []
        async for chunk in graph.astream(
            {
                "user_id": "student",
                "session_id": "memory",
                "message": message,
                "attachment_ids": [],
                "conversation": [{"role": "user", "content": message}],
            },
            config,
            stream_mode=["custom", "updates"],
        ):
            chunks.append(chunk)
        return chunks

    first_chunks = await run_turn("Em chưa biết bắt đầu")
    await run_turn("Em đã thử quy tắc lũy thừa")

    assert ("custom", {"type": "token", "content": "Gợi ý "}) in first_chunks
    system_prompt = prompts[-1][0].content
    second_prompt = prompts[-1][-1].content
    assert "kiến thức chuẩn của chương trình THPT" in system_prompt
    assert "Chỉ dùng học liệu" not in system_prompt
    assert "chưa có học liệu đã duyệt" not in second_prompt
    assert "Em chưa biết bắt đầu" in second_prompt
    assert "Gợi ý từng bước" in second_prompt
    state = await graph.aget_state(config)
    assert [item["role"] for item in state.values["conversation"]] == [
        "user", "assistant", "user", "assistant"
    ]


@pytest.mark.asyncio
async def test_only_assessable_chat_answers_update_mastery(monkeypatch):
    async def fake_extract(self, *, message, images):
        return ProblemExtraction(
            subject=Subject.MATH,
            problem_markdown="Tính đạo hàm của x^2",
            skill_ids=["math.derivatives"],
            confidence=0.95,
        )

    class FakeAssessment:
        async def ainvoke(self, messages):
            latest = messages[-1].content
            if "xin gợi ý" in latest:
                return TutorAssessment(assessable=False, confidence=0.95)
            return TutorAssessment(
                assessable=True,
                answer_correct=True,
                skill_ids=["math.derivatives", "physics.nuclear"],
                confidence=0.9,
            )

    class StreamingModel:
        async def astream(self, messages):
            yield AIMessageChunk(content="Mình đã xem cách làm của bạn.")

        def with_structured_output(self, schema, **kwargs):
            return FakeAssessment()

    class FakeEmbedding:
        async def aembed_query(self, query):
            return [0.0] * 1024

    monkeypatch.setattr("education.tutor_graph.ProblemPerception.extract", fake_extract)
    monkeypatch.setattr("education.tutor_graph.get_chat_model", lambda role: StreamingModel())
    monkeypatch.setattr("education.tutor_graph.get_embedding_model", lambda: FakeEmbedding())
    repository = MemoryRepository()
    graph = build_tutor_graph(repository, _Storage(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "student:observed"}}

    async def run(message):
        return await graph.ainvoke(
            {
                "user_id": "student",
                "session_id": "observed",
                "message": message,
                "attachment_ids": [],
                "conversation": [{"role": "user", "content": message}],
            },
            config,
        )

    first = await run("Em tính được đạo hàm là 2x")
    second = await run("Cho em xin gợi ý")
    records = await repository.get_mastery("student")

    assert first["assessed_skill_ids"] == ["math.derivatives"]
    assert second["assessed_skill_ids"] == []
    assert len(records) == 1
    assert records[0].attempts == 1
    assert records[0].mastery == 0.65
