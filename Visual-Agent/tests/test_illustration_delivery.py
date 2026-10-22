import asyncio
import uuid

import pytest

import api
from education.models import AuthUser
from services.repository import MemoryRepository
from tools.visualization.models import StemVisualizationResult


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
async def test_late_illustration_is_delivered_and_upserted_into_history(monkeypatch, success):
    repository = MemoryRepository()
    monkeypatch.setattr(api, "_REPOSITORY", repository)
    class Service:
        async def visualize(self, **kwargs):
            await asyncio.sleep(0.02)
            return StemVisualizationResult(success=success, subject="math", concept="Ví dụ", explanation="", key_takeaway="",
                scene_data={"template": "function_graph"} if success else None,
                error=None if success else "Chưa hỗ trợ", status="completed" if success else "unsupported")
    monkeypatch.setattr("tools.visualization.streaming.SceneIllustrationService", Service)
    class Graph:
        async def astream(self, graph_input, **kwargs):
            assert graph_input["visualization_external"]
            yield "updates", {"visualization_router": {"visualization_decision": {"should_visualize": True, "subject": "math", "question": "y=x^2"}}}
            yield "custom", {"type": "token", "content": "Lời giải"}
            yield "updates", {"tutor": {"tutor_response": {"message": "Lời giải", "hint_level": 0, "feedback_type": "guidance"}}}
    session = str(uuid.uuid4())
    request = api.TutorRequest(session_id=session, message="vẽ đồ thị", subject="math")
    events = [event async for event in api.tutor_events(request, AuthUser(id="student"), Graph())]
    types = [event["type"] for event in events]
    assert types.index("token") < types.index("visualization") < types.index("done")
    result = next(event for event in events if event["type"] == "visualization")
    assert result["success"] == success
    history = await repository.get_chat_history("student", session)
    answers = [message for message in history if message.role == "assistant"]
    assert len(answers) == 1 and answers[0].content == "Lời giải"
    assert answers[0].metadata["visualization"]["success"] == success
