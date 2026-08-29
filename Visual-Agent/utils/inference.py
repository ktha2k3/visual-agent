import uuid

from langgraph.checkpoint.memory import MemorySaver

from education.tutor_graph import build_tutor_graph
from services.repository import MemoryRepository
from services.storage import AssetStorage

_repository = MemoryRepository()
_tutor = build_tutor_graph(
    _repository, AssetStorage(_repository), checkpointer=MemorySaver()
)


def run_supervisor(user_input: str, recursion_limit: int = 50):
    """Backward-compatible CLI function backed by the education Tutor Graph."""
    session_id = str(uuid.uuid4())
    result = _tutor.invoke(
        {
            "user_id": "cli-user",
            "session_id": session_id,
            "message": user_input,
            "attachment_ids": [],
        },
        {
            "recursion_limit": recursion_limit,
            "configurable": {"thread_id": f"cli-user:{session_id}"},
        },
    )
    response = result.get("tutor_response", {})
    return [response.get("message", "")]
