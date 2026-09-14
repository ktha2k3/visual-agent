import asyncio
import hashlib
import hmac
import json

from fastapi.testclient import TestClient
import pytest

import api
import services.content as content_service
from education.models import AuthUser, ChatMessage, Subject
from services.auth import get_current_user
from services.repository import MemoryRepository
from tools.visualization.artifacts import VisualizationArtifactStore

app = api.app


@pytest.fixture(autouse=True)
def isolated_repository(monkeypatch):
    repository = MemoryRepository()
    monkeypatch.setattr(api, "_REPOSITORY", repository)
    return repository


def test_health_and_development_identity():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        me = client.get("/api/me")
        assert me.status_code == 200
        assert me.json()["auth_mode"] == "development"


def test_visualization_artifact_endpoint_enforces_ownership(tmp_path, monkeypatch):
    store = VisualizationArtifactStore(tmp_path)
    artifact_id = "a" * 64
    owner_id = "11111111-1111-4111-8111-111111111111"
    run_dir = store.initialize(
        artifact_id,
        user_id=owner_id,
        question="Đạo hàm là gì?",
        subject="math",
        model="gemma4:31b-cloud",
        version="test",
    )
    video = run_dir / "media" / "animation.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video" * 300)
    store.update_manifest(
        artifact_id,
        status="completed",
        render={"video_relative_path": "media/animation.mp4"},
    )
    monkeypatch.setattr(api, "_VISUALIZATION_STORE", store)
    app.dependency_overrides[get_current_user] = lambda: AuthUser(id=owner_id)
    try:
        with TestClient(app) as client:
            allowed = client.get(f"/api/visualizations/{artifact_id}/video")
        assert allowed.status_code == 200
        assert allowed.headers["content-type"].startswith("video/mp4")
        app.dependency_overrides[get_current_user] = lambda: AuthUser(id="intruder")
        with TestClient(app) as client:
            denied = client.get(f"/api/visualizations/{artifact_id}/video")
        assert denied.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_exam_bank_routes_are_removed():
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]
    assert not any("exam-plan" in path for path in paths)
    assert "/api/admin/exam-sync" not in paths
    assert not any("review-queue" in path for path in paths)


def test_every_user_can_upload_private_knowledge(monkeypatch, isolated_repository):
    class NoCloudEmbeddings:
        async def aembed_documents(self, _texts):
            raise RuntimeError("cloud has no embedding endpoint")

    monkeypatch.setattr(content_service, "get_embedding_model", lambda: NoCloudEmbeddings())
    learner_id = "11111111-1111-4111-8111-111111111111"
    app.dependency_overrides[get_current_user] = lambda: AuthUser(
        id=learner_id, email="learner@example.com", is_admin=False
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/knowledge",
                data={"subject": "math", "title": "De cuong dao ham"},
                files={
                    "file": (
                        "de-cuong.md",
                        "# Dao ham\n\nDao ham va ung dung khao sat ham so lop 12.",
                        "text/markdown",
                    )
                },
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 202
    assert response.json()["visibility"] == "private"
    assert response.json()["status"] == "pending"
    job_id = response.json()["job_id"]
    job = isolated_repository.knowledge_jobs[job_id]
    assert job.status == "completed"
    assert job.embedding_status == "lexical_fallback"
    assert job.chunks == 1
    stored = next(iter(isolated_repository.content_documents.values()))
    assert stored["user_id"] == learner_id
    app.dependency_overrides[get_current_user] = lambda: AuthUser(
        id=learner_id, email="learner@example.com", is_admin=False
    )
    try:
        with TestClient(app) as client:
            documents = client.get("/api/knowledge/documents?subject=math").json()
    finally:
        app.dependency_overrides.clear()
    assert documents["items"][0]["chunks"] == 1


def test_knowledge_webhook_requires_hmac_and_known_job(
    monkeypatch, isolated_repository
):
    secret = "test-webhook-secret"
    monkeypatch.setattr(
        api,
        "_SETTINGS",
        api._SETTINGS.__class__(
            **{
                **api._SETTINGS.__dict__,
                "knowledge_webhook_secret": secret,
            }
        ),
    )
    payload = json.dumps({"job_id": "missing"}).encode()
    with TestClient(app) as client:
        rejected = client.post(
            "/api/webhooks/knowledge-uploaded",
            content=payload,
            headers={"x-knowledge-signature": "sha256=invalid"},
        )
        signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        missing = client.post(
            "/api/webhooks/knowledge-uploaded",
            content=payload,
            headers={"x-knowledge-signature": f"sha256={signature}"},
        )
    assert rejected.status_code == 401
    assert missing.status_code == 404


def test_tutor_rejects_empty_request_before_model_call():
    with TestClient(app) as client:
        response = client.post(
            "/api/tutor/stream",
            json={"session_id": "2216f7aa-a088-4f61-bbe3-f3192d68c660", "message": "", "attachment_ids": []},
        )
        assert response.status_code == 422


def test_chat_session_is_created_by_backend_and_listed(isolated_repository):
    with TestClient(app) as client:
        created = client.post(
            "/api/chat/sessions",
            json={"subject": "math", "learning_mode": "concept"},
        )
        listed = client.get("/api/chat/sessions?subject=math")
        changed = client.patch(
            f"/api/chat/sessions/{created.json()['id']}",
            json={"learning_mode": "guided_problem"},
        )
        history = client.get(f"/api/history/{created.json()['id']}")

    assert created.status_code == 201
    assert listed.status_code == 200
    assert listed.json()["items"][0]["session_id"] == created.json()["id"]
    assert listed.json()["items"][0]["message_count"] == 0
    assert listed.json()["items"][0]["learning_mode"] == "concept"
    assert changed.status_code == 200
    assert changed.json()["learning_mode"] == "guided_problem"
    assert history.json()["learning_mode"] == "guided_problem"


def test_public_flows_only_accept_math_and_physics():
    with TestClient(app) as client:
        math = client.post("/api/chat/sessions", json={"subject": "math"})
        physics = client.post("/api/chat/sessions", json={"subject": "physics"})
        chemistry = client.post("/api/chat/sessions", json={"subject": "chemistry"})
        biology = client.post("/api/chat/sessions", json={"subject": "biology"})

    assert math.status_code == 201
    assert physics.status_code == 201
    assert chemistry.status_code == 422
    assert biology.status_code == 422
    assert "Toán và Vật lí" in chemistry.json()["detail"]


def test_chat_session_lists_are_separated_by_subject():
    with TestClient(app) as client:
        math_id = client.post("/api/chat/sessions", json={"subject": "math"}).json()["id"]
        physics_id = client.post("/api/chat/sessions", json={"subject": "physics"}).json()["id"]
        math_items = client.get("/api/chat/sessions?subject=math").json()["items"]
        physics_items = client.get("/api/chat/sessions?subject=physics").json()["items"]

    assert [item["session_id"] for item in math_items] == [math_id]
    assert [item["session_id"] for item in physics_items] == [physics_id]


def test_tutor_websocket_streams_token_and_done(monkeypatch):
    graph_inputs = []

    class FakeTutor:
        async def astream(self, graph_input, config, stream_mode):
            graph_inputs.append(graph_input)
            yield ("custom", {"type": "token", "content": "Gợi "})
            yield ("custom", {"type": "token", "content": "ý"})
            yield (
                "updates",
                {
                    "tutor": {
                        "tutor_response": {
                            "message": "Gợi ý",
                            "hint_level": 0,
                            "feedback_type": "guidance",
                            "citations": [],
                        }
                    }
                },
            )
            yield (
                "updates",
                {
                    "assessment": {
                        "tutor_response": {
                            "message": "Gợi ý",
                            "hint_level": 0,
                            "feedback_type": "guidance",
                            "evidence": 0.75,
                            "citations": [],
                        }
                    }
                },
            )

    monkeypatch.setattr(api, "build_tutor_graph", lambda *args, **kwargs: FakeTutor())
    session_id = "2216f7aa-a088-4f61-bbe3-f3192d68c660"
    with TestClient(app) as client:
        with client.websocket_connect("/api/tutor/ws") as socket:
            socket.send_json(
                {
                    "request": {
                        "session_id": session_id,
                        "request_id": "11111111-1111-4111-8111-111111111111",
                        "message": "Giúp em giải bài này",
                        "attachment_ids": [],
                        "subject": "math",
                        "learning_mode": "concept",
                    }
                }
            )
            events = []
            while not events or events[-1]["type"] != "done":
                events.append(socket.receive_json())
        with client.websocket_connect("/api/tutor/ws") as socket:
            socket.send_json(
                {
                    "request": {
                        "session_id": session_id,
                        "request_id": "22222222-2222-4222-8222-222222222222",
                        "message": "Em làm bước tiếp theo thế nào?",
                        "attachment_ids": [],
                    }
                }
            )
            while socket.receive_json()["type"] != "done":
                pass
        history = client.get(f"/api/history/{session_id}").json()
        sessions = client.get("/api/chat/sessions").json()

    assert [event["type"] for event in events] == [
        "token", "token", "stage", "hint", "stage", "done"
    ]
    assert "".join(
        event["content"] for event in events if event["type"] == "token"
    ) == "Gợi ý"
    assert sum(event["type"] == "hint" for event in events) == 1
    assert history["persistent"] is True
    assert sessions["items"][0]["session_id"] == session_id
    assert sessions["items"][0]["subject"] == "math"
    assert sessions["items"][0]["learning_mode"] == "concept"
    assert history["learning_mode"] == "concept"
    assert [item["role"] for item in history["messages"]] == [
        "user", "assistant", "user", "assistant"
    ]
    assert graph_inputs[1]["conversation"] == [
        {"role": "user", "content": "Giúp em giải bài này"},
        {"role": "assistant", "content": "Gợi ý"},
        {"role": "user", "content": "Em làm bước tiếp theo thế nào?"},
    ]
    assert graph_inputs[0]["learning_mode"] == "concept"
    assert graph_inputs[1]["learning_mode"] == "concept"


def test_tutor_sse_preserves_realtime_token_chunks(monkeypatch):
    class FakeTutor:
        async def astream(self, graph_input, config, stream_mode):
            yield ("custom", {"type": "token", "content": "First "})
            yield ("custom", {"type": "token", "content": "hint"})
            yield (
                "updates",
                {
                    "supervisor": {
                        "learning_plan": {"intent": "understand_concept"}
                    }
                },
            )
            yield (
                "updates",
                {
                    "tutor": {
                        "tutor_response": {
                            "message": "First hint",
                            "hint_level": 0,
                            "feedback_type": "guidance",
                            "citations": [],
                        }
                    }
                },
            )

    monkeypatch.setattr(api, "build_tutor_graph", lambda *args, **kwargs: FakeTutor())
    with TestClient(app) as client:
        response = client.post(
            "/api/tutor/stream",
            json={
                "session_id": "4216f7aa-a088-4f61-bbe3-f3192d68c660",
                "request_id": "33333333-3333-4333-8333-333333333333",
                "message": "Show one hint",
                "attachment_ids": [],
                "subject": "math",
                "learning_mode": "concept",
            },
        )

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == [
        "token", "token", "stage", "learning_plan", "mode_selected",
        "stage", "hint", "done"
    ]
    assert [
        event["content"] for event in events if event["type"] == "token"
    ] == ["First ", "hint"]
    assert next(
        event for event in events if event["type"] == "mode_selected"
    )["learning_mode"] == "concept"


def test_diagnostic_routes_are_removed():
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert not any(path.startswith("/api/diagnostic") for path in paths)


def test_tutor_session_rejects_cross_subject_requests(isolated_repository):
    session_id = "3216f7aa-a088-4f61-bbe3-f3192d68c660"
    asyncio.run(
        isolated_repository.save_chat_message(
            ChatMessage(
                user_id="00000000-0000-4000-8000-000000000001",
                session_id=session_id,
                request_id="seed-request",
                role="user",
                content="Đạo hàm là gì?",
                metadata={"subject": "math"},
            )
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/tutor/stream",
            json={
                "session_id": session_id,
                "message": "Môn Vật lí. Giải thích định luật Newton",
                "subject": "physics",
            },
        )

    assert response.status_code == 200
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [item["type"] for item in events] == ["error", "done"]
    assert "thuộc môn math" in events[0]["content"]
    history = asyncio.run(
        isolated_repository.get_chat_history(
            "00000000-0000-4000-8000-000000000001", session_id
        )
    )
    assert len(history) == 1
