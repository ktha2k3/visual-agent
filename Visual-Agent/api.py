import asyncio
import hashlib
import hmac
import importlib.util
import json
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from education.models import (
    AuthUser,
    ChatMessage,
    KnowledgeIngestionJob,
    ReviewPatch,
    Subject,
)
from education.curriculum import SKILL_BY_ID
from education.learning_profile import summarize_learning_profile
from education.exam_research import LearningPathSupervisor
from education.recommendations import rank_recommendations
from education.tutor_graph import build_tutor_graph, state_json
from exam.jobs import enqueue_initial_sync
from services.auth import authenticate_access_token, get_current_user, require_admin
from services.content import ContentIngestionService
from services.knowledge import KnowledgeFileError, read_knowledge_payload
from services.knowledge_ingestion import (
    KnowledgeIngestionProcessor,
    KnowledgeUploadStorage,
    enqueue_knowledge_job,
)
from services.repository import get_repository
from services.storage import AssetStorage
from tools.visualization.artifacts import VisualizationArtifactStore
from utils.config import get_settings
from utils.ollama import is_ollama_cloud, normalize_ollama_model

_SETTINGS = get_settings()
_REDIS_URL = _SETTINGS.redis_url
_DEBUG = _SETTINGS.debug_stream
_REPOSITORY = get_repository()
_STORAGE = AssetStorage(_REPOSITORY, _SETTINGS)
_KNOWLEDGE_STORAGE = KnowledgeUploadStorage(_SETTINGS)
_VISUALIZATION_STORE = VisualizationArtifactStore(
    _SETTINGS.visualization_artifact_dir
)


class _RateLimiter:
    """Small single-instance guard; production gateways should enforce a second limit."""

    def __init__(self, requests: int = 30, window_seconds: int = 60):
        self.requests = requests
        self.window_seconds = window_seconds
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        bucket = self.hits[key]
        while bucket and bucket[0] <= now - self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.requests:
            raise HTTPException(status_code=429, detail="Too many requests")
        bucket.append(now)


_RATE_LIMITER = _RateLimiter()


class KnowledgeWebhookEvent(BaseModel):
    job_id: str


async def _process_knowledge_locally(job_id: str) -> None:
    await KnowledgeIngestionProcessor(
        _REPOSITORY, _KNOWLEDGE_STORAGE
    ).process(job_id)


async def _dispatch_knowledge_job(job_id: str) -> None:
    if not enqueue_knowledge_job(job_id):
        await _process_knowledge_locally(job_id)


def _dbg(*args):
    if _DEBUG:
        print("[DBG]", *args, flush=True)


# ── Startup / shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = None

    if _REDIS_URL:
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver

            # from_conn_string returns an async context manager — use it correctly
            async with AsyncRedisSaver.from_conn_string(_REDIS_URL) as cp:
                await cp.setup()
                print(f"[memory] Redis connected → {_REDIS_URL}", flush=True)
                app.state.tutor = build_tutor_graph(
                    _REPOSITORY, _STORAGE, checkpointer=cp
                )
                app.state.checkpointer = cp
                yield  # server runs here, inside the async context
            return  # skip MemorySaver fallback below
        except Exception as exc:
            print(
                f"[memory] Redis unavailable ({exc}), falling back to MemorySaver",
                flush=True,
            )

    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    print(
        "[memory] Using in-memory checkpointer (sessions lost on restart)", flush=True
    )
    app.state.tutor = build_tutor_graph(
        _REPOSITORY, _STORAGE, checkpointer=checkpointer
    )
    app.state.checkpointer = checkpointer
    yield


app = FastAPI(title="Visual Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_SETTINGS.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def ip_rate_limit(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "unknown"
        try:
            _RATE_LIMITER.check(f"ip:{client_ip}")
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


# ── Request model ──────────────────────────────────────────────────────────────
class InferenceRequest(BaseModel):
    message: str
    session_id: str = "default"


class TutorRequest(BaseModel):
    session_id: str
    request_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    message: str = Field(default="", max_length=50_000)
    display_message: str | None = Field(default=None, max_length=50_000)
    attachment_ids: list[str] = Field(default_factory=list)
    knowledge_document_ids: list[str] = Field(default_factory=list, max_length=50)
    subject: Subject | None = None


class ExamSyncRequest(BaseModel):
    extract: bool = True


class ContentIngestRequest(BaseModel):
    title: str = Field(min_length=3, max_length=300)
    subject: Subject
    source_url: HttpUrl
    publisher: str | None = Field(default=None, max_length=200)
    rights_note: str = Field(min_length=3, max_length=500)
    content: str = Field(min_length=20, max_length=200_000)
    skill_ids: list[str] = Field(default_factory=list, max_length=20)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _effective_agent(metadata: dict) -> str:
    checkpoint_ns: str = metadata.get("langgraph_checkpoint_ns", "")
    if checkpoint_ns:
        return checkpoint_ns.split("|")[0].split(":")[0]
    return metadata.get("langgraph_node", "supervisor")


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


# ── SSE stream ─────────────────────────────────────────────────────────────────
async def event_stream(message: str, session_id: str, supervisor):
    config = {
        "recursion_limit": 50,
        "configurable": {"thread_id": session_id},
    }
    seen_agents: set[str] = set()
    supervisor_text_sent = False

    def _start(agent: str) -> str | None:
        if agent not in seen_agents:
            seen_agents.add(agent)
            return f"data: {json.dumps({'type': 'agent_start', 'agent': agent})}\n\n"
        return None

    try:
        async for raw in supervisor.astream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            # Detect format: multi-mode → (mode_str, payload) or legacy raw types
            if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], str):
                mode, payload = raw
            elif isinstance(raw, tuple) and len(raw) == 2:
                mode, payload = "messages", raw
            elif isinstance(raw, dict):
                mode, payload = "updates", raw
            else:
                _dbg(f"unknown raw type: {type(raw)}")
                continue

            _dbg(f"mode={mode!r}")

            # ── Token streaming ────────────────────────────────────────────
            if mode == "messages":
                chunk, metadata = payload
                agent = _effective_agent(metadata)
                node = metadata.get("langgraph_node", "?")
                ns = metadata.get("langgraph_checkpoint_ns", "")
                _dbg(
                    f"  messages | node={node!r} ns={ns!r} agent={agent!r} type={type(chunk).__name__}"
                )

                start = _start(agent)
                if start:
                    yield start

                if isinstance(chunk, ToolMessage):
                    tool_name = chunk.name or ""
                    _dbg(
                        f"  ToolMessage tool={tool_name!r} content={str(chunk.content)[:80]!r}"
                    )
                    if "transfer" in tool_name.lower():
                        if "back" not in tool_name.lower():
                            # Routing to a sub-agent — reset flag so updates fallback can run later
                            supervisor_text_sent = False
                            _dbg("  → supervisor_text_sent reset to False")
                        yield f"data: {json.dumps({'type': 'handoff', 'agent': agent, 'tool': tool_name})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'tool_result', 'agent': agent, 'tool': tool_name, 'content': str(chunk.content)[:1000]})}\n\n"

                elif isinstance(chunk, AIMessageChunk):
                    text = _extract_text(chunk.content)
                    tcs = [
                        tc.get("name")
                        for tc in (chunk.tool_call_chunks or [])
                        if tc.get("name")
                    ]
                    _dbg(f"  AIMessageChunk text={text[:60]!r} tool_calls={tcs}")

                    if text:
                        if agent == "supervisor":
                            supervisor_text_sent = True
                        yield f"data: {json.dumps({'type': 'token', 'agent': agent, 'content': text})}\n\n"

                    for tc in chunk.tool_call_chunks or []:
                        if tc.get("name") and "transfer" not in tc["name"].lower():
                            yield f"data: {json.dumps({'type': 'tool_call', 'agent': agent, 'tool': tc['name'], 'args': str(tc.get('args', ''))})}\n\n"

            # ── State updates — supervisor fallback ────────────────────────
            elif mode == "updates":
                nodes = list(payload.keys()) if isinstance(payload, dict) else []
                _dbg(
                    f"  updates | nodes={nodes} supervisor_text_sent={supervisor_text_sent}"
                )

                if not supervisor_text_sent:
                    for node, state in (payload or {}).items():
                        if node != "supervisor" or not isinstance(state, dict):
                            continue
                        msgs = state.get("messages", [])
                        _dbg(f"  supervisor update: {len(msgs)} messages")
                        for msg in reversed(msgs):
                            text = _extract_text(getattr(msg, "content", ""))
                            has_tc = bool(getattr(msg, "tool_calls", None))
                            _dbg(
                                f"    msg={type(msg).__name__} has_tc={has_tc} text={text[:60]!r}"
                            )
                            if text and not has_tc:
                                supervisor_text_sent = True
                                start = _start("supervisor")
                                if start:
                                    yield start
                                yield f"data: {json.dumps({'type': 'token', 'agent': 'supervisor', 'content': text})}\n\n"
                                _dbg(
                                    "  → supervisor final text emitted via updates fallback"
                                )
                                break

    except Exception as exc:
        print(f"[stream error] {exc}", flush=True)
        yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    _dbg(f"stream done | supervisor_text_sent={supervisor_text_sent}")
    yield f"data: {json.dumps({'type': 'done', 'supervisor_responded': supervisor_text_sent})}\n\n"


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def tutor_events(request: TutorRequest, user: AuthUser, tutor):
    """Yield transport-neutral public events for SSE and WebSocket clients."""
    try:
        uuid.UUID(request.session_id)
    except ValueError:
        yield {"type": "error", "content": "session_id must be a UUID"}
        yield {"type": "done"}
        return
    config = {
        "recursion_limit": 30,
        "configurable": {"thread_id": f"{user.id}:{request.session_id}"},
    }
    try:
        stored_history = await _REPOSITORY.get_chat_history(
            user.id, request.session_id, limit=24
        )
    except Exception as exc:
        print(f"[chat history read error] {exc}", flush=True)
        yield {"type": "error", "content": "Không thể tải lịch sử hội thoại"}
        yield {"type": "done"}
        return
    checkpoint_has_conversation = False
    try:
        if hasattr(tutor, "aget_state"):
            snapshot = await tutor.aget_state(config)
            checkpoint_has_conversation = bool(
                (snapshot.values or {}).get("conversation")
            )
    except Exception:
        # A missing/expired checkpoint is expected; database history restores it.
        pass
    visible_message = (
        request.display_message or request.message or "Phân tích tệp đính kèm."
    ).strip()
    conversation_input = [{"role": "user", "content": visible_message}]
    if not checkpoint_has_conversation:
        conversation_input = [
            {"role": item.role, "content": item.content}
            for item in stored_history
            if item.request_id != str(request.request_id)
        ] + conversation_input
    graph_input = {
        "user_id": user.id,
        "session_id": request.session_id,
        "message": request.message,
        "attachment_ids": request.attachment_ids,
        "knowledge_document_ids": request.knowledge_document_ids,
        "conversation": conversation_input,
    }
    stage_labels = {
        "perception": "Đang đọc và chuẩn hóa đề bài",
        "clarification": "Cần xác nhận lại nội dung đề",
        "visualization_router": "Đang xác định nhu cầu trực quan hóa",
        "visualization_tool": "Đang tạo animation minh họa",
        "retrieval": "Đang đối chiếu học liệu đã duyệt",
        "tutor": "Đang chuẩn bị gợi ý phù hợp",
        "assessment": "Đang đánh giá bài làm",
        "mastery": "Đang cập nhật tiến bộ",
    }
    visualization_progress_labels = {
        "cache_hit": "Đang dùng lại animation đã tạo",
        "concept_analysis": "Đang phân tích khái niệm và công thức",
        "visualization_planning": "Đang thiết kế chuyển động trực quan",
        "code_generation": "Đang tạo scene Manim",
        "rendering": "Đang render animation",
        "repairing": "Đang tự sửa scene sau lỗi render",
        "completed": "Animation đã sẵn sàng",
    }
    current_visualization: dict | None = None
    try:
        await _REPOSITORY.save_chat_message(
            ChatMessage(
                user_id=user.id,
                session_id=request.session_id,
                request_id=str(request.request_id),
                role="user",
                content=visible_message,
                metadata={
                    "attachment_ids": request.attachment_ids,
                    "subject": request.subject.value if request.subject else None,
                },
            )
        )
        async for raw in tutor.astream(
            graph_input,
            config=config,
            stream_mode=["custom", "updates"],
        ):
            if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], str):
                mode, payload = raw
            elif isinstance(raw, dict):
                mode, payload = "updates", raw
            else:
                continue
            if mode == "custom":
                if isinstance(payload, dict) and payload.get("type") == "token":
                    yield {
                        "type": "token",
                        "content": str(payload.get("content", "")),
                    }
                elif (
                    isinstance(payload, dict)
                    and payload.get("type") == "visualization_progress"
                ):
                    progress_stage = str(payload.get("stage", ""))
                    yield {
                        "type": "stage",
                        "stage": f"visualization:{progress_stage}",
                        "label": visualization_progress_labels.get(
                            progress_stage, "Đang xử lý animation"
                        ),
                    }
                continue
            if mode != "updates" or not isinstance(payload, dict):
                continue
            for node, values in payload.items():
                if node not in stage_labels or not isinstance(values, dict):
                    continue
                yield {"type": "stage", "stage": node, "label": stage_labels[node]}
                if node == "perception" and values.get("problem"):
                    yield {
                        "type": "problem_extracted",
                        "problem": state_json(values["problem"]),
                    }
                if node == "visualization_tool" and values.get("visualization"):
                    visual = state_json(values["visualization"])
                    if visual.get("success"):
                        current_visualization = {
                            key: visual.get(key)
                            for key in (
                                "success",
                                "subject",
                                "concept",
                                "formulas",
                                "explanation",
                                "key_takeaway",
                                "artifact_id",
                                "video_url",
                                "scene_url",
                                "cached",
                                "attempts",
                            )
                        }
                        yield {"type": "visualization", **current_visualization}
                response = values.get("tutor_response")
                # Assessment may return the same response with evidence attached.
                # Only student-facing nodes are allowed to emit chat messages.
                if response and node in {"tutor", "clarification"}:
                    event_type = "hint" if response.get("feedback_type") in {"guidance", "clarification"} else "feedback"
                    await _REPOSITORY.save_chat_message(
                        ChatMessage(
                            user_id=user.id,
                            session_id=request.session_id,
                            request_id=str(request.request_id),
                            role="assistant",
                            content=str(response.get("message", "")).strip(),
                            metadata={
                                "event_type": event_type,
                                "hint_level": response.get("hint_level", 0),
                                "subject": request.subject.value
                                if request.subject
                                else None,
                                "visualization": current_visualization,
                            },
                        )
                    )
                    yield {"type": event_type, **state_json(response)}
                    for citation in response.get("citations", []):
                        yield {"type": "citation", **state_json(citation)}
                for mastery in values.get("mastery_updates", []):
                    yield {"type": "mastery_updated", **state_json(mastery)}
                if values.get("recommendations"):
                    yield {
                        "type": "recommendations_updated",
                        "recommendations": state_json(values["recommendations"]),
                    }
    except Exception as exc:
        print(f"[tutor stream error] {exc}", flush=True)
        yield {"type": "error", "content": "Tutor request failed"}
    yield {"type": "done"}


async def tutor_event_stream(request: TutorRequest, user: AuthUser, tutor):
    async for payload in tutor_events(request, user, tutor):
        yield _sse(payload)


async def legacy_tutor_event_stream(request: InferenceRequest, user: AuthUser, tutor):
    """One-release compatibility adapter for the former Agent Activity UI."""
    try:
        normalized_session = str(uuid.UUID(request.session_id))
    except ValueError:
        normalized_session = str(uuid.uuid5(uuid.NAMESPACE_URL, request.session_id))
    sent_text = False
    yield _sse({"type": "agent_start", "agent": "supervisor"})
    tutor_request = TutorRequest(
        session_id=normalized_session,
        message=request.message,
        attachment_ids=[],
    )
    async for frame in tutor_event_stream(tutor_request, user, tutor):
        try:
            payload = json.loads(frame.removeprefix("data: ").strip())
        except (ValueError, TypeError):
            continue
        if payload.get("type") == "token":
            sent_text = True
            yield _sse(
                {"type": "token", "agent": "supervisor", "content": payload.get("content", "")}
            )
        elif payload.get("type") in {"hint", "feedback"} and not sent_text:
            sent_text = True
            yield _sse(
                {"type": "token", "agent": "supervisor", "content": payload.get("message", "")}
            )
        elif payload.get("type") == "error":
            yield _sse(payload)
        elif payload.get("type") == "done":
            yield _sse({"type": "done", "supervisor_responded": sent_text})


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/api/stream")
async def stream_inference(
    request: InferenceRequest,
    req: Request,
    user: AuthUser = Depends(get_current_user),
):
    _RATE_LIMITER.check(f"legacy:{user.id}")
    return StreamingResponse(
        legacy_tutor_event_stream(request, user, req.app.state.tutor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/uploads")
async def upload_asset(
    file: UploadFile = File(...),
    user: AuthUser = Depends(get_current_user),
):
    _RATE_LIMITER.check(f"upload:{user.id}")
    record = await _STORAGE.save(user, file)
    return {"asset_id": record.id, "preview_url": record.preview_url}


@app.post("/api/upload", deprecated=True)
async def upload_image_compat(
    file: UploadFile = File(...),
    user: AuthUser = Depends(get_current_user),
):
    record = await _STORAGE.save(user, file)
    return {
        "asset_id": record.id,
        "preview_url": record.preview_url,
        "path": f"asset:{record.id}",
    }


@app.get("/api/assets/{asset_id}")
async def get_asset(
    asset_id: str,
    user: AuthUser = Depends(get_current_user),
):
    record = await _REPOSITORY.get_asset(asset_id, user.id)
    if not record:
        raise HTTPException(status_code=404, detail="Asset not found")
    return Response(await _STORAGE.read(record), media_type=record.mime_type)


@app.get("/api/visualizations/{artifact_id}/video")
async def get_visualization_video(
    artifact_id: str,
    user: AuthUser = Depends(get_current_user),
):
    path = _VISUALIZATION_STORE.resolve_owned_file(
        artifact_id, user.id, kind="video"
    )
    if not path:
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(path, media_type="video/mp4", filename="stem-animation.mp4")


@app.get("/api/visualizations/{artifact_id}/scene")
async def get_visualization_scene(
    artifact_id: str,
    user: AuthUser = Depends(get_current_user),
):
    path = _VISUALIZATION_STORE.resolve_owned_file(
        artifact_id, user.id, kind="scene"
    )
    if not path:
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(
        path, media_type="text/x-python; charset=utf-8", filename="stem_scene.py"
    )


@app.post("/api/tutor/stream")
async def stream_tutor(
    request: TutorRequest,
    req: Request,
    user: AuthUser = Depends(get_current_user),
):
    if not request.message.strip() and not request.attachment_ids:
        raise HTTPException(status_code=422, detail="Message or attachment is required")
    _RATE_LIMITER.check(f"tutor:{user.id}")
    return StreamingResponse(
        tutor_event_stream(request, user, req.app.state.tutor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/api/tutor/ws")
async def tutor_websocket(websocket: WebSocket):
    """Stream one tutor turn as JSON frames; auth is supplied in the start frame."""
    await websocket.accept()
    try:
        start = await websocket.receive_json()
        token = start.get("access_token") if isinstance(start, dict) else None
        request_data = start.get("request", start) if isinstance(start, dict) else {}
        user = await authenticate_access_token(token)
        request = TutorRequest.model_validate(request_data)
        if not request.message.strip() and not request.attachment_ids:
            await websocket.send_json(
                {"type": "error", "content": "Message or attachment is required"}
            )
            await websocket.send_json({"type": "done"})
            return
        client_ip = websocket.client.host if websocket.client else "unknown"
        _RATE_LIMITER.check(f"ws-ip:{client_ip}")
        _RATE_LIMITER.check(f"tutor:{user.id}")
        async for event in tutor_events(request, user, websocket.app.state.tutor):
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    except (HTTPException, ValidationError) as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else "Invalid tutor request"
        await websocket.send_json({"type": "error", "content": detail})
        await websocket.send_json({"type": "done"})
    except Exception:
        await websocket.send_json({"type": "error", "content": "Tutor request failed"})
        await websocket.send_json({"type": "done"})
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/health")
async def health(req: Request):
    return {
        "status": "ok",
        "memory": type(req.app.state.checkpointer).__name__,
        "chat_history": type(_REPOSITORY).__name__,
        "inference": "ollama-cloud" if is_ollama_cloud(_SETTINGS.ollama_base_url) else "ollama-compatible",
        "model": normalize_ollama_model(_SETTINGS.tutor_model, _SETTINGS.ollama_base_url),
        "visualization_model": normalize_ollama_model(
            _SETTINGS.visualization_model, _SETTINGS.ollama_base_url
        ),
        "visualization_runtime": (
            "ready" if importlib.util.find_spec("manim") else "missing-manim"
        ),
        "embeddings": "configured" if _SETTINGS.embedding_base_url else "lexical-fallback",
    }


@app.get("/api/me")
async def get_me(user: AuthUser = Depends(get_current_user)):
    return {
        "id": user.id,
        "email": user.email,
        "is_admin": user.is_admin,
        "auth_mode": "supabase" if _SETTINGS.supabase_enabled else "development",
    }


@app.get("/api/me/mastery")
async def get_my_mastery(user: AuthUser = Depends(get_current_user)):
    records = [
        item for item in await _REPOSITORY.get_mastery(user.id) if item.attempts > 0
    ]
    summary = summarize_learning_profile(records)
    return {
        "items": [item.model_dump(mode="json") for item in records],
        "summary": summary.model_dump(mode="json"),
    }


@app.get("/api/me/recommendations")
async def get_my_recommendations(user: AuthUser = Depends(get_current_user)):
    mastery = await _REPOSITORY.get_mastery(user.id)
    trends = await _REPOSITORY.get_exam_trends()
    items = rank_recommendations(mastery, trends)
    return {"items": [item.model_dump(mode="json") for item in items]}


@app.get("/api/me/exam-plan/{subject}")
async def get_my_exam_plan(
    subject: Subject,
    target_score: float | None = Query(default=None, ge=0.0, le=10.0),
    user: AuthUser = Depends(get_current_user),
):
    if subject == Subject.UNKNOWN:
        raise HTTPException(status_code=422, detail="A STEM subject is required")
    _RATE_LIMITER.check(f"exam-plan:{user.id}")
    plan = await LearningPathSupervisor(_REPOSITORY).build_plan(
        user.id, subject, target_score=target_score
    )
    return plan.model_dump(mode="json")


@app.delete("/api/me", status_code=204)
async def delete_my_data(user: AuthUser = Depends(get_current_user)):
    await _STORAGE.delete_user_assets(user.id)
    await _REPOSITORY.delete_user_data(user.id)
    return Response(status_code=204)


@app.post("/api/admin/exam-sync", status_code=202)
async def admin_exam_sync(
    payload: ExamSyncRequest,
    user: AuthUser = Depends(require_admin),
):
    result = await asyncio.to_thread(enqueue_initial_sync, extract=payload.extract)
    return {"status": "queued" if isinstance(result, str) else "completed", "result": result}


@app.post("/api/admin/content", status_code=201)
async def admin_ingest_content(
    payload: ContentIngestRequest,
    user: AuthUser = Depends(require_admin),
):
    if payload.subject == Subject.UNKNOWN:
        raise HTTPException(status_code=422, detail="A STEM subject is required")
    try:
        document_id, chunks, embedding_status = await ContentIngestionService(
            _REPOSITORY
        ).ingest(
            user_id=user.id,
            **payload.model_dump(mode="json"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "document_id": document_id,
        "chunks": chunks,
        "embedding_status": embedding_status,
        "review_status": "approved",
        "deprecated": True,
    }


@app.post("/api/knowledge", status_code=202)
async def upload_knowledge(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subject: Subject = Form(...),
    title: str | None = Form(default=None, max_length=300),
    skill_ids: str = Form(default=""),
    rights_note: str = Form(
        default="Tài liệu do người dùng cung cấp để phục vụ học tập cá nhân",
        max_length=500,
    ),
    user: AuthUser = Depends(get_current_user),
):
    """Stage a private document and trigger asynchronous RAG ingestion."""
    if subject == Subject.UNKNOWN:
        raise HTTPException(status_code=422, detail="A STEM subject is required")
    _RATE_LIMITER.check(f"knowledge-upload:{user.id}")
    try:
        data, filename, mime_type = await read_knowledge_payload(
            file, _SETTINGS.upload_max_bytes
        )
    except KnowledgeFileError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    try:
        parsed_skill_ids = json.loads(skill_ids) if skill_ids.strip().startswith("[") else [
            item.strip() for item in skill_ids.split(",") if item.strip()
        ]
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="skill_ids must be a JSON array or CSV") from exc
    if not isinstance(parsed_skill_ids, list) or not all(
        isinstance(item, str) for item in parsed_skill_ids
    ):
        raise HTTPException(status_code=422, detail="skill_ids must contain strings")
    parsed_skill_ids = list(dict.fromkeys(parsed_skill_ids))
    if len(parsed_skill_ids) > 20:
        raise HTTPException(status_code=422, detail="At most 20 skill_ids are allowed")
    invalid_skills = [
        item
        for item in parsed_skill_ids
        if item not in SKILL_BY_ID or SKILL_BY_ID[item].subject != subject
    ]
    if invalid_skills:
        raise HTTPException(
            status_code=422,
            detail=f"Skills do not belong to {subject.value}: {', '.join(invalid_skills)}",
        )

    document_title = (title or Path(filename).stem).strip()
    if len(document_title) < 3:
        raise HTTPException(status_code=422, detail="title must contain at least 3 characters")

    job_id = str(uuid.uuid4())
    try:
        storage_path = await _KNOWLEDGE_STORAGE.save(
            job_id, filename, data, mime_type
        )
        job = await _REPOSITORY.create_knowledge_job(
            KnowledgeIngestionJob(
                id=job_id,
                user_id=user.id,
                title=document_title,
                subject=subject,
                original_filename=filename,
                storage_path=storage_path,
                rights_note=rights_note,
                skill_ids=parsed_skill_ids,
                size_bytes=len(data),
            )
        )
    except Exception as exc:
        if "storage_path" in locals():
            try:
                await _KNOWLEDGE_STORAGE.delete(storage_path)
            except Exception:
                pass
        raise HTTPException(status_code=503, detail="Could not queue knowledge upload") from exc

    # The response is returned before extraction/embedding starts. Redis/RQ is used
    # when available; local development falls back to an in-process background task.
    background_tasks.add_task(_dispatch_knowledge_job, job.id)
    return {
        "job_id": job.id,
        "status": job.status,
        "title": job.title,
        "filename": job.original_filename,
        "subject": job.subject.value,
        "visibility": "private",
        "status_url": f"/api/knowledge/jobs/{job.id}",
    }


@app.get("/api/knowledge/jobs")
async def list_knowledge_jobs(
    subject: Subject | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    user: AuthUser = Depends(get_current_user),
):
    if subject == Subject.UNKNOWN:
        raise HTTPException(status_code=422, detail="A STEM subject is required")
    jobs = await _REPOSITORY.list_knowledge_jobs(user.id, subject, limit)
    return {"items": [item.model_dump(mode="json") for item in jobs]}


@app.get("/api/knowledge/documents")
async def list_knowledge_documents(
    subject: Subject | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    user: AuthUser = Depends(get_current_user),
):
    if subject == Subject.UNKNOWN:
        raise HTTPException(status_code=422, detail="A STEM subject is required")
    documents = await _REPOSITORY.list_knowledge_documents(user.id, subject, limit)
    return {"items": [item.model_dump(mode="json") for item in documents]}


@app.get("/api/knowledge/jobs/{job_id}")
async def get_knowledge_job(
    job_id: str,
    user: AuthUser = Depends(get_current_user),
):
    job = await _REPOSITORY.get_knowledge_job(job_id, user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Knowledge job not found")
    return job.model_dump(mode="json")


@app.post("/api/webhooks/knowledge-uploaded", status_code=202)
async def knowledge_uploaded_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Signed, idempotent callback for storage/webhook integrations."""
    secret = _SETTINGS.knowledge_webhook_secret
    if not secret:
        raise HTTPException(status_code=503, detail="Knowledge webhook is not configured")
    body = await request.body()
    provided = request.headers.get("x-knowledge-signature", "")
    if provided.startswith("sha256="):
        provided = provided.removeprefix("sha256=")
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    try:
        event = KnowledgeWebhookEvent.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="Invalid webhook payload") from exc
    job = await _REPOSITORY.get_knowledge_job(event.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Knowledge job not found")
    background_tasks.add_task(_dispatch_knowledge_job, job.id)
    return {"accepted": True, "job_id": job.id, "status": job.status}


@app.get("/api/admin/review-queue")
async def admin_review_queue(
    limit: int = 100,
    user: AuthUser = Depends(require_admin),
):
    items = await _REPOSITORY.list_review_queue(min(max(limit, 1), 200))
    return {"items": [item.model_dump(mode="json") for item in items]}


@app.patch("/api/admin/review-queue/{question_id}")
async def admin_patch_review(
    question_id: str,
    patch: ReviewPatch,
    user: AuthUser = Depends(require_admin),
):
    item = await _REPOSITORY.patch_review(question_id, patch)
    if not item:
        raise HTTPException(status_code=404, detail="Question not found")
    return item.model_dump(mode="json")


@app.get("/api/chat/sessions")
async def list_chat_sessions(
    limit: int = Query(default=50, ge=1, le=100),
    user: AuthUser = Depends(get_current_user),
):
    sessions = await _REPOSITORY.list_chat_sessions(user.id, limit)
    return {"items": [item.model_dump(mode="json") for item in sessions]}


@app.get("/api/history/{session_id}")
async def get_history(
    session_id: str,
    req: Request,
    user: AuthUser = Depends(get_current_user),
):
    try:
        uuid.UUID(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="session_id must be a UUID") from exc
    config = {"configurable": {"thread_id": f"{user.id}:{session_id}"}}
    persistent_messages = await _REPOSITORY.get_chat_history(
        user.id, session_id, limit=100
    )
    values = {}
    try:
        graph = req.app.state.tutor
        if hasattr(graph, "aget_state"):
            state = await graph.aget_state(config)
        else:
            state = graph.get_state(config)
        values = state.values or {}
    except Exception:
        # Durable messages remain available even when checkpoints expired.
        pass
    messages = [
        {
            "id": item.id,
            "request_id": item.request_id,
            "role": item.role,
            "content": item.content,
            "metadata": item.metadata,
            "created_at": item.created_at.isoformat(),
        }
        for item in persistent_messages
    ]
    if not messages:
        messages = values.get("conversation", [])
    return {
        "session_id": session_id,
        "problem": values.get("problem"),
        "hint_level": values.get("hint_level", 0),
        "last_response": values.get("tutor_response"),
        "messages": messages,
        "persistent": bool(persistent_messages),
    }
