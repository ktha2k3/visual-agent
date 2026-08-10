import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage
from pydantic import BaseModel

from agents.supervisor_agent import build_supervisor

_REDIS_URL    = os.getenv("REDIS_URL", "")
_DEBUG        = os.getenv("DEBUG_STREAM", "false").lower() == "true"
_UPLOAD_DIR   = Path("assets/uploads")
_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


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
                app.state.supervisor    = build_supervisor(checkpointer=cp)
                app.state.checkpointer  = cp
                yield   # server runs here, inside the async context
            return      # skip MemorySaver fallback below
        except Exception as exc:
            print(f"[memory] Redis unavailable ({exc}), falling back to MemorySaver", flush=True)

    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    print("[memory] Using in-memory checkpointer (sessions lost on restart)", flush=True)
    app.state.supervisor   = build_supervisor(checkpointer=checkpointer)
    app.state.checkpointer = checkpointer
    yield


app = FastAPI(title="Visual Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model ──────────────────────────────────────────────────────────────
class InferenceRequest(BaseModel):
    message: str
    session_id: str = "default"


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
            p.get("text", "") for p in content
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
                node  = metadata.get("langgraph_node", "?")
                ns    = metadata.get("langgraph_checkpoint_ns", "")
                _dbg(f"  messages | node={node!r} ns={ns!r} agent={agent!r} type={type(chunk).__name__}")

                start = _start(agent)
                if start:
                    yield start

                if isinstance(chunk, ToolMessage):
                    tool_name = chunk.name or ""
                    _dbg(f"  ToolMessage tool={tool_name!r} content={str(chunk.content)[:80]!r}")
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
                    tcs  = [tc.get("name") for tc in (chunk.tool_call_chunks or []) if tc.get("name")]
                    _dbg(f"  AIMessageChunk text={text[:60]!r} tool_calls={tcs}")

                    if text:
                        if agent == "supervisor":
                            supervisor_text_sent = True
                        yield f"data: {json.dumps({'type': 'token', 'agent': agent, 'content': text})}\n\n"

                    for tc in (chunk.tool_call_chunks or []):
                        if tc.get("name") and "transfer" not in tc["name"].lower():
                            yield f"data: {json.dumps({'type': 'tool_call', 'agent': agent, 'tool': tc['name'], 'args': str(tc.get('args', ''))})}\n\n"

            # ── State updates — supervisor fallback ────────────────────────
            elif mode == "updates":
                nodes = list(payload.keys()) if isinstance(payload, dict) else []
                _dbg(f"  updates | nodes={nodes} supervisor_text_sent={supervisor_text_sent}")

                if not supervisor_text_sent:
                    for node, state in (payload or {}).items():
                        if node != "supervisor" or not isinstance(state, dict):
                            continue
                        msgs = state.get("messages", [])
                        _dbg(f"  supervisor update: {len(msgs)} messages")
                        for msg in reversed(msgs):
                            text         = _extract_text(getattr(msg, "content", ""))
                            has_tc       = bool(getattr(msg, "tool_calls", None))
                            _dbg(f"    msg={type(msg).__name__} has_tc={has_tc} text={text[:60]!r}")
                            if text and not has_tc:
                                supervisor_text_sent = True
                                start = _start("supervisor")
                                if start:
                                    yield start
                                yield f"data: {json.dumps({'type': 'token', 'agent': 'supervisor', 'content': text})}\n\n"
                                _dbg("  → supervisor final text emitted via updates fallback")
                                break

    except Exception as exc:
        print(f"[stream error] {exc}", flush=True)
        yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    _dbg(f"stream done | supervisor_text_sent={supervisor_text_sent}")
    yield f"data: {json.dumps({'type': 'done', 'supervisor_responded': supervisor_text_sent})}\n\n"


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/api/stream")
async def stream_inference(request: InferenceRequest, req: Request):
    return StreamingResponse(
        event_stream(request.message, request.session_id, req.app.state.supervisor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "image.jpg").suffix.lower()
    if ext not in _ALLOWED_EXTS:
        ext = ".jpg"
    dest = _UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": str(dest.resolve())}


@app.get("/health")
async def health(req: Request):
    return {"status": "ok", "memory": type(req.app.state.checkpointer).__name__}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str, req: Request):
    config = {"configurable": {"thread_id": session_id}}
    try:
        state    = req.app.state.supervisor.get_state(config)
        messages = state.values.get("messages", [])
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": [
                {"role": getattr(m, "type", "?"), "preview": str(getattr(m, "content", ""))[:120]}
                for m in messages
            ],
        }
    except Exception as exc:
        return {"session_id": session_id, "error": str(exc)}
