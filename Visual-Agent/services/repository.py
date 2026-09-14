from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import threading
import unicodedata
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from education.models import (
    AssetRecord,
    AuthUser,
    ChatMessage,
    ChatSessionRecord,
    ChatSessionSummary,
    Citation,
    KnowledgeIngestionJob,
    KnowledgeDocumentSummary,
    LearningMode,
    MasteryRecord,
    Subject,
)
from utils.config import Settings, get_settings


class TutorRepository(Protocol):
    async def save_asset(self, record: AssetRecord) -> None: ...
    async def get_asset(self, asset_id: str, user_id: str) -> AssetRecord | None: ...
    async def list_assets(self, user_id: str) -> list[AssetRecord]: ...
    async def list_expired_assets(self, before: datetime) -> list[AssetRecord]: ...
    async def delete_asset_records(self, asset_ids: list[str]) -> None: ...
    async def get_mastery(self, user_id: str) -> list[MasteryRecord]: ...
    async def upsert_mastery(self, record: MasteryRecord) -> MasteryRecord: ...
    async def save_chat_message(self, message: ChatMessage) -> ChatMessage: ...
    async def create_chat_session(
        self,
        user_id: str,
        subject: Subject,
        learning_mode: LearningMode = LearningMode.GUIDED_PROBLEM,
    ) -> ChatSessionRecord: ...
    async def get_chat_session(
        self, user_id: str, session_id: str
    ) -> ChatSessionRecord | None: ...
    async def update_chat_session_mode(
        self, user_id: str, session_id: str, learning_mode: LearningMode
    ) -> ChatSessionRecord | None: ...
    async def get_chat_history(
        self, user_id: str, session_id: str, limit: int = 100
    ) -> list[ChatMessage]: ...
    async def list_chat_sessions(
        self,
        user_id: str,
        limit: int = 50,
        subject: Subject | None = None,
    ) -> list[ChatSessionSummary]: ...
    async def retrieve_context(
        self,
        subject: Subject,
        query: str,
        skill_ids: list[str],
        limit: int = 5,
        query_embedding: list[float] | None = None,
        user_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[Citation]: ...
    async def save_approved_content(
        self, document: dict[str, Any], chunks: list[dict[str, Any]]
    ) -> str: ...
    async def list_knowledge_documents(
        self, user_id: str, subject: Subject | None = None, limit: int = 50
    ) -> list[KnowledgeDocumentSummary]: ...
    async def create_knowledge_job(
        self, job: KnowledgeIngestionJob
    ) -> KnowledgeIngestionJob: ...
    async def get_knowledge_job(
        self, job_id: str, user_id: str | None = None
    ) -> KnowledgeIngestionJob | None: ...
    async def list_knowledge_jobs(
        self, user_id: str, subject: Subject | None = None, limit: int = 20
    ) -> list[KnowledgeIngestionJob]: ...
    async def claim_knowledge_job(
        self, job_id: str
    ) -> KnowledgeIngestionJob | None: ...
    async def update_knowledge_job(
        self, job_id: str, updates: dict[str, Any]
    ) -> KnowledgeIngestionJob | None: ...
    async def save_problem(
        self, user_id: str, session_id: str, problem: dict[str, Any], asset_id: str | None
    ) -> str: ...
    async def save_hint_event(
        self, user_id: str, session_id: str, problem_id: str, response: dict[str, Any]
    ) -> None: ...
    async def save_attempt(
        self, user_id: str, session_id: str, problem_id: str, answer: str,
        answer_correct: bool, evidence: float | None, misconception: str | None,
    ) -> None: ...
    async def delete_user_data(self, user_id: str) -> None: ...


class MemoryRepository:
    """Useful local fallback and deterministic test repository."""

    def __init__(self):
        self.assets: dict[str, AssetRecord] = {}
        self.mastery: dict[tuple[str, str], MasteryRecord] = {}
        self.content_documents: dict[str, dict[str, Any]] = {}
        self.content_chunks: list[dict[str, Any]] = []
        self.knowledge_jobs: dict[str, KnowledgeIngestionJob] = {}
        self.problems: list[dict[str, Any]] = []
        self.hint_events: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []
        self.chat_messages: dict[tuple[str, str, str, str], ChatMessage] = {}
        self.chat_sessions: dict[tuple[str, str], ChatSessionRecord] = {}
        self._lock = asyncio.Lock()

    async def save_asset(self, record: AssetRecord) -> None:
        async with self._lock:
            self.assets[record.id] = record

    async def get_asset(self, asset_id: str, user_id: str) -> AssetRecord | None:
        record = self.assets.get(asset_id)
        return record if record and record.user_id == user_id else None

    async def list_assets(self, user_id: str) -> list[AssetRecord]:
        return [item for item in self.assets.values() if item.user_id == user_id]

    async def list_expired_assets(self, before: datetime) -> list[AssetRecord]:
        return [item for item in self.assets.values() if item.created_at < before]

    async def delete_asset_records(self, asset_ids: list[str]) -> None:
        for asset_id in asset_ids:
            self.assets.pop(asset_id, None)

    async def get_mastery(self, user_id: str) -> list[MasteryRecord]:
        return [item for (uid, _), item in self.mastery.items() if uid == user_id]

    async def upsert_mastery(self, record: MasteryRecord) -> MasteryRecord:
        self.mastery[(record.user_id, record.skill_id)] = record
        return record

    async def save_chat_message(self, message: ChatMessage) -> ChatMessage:
        subject_value = message.metadata.get("subject")
        try:
            subject = Subject(subject_value)
        except (TypeError, ValueError):
            subject = Subject.UNKNOWN
        session_key = (message.user_id, message.session_id)
        existing_session = self.chat_sessions.get(session_key)
        try:
            learning_mode = LearningMode(message.metadata.get("learning_mode"))
        except (TypeError, ValueError):
            learning_mode = LearningMode.GUIDED_PROBLEM
        if existing_session is None and subject != Subject.UNKNOWN:
            self.chat_sessions[session_key] = ChatSessionRecord(
                id=message.session_id,
                user_id=message.user_id,
                subject=subject,
                learning_mode=learning_mode,
            )
        elif existing_session is not None:
            self.chat_sessions[session_key] = existing_session.model_copy(
                update={"updated_at": datetime.now().astimezone()}
            )
        key = (
            message.user_id,
            message.session_id,
            message.request_id,
            message.role,
        )
        existing = self.chat_messages.get(key)
        stored = message.model_copy(
            update={"id": existing.id if existing else message.id or str(uuid.uuid4())}
        )
        self.chat_messages[key] = stored
        return stored

    async def create_chat_session(
        self,
        user_id: str,
        subject: Subject,
        learning_mode: LearningMode = LearningMode.GUIDED_PROBLEM,
    ) -> ChatSessionRecord:
        session = ChatSessionRecord(
            user_id=user_id, subject=subject, learning_mode=learning_mode
        )
        self.chat_sessions[(user_id, session.id)] = session
        return session

    async def get_chat_session(
        self, user_id: str, session_id: str
    ) -> ChatSessionRecord | None:
        return self.chat_sessions.get((user_id, session_id))

    async def update_chat_session_mode(
        self, user_id: str, session_id: str, learning_mode: LearningMode
    ) -> ChatSessionRecord | None:
        key = (user_id, session_id)
        session = self.chat_sessions.get(key)
        if session is None:
            return None
        updated = session.model_copy(
            update={
                "learning_mode": learning_mode,
                "updated_at": datetime.now().astimezone(),
            }
        )
        self.chat_sessions[key] = updated
        return updated

    async def get_chat_history(
        self, user_id: str, session_id: str, limit: int = 100
    ) -> list[ChatMessage]:
        messages = [
            item
            for item in self.chat_messages.values()
            if item.user_id == user_id and item.session_id == session_id
        ]
        return sorted(messages, key=lambda item: item.created_at)[-limit:]

    async def list_chat_sessions(
        self,
        user_id: str,
        limit: int = 50,
        subject: Subject | None = None,
    ) -> list[ChatSessionSummary]:
        messages = sorted(
            [item for item in self.chat_messages.values() if item.user_id == user_id],
            key=lambda item: item.created_at,
        )
        sessions = [
            item for (uid, _), item in self.chat_sessions.items() if uid == user_id
        ]
        return _summarize_chat_sessions(messages, limit, subject, sessions)

    async def retrieve_context(
        self,
        subject: Subject,
        query: str,
        skill_ids: list[str],
        limit: int = 5,
        query_embedding: list[float] | None = None,
        user_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[Citation]:
        if not user_id:
            return []
        matches = [
            item
            for item in self.content_chunks
            if item["subject"] == subject.value
            and item.get("user_id") in {None, user_id}
            and (
                not document_ids
                or item.get("user_id") is None
                or item.get("document_id") in set(document_ids)
            )
        ]
        matches = _rank_content_chunks(matches, query, skill_ids, query_embedding)
        return [
            Citation(
                title=item["title"], url=item["source_url"], excerpt=item["content"][:500]
            )
            for item in matches[:limit]
        ]

    async def save_approved_content(
        self, document: dict[str, Any], chunks: list[dict[str, Any]]
    ) -> str:
        document_id = str(uuid.uuid4())
        self.content_documents[document_id] = {
            **document,
            "id": document_id,
            "created_at": document.get("created_at") or datetime.now(),
        }
        self.content_chunks.extend({**item, "document_id": document_id} for item in chunks)
        return document_id

    async def list_knowledge_documents(
        self, user_id: str, subject: Subject | None = None, limit: int = 50
    ) -> list[KnowledgeDocumentSummary]:
        items = []
        for document_id, document in self.content_documents.items():
            if document.get("user_id") != user_id:
                continue
            document_subject = Subject(document["subject"])
            if subject is not None and document_subject != subject:
                continue
            items.append(
                KnowledgeDocumentSummary(
                    id=document_id,
                    user_id=user_id,
                    title=document["title"],
                    subject=document_subject,
                    original_filename=document.get("original_filename"),
                    size_bytes=document.get("size_bytes"),
                    embedding_status=document.get("embedding_status", "ready"),
                    review_status=document.get("review_status", "approved"),
                    chunks=sum(
                        1
                        for item in self.content_chunks
                        if item.get("document_id") == document_id
                    ),
                    created_at=document["created_at"],
                )
            )
        return sorted(items, key=lambda item: item.created_at, reverse=True)[:limit]

    async def create_knowledge_job(
        self, job: KnowledgeIngestionJob
    ) -> KnowledgeIngestionJob:
        self.knowledge_jobs[job.id] = job
        return job

    async def get_knowledge_job(
        self, job_id: str, user_id: str | None = None
    ) -> KnowledgeIngestionJob | None:
        job = self.knowledge_jobs.get(job_id)
        return job if job and (user_id is None or job.user_id == user_id) else None

    async def list_knowledge_jobs(
        self, user_id: str, subject: Subject | None = None, limit: int = 20
    ) -> list[KnowledgeIngestionJob]:
        jobs = [
            item
            for item in self.knowledge_jobs.values()
            if item.user_id == user_id and (subject is None or item.subject == subject)
        ]
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)[:limit]

    async def claim_knowledge_job(
        self, job_id: str
    ) -> KnowledgeIngestionJob | None:
        async with self._lock:
            job = self.knowledge_jobs.get(job_id)
            if not job or job.status != "pending":
                return None
            claimed = job.model_copy(
                update={"status": "processing", "updated_at": datetime.now()}
            )
            self.knowledge_jobs[job_id] = claimed
            return claimed

    async def update_knowledge_job(
        self, job_id: str, updates: dict[str, Any]
    ) -> KnowledgeIngestionJob | None:
        async with self._lock:
            job = self.knowledge_jobs.get(job_id)
            if not job:
                return None
            updated = job.model_copy(
                update={**updates, "updated_at": datetime.now()}
            )
            self.knowledge_jobs[job_id] = updated
            return updated

    async def save_problem(
        self, user_id: str, session_id: str, problem: dict[str, Any], asset_id: str | None
    ) -> str:
        problem_id = str(len(self.problems) + 1)
        self.problems.append({"id": problem_id, "user_id": user_id, "session_id": session_id, **problem})
        return problem_id

    async def save_hint_event(
        self, user_id: str, session_id: str, problem_id: str, response: dict[str, Any]
    ) -> None:
        self.hint_events.append({"user_id": user_id, "session_id": session_id, "problem_id": problem_id, **response})

    async def save_attempt(
        self, user_id: str, session_id: str, problem_id: str, answer: str,
        answer_correct: bool, evidence: float | None, misconception: str | None,
    ) -> None:
        self.attempts.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "problem_id": problem_id,
                "answer": answer,
                "answer_correct": answer_correct,
                "evidence": evidence,
                "misconception": misconception,
            }
        )

    async def delete_user_data(self, user_id: str) -> None:
        self.assets = {key: value for key, value in self.assets.items() if value.user_id != user_id}
        self.mastery = {key: value for key, value in self.mastery.items() if key[0] != user_id}
        self.chat_messages = {
            key: value for key, value in self.chat_messages.items() if key[0] != user_id
        }
        self.chat_sessions = {
            key: value for key, value in self.chat_sessions.items() if key[0] != user_id
        }
        owned_documents = {
            key for key, value in self.content_documents.items() if value.get("user_id") == user_id
        }
        self.content_documents = {
            key: value for key, value in self.content_documents.items() if key not in owned_documents
        }
        self.content_chunks = [
            item for item in self.content_chunks if item.get("document_id") not in owned_documents
        ]
        self.knowledge_jobs = {
            key: value
            for key, value in self.knowledge_jobs.items()
            if value.user_id != user_id
        }
        self.problems = [item for item in self.problems if item["user_id"] != user_id]
        self.hint_events = [item for item in self.hint_events if item["user_id"] != user_id]
        self.attempts = [item for item in self.attempts if item["user_id"] != user_id]


class LocalRepository(MemoryRepository):
    """Development repository with durable SQLite chat history and knowledge."""

    def __init__(self, database_path: str):
        super().__init__()
        configured = Path(database_path)
        self.database_path = (
            configured
            if configured.is_absolute()
            else Path(__file__).resolve().parents[1] / configured
        ).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._sqlite_lock = threading.Lock()
        self._initialize_chat_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_chat_database(self) -> None:
        with self._sqlite_lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tutoring_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    subject TEXT NOT NULL CHECK (
                        subject IN ('math', 'physics', 'chemistry', 'biology')
                    ),
                    learning_mode TEXT NOT NULL DEFAULT 'guided_problem' CHECK (
                        learning_mode IN ('concept', 'guided_problem')
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            session_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(tutoring_sessions)"
                ).fetchall()
            }
            if "learning_mode" not in session_columns:
                connection.execute(
                    "ALTER TABLE tutoring_sessions ADD COLUMN learning_mode "
                    "TEXT NOT NULL DEFAULT 'guided_problem' CHECK "
                    "(learning_mode IN ('concept', 'guided_problem'))"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL CHECK (length(content) BETWEEN 1 AND 50000),
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE (user_id, session_id, request_id, role)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    publisher TEXT,
                    rights_note TEXT,
                    original_filename TEXT,
                    mime_type TEXT,
                    size_bytes INTEGER,
                    embedding_status TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    skill_ids_json TEXT NOT NULL DEFAULT '[]',
                    content TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    embedding_json TEXT,
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    rights_note TEXT NOT NULL,
                    skill_ids_json TEXT NOT NULL DEFAULT '[]',
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'processing', 'completed', 'failed')
                    ),
                    document_id TEXT,
                    chunks INTEGER,
                    embedding_status TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS knowledge_chunks_user_subject_idx
                ON knowledge_chunks(user_id, subject, review_status)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS tutoring_sessions_user_subject_updated_idx
                ON tutoring_sessions(user_id, subject, updated_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS chat_messages_user_session_created_idx
                ON chat_messages(user_id, session_id, created_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS knowledge_jobs_user_created_idx
                ON knowledge_ingestion_jobs(user_id, created_at DESC)
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO tutoring_sessions (
                    id, user_id, subject, learning_mode, created_at, updated_at
                )
                SELECT
                    session_id,
                    user_id,
                    CASE json_extract(metadata_json, '$.subject')
                        WHEN 'physics' THEN 'physics'
                        WHEN 'chemistry' THEN 'chemistry'
                        WHEN 'biology' THEN 'biology'
                        ELSE 'math'
                    END,
                    'guided_problem',
                    MIN(created_at),
                    MAX(created_at)
                FROM chat_messages
                GROUP BY session_id, user_id
                """
            )

    def _create_chat_session_sync(
        self, user_id: str, subject: Subject, learning_mode: LearningMode
    ) -> ChatSessionRecord:
        session = ChatSessionRecord(
            user_id=user_id, subject=subject, learning_mode=learning_mode
        )
        with self._sqlite_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tutoring_sessions (
                    id, user_id, subject, learning_mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.user_id,
                    session.subject.value,
                    session.learning_mode.value,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )
        return session

    async def create_chat_session(
        self,
        user_id: str,
        subject: Subject,
        learning_mode: LearningMode = LearningMode.GUIDED_PROBLEM,
    ) -> ChatSessionRecord:
        return await asyncio.to_thread(
            self._create_chat_session_sync, user_id, subject, learning_mode
        )

    def _get_chat_session_sync(
        self, user_id: str, session_id: str
    ) -> ChatSessionRecord | None:
        with self._sqlite_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tutoring_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        return ChatSessionRecord.model_validate(dict(row)) if row else None

    async def get_chat_session(
        self, user_id: str, session_id: str
    ) -> ChatSessionRecord | None:
        return await asyncio.to_thread(
            self._get_chat_session_sync, user_id, session_id
        )

    def _update_chat_session_mode_sync(
        self, user_id: str, session_id: str, learning_mode: LearningMode
    ) -> ChatSessionRecord | None:
        now = datetime.now().astimezone().isoformat()
        with self._sqlite_lock, self._connect() as connection:
            connection.execute(
                "UPDATE tutoring_sessions SET learning_mode = ?, updated_at = ? "
                "WHERE id = ? AND user_id = ?",
                (learning_mode.value, now, session_id, user_id),
            )
            row = connection.execute(
                "SELECT * FROM tutoring_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        return ChatSessionRecord.model_validate(dict(row)) if row else None

    async def update_chat_session_mode(
        self, user_id: str, session_id: str, learning_mode: LearningMode
    ) -> ChatSessionRecord | None:
        return await asyncio.to_thread(
            self._update_chat_session_mode_sync,
            user_id,
            session_id,
            learning_mode,
        )

    def _save_chat_message_sync(self, message: ChatMessage) -> ChatMessage:
        message_id = message.id or str(uuid.uuid4())
        subject_value = message.metadata.get("subject")
        normalized_subject = (
            subject_value
            if subject_value in {"math", "physics", "chemistry", "biology"}
            else None
        )
        try:
            normalized_mode = LearningMode(
                message.metadata.get("learning_mode")
            ).value
        except (TypeError, ValueError):
            normalized_mode = LearningMode.GUIDED_PROBLEM.value
        now = datetime.now().astimezone().isoformat()
        with self._sqlite_lock, self._connect() as connection:
            existing_session = connection.execute(
                "SELECT user_id, subject FROM tutoring_sessions WHERE id = ?",
                (message.session_id,),
            ).fetchone()
            if existing_session and existing_session["user_id"] != message.user_id:
                raise PermissionError("Chat session belongs to another user")
            if (
                existing_session
                and normalized_subject
                and existing_session["subject"] != normalized_subject
            ):
                raise ValueError("Chat session belongs to another subject")
            if not existing_session:
                if not normalized_subject:
                    normalized_subject = Subject.MATH.value
                connection.execute(
                    """
                    INSERT INTO tutoring_sessions (
                        id, user_id, subject, learning_mode, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.session_id,
                        message.user_id,
                        normalized_subject,
                        normalized_mode,
                        message.created_at.isoformat(),
                        now,
                    ),
                )
            else:
                connection.execute(
                    "UPDATE tutoring_sessions SET updated_at = ? WHERE id = ?",
                    (now, message.session_id),
                )
            connection.execute(
                """
                INSERT INTO chat_messages (
                    id, user_id, session_id, request_id, role, content,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, session_id, request_id, role) DO UPDATE SET
                    content = excluded.content,
                    metadata_json = excluded.metadata_json
                """,
                (
                    message_id,
                    message.user_id,
                    message.session_id,
                    message.request_id,
                    message.role,
                    message.content,
                    json.dumps(message.metadata, ensure_ascii=False),
                    message.created_at.isoformat(),
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM chat_messages
                WHERE user_id = ? AND session_id = ? AND request_id = ? AND role = ?
                """,
                (
                    message.user_id,
                    message.session_id,
                    message.request_id,
                    message.role,
                ),
            ).fetchone()
        return ChatMessage(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            request_id=row["request_id"],
            role=row["role"],
            content=row["content"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
        )

    async def save_chat_message(self, message: ChatMessage) -> ChatMessage:
        return await asyncio.to_thread(self._save_chat_message_sync, message)

    def _get_chat_history_sync(
        self, user_id: str, session_id: str, limit: int
    ) -> list[ChatMessage]:
        with self._sqlite_lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM chat_messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (user_id, session_id, limit),
            ).fetchall()
        return [
            ChatMessage(
                id=row["id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                request_id=row["request_id"],
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata_json"]),
                created_at=row["created_at"],
            )
            for row in reversed(rows)
        ]

    async def get_chat_history(
        self, user_id: str, session_id: str, limit: int = 100
    ) -> list[ChatMessage]:
        return await asyncio.to_thread(
            self._get_chat_history_sync, user_id, session_id, limit
        )

    def _list_chat_sessions_sync(
        self, user_id: str, limit: int, subject: Subject | None
    ) -> list[ChatSessionSummary]:
        with self._sqlite_lock, self._connect() as connection:
            session_rows = connection.execute(
                """
                SELECT * FROM tutoring_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            rows = connection.execute(
                """
                SELECT * FROM chat_messages
                WHERE user_id = ?
                ORDER BY created_at ASC
                """,
                (user_id,),
            ).fetchall()
        messages = [
            ChatMessage(
                id=row["id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                request_id=row["request_id"],
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]
        sessions = [
            ChatSessionRecord(
                id=row["id"],
                user_id=row["user_id"],
                subject=row["subject"],
                learning_mode=row["learning_mode"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in session_rows
        ]
        return _summarize_chat_sessions(messages, limit, subject, sessions)

    async def list_chat_sessions(
        self,
        user_id: str,
        limit: int = 50,
        subject: Subject | None = None,
    ) -> list[ChatSessionSummary]:
        return await asyncio.to_thread(
            self._list_chat_sessions_sync, user_id, limit, subject
        )

    def _save_approved_content_sync(
        self, document: dict[str, Any], chunks: list[dict[str, Any]]
    ) -> str:
        document_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        with self._sqlite_lock, self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                INSERT INTO knowledge_documents (
                    id, user_id, title, subject, source_url, publisher, rights_note,
                    original_filename, mime_type, size_bytes, embedding_status,
                    review_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    document["user_id"],
                    document["title"],
                    document["subject"],
                    document["source_url"],
                    document.get("publisher"),
                    document.get("rights_note"),
                    document.get("original_filename"),
                    document.get("mime_type"),
                    document.get("size_bytes"),
                    document.get("embedding_status", "ready"),
                    document.get("review_status", "approved"),
                    created_at,
                ),
            )
            connection.executemany(
                """
                INSERT INTO knowledge_chunks (
                    id, document_id, user_id, title, subject, skill_ids_json,
                    content, source_url, embedding_json, review_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(uuid.uuid4()),
                        document_id,
                        item["user_id"],
                        item["title"],
                        item["subject"],
                        json.dumps(item.get("skill_ids", []), ensure_ascii=False),
                        item["content"],
                        item.get("source_url", ""),
                        json.dumps(item["embedding"]) if item.get("embedding") else None,
                        item.get("review_status", "approved"),
                        created_at,
                    )
                    for item in chunks
                ],
            )
        return document_id

    async def save_approved_content(
        self, document: dict[str, Any], chunks: list[dict[str, Any]]
    ) -> str:
        return await asyncio.to_thread(self._save_approved_content_sync, document, chunks)

    def _list_knowledge_documents_sync(
        self, user_id: str, subject: Subject | None, limit: int
    ) -> list[KnowledgeDocumentSummary]:
        query = """
            SELECT d.*, COUNT(c.id) AS chunk_count
            FROM knowledge_documents d
            LEFT JOIN knowledge_chunks c ON c.document_id = d.id
            WHERE d.user_id = ?
        """
        params: list[Any] = [user_id]
        if subject is not None:
            query += " AND d.subject = ?"
            params.append(subject.value)
        query += " GROUP BY d.id ORDER BY d.created_at DESC LIMIT ?"
        params.append(limit)
        with self._sqlite_lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            KnowledgeDocumentSummary(
                id=row["id"],
                user_id=row["user_id"],
                title=row["title"],
                subject=row["subject"],
                original_filename=row["original_filename"],
                size_bytes=row["size_bytes"],
                embedding_status=row["embedding_status"],
                review_status=row["review_status"],
                chunks=row["chunk_count"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def list_knowledge_documents(
        self, user_id: str, subject: Subject | None = None, limit: int = 50
    ) -> list[KnowledgeDocumentSummary]:
        return await asyncio.to_thread(
            self._list_knowledge_documents_sync, user_id, subject, limit
        )

    @staticmethod
    def _knowledge_job_from_row(row: sqlite3.Row) -> KnowledgeIngestionJob:
        return KnowledgeIngestionJob(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            subject=row["subject"],
            original_filename=row["original_filename"],
            storage_path=row["storage_path"],
            rights_note=row["rights_note"],
            skill_ids=json.loads(row["skill_ids_json"]),
            size_bytes=row["size_bytes"],
            status=row["status"],
            document_id=row["document_id"],
            chunks=row["chunks"],
            embedding_status=row["embedding_status"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _create_knowledge_job_sync(
        self, job: KnowledgeIngestionJob
    ) -> KnowledgeIngestionJob:
        with self._sqlite_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO knowledge_ingestion_jobs (
                    id, user_id, title, subject, original_filename, storage_path,
                    rights_note, skill_ids_json, size_bytes, status, document_id,
                    chunks, embedding_status, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.user_id,
                    job.title,
                    job.subject.value,
                    job.original_filename,
                    job.storage_path,
                    job.rights_note,
                    json.dumps(job.skill_ids, ensure_ascii=False),
                    job.size_bytes,
                    job.status,
                    job.document_id,
                    job.chunks,
                    job.embedding_status,
                    job.error,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )
        return job

    async def create_knowledge_job(
        self, job: KnowledgeIngestionJob
    ) -> KnowledgeIngestionJob:
        return await asyncio.to_thread(self._create_knowledge_job_sync, job)

    def _get_knowledge_job_sync(
        self, job_id: str, user_id: str | None
    ) -> KnowledgeIngestionJob | None:
        query = "SELECT * FROM knowledge_ingestion_jobs WHERE id = ?"
        params: list[Any] = [job_id]
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        with self._sqlite_lock, self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return self._knowledge_job_from_row(row) if row else None

    async def get_knowledge_job(
        self, job_id: str, user_id: str | None = None
    ) -> KnowledgeIngestionJob | None:
        return await asyncio.to_thread(self._get_knowledge_job_sync, job_id, user_id)

    def _list_knowledge_jobs_sync(
        self, user_id: str, subject: Subject | None, limit: int
    ) -> list[KnowledgeIngestionJob]:
        query = "SELECT * FROM knowledge_ingestion_jobs WHERE user_id = ?"
        params: list[Any] = [user_id]
        if subject is not None:
            query += " AND subject = ?"
            params.append(subject.value)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._sqlite_lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._knowledge_job_from_row(row) for row in rows]

    async def list_knowledge_jobs(
        self, user_id: str, subject: Subject | None = None, limit: int = 20
    ) -> list[KnowledgeIngestionJob]:
        return await asyncio.to_thread(
            self._list_knowledge_jobs_sync, user_id, subject, limit
        )

    def _claim_knowledge_job_sync(
        self, job_id: str
    ) -> KnowledgeIngestionJob | None:
        now = datetime.now().isoformat()
        with self._sqlite_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE knowledge_ingestion_jobs
                SET status = 'processing', updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now, job_id),
            )
            if cursor.rowcount != 1:
                return None
            row = connection.execute(
                "SELECT * FROM knowledge_ingestion_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._knowledge_job_from_row(row)

    async def claim_knowledge_job(
        self, job_id: str
    ) -> KnowledgeIngestionJob | None:
        return await asyncio.to_thread(self._claim_knowledge_job_sync, job_id)

    def _update_knowledge_job_sync(
        self, job_id: str, updates: dict[str, Any]
    ) -> KnowledgeIngestionJob | None:
        allowed = {
            "status",
            "document_id",
            "chunks",
            "embedding_status",
            "error",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if not values:
            return self._get_knowledge_job_sync(job_id, None)
        values["updated_at"] = datetime.now().isoformat()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._sqlite_lock, self._connect() as connection:
            connection.execute(
                f"UPDATE knowledge_ingestion_jobs SET {assignments} WHERE id = ?",
                [*values.values(), job_id],
            )
            row = connection.execute(
                "SELECT * FROM knowledge_ingestion_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._knowledge_job_from_row(row) if row else None

    async def update_knowledge_job(
        self, job_id: str, updates: dict[str, Any]
    ) -> KnowledgeIngestionJob | None:
        return await asyncio.to_thread(self._update_knowledge_job_sync, job_id, updates)

    def _retrieve_context_sync(
        self,
        user_id: str,
        subject: Subject,
        query: str,
        skill_ids: list[str],
        limit: int,
        query_embedding: list[float] | None,
        document_ids: list[str] | None,
    ) -> list[Citation]:
        with self._sqlite_lock, self._connect() as connection:
            query_sql = """
                SELECT title, content, source_url, skill_ids_json, embedding_json
                FROM knowledge_chunks
                WHERE user_id = ? AND subject = ? AND review_status = 'approved'
            """
            params: list[Any] = [user_id, subject.value]
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                query_sql += f" AND document_id IN ({placeholders})"
                params.extend(document_ids)
            rows = connection.execute(query_sql, params).fetchall()
        chunks = [
            {
                "title": row["title"],
                "content": row["content"],
                "source_url": row["source_url"],
                "skill_ids": json.loads(row["skill_ids_json"]),
                "embedding": json.loads(row["embedding_json"]) if row["embedding_json"] else None,
            }
            for row in rows
        ]
        ranked = _rank_content_chunks(chunks, query, skill_ids, query_embedding)
        return [
            Citation(title=item["title"], url=item["source_url"], excerpt=item["content"][:500])
            for item in ranked[:limit]
        ]

    async def retrieve_context(
        self,
        subject: Subject,
        query: str,
        skill_ids: list[str],
        limit: int = 5,
        query_embedding: list[float] | None = None,
        user_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[Citation]:
        if not user_id:
            return []
        return await asyncio.to_thread(
            self._retrieve_context_sync,
            user_id,
            subject,
            query,
            skill_ids,
            limit,
            query_embedding,
            document_ids,
        )

    async def delete_user_data(self, user_id: str) -> None:
        await super().delete_user_data(user_id)

        def delete_messages() -> None:
            with self._sqlite_lock, self._connect() as connection:
                connection.execute(
                    "DELETE FROM chat_messages WHERE user_id = ?", (user_id,)
                )
                connection.execute(
                    "DELETE FROM tutoring_sessions WHERE user_id = ?", (user_id,)
                )
                connection.execute(
                    "DELETE FROM knowledge_chunks WHERE user_id = ?", (user_id,)
                )
                connection.execute(
                    "DELETE FROM knowledge_documents WHERE user_id = ?", (user_id,)
                )
                connection.execute(
                    "DELETE FROM knowledge_ingestion_jobs WHERE user_id = ?", (user_id,)
                )

        await asyncio.to_thread(delete_messages)


class PostgresChatRepository(LocalRepository):
    """Use PostgreSQL for durable chat data while retaining local non-chat fallbacks."""

    def __init__(self, database_url: str, local_database_path: str):
        super().__init__(local_database_path)
        self.database_url = database_url
        self._pg_pool: Any | None = None
        self._pg_pool_lock = asyncio.Lock()

    async def _get_pg_pool(self):
        if self._pg_pool is not None:
            return self._pg_pool
        async with self._pg_pool_lock:
            if self._pg_pool is not None:
                return self._pg_pool
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            pool = AsyncConnectionPool(
                conninfo=self.database_url,
                min_size=1,
                max_size=8,
                open=False,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
            await pool.open(wait=True)
            async with pool.connection() as connection:
                await connection.execute("CREATE SCHEMA IF NOT EXISTS visual_agent")
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS visual_agent.tutoring_sessions (
                        id uuid PRIMARY KEY,
                        user_id text NOT NULL,
                        subject text NOT NULL CHECK (
                            subject IN ('math', 'physics', 'chemistry', 'biology')
                        ),
                        learning_mode text NOT NULL DEFAULT 'guided_problem' CHECK (
                            learning_mode IN ('concept', 'guided_problem')
                        ),
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                await connection.execute(
                    """
                    ALTER TABLE visual_agent.tutoring_sessions
                    ADD COLUMN IF NOT EXISTS learning_mode text NOT NULL
                    DEFAULT 'guided_problem'
                    """
                )
                await connection.execute(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint
                            WHERE conname = 'tutoring_sessions_learning_mode_check'
                              AND conrelid = 'visual_agent.tutoring_sessions'::regclass
                        ) THEN
                            ALTER TABLE visual_agent.tutoring_sessions
                            ADD CONSTRAINT tutoring_sessions_learning_mode_check
                            CHECK (learning_mode IN ('concept', 'guided_problem'));
                        END IF;
                    END
                    $$
                    """
                )
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS visual_agent.chat_messages (
                        id uuid PRIMARY KEY,
                        user_id text NOT NULL,
                        session_id uuid NOT NULL REFERENCES
                            visual_agent.tutoring_sessions(id) ON DELETE CASCADE,
                        request_id text NOT NULL,
                        role text NOT NULL CHECK (role IN ('user', 'assistant')),
                        content text NOT NULL CHECK (
                            char_length(content) BETWEEN 1 AND 50000
                        ),
                        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        UNIQUE (user_id, session_id, request_id, role)
                    )
                    """
                )
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS tutoring_sessions_user_subject_updated_idx
                    ON visual_agent.tutoring_sessions(user_id, subject, updated_at DESC)
                    """
                )
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chat_messages_user_session_created_idx
                    ON visual_agent.chat_messages(user_id, session_id, created_at)
                    """
                )
            self._pg_pool = pool
            return pool

    async def healthcheck(self) -> bool:
        pool = await self._get_pg_pool()
        async with pool.connection() as connection:
            await connection.execute("SELECT 1")
        return True

    async def create_chat_session(
        self,
        user_id: str,
        subject: Subject,
        learning_mode: LearningMode = LearningMode.GUIDED_PROBLEM,
    ) -> ChatSessionRecord:
        session = ChatSessionRecord(
            user_id=user_id, subject=subject, learning_mode=learning_mode
        )
        pool = await self._get_pg_pool()
        async with pool.connection() as connection:
            row = await (
                await connection.execute(
                    """
                    INSERT INTO visual_agent.tutoring_sessions (
                        id, user_id, subject, learning_mode, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        session.id,
                        session.user_id,
                        session.subject.value,
                        session.learning_mode.value,
                        session.created_at,
                        session.updated_at,
                    ),
                )
            ).fetchone()
        return ChatSessionRecord.model_validate(row)

    async def get_chat_session(
        self, user_id: str, session_id: str
    ) -> ChatSessionRecord | None:
        pool = await self._get_pg_pool()
        async with pool.connection() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT * FROM visual_agent.tutoring_sessions
                    WHERE id = %s AND user_id = %s
                    """,
                    (session_id, user_id),
                )
            ).fetchone()
        return ChatSessionRecord.model_validate(row) if row else None

    async def update_chat_session_mode(
        self, user_id: str, session_id: str, learning_mode: LearningMode
    ) -> ChatSessionRecord | None:
        pool = await self._get_pg_pool()
        async with pool.connection() as connection:
            row = await (
                await connection.execute(
                    """
                    UPDATE visual_agent.tutoring_sessions
                    SET learning_mode = %s, updated_at = now()
                    WHERE id = %s AND user_id = %s
                    RETURNING *
                    """,
                    (learning_mode.value, session_id, user_id),
                )
            ).fetchone()
        return ChatSessionRecord.model_validate(row) if row else None

    async def _ensure_pg_session(
        self, connection, message: ChatMessage
    ) -> None:
        row = await (
            await connection.execute(
                """
                SELECT user_id, subject
                FROM visual_agent.tutoring_sessions
                WHERE id = %s
                FOR UPDATE
                """,
                (message.session_id,),
            )
        ).fetchone()
        subject_value = message.metadata.get("subject")
        normalized_subject = (
            subject_value
            if subject_value in {"math", "physics", "chemistry", "biology"}
            else None
        )
        try:
            learning_mode = LearningMode(
                message.metadata.get("learning_mode")
            ).value
        except (TypeError, ValueError):
            learning_mode = LearningMode.GUIDED_PROBLEM.value
        if row:
            if row["user_id"] != message.user_id:
                raise PermissionError("Chat session belongs to another user")
            if normalized_subject and row["subject"] != normalized_subject:
                raise ValueError("Chat session belongs to another subject")
            await connection.execute(
                """
                UPDATE visual_agent.tutoring_sessions
                SET updated_at = now()
                WHERE id = %s
                """,
                (message.session_id,),
            )
            return
        await connection.execute(
            """
            INSERT INTO visual_agent.tutoring_sessions (
                id, user_id, subject, learning_mode, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, now())
            """,
            (
                message.session_id,
                message.user_id,
                normalized_subject or Subject.MATH.value,
                learning_mode,
                message.created_at,
            ),
        )

    async def save_chat_message(self, message: ChatMessage) -> ChatMessage:
        from psycopg.types.json import Jsonb

        pool = await self._get_pg_pool()
        message_id = message.id or str(uuid.uuid4())
        async with pool.connection() as connection:
            async with connection.transaction():
                await self._ensure_pg_session(connection, message)
                row = await (
                    await connection.execute(
                        """
                        INSERT INTO visual_agent.chat_messages (
                            id, user_id, session_id, request_id, role, content,
                            metadata, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, session_id, request_id, role)
                        DO UPDATE SET
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata
                        RETURNING *
                        """,
                        (
                            message_id,
                            message.user_id,
                            message.session_id,
                            message.request_id,
                            message.role,
                            message.content,
                            Jsonb(message.metadata),
                            message.created_at,
                        ),
                    )
                ).fetchone()
        return ChatMessage.model_validate(row)

    async def get_chat_history(
        self, user_id: str, session_id: str, limit: int = 100
    ) -> list[ChatMessage]:
        pool = await self._get_pg_pool()
        async with pool.connection() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT * FROM visual_agent.chat_messages
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (user_id, session_id, limit),
                )
            ).fetchall()
        return [ChatMessage.model_validate(row) for row in reversed(rows)]

    async def list_chat_sessions(
        self,
        user_id: str,
        limit: int = 50,
        subject: Subject | None = None,
    ) -> list[ChatSessionSummary]:
        pool = await self._get_pg_pool()
        params: list[Any] = [user_id]
        subject_clause = ""
        if subject is not None:
            subject_clause = "AND subject = %s"
            params.append(subject.value)
        params.append(limit)
        async with pool.connection() as connection:
            session_rows = await (
                await connection.execute(
                    f"""
                    SELECT * FROM visual_agent.tutoring_sessions
                    WHERE user_id = %s {subject_clause}
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    params,
                )
            ).fetchall()
            session_ids = [row["id"] for row in session_rows]
            if session_ids:
                message_rows = await (
                    await connection.execute(
                        """
                        SELECT * FROM visual_agent.chat_messages
                        WHERE user_id = %s AND session_id = ANY(%s)
                        ORDER BY created_at ASC
                        """,
                        (user_id, session_ids),
                    )
                ).fetchall()
            else:
                message_rows = []
        sessions = [ChatSessionRecord.model_validate(row) for row in session_rows]
        messages = [ChatMessage.model_validate(row) for row in message_rows]
        return _summarize_chat_sessions(messages, limit, subject, sessions)

    async def delete_user_data(self, user_id: str) -> None:
        pool = await self._get_pg_pool()
        async with pool.connection() as connection:
            await connection.execute(
                "DELETE FROM visual_agent.tutoring_sessions WHERE user_id = %s",
                (user_id,),
            )
        await super().delete_user_data(user_id)


class SupabaseRepository:
    """PostgREST-backed repository. User reads carry their JWT so RLS is active."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = f"{settings.supabase_url}/rest/v1"

    def _headers(self, token: str | None = None, *, service: bool = False) -> dict[str, str]:
        key = (
            self.settings.supabase_service_role_key
            if service
            else self.settings.supabase_publishable_key
        )
        bearer = token or key
        return {
            "apikey": key,
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        service: bool = False,
        json: Any = None,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        merged = self._headers(token, service=service)
        if headers:
            merged.update(headers)
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.request(
                method,
                f"{self.base_url}/{path}",
                headers=merged,
                json=json,
                params=params,
            )
        response.raise_for_status()
        return response.json() if response.content else None

    async def save_asset(self, record: AssetRecord) -> None:
        await self._request(
            "POST", "assets", service=True, json=record.model_dump(mode="json")
        )

    async def get_asset(self, asset_id: str, user_id: str) -> AssetRecord | None:
        rows = await self._request(
            "GET",
            "assets",
            service=True,
            params={"id": f"eq.{asset_id}", "user_id": f"eq.{user_id}", "limit": "1"},
        )
        return AssetRecord.model_validate(rows[0]) if rows else None

    async def list_assets(self, user_id: str) -> list[AssetRecord]:
        rows = await self._request(
            "GET", "assets", service=True, params={"user_id": f"eq.{user_id}"}
        )
        return [AssetRecord.model_validate(row) for row in rows]

    async def list_expired_assets(self, before: datetime) -> list[AssetRecord]:
        rows = await self._request(
            "GET",
            "assets",
            service=True,
            params={"created_at": f"lt.{before.isoformat()}"},
        )
        return [AssetRecord.model_validate(row) for row in rows]

    async def delete_asset_records(self, asset_ids: list[str]) -> None:
        if asset_ids:
            await self._request(
                "DELETE",
                "assets",
                service=True,
                params={"id": f"in.({','.join(asset_ids)})"},
            )

    async def get_mastery(self, user_id: str) -> list[MasteryRecord]:
        rows = await self._request(
            "GET", "mastery_records", service=True, params={"user_id": f"eq.{user_id}"}
        )
        return [MasteryRecord.model_validate(row) for row in rows]

    async def upsert_mastery(self, record: MasteryRecord) -> MasteryRecord:
        rows = await self._request(
            "POST",
            "mastery_records",
            service=True,
            json=record.model_dump(mode="json"),
            params={"on_conflict": "user_id,skill_id"},
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )
        return MasteryRecord.model_validate(rows[0])

    async def create_chat_session(
        self,
        user_id: str,
        subject: Subject,
        learning_mode: LearningMode = LearningMode.GUIDED_PROBLEM,
    ) -> ChatSessionRecord:
        session = ChatSessionRecord(
            user_id=user_id, subject=subject, learning_mode=learning_mode
        )
        rows = await self._request(
            "POST",
            "tutoring_sessions",
            service=True,
            json=session.model_dump(mode="json"),
        )
        return ChatSessionRecord.model_validate(rows[0])

    async def get_chat_session(
        self, user_id: str, session_id: str
    ) -> ChatSessionRecord | None:
        rows = await self._request(
            "GET",
            "tutoring_sessions",
            service=True,
            params={
                "id": f"eq.{session_id}",
                "user_id": f"eq.{user_id}",
                "limit": "1",
            },
        )
        return ChatSessionRecord.model_validate(rows[0]) if rows else None

    async def update_chat_session_mode(
        self, user_id: str, session_id: str, learning_mode: LearningMode
    ) -> ChatSessionRecord | None:
        rows = await self._request(
            "PATCH",
            "tutoring_sessions",
            service=True,
            params={"id": f"eq.{session_id}", "user_id": f"eq.{user_id}"},
            json={
                "learning_mode": learning_mode.value,
                "updated_at": datetime.now().isoformat(),
            },
        )
        return ChatSessionRecord.model_validate(rows[0]) if rows else None

    async def save_chat_message(self, message: ChatMessage) -> ChatMessage:
        await self._ensure_session(
            message.user_id,
            message.session_id,
            message.metadata.get("subject"),
            message.metadata.get("learning_mode"),
        )
        rows = await self._request(
            "POST",
            "chat_messages",
            service=True,
            json=message.model_dump(mode="json", exclude_none=True),
            params={"on_conflict": "user_id,session_id,request_id,role"},
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )
        return ChatMessage.model_validate(rows[0])

    async def get_chat_history(
        self, user_id: str, session_id: str, limit: int = 100
    ) -> list[ChatMessage]:
        rows = await self._request(
            "GET",
            "chat_messages",
            service=True,
            params={
                "or": f"(user_id.eq.{user_id},user_id.is.null)",
                "session_id": f"eq.{session_id}",
                "order": "created_at.desc",
                "limit": str(limit),
            },
        )
        return [ChatMessage.model_validate(row) for row in reversed(rows)]

    async def list_chat_sessions(
        self,
        user_id: str,
        limit: int = 50,
        subject: Subject | None = None,
    ) -> list[ChatSessionSummary]:
        session_params = {
            "user_id": f"eq.{user_id}",
            "order": "updated_at.desc",
            "limit": str(limit),
        }
        if subject is not None:
            session_params["subject"] = f"eq.{subject.value}"
        session_rows = await self._request(
            "GET",
            "tutoring_sessions",
            service=True,
            params=session_params,
        )
        session_ids = [row["id"] for row in session_rows]
        rows = []
        if session_ids:
            rows = await self._request(
                "GET",
                "chat_messages",
                service=True,
                params={
                    "user_id": f"eq.{user_id}",
                    "session_id": f"in.({','.join(session_ids)})",
                    "order": "created_at.asc",
                    "limit": "5000",
                },
            )
        return _summarize_chat_sessions(
            [ChatMessage.model_validate(row) for row in rows],
            limit,
            subject,
            [ChatSessionRecord.model_validate(row) for row in session_rows],
        )

    async def retrieve_context(
        self,
        subject: Subject,
        query: str,
        skill_ids: list[str],
        limit: int = 5,
        query_embedding: list[float] | None = None,
        user_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[Citation]:
        if not user_id:
            return []
        if (
            not document_ids
            and query_embedding
            and len(query_embedding) == self.settings.embedding_dimensions
        ):
            rows = await self._request(
                "POST",
                "rpc/match_user_content_chunks",
                service=True,
                json={
                    "query_embedding": query_embedding,
                    "match_user_id": user_id,
                    "match_subject": subject.value,
                    "match_count": limit,
                },
            )
            return [
                Citation(
                    title=row.get("title") or "Học liệu CTGDPT 2018",
                    url=row.get("source_url") or "",
                    excerpt=row.get("content", "")[:500],
                )
                for row in rows
            ]
        # Approved-only lexical fallback while the embedding endpoint is unavailable.
        params = {
            "subject": f"eq.{subject.value}",
            "or": f"(user_id.eq.{user_id},user_id.is.null)",
            "review_status": "eq.approved",
            "select": "title,content,source_url,skill_ids",
            "limit": "100",
        }
        if document_ids:
            params.pop("or")
            params["user_id"] = f"eq.{user_id}"
            params["document_id"] = f"in.({','.join(document_ids)})"
        rows = await self._request(
            "GET",
            "content_chunks",
            service=True,
            params=params,
        )
        rows = _rank_content_chunks(rows, query, skill_ids, None)[:limit]
        return [
            Citation(
                title=row.get("title") or "Học liệu CTGDPT 2018",
                url=row.get("source_url") or "",
                excerpt=row.get("content", "")[:500],
            )
            for row in rows
        ]

    async def save_approved_content(
        self, document: dict[str, Any], chunks: list[dict[str, Any]]
    ) -> str:
        rows = await self._request(
            "POST", "content_documents", service=True, json=document
        )
        document_id = rows[0]["id"]
        await self._request(
            "POST",
            "content_chunks",
            service=True,
            json=[{**item, "document_id": document_id} for item in chunks],
            headers={"Prefer": "return=minimal"},
        )
        return document_id

    async def list_knowledge_documents(
        self, user_id: str, subject: Subject | None = None, limit: int = 50
    ) -> list[KnowledgeDocumentSummary]:
        params = {
            "user_id": f"eq.{user_id}",
            "order": "created_at.desc",
            "limit": str(limit),
        }
        if subject is not None:
            params["subject"] = f"eq.{subject.value}"
        rows = await self._request(
            "GET", "content_documents", service=True, params=params
        )
        document_ids = [row["id"] for row in rows]
        chunk_counts: dict[str, int] = defaultdict(int)
        if document_ids:
            chunk_rows = await self._request(
                "GET",
                "content_chunks",
                service=True,
                params={
                    "select": "document_id",
                    "document_id": f"in.({','.join(document_ids)})",
                },
            )
            for chunk in chunk_rows:
                chunk_counts[chunk["document_id"]] += 1
        return [
            KnowledgeDocumentSummary(
                **row,
                chunks=chunk_counts[row["id"]],
            )
            for row in rows
        ]

    async def create_knowledge_job(
        self, job: KnowledgeIngestionJob
    ) -> KnowledgeIngestionJob:
        rows = await self._request(
            "POST",
            "knowledge_ingestion_jobs",
            service=True,
            json=job.model_dump(mode="json"),
        )
        return KnowledgeIngestionJob.model_validate(rows[0])

    async def get_knowledge_job(
        self, job_id: str, user_id: str | None = None
    ) -> KnowledgeIngestionJob | None:
        params = {"id": f"eq.{job_id}", "limit": "1"}
        if user_id is not None:
            params["user_id"] = f"eq.{user_id}"
        rows = await self._request(
            "GET", "knowledge_ingestion_jobs", service=True, params=params
        )
        return KnowledgeIngestionJob.model_validate(rows[0]) if rows else None

    async def list_knowledge_jobs(
        self, user_id: str, subject: Subject | None = None, limit: int = 20
    ) -> list[KnowledgeIngestionJob]:
        params = {
            "user_id": f"eq.{user_id}",
            "order": "created_at.desc",
            "limit": str(limit),
        }
        if subject is not None:
            params["subject"] = f"eq.{subject.value}"
        rows = await self._request(
            "GET", "knowledge_ingestion_jobs", service=True, params=params
        )
        return [KnowledgeIngestionJob.model_validate(row) for row in rows]

    async def claim_knowledge_job(
        self, job_id: str
    ) -> KnowledgeIngestionJob | None:
        rows = await self._request(
            "PATCH",
            "knowledge_ingestion_jobs",
            service=True,
            params={"id": f"eq.{job_id}", "status": "eq.pending"},
            json={"status": "processing", "updated_at": datetime.now().isoformat()},
        )
        return KnowledgeIngestionJob.model_validate(rows[0]) if rows else None

    async def update_knowledge_job(
        self, job_id: str, updates: dict[str, Any]
    ) -> KnowledgeIngestionJob | None:
        allowed = {
            "status",
            "document_id",
            "chunks",
            "embedding_status",
            "error",
        }
        payload = {key: value for key, value in updates.items() if key in allowed}
        payload["updated_at"] = datetime.now().isoformat()
        rows = await self._request(
            "PATCH",
            "knowledge_ingestion_jobs",
            service=True,
            params={"id": f"eq.{job_id}"},
            json=payload,
        )
        return KnowledgeIngestionJob.model_validate(rows[0]) if rows else None

    async def _ensure_session(
        self,
        user_id: str,
        session_id: str,
        subject: str | None,
        learning_mode: str | None = None,
    ) -> None:
        normalized_subject = (
            subject
            if subject in {"math", "physics", "chemistry", "biology"}
            else None
        )
        try:
            normalized_mode = LearningMode(learning_mode).value
        except (TypeError, ValueError):
            normalized_mode = LearningMode.GUIDED_PROBLEM.value
        existing = await self._request(
            "GET",
            "tutoring_sessions",
            service=True,
            params={"id": f"eq.{session_id}", "limit": "1"},
        )
        if existing:
            session = existing[0]
            if session["user_id"] != user_id:
                raise PermissionError("Chat session belongs to another user")
            if (
                normalized_subject
                and session.get("subject")
                and session["subject"] != normalized_subject
            ):
                raise ValueError("Chat session belongs to another subject")
            updates = {"updated_at": datetime.now().isoformat()}
            if normalized_subject and not session.get("subject"):
                updates["subject"] = normalized_subject
            await self._request(
                "PATCH",
                "tutoring_sessions",
                service=True,
                params={"id": f"eq.{session_id}", "user_id": f"eq.{user_id}"},
                json=updates,
                headers={"Prefer": "return=minimal"},
            )
            return
        await self._request(
            "POST",
            "tutoring_sessions",
            service=True,
            json={
                "id": session_id,
                "user_id": user_id,
                "subject": normalized_subject or Subject.MATH.value,
                "learning_mode": normalized_mode,
                "updated_at": datetime.now().isoformat(),
            },
            headers={"Prefer": "return=minimal"},
        )

    async def save_problem(
        self, user_id: str, session_id: str, problem: dict[str, Any], asset_id: str | None
    ) -> str:
        await self._ensure_session(user_id, session_id, problem["subject"])
        rows = await self._request(
            "POST",
            "problems",
            service=True,
            json={
                "user_id": user_id,
                "session_id": session_id,
                "asset_id": asset_id,
                "subject": problem["subject"],
                "problem_markdown": problem.get("problem_markdown", ""),
                "extraction": problem,
                "confidence": problem.get("confidence", 0),
            },
        )
        return rows[0]["id"]

    async def save_hint_event(
        self, user_id: str, session_id: str, problem_id: str, response: dict[str, Any]
    ) -> None:
        await self._request(
            "POST", "hint_events", service=True,
            json={
                "user_id": user_id, "session_id": session_id, "problem_id": problem_id,
                "hint_level": response.get("hint_level", 0), "content": response.get("message", ""),
            },
            headers={"Prefer": "return=minimal"},
        )

    async def save_attempt(
        self, user_id: str, session_id: str, problem_id: str, answer: str,
        answer_correct: bool, evidence: float | None, misconception: str | None,
    ) -> None:
        await self._request(
            "POST", "attempts", service=True,
            json={
                "user_id": user_id, "session_id": session_id, "problem_id": problem_id,
                "answer": answer, "answer_correct": answer_correct,
                "evidence": evidence, "misconception": misconception,
            },
            headers={"Prefer": "return=minimal"},
        )

    async def delete_user_data(self, user_id: str) -> None:
        for table in ("knowledge_ingestion_jobs", "content_documents", "chat_messages", "hint_events", "attempts", "problems", "tutoring_sessions", "mastery_records", "assets", "profiles"):
            await self._request(
                "DELETE", table, service=True, params={"user_id": f"eq.{user_id}"}
            )


def _summarize_chat_sessions(
    messages: list[ChatMessage],
    limit: int,
    subject_filter: Subject | None = None,
    sessions: list[ChatSessionRecord] | None = None,
) -> list[ChatSessionSummary]:
    grouped: dict[str, list[ChatMessage]] = defaultdict(list)
    for message in messages:
        grouped[message.session_id].append(message)
    session_by_id = {session.id: session for session in sessions or []}
    summaries: list[ChatSessionSummary] = []
    for session_id in set(grouped) | set(session_by_id):
        items = grouped.get(session_id, [])
        session = session_by_id.get(session_id)
        if not items and session is not None:
            summary = ChatSessionSummary(
                session_id=session.id,
                title="Phiên mới",
                last_message="Chưa có tin nhắn",
                message_count=0,
                subject=session.subject,
                learning_mode=session.learning_mode,
                updated_at=session.updated_at,
            )
            if subject_filter is None or summary.subject == subject_filter:
                summaries.append(summary)
            continue
        items.sort(key=lambda item: item.created_at)
        first_user = next((item for item in items if item.role == "user"), items[0])
        subject_value = next(
            (
                item.metadata.get("subject")
                for item in reversed(items)
                if item.metadata.get("subject")
            ),
            None,
        )
        try:
            subject = (
                Subject(subject_value)
                if subject_value
                else session.subject if session is not None else None
            )
        except ValueError:
            subject = None
        title = " ".join(first_user.content.split())
        last_message = " ".join(items[-1].content.split())
        summary = ChatSessionSummary(
            session_id=session_id,
            title=(title[:77] + "...") if len(title) > 80 else title,
            last_message=(last_message[:117] + "...")
            if len(last_message) > 120
            else last_message,
            message_count=len(items),
            subject=subject,
            learning_mode=(
                session.learning_mode
                if session is not None
                else LearningMode.GUIDED_PROBLEM
            ),
            updated_at=items[-1].created_at,
        )
        if subject_filter is None or summary.subject == subject_filter:
            summaries.append(summary)
    return sorted(summaries, key=lambda item: item.updated_at, reverse=True)[:limit]


def _rank_content_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    skill_ids: list[str],
    query_embedding: list[float] | None,
) -> list[dict[str, Any]]:
    def search_terms(value: str) -> set[str]:
        normalized = unicodedata.normalize("NFKD", value.casefold())
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        normalized = normalized.replace("đ", "d")
        return set(re.findall(r"\w+", normalized, flags=re.UNICODE))

    terms = search_terms(query)
    target_skills = set(skill_ids)

    def score(item: dict[str, Any]) -> float:
        content_terms = search_terms(str(item.get("content", "")))
        lexical = len(terms & content_terms) / max(len(terms), 1)
        skill_match = 1.0 if target_skills & set(item.get("skill_ids", [])) else 0.0
        semantic = 0.0
        vector = item.get("embedding")
        if query_embedding and vector and len(query_embedding) == len(vector):
            dot = sum(left * right for left, right in zip(query_embedding, vector))
            left_norm = math.sqrt(sum(value * value for value in query_embedding))
            right_norm = math.sqrt(sum(value * value for value in vector))
            if left_norm and right_norm:
                semantic = dot / (left_norm * right_norm)
        return semantic * 0.7 + lexical * 0.25 + skill_match * 0.05

    scored = [(score(item), item) for item in chunks]
    return [item for value, item in sorted(scored, key=lambda pair: pair[0], reverse=True) if value > 0]


_local_repository = LocalRepository(get_settings().chat_database_path)


def get_repository() -> TutorRepository:
    settings = get_settings()
    if settings.supabase_service_enabled:
        return SupabaseRepository(settings)
    if settings.database_url:
        return PostgresChatRepository(
            settings.database_url, settings.chat_database_path
        )
    return _local_repository
