import sqlite3

import pytest

from education.models import ChatMessage, LearningMode, Subject
from services.repository import LocalRepository, MemoryRepository


@pytest.mark.asyncio
async def test_empty_chat_session_is_persisted_and_listed(tmp_path):
    database = tmp_path / "chat.sqlite3"
    repository = LocalRepository(str(database))
    created = await repository.create_chat_session(
        "user-a", Subject.MATH, LearningMode.CONCEPT
    )

    restarted = LocalRepository(str(database))
    sessions = await restarted.list_chat_sessions(
        "user-a", subject=Subject.MATH
    )

    assert sessions[0].session_id == created.id
    assert sessions[0].message_count == 0
    assert sessions[0].title == "Phiên mới"
    assert sessions[0].learning_mode == LearningMode.CONCEPT

    updated = await restarted.update_chat_session_mode(
        "user-a", created.id, LearningMode.GUIDED_PROBLEM
    )
    persisted = await restarted.get_chat_session("user-a", created.id)
    assert updated is not None
    assert updated.learning_mode == LearningMode.GUIDED_PROBLEM
    assert persisted is not None
    assert persisted.learning_mode == LearningMode.GUIDED_PROBLEM


@pytest.mark.asyncio
async def test_local_chat_history_survives_repository_restart(tmp_path):
    database = tmp_path / "chat.sqlite3"
    first = LocalRepository(str(database))
    message = ChatMessage(
        user_id="user-a",
        session_id="session-a",
        request_id="request-a",
        role="user",
        content="Cách làm của em",
    )
    await first.save_chat_message(message)
    await first.save_chat_message(message.model_copy(update={"content": "Cách làm đã sửa"}))

    restarted = LocalRepository(str(database))
    history = await restarted.get_chat_history("user-a", "session-a")
    sessions = await restarted.list_chat_sessions("user-a")

    assert len(history) == 1
    assert history[0].content == "Cách làm đã sửa"
    assert len(sessions) == 1
    assert sessions[0].session_id == "session-a"
    assert sessions[0].title == "Cách làm đã sửa"
    assert await restarted.get_chat_history("user-b", "session-a") == []


@pytest.mark.asyncio
async def test_local_repository_migrates_legacy_sessions_to_default_mode(tmp_path):
    database = tmp_path / "legacy-chat.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE tutoring_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO tutoring_sessions VALUES (?, ?, ?, ?, ?)",
            ("session-a", "user-a", "math", "2026-01-01", "2026-01-01"),
        )

    repository = LocalRepository(str(database))
    session = await repository.get_chat_session("user-a", "session-a")

    assert session is not None
    assert session.learning_mode == LearningMode.GUIDED_PROBLEM


@pytest.mark.asyncio
async def test_chat_sessions_are_filtered_by_subject():
    repository = MemoryRepository()
    await repository.save_chat_message(
        ChatMessage(
            user_id="student",
            session_id="math-session",
            request_id="math-request",
            role="user",
            content="Đạo hàm là gì?",
            metadata={"subject": "math"},
        )
    )
    await repository.save_chat_message(
        ChatMessage(
            user_id="student",
            session_id="physics-session",
            request_id="physics-request",
            role="user",
            content="Định luật Newton",
            metadata={"subject": "physics"},
        )
    )

    math_sessions = await repository.list_chat_sessions(
        "student", subject=Subject.MATH
    )
    physics_sessions = await repository.list_chat_sessions(
        "student", subject=Subject.PHYSICS
    )

    assert [item.session_id for item in math_sessions] == ["math-session"]
    assert [item.session_id for item in physics_sessions] == ["physics-session"]
