import pytest

from education.models import ChatMessage, ExamQuestion, ExamSource, ReviewPatch, ReviewStatus, Subject
from services.repository import LocalRepository, MemoryRepository


@pytest.mark.asyncio
async def test_exam_sync_storage_is_idempotent_and_requires_review():
    repository = MemoryRepository()
    source = await repository.save_exam_source(
        ExamSource(url="https://vqa.moet.gov.vn/x", publisher="MOET", exam_year=2026, subject=Subject.MATH)
    )
    artifact = {
        "source_id": source.id,
        "storage_path": "private/original.png",
        "original_url": "https://vqa.moet.gov.vn/original.png",
        "mime_type": "image/png",
        "checksum": "abc",
    }
    first_artifact_id = await repository.save_exam_artifact(artifact)
    second_artifact_id = await repository.save_exam_artifact(artifact)
    assert first_artifact_id == second_artifact_id
    assert len(repository.exam_artifacts) == 1
    first = ExamQuestion(
        source_id=source.id,
        exam_year=2026,
        subject=Subject.MATH,
        question_number="1",
        stem_markdown="Câu hỏi",
        confidence=0.9,
    )
    await repository.save_exam_questions([first])
    await repository.save_exam_questions([first.model_copy(update={"stem_markdown": "Câu hỏi đã OCR lại"})])
    assert len(repository.exam_questions) == 1
    pending = await repository.list_review_queue()
    assert pending[0].review_status == ReviewStatus.PENDING
    approved = await repository.patch_review(first.id, ReviewPatch(review_status=ReviewStatus.APPROVED))
    assert approved.review_status == ReviewStatus.APPROVED


@pytest.mark.asyncio
async def test_exam_trends_do_not_fabricate_templates_without_reviewed_questions():
    repository = MemoryRepository()
    assert await repository.get_exam_trends() == []


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
