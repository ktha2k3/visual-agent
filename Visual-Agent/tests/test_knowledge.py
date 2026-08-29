import io
import zipfile

import pytest
from fastapi import UploadFile

from education.models import KnowledgeIngestionJob, Subject
from services.content import split_content
from services.knowledge import KnowledgeFileError, read_knowledge_upload
from services.repository import LocalRepository, MemoryRepository


def _docx_bytes(text: str) -> bytes:
    stream = io.BytesIO()
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("[Content_Types].xml", "<Types />")
    return stream.getvalue()


@pytest.mark.asyncio
async def test_knowledge_upload_validates_and_extracts_real_document_content():
    upload = UploadFile(
        filename="de-cuong.docx",
        file=io.BytesIO(_docx_bytes("Dao ham va ung dung trong khao sat ham so")),
    )
    text, mime_type, filename, size = await read_knowledge_upload(upload, 10 * 1024 * 1024)
    assert text == "Dao ham va ung dung trong khao sat ham so"
    assert mime_type.endswith("wordprocessingml.document")
    assert filename == "de-cuong.docx"
    assert size > 0

    fake = UploadFile(filename="de-cuong.pdf", file=io.BytesIO(b"not a pdf"))
    with pytest.raises(KnowledgeFileError) as exc:
        await read_knowledge_upload(fake, 10 * 1024 * 1024)
    assert exc.value.status_code == 415


def test_knowledge_chunker_caps_very_long_paragraphs():
    chunks = split_content("word " * 1000, max_chars=250)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 250 for chunk in chunks)


@pytest.mark.asyncio
async def test_local_knowledge_is_persistent_and_owner_scoped(tmp_path):
    database = tmp_path / "knowledge.sqlite3"
    first = LocalRepository(str(database))
    document = {
        "user_id": "user-a",
        "title": "De cuong dao ham",
        "subject": "math",
        "source_url": "knowledge:private",
        "rights_note": "personal",
        "embedding_status": "lexical_fallback",
        "review_status": "approved",
    }
    chunks = [
        {
            "user_id": "user-a",
            "title": "De cuong dao ham",
            "subject": "math",
            "skill_ids": ["math.derivatives"],
            "content": "Đạo hàm dùng để khảo sát chiều biến thiên của hàm số.",
            "source_url": "knowledge:private",
            "embedding": None,
            "review_status": "approved",
        }
    ]
    await first.save_approved_content(document, chunks)

    restarted = LocalRepository(str(database))
    own = await restarted.retrieve_context(
        Subject.MATH,
        "khao sat dao ham",
        ["math.derivatives"],
        user_id="user-a",
    )
    other = await restarted.retrieve_context(
        Subject.MATH,
        "khao sat dao ham",
        ["math.derivatives"],
        user_id="user-b",
    )
    documents = await restarted.list_knowledge_documents("user-a", Subject.MATH)
    assert [item.title for item in own] == ["De cuong dao ham"]
    assert other == []
    assert len(documents) == 1
    assert documents[0].title == "De cuong dao ham"
    assert documents[0].chunks == 1


@pytest.mark.asyncio
async def test_knowledge_job_is_persistent_and_claimed_once(tmp_path):
    database = tmp_path / "knowledge-jobs.sqlite3"
    first = LocalRepository(str(database))
    job = KnowledgeIngestionJob(
        user_id="user-a",
        title="De cuong dao ham",
        subject=Subject.MATH,
        original_filename="de-cuong.md",
        storage_path=str(tmp_path / "staged.md"),
        rights_note="personal",
        size_bytes=128,
    )
    await first.create_knowledge_job(job)

    restarted = LocalRepository(str(database))
    stored = await restarted.get_knowledge_job(job.id, "user-a")
    claimed = await restarted.claim_knowledge_job(job.id)
    duplicate = await restarted.claim_knowledge_job(job.id)

    assert stored is not None and stored.status == "pending"
    assert claimed is not None and claimed.status == "processing"
    assert duplicate is None


@pytest.mark.asyncio
async def test_memory_knowledge_does_not_leak_between_users():
    repository = MemoryRepository()
    await repository.save_approved_content(
        {"user_id": "user-a", "title": "Private", "subject": "biology"},
        [
            {
                "user_id": "user-a",
                "title": "Private",
                "subject": "biology",
                "skill_ids": [],
                "content": "Di truyen quan the",
                "source_url": "knowledge:private",
            }
        ],
    )
    assert await repository.retrieve_context(
        Subject.BIOLOGY, "di truyen", [], user_id="user-b"
    ) == []


@pytest.mark.asyncio
async def test_selected_document_scopes_private_rag_context():
    repository = MemoryRepository()
    selected_id = await repository.save_approved_content(
        {"user_id": "user-a", "title": "Selected", "subject": "math"},
        [{
            "user_id": "user-a",
            "title": "Selected",
            "subject": "math",
            "skill_ids": [],
            "content": "dao ham ham so bac ba",
            "source_url": "knowledge:selected",
        }],
    )
    await repository.save_approved_content(
        {"user_id": "user-a", "title": "Other", "subject": "math"},
        [{
            "user_id": "user-a",
            "title": "Other",
            "subject": "math",
            "skill_ids": [],
            "content": "dao ham ham so bac ba",
            "source_url": "knowledge:other",
        }],
    )
    matches = await repository.retrieve_context(
        Subject.MATH,
        "dao ham bac ba",
        [],
        user_id="user-a",
        document_ids=[selected_id],
    )
    assert [item.title for item in matches] == ["Selected"]
