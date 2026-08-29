from __future__ import annotations

from education.models import Subject
from services.repository import TutorRepository
from utils.config import get_settings
from utils.model_factory import get_embedding_model


def split_content(text: str, max_chars: int = 1200) -> list[str]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            remaining = paragraph
            while len(remaining) > max_chars:
                split_at = remaining.rfind(" ", 0, max_chars + 1)
                if split_at < max_chars // 2:
                    split_at = max_chars
                chunks.append(remaining[:split_at].strip())
                remaining = remaining[split_at:].strip()
            current = remaining
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


class ContentIngestionService:
    def __init__(self, repository: TutorRepository):
        self.repository = repository
        self.embeddings = get_embedding_model()
        self.embedding_dimensions = get_settings().embedding_dimensions

    async def ingest(
        self,
        *,
        user_id: str,
        title: str,
        subject: Subject,
        source_url: str,
        publisher: str | None,
        rights_note: str,
        content: str,
        skill_ids: list[str],
        original_filename: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
    ) -> tuple[str, int, str]:
        texts = split_content(content)
        if not texts:
            raise ValueError("Content is empty")
        embedding_status = "ready"
        try:
            vectors: list[list[float] | None] = await self.embeddings.aembed_documents(texts)
            if any(
                vector is None or len(vector) != self.embedding_dimensions
                for vector in vectors
            ):
                raise ValueError(
                    "EMBEDDING_MODEL must return "
                    f"{self.embedding_dimensions:,}-dimensional vectors"
                )
        except ValueError:
            raise
        except Exception:
            # Ollama Cloud currently has no embedding endpoint. Persisting the chunks
            # still enables owner-scoped lexical RAG until one is configured.
            vectors = [None] * len(texts)
            embedding_status = "lexical_fallback"
        document = {
            "user_id": user_id,
            "title": title,
            "subject": subject.value,
            "source_url": source_url,
            "publisher": publisher,
            "rights_note": rights_note,
            "original_filename": original_filename,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "embedding_status": embedding_status,
            "review_status": "approved",
        }
        chunks = [
            {
                "user_id": user_id,
                "title": title,
                "subject": subject.value,
                "skill_ids": skill_ids,
                "content": text,
                "source_url": source_url,
                "embedding": vector,
                "review_status": "approved",
            }
            for text, vector in zip(texts, vectors, strict=True)
        ]
        return (
            await self.repository.save_approved_content(document, chunks),
            len(chunks),
            embedding_status,
        )
