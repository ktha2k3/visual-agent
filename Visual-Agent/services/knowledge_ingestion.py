from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from education.models import KnowledgeIngestionJob
from services.content import ContentIngestionService
from services.knowledge import extract_document_text
from services.repository import TutorRepository, get_repository
from utils.config import Settings, get_settings


class KnowledgeUploadStorage:
    """Private staging storage shared by the upload API and ingestion worker."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.local_dir = (
            Path(__file__).resolve().parent.parent / "assets" / "knowledge-uploads"
        )

    async def save(
        self, job_id: str, filename: str, data: bytes, mime_type: str
    ) -> str:
        extension = Path(filename).suffix.lower()
        object_path = f"knowledge/{job_id}{extension}"
        if self.settings.supabase_service_enabled:
            url = (
                f"{self.settings.supabase_url}/storage/v1/object/"
                f"{self.settings.supabase_storage_bucket}/{object_path}"
            )
            headers = {
                "apikey": self.settings.supabase_service_role_key,
                "Authorization": f"Bearer {self.settings.supabase_service_role_key}",
                "Content-Type": mime_type,
                "x-upsert": "false",
            }
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(url, headers=headers, content=data)
            response.raise_for_status()
            return object_path

        self.local_dir.mkdir(parents=True, exist_ok=True)
        destination = (self.local_dir / f"{job_id}{extension}").resolve()
        if self.local_dir.resolve() not in destination.parents:
            raise ValueError("Invalid knowledge upload path")
        await asyncio.to_thread(destination.write_bytes, data)
        return str(destination)

    async def read(self, storage_path: str) -> bytes:
        if self.settings.supabase_service_enabled:
            url = (
                f"{self.settings.supabase_url}/storage/v1/object/authenticated/"
                f"{self.settings.supabase_storage_bucket}/{storage_path}"
            )
            headers = {
                "apikey": self.settings.supabase_service_role_key,
                "Authorization": f"Bearer {self.settings.supabase_service_role_key}",
            }
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.content

        path = Path(storage_path).resolve()
        if self.local_dir.resolve() not in path.parents:
            raise ValueError("Invalid knowledge upload path")
        return await asyncio.to_thread(path.read_bytes)

    async def delete(self, storage_path: str) -> None:
        if self.settings.supabase_service_enabled:
            url = (
                f"{self.settings.supabase_url}/storage/v1/object/"
                f"{self.settings.supabase_storage_bucket}"
            )
            headers = {
                "apikey": self.settings.supabase_service_role_key,
                "Authorization": f"Bearer {self.settings.supabase_service_role_key}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.request(
                    "DELETE", url, headers=headers, json={"prefixes": [storage_path]}
                )
            response.raise_for_status()
            return

        path = Path(storage_path).resolve()
        if self.local_dir.resolve() in path.parents and path.is_file():
            await asyncio.to_thread(path.unlink)


class KnowledgeIngestionProcessor:
    def __init__(
        self,
        repository: TutorRepository,
        storage: KnowledgeUploadStorage | None = None,
    ):
        self.repository = repository
        self.storage = storage or KnowledgeUploadStorage()

    async def process(self, job_id: str) -> KnowledgeIngestionJob | None:
        job = await self.repository.claim_knowledge_job(job_id)
        if not job:
            return await self.repository.get_knowledge_job(job_id)

        try:
            data = await self.storage.read(job.storage_path)
            content, mime_type = await asyncio.to_thread(
                extract_document_text, data, job.original_filename
            )
            document_id, chunks, embedding_status = await ContentIngestionService(
                self.repository
            ).ingest(
                user_id=job.user_id,
                title=job.title,
                subject=job.subject,
                source_url=f"knowledge:{job.id}",
                publisher=None,
                rights_note=job.rights_note,
                content=content,
                skill_ids=job.skill_ids,
                original_filename=job.original_filename,
                mime_type=mime_type,
                size_bytes=job.size_bytes,
            )
            return await self.repository.update_knowledge_job(
                job.id,
                {
                    "status": "completed",
                    "document_id": document_id,
                    "chunks": chunks,
                    "embedding_status": embedding_status,
                    "error": None,
                },
            )
        except Exception as exc:
            return await self.repository.update_knowledge_job(
                job.id,
                {"status": "failed", "error": str(exc)[:500]},
            )
        finally:
            try:
                await self.storage.delete(job.storage_path)
            except Exception:
                pass


async def process_knowledge_job(job_id: str) -> KnowledgeIngestionJob | None:
    """Async entrypoint used by FastAPI background tasks and webhook delivery."""
    return await KnowledgeIngestionProcessor(get_repository()).process(job_id)


def run_knowledge_job(job_id: str) -> None:
    """Synchronous RQ entrypoint."""
    asyncio.run(process_knowledge_job(job_id))


def enqueue_knowledge_job(job_id: str) -> bool:
    """Enqueue in Redis when available; callers can fall back to a local task."""
    settings = get_settings()
    if not settings.redis_url:
        return False
    try:
        from redis import Redis
        from rq import Queue

        connection = Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=1,
            socket_timeout=2,
        )
        Queue("knowledge-ingestion", connection=connection).enqueue(
            run_knowledge_job,
            job_id,
            job_id=f"knowledge:{job_id}",
            job_timeout=600,
            result_ttl=3600,
        )
        return True
    except Exception:
        return False
