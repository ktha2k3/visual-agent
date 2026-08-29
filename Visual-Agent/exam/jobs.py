from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from education.models import ExamSource
from exam.extractor import ExamExtractor
from exam.fetcher import SourceFetcher
from exam.sources import SourceDefinition, recent_source_definitions
from services.repository import TutorRepository, get_repository
from services.storage import AssetStorage
from services.exam_storage import ExamArtifactStorage


class ExamSyncService:
    def __init__(self, repository: TutorRepository):
        self.repository = repository
        self.fetcher = SourceFetcher()
        self.extractor = ExamExtractor()
        self.artifacts = ExamArtifactStorage(repository)

    async def sync_source(
        self, definition: SourceDefinition, *, extract: bool = True
    ) -> dict:
        fetched = await self.fetcher.fetch(definition.url)
        source = definition.to_model()
        source.url = fetched.final_url
        source.checksum = fetched.checksum
        source.fetched_at = datetime.now(timezone.utc)
        source = await self.repository.save_exam_source(source)

        questions = []
        failures = []
        if extract:
            for artifact_url in fetched.artifact_urls:
                try:
                    content, mime, checksum = await self.fetcher.fetch_artifact(artifact_url)
                    await self.artifacts.save(
                        source,
                        original_url=artifact_url,
                        content=content,
                        mime_type=mime,
                        checksum=checksum,
                    )
                    if not mime.startswith("image/"):
                        continue
                    questions.extend(await self.extractor.extract_image(source, content, mime))
                except Exception as exc:
                    failures.append({"url": artifact_url, "error": str(exc)})
            await self.repository.save_exam_questions(questions)
        return {
            "url": source.url,
            "subject": source.subject.value,
            "year": source.exam_year,
            "checksum": source.checksum,
            "artifacts_found": len(fetched.artifact_urls),
            "questions_pending_review": len(questions),
            "failures": failures,
        }

    async def sync_initial_sources(self, *, extract: bool = True) -> list[dict]:
        results = []
        for definition in recent_source_definitions():
            try:
                results.append(await self.sync_source(definition, extract=extract))
            except Exception as exc:
                results.append(
                    {
                        "url": definition.url,
                        "subject": definition.subject.value,
                        "year": definition.exam_year,
                        "error": str(exc),
                    }
                )
        return results


def run_initial_sync(*, extract: bool = True) -> list[dict]:
    return asyncio.run(ExamSyncService(get_repository()).sync_initial_sources(extract=extract))


def enqueue_initial_sync(*, extract: bool = True) -> str | list[dict]:
    from utils.config import get_settings

    settings = get_settings()
    if not settings.redis_url:
        return run_initial_sync(extract=extract)
    from redis import Redis
    from rq import Queue

    job = Queue("exam-ingestion", connection=Redis.from_url(settings.redis_url)).enqueue(
        run_initial_sync,
        extract=extract,
        job_timeout=1800,
        result_ttl=86400,
    )
    return job.id


def run_upload_cleanup() -> int:
    repository = get_repository()
    return asyncio.run(AssetStorage(repository).cleanup_expired())
