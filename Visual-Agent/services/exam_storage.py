from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlparse

import httpx

from education.models import ExamSource
from services.repository import TutorRepository
from utils.config import Settings, get_settings


class ExamArtifactStorage:
    def __init__(self, repository: TutorRepository, settings: Settings | None = None):
        self.repository = repository
        self.settings = settings or get_settings()
        self.local_dir = Path(__file__).resolve().parents[1] / "assets" / "exam-artifacts"

    async def save(
        self,
        source: ExamSource,
        *,
        original_url: str,
        content: bytes,
        mime_type: str,
        checksum: str,
    ) -> str:
        extension = Path(urlparse(original_url).path).suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp", ".pdf"}:
            extension = ".pdf" if mime_type == "application/pdf" else ".bin"
        storage_path = f"{source.exam_year}/{source.subject.value}/{checksum}{extension}"
        if self.settings.supabase_service_enabled:
            url = (
                f"{self.settings.supabase_url}/storage/v1/object/"
                f"{self.settings.supabase_exam_bucket}/{storage_path}"
            )
            headers = {
                "apikey": self.settings.supabase_service_role_key,
                "Authorization": f"Bearer {self.settings.supabase_service_role_key}",
                "Content-Type": mime_type,
                "x-upsert": "true",
            }
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, headers=headers, content=content)
            response.raise_for_status()
        else:
            self.local_dir.mkdir(parents=True, exist_ok=True)
            path = self.local_dir / f"{checksum}{extension}"
            if not path.exists():
                await asyncio.to_thread(path.write_bytes, content)
            storage_path = str(path.resolve())
        return await self.repository.save_exam_artifact(
            {
                "source_id": source.id,
                "storage_path": storage_path,
                "original_url": original_url,
                "mime_type": mime_type,
                "checksum": checksum,
            }
        )
