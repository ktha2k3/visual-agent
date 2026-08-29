from __future__ import annotations

import asyncio
import io
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from education.models import AssetRecord, AuthUser
from services.repository import TutorRepository
from utils.config import Settings, get_settings

ALLOWED_FORMATS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "GIF": ("image/gif", ".gif"),
    "WEBP": ("image/webp", ".webp"),
    "BMP": ("image/bmp", ".bmp"),
}


async def read_validated_image(upload: UploadFile, max_bytes: int) -> tuple[bytes, str, str]:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > max_bytes:
            limit_mb = max_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the {limit_mb:g} MB limit",
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            detected = image.format
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="File is not a valid image") from exc
    if detected not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415, detail="Unsupported image format")
    mime, extension = ALLOWED_FORMATS[detected]
    return data, mime, extension


class AssetStorage:
    def __init__(self, repository: TutorRepository, settings: Settings | None = None):
        self.repository = repository
        self.settings = settings or get_settings()
        self.local_dir = Path(__file__).resolve().parent.parent / "assets" / "uploads"

    async def save(self, user: AuthUser, upload: UploadFile) -> AssetRecord:
        data, mime, extension = await read_validated_image(
            upload, self.settings.upload_max_bytes
        )
        asset_id = str(uuid.uuid4())
        storage_path = f"{user.id}/{asset_id}{extension}"
        preview_url: str | None
        if self.settings.supabase_service_enabled:
            await self._upload_supabase(storage_path, data, mime)
            preview_url = await self._sign_supabase(storage_path)
        else:
            self.local_dir.mkdir(parents=True, exist_ok=True)
            destination = self.local_dir / f"{asset_id}{extension}"
            await asyncio.to_thread(destination.write_bytes, data)
            storage_path = str(destination.resolve())
            preview_url = f"/api/assets/{asset_id}"
        record = AssetRecord(
            id=asset_id,
            user_id=user.id,
            storage_path=storage_path,
            mime_type=mime,
            size_bytes=len(data),
            preview_url=preview_url,
        )
        await self.repository.save_asset(record)
        return record

    async def read(self, record: AssetRecord) -> bytes:
        if self.settings.supabase_service_enabled:
            url = (
                f"{self.settings.supabase_url}/storage/v1/object/authenticated/"
                f"{self.settings.supabase_storage_bucket}/{record.storage_path}"
            )
            headers = {
                "apikey": self.settings.supabase_service_role_key,
                "Authorization": f"Bearer {self.settings.supabase_service_role_key}",
            }
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.content
        path = Path(record.storage_path).resolve()
        if self.local_dir.resolve() not in path.parents:
            raise HTTPException(status_code=403, detail="Invalid asset path")
        return await asyncio.to_thread(path.read_bytes)

    async def delete_user_assets(self, user_id: str) -> int:
        """Delete private objects before their database ownership rows are removed."""
        records = await self.repository.list_assets(user_id)
        if not records:
            return 0
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
                    "DELETE",
                    url,
                    headers=headers,
                    json={"prefixes": [item.storage_path for item in records]},
                )
            response.raise_for_status()
            return len(records)

        root = self.local_dir.resolve()
        deleted = 0
        for record in records:
            path = Path(record.storage_path).resolve()
            if root not in path.parents:
                continue
            if path.is_file():
                await asyncio.to_thread(path.unlink)
                deleted += 1
        return deleted

    async def cleanup_expired(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.settings.upload_retention_days
        )
        records = await self.repository.list_expired_assets(cutoff)
        if not records:
            return 0
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
                    "DELETE", url, headers=headers,
                    json={"prefixes": [item.storage_path for item in records]},
                )
            response.raise_for_status()
        else:
            root = self.local_dir.resolve()
            for record in records:
                path = Path(record.storage_path).resolve()
                if root in path.parents and path.is_file():
                    await asyncio.to_thread(path.unlink)
        await self.repository.delete_asset_records([item.id for item in records])
        return len(records)

    async def _upload_supabase(self, storage_path: str, data: bytes, mime: str) -> None:
        url = (
            f"{self.settings.supabase_url}/storage/v1/object/"
            f"{self.settings.supabase_storage_bucket}/{storage_path}"
        )
        headers = {
            "apikey": self.settings.supabase_service_role_key,
            "Authorization": f"Bearer {self.settings.supabase_service_role_key}",
            "Content-Type": mime,
            "x-upsert": "false",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, headers=headers, content=data)
        response.raise_for_status()

    async def _sign_supabase(self, storage_path: str) -> str:
        url = (
            f"{self.settings.supabase_url}/storage/v1/object/sign/"
            f"{self.settings.supabase_storage_bucket}/{storage_path}"
        )
        headers = {
            "apikey": self.settings.supabase_service_role_key,
            "Authorization": f"Bearer {self.settings.supabase_service_role_key}",
        }
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(url, headers=headers, json={"expiresIn": 900})
        response.raise_for_status()
        signed = response.json()["signedURL"]
        return f"{self.settings.supabase_url}/storage/v1{signed}"
