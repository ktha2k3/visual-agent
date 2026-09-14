"""Central application configuration.

The project intentionally keeps configuration dependency-free so the API, CLI,
workers and tests all read the same environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(value: str | None, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_api_key: str
    tutor_model: str
    vision_model: str
    research_model: str
    visualization_model: str
    embedding_model: str
    embedding_base_url: str
    embedding_dimensions: int
    redis_url: str
    supabase_url: str
    supabase_publishable_key: str
    supabase_service_role_key: str
    supabase_storage_bucket: str
    auth_required: bool
    admin_emails: tuple[str, ...]
    cors_origins: tuple[str, ...]
    upload_max_bytes: int
    upload_retention_days: int
    debug_stream: bool
    chat_database_path: str
    database_url: str
    app_timezone: str
    knowledge_webhook_secret: str
    visualization_artifact_dir: str
    visualization_version: str
    visualization_quality: str
    visualization_render_timeout: float
    visualization_max_render_retries: int
    visualization_max_concurrency: int

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_publishable_key)

    @property
    def supabase_service_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Import lazily to avoid a configuration import cycle.
    from utils.ollama import get_ollama_base_url

    ollama_base_url = get_ollama_base_url()
    default_model = os.getenv("MODEL_NAME", "gemma4:31b")
    return Settings(
        ollama_base_url=ollama_base_url,
        ollama_api_key=os.getenv("OLLAMA_API_KEY", ""),
        tutor_model=os.getenv("TUTOR_MODEL", default_model),
        vision_model=os.getenv("VISION_MODEL", default_model),
        research_model=os.getenv(
            "RESEARCH_MODEL",
            os.getenv("MODEL_RESEARCH_NAME", "gpt-oss:120b"),
        ),
        visualization_model=os.getenv(
            "VISUALIZATION_MODEL",
            os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud"),
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "").strip(),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "768")),
        redis_url=os.getenv("REDIS_URL", ""),
        supabase_url=os.getenv("SUPABASE_URL", "").rstrip("/"),
        supabase_publishable_key=os.getenv(
            "SUPABASE_PUBLISHABLE_KEY", os.getenv("SUPABASE_ANON_KEY", "")
        ),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        supabase_storage_bucket=os.getenv("SUPABASE_STORAGE_BUCKET", "uploads"),
        auth_required=_as_bool(os.getenv("AUTH_REQUIRED"), default=False),
        admin_emails=tuple(
            email.lower() for email in _csv(os.getenv("ADMIN_EMAILS"))
        ),
        cors_origins=_csv(
            os.getenv("CORS_ORIGINS"),
            ("http://localhost:3000", "http://localhost:5173"),
        ),
        upload_max_bytes=int(os.getenv("UPLOAD_MAX_BYTES", str(25 * 1024 * 1024))),
        upload_retention_days=int(os.getenv("UPLOAD_RETENTION_DAYS", "30")),
        debug_stream=_as_bool(os.getenv("DEBUG_STREAM"), default=False),
        chat_database_path=os.getenv("CHAT_DATABASE_PATH", "data/chat_history.sqlite3"),
        database_url=os.getenv("DATABASE_URL", "").strip(),
        app_timezone=os.getenv("APP_TIMEZONE", "Asia/Ho_Chi_Minh"),
        knowledge_webhook_secret=os.getenv("KNOWLEDGE_WEBHOOK_SECRET", ""),
        visualization_artifact_dir=os.getenv(
            "VISUALIZATION_ARTIFACT_DIR", "data/visualizations"
        ),
        visualization_version=os.getenv("VISUALIZATION_VERSION", "stem-v1"),
        visualization_quality=os.getenv("VISUALIZATION_QUALITY", "l"),
        visualization_render_timeout=float(
            os.getenv("VISUALIZATION_RENDER_TIMEOUT", "300")
        ),
        visualization_max_render_retries=max(
            0, int(os.getenv("MAX_RENDER_RETRIES", "3"))
        ),
        visualization_max_concurrency=max(
            1, int(os.getenv("VISUALIZATION_MAX_CONCURRENCY", "2"))
        ),
    )
