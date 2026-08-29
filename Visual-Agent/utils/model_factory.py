"""Configured model constructors used by every graph and tool."""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config import get_settings
from utils.ollama import is_ollama_cloud, normalize_ollama_model


class UnavailableCloudEmbeddings:
    """Explicit fallback because Ollama Cloud does not expose embeddings today."""

    async def aembed_query(self, text: str) -> list[float]:
        raise RuntimeError(
            "Ollama Cloud does not expose embeddings; configure EMBEDDING_BASE_URL "
            "for a local Ollama or OpenAI-compatible embedding endpoint"
        )

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "Ollama Cloud does not expose embeddings; configure EMBEDDING_BASE_URL "
            "for a local Ollama or OpenAI-compatible embedding endpoint"
        )


def get_chat_model(role: str = "tutor", *, temperature: float = 0.1) -> ChatOpenAI:
    settings = get_settings()
    names = {
        "tutor": settings.tutor_model,
        "vision": settings.vision_model,
        "research": settings.research_model,
        "visualization": settings.visualization_model,
    }
    if role not in names:
        raise ValueError(f"Unknown model role: {role}")
    return ChatOpenAI(
        model=normalize_ollama_model(names[role], settings.ollama_base_url),
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
        temperature=temperature,
    )


def get_embedding_model() -> OpenAIEmbeddings | UnavailableCloudEmbeddings:
    settings = get_settings()
    embedding_url = settings.embedding_base_url
    if not embedding_url and is_ollama_cloud(settings.ollama_base_url):
        return UnavailableCloudEmbeddings()
    embedding_url = embedding_url or settings.ollama_base_url
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=embedding_url,
        api_key=settings.ollama_api_key,
        check_embedding_ctx_length=False,
    )
