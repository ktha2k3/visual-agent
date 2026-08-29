import os
from urllib.parse import urlparse


def get_ollama_base_url() -> str:
    """Return a normalized Ollama OpenAI-compatible base URL.

    Supports local and remote Ollama endpoints.
    - `http://localhost:11434/api` -> `http://localhost:11434/v1`
    - `https://ollama.com/api` -> `https://ollama.com/v1`
    - `http://localhost:11434/v1` stays unchanged
    - `https://ollama.com/v1` stays unchanged
    """
    url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1").strip()
    if url.endswith("/api"):
        url = url[: -len("/api")]
    if url.endswith("/api/"):
        url = url[: -len("/api/")]
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def is_ollama_cloud(base_url: str | None = None) -> bool:
    url = base_url or get_ollama_base_url()
    return (urlparse(url).hostname or "").lower() == "ollama.com"


def normalize_ollama_model(model: str, base_url: str | None = None) -> str:
    """Direct cloud uses normal tags; local cloud routing uses ``-cloud``."""
    if is_ollama_cloud(base_url) and model.endswith("-cloud"):
        return model[: -len("-cloud")]
    return model
