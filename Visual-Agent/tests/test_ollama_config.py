import pytest

from utils.model_factory import UnavailableCloudEmbeddings
from utils.ollama import is_ollama_cloud, normalize_ollama_model


def test_direct_cloud_model_names_drop_local_routing_suffix():
    assert is_ollama_cloud("https://ollama.com/v1")
    assert normalize_ollama_model("gemma4:31b-cloud", "https://ollama.com/v1") == "gemma4:31b"
    assert normalize_ollama_model("gpt-oss:120b-cloud", "http://localhost:11434/v1") == "gpt-oss:120b-cloud"


@pytest.mark.asyncio
async def test_cloud_embedding_fallback_is_explicit():
    embeddings = UnavailableCloudEmbeddings()
    with pytest.raises(RuntimeError, match="EMBEDDING_BASE_URL"):
        await embeddings.aembed_query("xin chào")
