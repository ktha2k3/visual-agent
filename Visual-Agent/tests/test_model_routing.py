import pytest

from utils.config import get_settings
from utils.model_factory import get_chat_model, get_embedding_model


@pytest.fixture(autouse=True)
def configured_models(monkeypatch):
    values={
        'RIVER_API_KEY':'river-test-key', 'RIVER_BASE_URL':'https://api.river.run/v1',
        'LLM_MODEL_NAME':'deepseek-ai/DeepSeek-V4.1-Flash',
        'OLLAMA_API_KEY':'ollama-test-key', 'OLLAMA_BASE_URL':'https://ollama.com/v1',
        'VISION_MODEL':'gemma4:31b-cloud', 'RESEARCH_MODEL':'gpt-oss:120b-cloud',
        'EMBEDDING_BASE_URL':'http://embeddings:11434/v1',
    }
    for name,value in values.items():monkeypatch.setenv(name,value)
    for name in ['TUTOR_MODEL','VISUALIZATION_MODEL']:
        monkeypatch.delenv(name,raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.parametrize('role',['tutor','visualization'])
def test_main_roles_use_river_deepseek_with_river_credentials(role):
    model=get_chat_model(role)
    assert model.model_name=='deepseek-ai/DeepSeek-V4.1-Flash'
    assert model.openai_api_base=='https://api.river.run/v1'
    assert model.openai_api_key.get_secret_value()=='river-test-key'


@pytest.mark.parametrize('role,expected',[('vision','gemma4:31b'),('research','gpt-oss:120b')])
def test_image_and_research_roles_stay_on_ollama(role,expected):
    model=get_chat_model(role)
    assert model.model_name==expected
    assert model.openai_api_base=='https://ollama.com/v1'
    assert model.openai_api_key.get_secret_value()=='ollama-test-key'


def test_missing_river_key_does_not_silently_use_ollama(monkeypatch):
    monkeypatch.setenv('RIVER_API_KEY','')
    get_settings.cache_clear()
    with pytest.raises(ValueError,match='RIVER_API_KEY'):
        get_chat_model('tutor')


def test_embeddings_keep_their_dedicated_endpoint():
    embeddings=get_embedding_model()
    assert embeddings.openai_api_base=='http://embeddings:11434/v1'
    assert embeddings.openai_api_key.get_secret_value()=='ollama-test-key'
