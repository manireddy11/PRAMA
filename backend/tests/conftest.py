import os

# Pin every setting before the app is imported. Environment variables take
# precedence over backend/.env, so tests never read a developer's real keys
# or reach external providers.
TEST_ENV = {
    "JINA_API_KEY": "",
    "JINA_SEARCH_URL": "https://s.jina.test/",
    "JINA_READER_URL": "https://r.jina.test/",
    "GROQ_API_KEY": "",
    "GROQ_MODEL": "test-model",
    "UPSTASH_REDIS_REST_URL": "",
    "UPSTASH_REDIS_REST_TOKEN": "",
    "PINECONE_API_KEY": "",
    "PINECONE_HOST": "",
    "SUPABASE_URL": "",
    "SUPABASE_SECRET_KEY": "",
    "DEMO_MODE": "false",
    "ENABLE_RERANKING": "true",
    "MAX_SUBQUERIES": "3",
    "MAX_SOURCES": "6",
    "FETCH_FULL_TEXT": "true",
    "CITATION_THRESHOLD": "0.62",
}
os.environ.update(TEST_ENV)

import pytest  # noqa: E402

import research  # noqa: E402


@pytest.fixture(autouse=True)
def no_local_models(monkeypatch):
    """Make the embedding and reranking models unavailable.

    The pipeline then uses its lexical fallbacks, which are deterministic and
    need no model download.
    """
    def unavailable():
        raise RuntimeError("models are disabled in tests")

    monkeypatch.setattr(research, "embedder", unavailable)
    monkeypatch.setattr(research, "reranker", unavailable)
