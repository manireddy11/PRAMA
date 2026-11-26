import json
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

import main
import research
from research import JinaAuthError


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def post(client, body, *, session=True):
    headers = {"x-session-id": str(uuid4())} if session else {}
    return client.post("/search", params={"thread": str(uuid4())}, json=body, headers=headers)


def events(response):
    parsed = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        parsed.append((lines["event"], json.loads(lines["data"])))
    return parsed


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "demo": False, "persistence": False}


def test_history_is_empty_without_supabase(client):
    response = client.get("/search", headers={"x-session-id": str(uuid4())})
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize("body, session", [
    ({"question": "heat pumps"}, False),
    ({"question": ""}, True),
    ({"question": "heat pumps", "mode": "images"}, True),
    ({"question": "x" * 2001}, True),
])
def test_search_rejects_invalid_requests(client, body, session):
    assert post(client, body, session=session).status_code == 422


def test_search_rejects_blank_question(client):
    response = post(client, {"question": "   "})
    assert response.status_code == 422
    assert response.json()["detail"] == "Enter a question."


def test_search_requires_groq_key_outside_demo_mode(client):
    response = post(client, {"question": "heat pumps"})
    assert response.status_code == 503


def test_demo_web_search_streams_events_in_order(client, monkeypatch):
    monkeypatch.setattr(research.settings, "demo_mode", True)
    response = post(client, {"question": "heat pumps", "mode": "web"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    stream = events(response)
    kinds = [kind for kind, _ in stream]
    assert kinds[0] == "status"
    assert kinds.index("sources") < kinds.index("token") < kinds.index("done")
    assert kinds[-1] == "done"
    sources = next(data for kind, data in stream if kind == "sources")
    assert len(sources) == 2
    answer = "".join(data["text"] for kind, data in stream if kind == "token")
    assert answer.startswith("**Demo mode**")
    assert stream[-1][1] == {"persisted": False}


def test_demo_chat_mode_has_no_sources(client, monkeypatch):
    monkeypatch.setattr(research.settings, "demo_mode", True)
    stream = events(post(client, {"question": "hello", "mode": "chat"}))
    assert next(data for kind, data in stream if kind == "sources") == []


def failing_retrieval(error):
    async def retrieve(client, query):
        raise error
    return retrieve


def status_error(code):
    request = httpx.Request("GET", "https://s.jina.test/")
    return httpx.HTTPStatusError("failed", request=request, response=httpx.Response(code, request=request))


@pytest.mark.parametrize("error, message", [
    (JinaAuthError("Jina Search needs JINA_API_KEY in backend/.env."), "Jina Search needs JINA_API_KEY"),
    (httpx.ReadTimeout("slow"), "took too long"),
    (status_error(401), "rejected the API key"),
    (status_error(429), "rate limit"),
    (status_error(500), "could not finish"),
    (ValueError("boom"), "could not finish"),
])
def test_pipeline_errors_become_error_events(client, monkeypatch, error, message):
    monkeypatch.setattr(research.settings, "groq_api_key", "key")
    monkeypatch.setattr(main, "retrieve_sources", failing_retrieval(error))
    stream = events(post(client, {"question": "heat pumps"}))
    kind, data = stream[-1]
    assert kind == "error"
    assert message in data["message"]


def test_web_search_without_sources_explains_instead_of_calling_groq(client, monkeypatch):
    monkeypatch.setattr(research.settings, "groq_api_key", "key")

    async def no_sources(client, query):
        return [], ["heat pumps"]

    async def unexpected(*args, **kwargs):
        raise AssertionError("Groq should not be called without sources")

    monkeypatch.setattr(main, "retrieve_sources", no_sources)
    monkeypatch.setattr(main, "generate_answer", unexpected)
    stream = events(post(client, {"question": "heat pumps"}))
    answer = "".join(data["text"] for kind, data in stream if kind == "token")
    assert answer.startswith("I couldn't find web sources")
    assert stream[-1][0] == "done"
