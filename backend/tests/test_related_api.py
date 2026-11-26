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


BODY = {"question": "best AI courses", "answer": "IBM AI Engineering covers PyTorch [1].",
        "sources": [{"url": "https://a.example", "title": "A", "content": "IBM AI Engineering covers PyTorch."}]}


def test_related_returns_items_and_analysis(client, monkeypatch):
    async def fake(client, request):
        assert request.sources[0].url == "https://a.example"
        return {"entities": ["IBM AI Engineering"], "queries": ["IBM AI Engineering courses"], "items": [],
                "analysis": {"documents": []}}

    monkeypatch.setattr(main, "research_intelligence", fake)
    response = client.post("/related", json=BODY)
    assert response.status_code == 200
    assert response.json()["queries"] == ["IBM AI Engineering courses"]


@pytest.mark.parametrize("error, status", [
    (JinaAuthError("Jina Search needs JINA_API_KEY in backend/.env."), 503),
    (httpx.ReadTimeout("slow"), 502),
    (ValueError("boom"), 502),
])
def test_related_maps_errors(client, monkeypatch, error, status):
    async def failing(client, request):
        raise error

    monkeypatch.setattr(main, "research_intelligence", failing)
    response = client.post("/related", json=BODY)
    assert response.status_code == status
    assert "boom" not in response.text


@pytest.mark.parametrize("body", [
    {"question": "", "answer": "a"},
    {"question": "q"},
    {"question": "q", "answer": "a", "sources": [{"url": "https://a.example"}] * 9},
    {"question": "q", "answer": "a", "sources": [{"url": "https://a.example", "content": "x" * 8001}]},
])
def test_related_validates_input(client, body):
    assert client.post("/related", json=body).status_code == 422


def test_related_demo_mode(client, monkeypatch):
    monkeypatch.setattr(research.settings, "demo_mode", True)
    data = client.post("/related", json=BODY).json()
    assert data["items"] and data["analysis"]["common_findings"]
