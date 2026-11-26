import asyncio
import json
import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from intelligence import RelatedRequest, research_intelligence

from research import JinaAuthError, Query, Source, generate_answer, retrieve_sources, settings, verify_answer

log = logging.getLogger(__name__)
PERSISTENCE = bool(settings.supabase_url and settings.supabase_secret_key)


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as client:
        app.state.client = client
        yield


app = FastAPI(title="Prama", lifespan=lifespan)


def event(kind, payload):
    return f"event: {kind}\ndata: {json.dumps(payload)}\n\n"


async def database(method, **kwargs):
    response = await app.state.client.request(
        method, settings.supabase_url.rstrip("/") + "/rest/v1/research_turns",
        headers={"apikey": settings.supabase_secret_key}, **kwargs,
    )
    response.raise_for_status()
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "demo": settings.demo_mode, "persistence": PERSISTENCE}


@app.get("/search")
async def history(thread: UUID | None = None, x_session_id: UUID = Header()):
    if not PERSISTENCE:
        return []
    params = {"session_id": f"eq.{x_session_id}", "order": "created_at.desc", "limit": 100,
              "select": "id,thread_id,question,answer,sources,created_at"}
    if thread:
        params["thread_id"] = f"eq.{thread}"
    try:
        return (await database("GET", params=params)).json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Saved history is unavailable.") from exc


def chunks(text, size=28):
    for index in range(0, len(text), size):
        yield text[index:index + size]


@app.post("/search")
async def search(query: Query, thread: UUID, x_session_id: UUID = Header()):
    query.question = query.question.strip()
    if not query.question:
        raise HTTPException(422, "Enter a question.")
    if not settings.demo_mode and not settings.groq_api_key:
        raise HTTPException(503, "Set GROQ_API_KEY in backend/.env, or enable demo mode.")

    async def stream():
        yield event("status", {"message": "Planning searches" if query.mode == "web" else "Preparing answer"})
        try:
            sources = []
            queries = []
            if settings.demo_mode:
                if query.mode == "web":
                    sources = [
                        Source(url="https://jina.ai/reader/", title="Jina Reader",
                               content="Jina Reader converts a URL into clean, LLM-friendly Markdown, including full page text for grounding."),
                        Source(url="https://console.groq.com/docs/text-chat", title="Groq text generation",
                               content="Groq streams low-latency model completions over an OpenAI-compatible chat API."),
                    ]
            elif query.mode == "web":
                yield event("status", {"message": "Searching the web with Jina"})
                sources, queries = await retrieve_sources(app.state.client, query)
                yield event("status", {"message": f"Reading {len(sources)} sources from {len(queries)} queries"})
            public = [source.model_dump(mode="json") for source in sources]
            yield event("sources", public)
            yield event("status", {"message": "Writing answer"})
            answer = ""
            verification = None
            if settings.demo_mode:
                answer = ("**Demo mode**\n\nThis sample shows how Prama streams an answer."
                          " Jina Reader extracts full pages [1], and Groq writes the cited answer [2]."
                          " Live responses require provider credentials.")
            elif query.mode == "web" and not sources:
                answer = ("I couldn't find web sources for this question. "
                          "Try rephrasing it or asking something more specific.")
            else:
                answer = await generate_answer(app.state.client, query, sources)
                if query.mode == "web" and sources:
                    yield event("status", {"message": "Verifying citations"})
                    answer, verification = await verify_answer(app.state.client, answer, sources)
                    yield event("verification", verification)
            if not answer.strip():
                raise RuntimeError("The model did not return an answer")
            for chunk in chunks(answer):
                yield event("token", {"text": chunk})
                await asyncio.sleep(0.008)
            saved = False
            if PERSISTENCE and not settings.demo_mode:
                try:
                    await database("POST", json={"id": str(uuid4()), "session_id": str(x_session_id),
                                                  "thread_id": str(thread), "question": query.question,
                                                  "answer": answer, "sources": public})
                    saved = True
                except httpx.HTTPError:
                    yield event("status", {"message": "Supabase could not save this answer. It is available in this browser."})
            yield event("done", {"persisted": saved})
        except JinaAuthError as exc:
            yield event("error", {"message": str(exc)})
        except (httpx.TimeoutException, asyncio.TimeoutError):
            log.error("Research pipeline timed out")
            yield event("error", {"message": "Jina took too long to respond. Try again in a moment."})
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            log.error("Research pipeline HTTP %s", code)
            if code in (401, 403):
                message = "Jina or Groq rejected the API key. Check JINA_API_KEY and GROQ_API_KEY."
            elif code == 429:
                message = "Jina or Groq rate limit reached. Try again in a moment."
            else:
                message = "The answer could not finish. Try again in a moment."
            yield event("error", {"message": message})
        except Exception as exc:
            log.error("Research pipeline failed: %s", type(exc).__name__)
            yield event("error", {"message": "The answer could not finish. Try again in a moment."})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


DEMO_INTELLIGENCE = {
    "entities": ["Jina Reader", "Groq"],
    "queries": [],
    "items": [
        {"url": "https://jina.ai/reader/", "title": "Jina Reader", "domain": "jina.ai", "published": "",
         "description": "Convert a URL into LLM-friendly input.",
         "excerpt": "Reader converts any URL to clean Markdown so language models can ground answers in full page text.",
         "entities": ["Jina Reader"], "answer_sentence": "Jina Reader extracts full pages.", "match": "strong",
         "score": 0.84},
    ],
    "analysis": {
        "documents": [
            {"id": "S1", "label": "[1]", "url": "https://jina.ai/reader/", "title": "Jina Reader"},
            {"id": "R1", "label": "R1", "url": "https://jina.ai/reader/", "title": "Jina Reader"},
        ],
        "similarity": [[1.0, 0.9], [0.9, 1.0]],
        "clusters": [{"label": "Jina Reader", "documents": ["S1", "R1"]}],
        "common_findings": [{"text": "Jina Reader converts a URL into clean Markdown for language models.",
                             "sources": [{"document": "S1", "text": "Jina Reader converts a URL into clean Markdown."},
                                         {"document": "R1", "text": "Reader converts any URL to clean Markdown."}]}],
        "conflicts": [],
        "unique_insights": [],
    },
}


@app.post("/related")
async def related(request: RelatedRequest):
    """Related pages for an answer and a cross-document analysis; shown below the answer."""
    if settings.demo_mode:
        return DEMO_INTELLIGENCE
    try:
        return await research_intelligence(app.state.client, request)
    except JinaAuthError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        log.error("Related research failed: %s", type(exc).__name__)
        raise HTTPException(502, "Related research is unavailable right now.") from exc

