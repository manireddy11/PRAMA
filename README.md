<p align="center">
  <img src="brand/prama-logo.svg" alt="Prama — Ask. Verify. Know." width="420" />
</p>

<p align="center"><strong>Ask. Verify. Know.</strong><br />Answers you can check.</p>


<p align="center">
  <img src="./Screenshot%202026-11-26%20135628.png" alt="PRAMA Screenshot" width="700" />
</p>



> Prama comes from Sanskrit *pramāṇa*: a valid source of knowledge, or evidence.
> A P whose bowl is a lens, with the source at its centre.

Prama is a local research assistant with web search, direct chat, follow-up threads, source previews, and verified inline citations. The Next.js app proxies HTTP events from FastAPI on the same origin.

A web question follows a Perplexity-style pipeline:

1. Groq rewrites the question into up to three sub-queries.
2. Jina Search runs each sub-query and the results are merged and de-duplicated.
3. Jina Reader extracts the full text of the strongest pages.
4. BGE embeddings rank passages against every sub-query, and a cross-encoder re-ranks the shortlist.
5. Groq writes the answer from the selected passages, citing a source for every factual sentence.
6. Every citation is checked against the retrieved passages. Unsupported sentences are removed, repaired citations are corrected, and the streamed answer reports how many sentences were verified.

Upstash Redis caches search results for 15 minutes, and Pinecone keeps a `prama-web` namespace of previously seen pages for extra recall.

![PRAMA Screenshot 2](./Screenshot%202026-11-26%20135711.png)


## Run locally

Use Node.js 22+ and Python 3.11 or 3.12. Open two PowerShell terminals from the repository.

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env: set GROQ_API_KEY and JINA_API_KEY for live answers,
# or DEMO_MODE=true for the sample stream.
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev -- --hostname 127.0.0.1
```

Open <http://127.0.0.1:3000>. Demo mode streams a fixed, labeled sample response. API configuration stays in the backend. Next.js uses `BACKEND_URL` when FastAPI runs at a different address.

## Enable live providers

Set `DEMO_MODE=false` in `backend/.env`. All answers need `GROQ_API_KEY`. Web search first calls [Jina Search](https://jina.ai/search/) through its URL path (`https://s.jina.ai/<query>`) with no key; if that is unavailable, Prama retries the keyed `?q=` endpoint using `JINA_API_KEY`. Set the key when Jina requires authentication. [Jina Reader](https://jina.ai/reader/) (`r.jina.ai`) extracts full pages. The default Groq model is `openai/gpt-oss-20b`.

`MAX_SUBQUERIES` (3), `MAX_SOURCES` (6), and `FETCH_FULL_TEXT` tune retrieval. `CITATION_THRESHOLD` (default `0.62`) is the minimum embedding similarity for a sentence to count as supported by a passage; lower it for looser matching or raise it for stricter answers. `ENABLE_RERANKING=false` skips the cross-encoder. Optional integrations are Upstash Redis for the search cache, Pinecone for vector recall, and Supabase for saved history.

History is scoped to a randomly generated browser session. Completed turns are also available in local browser storage. Run the app on localhost for personal use. Public deployment requires account authentication and abuse controls for paid API calls.

## License

MIT. The Montserrat font is licensed under the SIL Open Font License; see `frontend/public/fonts/OFL.txt`.
