"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { BookOpen, Check, Compass, ExternalLink, Globe2, History, Menu, MessageSquare, Moon, Plus, Send, Square, Sun, X } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";

type Source = { url: string; title: string; content: string };
type Verification = { sentences: number; cited: number; repaired: number; removed: number; sources: number };
type Turn = { question: string; answer: string; sources: Source[]; verification?: Verification };
type Thread = { id: string; turns: Turn[] };
type SavedTurn = Turn & { thread_id: string };
type Mode = "web" | "chat";
type RelatedItem = {
  url: string; title: string; domain: string; published: string; description: string; excerpt: string;
  entities: string[]; answer_sentence: string; match: "strong" | "related";
};
type DocumentRef = { id: string; label: string; url: string; title: string };
type Quote = { document: string; text: string };
type Analysis = {
  documents: DocumentRef[]; similarity: number[][]; clusters: { label: string; documents: string[] }[];
  common_findings: { text: string; sources: Quote[] }[]; conflicts: { reason: string; claims: Quote[] }[];
  unique_insights: Quote[];
};
type Intelligence = { items: RelatedItem[]; analysis: Analysis };
type IntelligenceState = { status: "loading" | "done" | "error"; data?: Intelligence };

const prompts = [
  ["Career", "How to become top LinkedIn creator"],
  ["Marketing", "Ad campaign for your brand"],
  ["Jobs", "Search for most recent AI engineering jobs"],
  ["Learning", "Best AI engineering courses"],
];

async function readEvents(response: Response, onEvent: (name: string, data: string) => void) {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const name = block.match(/^event: (.+)$/m)?.[1];
        const data = block.match(/^data: (.+)$/m)?.[1];
        if (name && data) onEvent(name, data);
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

function PramaMark({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" focusable="false">
      <rect x="8" y="7" width="11" height="47" rx="5.5" fill="currentColor" />
      <circle cx="36" cy="25" r="13.5" fill="none" stroke="currentColor" strokeWidth="9" />
      <circle cx="36" cy="25" r="6" fill="#F5A800" />
    </svg>
  );
}

function hostOf(url: string) {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
}

function CitationChip({ number, source }: { number: number; source?: Source }) {
  return (
    <span className="citation-wrap">
      <a className="citation" href={source?.url ?? "#"} target="_blank" rel="noopener noreferrer"
        aria-label={source ? `Source ${number}: ${source.title}` : `Source ${number}`}>{number}</a>
      {source && (
        <span className="citation-card" role="tooltip">
          <span className="citation-domain">{hostOf(source.url)}</span>
          <strong>{source.title}</strong>
          <span className="citation-snippet">{source.content.slice(0, 240)}{source.content.length > 240 ? "…" : ""}</span>
        </span>
      )}
    </span>
  );
}

const PROTECTED = /(`[^`]*`|\[[^\]]*\]\([^)]*\)|https?:\/\/\S+)/;

function answerMarkdown(turn: Turn, mentions: string[] = []) {
  let text = turn.answer.replace(/\[(\d+)\](?!\()/g, (citation, number) => {
    const source = turn.sources[Number(number) - 1];
    return source ? citation + "(" + source.url.replaceAll("(", "%28").replaceAll(")", "%29") + ")" : citation;
  });
  // Link the first mention of each entity that has related pages, outside code, links and URLs.
  mentions.forEach((name, index) => {
    const pattern = new RegExp("(?<![\\w])" + name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?![\\w])");
    const parts = text.split(PROTECTED);
    for (let part = 0; part < parts.length; part += 2) {
      if (pattern.test(parts[part])) {
        parts[part] = parts[part].replace(pattern, (match) => `[${match}](#related-${index})`);
        text = parts.join("");
        return;
      }
    }
  });
  return text;
}

function groupByEntity(items: RelatedItem[]) {
  const groups = new Map<string, RelatedItem[]>();
  for (const item of items) {
    for (const entity of item.entities) {
      if (/[[\]()]/.test(entity)) continue;
      const pages = groups.get(entity) ?? [];
      if (pages.length < 3 && !pages.some((page) => page.url === item.url)) groups.set(entity, [...pages, item]);
    }
  }
  return [...groups.entries()].map(([name, pages]) => ({ name, pages }));
}

function RelatedMention({ label, pages }: { label: string; pages: RelatedItem[] }) {
  return (
    <span className="mention-wrap">
      <button type="button" className="mention" aria-label={`Related pages about ${label}`}>
        {label}<ExternalLink size={11} aria-hidden="true" />
      </button>
      <span className="mention-card" role="tooltip">
        <span className="citation-domain">Read more about {label}</span>
        {pages.map((page) => {
          const summary = page.description || page.excerpt;
          return (
            <a key={page.url} className="mention-page" href={page.url} target="_blank" rel="noopener noreferrer">
              <small>{page.domain}{formatDate(page.published) && <> · {formatDate(page.published)}</>}</small>
              <strong>{page.title}</strong>
              <span>{summary.slice(0, 160)}{summary.length > 160 ? "…" : ""}</span>
            </a>
          );
        })}
      </span>
    </span>
  );
}

function formatDate(value: string) {
  const date = new Date(value);
  return value && !Number.isNaN(date.getTime())
    ? date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "";
}

function DocumentChip({ reference, quote }: { reference?: DocumentRef; quote?: string }) {
  if (!reference) return null;
  return (
    <a className="doc-chip" href={reference.url} target="_blank" rel="noopener noreferrer"
      title={quote ? `${reference.title}: “${quote}”` : reference.title}>
      {reference.label.startsWith("[") && <span>{reference.label}</span>} {hostOf(reference.url)}
    </a>
  );
}

function CrossDocumentAnalysis({ analysis }: { analysis: Analysis }) {
  const byId = new Map(analysis.documents.map((document) => [document.id, document]));
  const { common_findings: common, conflicts, unique_insights: unique, clusters } = analysis;
  if (!common.length && !conflicts.length && !unique.length) return null;
  const agreeing = new Set(common.flatMap((finding) => finding.sources.map((quote) => quote.document))).size;
  return (
    <details className="analysis">
      <summary>
        {common.length > 0 && <span className="badge-agree"><Check size={13} aria-hidden="true" /> {agreeing} sources agree on {common.length} {common.length === 1 ? "point" : "points"}</span>}
        {conflicts.length > 0 && <span className="badge-conflict">⚠ {conflicts.length} possible {conflicts.length === 1 ? "conflict" : "conflicts"}</span>}
        {unique.length > 0 && <span>{unique.length} unique {unique.length === 1 ? "insight" : "insights"}</span>}
      </summary>
      <p className="analysis-note">Quotes from the answer’s sources and related pages. Similar wording means two sources discuss
        the same thing, not that they are correct.</p>
      {common.length > 0 && <section>
        <h3>Common findings</h3>
        <ul>{common.map((finding, index) => (
          <li key={index}>
            <q>{finding.text}</q>
            <div className="chips">Found in {finding.sources.length} sources: {finding.sources.map((quote) => (
              <DocumentChip key={quote.document} reference={byId.get(quote.document)} quote={quote.text} />
            ))}</div>
          </li>
        ))}</ul>
      </section>}
      {conflicts.length > 0 && <section>
        <h3>Possible conflicts</h3>
        <ul>{conflicts.map((conflict, index) => (
          <li key={index} className="conflict">
            <span className="analysis-reason">{conflict.reason}</span>
            {conflict.claims.map((quote) => (
              <div key={quote.document}><q>{quote.text}</q> <DocumentChip reference={byId.get(quote.document)} /></div>
            ))}
          </li>
        ))}</ul>
      </section>}
      {unique.length > 0 && <section>
        <h3>Unique insights</h3>
        <ul>{unique.map((quote) => (
          <li key={quote.document}><q>{quote.text}</q> <DocumentChip reference={byId.get(quote.document)} /></li>
        ))}</ul>
      </section>}
      {clusters.length > 0 && <section>
        <h3>Related documents</h3>
        <ul>{clusters.map((cluster, index) => (
          <li key={index}><strong>{cluster.label}</strong> <div className="chips">{cluster.documents.map((id) => (
            <DocumentChip key={id} reference={byId.get(id)} />
          ))}</div></li>
        ))}</ul>
      </section>}
    </details>
  );
}

type MentionGroup = { name: string; pages: RelatedItem[] };

function RelatedResearch({ state, unplaced }: { state?: IntelligenceState; unplaced: MentionGroup[] }) {
  if (!state || state.status === "error") return null;
  if (state.status === "loading") {
    return <p className="related-loading" aria-busy="true">Comparing sources and finding related pages…</p>;
  }
  const analysis = state.data?.analysis;
  return (
    <div className="related">
      {unplaced.length > 0 && <p className="read-more"><Compass size={14} aria-hidden="true" /> Read more:{" "}
        {unplaced.map((group) => <RelatedMention key={group.name} label={group.name} pages={group.pages} />)}
      </p>}
      {analysis && <CrossDocumentAnalysis analysis={analysis} />}
    </div>
  );
}

export default function Home() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [current, setCurrent] = useState<Thread | null>(null);
  const [mode, setMode] = useState<Mode>("web");
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [menu, setMenu] = useState(false);
  const [selectedSource, setSelectedSource] = useState<Source | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  // Related research is keyed by "<thread id>:<turn index>" and kept apart from the turns, so it can
  // arrive after the answer (even while the next question runs) without overwriting thread state.
  const [research, setResearch] = useState<Record<string, IntelligenceState>>({});
  useEffect(() => {
    try { setResearch(JSON.parse(localStorage.getItem("prama-research") || "{}")); } catch {}
  }, []);

  function loadResearch(key: string, body: object) {
    setResearch((all) => ({ ...all, [key]: { status: "loading" } }));
    fetch("/api/related", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) })
      .then(async (response) => {
        if (!response.ok) throw new Error("Related research is unavailable.");
        const data: Intelligence = await response.json();
        setResearch((all) => {
          const next = { ...all, [key]: { status: "done" as const, data } };
          const kept = Object.entries(next).filter(([, value]) => value.status === "done").slice(-30);
          try { localStorage.setItem("prama-research", JSON.stringify(Object.fromEntries(kept))); } catch {}
          return next;
        });
      })
      .catch(() => setResearch((all) => ({ ...all, [key]: { status: "error" } })));
  }
  const session = useRef("");
  const request = useRef<AbortController | null>(null);
  const composer = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const key = localStorage.getItem("prama-session") || localStorage.getItem("fieldwork-session") || crypto.randomUUID();
    localStorage.setItem("prama-session", key);
    session.current = key;
    const savedTheme = localStorage.getItem("prama-theme") as "light" | "dark" | null;
    const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const initial = savedTheme ?? (systemDark ? "dark" : "light");
    setTheme(initial);
    document.documentElement.dataset.theme = initial;
    try { setThreads(JSON.parse(localStorage.getItem("prama-threads") || localStorage.getItem("fieldwork-threads") || "[]")); } catch {}
    const controller = new AbortController();
    fetch("/api/search", { headers: { "x-session-id": key }, signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) return;
        const rows: SavedTurn[] = await response.json();
        const grouped = new Map<string, Thread>();
        for (const row of rows) {
          const thread = grouped.get(row.thread_id) || { id: row.thread_id, turns: [] };
          thread.turns.unshift(row);
          grouped.set(row.thread_id, thread);
        }
        setThreads((local) => [...local, ...Array.from(grouped.values()).filter((thread) => !local.some(({ id }) => id === thread.id))]);
      })
      .catch(() => {});
    return () => controller.abort();
  }, []);

  function open(thread: Thread | null) {
    if (request.current) return;
    setCurrent(thread);
    setQuestion("");
    setError("");
    setStatus("");
    setMenu(false);
    setSelectedSource(null);
    window.scrollTo(0, 0);
    composer.current?.focus();
  }

  async function submit(value: string) {
    const clean = value.trim();
    if (!clean || request.current || !session.current) return;
    const thread = current || { id: crypto.randomUUID(), turns: [] };
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setQuestion("");
    setError("");
    setStatus("Starting research");
    let turn: Turn = { question: clean, answer: "", sources: [] };
    let completed = false;
    const update = () => setCurrent({ ...thread, turns: [...thread.turns, turn] });
    update();
    try {
      const response = await fetch("/api/search?thread=" + thread.id, {
        method: "POST",
        signal: controller.signal,
        headers: { "content-type": "application/json", "x-session-id": session.current },
        body: JSON.stringify({ question: clean, mode, history: thread.turns.slice(-8).map(({ question, answer }) => ({ question, answer })) }),
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(typeof data.detail === "string" ? data.detail : "The request could not be processed.");
      }
      await readEvents(response, (kind, raw) => {
        const data = JSON.parse(raw);
        if (kind === "status") setStatus(data.message);
        if (kind === "sources") turn = { ...turn, sources: data };
        if (kind === "verification") turn = { ...turn, verification: data };
        if (kind === "token") turn = { ...turn, answer: turn.answer + data.text };
        if (kind === "error") throw new Error(data.message);
        if (kind === "done") completed = true;
        update();
      });
      if (!completed) throw new Error("The stream ended early. Try again.");
      if (mode === "web" && turn.sources.length > 0) {
        // Runs after the answer is complete and never blocks the next question.
        loadResearch(thread.id + ":" + thread.turns.length, {
          question: clean,
          answer: turn.answer,
          sources: turn.sources.slice(0, 8).map(({ url, title, content }) => ({ url, title, content: content.slice(0, 8000) })),
        });
      }
    } catch (failure) {
      setError(controller.signal.aborted ? "Answer stopped." : failure instanceof Error ? failure.message : "Something went wrong.");
      if (!turn.answer) {
        setCurrent(thread.turns.length ? thread : null);
        setQuestion(clean);
      }
    } finally {
      if (completed || turn.answer) {
        const updated = { ...thread, turns: [...thread.turns, turn] };
        setThreads((existing) => {
          const next = [updated, ...existing.filter(({ id }) => id !== updated.id)].slice(0, 30);
          try { localStorage.setItem("prama-threads", JSON.stringify(next)); } catch {}
          return next;
        });
      }
      request.current = null;
      setBusy(false);
      setStatus((message) => message.startsWith("Supabase") ? message : "");
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void submit(question);
  }

  return (
    <div className="shell">
      <aside className={"rail " + (menu ? "rail-open" : "")}>
        <button className="brand" onClick={() => open(null)}><PramaMark className="brand-mark" /> Prama</button>
        <Button className="new-search" onClick={() => open(null)} disabled={busy}><Plus size={17} /> New search</Button>
        <div className="history-label"><History size={16} /> Recent searches</div>
        <nav aria-label="Recent searches">
          {threads.length === 0 && <p className="history-empty">Your searches will appear here.</p>}
          {threads.map((thread) => (
            <button key={thread.id} className={"history-item " + (current?.id === thread.id ? "active" : "")} onClick={() => open(thread)} disabled={busy}>
              <MessageSquare size={15} /><span>{thread.turns[0].question}</span>
            </button>
          ))}
        </nav>
      </aside>
      <main>
        <header className="top">
          <Button className="mobile-menu" aria-label={menu ? "Close menu" : "Open menu"} onClick={() => setMenu(!menu)}>{menu ? <X size={20} /> : <Menu size={20} />}</Button>
          <Button className="theme-toggle" aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"} onClick={() => {
            const next = theme === "light" ? "dark" : "light";
            setTheme(next);
            localStorage.setItem("prama-theme", next);
            document.documentElement.dataset.theme = next;
          }}>{theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}</Button>
          <span><BookOpen size={16} /> Research workspace</span>
        </header>
        <div className={current ? "workspace conversation" : "workspace welcome"}>
          {!current && <div className="intro"><div className="intro-icon"><PramaMark /></div><h1>Ask. Verify. Know.</h1><p>Answers you can check, grounded in sources you can open.</p></div>}
          {current?.turns.map((turn, index) => (
            <article className="turn" key={index}>
              {index === 0 ? <h1>{turn.question}</h1> : <h2>{turn.question}</h2>}
              <div className="turn-layout">
                <div className="turn-main">
                  {(() => {
                    const state = research[current.id + ":" + index];
                    const groups = state?.status === "done" ? groupByEntity(state.data?.items ?? []) : [];
                    const markdown = answerMarkdown(turn, groups.map((group) => group.name));
                    const unplaced = groups.filter((_, i) => !markdown.includes(`(#related-${i})`));
                    return <>
                      {turn.answer && <div className="answer"><Markdown remarkPlugins={[remarkGfm]} disallowedElements={["img"]} components={{
                        a: ({ children, href }) => {
                          const label = String(children);
                          const related = href?.match(/^#related-(\d+)$/);
                          if (related && groups[Number(related[1])]) {
                            return <RelatedMention label={label} pages={groups[Number(related[1])].pages} />;
                          }
                          if (/^\d+$/.test(label)) {
                            const number = Number(label);
                            return <CitationChip number={number} source={turn.sources[number - 1]} />;
                          }
                          return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
                        },
                      }}>{markdown}</Markdown></div>}
                      <RelatedResearch state={state} unplaced={unplaced} />
                    </>;
                  })()}
                </div>
                {turn.sources.length > 0 && <aside className="source-rail" aria-label="Sources">
                  <div className="source-heading">
                    <BookOpen size={16} /> Sources <span className="source-count">{turn.sources.length}</span>
                    {turn.verification && turn.verification.sentences > 0 && <span className="verified" title={`${turn.verification.cited} of ${turn.verification.sentences} sentences cited · ${turn.verification.repaired} repaired · ${turn.verification.removed} removed`}>
                      <Check size={13} /> Verified
                    </span>}
                  </div>
                  {turn.sources.map((source, sourceIndex) => (
                    <button key={source.url} className="source" onClick={() => setSelectedSource(source)}>
                      <small><span>{sourceIndex + 1}</span> {hostOf(source.url)}</small>
                      <strong>{source.title}</strong>
                      <p>{source.content.slice(0, 170)}{source.content.length > 170 ? "…" : ""}</p>
                    </button>
                  ))}
                </aside>}
              </div>
            </article>
          ))}
          <div className="feedback" aria-live="polite">{error ? <p role="alert">{error}</p> : status && <p>{status}{busy ? "…" : ""}</p>}</div>
          <form className="composer" onSubmit={onSubmit}>
            <textarea ref={composer} aria-label="Question" value={question} maxLength={2000} rows={2}
              placeholder={current ? "Ask a follow-up…" : "What do you want to find out?"}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit(question); } }} />
            <div className="composer-actions">
              <div className="mode" aria-label="Answer mode">
                <Button type="button" aria-pressed={mode === "web"} onClick={() => setMode("web")}><Globe2 size={16} /> Web search</Button>
                <Button type="button" aria-pressed={mode === "chat"} onClick={() => setMode("chat")}><MessageSquare size={16} /> Chat</Button>
              </div>
              {busy ? <Button type="button" className="send" variant="primary" aria-label="Stop answer" onClick={() => request.current?.abort()}><Square size={17} /></Button>
                : <Button type="submit" className="send" variant="primary" aria-label="Send question" disabled={!question.trim()}><Send size={17} /></Button>}
            </div>
          </form>
          {!current && <section className="suggestions" aria-label="Suggested questions">
            {prompts.map(([topic, prompt]) => <button key={topic} onClick={() => { setQuestion(prompt); composer.current?.focus(); }}><span>{topic}</span>{prompt}</button>)}
          </section>}
        </div>
      </main>
      {selectedSource && <div className="source-panel" role="region" aria-label="Source preview">
        <Button className="close" aria-label="Close source preview" onClick={() => setSelectedSource(null)}><X size={20} /></Button>
        <small>{hostOf(selectedSource.url)}</small>
        <h2>{selectedSource.title}</h2>
        <p>{selectedSource.content}</p>
        <a href={selectedSource.url} target="_blank" rel="noopener noreferrer">Open source <ExternalLink size={15} /></a>
      </div>}
    </div>
  );
}
