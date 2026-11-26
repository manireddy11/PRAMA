"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { BookOpen, Check, ExternalLink, Globe2, History, Menu, MessageSquare, Moon, Plus, Send, Square, Sun, X } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";

type Source = { url: string; title: string; content: string };
type Verification = { sentences: number; cited: number; repaired: number; removed: number; sources: number };
type Turn = { question: string; answer: string; sources: Source[]; verification?: Verification; followups?: string[] };
type Thread = { id: string; turns: Turn[] };
type SavedTurn = Turn & { thread_id: string };
type Mode = "web" | "chat";

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

function answerMarkdown(turn: Turn) {
  return turn.answer.replace(/\[(\d+)\](?!\()/g, (citation, number) => {
    const source = turn.sources[Number(number) - 1];
    return source ? citation + "(" + source.url.replaceAll("(", "%28").replaceAll(")", "%29") + ")" : citation;
  });
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
  const session = useRef("");
  const request = useRef<AbortController | null>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const [suggestionIndex, setSuggestionIndex] = useState(0);
  const [composerFocused, setComposerFocused] = useState(false);
  const latest = current?.turns[current.turns.length - 1];
  const followups = busy ? [] : latest?.followups ?? [];
  const followupKey = followups.join("\n");
  const suggestion = followups.length ? followups[suggestionIndex % followups.length] : "";

  // Show one follow-up at a time as placeholder text. Rotate slowly, and only while the
  // composer is idle, so the hint never changes while someone is reading or typing.
  useEffect(() => setSuggestionIndex(0), [followupKey]);
  useEffect(() => {
    if (followups.length < 2 || composerFocused || question) return;
    const timer = setInterval(() => setSuggestionIndex((index) => index + 1), 12000);
    return () => clearInterval(timer);
  }, [followupKey, followups.length, composerFocused, question]);

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
        if (kind === "followups") turn = { ...turn, followups: data };
        if (kind === "token") turn = { ...turn, answer: turn.answer + data.text };
        if (kind === "error") throw new Error(data.message);
        if (kind === "done") completed = true;
        update();
      });
      if (!completed) throw new Error("The stream ended early. Try again.");
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
                  {turn.answer && <div className="answer"><Markdown remarkPlugins={[remarkGfm]} disallowedElements={["img"]} components={{
                    a: ({ children, href }) => {
                      const label = String(children);
                      if (/^\d+$/.test(label)) {
                        const number = Number(label);
                        return <CitationChip number={number} source={turn.sources[number - 1]} />;
                      }
                      return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
                    },
                  }}>{answerMarkdown(turn)}</Markdown></div>}
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
              placeholder={suggestion || (current ? "Ask a follow-up…" : "What do you want to find out?")}
              onChange={(event) => setQuestion(event.target.value)}
              onFocus={() => setComposerFocused(true)}
              onBlur={() => setComposerFocused(false)}
              onKeyDown={(event) => {
                if (event.key === "Tab" && !event.shiftKey && suggestion && !question) { event.preventDefault(); setQuestion(suggestion); return; }
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit(question); }
              }} />
            <div className="composer-actions">
              <div className="mode" aria-label="Answer mode">
                <Button type="button" aria-pressed={mode === "web"} onClick={() => setMode("web")}><Globe2 size={16} /> Web search</Button>
                <Button type="button" aria-pressed={mode === "chat"} onClick={() => setMode("chat")}><MessageSquare size={16} /> Chat</Button>
              </div>
              {suggestion && !question && <span className="suggestion-hint" aria-hidden="true"><kbd>Tab</kbd> to use suggestion</span>}
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
