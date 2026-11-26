"""Answer-aware related content and cross-document analysis.

After an answer is written, find more web pages about the specific things it discusses, then
compare those pages with the answer's own sources: which findings several sources share, where
their figures differ, and what only one source adds. Every finding is a quote from a real page,
so it stays traceable. Similar wording means two passages discuss the same thing, not that they
agree. Uses the existing Jina search, BGE embedder and cross-encoder; no extra LLM calls.
"""

import asyncio
import logging
import re
from collections import Counter
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

import research
from research import (CITATION_RE, STOPWORDS, JinaAuthError, build_blocks, chunk_text, clean_text, lexical_similarity,
                      normalize_url, token_set)

log = logging.getLogger(__name__)

MAX_QUERIES = 3
MAX_ITEMS = 5
MAX_CANDIDATES = 12
MAX_ENTITY_QUERY_WORDS = 6
SENTENCES_PER_DOCUMENT = 15

# Thresholds for BGE cosine similarity, with the lexical fallback in brackets. BGE-small scores
# sit in a narrow band (about 0.6-0.9 for on-topic text), so these are not probabilities.
RELEVANT = (0.6, 0.15)  # a passage or sentence is about something the answer says
SAME_CLAIM = (0.88, 0.5)  # two sentences make nearly the same statement
POSSIBLE_CONFLICT = (0.85, 0.4)  # close enough to compare their figures
NEGATION_CONFLICT = (0.9, 0.55)  # close enough that a negation is meaningful
UNIQUE_BELOW = (0.7, 0.25)  # no other document has a sentence this close
UNIQUE_RELEVANT = (0.75, 0.3)  # and the sentence is closely about what the answer says
SIMILAR_DOCUMENTS = (0.9, 0.35)  # two documents cover the same ground (also must stand out from the rest)
DUPLICATE_PAGE = 0.92  # two pages whose best passages are this close repeat each other
STRONG_MATCH = 0.8

QUERY_FILLER = {"best", "top", "good", "great", "latest", "new", "list", "ways", "tips", "guide", "should", "could"}
NEGATIONS = {"not", "no", "never", "cannot", "can't", "doesn't", "don't", "isn't", "aren't", "won't", "without"}
LIST_HEAD_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(?:\*\*)?([^:\n–—*]{3,60}?)(?:\*\*)?\s*[:–—]", re.M)
BOLD_RE = re.compile(r"\*\*([^*\n]{3,80})\*\*")
PROPER_RE = re.compile(r"\b[A-Z][\w&+.-]*(?:\s+(?:of|for|and|&)?\s*[A-Z][\w&+.-]*)+")
TABLE_ROW_RE = re.compile(r"^\s*\|\s*([^|\n]{3,60}?)\s*\|")
TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an|this|these|those|its|their)\s+", re.I)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[^a-z\s])")
NUMBER_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
MEASURE_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(%|°\s?[cf]|[a-z$€£]+)", re.I)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
BARE_URL_RE = re.compile(r"https?://\S+")
MARKDOWN_MARK_RE = re.compile(r"(?:^|(?<=\s))#{1,6}\s+|\*\*|__|`")
UNIT_FILLER = {"and", "or", "to", "of", "the", "a", "in", "on", "for", "per", "with", "more", "than", "up", "at"}


class SourceDocument(BaseModel):
    url: str = Field(max_length=2000)
    title: str = Field(default="", max_length=300)
    content: str = Field(default="", max_length=8000)


class RelatedRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=40000)
    sources: list[SourceDocument] = Field(default_factory=list, max_length=8)
    entities: list[str] = Field(default_factory=list, max_length=8)


# --- what the answer is about ------------------------------------------------------------------


def answer_sentences(answer):
    """The answer's own statements, without citations, Markdown, notes, or lead-in lines."""
    blocks, _ = build_blocks(answer)
    sentences = []
    for block in blocks:
        if block["type"] != "prose":
            continue
        for sentence in block["sentences"]:
            if sentence.lstrip().startswith(">"):
                continue
            text = clean_text(CITATION_RE.sub("", sentence).replace("**", ""))
            text = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", text))
            if text.endswith(":") or len(re.findall(r"[A-Za-z0-9']+", text)) < 5:
                continue
            sentences.append(text)
    return sentences[:24]


def table_first_cells(text):
    lines = text.splitlines()
    cells = []
    for index, line in enumerate(lines):
        match = TABLE_ROW_RE.match(line)
        is_header = index + 1 < len(lines) and TABLE_RULE_RE.match(lines[index + 1])
        if match and not TABLE_RULE_RE.match(line) and not is_header:
            cells.append(match.group(1))
    return cells


def entities_from_answer(answer):
    """Names the answer emphasises: list-item heads, bold text, table rows, repeated proper nouns."""
    text = CITATION_RE.sub("", answer)
    found = {}

    def add(name, weight):
        name = LEADING_ARTICLE_RE.sub("", clean_text(name.replace("**", "")).strip(" .,:;*_`\"'()"))
        if not 3 <= len(name) <= 80 or name.endswith("?") or not token_set(name):
            return
        # Names start with a capital or digit, are short, and are not measurements or phrases.
        if not re.match(r"[A-Z0-9]", name) or len(query_name(name).split()) > 5 or re.search(r"[%°]", name):
            return
        if sum(char.isdigit() for char in name) > len(name) / 3:
            return
        key = name.lower()
        label, total, order = found.get(key, (name, 0, len(found)))
        found[key] = (label, total + weight, order)

    for match in LIST_HEAD_RE.finditer(text):
        add(match.group(1), 3)
    for match in BOLD_RE.finditer(text):
        add(match.group(1), 2)
    for cell in table_first_cells(text):
        add(cell, 2)
    for match in PROPER_RE.finditer(text):
        add(match.group(0), 1)
    ranked = sorted(found.values(), key=lambda item: (-item[1], item[2]))
    return [label for label, total, _ in ranked if total >= 2][:6]


def query_name(entity):
    """A search-friendly form of an entity: no brackets or subtitle, at most six words."""
    name = re.sub(r"\([^)]*\)", " ", entity)
    name = re.split(r"\s[:–—|]\s|:\s", name)[0]
    words = clean_text(re.sub(r"[^\w\s&+.'-]", " ", name)).split()
    return " ".join(words[:MAX_ENTITY_QUERY_WORDS])


def context_words(question, entity):
    """Up to two topic words from the question, so an entity query stays on the question's topic."""
    taken = {word.rstrip("s") for word in token_set(entity)}
    words = []
    for word in re.findall(r"[a-z0-9']+", question.lower()):
        if (len(word) > 2 and word not in STOPWORDS and word not in QUERY_FILLER and word.rstrip("s") not in taken
                and word not in words and not re.fullmatch(r"(?:19|20)\d\d", word)):
            words.append(word)
    return words[:2]


def related_queries(question, entities):
    names = [name for name in (query_name(entity) for entity in entities) if name]
    candidates = [" ".join([name, *context_words(question, name)]) for name in names[:2]]
    if len(names) >= 2:
        candidates.append(f"{names[0]} vs {names[1]}")
    queries = []
    for value in candidates:
        # Never repeat the question itself or a query already planned here.
        if value.lower() in {query.lower() for query in queries} or lexical_similarity(value, question) >= 0.8:
            continue
        queries.append(value)
    return queries[:MAX_QUERIES]


def mentions(entity, text_lower):
    name = query_name(entity).lower() or entity.lower()
    if name in text_lower or entity.lower() in text_lower:
        return True
    tokens = token_set(name)
    return len(tokens) >= 2 and all(token in text_lower for token in tokens)


# --- similarity helpers --------------------------------------------------------------------------


async def embed(texts):
    """Normalised BGE vectors (numpy) for texts, or None when the model is unavailable."""
    if not texts:
        return None
    try:
        async with research.model_lock:
            model = await asyncio.to_thread(research.embedder)
            return await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)
    except Exception:
        log.warning("Embeddings unavailable; using lexical overlap")
        return None


def word_overlap(left, right):
    """Symmetric lexical similarity (Jaccard) for when embeddings are unavailable."""
    a, b = token_set(left), token_set(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def similarity_matrix(left_texts, right_texts, left_vectors=None, right_vectors=None):
    if left_vectors is not None and right_vectors is not None:
        return (left_vectors @ right_vectors.T).tolist()
    return [[word_overlap(left, right) for right in right_texts] for left in left_texts]


def pick(thresholds, embedded):
    return thresholds[0] if embedded else thresholds[1]


def plain_text(text):
    """Page text without Markdown links, images, headings, emphasis or bare URLs."""
    text = MARKDOWN_IMAGE_RE.sub(" ", str(text or ""))
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = MARKDOWN_MARK_RE.sub(" ", BARE_URL_RE.sub(" ", text))
    return clean_text(text.replace("[]", " "))


def split_sentences(text):
    sentences = []
    for part in SENTENCE_SPLIT_RE.split(plain_text(text)):
        part = re.sub(r"^[^\w\"“(]+", "", part.strip())
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", part)
        letters = sum(char.isalpha() for char in part)
        # Keep readable statements; drop navigation text, link lists, table rows and fragments
        # that start mid-sentence (passages are cut at fixed lengths).
        if (8 <= len(words) <= 70 and letters >= 0.6 * len(part) and not part.endswith(":")
                and re.match(r"[A-Z0-9\"“(]", part) and part.count("|") < 2 and not re.search(r"\d\.\s+\d\.", part)):
            sentences.append(part)
    return sentences


def excerpt_for(passage, sentence, limit=320):
    passage = plain_text(passage)
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", passage) if part.strip()]
    if not parts:
        return passage[:limit]
    start = max(range(len(parts)), key=lambda index: lexical_similarity(sentence, parts[index]))
    text = parts[start]
    for extra in parts[start + 1:]:
        if len(text) + len(extra) + 1 > limit:
            break
        text += " " + extra
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def domain_of(url):
    return urlsplit(str(url)).netloc.removeprefix("www.")


def title_key(title):
    return " ".join(sorted(token_set(title)))


# --- related content -------------------------------------------------------------------------------


async def find_related(client, request, entities):
    """Pages about the answer's entities that the answer did not already cite.

    Returns (items, queries, documents) where documents carry the passages used for analysis.
    """
    queries = related_queries(request.question, entities)
    if not queries:
        return [], queries, []
    results = await asyncio.gather(*(research.cached_search(client, value) for value in queries),
                                   return_exceptions=True)
    excluded = {normalize_url(source.url) for source in request.sources}
    seen_titles = {title_key(source.title) for source in request.sources if source.title}
    candidates, failures = {}, []
    for result in results:
        if isinstance(result, Exception):
            log.warning("Related search failed: %s", type(result).__name__)
            failures.append(result)
            continue
        for source in result:
            key, title = normalize_url(source.url), title_key(source.title)
            if key in excluded or key in candidates or (title and title in seen_titles):
                continue
            text = (source.title + " " + source.content).lower()
            matched = [entity for entity in entities if mentions(entity, text)]
            if matched:  # the entity gate keeps out pages that only share broad keywords
                candidates[key] = (source, matched)
                seen_titles.add(title)
    if not candidates:
        for failure in failures:
            if isinstance(failure, JinaAuthError) or (
                    isinstance(failure, httpx.HTTPStatusError) and failure.response.status_code in (401, 403, 429)):
                raise failure
        return [], queries, []

    docs = list(candidates.values())[:MAX_CANDIDATES]
    sentences = answer_sentences(request.answer) or [request.question]
    passages = []
    for index, (source, matched) in enumerate(docs):
        parts = chunk_text(source.content, 1000, 150)
        focused = [part for part in parts if any(mentions(entity, part.lower()) for entity in matched)]
        for part in (focused + [part for part in parts if part not in focused])[:4]:
            passages.append((index, part))

    sentence_vectors = await embed(sentences)
    passage_vectors = await embed([part for _, part in passages]) if sentence_vectors is not None else None
    embedded = passage_vectors is not None
    if embedded:
        similarity = similarity_matrix([], [], passage_vectors, sentence_vectors)
    else:  # share of the answer sentence's words that the passage contains
        similarity = [[lexical_similarity(sentence, part) for sentence in sentences] for _, part in passages]

    best, by_doc = {}, {}
    for row, (doc, _) in enumerate(passages):
        column = max(range(len(sentences)), key=lambda col: similarity[row][col])
        score = float(similarity[row][column])
        by_doc.setdefault(doc, []).append((score, row))
        if score > best.get(doc, (-1.0, 0, 0))[0]:
            best[doc] = (score, row, column)
    ranked = [(doc, *values) for doc, values in best.items() if values[0] >= pick(RELEVANT, embedded)]
    order = [score for _, score, _, _ in ranked]
    if research.settings.enable_reranking and ranked:
        try:
            async with research.model_lock:
                model = await asyncio.to_thread(research.reranker)
                scores = await asyncio.to_thread(
                    model.predict, [(sentences[column], passages[row][1]) for _, _, row, column in ranked])
            order = [float(score) for score in scores]
        except Exception:
            log.warning("Cross-encoder reranking unavailable for related content")
    ranked = [item for _, item in sorted(zip(order, ranked), key=lambda pair: pair[0], reverse=True)]

    items, documents, chosen_rows = [], [], []
    for doc, score, row, column in ranked:
        if embedded and any(float(passage_vectors[row] @ passage_vectors[other]) >= DUPLICATE_PAGE
                            for other in chosen_rows):
            continue
        source, matched = docs[doc]
        chosen_rows.append(row)
        items.append({
            "url": str(source.url),
            "title": source.title,
            "domain": domain_of(source.url),
            "published": source.published,
            "description": source.description,
            "excerpt": excerpt_for(passages[row][1], sentences[column]),
            "entities": matched[:3],
            "answer_sentence": sentences[column][:240],
            "match": "strong" if embedded and score >= STRONG_MATCH else "related",
            "score": round(score, 3),
        })
        top_passages = [passages[r][1] for _, r in sorted(by_doc[doc], reverse=True)[:2]]
        documents.append({"url": str(source.url), "title": source.title, "text": "\n".join(top_passages)})
        if len(items) >= MAX_ITEMS:
            break
    return items, queries, documents


# --- cross-document analysis -----------------------------------------------------------------------


def numbers_in(sentence):
    values = set()
    for raw in NUMBER_RE.findall(sentence):
        value = raw.replace(",", "")
        if re.fullmatch(r"(?:19|20)\d\d", value):  # years differ for many harmless reasons
            continue
        values.add(value.rstrip("0").rstrip(".") if "." in value else value)
    return values


def negated(sentence):
    return bool(NEGATIONS & set(re.findall(r"[a-z']+", sentence.lower())))


def measures(sentence):
    """{unit: {numbers}} for figures followed by a unit, e.g. "405 miles" or "-15°F"."""
    found = {}
    for number, unit in MEASURE_RE.findall(sentence):
        unit = re.sub(r"\s", "", unit.lower())
        value = number.replace(",", "")
        if unit in UNIT_FILLER or re.fullmatch(r"(?:19|20)\d\d", value):
            continue
        value = value.rstrip("0").rstrip(".") if "." in value else value
        found.setdefault(unit.rstrip("s") if len(unit) > 3 else unit, set()).add(value)
    return found


def differs(left, right):
    """Why two closely related sentences may disagree, or None.

    Only flags the same unit with no figure in common ("405 miles" vs "412 miles"), or a negation
    on otherwise near-identical wording. This is a prompt to check the sources, not a verdict.
    """
    left_measures, right_measures = measures(left), measures(right)
    for unit in left_measures.keys() & right_measures.keys():
        if not left_measures[unit] & right_measures[unit]:
            return "different figures"
    shared = token_set(left) & token_set(right)
    if negated(left) != negated(right) and len(shared) >= 0.6 * min(len(token_set(left)), len(token_set(right))):
        return "one source negates the other"
    return None


def cluster_label(members, documents, entities):
    """The first answer entity that every document in the group mentions, else a neutral label."""
    counts = Counter()
    for index in members:
        text = (documents[index]["title"] + " " + documents[index]["text"]).lower()
        counts.update(entity for entity in entities if mentions(entity, text))
    shared = [entity for entity in entities if counts[entity] == len(members)]
    return shared[0] if shared else "Similar coverage"


async def analyze_documents(documents, answer_text, entities):
    """Compare documents sentence by sentence.

    documents: [{"id", "label", "url", "title", "text"}]. Returns common findings (one quote plus
    every source that says nearly the same), possible conflicts (close statements with different
    figures or a negation), unique insights, groups of similar documents and the similarity matrix.
    """
    empty = {"documents": [], "similarity": [], "clusters": [], "common_findings": [], "conflicts": [],
             "unique_insights": []}
    if len(documents) < 2:
        return empty
    targets = answer_sentences(answer_text) or [answer_text[:500]]
    per_doc = [split_sentences(doc["text"])[:SENTENCES_PER_DOCUMENT] for doc in documents]
    flat = [(index, sentence) for index, sentences in enumerate(per_doc) for sentence in sentences]
    if not flat:
        return empty
    texts = [sentence for _, sentence in flat]

    vectors = await embed(texts + targets)
    embedded = vectors is not None
    sentence_vectors = vectors[: len(texts)] if embedded else None
    target_vectors = vectors[len(texts):] if embedded else None
    relevance = [max(row) for row in similarity_matrix(texts, targets, sentence_vectors, target_vectors)]

    # Keep each document's on-topic sentences, most relevant first.
    keep = sorted((i for i in range(len(flat)) if relevance[i] >= pick(RELEVANT, embedded)),
                  key=lambda i: relevance[i], reverse=True)
    kept = {}
    for i in keep:
        if len(kept.setdefault(flat[i][0], [])) < 8:
            kept[flat[i][0]].append(i)
    rows = [i for doc in sorted(kept) for i in kept[doc]]
    if not rows:
        return empty
    pair = similarity_matrix([texts[i] for i in rows], [texts[i] for i in rows],
                             sentence_vectors[rows] if embedded else None, sentence_vectors[rows] if embedded else None)
    position = {i: p for p, i in enumerate(rows)}

    def sim(a, b):
        return float(pair[position[a]][position[b]])

    def quote(i):
        return {"document": documents[flat[i][0]]["id"], "text": texts[i]}

    # Common findings: greedy grouping of near-identical statements from different documents.
    groups = []
    for i in sorted(rows, key=lambda r: relevance[r], reverse=True):
        for group in groups:
            # Same wording with different figures is a possible conflict, not a shared finding.
            if sim(i, group[0]) >= pick(SAME_CLAIM, embedded) and not differs(texts[i], texts[group[0]]):
                if flat[i][0] not in {flat[g][0] for g in group}:
                    group.append(i)
                break
        else:
            groups.append([i])
    common = [group for group in groups if len(group) >= 2]
    common_findings = [{"text": texts[group[0]], "sources": [quote(i) for i in group]} for group in common[:5]]

    # Possible conflicts: near-identical statements about the same entity, from different
    # documents, that give different figures for the same unit (or where one negates the other).
    conflicts, used = [], set()
    candidates = sorted(((sim(a, b), a, b) for a in rows for b in rows
                         if a < b and flat[a][0] != flat[b][0]), reverse=True)
    for score, a, b in candidates:
        if score < pick(POSSIBLE_CONFLICT, embedded) or len(conflicts) >= 3:
            break
        if a in used or b in used or not any(
                mentions(entity, texts[a].lower()) and mentions(entity, texts[b].lower()) for entity in entities):
            continue
        reason = differs(texts[a], texts[b])
        if reason == "one source negates the other" and score < pick(NEGATION_CONFLICT, embedded):
            reason = None
        if reason:
            conflicts.append({"reason": reason, "claims": [quote(a), quote(b)]})
            used.update((a, b))

    # Unique insights: a document's most relevant statement that no other document comes close to.
    unique, in_common = [], {i for group in common for i in group}
    for doc in sorted(kept):
        for i in kept[doc]:
            if relevance[i] < pick(UNIQUE_RELEVANT, embedded) or re.search(r"\s[*•]\s", texts[i]):
                continue
            others = [sim(i, j) for j in rows if flat[j][0] != doc]
            if i not in in_common and i not in used and (not others or max(others) < pick(UNIQUE_BELOW, embedded)):
                unique.append(quote(i))
                break
    unique_insights = unique[:4]

    # Document similarity matrix and groups of documents that cover the same ground.
    doc_ids = sorted(kept)
    if embedded:
        import numpy as np

        doc_vectors = np.stack([sentence_vectors[kept[doc]].mean(axis=0) for doc in doc_ids])
        doc_vectors = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        matrix = similarity_matrix([], [], doc_vectors, doc_vectors)
    else:
        joined = [" ".join(texts[i] for i in kept[doc]) for doc in doc_ids]
        matrix = similarity_matrix(joined, joined)
    parent = list(range(len(doc_ids)))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    # Pages on one topic all look alike on average, so a pair must also stand out from the rest:
    # at least one standard deviation above the mean pairwise similarity.
    off_diagonal = [matrix[a][b] for a in range(len(doc_ids)) for b in range(a + 1, len(doc_ids))]
    if off_diagonal:
        mean = sum(off_diagonal) / len(off_diagonal)
        spread = (sum((value - mean) ** 2 for value in off_diagonal) / len(off_diagonal)) ** 0.5
        threshold = max(pick(SIMILAR_DOCUMENTS, embedded), mean + spread)
        for a in range(len(doc_ids)):
            for b in range(a + 1, len(doc_ids)):
                if matrix[a][b] >= threshold:
                    parent[root(a)] = root(b)
    grouped = {}
    for index, doc in enumerate(doc_ids):
        grouped.setdefault(root(index), []).append(doc)
    clusters = [{"label": cluster_label(members, documents, entities),
                 "documents": [documents[m]["id"] for m in members]}
                for members in grouped.values()
                if 2 <= len(members) <= max(2, len(doc_ids) // 2)][:4]  # a group of nearly everything says nothing

    return {
        "documents": [{key: documents[doc][key] for key in ("id", "label", "url", "title")} for doc in doc_ids],
        "similarity": [[round(float(value), 2) for value in row] for row in matrix],
        "clusters": clusters,
        "common_findings": common_findings,
        "conflicts": conflicts,
        "unique_insights": unique_insights,
    }


async def research_intelligence(client, request):
    """Related pages for an answer plus a cross-document analysis of them and the answer's sources."""
    entities = [clean_text(entity)[:100] for entity in request.entities if clean_text(entity)]
    entities = entities or entities_from_answer(request.answer)
    items, queries = [], []
    documents = [{"id": f"S{index}", "label": f"[{index}]", "url": source.url, "title": source.title,
                  "text": source.content} for index, source in enumerate(request.sources, 1) if source.content]
    if entities:
        items, queries, related_docs = await find_related(client, request, entities)
        documents += [{"id": f"R{index}", "label": f"R{index}", **doc} for index, doc in enumerate(related_docs, 1)]
    analysis = await analyze_documents(documents, request.answer, entities)
    return {"entities": entities, "queries": queries, "items": items, "analysis": analysis}
