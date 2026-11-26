import asyncio
import re
import zlib

import httpx
import numpy as np
import pytest

import intelligence
import research
from intelligence import (RelatedRequest, analyze_documents, answer_sentences, differs, entities_from_answer,
                          find_related, measures, numbers_in, query_name, related_queries, research_intelligence,
                          split_sentences)
from research import JinaAuthError, Source


class BagOfWords:
    """Deterministic stand-in for BGE: hashed word counts, L2-normalised."""

    def encode(self, texts, normalize_embeddings=True):
        vectors = np.zeros((len(texts), 512))
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vectors[row, zlib.crc32(word.encode()) % 512] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


@pytest.fixture
def embeddings(monkeypatch):
    model = BagOfWords()
    monkeypatch.setattr(research, "embedder", lambda: model)


def run(coroutine):
    return asyncio.run(coroutine)


# --- reading the answer ------------------------------------------------------------------------


def test_answer_sentences_skip_notes_lead_ins_and_fragments():
    answer = ("Electric cars with the longest range include:\n"
              "- **Lucid Air** offers up to 512 miles of range [1].\n"
              "Short one [2].\n\n"
              "> Some claims could not be verified against the retrieved sources and were omitted.")
    assert answer_sentences(answer) == ["Lucid Air offers up to 512 miles of range."]


def test_entities_from_list_heads_bold_tables_and_repeated_names():
    answer = (
        "- **Deep Learning Specialization**: taught on Coursera [1].\n"
        "- IBM AI Engineering: covers PyTorch [2].\n\n"
        "| Course | Hours |\n|---|---|\n| Google AI Essentials | 10 |\n\n"
        "Andrew Ng teaches it. Andrew Ng also founded DeepLearning.AI."
    )
    entities = entities_from_answer(answer)
    assert entities[:2] == ["Deep Learning Specialization", "IBM AI Engineering"]
    assert "Google AI Essentials" in entities and "Andrew Ng" in entities
    assert "Course" not in entities  # table header
    assert "Coursera" not in entities  # mentioned once without emphasis


def test_query_name_drops_brackets_and_subtitles():
    assert query_name("Machine Learning Specialization (DeepLearning.AI)") == "Machine Learning Specialization"
    assert query_name("The AI Engineer Course 2026: Complete AI Engineer Bootcamp") == "The AI Engineer Course 2026"


def test_related_queries_focus_on_entities_and_never_repeat_the_question():
    queries = related_queries("What are the best AI engineering courses in 2026?",
                              ["Deep Learning Specialization", "IBM AI Engineering"])
    assert queries == ["Deep Learning Specialization engineering courses", "IBM AI Engineering courses",
                       "Deep Learning Specialization vs IBM AI Engineering"]
    assert related_queries("heat pumps cold climate", ["Heat pumps"]) == []


# --- related content -----------------------------------------------------------------------------

ANSWER = ("IBM AI Engineering covers PyTorch, Keras and deep learning projects [1]. "
          "The Deep Learning Specialization is taught by Andrew Ng on Coursera [2].")
ENTITIES = ["IBM AI Engineering", "Deep Learning Specialization"]


def page(name, content, title=None, **extra):
    return Source(url=f"https://{name}.example/page", title=title or name.title(), content=content, **extra)


SYLLABUS = page("syllabus", "The IBM AI Engineering certificate covers PyTorch, Keras and deep learning projects "
                            "in six courses with hands-on labs.", description="Official syllabus",
                published="2026-03-01")
REVIEW = page("review", "Review: the Deep Learning Specialization by Andrew Ng on Coursera is taught clearly.")
GENERIC = page("generic", "The future of artificial intelligence will change engineering and many industries.")


def related(monkeypatch, pages_by_query, sources=(), entities=ENTITIES):
    searched = []

    async def fake_search(client, query):
        searched.append(query)
        result = pages_by_query(query)
        if isinstance(result, Exception):
            raise result
        return [item.model_copy() for item in result]

    monkeypatch.setattr(research, "cached_search", fake_search)
    request = RelatedRequest(question="best AI engineering courses", answer=ANSWER, sources=list(sources))
    items, queries, documents = run(find_related(None, request, entities))
    return items, queries, documents, searched


def test_find_related_keeps_only_pages_about_answer_entities(monkeypatch):
    items, queries, documents, searched = related(monkeypatch, lambda query: [SYLLABUS, GENERIC, REVIEW])
    assert [item["url"] for item in items] == ["https://syllabus.example/page", "https://review.example/page"]
    first = items[0]
    assert first["entities"] == ["IBM AI Engineering"]
    assert first["answer_sentence"].startswith("IBM AI Engineering covers PyTorch")
    assert (first["description"], first["published"], first["domain"]) == ("Official syllabus", "2026-03-01",
                                                                           "syllabus.example")
    assert "PyTorch" in first["excerpt"]
    assert [doc["url"] for doc in documents] == [item["url"] for item in items]
    assert len(searched) == len(queries) == 3


def test_find_related_skips_answer_sources_and_duplicate_titles(monkeypatch):
    copy = page("mirror", SYLLABUS.content, title=SYLLABUS.title)
    items, *_ = related(monkeypatch, lambda query: [SYLLABUS, copy, REVIEW],
                        sources=[{"url": "https://review.example/page/", "title": "Review", "content": "x"}])
    assert [item["url"] for item in items] == ["https://syllabus.example/page"]


def test_find_related_drops_near_identical_pages(monkeypatch, embeddings):
    copy = page("mirror", SYLLABUS.content + " Enrol today.", title="IBM AI Engineering on another site")
    items, *_ = related(monkeypatch, lambda query: [SYLLABUS, copy])
    assert len(items) == 1


def test_find_related_raises_auth_errors_when_every_search_fails(monkeypatch):
    with pytest.raises(JinaAuthError):
        related(monkeypatch, lambda query: JinaAuthError("Jina Search needs JINA_API_KEY in backend/.env."))


def test_find_related_tolerates_one_failed_search(monkeypatch):
    items, *_ = related(monkeypatch, lambda query: httpx.ConnectError("down") if " vs " in query else [SYLLABUS])
    assert [item["url"] for item in items] == ["https://syllabus.example/page"]


# --- cross-document analysis ---------------------------------------------------------------------


def test_entities_reject_phrases_and_measurements():
    answer = ("- **moving heat from the outside air into the building**: the basic idea [1].\n"
              "- **‑25 °C (‑13 °F) or lower**: still works [2].\n"
              "- **Variable-speed compressors**: adjust output [3].")
    assert entities_from_answer(answer) == ["Variable-speed compressors"]


def test_plain_text_and_sentences_drop_markdown_and_fragments():
    text = ("ying a suitable level of low temperature performance and efficiency for every home. "
            "[← All posts](https://example.com/blog) ## **Heat pumps** work well in very cold weather today, "
            "according to [field tests](https://example.com/tests). 1. [](https://x.example) 2. [Categories](https://x.example)")
    assert split_sentences(text) == [
        "All posts Heat pumps work well in very cold weather today, according to field tests."]


def test_measures_pair_numbers_with_units():
    assert measures("Range is 405 miles (652 km), about 20% more, at -15°F, since 2019.") == {
        "mile": {"405"}, "km": {"652"}, "%": {"20"}, "°f": {"15"}}


def test_numbers_ignore_years_and_formatting():
    assert numbers_in("In 2026 the Lucid Air reached 5,120 miles and 3.50 kWh") == {"5120", "3.5"}


@pytest.mark.parametrize("left, right, expected", [
    ("The Lucid Air has an EPA range of 512 miles on one charge.",
     "The Lucid Air has an EPA range of 516 miles on one charge.", "different figures"),
    ("Heat pumps work well in very cold climates with modern compressors.",
     "Heat pumps do not work well in very cold climates with modern compressors.", "one source negates the other"),
    ("The Lucid Air has an EPA range of 512 miles on one charge.",
     "The Lucid Air has an EPA range of 512 miles, the best in 2025.", None),
    ("The Lucid Air has an EPA range of 512 miles on one charge.",
     "The Lucid Air charges at 300 kW at the fastest stations.", None),  # different units are not a conflict
])
def test_differs(left, right, expected):
    assert differs(left, right) == expected


ANALYSIS_ANSWER = (
    "The Lucid Air Grand Touring offers the longest EPA range of any electric sedan today [1]. "
    "The Lucid Air Grand Touring has an EPA range of 512 miles on a single charge [1]. "
    "Owners report that winter driving cuts the electric range of the Lucid Air Grand Touring [2]."
)
DOCUMENTS = [
    {"id": "S1", "label": "[1]", "url": "https://a.example", "title": "A",
     "text": "The Lucid Air Grand Touring offers the longest EPA range of any electric sedan on sale today. "
             "The Lucid Air Grand Touring has an EPA range of 512 miles on a single charge."},
    {"id": "R1", "label": "R1", "url": "https://b.example", "title": "B",
     "text": "The Lucid Air Grand Touring offers the longest EPA range of any electric sedan on sale now. "
             "The Lucid Air Grand Touring has an EPA range of 516 miles on a single charge."},
    {"id": "R2", "label": "R2", "url": "https://c.example", "title": "C",
     "text": "Owners of the Lucid Air Grand Touring report that winter driving cuts its electric range by about "
             "a fifth compared with the official rating."},
]


def test_analysis_finds_common_findings_conflicts_and_unique_insights(embeddings):
    analysis = run(analyze_documents(DOCUMENTS, ANALYSIS_ANSWER, ["Lucid Air Grand Touring"]))
    common = analysis["common_findings"]
    assert len(common) == 1
    assert {quote["document"] for quote in common[0]["sources"]} == {"S1", "R1"}
    assert all(quote["text"] in next(d["text"] for d in DOCUMENTS if d["id"] == quote["document"])
               for quote in common[0]["sources"])  # every quote is traceable to its document

    conflict = analysis["conflicts"][0]
    assert conflict["reason"] == "different figures"
    assert sorted(re.search(r"\d+", claim["text"]).group() for claim in conflict["claims"]) == ["512", "516"]

    assert [quote["document"] for quote in analysis["unique_insights"]] == ["R2"]

    matrix = analysis["similarity"]
    assert len(matrix) == len(analysis["documents"]) == 3
    assert all(matrix[i][i] == pytest.approx(1.0) and matrix[i][j] == matrix[j][i]
               for i in range(3) for j in range(3))
    assert analysis["clusters"] == [{"label": "Lucid Air Grand Touring", "documents": ["S1", "R1"]}]


def test_conflicts_need_an_entity_both_sentences_mention(embeddings):
    analysis = run(analyze_documents(DOCUMENTS, ANALYSIS_ANSWER, ["Tesla Model S"]))
    assert analysis["conflicts"] == []


def test_document_groups_never_swallow_most_documents(embeddings):
    copies = [{**DOCUMENTS[0], "id": f"R{index}", "label": f"R{index}"} for index in range(1, 5)]
    analysis = run(analyze_documents([DOCUMENTS[0], *copies], ANALYSIS_ANSWER, ["Lucid Air Grand Touring"]))
    assert analysis["clusters"] == []


def test_analysis_works_without_embeddings():
    analysis = run(analyze_documents(DOCUMENTS, ANALYSIS_ANSWER, ["Lucid Air Grand Touring"]))
    assert {quote["document"] for quote in analysis["common_findings"][0]["sources"]} == {"S1", "R1"}
    assert analysis["conflicts"][0]["reason"] == "different figures"


def test_analysis_needs_two_documents():
    analysis = run(analyze_documents(DOCUMENTS[:1], ANALYSIS_ANSWER, []))
    assert analysis["common_findings"] == [] and analysis["documents"] == []


def test_research_intelligence_labels_documents_for_traceability(monkeypatch, embeddings):
    async def fake_search(client, query):
        return [page("b", DOCUMENTS[1]["text"], title="Lucid Air review"),
                page("c", DOCUMENTS[2]["text"], title="Lucid Air winter range")]

    monkeypatch.setattr(research, "cached_search", fake_search)
    request = RelatedRequest(question="Which electric car has the longest range?", answer=ANALYSIS_ANSWER,
                             sources=[{"url": "https://a.example/page", "title": "A", "content": DOCUMENTS[0]["text"]}],
                             entities=["Lucid Air Grand Touring"])
    result = run(research_intelligence(None, request))
    assert result["entities"] == ["Lucid Air Grand Touring"]
    assert [item["url"] for item in result["items"]] == ["https://b.example/page", "https://c.example/page"]
    labels = {document["id"]: document["label"] for document in result["analysis"]["documents"]}
    assert labels == {"S1": "[1]", "R1": "R1", "R2": "R2"}
    finding_docs = {quote["document"] for quote in result["analysis"]["common_findings"][0]["sources"]}
    assert finding_docs == {"S1", "R1"}


def test_research_intelligence_without_entities_skips_web_search(monkeypatch):
    async def unexpected(client, query):
        raise AssertionError("no search without entities")

    monkeypatch.setattr(research, "cached_search", unexpected)
    request = RelatedRequest(question="why", answer="it depends on what you want to learn and how much time you have.")
    assert run(research_intelligence(None, request))["items"] == []


def test_intelligence_module_reuses_research_components():
    assert intelligence.research is research
