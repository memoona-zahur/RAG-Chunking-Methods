"""Evaluation baseline for the chunking comparison.

Each question carries hand-defined *required facts* (how we judge "completeness").
For every chunking method we embed + index its chunks, retrieve top-k, then score:

- fact coverage : which required facts appear in the retrieved context
- precision@k / recall@k against chunks that contain any required fact
- hit-rate@k (the agentic-side metric): did at least one relevant unit land in top-k?
- mid-sentence cuts & table cuts produced by the method itself
- tokens sent per question (chars/4 heuristic) and LLM calls

The chunkless/agentic path is scored separately (analyze_agentic): it is judged
on the paragraphs the agent ACTUALLY navigated to, not on a fresh embedding
search over the same units - otherwise it would score identically to paragraph
chunking and the grid would tell you nothing about the navigator.
"""

from __future__ import annotations

import math


def est_tokens(texts: list[str]) -> int:
    return sum(math.ceil(len(t) / 4) for t in texts)


def fact_coverage(context_texts: list[str], required_facts: list[str]) -> tuple[int, int]:
    hay = " ".join(context_texts).lower()
    found = [f for f in required_facts if f in hay]
    return len(found), len(required_facts)


def relevant_chunk_ids(chunks, required_facts: list[str]) -> set[int]:
    """A chunk is 'relevant' if it contains at least one required fact."""
    return {
        c.idx
        for c in chunks
        if any(f in c.text.lower() for f in required_facts)
    }


def precision_recall(retrieved: list[dict], relevant: set[int], k: int) -> tuple[float, float]:
    top = [h["idx"] for h in retrieved[:k]]
    hit = len(set(top) & relevant)
    p = hit / k if k else 0.0
    r = hit / len(relevant) if relevant else 0.0
    return round(p, 2), round(r, 2)


def hit_rate(retrieved: list[dict], relevant: set[int], k: int) -> float:
    """Binary "did we find ANYTHING relevant in the top-k" — the standard
    metric for agentic navigation, where only 'found or not' matters."""
    top = [h["idx"] for h in retrieved[:k]]
    return 1.0 if set(top) & relevant else 0.0


# (question, [required facts as lowercase substrings]).
# 14 questions × 2 facts = 28 required facts. Each fact exists verbatim in
# demo_document.md. Some questions pair facts inside ONE paragraph (Q5-Q14);
# others deliberately split facts across two paragraphs (Q1, Q2, Q4) so a method
# must retrieve both units to pass. Every pair was checked against the document.
QUESTIONS: list[tuple[str, list[str]]] = [
    (
        "How long can the Home battery run alone, and what warranty applies to it?",
        ["eight hours", "ten years"],
    ),
    (
        "What does the Solar Home Plus kit cost and what is its warranty?",
        ["3,499", "24 months"],
    ),
    (
        "Which kit carries model number SHM-400?",
        ["shm-400", "solar home mini"],
    ),
    (
        "What should a coastal-area user prepare at install, and which maintenance habit matters most?",
        ["corrosion protection kit", "heat sinks"],
    ),
    (
        "What is the entry-level kit called, and how large is its battery?",
        ["solar home mini", "5 kwh"],
    ),
    (
        "Which kit lets a larger home run a washing machine or a water pump in the evening?",
        ["washing machine", "water pump"],
    ),
    (
        "Where must the battery pack be installed, and how far from walls?",
        ["indoors", "fifty centimeters"],
    ),
    (
        "How should roof panels be aimed and tilted?",
        ["southern hemisphere", "ten degrees"],
    ),
    (
        "At what charge levels does the inverter stop charging and start feeding the home directly?",
        ["one hundred percent", "eighty percent"],
    ),
    (
        "How long is the battery pack rated, and what cycle count backs that rating?",
        ["ten years", "six thousand"],
    ),
    (
        "What does the coastal corrosion protection kit consist of?",
        ["marine-grade coating", "sealed junction box"],
    ),
    (
        "What must an owner upload to claim an in-warranty inverter fault, and how fast is service?",
        ["diagnostic summary", "seven days"],
    ),
    (
        "At a light draw of five hundred watts, how long does a fully charged ten kWh pack last?",
        ["sixteen hours", "five hundred watt"],
    ),
    (
        "What, besides dust, can cut panel production, and by how much?",
        ["bird droppings", "ten percent"],
    ),
]

N_FACTS = len(QUESTIONS) * 2  # total required facts any method must find

K = 2  # retrieval depth used across every method for a fair baseline


def analyze_method(
    method: str,
    chunks,
    build_store,
    search,
) -> list[dict]:
    """Run all questions through one method's chunks. Returns per-question rows."""
    rows: list[dict] = []
    store = build_store(chunks, method)
    for q, facts in QUESTIONS:
        hits = search(store, q, k=10)
        context = [h["text"] for h in hits[:K]]
        cov, total = fact_coverage(context, facts)
        relevant = relevant_chunk_ids(chunks, facts)
        p, r = precision_recall(hits, relevant, K)
        hr = hit_rate(hits, relevant, K)
        rows.append(
            {
                "method": method,
                "question": q,
                "facts": (cov, total),
                "precision": p,
                "recall": r,
                "hit_rate": hr,
                "tokens": est_tokens(context),
                "calls": 1,
            }
        )
    return rows


def analyze_agentic(
    units,
    chosen_idx_by_question: list[list[int]],
    questions: list[tuple[str, list[str]]] | None = None,
    k: int = K,
) -> list[dict]:
    """Score the chunkless path on the units the agent ACTUALLY navigated to.

    Unlike analyze_method, here the "retrieved" set is the LLM-navigated (or,
    offline, embedding-ranked) paragraphs the agent chose to read in full - not
    a fresh embedding search over the same units. That is what the reviewer
    could not see before: the agentic row now reflects the navigator's picks.

    ``units`` is chunkers.chunk_agentic(document_text); ``chosen_idx_by_question``
    holds the unit indices picked per question (agentic.py 'chosen_indices').
    """
    if questions is None:
        questions = QUESTIONS
    rows: list[dict] = []
    for (q, facts), chosen_idx in zip(questions, chosen_idx_by_question):
        chosen = [units[i].text for i in chosen_idx]
        cov, total = fact_coverage(chosen, facts)
        relevant = relevant_chunk_ids(units, facts)
        retrieved = [{"idx": i} for i in chosen_idx]
        p, r = precision_recall(retrieved, relevant, k)
        hr = hit_rate(retrieved, relevant, k)
        rows.append(
            {
                "method": "agentic",
                "question": q,
                "facts": (cov, total),
                "precision": p,
                "recall": r,
                "hit_rate": hr,
                "tokens": est_tokens(chosen),
                "calls": 2,
            }
        )
    return rows