"""Evaluation baseline for the chunking comparison.

Each question carries hand-defined *required facts* (how we judge "completeness").
For every chunking method we embed + index its chunks, retrieve top-k, then score:

- fact coverage : which required facts appear in the retrieved context
- precision@k / recall@k against chunks that contain any required fact
- hit-rate@k (the agentic-side metric): did at least one relevant unit land in top-k?
- mid-sentence cuts & table cuts produced by the method itself
- tokens sent per question (chars/4 heuristic) and LLM calls
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


# (question, [required facts as lowercase substrings])
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
]

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