"""The chunkless/agentic side: no chunks at all.

Navigation units are whole paragraphs (chunkers.chunk_agentic keeps them
untouched). The agent scans the paragraphs, picks the ones worth reading, and
reads them IN FULL - so no fact can ever be severed by a boundary. Its work is
TWO REAL LLM CALLS per question: a navigation call that picks the paragraphs,
then an answer call that reads the chosen ones completely.

Offline (no key), navigation degrades to the embedding-based ranking and the
answer to the deterministic dry-run; the call structure stays the same.
"""

from __future__ import annotations

import re

import chunkers
import embed_store as es
import llm

NAV_SYSTEM_PROMPT = (
    "You are a document navigator for a retrieval engine. Below is a numbered list "
    "of whole paragraphs from a manual (paragraphs start at 1). Reply with ONLY the "
    "comma-separated numbers of the two paragraphs you would READ IN FULL to answer "
    "the question. Do not explain."
)
_UINT = re.compile(r"\d+")


def parse_navigation(reply: str, n_units: int, pick: int = 2) -> list[int]:
    """Extract up to ``pick`` valid, unique 1..n_units indices from the LLM reply."""
    idxs: list[int] = []
    for m in _UINT.findall(reply or ""):
        v = int(m)
        if 1 <= v <= n_units and v not in idxs:
            idxs.append(v)
            if len(idxs) >= pick:
                break
    return idxs


def navigate(document_text: str, query: str, k_units: int = 2) -> tuple[list[str], dict]:
    """Return (chosen full paragraphs, metadata). One real navigation LLM call
    (embedding-ranking fallback when offline)."""
    import numpy as np

    units = chunkers.chunk_agentic(document_text)
    texts = [u.text for u in units]
    vecs = np.asarray(es.embed([query] + texts))
    qv, unit_vecs = vecs[0], vecs[1:]
    scored = sorted(
        ((float(qv @ u_vec), i) for i, u_vec in enumerate(unit_vecs)),
        key=lambda t: t[0],
        reverse=True,
    )

    via, picked, nav_res = "knn (embedding)", [], None
    if llm.backend() == "groq":
        numbered = "\n\n".join(f"{i + 1}. {t[:900]}" for i, t in enumerate(texts))
        nav_res = llm.ask_robust(
            [
                {"role": "system", "content": NAV_SYSTEM_PROMPT},
                {"role": "user", "content": f"Paragraphs:\n{numbered}\n\nQuestion: {query}"},
            ]
        )
        picked = parse_navigation(nav_res["reply"], len(units), k_units)
        if picked:
            via = "llm navigation"
            chosen_idx = [p - 1 for p in picked]
        else:
            chosen_idx = None
    else:
        chosen_idx = None

    if chosen_idx is None:
        chosen_idx = [i for _, i in scored[:k_units]]
        via = "knn (embedding)"

    chosen = [units[i].text for i in chosen_idx]
    return chosen, {
        "calls": 2,
        "units_considered": len(units),
        "chosen": k_units,
        "via": via,
        "nav_tokens_in": (nav_res["input_tokens"] if nav_res else 0),
        "nav_tokens_out": (nav_res["output_tokens"] if nav_res else 0),
    }


def answer(document_text: str, question: str, k_units: int = 2) -> dict:
    """Full chunkless pipeline: navigate -> read whole paragraphs -> answer.

    Both calls are real when a Groq key is present; the reported token counts
    are the SUM of the navigation and answer calls (the true per-question bill).
    """
    chosen, nav = navigate(document_text, question, k_units=k_units)
    context = "\n\n---\n\n".join(chosen)
    messages = [
        {
            "role": "system",
            "content": "Answer ONLY from the provided manual text. Cite the sentences you use.",
        },
        {"role": "user", "content": f"Manual excerpts:\n\n{context}\n\nQuestion: {question}"},
    ]
    res = llm.ask_robust(messages)
    return {
        "question": question,
        "chosen_units": nav,
        "context_tokens": es_est_tokens(chosen),
        "reply": res["reply"],
        "backend": res["backend"],
        "model": res["model"],
        "input_tokens": res["input_tokens"] + nav["nav_tokens_in"],
        "output_tokens": res["output_tokens"] + nav["nav_tokens_out"],
    }


def es_est_tokens(texts: list[str]) -> int:
    from evaluate import est_tokens

    return est_tokens(texts)