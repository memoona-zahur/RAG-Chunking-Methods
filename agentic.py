"""The chunkless/agentic side: no chunks at all.

Navigation units are whole paragraphs (chunkers.chunk_agentic keeps them
untouched). The agent scans paragraphs, picks the most relevant, reads the
chosen ones IN FULL - so no fact can ever be severed by a boundary. We model
its work as: 1 navigation/selection call + 1 answer call per question.
"""

from __future__ import annotations

import chunkers
import embed_store as es
import llm


def navigate(document_text: str, query: str, k_units: int = 2) -> tuple[list[str], dict]:
    """Return (chosen full paragraphs, metadata). Two modeling calls on Groq."""
    import numpy as np

    units = chunkers.chunk_agentic(document_text)
    texts = [u.text for u in units]
    vecs = np.asarray(es.embed([query] + texts))
    qv, unit_vecs = vecs[0], vecs[1:]
    scored = sorted(
        ((float(qv @ u_vec), u) for u_vec, u in zip(unit_vecs, units)),
        key=lambda t: t[0],
        reverse=True,
    )
    best = [u.text for _, u in scored[:k_units]]
    return best, {"calls": 2, "units_considered": len(units), "chosen": k_units}


def answer(document_text: str, question: str, k_units: int = 2) -> dict:
    """Full chunkless pipeline: navigate -> read whole paragraphs -> answer."""
    chosen, nav = navigate(document_text, question, k_units=k_units)
    context = "\n\n---\n\n".join(chosen)
    messages = [
        {
            "role": "system",
            "content": "Answer ONLY from the provided manual text. Cite the sentences you use.",
        },
        {"role": "user", "content": f"Manual excerpts:\n\n{context}\n\nQuestion: {question}"},
    ]
    res = llm.ask(messages)
    return {
        "question": question,
        "chosen_units": nav,
        "context_tokens": es_est_tokens(chosen),
        "reply": res["reply"],
        "backend": res["backend"],
        "model": res["model"],
        "input_tokens": res["input_tokens"],
        "output_tokens": res["output_tokens"],
    }


def es_est_tokens(texts: list[str]) -> int:
    from evaluate import est_tokens

    return est_tokens(texts)