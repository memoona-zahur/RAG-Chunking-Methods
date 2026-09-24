# RAG Chunking — a Head-to-Head of 8 Strategies on One Long Document

A self-contained, runnable comparison of chunking methods for Retrieval-Augmented
Generation, built around **one long, paragraph-based document** and **fourteen
fact-checkable questions** (28 required facts). It includes the usual suspects
(fixed-size char, fixed-size token, sentence, paragraph, structural, recursive,
semantic) and the **chunkless / agentic** far end, so the "to chunk or not to
chunk" decision is shown with real numbers.

Everything is computed live — there are no hand-written results anywhere.

## What this demo illustrates

> A 14-question, 1-document, 1-embedding-model *micro-benchmark* — it shows the
> *pattern* (structure wins, fixed-size leaks, agentic finds facts but bills for
> it); the exact numbers will shift on your own corpus. Caveats in
> [Honest limits](#honest-limits--read-before-you-generalize).

| Takeaway | Illustrated by |
|---|---|
| Fixed-size splitting severs sentences and table rows | `fixed_char` cuts 23 sentences + 1 table row in this document |
| Structure-aware chunking recovers completeness at one call per question | `structural` finds 28/28 required facts, all 14 questions complete |
| Structural's completeness isn't free — its top-2 context is big | the grid's **ctx** column: ~906 tokens/question vs ~242 for `fixed_char` — k=2 is not an equal-token bet |
| Blind recursive overlap is not a magic bullet | `recursive` finds only 21/28 required facts |
| Chunkless/agentic never severs a fact, and its LLM navigator reads the manual well — but bills like a taxi | the agentic row is scored on the paragraphs it **actually navigated to** (27/28, its only miss is Q1 — two facts in separate paragraphs), yet pays ~2.5× structural's price and 2× its calls |
| Chunked context is predictable; agentic context is what the navigator picks + a full-document scan | cost drill-down at the end of the run: real tokens straight from the model's usage object |

## Run it

```bash
# CPU-only torch FIRST — avoids the ~6 GB CUDA download (this demo never uses a GPU).
# Skip this line only if you have an NVIDIA GPU and want the CUDA build.
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt
python3 demo.py
```

In no time at all, the terminal prints the whole lesson. To *understand* the
strategies — read them one by one, with the live numbers — open
[`REPORT.md`](REPORT.md): it is this repo's **slide deck**, written to be read
end-to-end. The recorded transcript is `evidence/demo_output.txt` (UTF-8).

> Windows tip: for glyphs like ✂ and ✅ in the console, run in a UTF-8 terminal
> (VS Code / Windows Terminal). GitHub renders the committed transcript cleanly
> because it is valid UTF-8.

### Get live LLM answers (recommended)

Create `.env` (copy of `.env.example`):

```dotenv
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

With a key, the agentic side makes **two real LLM calls per question** (a navigation
call that picks the paragraphs + the answer call), the cost table's tokens come
straight from the model's usage object, and the grid's **agentic row is scored on
the paragraphs the agent navigated to** (not on a separate embedding search).
Without a key, the demo still runs end-to-end with a deterministic dry-run backend
that reports the required facts present in the retrieved context, so the mechanics
— chunk counts, cuts, and every P/R/HR score — are the same; only the token-dollar
columns differ, and the offline navigator falls back to embedding ranking.

## What's inside

```
demo.py             the one-command walkthrough (THE artifact)
demo_document.md    the long Solar Home manual (1628 words, 9 paragraphs, 1 table)
REPORT.md           the slide deck: every strategy explained + judged, read it end-to-end
chunkers.py         8 chunking strategies (incl. the chunkless far end)
embed_store.py      all-MiniLM-L6-v2 + in-memory Qdrant
evaluate.py         the baseline: required facts, precision@k, recall@k, hit-rate@k
agentic.py          chunkless agent: REAL navigation + answer calls, reads whole paragraphs
llm.py              hybrid: Groq when key present, deterministic dry-run otherwise
costs.py            token/call/cost accounting
tests/              pure-logic tests (no model, no network)
evidence/           recorded run output
verify_project.py   sanity + hygiene check before you commit/share
```

## The evaluation baseline (how we judge)

Each of the **14 questions** carries **2 hand-defined required facts** (= 28 facts).
A method only gets ✅ if **both** facts survive retrieval into its top-2 context —
the LLM is told nothing. Fine-grained scores: fact coverage, precision@2,
recall@2, hit-rate@2 (did the top-2 contain *any* relevant unit), plus
mid-sentence cuts, table-row cuts, and the **ctx** column — the approximate token
budget each method's top-2 spends, so you can see the ruler is *not* "equal
tokens". The chunkless/agentic row is judged on the paragraphs its navigator
activated, not on a fresh embedding search.

## Honest limits — read before you generalize

- One manual, fourteen questions, one embedding model: a micro-benchmark, not a law.
- A production decision needs your own document, question set, and retrieval tuning (reranking, hybrid BM25).
- `$` figures use example pricing — verify current rates for `openai/gpt-oss-120b` in the Groq console.
- Deliberately out of scope: hyDE, hybrid BM25, meta-refinement, agent frameworks.