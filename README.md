# RAG Chunking — a Head-to-Head of 8 Strategies on One Long Document

A self-contained, runnable comparison of chunking methods for Retrieval-Augmented
Generation, built around **one long, paragraph-based document** and **four
fact-checkable questions**. It includes the usual suspects (fixed-size char,
fixed-size token, sentence, paragraph, structural, recursive, semantic) and the
**chunkless / agentic** far end, so the "to chunk or not to chunk" decision is
shown with real numbers.

Everything is computed live — there are no hand-written results anywhere.

## What this demo proves

| Takeaway | Proven by |
|---|---|
| Fixed-size splitting severs sentences and table rows | `fixed_char` cuts 23 sentences + 1 table row in this document |
| Structure-aware chunking recovers completeness at one small call | `structural` finds 8/8 required facts, all 4 questions complete |
| Blind recursive overlap is not a magic bullet | `recursive` finds only 4/8 required facts here |
| Chunkless/agentic never severs a fact — but it bills like a taxi (2 real calls/question, ≈4.2× the price) | agentic section + cost drill-down: real navigation call + real answer call, every token from the model's usage object |
| Chunked context is predictable (top-k × chunk size); agentic context grows with section size | cost drill-down at the end of the run: one chunked method vs the chunkless path, per question |

## Run it

```bash
pip install -r requirements.txt
python demo.py
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
call that picks the paragraphs + the answer call), and the cost table's tokens come
straight from the model's usage object. Without a key, the demo still runs
end-to-end with a deterministic dry-run backend that reports the required facts
present in the retrieved context, so the mechanics — chunk counts, cuts, and every
P/R/HR score — are the same; only the token-dollar columns and the agent's
navigation choice differ (the offline agent falls back to embedding ranking).

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

Each of the 4 questions carries **2 hand-defined required facts**. A method only
gets ✅ if **both** facts survive retrieval into its top-2 context — the LLM is
told nothing. Fine-grained scores: fact coverage, precision@2, recall@2,
hit-rate@2 (did the top-2 contain *any* relevant unit — the agentic-side metric),
plus mid-sentence cuts and table-row cuts produced by the method itself.

## Honest limits — read before you generalize

- One manual, four questions, one embedding model: a micro-benchmark, not a law.
- A production decision needs your own document, question set, and retrieval tuning (reranking, hybrid BM25).
- `$` figures use example pricing — verify current rates for `openai/gpt-oss-120b` in the Groq console.
- Deliberately out of scope: hyDE, hybrid BM25, meta-refinement, agent frameworks.