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
| Chunkless/agentic never severs a fact — but it reads whole sections at 2 calls/question | agentic section: whole paragraphs read in full, 2 modeled calls each |
| Chunked context is predictable (top-k × chunk size); agentic context grows with section size | cost drill-down at the end of the run |

## Run it

```bash
pip install -r requirements.txt
python demo.py
```

Your terminal prints the colored walkthrough; the terminal session is the demo.
The `evidence/demo_output.txt` file in this repo is the recorded run.

### Get live LLM answers (optional)

Create `.env` (copy of `.env.example`):

```dotenv
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

No key → the demo still runs end-to-end with a deterministic dry-run backend
that reports the required facts present in the retrieved context, so the
mechanics — and every number — are identical either way.

## What's inside

```
demo.py             the one-command walkthrough (THE artifact)
demo_document.md    the long Solar Home manual (1628 words, 9 paragraphs, 1 table)
chunkers.py         8 chunking strategies (incl. the chunkless far end)
embed_store.py      all-MiniLM-L6-v2 + in-memory Qdrant
evaluate.py         the baseline: required facts, precision@k, recall@k
agentic.py          chunkless navigation: pick whole paragraphs, read them in full
llm.py              hybrid: Groq when key present, dry-run otherwise
costs.py            token/call/cost accounting
tests/              pure-logic tests (no model, no network)
evidence/           recorded run output
verify_project.py   sanity + hygiene check before you commit/share
```

## The evaluation baseline (how we judge)

Each of the 4 questions carries **2 hand-defined required facts**. A method only
gets ✅ if **both** facts survive retrieval into its top-2 context — the LLM is
told nothing. Fine-grained scores: fact coverage, precision@2, recall@2, plus
mid-sentence cuts and table-row cuts produced by the method itself.

## Honest limits — read before you generalize

- One manual, four questions, one embedding model: a micro-benchmark, not a law.
- A production decision needs your own document, question set, and retrieval tuning (reranking, hybrid BM25).
- `$` figures use example llama-3.3-70b pricing — verify current rates in the Groq console.
- Deliberately out of scope: hyDE, hybrid BM25, meta-refinement, agent frameworks.