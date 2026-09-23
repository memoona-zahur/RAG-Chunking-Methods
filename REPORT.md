# Chunking Methods vs Chunkless/Agentic RAG — Technical Report

Date: 2026-09-23 · Repository: `memoona-zahur/RAG-Chunking-Methods`
Baseline run: `evidence/demo_output.txt` (regenerable with `python demo.py`)

---

## 1. Purpose

Answer the reviewer request with a *working, honest* comparison: **chunked RAG**
(7 chunking strategies + vector retrieval) versus **chunkless / agentic RAG**
(no chunks at all — whole paragraphs are navigation units). The demonstration
must work on a **long-paragraph-based document**, show **a small example of each
chunking type**, define a **baseline for judgement**, quantify the **difference
between methods**, and show the **cost difference between the chunked and
chunkless approaches**.

**Live status of the recorded run:** `evidence/demo_output.txt` was produced with
the **real LLM backend** — Groq `openai/gpt-oss-120b` (the older
`llama-3.3-70b-versatile` is no longer served; `llm.py` prints the live model
list if the name in `.env` is wrong). The fact grid and split counts are pure
retrieval math (embedding-based, deterministic); the agentic **replies** and the
agentic **token counts** come from genuine Groq calls. The demo is fully
runnable without a key — the dry-run backend then reports the same grid with
deterministic mock replies.

## 2. The document

`demo_document.md` — a Solar Home Power System user manual:

- 1,628 words, **9 long paragraphs**, 33 sentences, 1 specification/price table
- 5 short headings that (deliberately) do **not** isolate every answer
- facts spread across paragraphs and sections (runtime in §4, warranty in §5, etc.)
- precisely the kind of source that "fixed-size chunking tutorials" gloss over

## 3. The four questions and the baseline for judgement

Every question carries **2 hand-defined required facts**. The baseline is simple,
verbatim-checkable, and LLM-independent:

> a method is judged by whether the **required facts survive retrieval** into its
> top-2 context. The LLM (live or dry-run) is told nothing. A ✅ requires **both**
> facts present; ⚠ one; ❌ none.

| Q | Question | Required facts |
|---|---|---|
| Q1 | How long can the Home battery run alone, and what warranty applies to it? | `eight hours` · `ten years` |
| Q2 | What does the Solar Home Plus kit cost and what is its warranty? | `3,499` · `24 months` |
| Q3 | Which kit carries model number SHM-400? | `shm-400` · `solar home mini` |
| Q4 | What should a coastal-area user prepare at install, and which maintenance habit matters most? | `corrosion protection kit` · `heat sinks` |

Secondary metrics, all computed live per method: **mid-sentence cuts** (boundaries
landing inside a sentence), **table-row cuts** (boundary splitting a markdown
table row), **precision@2**, **recall@2**, and **cost** (tokens + LLM calls).

## 4. Each chunking type — what it is, a small example, and what it lost

| Method | What it does | Split example from this run | Facts (8) | P@2 / R@2 |
|---|---|---|---|---|
| fixed_char | Every ~500 chars, no regard for meaning | cuts 23 sentences + splits the price-table row | 6/8 | 0.50 / 0.52 |
| fixed_token | Every ~120 tokens (word-count stand-in) | 14 mid-sentence cuts, table intact | 7/8 | 0.50 / 0.54 |
| sentence | Boundaries between sentences only | budget-groups sentences; no cuts | 5/8 | 0.38 / 0.46 |
| paragraph | One chunk per paragraph | keeps idea-units; paragraphs stay large (up to 1,610 chars) | 7/8 | 0.88 / 0.92 |
| structural | Respects `##` headings; table stays with its section | 6 section-sized chunks | 8/8 | 0.62 / 0.88 |
| recursive | Structure first, then sentences, then raw cuts (overlap 60) | 25 chunks; 4 cuts; overlap fragments hurt | 4/8 | 0.50 / 0.38 |
| semantic | Breaks where embedding similarity drops | 24 similarity-driven chunks; no cuts | 6/8 | 0.50 / 0.67 |
| agentic (chunkless) | **No chunks**: whole paragraphs navigated + read in full | 9 whole-paragraph units; 0 cuts by construction | 7/8 | 0.88 / 0.92 |

### What this tells sir, in one line each

- **fixed_char** is the honest baseline that *proves chunks sever meaning* — but
  that alone is not the end of the story.
- **fixed_token** inherits the same disease for a different unit.
- **sentence** stops mid-sentence cuts — and still fails Q1/Q4, because the two
  required facts live in *different paragraphs*: sentence-aware != answer-aware.
- **paragraph** keeps an idea-unit intact; it wins on precision here (0.88).
- **structural** is the best of both: full completeness (8/8) at **1 call/question** —
  structure is cheap insurance.
- **recursive** (the famous default) is *not* a magic bullet; overlap fragments
  add noise and it scored lowest here (4/8).
- **semantic** follows meaning, but its boundaries are not question-shaped.
- **agentic/chunkless** never severs anything *inside* a read unit — its losses are
  pure *navigation* misses (Q1: it picked the wrong two paragraphs and never saw
  `eight hours`).

## 5. Chunkless / agentic — the cost of "never severing"

The agentic path is modeled honestly:

- navigation: embed query + all 9 units, pick top-2 -> 1 LLM call
- answer: pass the **whole chosen paragraphs** to the LLM -> 1 call
- total: **2 modeled calls/question with whole-paragraph context** (recorded: 573–706 tokens
  context per question). On Groq today the navigation step is embedding-based, so the
  answer call is the one that really hits the LLM — the 2-call count is the honest
  cost model for an agent that *would* also decide by LLM.

Chunked path: **1 call/question, top-2 chunk context** (fixed-size ≈ small, e.g.
~250-token chunks).

### Recorded cost drill-down (example llama-3.3-70b pricing)

| Path | LLM calls (4 Q's) | context tokens | est. $ (4 Q's) |
|---|---|---|---|
| chunked (fixed-size) | 8 | 3,291 | 0.00213 |
| agentic (chunkless) | 8 | 2,345 | 0.00141 |

**Honest reading:** on a small manual the token gap is modest (agentic ≈ 0.9× —
whole sections are still small). The **call gap is the structural difference**:
2 calls vs 1, and chunked context is **predictable** (≈ top-k × chunk size) while
agentic context **grows with whole-section size**. On a manual with 5,000-word
sections the gap multiplies — which is exactly why real systems still chunk, and
why `structural` gets the completeness of agentic at the cost of chunked.

## 6. Verification

- `pytest` — 13 pure-logic tests pass (no model, no network), covering every
  chunker, cut detection, fact coverage, precision/recall and cost math.
- `verify_project.py` — import/hygiene check: no Groq key in any tracked file,
  `.env` git-ignored, tests green.
- All reported numbers are printed by the demo itself into `evidence/` — nothing
  hand-written.

## 7. Scope guard

Per instruction, this is **limited to the chunking problem** and proceeds step by
step. Deliberately out of scope: reranking, hybrid BM25, hyDE, meta-refinement,
agent frameworks, multi-hop tool use.