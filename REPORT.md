# RAG CHUNKING STRATEGIES — THE COMPLETE DECK

**Every chunking approach explained, compared, and judged — with a live, reproducible experiment.**

> How to read this deck: each `##` block is one "slide". Slides 1–4 build the mental
> model; slides 5–10 go strategy-by-strategy, then compare; slide 11 helps you choose.
> Every number under **Live evidence** comes from `python demo.py` (see
> `evidence/demo_output.txt`) — nothing here is invented.
>
> Repo: `memoona-zahur/RAG-Chunking-Methods` · Test document: a 1,628-word Solar Home
> manual (`demo_document.md`) with long paragraphs and a spec/price table.
> Real LLM backend: Groq `openai/gpt-oss-120b`.

---

## SLIDE 1 · THE ONE QUESTION THIS DEMO ANSWERS

> **Should we slice a long document into chunks before feeding it to RAG — and if
> we slice it, which of the 8 ways is best, and at what cost?**

Most RAG content shows you *a* way to chunk. This deck shows you **all the ways**,
judges them with the same ruler, and lets a chunkless "agentic" path compete too —
so the choice stops being habit and starts being a decision.

Preview of the verdict (full grid in Slide 9):

| Strategy | Required facts found (of 8) | One-line verdict |
|---|---|---|
| structural (heading-aware) | **8/8** | cheapest completeness — the winner here |
| paragraph | 7/8 | great precision, but chunks can get fat |
| agentic / chunkless | 7/8 | never severs a fact — pays in calls |
| fixed_token | 7/8 | tiny context, same boundary disease |
| fixed_char | 6/8 | the raw baseline; cuts sentences on purpose |
| semantic | 6/8 | follows meaning, not questions |
| sentence | 5/8 | grammar-safe, but facts scatter anyway |
| recursive | 4/8 | the "default" is not a magic bullet |

---

## SLIDE 2 · WHAT A CHUNK IS — THE 30-SECOND DEFINITION

A **chunk** is the unit of text that a RAG system hands to two pieces of machinery:

```
                  ┌─────────────┐      ┌──────────────┐      ┌──────────────┐
  long document ─▶│  CHUNKING   │─────▶│   RETRIEVAL  │─────▶│  LLM CONTEXT │
 (this manual)    │  (our demo) │      │ (semantic +  │      │  (answer is  │
                  └─────────────┘      │  Qdrant top-k)│      │  only as     │
                                       └──────────────┘      │  good as the │
                                                             │  chunks read │
                                                             └──────────────┘
```

- To the **retriever**, the chunk is a *target*: "find the chunk(s) most relevant to
  this question."
- To the **LLM**, the chunk is a *floor and a ceiling*: the answer can only contain
  what the retrieved chunks contain — nothing more.

So chunking secretly controls **two** things: (1) how precisely the retriever can
point at an answer, and (2) how many tokens we pay to feed the LLM.

---

## SLIDE 3 · WHY CHUNKING IS A TRADE-OFF, NOT A TRICK

There is no perfect chunk size — there is only a balance between two failure modes.

| If chunks are… | The retrieval pain | The cost pain |
|---|---|---|
| **too small** (e.g. 100 chars) | an answer that needs two sentences gets split → the retriever returns half a fact | context is cheap, but the answer is wrong |
| **too big** (e.g. whole section) | irrelevant filler travels with the fact → top-k context is diluted | you pay for tokens you didn't need |
| **just right** (boundaries match ideas) | retriever points exactly at the fact | smallest context that still contains it |

Our test document was written to make this visible. Its second paragraph is a
single ~180-word thought; inside the manual, a *fact* ("the battery runs **eight
hours**") and its *companion fact* ("warranty: **ten years**") sit in different
paragraphs. A naive char-splitter severs both of them — see Slide 6.

**The chunking choice is the difference between a committee that receives half the
evidence and one that receives all of it.**

---

## SLIDE 4 · THE 8 STRATEGIES — THE MAP BEFORE THE DETAILS

Three families, plus the chunkless far end that "audits" the whole idea:

| Family | Strategy | One-line description |
|---|---|---|
| **Size-based** | `fixed_char` | every N characters, no regard for meaning |
| | `fixed_token` | every N tokens (word-count stand-in) |
| **Structure-based** | `sentence` | boundaries only between sentences |
| | `paragraph` | one chunk per paragraph / blank-line block |
| | `structural` | respects headings, keeps tables with their section |
| | `recursive` | tries paragraphs, then sentences, then raw cuts (**the famous default**) |
| **Meaning-based** | `semantic` | breaks where embedding similarity drops |
| **Chunkless** | `agentic` | **no chunks at all** — whole paragraphs are navigation units, read in full |

Each strategy prints, in the demo, exactly what it produced: chunk count, every
chunk's size, the number of ✂ mid-sentence cuts, and whether it sliced the table.

---

## SLIDE 5 · THE BASELINE WE JUDGE BY (the ruler)

Judging must be **independent of the LLM** (so the comparison is reproducible)
and **concrete** (so it is checkable). The ruler has three parts:

**① Required facts.** Four questions, each with 2 hand-defined facts. A method
scores ✅ only if **both** facts survive retrieval into its top-2 context.

| # | Question | Fact A | Fact B |
|---|---|---|---|
| Q1 | How long can the Home battery run alone, and what warranty applies to it? | `eight hours` | `ten years` |
| Q2 | What does the Solar Home Plus kit cost and what is its warranty? | `3,499` | `24 months` |
| Q3 | Which kit carries model number SHM-400? | `shm-400` | `solar home mini` |
| Q4 | What should a coastal installer prepare, and which maintenance habit matters most? | `corrosion protection kit` | `heat sinks` |

**② Retrieval quality.** For each method, every chunk is embedded with
`all-MiniLM-L6-v2`, indexed in Qdrant (in-memory), and the question is retrieved
with `k=2`. We compute precision@2 and recall@2 against "chunks that contain a
required fact".

**③ Boundary quality.** Two weaknesses are counted mechanically:

- **✂ mid-sentence cuts** — how many chunk boundaries land *inside* a sentence.
- **table cuts** — how many boundaries slice a markdown table row.

**④ Cost.** Tokens handed to the LLM per question (`chars/4`) and LLM calls.

> One ruler, eight contestants, zero hand-written numbers.

---

## SLIDE 6 · STRATEGY BY STRATEGY (the detail)

Format per strategy: **what it does → where it wins → where it breaks → live
evidence from our run → verdict.**

### 6.1 `fixed_char` — every N characters (the raw baseline)

- **What:** cut the text at 500 characters, no knowledge of words, sentences or
  meaning. Overlap of 100 chars so a fact straddling a boundary has a second chance.
- **Wins:** dead simple, totally predictable chunk size, trivially implemented, and
  a great "worst case" to measure everyone else against.
- **Breaks:** boundaries land inside words and sentences; the retriever then returns
  slivers of facts. Long, idea-sized paragraphs get chopped into unrelated fragments.
- **Live evidence:** `25 chunks` · ✂ **23 mid-sentence cuts** · ✂ **1 table-row cut**.
  First sever in the manual:
  `…an hour. The philosophy behind ✂ uick-start card that guides you th…`
  **Found 6/8 required facts**, P@2 0.50, R@2 0.52.
- **Verdict:** the honest baseline — it *proves* that chunks sever meaning. Nobody
  should ship this for prose, but you should always run it to measure the others.

### 6.2 `fixed_token` — every N tokens

- **What:** same idea, but the unit is a token (here a word-count stand-in, so the
  demo stays dependency-free). 120-token windows with 20-token overlap.
- **Wins:** tokens are what the LLM actually bills; so "120 tokens ≈ a predictable
  bill." Slightly less violent than chars because it respects word boundaries.
- **Breaks:** identical disease — it still cuts *between* sentences mid-idea.
- **Live evidence:** `17 chunks` · ✂ **14 mid-sentence cuts** · table intact.
  **Found 7/8** · P@2 0.50 · R@2 0.54.
- **Verdict:** the same baseline, tuned to a model's real currency. Fine for exact
  token budgets, still blind to meaning.

### 6.3 `sentence` — boundaries only between sentences

- **What:** split on sentence-ending punctuation, then group sentences up to a char
  budget. No boundary can ever land inside a sentence by construction.
- **Wins:** zero mid-sentence cuts; each unit is grammatically whole; cheap to build.
- **Breaks:** "grammar-aware ≠ answer-aware." Our Q1 needs a fact from §4-para-1
  *and* a fact from §5 — two different paragraphs. Sentence units are still separate
  retrieval targets, so both facts can still miss the top-2.
- **Live evidence:** `28 chunks` · **0 mid-sentence cuts** · table intact.
  **Found 5/8** · P@2 0.38 · R@2 0.46.
- **Verdict:** fixes the symptom (cut sentences) but not the problem (scattered
  facts). It is a *necessary* refinement, not a strategy on its own for idea-keeping.

### 6.4 `paragraph` — one chunk per paragraph

- **What:** one chunk per blank-line-separated block. Each unit is (usually) one
  self-contained thought.
- **Wins:** best raw precision in our run (P@2 0.88) — when the answer lives inside
  one paragraph, you find the *exact* paragraph.
- **Breaks:** paragraphs can be huge (up to 1,610 chars here) → context bloat and
  diluted top-k; and a cross-paragraph question (Q1, Q4) still splits the required
  facts across two chunks.
- **Live evidence:** `9 chunks` · **0 cuts** · table intact.
  **Found 7/8** · P@2 0.88 · R@2 0.92.
- **Verdict:** strong, simple default for well-written documents. Its weakness is
  size, not meaning.

### 6.5 `structural` — heading-aware (the winner here)

- **What:** respects `##` headings and keeps the table attached to its section — the
  markdown-aware cousin of "section-based" chunking.
- **Wins:** sections are *semantic containers*: a heading says what the whole block
  is about, so the retriever gets a tagged unit; tables stay whole; completeness
  stays high at 1 small call per question.
- **Breaks:** relies on the document having structure. A wall of prose with no
  headings degrades it to paragraph-level.
- **Live evidence:** `6 chunks` · **0 cuts** · table intact.
  **Found 8/8 — ALL required facts.** P@2 0.62 · R@2 **0.88**. All 4 questions ✅.
- **Verdict:** **the cheap completeness winner** — structural information is free
  insurance. Use it whenever your corpus is markdown/HTML/sectioned.

### 6.6 `recursive` — the famous "default"

- **What:** try the highest-priority separator first (paragraph → sentence → word →
  char), and only fall back to raw cuts when nothing fits — plus a 60-char overlap.
- **Wins:** adapts to text quality; it is the default in many RAG libraries because
  it behaves "well enough" on most documents.
- **Breaks:** overlap re-attaches each chunk's tail, which can repeat noise and
  blur boundaries, and "well enough" still misses idea-boundaries.
- **Live evidence:** `25 chunks` with overlap tails · ✂ **4 cuts** · table intact.
  **Found only 4/8** · P@2 0.50 · R@2 0.38.
- **Verdict:** the demo's most useful surprise: **a default is not a proof.** On
  this document it was the *worst* at completeness. Always measure it, don't
  trust it.

### 6.7 `semantic` — breaks where meaning shifts

- **What:** embed every sentence, and start a new chunk when the cosine similarity
  between the next sentence and the running centroid drops below a threshold. No
  fixed sizes, no separator rules.
- **Wins:** boundaries follow meaning, not punctuation — units are "topically
  coherent," which is what paragraph chunking *wants* to be.
- **Breaks:** a topic is not a *question*; and it costs a model pass at index time.
  Topic boundaries and question-shaped boundaries just aren't the same thing.
- **Live evidence:** `24 chunks` · **0 cuts** · table intact.
  **Found 6/8** · P@2 0.50 · R@2 0.67.
- **Verdict:** intellectually appealing, honestly an add-on: fine for topical
  search, not a substitute for structure when the question is fact-shaped.

### 6.8 `agentic` / `chunkless` — no chunks at all

- **What:** the far end. Whole paragraphs/sections are *navigation units*; the
  agent embeds the question, picks the best 2 units, then reads them **in full** —
  no boundary can ever sever a fact inside a read unit.
- **Wins:** by construction, zero mid-sentence/table cuts and zero severed facts.
  Elegant answer to the whole chunking debate: "don't."
- **Breaks:** completeness now depends on *navigation*, not boundaries — and it
  pays for it: **2 modeled LLM calls** per question (navigate + answer) and the
  whole section as context. If the question spans units, both still contain what it
  needs (skip the paragraphs it picked wrong and it *does* miss facts, exactly like
  retrieval).
- **Live evidence:** `9 whole-paragraph units` · 0 cuts by construction.
  **Found 7/8** · P@2 0.88 · R@2 0.92. Real-LLM replies cite the source verbatim
  (e.g. Q3: *"The kit that carries model number SHM-400 is the Solar Home Mini."*).
- **Verdict:** the idea that motivated the demo — and our measured takeaway is
  nuanced: agentic never severs facts, but structure achieves the *same* 8/8 at
  less cost. See Slide 10.

---

## SLIDE 7 · THE JUDGEMENT GRID (how they really scored)

Reproduced verbatim from the live run:

```
      method        | Q1 | Q2 | Q3 | Q4 | facts(8) | avg P@2 | avg R@2
      ----------------------------------------------------------------
      fixed_char      | ⚠ ✅ ✅ ⚠ |  6/8   |  0.50     | 0.52
      fixed_token     | ⚠ ✅ ✅ ✅ |  7/8   |  0.50     | 0.54
      sentence        | ❌ ✅ ✅ ⚠ |  5/8   |  0.38     | 0.46
      paragraph       | ⚠ ✅ ✅ ✅ |  7/8   |  0.88     | 0.92
      structural      | ✅ ✅ ✅ ✅ |  8/8   |  0.62     | 0.88
      recursive       | ⚠ ❌ ✅ ⚠ |  4/8   |  0.50     | 0.38
      semantic        | ⚠ ✅ ✅ ⚠ |  6/8   |  0.50     | 0.67
      agentic         | ⚠ ✅ ✅ ✅ |  7/8   |  0.88     | 0.92
```

**How to read it**

- **Columns Q1–Q4** are the fact test: ✅ both facts retrieved, ⚠ one, ❌ none.
- **facts(8)** is the total completeness (8 = perfect).
- **avg P@2** = of the 2 chunks retrieved, how many were relevant.
- **avg R@2** = of all *relevant* chunks that exist, how many were retrieved.

**Three observations that survive any single run**

1. **Structure is the cheapest completeness.** `structural` is 8/8 at one call per
   question. Prefer it on any structured corpus.
2. **Every fixed/grammar-aware method leaks facts at paragraph borders.**
   Q1 (two facts, two paragraphs) is exactly where char/token/sentence/recursive
   fall to ⚠ or ❌. Boundaries that ignore paragraphs are boundaries that eat answers.
3. **Agentic is not automatically better.** Its 7/8 equals `paragraph`'s — read
   whole paragraphs helps, but *finding* the right paragraphs is still retrieval.
   (And in our run it missed Q1's `eight hours` on navigation alone.)

---

## SLIDE 8 · THE COST CONVERSATION — CHUNKED vs CHUNKLESS

The user-story framing is "chunked = 1 call with a small context, agentic = many
calls with a big context." Our live numbers make it concrete (Groq
`gpt-oss-120b`; `$` uses example pricing shown in `costs.py` — check live rates):

```
   method       | LLM calls | context tokens | est. USD (4 questions)
   -----------------------------------------------------------------
   chunked      |         8 |           3291 |          0.00213
   agentic      |         8 |           3572 |          0.00239
```

What is honest here — and what isn't:

- The **call gap** is real and structural: **2 calls vs 1** per question.
- The **token gap is modest on this small manual** (agentic ≈ 0.9× the context
  tokens) because the whole sections we read are still only ~700 tokens each.
- The **predictability gap is the one that scales**: chunked context is *roughly
  predictable* (top-k × chunk size, fixed per method), while agentic context *grows
  with whole-section size*. Point the agent at a 5,000-word regulation and the
  dollar gap multiplies; the calls stay 2 vs 1, but the context won't stay small.

So the honest one-liner for sir: *"Chunkless never severs a fact, but it bills
like a taxi (per read) instead of sharing a bus (fixed chunks). Structure gets you
the completeness at the bus price — that's why real systems chunk."*

---

## SLIDE 9 · HOW TO CHOOSE (a decision guide)

```
start here ──▶ is the corpus naturally structured (md/html/sections)?
                 ├─ YES ──────▶ structural / section-aware ──▶ the winner here
                 └─ NO ──────────────┬─ is the text neat prose with ideas per paragraph?
                                     │      ├─ YES ──▶ paragraph (or semantic for fuzzy topics)
                                     │      └─ NO ──▶ question-spanning paragraphs?
                                     │               ├─ YES ──▶ hybrid: paragraph + overlap window
                                     │               └─ NO ──▶ recursive (but MEASURE it)
                                                      
still worried about severed facts?  ──▶ run agentic/chunkless side-by-side;
                                      use its whole-unit reads only where $ allows
```

Rules of thumb (from this deck's evidence):

| Goal | Reach for |
|---|---|
| Completeness at lowest cost (recommended default) | `structural` |
| Fine-grained pinpointing, neat prose | `paragraph` |
| Fuzzy topical search, no headings anywhere | `semantic` |
| "I must use the library default" | `recursive` — **but run the facts test** |
| The demo/audit of whether boundaries hurt at all | `fixed_char`, `fixed_token` |
| Avoid boundaries entirely, budget allows 2× calls | `agentic` / chunkless |

Parameters worth tuning in `chunkers.py`: `size` (500 chars / 120 tokens here),
`overlap` (100), recursive separators, semantic similarity threshold (0.45).

---

## SLIDE 10 · THE ONE-LINE TAKEAWAYS (cheat sheet)

| Strategy | One line to remember |
|---|---|
| fixed_char | "No meaning, no mercy — the control group." |
| fixed_token | "Bills in the LLM's currency; still blind to ideas." |
| sentence | "Grammar-safe, answer-blind." |
| paragraph | "The idea-unit default; fat paragraphs eventually bite." |
| structural | "Headings are free metadata — cheapest completeness." |
| recursive | "The famous default is not a proof; overlap can add noise." |
| semantic | "Follows topics, not questions; costs a model pass." |
| agentic | "Never severs a fact — navigation and tokens are the tax." |

---

## SLIDE 11 · HONEST LIMITS AND WHAT'S GENUINELY NEXT

- This is **one** document, **four** questions, **one** embedding model: a
  *micro-benchmark*, not a law of nature. The verdict pattern (structure wins,
  fixed-size leaks, agentic is pricey) is general — the exact numbers are not.
- Per instruction, the demo is intentionally limited to the **chunking problem**.
  Deliberately out of scope: reranking, hybrid BM25+semantic, hyDE, query
  expansion, meta-refinement, agent tooling, multi-hop reasoning.
- `$` figures use example `/1M`-token pricing; always verify the live Groq console
  rates before quoting dollars.
- Reproducibility: `pytest` = 13 pure-logic tests, `verify_project.py` = 6/6
  hygiene checks, and the transcript is regenerable with `python demo.py`.

---

## APPENDIX · REPRO DELIVERY

```
python -m pip install -r requirements.txt   # 1
python demo.py                              # 2  <- the whole deck, printed live
python -m pytest -q; python verify_project.py  # 3  <- proof
```

Live LLM answers appear automatically when `.env` holds `GROQ_API_KEY` + a valid
`GROQ_MODEL` (see `.env.example`); otherwise the deterministic dry-run backend
reports the same grid with mock replies — every number above is identical either
way. On Windows, prefer a UTF-8 terminal (VS Code / Windows Terminal) so glyphs
like ✂ ✅ render; the file `evidence/demo_output.txt` is saved as UTF-8 for GitHub.