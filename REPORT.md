# RAG CHUNKING STRATEGIES — THE COMPLETE DECK

**Every chunking approach explained, compared, and judged — with a live, reproducible experiment.**

> How to read this deck: each `##` block is one "slide". The first four blocks build the
> mental model; the middle ones go strategy-by-strategy, then compare; the last two help
> you choose. Every number under **Live evidence** comes from `python3 demo.py` (see
> `evidence/demo_output.txt`) — nothing here is invented.
>
> Repo: `memoona-zahur/RAG-Chunking-Methods` · Test document: a 1,628-word Solar Home
> manual (`demo_document.md`) with long paragraphs and a spec/price table.
> Backend: Groq `openai/gpt-oss-120b` — the committed transcript is a REAL live run,
> every token is measured from the model's usage object, and the agentic side makes
> two real LLM calls per question (navigation + answer). No key → the deterministic
> dry-run backend reproduces the same retrieval numbers with mock replies (APPENDIX).

---

## THE ONE QUESTION THIS DEMO ANSWERS

> **Should we slice a long document into chunks before feeding it to RAG — and if
> we slice it, which of the 8 ways is best, and at what cost?**

Most RAG content shows you *a* way to chunk. This deck shows you **all the ways**,
judges them with the same ruler, and lets a chunkless "agentic" path compete too —
so the choice stops being habit and starts being a decision.

Preview of the verdict (full grid in [THE JUDGEMENT GRID](#the-judgement-grid-how-they-really-scored)):

| Strategy | Required facts found (of 28) | One-line verdict |
|---|---|---|
| structural (heading-aware) | **28/28** | perfect completeness at 1 call/question — the cheapest perfection here |
| agentic / chunkless | **27/28** | navigator found every pair but Q1 (its two facts live in separate paragraphs) — pays ~2.5× the price, 2× the calls |
| paragraph | 26/28 | great recall, but chunks can get fat |
| semantic | 26/28 | follows meaning, not questions |
| fixed_token | 24/28 | tiny context, same boundary disease |
| fixed_char | 23/28 | the raw baseline; cuts sentences on purpose |
| sentence | 23/28 | grammar-safe, but facts scatter anyway |
| recursive | 21/28 | the "default" is not a magic bullet |

---

## CHUNKING STRATEGIES AT A GLANCE

The eight strategies in this demo, in beginner-friendly English. The details,
method-by-method, come later in the deck.

| Method | How it splits | Main characteristic |
|---|---|---|
| **Fixed-character** `fixed_char` | Cuts the text every ~500 characters, snapping to the nearest space. | The raw baseline — cheap and predictable, but boundaries land inside sentences and table rows, so facts get severed. |
| **Fixed-token** `fixed_token` | Cuts every ~120 tokens and steps forward with an overlap; words stand in for real tokens. | The token-tuned sibling of fixed-character — still blind to sentence and paragraph meaning. |
| **Sentence-based** `sentence` | Groups sentences up to a size budget; boundaries land only between sentences. | Grammar-safe: it never cuts mid-sentence, but related sentences can still end up in different chunks. |
| **Paragraph-based** `paragraph` | Makes one chunk per paragraph (a blank-line block; headings and tables stay attached). | Keeps one idea-unit together, but paragraphs can grow large and dilute retrieval. |
| **Structural / heading-based** `structural` | Splits at each `##` heading, so a whole section (and its table) is one chunk. | Turns headings into free structure markers — sections stay intact and easy for a retriever to find. |
| **Recursive** `recursive` | Tries the most meaningful separator first (paragraph → line → sentence → space) until every piece fits; overlaps the tails. | The familiar "default" splitter — sensible out of the box, but not a magic bullet. |
| **Semantic** `semantic` | Embeds each sentence with the local model and merges neighbours while the meaning stays similar (above a threshold). | Follows topic shifts instead of character counts, so boundaries match ideas — at the cost of an embedding pass. |
| **Agentic / chunkless** `agentic` | Does not split at all: whole paragraphs become *navigation units*, and an LLM picks which ones to read in full. | Nothing is ever severed by a boundary, but navigation bills for it: ~2× the calls and ~2.5× the price of a chunked method. |

> ℹ️ **"Chunkless" here is one flavor, not the whole category.** This demo's agentic
> path uses an LLM navigator to choose paragraphs — traditional chunking retrieves
> fixed chunks instead. Other chunkless RAG designs (e.g. no navigation step at all)
> work differently, so don't generalize from this one implementation.

---

## WHAT A CHUNK IS — THE 30-SECOND DEFINITION

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

## WHY CHUNKING IS A TRADE-OFF, NOT A TRICK

There is no perfect chunk size — there is only a balance between two failure modes.

| If chunks are… | The retrieval pain | The cost pain |
|---|---|---|
| **too small** (e.g. 100 chars) | an answer needs two sentences, gets split → the retriever returns half a fact | context is cheap, but the answer is wrong |
| **too big** (e.g. whole section) | irrelevant filler travels with the fact → top-k context is diluted | you pay for tokens you didn't need |
| **just right** (boundaries match ideas) | retriever points exactly at the fact | smallest context that still contains it |

Our test document was written to make this visible. Its second paragraph is a single
~180-word thought; inside the manual, a *fact* ("the battery runs **eight hours**")
and its *companion fact* ("warranty: **ten years**") sit in different paragraphs. A
naive char-splitter severs both of them — see [STRATEGY BY STRATEGY](#strategy-by-strategy-the-detail).

**The chunking choice is the difference between a committee that receives half the
evidence and one that receives all of it.**

---

## THE 8 STRATEGIES — THE MAP BEFORE THE DETAILS

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

## THE BASELINE WE JUDGE BY (the ruler)

Judging must be **independent of the LLM** (so the comparison is reproducible) and
**concrete** (so it is checkable). The ruler has three parts:

**① Required facts.** Fourteen questions, each with 2 hand-defined facts (= 28 facts).
A method scores ✅ only if **both** facts survive retrieval into its top-2 context.
All facts are verbatim substrings of the manual; a few questions pair facts that
live in *different* paragraphs (Q1, Q2, Q4) so a method must retrieve two units to
pass, while the rest are same-paragraph checks.

| # | Question | Fact A + Fact B |
|---|---|---|
| Q1 | How long can the Home battery run alone, and what warranty applies to it? | `eight hours` + `ten years` |
| Q2 | What does the Solar Home Plus kit cost and what is its warranty? | `3,499` + `24 months` |
| Q3 | Which kit carries model number SHM-400? | `shm-400` + `solar home mini` |
| Q4 | What should a coastal-area user prepare at install, and which maintenance habit matters most? | `corrosion protection kit` + `heat sinks` |
| Q5 | What is the entry-level kit called, and how large is its battery? | `solar home mini` + `5 kwh` |
| Q6 | Which kit lets a larger home run a washing machine or water pump in the evening? | `washing machine` + `water pump` |
| Q7 | Where must the battery pack be installed, and how far from walls? | `indoors` + `fifty centimeters` |
| Q8 | How should roof panels be aimed and tilted? | `southern hemisphere` + `ten degrees` |
| Q9 | At what charge levels does the inverter stop charging and start feeding the home? | `one hundred percent` + `eighty percent` |
| Q10 | How long is the battery rated, and what cycle count backs it? | `ten years` + `six thousand` |
| Q11 | What does the coastal corrosion protection kit consist of? | `marine-grade coating` + `sealed junction box` |
| Q12 | What must an owner upload to claim an in-warranty inverter fault, and how fast is service? | `diagnostic summary` + `seven days` |
| Q13 | At a light draw of five hundred watts, how long does a full ten kWh pack last? | `sixteen hours` + `five hundred watt` |
| Q14 | What, besides dust, can cut panel production, and by how much? | `bird droppings` + `ten percent` |

**② Retrieval quality.** For each method, every chunk is embedded with
`all-MiniLM-L6-v2`, indexed in Qdrant (in-memory), and the question is retrieved with
`k=2`. We compute precision@2 and recall@2 against "chunks that contain a required fact",
plus **hit-rate@2** — *"did at least one relevant unit land in our top-2?"* — the metric
most quoted for agentic navigation, where "found or not" matters more than exact rank.
The **ctx** column reports the approximate tokens in each method's top-2
(`chars/4`), making the single most important caveat visible: **k=2 is not an
equal-token bet.** `structural` spends ~906 tokens/query on top-2; `fixed_char`
spends ~242. The grid lets you weigh completeness *against* the context it cost.
The chunkless row is judged on the units its navigator actually picked (not on a
fresh embedding search over the same paragraphs), so the grid scores the agent,
not a duplicated paragraph row.

**③ Boundary quality.** Two weaknesses are counted mechanically:

- **✂ mid-sentence cuts** — how many chunk boundaries land *inside* a sentence.
- **table cuts** — how many boundaries slice a markdown table row.

**④ Cost.** Tokens handed to the LLM per question (`chars/4`) and LLM calls.

> One ruler, eight contestants, zero hand-written numbers.

---

## STRATEGY BY STRATEGY (the detail)

Format per strategy: **what it does → where it wins → where it breaks → live evidence
from our run → verdict.**

### `fixed_char` — every N characters (the raw baseline)

- **What:** cut the text at 500 characters, no knowledge of words, sentences or
  meaning. Overlap of 100 chars so a fact straddling a boundary has a second chance.
- **Wins:** dead simple, totally predictable chunk size, trivially implemented, and a
  great "worst case" to measure everyone else against.
- **Breaks:** boundaries land inside words and sentences; the retriever then returns
  slivers of facts. Long, idea-sized paragraphs get chopped into unrelated fragments.
- **Live evidence:** `25 chunks` · ✂ **23 mid-sentence cuts** · ✂ **1 table-row cut**.
  First sever in the manual: `…an hour. The philosophy behind ✂ uick-start card that
  guides you th…` **Found 23/28 required facts**, P@2 0.57, R@2 0.71.
- **Verdict:** the honest baseline — it *shows* that chunks sever meaning. Nobody
  should ship this for prose, but you should always run it to measure the others.

### `fixed_token` — every N tokens

- **What:** same idea, but the unit is a token (here a word-count stand-in, so the demo
  stays dependency-free). 120-token windows with 20-token overlap.
- **Wins:** tokens are what the LLM actually bills; so "120 tokens ≈ a predictable
  bill." Slightly less violent than chars because it respects word boundaries.
- **Breaks:** identical disease — it still cuts *between* sentences mid-idea.
- **Live evidence:** `17 chunks` · ✂ **14 mid-sentence cuts** · table intact.
  **Found 24/28** · P@2 0.54 · R@2 0.65.
- **Verdict:** the same baseline, tuned to a model's real currency. Fine for exact
  token budgets, still blind to meaning.

### `sentence` — boundaries only between sentences

- **What:** split on sentence-ending punctuation, then group sentences up to a char
  budget. No boundary can ever land inside a sentence by construction.
- **Wins:** zero mid-sentence cuts; each unit is grammatically whole; cheap to build.
- **Breaks:** "grammar-aware ≠ answer-aware." Our Q1 needs a fact from §4-para-1 *and*
  a fact from §5 — two different paragraphs. Sentence units are still separate
  retrieval targets, so both facts can still miss the top-2.
- **Live evidence:** `28 chunks` · **0 mid-sentence cuts** · table intact.
  **Found 23/28** · P@2 0.46 · R@2 0.77.
- **Verdict:** fixes the symptom (cut sentences) but not the problem (scattered
  facts). It is a *necessary* refinement, not a strategy on its own for idea-keeping.

### `paragraph` — one chunk per paragraph

- **What:** one chunk per blank-line-separated block. Each unit is (usually) one
  self-contained thought.
- **Wins:** strong recall in our run (R@2 0.91) — when the answer lives inside one
  paragraph, you find the *exact* paragraph.
- **Breaks:** paragraphs can be huge (up to 1,610 chars here) → context bloat and
  diluted top-k; and a cross-paragraph question (Q1, Q4) still splits the required
  facts across two chunks.
- **Live evidence:** `9 navigation units` (whole paragraphs — nothing severed) · **0 cuts** · table intact.
  **Found 26/28** · P@2 0.61 · R@2 0.91.
- **Verdict:** strong, simple default for well-written documents. Its weakness is size,
  not meaning.

### `structural` — heading-aware (the winner here)

- **What:** respects `##` headings and keeps the table attached to its section — the
  markdown-aware cousin of "section-based" chunking.
- **Wins:** sections are *semantic containers*: a heading says what the whole block is
  about, so the retriever gets a tagged unit; tables stay whole; completeness stays
  perfect at 1 call per question.
- **Breaks:** relies on the document having structure. A wall of prose with no headings
  degrades it to paragraph-level. And its *completeness is bought with context*:
  top-2 here runs ~906 tokens/question — roughly a third of the manual — versus ~242
  for `fixed_char` (see the **ctx** column).
- **Live evidence:** `6 chunks` · **0 cuts** · table intact.
  **Found 28/28 — ALL required facts.** P@2 0.54 · R@2 **0.93**. All 14 questions ✅.
- **Verdict:** **the cheap perfection winner** — among the two methods that found
  everything, `structural` gets there at ~40% of the agentic path's price and stays
  section-sized. Use it whenever your corpus is markdown/HTML/sectioned.

### `recursive` — the famous "default"

- **What:** try the highest-priority separator first (paragraph → sentence → word →
  char), and only fall back to raw cuts when nothing fits — plus a 60-char overlap.
- **Wins:** adapts to text quality; it is the default in many RAG libraries because it
  behaves "well enough" on most documents.
- **Breaks:** overlap re-attaches each chunk's tail, which can repeat noise and blur
  boundaries, and "well enough" still misses idea-boundaries.
- **Live evidence:** `25 chunks` with overlap tails · ✂ **4 cuts** · table intact.
  **Found only 21/28** · P@2 0.46 · R@2 0.67.
- **Verdict:** the demo's most useful surprise: **a default is not a proof.** On this
  document it was the *worst* at completeness. Always measure it, don't trust it.

### `semantic` — breaks where meaning shifts

- **What:** embed every sentence, and start a new chunk when the cosine similarity
  between the next sentence and the running centroid drops below a threshold. No fixed
  sizes, no separator rules.
- **Wins:** boundaries follow meaning, not punctuation — units are "topically coherent,"
  which is what paragraph chunking *wants* to be.
- **Breaks:** a topic is not a *question*; and it costs a model pass at index time.
  Topic boundaries and question-shaped boundaries just aren't the same thing.
- **Live evidence:** `24 chunks` · **0 cuts** · table intact.
  **Found 26/28** · P@2 0.54 · R@2 0.90.
- **Verdict:** intellectually appealing, honestly an add-on: fine for topical search,
  not a substitute for structure when the question is fact-shaped.

### `agentic` / `chunkless` — no chunks at all

- **What:** the far end. Whole paragraphs/sections are *navigation units*; the agent
  embeds the question, picks the best 2 units, then reads them **in full** — no boundary
  can ever sever a fact inside a read unit.
- **Wins:** by construction, zero mid-sentence/table cuts and zero severed facts. An
  elegant answer to the whole chunking debate: "don't."
- **Breaks:** completeness now depends on *navigation*, not boundaries — and it pays
  for it: **2 REAL LLM calls per question** (a navigation call that picks the units +
  an answer call), and the whole read sections as context. The navigation call also has
  to *scan the full document listing* to choose, which is visible in the token bill. If
  the agent picks the wrong units it misses facts, exactly like retrieval.
- **Live evidence:** `9 whole-paragraph units` · 0 cuts by construction, and the agent
  really navigates: in this transcript the LLM picked the paragraphs on every question
  (`via llm navigation`) — for Q2 it went straight to the spec table. Scored on the
  paragraphs it *actually navigated to*, not on an embedding search: **Found 27/28 —
  every fact pair but Q1** (its two facts live in separate paragraphs and the navigator's
  pair of units caught only one). P@2 0.64 · R@2 0.95 · **HR@2 1.00**, context ~571
  tokens/question.
  The replies cite the source verbatim (e.g. Q14: *"a layer of dust or bird droppings
  can reduce output… the loss can be ten percent or more"*).
- **Verdict:** the idea that motivated the demo — and our measured takeaway is nuanced:
  the LLM navigator is excellent on this manual (27/28, one paragraph-border miss), so
  the differentiator is NOT facts, it is **cost and predictability**: ~2× the calls and
  ~2.5× the price of `structural`, which matches or beats it in fewer, smaller reads —
  see [COST CONVERSATION](#the-cost-conversation-chunked-vs-chunkless). (Offline, with no key,
  the navigator degrades to embedding ranking, so its row then mirrors `paragraph`'s —
  honestly: offline there IS no agent to score.)
  See [THE ONE-LINE TAKEAWAYS](#the-one-line-takeaways-cheat-sheet).

---

## CHUNKLESS vs AGENTIC — TWO DIFFERENT IDEAS, NOT SYNONYMS

These two terms get mixed up all the time, but they answer different questions.

**Chunkless RAG — about HOW the document is stored & read.** No fixed-size chunks.
The document keeps its natural structure: whole paragraphs, sections and tables stay
intact as *navigation units*, and the system moves through them like a human reading a
manual section-by-section — a sentence or a table row is never cut in half. The
downside: it reads bigger pieces, so each context costs more tokens.

**Agentic RAG — about WHO decides what to read.** Instead of a fixed *"embed the
question → return the top-2 similar chunks"* step, an LLM *agent* runs a loop:
plan (what do I need?) → search (which section / tool / source?) → read → judge
(is this enough?) → retry with a better plan if not. It can even split a complex
question into sub-questions and use multiple tools. The downside: more calls per
question → higher latency and more tokens.

**Why the confusion?** Both reject the naive *"cut into fixed-size pieces and hope the
top-k happens to contain the answer"*, so they *feel* similar. But they are independent
axes: *chunkless* is about how text is segmented, *agentic* is about who controls
retrieval. A system can be chunked+agentic or chunkless+static — they are not two ends
of one line.

**How this demo implements both at once.** Its 8th method is chunkless **and** agentic:
whole paragraphs are the navigation units (chunkless), and an LLM navigator decides
which ones to read in full (agentic). Its real measured fact-survival and token cost
against the 7 chunked methods are in the [judgement grid](#the-judgement-grid-how-they-really-scored)
and the [cost conversation](#the-cost-conversation-chunked-vs-chunkless).

**When to use which:** **Chunked** → FAQs, knowledge bases, chatbots — high volume,
well-sectioned content, fast and cheap. **Chunkless/Agentic** → contracts, financial
or technical reports, research papers — cross-section answers, tables, complete
context, when you can afford higher latency and cost.

---

## WORKED EXAMPLES — 2 INPUTS, 8 FATES

The strategy detail above explains **what** each method does; this section shows the
**same two inputs being cut by all eight methods** — boundary-blind ones slice inside a
sentence, boundary-respecting ones do not. Every split below is produced by
`chunkers.py`, nothing hand-staged.

**EXAMPLE 1 — two headings, one long paragraph, a closing** (619 chars):

> ## Safety
>
> Always disconnect the power before touching the inverter terminals, and wait at least
> five minutes after switching the system off so the capacitors can drain.
>
> ## Warranty
>
> The battery carries a ten-year warranty against manufacturing defects, and every panel
> a twenty-five-year warranty, provided the unit is installed by a certified technician
> and the online registration form is completed within thirty days of the first startup;
> replacement units are shipped within ten working days, and service visits are free
> during the first two years.
>
> ## Recycling
>
> Return the packaging to any participating dealer.

| Method | Chunks | What actually happened |
|---|---|---|
| fixed_char | 2 | split the long Warranty paragraph mid-word (`…eted within thir…`) — blind to sentences |
| fixed_token | 1 | whole input ≈ 110 words < 120 tokens → kept whole this time |
| sentence | 1 | 4 sentences easily fit the budget → no boundary, all intact |
| paragraph | 3 | three blank-line blocks, headings included, Warranty kept whole |
| structural | 3 | three sections, each `## …` block kept whole (its home turf) |
| recursive | 2 | Warranty paragraph > size → fallback split, otherwise whole |
| semantic | 1 | sentences topically coherent → no drop, stays whole |
| agentic | 3 | whole-paragraph units, read in full — nothing severed |

**EXAMPLE 2 — one long run-on sentence, no heading at all** (744 chars):

> The Home battery stores cheap daytime solar energy during the day and releases it at
> night, and it can power a full house for roughly eight hours at a steady
> one-thousand-watt draw, or about sixteen hours when the load stays near five hundred
> watts, because a fully charged ten-kilowatt-hour pack makes about eight usable
> kilowatt-hours available before depth-of-discharge protection steps in, and when the
> battery is paired with the Solar Home Mini kit it keeps the refrigerator, the router
> and the lights running through the evening and during outages, so the philosophy of
> the whole series is simple — store while the sun shines, spend only what you need
> while it does not, and let the automatic transfer switch decide where every watt goes.

| Method | Chunks | What actually happened |
|---|---|---|
| fixed_char | 2 | sliced the single sentence — chunk 2 opens mid-word: `…hen the battery is paired…` (the boundary split the word "when") |
| fixed_token | 2 | same story, the cut just lands a few words later: chunk 2 opens `while the sun shines…` |
| sentence | 1 | one sentence = one unit — nothing severed by construction |
| paragraph | 1 | one paragraph = one unit — nothing severed |
| structural | 1 | no headings here, degrades to paragraph level — and still whole |
| recursive | 2 | paragraph too big → fell back to separators, cut mid-idea, and re-emitted the overlap tail (`…and during outages, so the philosophy…` appears on both sides) |
| semantic | 1 | one coherent topic → stays one chunk |
| agentic | 1 | one whole paragraph, read in full |

**What these two tables teach in one glance:** the size-based and fallback methods
(`fixed_char`, `fixed_token`, `recursive`) cut *inside* an idea; the boundary-aware ones
(`sentence`, `paragraph`, `structural`, plus `agentic`'s no-chunk far end) cut
*between* ideas. Example 2 is exactly why `sentence`/`paragraph` show 0 mid-sentence
cuts while `fixed_char` shows 23 on the real document.

---

## THE JUDGEMENT GRID (how they really scored)

Reproduced from the live run — a clean markdown table (the demo prints the full
Q1–Q14 grid; `ctx` = approximate tokens in each method's top-2 context):

| Method | Facts (28) | Avg P@2 | Avg R@2 | Avg HR@2 | ctx |
|---|---:|---:|---:|---:|---:|
| fixed_char | 23/28 | 0.57 | 0.71 | 1.00 | 242 |
| fixed_token | 24/28 | 0.54 | 0.65 | 0.93 | 357 |
| sentence | 23/28 | 0.46 | 0.77 | 0.86 | 180 |
| paragraph | 26/28 | 0.61 | 0.91 | 1.00 | 555 |
| structural | **28/28** | 0.54 | 0.93 | 1.00 | 906 |
| recursive | 21/28 | 0.46 | 0.67 | 0.86 | 252 |
| semantic | 26/28 | 0.54 | 0.90 | 1.00 | 285 |
| agentic | **27/28** | 0.64 | 0.95 | 1.00 | 571 |

**How to read it**

- **Facts (28)** is total completeness (28 = perfect; each question needs both of its 2 facts).
- **Avg P@2** = of the 2 units retrieved/navigated, how many were relevant.
- **Avg R@2** = of all *relevant* units that exist, how many were retrieved/navigated.
- **Avg HR@2** = did the top-2/navigated set contain *any* relevant unit (1 = yes), averaged.
- **ctx** = the token budget the ruler really spends — where the "equal k, unequal
  cost" caveat lives. Structural reaches perfection with ~3.7× `fixed_char`'s context.

**Four observations that survive any single run**

1. **Structure is the cheapest perfection.** `structural` finds 28/28 at one call per
   question — the only cost is context size (ctx 906), which the grid prints honestly.
2. **Every method that ignores paragraph boundaries leaks facts there.** Q1 (two
   facts, two paragraphs) is exactly where char/token/sentence/recursive fall to ⚠ or ❌ —
   and it is the *only* question the agentic navigator missed (its two picked paragraphs
   held one fact). Boundaries that ignore paragraphs are boundaries that eat answers.
3. **Scoring the navigator changes the agentic story.** Judged on the paragraphs it
   actually picked, the LLM navigator is 27/28 — better than every fixed method and
   mortally close to `structural`, and its single miss is the same paragraph-border
   question, not navigation randomness. So the differentiator here is **cost and
   predictability**, not facts (2× calls, ~2.5× price — see the cost conversation).
   Offline, the navigator becomes embedding ranking and its row legitimately mirrors
   `paragraph`'s.
4. **Hit-rate can flatter a retriever.** `paragraph`/`fixed_char`/`semantic` all reach
   HR@2 1.00 yet land at 26/28, 23/28, 26/28 — they *always* find *a* relevant unit but
   not always *both* facts. "We found something" ≠ "we found the answer": that is
   exactly why this demo judges facts, not just hits.

---

## THE COST CONVERSATION — CHUNKED vs CHUNKLESS

The framing is "chunked = 1 call per question, agentic = 2 calls and a full-document
scan." Our live numbers make it concrete (`$` uses the example pricing in `costs.py`
— verify live Groq rates before quoting dollars). We cost the **weak baseline**
(`fixed_char`) *and* the **recommended method** (`structural`), so the cheap path is
the one you'd actually ship, not a strawman:

| Path | LLM calls (14 Q's) | Tokens in (14 Q's) | Tokens out (14 Q's) | Est. USD (14 Q's) | USD / question |
|---|---:|---:|---:|---:|---:|
| chunked (`fixed_char`) | 14 | 4,170 | 4,038 | 0.00565 | 0.000404 |
| chunked (`structural`) | 14 | 11,547 | 3,473 | 0.00956 | 0.000683 |
| agentic / chunkless | 28 | 29,757 | 8,193 | 0.02403 | 0.001716 |

What is honest here — and what isn't:

- **Every number above is a REAL LLM call.** All three paths actually answered on Groq
  (`openai/gpt-oss-120b`) in this run; input *and* output tokens come from the model's
  usage object, not a model.
- **The call gap is structural: 2 vs 1 per question** (navigate + answer). Because there
  are two calls, agentic also pays answer output tokens *twice over* — part of its 8,193
  vs ~3.7k in output.
- **The input-token gap is real and big even on this tiny manual — agentic sends ≈ 7.1×
  `fixed_char`'s context and ≈ 2.6× `structural`'s** (29,757 vs 4,170 / 11,547): it reads
  whole paragraphs, and its navigation call scans the whole numbered document.
- **The dollar gap lands at ≈ 2.5× per question vs `structural`** (0.001716 vs 0.000683)
  — and `structural` reaches 28/28 completeness while staying at one call.
  Against `fixed_char` the ratio is ≈ 4.2×, but that comparison is the apples-to-oranges
  one: `fixed_char` is the weak baseline, not the recommendation.
- **The predictability gap is the one that scales**: a chunked top-2 is *roughly
  predictable* (top-k × chunk size), while agentic context *grows with whole-section size
  plus the scan*. Point it at a 5,000-word regulation and the 2.5× on this manual becomes
  more: the calls stay 2 vs 1, but the context will not stay small.

So the honest one-liner: *"Chunkless never severs a fact, and its navigator reads the
manual well — but it bills like a taxi (per read) instead of sharing a bus (fixed
chunks). Structure gets you 28/28 at the bus price — that's why real systems chunk."*

---

## HOW TO CHOOSE (a decision guide)

```
start here ──▶ is the corpus naturally structured (md/html/sections)?
                 ├─ YES ──────▶ structural / section-aware ──▶ the winner here
                 └─ NO ──────────────┬─ is the text neat prose with one idea per paragraph?
                                      │      ├─ YES ──▶ paragraph (or semantic for fuzzy topics)
                                      │      └─ NO ──▶ do facts span paragraphs?
                                      │               ├─ YES ──▶ hybrid: paragraph + overlap window
                                      │               └─ NO ──▶ recursive (but MEASURE it)

still worried about severed facts? ──▶ run agentic/chunkless side-by-side;
                                     use its whole-unit reads only where $ allows
```

Rules of thumb (from this deck's evidence):

| Goal | Reach for |
|---|---|
| Completeness at lowest cost (recommended default) | `structural` — 28/28, 1 call/question |
| Fine-grained pinpointing, neat prose | `paragraph` |
| Fuzzy topical search, no headings anywhere | `semantic` |
| "I must use the library default" | `recursive` — **but run the facts test** |
| The demo/audit of whether boundaries hurt at all | `fixed_char`, `fixed_token` |
| Avoid boundaries entirely; budget allows ~2.5× price, 2× calls | `agentic` / chunkless — scored on its real navigation |

Parameters worth tuning in `chunkers.py`: `size` (500 chars / 120 tokens here), `overlap`
(100), recursive separators, semantic similarity threshold (0.45).

---

## THE ONE-LINE TAKEAWAYS (cheat sheet)

| Strategy | One line to remember |
|---|---|
| fixed_char | "No meaning, no mercy — the control group." |
| fixed_token | "Bills in the LLM's currency; still blind to ideas." |
| sentence | "Grammar-safe, answer-blind." |
| paragraph | "The idea-unit default; fat paragraphs eventually bite." |
| structural | "Headings are free metadata — cheapest 28/28 (watch the ctx column)." |
| recursive | "The famous default is not a proof; overlap can add noise." |
| semantic | "Follows topics, not questions; costs a model pass." |
| agentic | "Never severs a fact, navigator is sharp — the 2-call bill is the tax." |

---

## HONEST LIMITS AND WHAT'S GENUINELY NEXT

- This is **one** document, **fourteen** questions, **one** embedding model: a *micro-benchmark*,
  not a law of nature. The verdict pattern (structure wins, fixed-size leaks, agentic
  finds facts but bills for it) is general — the exact numbers are not.
- **k=2 is not an equal-token bet.** Methods were compared at equal retrieval depth, not
  equal context size; the **ctx** column shows what each top-2 really cost, so weigh
  completeness against context honestly (e.g. structural 906 vs fixed_char 242)
  before generalizing from this manual.
- **The grid is deterministic; the agentic row is stochastic.** Chunking and top-2
  retrieval are pure logic + fixed embeddings, so every chunked method reproduces its
  exact facts/cuts/P/R/HR on a rerun (in this *fixed environment* — a dependency or
  hardware change could flip a borderline tie). The dollar figures are not fully fixed
  even for chunked methods: input tokens are exact, but output tokens come from a
  stochastic answer and drift slightly (fixed_char out 3,810 → 4,038 across two runs).
  The LLM navigator varies in *both* facts and tokens: it was 28/28 on one live run and
  27/28 on another (each only missing Q1, whose two facts live in separate paragraphs).
  Treat the agentic row as a snapshot, not a guarantee.
- Per instruction, the demo is intentionally limited to the **chunking problem**.
  Deliberately out of scope: reranking, hybrid BM25+semantic, hyDE, query expansion,
  meta-refinement, agent tooling, multi-hop reasoning.
- **Long-context degradation ("lost in the middle") is NOT measured here** — the agentic
  side reads whole paragraphs, but proving recall drops toward the middle of a long
  context needs a separate, position-controlled benchmark. Out of scope with the demo.
- `$` figures use example `/1M`-token pricing; always verify the live Groq console rates
  before quoting dollars.
- Reproducibility: `pytest` = 17 pure-logic tests, `verify_project.py` = 6/6 hygiene
  checks, and the transcript is regenerable with `python3 demo.py > evidence/demo_output.txt`.

---

## APPENDIX · REPRO DELIVERY

```
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu  # 0  <- CPU-only, skips ~6 GB CUDA
python3 -m pip install -r requirements.txt   # 1
python3 demo.py                              # 2  <- the whole deck, printed live
python3 -m pytest -q; python3 verify_project.py  # 3  <- proof
```

Live LLM answers appear automatically when `.env` holds `GROQ_API_KEY` + a valid
`GROQ_MODEL` (see `.env.example`); the agent makes **two real calls per question**
(navigation + answer), the grid's agentic row scores the paragraphs it **navigated
to**, and every token in the cost table is measured from the model's usage object.
The committed transcript (`python3 demo.py > evidence/demo_output.txt`) is from a
live run. Without a key, the deterministic dry-run backend reproduces the same chunk
counts, cuts, and the chunked methods' fact grid and P/R/HR scores with mock replies —
but the dry-run has no agent to score, so the agentic row then mirrors the embedding
ranking of `paragraph`, and the token-dollar columns reflect mock sizes. On
Windows, prefer a UTF-8 terminal (VS Code / Windows Terminal) so glyphs like ✂ ✅ render;
the file `evidence/demo_output.txt` is saved as UTF-8 for GitHub.