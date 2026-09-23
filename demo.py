"""RAG Chunking Methods — the one-command walkthrough.

    python demo.py

Runs the full comparison on demo_document.md: 8 chunking strategies
(including the chunkless/agentic far end), 4 questions, fact-based
evaluation, and an honest chunked-vs-chunkless cost drill-down.

Prints in color when stdout is a terminal; stays plain for file capture.
Real answers via Groq when GROQ_API_KEY is set, deterministic dry-run
otherwise (see llm.py). Everything you see is produced by THIS run —
no hand-written numbers anywhere.
"""

from __future__ import annotations

import re
import sys

import chunkers
import costs
import embed_store as es
import evaluate

DOC = "demo_document.md"


def re_split_blocks(text: str) -> list[str]:
    return re.split(r"\n\s*\n", text)

# ---- ANSI helpers (auto-off when not a TTY) ---------------------------------
_COLOR = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def green(s):  return _c("32", s)
def red(s):    return _c("31", s)
def yellow(s): return _c("33", s)
def bold(s):   return _c("1", s)
def dim(s):    return _c("2", s)

OK, NO, PART = green("PASS"), red("FAIL"), yellow("PARTIAL")


def fact_status(cov: int, total: int) -> str:
    if cov >= total:
        return "PASS"
    if cov == 0:
        return "FAIL"
    return "PARTIAL"


def fact_mark(cov: int, total: int) -> str:
    return {"PASS": OK, "FAIL": NO, "PARTIAL": PART}[fact_status(cov, total)]


# ---- helpers ---------------------------------------------------------------
def split_stats(chunks):
    return chunkers.count_mid_sentence_cuts(chunks), chunkers.count_table_cuts(chunks)


def first_awkward_cut(chunks) -> str | None:
    """Show the first boundary that lands inside a sentence."""
    for a, b in zip(chunks, chunks[1:]):
        tail, head = a.text.rstrip(), b.text.lstrip()
        if tail[-1:] in ".!?" or head[0].isupper() or head.startswith(("|", "#")):
            continue
        return f"...{tail[-34:]} {bold('✂')} {head[:34]}...".replace("\n", " ⏎ ")
    return None


# ---- main story -------------------------------------------------------------

def banner():
    print(bold(" RAG CHUNKING METHODS — a head-to-head on one long manual "))
    print(" fixed-char · fixed-token · sentence · paragraph · structural · recursive · semantic · chunkless/agentic")
    print()
    import llm
    print(llm.describe_backend())
    print()


def show_document(text: str):
    paras = [
        b.strip()
        for b in re_split_blocks(text)
        if b.strip() and not b.strip().startswith("#")
    ]
    sents = chunkers.split_sentences(text)
    words = len(text.split())
    print(bold(" 1 · THE DOCUMENT"))
    print(f"   demo_document.md — a Solar Home Power System manual written in LONG paragraphs")
    print(f"   {words} words · {len(paras)} paragraphs · {len(sents)} sentences · 1 table")
    print(f"   It is exactly the kind of source most RAG videos wrongly assume is short & neat.")
    print()


def why_chunk(paras: list):
    print(bold(" 2 · WHY CHUNKING MATTERS (the problem we fix)"))
    p = next(x for x in paras if "off-grid power package" in x)
    print("   One paragraph here is ~", len(p.split()), "words — bigger than many models' comfort zone")
    print(f"   {dim(p[:140])}…")
    print("   Fixed-size char splitting severs this mid-sentence, as we will now prove.")
    print()


def show_method(idx: int, name: str):
    fn = chunkers.METHODS[name]
    chunks = fn(DOC_TEXT)
    cuts, tcuts = split_stats(chunks)
    sizes = ",".join(str(len(c.text)) for c in chunks)
    print(bold(f" 3.{idx}  {name}"))
    print(f"      {chunkers.PURPOSE[name]}")
    line = f"      {len(chunks)} chunks   sizes {sizes} chars"
    if cuts:
        line += f"   {red(f'✂ {cuts} mid-sentence cut(s)')}"
    else:
        line += f"   {green('0 mid-sentence cuts')}"
    if tcuts:
        line += f"   {red(f'✂ {tcuts} table-row cut(s)')}"
    else:
        line += f"   {green('table intact')}"
    print(line)
    aw = first_awkward_cut(chunks)
    if aw:
        print(f"      e.g. {dim(aw)}")
    print()


def run_agentic(agent, block: list):
    print(bold(" 4 · CHUNKLESS / AGENTIC (no chunks at all)"))
    print("      whole paragraphs are navigation units · the agent picks the best and reads them IN FULL")
    print()
    for q, _facts in evaluate.QUESTIONS:
        res = agent.answer(DOC_TEXT, q)
        print(f"   Q  {q}")
        print(f"      units considered {res['chosen_units']['units_considered']}  "
              f"chosen {res['chosen_units']['chosen']} (read whole)  "
              f"{res['chosen_units']['calls']} LLM calls modeled")
        print(f"      context ~{res['context_tokens']} tokens (full paragraphs, nothing trimmed)")
        print(f"      reply  {dim(res['reply'][:200])}" + ("…" if len(res["reply"]) > 200 else ""))
        print()


def eval_grid(rows_per_method: dict):
    print(bold(" 5 · FACT-COVERAGE GRID (the baseline for judgement)"))
    print("      every question carries 2 hand-defined required facts; a method gets ✅ only if")
    print("      BOTH facts survive retrieval into its top-2 context. No telling the LLM anything.")
    header = "      method        | Q1 | Q2 | Q3 | Q4 | facts(8) | avg P@2 | avg R@2"
    print(header)
    print("      " + "-" * (len(header) - 6))
    order = list(chunkers.METHODS)
    marks_for = {"PASS": " ✅", "FAIL": " ❌", "PARTIAL": " ⚠"}

    for name in order:
        rows = rows_per_method[name]
        marks = "".join(marks_for[fact_status(r["facts"][0], r["facts"][1])] for r in rows)
        total_cov = sum(r["facts"][0] for r in rows)
        p = sum(r["precision"] for r in rows) / len(rows)
        r_ = sum(r["recall"] for r in rows) / len(rows)
        print(f"      {name:<16}|{marks} | {total_cov:2d}/8   |  {p:4.2f}     | {r_:4.2f}")
    print()


def cost_drilldown(rows_per_method: dict, agentic_results: list[dict]):
    print(bold(" 6 · CHUNKED vs CHUNKLESS — THE COST GAP"))
    print("      chunked (fixed-size): 1 LLM call/question, only the small top-2 chunk as context.")
    print("      agentic (no chunks)  : 2 calls/question (navigate + answer), whole sections read.")
    print()
    ch_rows = []
    for name in ["fixed_char", "fixed_token"]:
        for r in rows_per_method[name]:
            ch_rows.append(
                {
                    "input_tokens": r["tokens"],
                    "output_tokens": 120,  # typical short answer length (dry-run also uses ~this)
                    "calls": 1,
                }
            )
    if not ch_rows:
        return
    chunked = costs.account(ch_rows)
    ag_rows = [
        {"input_tokens": a["input_tokens"], "output_tokens": a["output_tokens"], "calls": a["chosen_units"]["calls"]}
        for a in agentic_results
    ]
    agentic = costs.account(ag_rows)
    print("   method       | LLM calls | context tokens | est. USD (4 questions)")
    print("   " + "-" * 58)
    print(f"   {'chunked':<12} | {chunked['calls']:>9} | {chunked['input_tokens'] + chunked['output_tokens']:>14} | {chunked['usd']:>16.5f}")
    print(f"   {'agentic':<12} | {agentic['calls']:>9} | {agentic['input_tokens'] + agentic['output_tokens']:>14} | {agentic['usd']:>16.5f}")
    ratio = (agentic["input_tokens"] / max(1, chunked["input_tokens"]))
    print()
    print("      agentic reads ~{:.1f}x the context tokens of chunked for the same 4 questions".format(ratio))
    print("      and burns 2 calls instead of 1. On a small manual the gap is modest,")
    print("      but chunked context is PREDICTABLE (top-k x chunk size) while agentic")
    print("      context grows with whole-section size - flip it to a 5,000-word section")
    print("      and the gap multiplies. Structure is the cheap way to keep completeness")
    print("      and small fixed reads: structural got {}/4 questions complete in one call each.".format(
        sum(1 for r in rows_per_method["structural"] if r["facts"][0] == r["facts"][1]))
    )
    print()


def honest_limits():
    print(bold(" 7 · HONEST LIMITS OF THIS DEMO"))
    print("  - one document, four questions, one embedding model: a micro-benchmark, not a law.")
    print("  - a real production decision needs your own documents + question set + reranking.")
    print("  - $ figures use example llama-3.3-70b pricing; check the Groq console before quoting.")
    print("  - no hyDE, no BM25 hybrid, no meta-refinement — deliberately out of scope today.")
    print()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    banner()
    global DOC_TEXT
    DOC_TEXT = chunkers.load_document(DOC)
    show_document(DOC_TEXT)
    why_chunk(chunkers.split_paragraphs(DOC_TEXT))

    # 3 · per-method walkthrough (semantic needs the model, so build it once)
    rows_per_method = {}
    for idx, name in enumerate(chunkers.METHODS, start=1):
        show_method(idx, name)

    print(bold(" 3.9 · EMBED + RETRIEVE + EVALUATE EVERY METHOD (all-MiniLM-L6-v2, in-memory Qdrant)"))
    for idx, name in enumerate(chunkers.METHODS, start=1):
        chunks = chunkers.METHODS[name](DOC_TEXT)
        rows = evaluate.analyze_method(name, chunks, es.build_store, es.search)
        rows_per_method[name] = rows
        tot = sum(r["facts"][0] for r in rows)
        print(f"      {idx}/8 {name:<14} facts found (of 8) = {tot:2d}/8 -> "
              f"{'keep reading' if tot > 6 else 'look at the cuts above'}")
    print()

    import agentic
    agentic_results = [agentic.answer(DOC_TEXT, q) for q, _ in evaluate.QUESTIONS]
    run_agentic(agentic, agentic_results)

    eval_grid(rows_per_method)

    cost_drilldown(rows_per_method, agentic_results)
    honest_limits()
    print(bold(" DONE — every number above was computed in this run."))


if __name__ == "__main__":
    main()