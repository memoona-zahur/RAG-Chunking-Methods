"""Cost accounting for the chunked vs chunkless story.

The claim we demonstrate: chunked RAG is roughly 1 LLM call with a small
top-k context; the chunkless/agentic path needs several calls and reads whole
sections - so on token volume and dollar cost it multiplies.

Pricing is *example* pricing for llama-3.3-70b-versatile on Groq - check the
current rates in the Groq console before quoting exact dollars to anyone.
"""

from __future__ import annotations

# per million input / output tokens (example, verify live)
INPUT_PER_MT = 0.59
OUTPUT_PER_MT = 0.79


def chunk_size_tokens(texts: list[str]) -> int:
    from evaluate import est_tokens

    return est_tokens(texts)


def cost_usd(input_tok: int, output_tok: int) -> float:
    return (input_tok / 1e6 * INPUT_PER_MT) + (output_tok / 1e6 * OUTPUT_PER_MT)


def account(rows: list[dict]) -> dict:
    """Aggregate per-question cost rows into a summary dict.

    rows entries: {input_tokens, output_tokens, calls} (all ints).
    """
    tot_in = sum(r["input_tokens"] for r in rows)
    tot_out = sum(r["output_tokens"] for r in rows)
    return {
        "input_tokens": tot_in,
        "output_tokens": tot_out,
        "calls": sum(r["calls"] for r in rows),
        "usd": cost_usd(tot_in, tot_out),
    }


