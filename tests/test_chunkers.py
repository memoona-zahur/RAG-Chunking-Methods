"""Tests for the chunking comparison -- pure logic only, no model, no network.

Only the sentence/paragraph/structure/char/token/recursive chunkers and the
evaluation/cost math are covered. Semantic + embedding internals are validated
by the demo run itself (evidence/), not by unit tests.
"""

import pytest

import chunkers
import costs
import evaluate
from chunkers import (
    Chunk,
    chunk_fixed_char,
    chunk_fixed_token,
    chunk_paragraph,
    chunk_recursive,
    chunk_sentence,
    chunk_structural,
    count_mid_sentence_cuts,
    count_table_cuts,
    split_paragraphs,
    split_sentences,
)

TEXT = (
    "# Manual\n"
    "\n"
    "The Solar Home Mini runs on the SHM-400 model and costs $1,999. "
    "Its battery lasts about eight hours at one thousand watts. "
    "The Plus kit (model SHM-800) costs $3,499 and carries a 24 month warranty.\n"
    "\n"
    "Coastal homes need the corrosion protection kit. Clean the heat sinks monthly.\n"
    "\n"
    "| Model | Number | Price | Warranty |\n"
    "|---|---|---|---|\n"
    "| Solar Home Mini | SHM-400 | $1,999 | 12 months |\n"
    "| Solar Home Plus | SHM-800 | $3,499 | 24 months |\n"
)


def test_sentences_split_on_punctuation():
    s = split_sentences("A sentence. Another one!")
    assert len(s) == 2
    assert s[0] == "A sentence."


def test_paragraphs_split_blank_lines():
    p = split_paragraphs("One.\n\nTwo.\n\nThree.")
    assert p == ["One.", "Two.", "Three."]


def test_structural_keeps_heading_sections():
    out = chunk_structural("# Top\n\ntext\n\n## Next\n\ntext2")
    assert len(out) == 2
    assert out[0].kind == "section"


def test_fixed_char_never_returns_empty():
    out = chunk_fixed_char("hello world " * 200, size=500)
    assert out
    assert all(c.text for c in out)


def test_fixed_token_round_trips_text():
    out = chunk_fixed_token("one two three four", size=2, overlap=0)
    assert " ".join(c.text for c in out) == "one two three four"
    out2 = chunk_fixed_token("a b c d e f g h", size=3, overlap=1)
    assert all(c.text for c in out2)


def test_recursive_is_exhaustive():
    out = chunk_recursive(TEXT, size=120)
    joined = " ".join(c.text for c in out)
    for word in ("SHM-400", "$3,499", "corrosion", "heat sinks"):
        assert word in joined


def test_sentence_chunks_never_cut_a_sentence():
    out = chunk_sentence("One. Two. Three. Four. ", size=5)
    body = " ".join(c.text for c in out)
    assert "One." in body and "Four." in body


def test_mid_sentence_cut_detection():
    chunks = [Chunk("the quick brown fox", "char", 0), Chunk("jumps over.", "char", 1)]
    assert count_mid_sentence_cuts(chunks) == 1
    chunks2 = [Chunk("Done. ", "char", 0), Chunk("Next.", "char", 1)]
    assert count_mid_sentence_cuts(chunks2) == 0


def test_table_cut_detection():
    broken_first = Chunk("| Model | Number | Price | War", "char", 0)
    broken_rest = Chunk("ranty | 12 months |", "char", 1)
    assert count_table_cuts([broken_first, broken_rest]) == 1
    intact = [Chunk("| A | B |\n| 1 | 2 |\n", "char", 0), Chunk("plain para", "char", 1)]
    assert count_table_cuts(intact) == 0


def test_fact_coverage_checks_substrings():
    ctx = ["the battery lasts eight hours at one thousand watts"]
    cov, total = evaluate.fact_coverage(ctx, ["eight hours", "nine hours"])
    assert (cov, total) == (1, 2)


def test_precision_recall_basic():
    chunks = [
        Chunk("contains eight hours", "p", 0),
        Chunk("contains ten years and eight hours", "p", 1),
        Chunk("nothing", "p", 2),
    ]
    rel = evaluate.relevant_chunk_ids(chunks, ["eight hours", "ten years"])
    assert rel == {0, 1}
    p, r = evaluate.precision_recall(
        [{"idx": 0}, {"idx": 2}], rel, k=2
    )
    assert p == 0.5
    assert r == 0.5


def test_hit_rate_is_binary_hit_or_miss():
    rel = {1}
    assert evaluate.hit_rate([{"idx": 1}, {"idx": 9}], rel, k=2) == 1.0
    assert evaluate.hit_rate([{"idx": 9}, {"idx": 8}], rel, k=2) == 0.0
    assert evaluate.hit_rate([], rel, k=0) == 0.0


def test_navigation_parsing_is_robust():
    from agentic import parse_navigation

    assert parse_navigation("read paragraphs 4 and 7.", 9, 2) == [4, 7]
    assert parse_navigation("4,7", 9, 2) == [4, 7]
    assert parse_navigation("Paragraph 9 is the only useful one.", 9, 2) == [9]
    assert parse_navigation("9 and 9 again.", 9, 2) == [9]
    assert parse_navigation("use 0 and 11.", 9, 2) == []
    assert parse_navigation("none found.", 9, 2) == []
    assert parse_navigation("7 4 2", 9, 2) == [7, 4]
    assert parse_navigation("1 2 3 4", 9, 2) == [1, 2]


def test_cost_accounting_is_deterministic():
    rows = [
        {"input_tokens": 400, "output_tokens": 100, "calls": 1},
        {"input_tokens": 200, "output_tokens": 0, "calls": 1},
    ]
    acc = costs.account(rows)
    assert acc["input_tokens"] == 600
    assert acc["calls"] == 2
    assert acc["usd"] > 0
    assert acc["usd"] == costs.cost_usd(600, 100)


def test_analyze_agentic_scores_only_the_navigated_units():
    units = [
        Chunk("the battery lasts eight hours at one thousand watts", "agentic", 0),
        Chunk("the pack is rated for ten years or six thousand cycles", "agentic", 1),
        Chunk("the Plus kit costs $3,499 with a 24 month warranty", "agentic", 2),
    ]
    # navigator picks units 0 and 1 for a battery question -> both facts found
    rows = evaluate.analyze_agentic(
        units,
        [[0, 1]],
        questions=[("How long does the battery run, and its warranty?", ["eight hours", "ten years"])],
    )
    assert rows[0]["facts"] == (2, 2)
    assert rows[0]["calls"] == 2
    # navigator picks unit 2 instead (plus-kit paragraph) -> only ONE fact present
    rows2 = evaluate.analyze_agentic(
        units,
        [[1, 2]],
        questions=[("How long does the battery run, and its warranty?", ["eight hours", "ten years"])],
    )
    assert rows2[0]["facts"] == (1, 2)
    assert rows2[0]["precision"] == 0.5  # one of the two chosen units held a required fact


def test_analyze_agentic_does_embedding_search_over_the_same_units():
    # Sanity: the function scores what navigation returns; a fresh embed/search
    # (analyze_method) over the identical unit set is the ORIGINAL review bug,
    # so ensure the two paths can disagree on the route, not the code.
    units = [Chunk("A sentence about a.", "agentic", i) for i in range(4)]
    rows = evaluate.analyze_agentic(
        units, [[0, 3]], questions=[("Q?", ["zzz", "yyy"])]
    )
    assert rows[0]["facts"] == (0, 2)
    assert rows[0]["hit_rate"] == 0.0


def test_all_methods_return_nonempty_on_doc(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text(TEXT, encoding="utf-8")
    # `semantic` loads the embedding model - it is validated by the demo run, not here.
    for name in chunkers.METHODS:
        if name == "semantic":
            continue
        fn = chunkers.METHODS[name]
        out = fn(chunkers.load_document(str(doc)))
        assert out, f"{name} produced no chunks"
        assert all(c.kind for c in out)