"""Eight chunking strategies for RAG, one long paragraph-based document.

Each method returns a list of Chunk namedtuples (text, kind, tag). The methods
are deliberately dependency-light and deterministic (semantic chunking excepted,
which needs the local embedding model). The 'agentic' method is the far end of
the spectrum: it does NOT split at all - it keeps whole paragraphs/sections as
navigation units, which is exactly what the chunkless/agentic approach relies on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
MARKDOWN_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
# Separators in priority order, mirroring RecursiveCharacterTextSplitter.
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    text: str
    kind: str          # char | token | sentence | paragraph | section | recursive | semantic | agentic
    idx: int


def load_document(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation. Table lines survive as units."""
    parts = [p.strip() for p in SENTENCE_BREAK.split(text) if p.strip()]
    # merge a trailing table line onto the sentence that introduced it
    merged: list[str] = []
    for p in parts:
        if merged and MARKDOWN_TABLE_LINE.match(p):
            merged[-1] += "\n" + p
            continue
        merged.append(p)
    return merged


def split_paragraphs(text: str) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    blocks = _merge_headings(blocks)
    # treat each markdown table as one block of its own (keep rows together)
    out: list[str] = []
    for block in blocks:
        if MARKDOWN_TABLE_LINE.match(block.splitlines()[0] if block else ""):
            rows = [r for r in block.splitlines() if r.strip().startswith("|")]
            out.append("\n".join(rows))
            continue
        out.append(block)
    return out


def _merge_headings(blocks: list[str]) -> list[str]:
    """Prefix each heading onto the block that follows it, so no unit is a bare title."""
    merged: list[str] = []
    pending: str | None = None
    for b in blocks:
        if b.startswith("#"):
            pending = b
        else:
            merged.append((pending + "\n\n" + b) if pending else b)
            pending = None
    if pending:
        merged.append(pending)
    return merged


def _word_safe(text: str, limit: int) -> str:
    """Cut at the nearest space at or before ``limit`` so words stay intact."""
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit + 1)
    return text[: cut if cut > 0 else limit]


def chunk_fixed_char(text: str, size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Fixed-size by characters. Boundaries land inside sentences/tables."""
    chunks: list[Chunk] = []
    start, i = 0, 0
    while start < len(text):
        cut = _word_safe(text[start:], size)
        pieces = cut.strip()
        if not pieces:
            break
        chunks.append(Chunk(pieces, "char", i))
        end = start + len(cut)          # consumed length INCLUDING stripped whitespace
        if end >= len(text):
            break
        nxt = max(0, end - overlap)
        if nxt <= start:                # safety: always make progress
            nxt = start + 1
        start, i = nxt, i + 1
    return chunks


def chunk_fixed_token(text: str, size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Fixed-size by token (word count stands in for real tokens, kata-style)."""
    words = text.split(" ")
    chunks: list[Chunk] = []
    step = size - overlap
    for i in range(0, len(words), step):
        piece = " ".join(words[i : i + size]).strip()
        if piece:
            chunks.append(Chunk(piece, "token", i // step))
    return chunks


def chunk_sentence(text: str, size: int = 500) -> list[Chunk]:
    """Sentence-aware: split at sentence boundaries, group up to a char budget."""
    sents = split_sentences(text)
    chunks: list[Chunk] = []
    buf, i = "", 0
    for s in sents:
        if buf and len(buf) + len(s) > size:
            chunks.append(Chunk(buf.strip(), "sentence", i))
            i += 1
            buf = s
        else:
            buf = (buf + " " + s).strip()
    if buf:
        chunks.append(Chunk(buf.strip(), "sentence", i))
    return chunks


def chunk_paragraph(text: str) -> list[Chunk]:
    """Paragraph-aware: one chunk per paragraph/topic block."""
    return [Chunk(b, "paragraph", i) for i, b in enumerate(split_paragraphs(text))]


def chunk_structural(text: str) -> list[Chunk]:
    """Structure-aware: split on headings, keeping tables attached to their section."""
    raw = re.split(r"(?=^## )", text, flags=re.M)
    return [Chunk(b.strip(), "section", i) for i, b in enumerate(raw) if b.strip()]


def chunk_recursive(text: str, size: int = 600, overlap: int = 60) -> list[Chunk]:
    """Recursive: try separators highest-priority first, splitting until all pieces fit."""
    pieces: list[str] = []

    def _split(chunk: str):
        if len(chunk) <= size:
            pieces.append(chunk)
            return
        for sep in RECURSIVE_SEPARATORS:
            if not sep:
                continue
            idx = chunk.rfind(sep, 0, size + 1)
            if idx > 0:
                _split(chunk[: idx + len(sep)].rstrip())
                _split(chunk[idx + len(sep) :].lstrip())
                return
        _split(chunk[:size])
        _split(chunk[size:])

    _split(text)
    out: list[Chunk] = []
    prev_tail = ""
    for p in pieces:
        if not p:
            continue
        if overlap and prev_tail:
            p = prev_tail + p
        out.append(Chunk(p.strip(), "recursive", len(out)))
        prev_tail = p[-overlap:] if len(p) > overlap else ""
    return out


def chunk_semantic(text: str, threshold: float = 0.45) -> list[Chunk]:
    """Semantic: embed sentences with the local model, break where meaning shifts.

    Consecutive sentences are merged while their cosine similarity to the running
    centroid stays above ``threshold``; a drop starts a new chunk.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sents = split_sentences(text)
    emb = model.encode(sents, normalize_embeddings=True)
    chunks: list[Chunk] = []
    cur: list[str] = [sents[0]]
    centroid = emb[0].copy()
    for i in range(1, len(sents)):
        sim = float(centroid @ emb[i])
        if sim >= threshold:
            cur.append(sents[i])
            centroid = (centroid * len(cur) + emb[i]) / (len(cur) + 1)
        else:
            chunks.append(Chunk(" ".join(cur), "semantic", len(chunks)))
            cur = [sents[i]]
            centroid = emb[i].copy()
    if cur:
        chunks.append(Chunk(" ".join(cur), "semantic", len(chunks)))
    return chunks


def chunk_agentic(text: str) -> list[Chunk]:
    """Chunkless far end: no splitting at all - whole paragraphs are the units.

    The 'agent' navigates these units (see agentic.py) and reads the chosen ones
    COMPLETELY, so nothing is ever severed by a boundary.
    """
    return [Chunk(b, "agentic", i) for i, b in enumerate(split_paragraphs(text))]


METHODS: dict[str, callable] = {
    "fixed_char": chunk_fixed_char,
    "fixed_token": chunk_fixed_token,
    "sentence": chunk_sentence,
    "paragraph": chunk_paragraph,
    "structural": chunk_structural,
    "recursive": chunk_recursive,
    "semantic": chunk_semantic,
    "agentic": chunk_agentic,
}

PURPOSE: dict[str, str] = {
    "fixed_char": "Every N characters, no regard for meaning - the baseline that cuts sentences/tables.",
    "fixed_token": "Every N tokens (word-count stand-in) - the token-tuned sibling of fixed_char.",
    "sentence": "Boundaries land between sentences, never inside one.",
    "paragraph": "One chunk per paragraph - keeps an idea-unit intact, but paragraphs can be large.",
    "structural": "Respects headings and keeps tables attached to their section.",
    "recursive": "Tries structure first (paragraphs, then sentences) before falling back to raw cuts.",
    "semantic": "Breaks where meaning shifts, using embedding similarity - no fixed sizes at all.",
    "agentic": "No chunks - whole paragraphs are navigation units read in full (chunkless RAG).",
}


def count_mid_sentence_cuts(chunks: list[Chunk]) -> int:
    """Number of boundaries that land inside a sentence (not after sentence-ending punctuation)."""
    cuts = 0
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text.rstrip()
        head = b.text.lstrip()
        if not tail or not head:
            continue
        ends_sentence = tail[-1:] in ".!?"
        starts_fresh = head[0].isupper() or head.startswith(("|", "#", "-"))
        if not ends_sentence and not starts_fresh:
            cuts += 1
    return cuts


def count_table_cuts(chunks: list[Chunk]) -> int:
    """Number of boundaries that split a markdown table row mid-line."""
    cuts = 0
    for a, b in zip(chunks, chunks[1:]):
        tail_lines = [l for l in a.text.splitlines() if l]
        if not tail_lines:
            continue
        last = tail_lines[-1]
        next_head = b.text.lstrip().splitlines()[0] if b.text.strip() else ""
        if last.startswith("|") and not last.endswith("|"):
            cuts += 1
        elif last.startswith("|") and next_head.startswith("|"):
            cuts += 1
    return cuts