# RAG Chunking Strategies

*Easy, concept-based notes from start to end*

## 1. What is Chunking?

- Chunking means breaking a large document into smaller, meaningful pieces called chunks.
- These chunks are later converted into embeddings and stored for retrieval in a RAG system.
- The main goal is to make each chunk useful enough to retrieve, while keeping enough context to understand its meaning.

*Simple flow:* Document → Chunks → Embeddings → Vector Database → Retrieval

## 2. Why is Chunking Important?

- Large documents may contain much more information than the query needs.
- Smaller, meaningful chunks make retrieval more focused.
- Very small chunks can lose important context.
- Very large chunks can contain unnecessary information and increase token usage.
- So, good chunking tries to balance context, relevance, and size.

## 3. Fixed-Character Chunking

- The document is divided into chunks of a fixed number of characters (roughly letters/words).
- Example: every 500 characters becomes one chunk.
- It is simple, fast, and easy to implement.
- A weakness is that it may split a sentence, paragraph, or idea in the middle.

**Remember:** Characters decide the boundary.

## 4. Fixed-Token Chunking

- The document is divided into chunks of a fixed number of tokens, where tokens are the small word pieces an LLM actually counts and bills for.
- Example: every 100 tokens becomes one chunk.
- Cost becomes easy to predict because the LLM bills by tokens, not by characters.
- The trade-off is the same as fixed-character: boundaries can still land mid-sentence.

**Remember:** Tokens decide the boundary, and tokens are what the LLM bills by.

## 5. Sentence-Based Chunking

- The document is divided using sentence boundaries.
- This helps keep complete sentences together.
- It is usually more meaningful than blindly cutting text at a fixed character position.
- However, a group of individually complete sentences may still lose the larger topic.

**Remember:** Sentences decide the boundary.

## 6. Paragraph-Based Chunking

- The document is divided according to paragraphs.
- A paragraph often represents one related idea.
- This can preserve more meaning than very small fixed-size chunks.
- The main limitation is that paragraph sizes can be very different.

**Remember:** Paragraphs decide the boundary.

## 7. Recursive Chunking

- Recursive chunking tries to split text using natural boundaries before making smaller cuts.
- A typical approach may try: document/section → paragraph → sentence → smaller text.
- The idea is to preserve meaningful structure as much as possible.
- It is useful when documents have different paragraph and sentence sizes.

**Remember:** Try larger natural boundaries first, then split further only when needed.

## 8. Semantic Chunking

- Semantic chunking creates chunks based on meaning rather than only size.
- Sentences discussing the same topic can stay together.
- A new chunk can begin when the meaning/topic changes significantly.
- It can improve retrieval because each chunk represents a more coherent idea.

**Remember:** Meaning decides the boundary.

## 9. Overlapping Chunks

- A small part of one chunk is repeated in the next chunk.
- Example: Chunk 1 contains A B C D E, while Chunk 2 starts with D E F G H.
- The overlap helps preserve information that appears near a chunk boundary.
- Too much overlap can increase storage and token usage.

**Remember:** Overlap helps protect context at boundaries.

## 10. Document-Based Chunking

- Chunks are created according to the document's natural structure.
- Examples include headings, sections, chapters, pages, and paragraphs.
- This works especially well when the document has a clear structure.
- The document structure, rather than an arbitrary size, guides the boundaries.

**Remember:** Document structure decides the boundary.

## 11. Chunkless RAG (No-Chunk)

- There are no chunks at all — the document is not split into size-based pieces.
- The system keeps whole paragraphs, sections, and tables intact as navigation units.
- The reader/navigator moves through these units and reads the chosen ones completely, so a sentence or a table row is never cut in half.
- The trade-off: whole pieces can be large, so each context costs more tokens.

**Remember:** No splitting — whole pieces stay intact, so nothing is ever severed.

## 12. Context-Aware Chunking

- The chunk is created while considering the context needed to understand it.
- A sentence such as "It increased by 20%" may be unclear by itself.
- Keeping the relevant subject or surrounding information makes the chunk easier to understand and retrieve.
- The goal is to avoid creating isolated chunks that lose necessary context.

**Remember:** Keep the context needed to understand the chunk.

## 13. Late Chunking

- Traditional approach: split the document first, then create embeddings.
- Late chunking first represents the document with broader context and creates chunk representations afterward.
- This can help individual chunks retain more document-level context.
- It is particularly useful when the meaning of a passage depends on information elsewhere in the document.

**Remember:** Traditional = chunk first → embed. Late chunking = encode context first → create chunk representations later.

## Quick Revision

| Strategy | Main Idea |
|---|---|
| Fixed-character | Fixed number of characters |
| Fixed-token | Fixed number of tokens (what the LLM bills by) |
| Sentence-based | Split at sentence boundaries |
| Paragraph-based | Split at paragraph boundaries |
| Recursive | Use natural boundaries from larger to smaller |
| Semantic | Meaning/topic decides the boundary |
| Overlap | Repeat boundary content between chunks |
| Document-based | Use headings/sections/chapters/structure |
| Chunkless (no-chunk) | No splitting — whole units stay intact |
| Context-aware | Keep necessary context with the chunk |
| Late chunking | Create chunk representations after broader document encoding |

> **Golden Point:** There is no single chunking strategy that is perfect for every document. The best approach depends on document structure, content, retrieval needs, and how much context each chunk requires.