# RAG Strategies

*Easy, concept-based notes from start to end*

## 1. What are RAG Strategies?

- RAG strategies are different techniques used to make retrieval and generation more accurate, relevant, and useful.
- Basic RAG follows a simple retrieve → provide context → generate flow.
- Advanced strategies improve different parts of this process, such as query understanding, retrieval, ranking, context, and validation.

**Remember:** RAG strategies = techniques used to improve a RAG system.

## 2. Reranking

- The retriever first finds multiple candidate documents or chunks.
- A reranker then checks those candidates against the user's query and puts the most relevant ones first.
- Only the best results are usually passed to the LLM.
- This can improve precision, but it adds extra processing cost.

**Remember:** Flow: Retrieve many → Rerank → Keep the best → LLM

## 3. Agentic RAG

- Traditional RAG often follows a fixed retrieval pipeline.
- Agentic RAG uses an agent that can decide dynamically what to search, which source to use, whether another search is needed, and whether the retrieved information is enough.
- The agent can modify or expand queries and perform multiple retrieval steps.
- Hybrid search can be one of the retrieval strategies used by an agent.

**Remember:** Agentic RAG = RAG + dynamic decision-making.

## 4. Chunkless vs Agentic — Two Different Ideas

- **Chunkless RAG** is about HOW the document is stored and read: no fixed-size chunks — whole paragraphs, sections, and tables stay intact as navigation units, so nothing is severed.
- **Agentic RAG** is about WHO decides what to read: an LLM agent plans, searches, reads, judges, and retries instead of a fixed top-k search, and may use multiple tools.
- They are independent axes — a system can be both at once: a chunkless corpus (whole units) guided by an agentic navigator that decides which units to read.

**Remember:** Chunkless = how text stays whole; Agentic = who controls retrieval. Not synonyms.

## 5. Knowledge Graphs

- A knowledge graph represents information using entities and relationships.
- Nodes represent entities, while edges represent relationships.
- Example: Ali → works_at → Google → located_in → USA.
- Knowledge graphs are useful for questions that require relationships or multiple steps.

**Remember:** Vector DB = similarity; Knowledge Graph = relationships.

## 6. Contextual Retrieval

- A retrieved chunk can sometimes be unclear when separated from its original document.
- Contextual retrieval adds useful document context to the chunk so that it is more understandable during retrieval.
- For example, "It increased by 20%" becomes more useful when the chunk also identifies what "it" refers to.
- The goal is to improve retrieval accuracy.

**Remember:** Chunk + useful context = better retrieval.

## 7. Query Expansion

- Query expansion makes the original user query richer by adding related words, synonyms, or phrases.
- Example: "car insurance" can be expanded with "auto insurance", "vehicle coverage", and similar terms.
- It helps the retriever find relevant information that may use different wording.
- The result is usually one richer query.

**Remember:** One question → one richer query.

## 8. Multi-Query RAG

- The LLM generates several alternative queries from the original question.
- Each query is used for retrieval.
- The retrieved results are combined and duplicates can be removed.
- This can improve recall because different queries may find different relevant information.

**Remember:** Query Expansion = one richer query. Multi-Query = multiple alternative queries.

## 9. Hybrid Search

- Hybrid search combines keyword search and vector/semantic search.
- Keyword search is useful for exact names, IDs, codes, and specific terms.
- Vector search is useful for meaning and semantically similar wording.
- Combining both can improve retrieval across different types of queries.

**Remember:** Keyword = exact; Vector = meaning; Hybrid = both.

## 10. Small-to-Big (Parent-Document) Retrieval

- Retrieval is split into two sizes instead of using one identical piece.
- Small pieces (sentences or short chunks) are embedded and matched against the query for precision.
- When a small piece matches, the system returns its larger parent unit (the whole paragraph or section) as context.
- This keeps retrieval precise while giving the LLM the full surrounding context it needs to answer.

**Remember:** Retrieve small (precise) → return big (full context).

## 11. Hierarchical RAG

- Information is organized into levels, such as company → department → document → section → chunk.
- Metadata can describe each level, such as date, author, department, document type, or category.
- Retrieval can move from broad information to more specific information.
- This is useful for large and well-structured knowledge bases.

**Remember:** Retrieve from broad → narrow → detailed.

## 12. Self-Reflective RAG

- The system checks whether the retrieved information or generated answer is sufficient.
- If the information is insufficient, it can modify the query, retrieve again, or revise the answer.
- This creates a feedback loop instead of relying on one retrieval step.
- The goal is to reduce weak or unsupported answers.

**Remember:** Retrieve → Generate → Check → Improve if needed.

## 13. CRAG (Corrective RAG)

- Like self-reflective RAG, the system evaluates the quality of what it retrieved before answering.
- If the retrieved content is weak or irrelevant, it takes corrective steps instead of answering anyway.
- The system may filter out bad chunks, run a web search, or re-retrieve from a better source.
- The goal is to avoid answers built on weak or wrong context.

**Remember:** Retrieve → Evaluate → Correct if weak → Generate.

## 14. Fine-Tuned Embeddings

- General embedding models may not represent specialized domain terminology perfectly.
- An embedding model can be adapted or fine-tuned for a specific domain or task.
- Examples include medical, legal, finance, e-commerce, or company-specific terminology.
- Better domain representations can improve semantic retrieval.

**Remember:** Adapt embeddings to the domain for better retrieval.

## 15. Advanced RAG Strategy Flow

*User Query → Query Expansion / Multi-Query → Hybrid Retrieval → Reranking → Relevant Context → LLM → Self-Reflection → Final Answer*

## Quick Revision

| Strategy | Main Idea |
|---|---|
| Reranking | Reorder retrieved candidates by relevance |
| Agentic RAG | Agent dynamically decides retrieval steps |
| Chunkless vs Agentic | Chunkless = how text stays whole; Agentic = who controls retrieval |
| Knowledge Graph | Use entities and relationships |
| Contextual Retrieval | Add useful context to retrieved chunks |
| Query Expansion | Make one query richer |
| Multi-Query RAG | Generate multiple alternative queries |
| Hybrid Search | Keyword + vector search |
| Small-to-Big (Parent-Document) | Retrieve small chunks, return larger parent context |
| Hierarchical RAG | Retrieve through information levels |
| Self-Reflective RAG | Check and improve retrieval/answer |
| CRAG (Corrective RAG) | Detect weak retrieval and correct it before answering |
| Fine-Tuned Embeddings | Adapt embeddings to a specific domain |

> **Golden Point:** Advanced RAG strategies mainly improve one or more of these areas: query understanding, retrieval, ranking, context, reasoning, and answer checking.