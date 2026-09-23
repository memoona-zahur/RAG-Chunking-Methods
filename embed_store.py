"""Local embeddings (all-MiniLM-L6-v2) + Qdrant in-memory store.

One shared model so every method/chunk-type produces comparable vectors. The
store is rebuilt per call - nothing persists, exactly like the kata style.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBED_MODEL = "all-MiniLM-L6-v2"
DIM = 384


@lru_cache(maxsize=1)
def embedder():
    import logging

    from sentence_transformers import SentenceTransformer

    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def build_store(chunks, method: str) -> QdrantClient:
    """Embed every chunk and index it (plus its text + method) in an in-memory store."""
    texts = [c.text for c in chunks]
    vectors = embed(texts)
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="chunks",
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name="chunks",
        points=[
            PointStruct(
                id=c.idx,
                vector=v,
                payload={"text": c.text, "kind": c.kind, "method": method, "idx": c.idx},
            )
            for c, v in zip(chunks, vectors)
        ],
    )
    return client


def search(client: QdrantClient, query: str, k: int = 3) -> list[dict]:
    qv = embed([query])[0]
    hits = client.query_points(collection_name="chunks", query=qv, limit=k).points
    return [
        {"idx": h.payload["idx"], "text": h.payload["text"], "score": h.score}
        for h in hits
    ]