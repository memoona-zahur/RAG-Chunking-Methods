"""Local embeddings (all-MiniLM-L6-v2) + Qdrant in-memory store.

One shared model so every method/chunk-type produces comparable vectors. The
store is rebuilt per call - nothing persists, exactly like the kata style.
"""

from __future__ import annotations

import logging
import os
import warnings
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBED_MODEL = "all-MiniLM-L6-v2"
DIM = 384

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests")


def _quiet_tqdm(iterable=None, **kwargs):
    kwargs["disable"] = True
    return __import__("tqdm.auto", fromlist=["tqdm"]).tqdm(iterable, **kwargs)


# transformers' core_model_loading renders a "Loading weights" bar via
# transformers.utils.logging.tqdm - patch it to disabled BEFORE sentence_transformers
# is imported (the demo re-imports this module each run).
try:
    from transformers.utils import logging as _logging

    _logging.tqdm = _quiet_tqdm
except Exception:
    pass

from sentence_transformers import SentenceTransformer

# Silence library chatter so `python demo.py` shows a clean, presentable console.
# The "unauthenticated requests" notice is NOT a literal string in any package: the
# HF server returns an `X-HF-Warning` header and huggingface_hub.utils._http forwards
# it via logging. Levels must be set AFTER that module is imported (its own import
# re-applies levels, which is why the earlier pre-import setLevel never held).
for _name in (
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "transformers",
    "sentence_transformers",
    "urllib3",
    "httpx",
    "httpcore",
):
    logging.getLogger(_name).setLevel(logging.ERROR)


@lru_cache(maxsize=1)
def embedder():
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