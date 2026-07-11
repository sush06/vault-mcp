"""Local embedding model wrapper — sentence-transformers' all-MiniLM-L6-v2.
Small, fast on CPU, no API key needed."""

from __future__ import annotations

import math
from functools import lru_cache

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


def embed(text: str) -> list[float]:
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    # Vectors from sentence-transformers are already L2-normalized, so dot
    # product == cosine similarity — but computed explicitly here so this
    # function is correct even if callers pass in un-normalized vectors.
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
