"""BM25 keyword search + embedding similarity, fused with Reciprocal Rank
Fusion (RRF) instead of a weighted blend — BM25 and cosine similarity
aren't on comparable scales, so RRF combines them by rank instead."""

from __future__ import annotations

from dataclasses import dataclass

from . import embeddings
from .storage import VaultStore

RRF_K = 60  # standard smoothing constant from the original RRF paper


@dataclass
class SearchResult:
    note_id: int
    title: str
    snippet: str
    score: float
    matched_by: str  # "keyword", "vector", or "both"


def _vector_search(store: VaultStore, query_vec: list[float], limit: int) -> list[tuple[int, float]]:
    import numpy as np

    all_emb = store.all_embeddings()
    if not all_emb:
        return []
    note_ids = list(all_emb.keys())
    # Stack all vectors into a matrix: shape (N, dim)
    matrix = np.array(list(all_emb.values()), dtype=np.float32)
    query_arr = np.array(query_vec, dtype=np.float32)
    # Vectors from sentence-transformers are L2-normalised, so dot == cosine sim.
    # One BLAS call instead of N Python iterations.
    scores: list[float] = (matrix @ query_arr).tolist()
    scored = sorted(zip(note_ids, scores), key=lambda x: x[1], reverse=True)
    return scored[:limit]


def _rrf_fuse(
    keyword_ranked: list[tuple[int, float]],
    vector_ranked: list[tuple[int, float]],
    k: int = RRF_K,
) -> dict[int, tuple[float, str]]:
    fused: dict[int, float] = {}
    origin: dict[int, set[str]] = {}

    for rank, (note_id, _score) in enumerate(keyword_ranked, start=1):
        fused[note_id] = fused.get(note_id, 0.0) + 1.0 / (k + rank)
        origin.setdefault(note_id, set()).add("keyword")

    for rank, (note_id, _score) in enumerate(vector_ranked, start=1):
        fused[note_id] = fused.get(note_id, 0.0) + 1.0 / (k + rank)
        origin.setdefault(note_id, set()).add("vector")

    result = {}
    for note_id, score in fused.items():
        origins = origin[note_id]
        label = "both" if len(origins) == 2 else next(iter(origins))
        result[note_id] = (score, label)
    return result


def _snippet(content: str, length: int = 160) -> str:
    flat = " ".join(content.split())
    return flat[:length] + ("…" if len(flat) > length else "")


def search(
    store: VaultStore,
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    candidate_pool: int = 20,
) -> list[SearchResult]:
    """mode: 'hybrid' (default), 'keyword', or 'vector' — the latter two
    exist mainly so the eval harness can compare strategies head-to-head."""

    keyword_ranked = store.keyword_search(query, limit=candidate_pool) if mode in ("hybrid", "keyword") else []
    vector_ranked = (
        _vector_search(store, embeddings.embed(query), limit=candidate_pool)
        if mode in ("hybrid", "vector")
        else []
    )

    if mode == "keyword":
        fused = {nid: (score, "keyword") for nid, score in keyword_ranked}
    elif mode == "vector":
        fused = {nid: (score, "vector") for nid, score in vector_ranked}
    else:
        fused = _rrf_fuse(keyword_ranked, vector_ranked)

    ranked_ids = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]

    results = []
    for note_id, (score, label) in ranked_ids:
        note = store.get_note(note_id)
        if note is None:
            continue
        results.append(
            SearchResult(
                note_id=note.id,
                title=note.title,
                snippet=_snippet(note.content),
                score=score,
                matched_by=label,
            )
        )
    return results
