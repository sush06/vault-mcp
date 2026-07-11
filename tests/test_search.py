import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vault_mcp.search import _rrf_fuse  # noqa: E402


def test_rrf_fuse_prefers_items_ranked_high_by_both():
    keyword_ranked = [(1, 10.0), (2, 8.0), (3, 5.0)]
    vector_ranked = [(2, 0.9), (1, 0.7), (4, 0.5)]

    fused = _rrf_fuse(keyword_ranked, vector_ranked)

    # note 1 is #1 keyword + #2 vector, note 2 is #2 keyword + #1 vector —
    # both appear in both rankings near the top, so both should outscore
    # note 3 and note 4, which each appear in only one ranking.
    assert fused[1][0] > fused[3][0]
    assert fused[2][0] > fused[4][0]
    assert fused[1][1] == "both"
    assert fused[3][1] == "keyword"
    assert fused[4][1] == "vector"


def test_rrf_fuse_handles_disjoint_rankings():
    keyword_ranked = [(1, 10.0)]
    vector_ranked = [(2, 0.9)]
    fused = _rrf_fuse(keyword_ranked, vector_ranked)
    assert set(fused.keys()) == {1, 2}
    # symmetric — both are rank 1 in their own list, so equal score
    assert fused[1][0] == pytest.approx(fused[2][0])


def test_rrf_fuse_empty_inputs():
    assert _rrf_fuse([], []) == {}
