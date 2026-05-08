from src.retrieval.hybrid_search import HybridSearch


def test_rrf_combines_results_without_loading_indexes():
    hybrid = object.__new__(HybridSearch)
    hybrid.rrf_k = 60

    vector_results = [
        {"chunk_id": 1, "chunk_text": "revenue recognition", "vector_rank": 1},
        {"chunk_id": 2, "chunk_text": "leases", "vector_rank": 2},
    ]

    bm25_results = [
        {"chunk_id": 2, "chunk_text": "leases", "bm25_rank": 1},
        {"chunk_id": 3, "chunk_text": "tax", "bm25_rank": 2},
    ]

    fused = HybridSearch.reciprocal_rank_fusion(hybrid, vector_results, bm25_results)

    assert len(fused) == 3
    assert "rrf_score" in fused[0]