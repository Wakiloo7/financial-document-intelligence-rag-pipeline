from collections import defaultdict

from src.retrieval.embedding_search import EmbeddingSearch
from src.retrieval.bm25_search import BM25Search
from src.utils.config_loader import load_config


class HybridSearch:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

        self.top_k_vector = int(self.config["retrieval"]["top_k_vector"])
        self.top_k_bm25 = int(self.config["retrieval"]["top_k_bm25"])
        self.top_k_final = int(self.config["retrieval"]["top_k_final"])
        self.rrf_k = int(self.config["retrieval"]["rrf_k"])

        self.embedding_search = EmbeddingSearch(self.config)
        self.bm25_search = BM25Search(self.config)

        self.embedding_search.load()
        self.bm25_search.load()

    def reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        scores = defaultdict(float)
        records = {}

        for result in vector_results:
            chunk_id = result["chunk_id"]
            rank = result.get("vector_rank", 999)
            scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            records[chunk_id] = result

        for result in bm25_results:
            chunk_id = result["chunk_id"]
            rank = result.get("bm25_rank", 999)
            scores[chunk_id] += 1.0 / (self.rrf_k + rank)

            if chunk_id in records:
                records[chunk_id].update(result)
            else:
                records[chunk_id] = result

        fused = []

        for chunk_id, score in scores.items():
            record = records[chunk_id]
            record["rrf_score"] = score
            fused.append(record)

        fused = sorted(fused, key=lambda x: x["rrf_score"], reverse=True)
        return fused

    def search(self, query: str, filters: dict | None = None) -> list[dict]:
        vector_results = self.embedding_search.search(
            query=query,
            top_k=self.top_k_vector,
            filters=filters,
        )

        bm25_results = self.bm25_search.search(
            query=query,
            top_k=self.top_k_bm25,
            filters=filters,
        )

        fused_results = self.reciprocal_rank_fusion(vector_results, bm25_results)
        return fused_results[: self.top_k_final]