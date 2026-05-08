from flashrank import Ranker, RerankRequest


class CrossEncoderReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, retrieved_chunks: list[dict], top_k: int = 5) -> list[dict]:
        passages = []

        for item in retrieved_chunks:
            passages.append(
                {
                    "id": str(item["chunk_id"]),
                    "text": item["chunk_text"],
                    "meta": item,
                }
            )

        request = RerankRequest(query=query, passages=passages)
        reranked = self.ranker.rerank(request)

        results = []

        for rank, item in enumerate(reranked[:top_k], start=1):
            record = item["meta"]
            record["rerank_rank"] = rank
            record["rerank_score"] = float(item["score"])
            results.append(record)

        return results