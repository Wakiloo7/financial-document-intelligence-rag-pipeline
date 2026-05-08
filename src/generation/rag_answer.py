import os
from dotenv import load_dotenv

from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.prompt_builder import build_prompt
from src.generation.langchain_rag_chain import generate_langchain_answer
from src.utils.config_loader import load_config


load_dotenv()


class FinancialRAGPipeline:
    def __init__(self):
        self.config = load_config()
        self.hybrid_search = HybridSearch(self.config)

        self.use_reranker = bool(self.config["retrieval"]["use_reranker"])
        self.top_k_final = int(self.config["retrieval"]["top_k_final"])

        if self.use_reranker:
            self.reranker = CrossEncoderReranker(
                model_name=self.config["retrieval"]["reranker_model"]
            )

    def filter_weak_chunks(self, chunks: list[dict], min_score: float = 0.10) -> list[dict]:
        """
        Keep only high-confidence chunks after re-ranking.

        If all chunks are below threshold, keep the top chunk so the user still
        receives the best available evidence.
        """
        if not chunks:
            return []

        filtered_chunks = []

        for item in chunks:
            score = item.get("rerank_score", item.get("rrf_score", 0))

            try:
                score = float(score)
            except Exception:
                score = 0.0

            if score >= min_score:
                filtered_chunks.append(item)

        return filtered_chunks if filtered_chunks else chunks[:1]

    def answer(
        self,
        question: str,
        company_name: str | None = None,
        form: str | None = None,
    ) -> dict:
        filters = {}

        if company_name:
            filters["company_name"] = company_name

        if form:
            filters["form"] = form

        retrieved = self.hybrid_search.search(
            query=question,
            filters=filters if filters else None,
        )

        if self.use_reranker and retrieved:
            retrieved = self.reranker.rerank(
                query=question,
                retrieved_chunks=retrieved,
                top_k=self.top_k_final,
            )

        retrieved = self.filter_weak_chunks(retrieved, min_score=0.10)

        prompt = build_prompt(question, retrieved)

        sources = [
            {
                "company_name": item.get("company_name", ""),
                "form": item.get("form", ""),
                "fiscal_year": item.get("fiscal_year", ""),
                "filed_date": item.get("filed_date", ""),
                "tag": item.get("tag", ""),
                "chunk_id": item.get("chunk_id", ""),
                "score": item.get("rerank_score", item.get("rrf_score", "")),
            }
            for item in retrieved
        ]

        use_langchain_llm = os.getenv("USE_LANGCHAIN_LLM", "false").lower() == "true"

        if use_langchain_llm:
            answer_text = generate_langchain_answer(question, retrieved)
            answer_mode = "langchain_llm_generation"
        else:
            answer_text = self.generate_extractive_answer(retrieved)
            answer_mode = "local_extractive_answer"

        return {
            "question": question,
            "answer": answer_text,
            "answer_mode": answer_mode,
            "filters": filters,
            "sources": sources,
            "retrieved_chunks": retrieved,
            "llm_prompt_preview": prompt[:4000],
        }

    def generate_extractive_answer(self, chunks: list[dict]) -> str:
        """
        Free local fallback answer.

        This does not call an LLM. It summarizes the top retrieved chunk
        extractively and attaches source metadata.
        """
        if not chunks:
            return "No relevant evidence was retrieved for this question."

        top_chunk = chunks[0]

        company = top_chunk.get("company_name", "Unknown company")
        form = top_chunk.get("form", "Unknown form")
        fiscal_year = top_chunk.get("fiscal_year", "Unknown fiscal year")
        tag = top_chunk.get("tag", "Unknown tag")
        chunk_id = top_chunk.get("chunk_id", "Unknown chunk")

        text = str(top_chunk.get("chunk_text", "")).replace("\n", " ")
        sentences = text.split(". ")

        selected_sentences = sentences[:3]
        summary = ". ".join(selected_sentences).strip()

        if summary and not summary.endswith("."):
            summary += "."

        return (
            f"{summary}\n\n"
            f"Sources: {company}, {form}, FY {fiscal_year}, {tag}, Chunk ID {chunk_id}."
        )