import argparse

from src.ingestion.document_loader import main as ingest_main
from src.retrieval.embedding_search import main as embedding_main
from src.retrieval.bm25_search import main as bm25_main
from src.evaluation.ragas_evaluation import main as evaluation_main
from src.generation.rag_answer import FinancialRAGPipeline


def run_query(question: str, company_name: str | None = None, form: str | None = None) -> None:
    rag = FinancialRAGPipeline()
    result = rag.answer(question=question, company_name=company_name, form=form)

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(source)

    print("\nPrompt Preview:")
    print(result["llm_prompt_preview"][:2000])


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Document Intelligence RAG Pipeline")

    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "ingest",
            "embeddings",
            "bm25",
            "evaluate",
            "query",
            "all",
        ],
    )

    parser.add_argument("--question", required=False)
    parser.add_argument("--company", required=False)
    parser.add_argument("--form", required=False)

    args = parser.parse_args()

    if args.task == "ingest":
        ingest_main()

    elif args.task == "embeddings":
        embedding_main()

    elif args.task == "bm25":
        bm25_main()

    elif args.task == "evaluate":
        evaluation_main()

    elif args.task == "query":
        if not args.question:
            raise ValueError("--question is required when task=query")

        run_query(
            question=args.question,
            company_name=args.company,
            form=args.form,
        )

    elif args.task == "all":
        ingest_main()
        embedding_main()
        bm25_main()
        evaluation_main()


if __name__ == "__main__":
    main()