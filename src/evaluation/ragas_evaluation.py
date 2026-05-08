from pathlib import Path
import pandas as pd

from src.generation.rag_answer import FinancialRAGPipeline


def evaluate_retrieval() -> None:
    questions_path = Path("data/evaluation/evaluation_questions.csv")

    if not questions_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {questions_path}")

    questions_df = pd.read_csv(questions_path)
    rag = FinancialRAGPipeline()

    rows = []

    for _, row in questions_df.iterrows():
        question = row["question"]
        expected_topic = row.get("expected_topic", "")

        result = rag.answer(question)

        retrieved_text = " ".join(
            [item.get("chunk_text", "") for item in result["retrieved_chunks"]]
        ).lower()

        topic_hit = str(expected_topic).lower() in retrieved_text if expected_topic else False

        rows.append(
            {
                "question": question,
                "expected_topic": expected_topic,
                "retrieved_source_count": len(result["sources"]),
                "topic_hit": topic_hit,
                "top_source": result["sources"][0] if result["sources"] else "",
            }
        )

    output_path = Path("data/processed/evaluation_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Evaluation results saved to {output_path}")


def main() -> None:
    evaluate_retrieval()


if __name__ == "__main__":
    main()