from pathlib import Path
import pickle
import re

import pandas as pd
from rank_bm25 import BM25Okapi

from src.utils.config_loader import load_config
from src.utils.logger import get_logger


logger = get_logger("bm25_search")


def tokenize(text: str) -> list[str]:
    text = str(text).lower()
    return re.findall(r"[a-zA-Z0-9_]+", text)


class BM25Search:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.chunks_file = Path(self.config["paths"]["chunks_file"])
        self.bm25_store = Path(self.config["paths"]["bm25_store"])
        self.metadata_file = Path(self.config["paths"]["chunk_metadata_file"])

    def build(self) -> None:
        chunks_df = pd.read_csv(self.chunks_file)
        tokenized_corpus = [tokenize(text) for text in chunks_df["chunk_text"].fillna("").tolist()]

        bm25 = BM25Okapi(tokenized_corpus)

        self.bm25_store.parent.mkdir(parents=True, exist_ok=True)

        with self.bm25_store.open("wb") as file:
            pickle.dump(
                {
                    "bm25": bm25,
                    "metadata": chunks_df,
                },
                file,
            )

        logger.info(f"Saved BM25 index to {self.bm25_store}")

    def load(self) -> None:
        with self.bm25_store.open("rb") as file:
            payload = pickle.load(file)

        self.bm25 = payload["bm25"]
        self.metadata = payload["metadata"]

    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = scores.argsort()[::-1]

        results = []

        for rank, idx in enumerate(ranked_indices, start=1):
            row = self.metadata.iloc[idx].to_dict()

            if filters:
                skip = False
                for key, value in filters.items():
                    if value and str(row.get(key, "")).lower() != str(value).lower():
                        skip = True
                        break
                if skip:
                    continue

            row["bm25_rank"] = rank
            row["bm25_score"] = float(scores[idx])
            results.append(row)

            if len(results) >= top_k:
                break

        return results


def main() -> None:
    bm25 = BM25Search()
    bm25.build()


if __name__ == "__main__":
    main()