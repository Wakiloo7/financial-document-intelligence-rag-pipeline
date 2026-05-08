from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from src.utils.config_loader import load_config
from src.utils.logger import get_logger


logger = get_logger("embedding_search")


class EmbeddingSearch:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

        self.model_name = self.config["embedding"]["model_name"]
        self.normalize_embeddings = bool(self.config["embedding"]["normalize_embeddings"])

        self.chunks_file = Path(self.config["paths"]["chunks_file"])
        self.embeddings_file = Path(self.config["paths"]["embeddings_file"])
        self.vector_store_path = Path(self.config["paths"]["vector_store"])
        self.metadata_file = Path(self.config["paths"]["chunk_metadata_file"])

        self.model = SentenceTransformer(self.model_name)

    def create_embeddings(self) -> None:
        chunks_df = pd.read_csv(self.chunks_file)

        texts = chunks_df["chunk_text"].fillna("").tolist()

        logger.info(f"Creating embeddings for {len(texts)} chunks using {self.model_name}")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype("float32")

        self.embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_file, embeddings)

        chunks_df.to_csv(self.metadata_file, index=False)

        logger.info(f"Saved embeddings to {self.embeddings_file}")
        logger.info(f"Saved metadata to {self.metadata_file}")

    def build_faiss_index(self) -> None:
        embeddings = np.load(self.embeddings_file).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.vector_store_path))

        logger.info(f"Saved FAISS index with {index.ntotal} vectors to {self.vector_store_path}")

    def load(self) -> None:
        self.index = faiss.read_index(str(self.vector_store_path))
        self.metadata = pd.read_csv(self.metadata_file)

    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k * 3)

        results = []

        for rank, idx in enumerate(indices[0], start=1):
            if idx < 0:
                continue

            row = self.metadata.iloc[idx].to_dict()

            if filters:
                skip = False
                for key, value in filters.items():
                    if value and str(row.get(key, "")).lower() != str(value).lower():
                        skip = True
                        break
                if skip:
                    continue

            row["vector_rank"] = rank
            row["vector_score"] = float(scores[0][rank - 1])
            results.append(row)

            if len(results) >= top_k:
                break

        return results


def main() -> None:
    searcher = EmbeddingSearch()
    searcher.create_embeddings()
    searcher.build_faiss_index()


if __name__ == "__main__":
    main()