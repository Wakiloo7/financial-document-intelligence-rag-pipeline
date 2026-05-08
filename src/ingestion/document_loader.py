from pathlib import Path
import pandas as pd

from src.preprocessing.text_cleaning import normalize_financial_text
from src.preprocessing.chunking import chunk_text
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


logger = get_logger("document_loader")


POSSIBLE_TEXT_COLUMNS = [
    "txt",
    "text",
    "value",
    "footnote",
    "note",
    "content",
    "disclosure",
]


def read_submission_metadata(sec_data_path: Path, sub_file: str) -> pd.DataFrame:
    sub_path = sec_data_path / sub_file

    if not sub_path.exists():
        raise FileNotFoundError(f"Missing SEC metadata file: {sub_path}")

    logger.info(f"Reading submission metadata from {sub_path}")

    sub_df = pd.read_csv(
        sub_path,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    keep_cols = [
        col for col in [
            "adsh",
            "cik",
            "name",
            "sic",
            "countryba",
            "stprba",
            "cityba",
            "form",
            "period",
            "fy",
            "fp",
            "filed",
        ]
        if col in sub_df.columns
    ]

    return sub_df[keep_cols].drop_duplicates("adsh")


def detect_text_column(columns: list[str]) -> str:
    normalized_columns = {col.lower(): col for col in columns}

    for candidate in POSSIBLE_TEXT_COLUMNS:
        if candidate.lower() in normalized_columns:
            return normalized_columns[candidate.lower()]

    raise ValueError(
        "Could not detect the text column in txt.tsv. "
        f"Available columns are: {list(columns)}"
    )


def process_txt_file(config: dict) -> pd.DataFrame:
    sec_data_path = Path(config["paths"]["sec_notes_data"])
    txt_path = sec_data_path / config["sec_files"]["txt"]
    sub_file = config["sec_files"]["sub"]

    processed_path = Path(config["paths"]["processed"])
    processed_path.mkdir(parents=True, exist_ok=True)

    chunks_file = Path(config["paths"]["chunks_file"])

    if not txt_path.exists():
        raise FileNotFoundError(f"Missing SEC txt file: {txt_path}")

    sub_df = read_submission_metadata(sec_data_path, sub_file)

    chunksize = int(config["processing"]["pandas_chunksize"])
    max_rows = int(config["processing"]["max_rows_from_txt"])
    min_text_length = int(config["processing"]["min_text_length"])

    chunk_size_words = int(config["chunking"]["chunk_size_words"])
    overlap_words = int(config["chunking"]["chunk_overlap_words"])
    min_chunk_words = int(config["chunking"]["min_chunk_words"])

    logger.info(f"Reading SEC note text from {txt_path}")
    logger.info(f"Processing up to {max_rows} rows from txt.tsv")

    all_chunks = []
    total_rows = 0
    global_chunk_id = 1
    text_column = None

    reader = pd.read_csv(
        txt_path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="ignore",
    )

    for txt_batch in reader:
        if text_column is None:
            logger.info(f"Detected txt.tsv columns: {list(txt_batch.columns)}")
            text_column = detect_text_column(list(txt_batch.columns))
            logger.info(f"Using text column: {text_column}")

        total_rows += len(txt_batch)

        if "adsh" in txt_batch.columns and "adsh" in sub_df.columns:
            txt_batch = txt_batch.merge(sub_df, on="adsh", how="left")
        else:
            logger.warning("Column 'adsh' not found. Proceeding without submission metadata join.")

        for _, row in txt_batch.iterrows():
            raw_text = row.get(text_column, "")
            cleaned_text = normalize_financial_text(raw_text)

            if len(cleaned_text) < min_text_length:
                continue

            chunks = chunk_text(
                cleaned_text,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
                min_chunk_words=min_chunk_words,
            )

            for position, chunk in enumerate(chunks, start=1):
                all_chunks.append(
                    {
                        "chunk_id": global_chunk_id,
                        "adsh": row.get("adsh", ""),
                        "tag": row.get("tag", ""),
                        "version": row.get("version", ""),
                        "ddate": row.get("ddate", ""),
                        "qtrs": row.get("qtrs", ""),
                        "iprx": row.get("iprx", ""),
                        "lang": row.get("lang", ""),
                        "dcml": row.get("dcml", ""),
                        "durp": row.get("durp", ""),
                        "datp": row.get("datp", ""),
                        "company_name": row.get("name", ""),
                        "cik": row.get("cik", ""),
                        "form": row.get("form", ""),
                        "fiscal_year": row.get("fy", ""),
                        "fiscal_period": row.get("fp", ""),
                        "filed_date": row.get("filed", ""),
                        "chunk_position": position,
                        "chunk_text": chunk,
                    }
                )
                global_chunk_id += 1

        logger.info(
            f"Processed {total_rows} rows so far. Created {len(all_chunks)} chunks."
        )

        if total_rows >= max_rows:
            break

    if not all_chunks:
        raise ValueError(
            "No chunks were created. Check the text column, min_text_length, and SEC txt.tsv content."
        )

    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv(chunks_file, index=False)

    logger.info(f"Created {len(chunks_df)} chunks.")
    logger.info(f"Saved chunks to {chunks_file}")

    return chunks_df


def main() -> None:
    config = load_config()
    process_txt_file(config)


if __name__ == "__main__":
    main()