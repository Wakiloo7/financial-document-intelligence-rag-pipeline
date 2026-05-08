def chunk_text(
    text: str,
    chunk_size_words: int = 300,
    overlap_words: int = 60,
    min_chunk_words: int = 50,
) -> list[str]:
    words = text.split()

    if len(words) < min_chunk_words:
        return []

    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]

        if len(chunk_words) >= min_chunk_words:
            chunks.append(" ".join(chunk_words))

        start += chunk_size_words - overlap_words

    return chunks