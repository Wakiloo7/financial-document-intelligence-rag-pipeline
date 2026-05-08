from src.preprocessing.chunking import chunk_text


def test_chunking_creates_multiple_chunks():
    text = " ".join(["financial"] * 1000)
    chunks = chunk_text(text, chunk_size_words=200, overlap_words=50, min_chunk_words=50)

    assert len(chunks) > 1


def test_chunking_drops_short_text():
    text = "short text"
    chunks = chunk_text(text, chunk_size_words=200, overlap_words=50, min_chunk_words=50)

    assert chunks == []