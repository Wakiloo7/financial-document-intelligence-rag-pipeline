import pandas as pd


def detect_table_like_text(text: str) -> bool:
    if not text:
        return False

    number_count = sum(char.isdigit() for char in text)
    comma_count = text.count(",")
    dollar_count = text.count("$")

    return number_count > 30 and (comma_count > 5 or dollar_count > 2)


def table_text_to_markdown_preview(text: str) -> str:
    """
    Lightweight placeholder for table-to-markdown conversion.

    SEC Financial Statement and Notes datasets already contain extracted disclosures.
    For PDF annual reports, this module can later be replaced with Unstructured,
    Camelot, Tabula, or Marker-based table extraction.
    """
    if not detect_table_like_text(text):
        return text

    return f"Possible financial table or numeric disclosure:\n\n{text}"