import re
import html


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = html.unescape(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def normalize_financial_text(text: str) -> str:
    text = clean_text(text)

    replacements = {
        "consolidated statements": "Consolidated Statements",
        "risk factors": "Risk Factors",
        "fair value": "Fair Value",
        "revenue recognition": "Revenue Recognition",
    }

    for old, new in replacements.items():
        text = re.sub(old, new, text, flags=re.IGNORECASE)

    return text