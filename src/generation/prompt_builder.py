from pathlib import Path


def load_prompt_template(path: str = "prompts/rag_prompt_template.txt") -> str:
    return Path(path).read_text(encoding="utf-8")


def build_context(chunks: list[dict]) -> str:
    context_blocks = []

    for item in chunks:
        source = (
            f"Company: {item.get('company_name', 'Unknown')} | "
            f"Form: {item.get('form', 'Unknown')} | "
            f"FY: {item.get('fiscal_year', 'Unknown')} | "
            f"Filed: {item.get('filed_date', 'Unknown')} | "
            f"Tag: {item.get('tag', 'Unknown')} | "
            f"Chunk ID: {item.get('chunk_id', 'Unknown')}"
        )

        context_blocks.append(
            f"[Source]\n{source}\n\n[Content]\n{item.get('chunk_text', '')}"
        )

    return "\n\n---\n\n".join(context_blocks)


def build_prompt(question: str, chunks: list[dict]) -> str:
    template = load_prompt_template()
    context = build_context(chunks)

    return template.format(
        question=question,
        context=context,
    )