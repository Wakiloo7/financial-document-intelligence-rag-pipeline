from pathlib import Path


def test_prompt_template_contains_required_fields():
    template = Path("prompts/rag_prompt_template.txt").read_text(encoding="utf-8")

    assert "{question}" in template
    assert "{context}" in template