from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()


def format_sources(chunks: list[dict]) -> str:
    source_lines = []

    for item in chunks:
        source_lines.append(
            " | ".join(
                [
                    f"Company: {item.get('company_name', 'Unknown')}",
                    f"Form: {item.get('form', 'Unknown')}",
                    f"Fiscal Year: {item.get('fiscal_year', 'Unknown')}",
                    f"Filed Date: {item.get('filed_date', 'Unknown')}",
                    f"Tag: {item.get('tag', 'Unknown')}",
                    f"Chunk ID: {item.get('chunk_id', 'Unknown')}",
                ]
            )
        )

    return "\n".join(source_lines)


def format_context(chunks: list[dict]) -> str:
    context_blocks = []

    for item in chunks:
        source = " | ".join(
            [
                f"Company: {item.get('company_name', 'Unknown')}",
                f"Form: {item.get('form', 'Unknown')}",
                f"Fiscal Year: {item.get('fiscal_year', 'Unknown')}",
                f"Filed Date: {item.get('filed_date', 'Unknown')}",
                f"Tag: {item.get('tag', 'Unknown')}",
                f"Chunk ID: {item.get('chunk_id', 'Unknown')}",
            ]
        )

        context_blocks.append(
            f"[Source]\n{source}\n\n[Content]\n{item.get('chunk_text', '')}"
        )

    return "\n\n---\n\n".join(context_blocks)


def generate_langchain_answer(question: str, chunks: list[dict]) -> str:
    """
    Generate a source-grounded answer using LangChain + OpenAI.

    Retrieval is handled before this function by:
    FAISS semantic search + BM25 keyword search + RRF + cross-encoder re-ranking.
    This function only receives the final retrieved chunks and generates an answer.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        return (
            "LangChain LLM generation is enabled, but OPENAI_API_KEY is missing. "
            "Please configure OPENAI_API_KEY in your local .env file."
        )

    context = format_context(chunks)
    sources = format_sources(chunks)

    system_prompt_path = Path("prompts/system_prompt.txt")

    if system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = (
            "You are a financial document intelligence assistant. "
            "Answer only using the retrieved context. "
            "Do not invent financial facts, figures, dates, or conclusions."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "user",
                """
Question:
{question}

Retrieved Financial Context:
{context}

Source Metadata:
{sources}

Instructions:
- Answer only using the retrieved financial context.
- Do not invent figures, dates, risks, accounting policies, or conclusions.
- If the context is insufficient, say that the available evidence is insufficient.
- Keep the answer concise and business-readable.
- End with a "Sources" line listing company, form, fiscal year, tag, and chunk id.

Answer:
""",
            ),
        ]
    )

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        api_key=api_key,
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
            "sources": sources,
        }
    )

    return response.content