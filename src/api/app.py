from fastapi import FastAPI
from pydantic import BaseModel

from src.generation.rag_answer import FinancialRAGPipeline


app = FastAPI(
    title="Financial Document Intelligence RAG API",
    description="Hybrid-search RAG API for SEC financial statement notes and disclosures.",
    version="1.0.0",
)

rag = FinancialRAGPipeline()


class QueryRequest(BaseModel):
    question: str
    company_name: str | None = None
    form: str | None = None


@app.get("/")
def root():
    return {
        "message": "Financial Document Intelligence RAG API is running.",
        "docs": "/docs",
    }


@app.post("/ask")
def ask(request: QueryRequest):
    return rag.answer(
        question=request.question,
        company_name=request.company_name,
        form=request.form,
    )