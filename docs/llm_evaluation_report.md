# LLM and RAG Evaluation Report

This document describes the evaluation approach used in the Financial Document Intelligence RAG Pipeline.

The project evaluates whether the system can retrieve relevant SEC filing evidence and produce source-grounded answers for financial disclosure questions.

## Evaluation Goals

The evaluation focuses on:

- Retrieval quality
- Context relevance
- Answer groundedness
- Citation correctness
- Hallucination risk
- Metadata filtering correctness
- Qualitative review of LLM/fine-tuning behavior

## Evaluation Dimensions

| Metric | Description |
|---|---|
| Context Relevance | Checks whether retrieved chunks directly address the user question. |
| Groundedness | Checks whether the answer is supported by the retrieved context. |
| Citation Correctness | Checks whether company, form, fiscal year, tag, and chunk ID are correctly attached to the answer. |
| Hallucination Risk | Checks whether the answer introduces facts not present in the retrieved context. |
| Metadata Filtering | Checks whether company/form filters prevent cross-company context contamination. |
| Answer Relevance | Checks whether the final answer directly addresses the user question. |

## Retrieval Evaluation

The retrieval pipeline combines:

- FAISS semantic search
- BM25 lexical search
- Reciprocal Rank Fusion
- Cross-encoder re-ranking

The strongest example query was:

Question:
What does the filing say about revenue recognition?

Filters:
company_name = SEI INVESTMENTS CO
form = 10-Q

Top retrieved chunk:

Company: SEI INVESTMENTS CO
Form: 10-Q
Fiscal Year: 2026
Filed Date: 20260427
Tag: RevenueFromContractWithCustomerPolicyTextBlock
Chunk ID: 8690
Re-rank Score: 0.9888

This result was considered high quality because the retrieved evidence directly contains the revenue recognition policy.

## Local Extractive Answer Evaluation

The local extractive mode does not call an external LLM. It creates a source-grounded answer from the top retrieved and re-ranked chunk.

Strengths:

- Fully reproducible
- No API cost
- Low hallucination risk
- Source metadata is preserved

Limitations:

- Less fluent than full LLM generation
- Mainly extractive rather than abstractive
- May require multiple chunks for broad questions

## LangChain and OpenAI Evaluation

The project includes optional LangChain and OpenAI answer generation.

When USE_LANGCHAIN_LLM=true and an OpenAI API key with available quota is configured, retrieved chunks are passed into a LangChain prompt for final answer generation.

The prompt instructs the model to:

- Answer only from retrieved context
- Avoid inventing facts, figures, dates, or conclusions
- Include source metadata
- State when the evidence is insufficient

## LLaMA Factory and LoRA Fine-tuning Evaluation

A compact LLaMA Factory and LoRA fine-tuning demo was added under:

experiments/llama_factory_lora_finetuning/

The experiment uses:

- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- LoRA supervised fine-tuning
- Alpaca-style financial RAG instruction data
- Source-grounded financial QA examples
- Qualitative evaluation examples

Training evidence:

Fine-tuning method: LoRA
Trainable parameters: 1,126,400
Training steps: 3
Train loss: 1.8529
Eval loss: 2.6567
Training completed successfully

The purpose of this experiment is not to produce a production-grade financial model, but to demonstrate the full workflow:

Instruction data preparation -> LoRA configuration -> LLaMA Factory training -> evaluation evidence

## Evaluation Output

A sample evaluation file is available at:

outputs/llm_evaluation_sample.csv

It includes:

- Question
- Expected topic
- Answer mode
- Retrieved source metadata
- Re-rank score
- Context relevance
- Groundedness
- Citation correctness
- Hallucination risk
- Qualitative notes

## Future Evaluation Improvements

Future extensions can include:

- RAGAS faithfulness
- RAGAS answer relevancy
- Context precision
- Context recall
- Human evaluation rubric
- Before/after fine-tuned model comparison
- Hallucination benchmark questions
- Citation accuracy scoring
