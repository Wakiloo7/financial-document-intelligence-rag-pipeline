# Financial Document Intelligence RAG Pipeline

End-to-end financial document intelligence pipeline using **SEC Financial Statement and Notes data**, **hybrid retrieval**, **FAISS vector search**, **BM25 keyword search**, **cross-encoder re-ranking**, **LangChain**, **optional OpenAI LLM generation**, **FastAPI**, **Docker**, **LLaMA Factory + LoRA fine-tuning**, **LLM/RAG evaluation**, and **CLIP multimodal retrieval**.

This project demonstrates how financial disclosure text can be ingested, cleaned, chunked, indexed, retrieved, re-ranked, filtered, evaluated, and served through an API for source-grounded financial question answering.

The goal is to build a production-style Retrieval-Augmented Generation architecture for financial documents, where answers are supported by traceable evidence from SEC filings.

---

## Project Overview

Financial documents such as SEC filings, annual reports, and financial statement notes contain large volumes of complex disclosure text. Searching these documents manually is time-consuming, and simple keyword search often misses relevant context.

This project builds a financial document intelligence pipeline that can answer questions over SEC disclosure data by combining:

- Semantic search using sentence-transformer embeddings and FAISS
- Keyword search using BM25
- Hybrid retrieval using Reciprocal Rank Fusion
- Cross-encoder re-ranking for better context selection
- Metadata filtering by company and filing form
- Source-grounded evidence extraction
- Local extractive answer generation without requiring an LLM API
- Optional LangChain + OpenAI answer generation
- FastAPI serving through a `/ask` endpoint
- Dockerized API deployment
- Retrieval and LLM evaluation
- LLaMA Factory + LoRA fine-tuning demo
- CLIP-based multimodal retrieval experiment

The system works as an **evidence-grounded retrieval and LLM-ready RAG pipeline**. It retrieves the most relevant financial disclosure chunks and prepares a structured prompt that can be used by LangChain, OpenAI, Azure OpenAI, a local LLM, or another language model for final answer generation.

By default, the project can run fully in local extractive mode without an API key. If `USE_LANGCHAIN_LLM=true` and an OpenAI API key with available quota is configured, the system uses LangChain to generate a natural-language answer from the retrieved SEC filing context.

---

## Business Problem

Financial analysts, auditors, researchers, and compliance teams often need to answer questions such as:

- What does a company disclose about revenue recognition?
- What accounting policies are mentioned in the filing?
- What does the filing say about fair value measurements?
- What debt, lease, tax, or risk-related disclosures are available?
- Which filing section supports a specific financial interpretation?
- Can the answer be traced back to the source company, form, fiscal year, and disclosure tag?

This project addresses that problem by creating a retrieval and answer-generation system that returns relevant financial evidence with metadata and source traceability.

---

## Key Features

- End-to-end financial document processing pipeline
- SEC Financial Statement and Notes dataset ingestion
- Metadata-rich chunk generation
- Text cleaning and financial disclosure preprocessing
- FAISS-based semantic vector search
- BM25 lexical keyword search
- Hybrid search using Reciprocal Rank Fusion
- Cross-encoder re-ranking with FlashRank
- Company and filing-form metadata filtering
- FastAPI application for real-time querying
- Dockerized API deployment
- Source-grounded retrieval output with company, form, year, filed date, tag, and chunk ID
- Local extractive answer mode when no API quota or key is available
- Optional LangChain + OpenAI answer-generation layer
- LLM-ready prompt generation
- Source-cited answer output
- Retrieval evaluation using financial question sets
- LLM/RAG evaluation report covering groundedness, context relevance, citation correctness, and hallucination risk
- LLaMA Factory + LoRA fine-tuning demo using TinyLlama and financial RAG-style instruction data
- CLIP multimodal retrieval experiment for text-to-image search over financial chart/document images
- Modular production-style Python project structure

---

## Architecture

```mermaid
flowchart LR
    A[SEC Financial Statement and Notes TSV Files] --> B[Ingestion Layer]
    B --> C[Text Cleaning and Normalization]
    C --> D[Metadata-Rich Chunking]
    D --> E[Sentence Transformer Embeddings]
    D --> F[BM25 Token Index]
    E --> G[FAISS Vector Index]
    F --> H[BM25 Lexical Search]
    G --> I[Hybrid Search with RRF]
    H --> I
    I --> J[Cross-Encoder Re-ranking]
    J --> K[Source-Grounded Evidence]
    K --> L[Prompt Builder]
    L --> M{Answer Mode}
    M --> N[Local Extractive Answer]
    M --> O[LangChain + OpenAI Answer]
    N --> P[FastAPI /ask Endpoint]
    O --> P
    P --> Q[Answer, Sources, Retrieved Chunks, Prompt Preview]
```

---

## Technology Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Financial Dataset | SEC Financial Statement and Notes Data Sets |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Lexical Search | BM25 |
| Hybrid Retrieval | Reciprocal Rank Fusion |
| Re-ranking | FlashRank Cross-Encoder |
| LLM Orchestration | LangChain |
| Optional LLM Generation | OpenAI API / GPT model |
| API Layer | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| LLM Fine-tuning | LLaMA Factory, LoRA, TinyLlama |
| Multimodal AI | CLIP |
| LLM/RAG Evaluation | Groundedness, context relevance, citation correctness, hallucination risk |
| Configuration | YAML, python-dotenv |
| Version Control | Git and GitHub |

---

## Dataset

This project uses the **SEC Financial Statement and Notes Data Sets**.

The dataset provides text and numeric information extracted from financial statements and notes filed with the SEC using XBRL. For this project, the main focus is on the disclosure text contained in the `txt.tsv` file, enriched with submission metadata from `sub.tsv`.

Used files:

| File | Purpose |
|---|---|
| `txt.tsv` | Financial disclosure text blocks and note values |
| `sub.tsv` | Company, filing, fiscal year, and submission metadata |
| `tag.tsv` | Financial tag definitions and disclosure labels |
| `num.tsv` | Numeric financial data, optional for future expansion |

The current pipeline processes a configurable sample of rows from `txt.tsv` to create financial disclosure chunks for retrieval.

Large SEC source files and generated vector indexes are not committed to GitHub. The repository includes source code, configuration, evaluation questions, documentation, and small sample outputs only.

---

## Project Structure

```text
financial-document-intelligence-rag-pipeline/

├── README.md
├── requirements.txt
├── .gitignore
├── .dockerignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── main.py
├── config/
│   └── config.yaml
├── data/
│   ├── raw_docs/
│   ├── processed/
│   │   └── .gitkeep
│   └── evaluation/
│       └── evaluation_questions.csv
├── outputs/
│   ├── api_response_sample.json
│   ├── evaluation_results_sample.csv
│   ├── llm_evaluation_sample.csv
│   └── sec_note_chunks_sample.csv
├── vector_store/
│   └── .gitkeep
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py
│   │   └── table_extractor.py
│   ├── preprocessing/
│   │   ├── text_cleaning.py
│   │   └── chunking.py
│   ├── retrieval/
│   │   ├── embedding_search.py
│   │   ├── bm25_search.py
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── prompt_builder.py
│   │   ├── langchain_rag_chain.py
│   │   └── rag_answer.py
│   ├── evaluation/
│   │   └── ragas_evaluation.py
│   ├── api/
│   │   └── app.py
│   └── utils/
│       ├── logger.py
│       └── config_loader.py
├── experiments/
│   ├── llama_factory_lora_finetuning/
│   │   ├── README.md
│   │   ├── RUN_COMMANDS.md
│   │   ├── requirements-llamafactory.txt
│   │   ├── configs/
│   │   │   └── tinyllama_lora_sft.yaml
│   │   ├── data/
│   │   │   ├── dataset_info.json
│   │   │   └── financial_instruction_sample.jsonl
│   │   └── outputs/
│   │       ├── lora_evaluation_sample.csv
│   │       └── training_summary.txt
│   └── clip_multimodal_retrieval/
│       ├── README.md
│       ├── requirements-clip.txt
│       ├── data/
│       │   ├── image_metadata.csv
│       │   └── images/
│       │       ├── revenue_chart_sample.png
│       │       ├── risk_summary_sample.png
│       │       └── balance_sheet_sample.png
│       ├── outputs/
│       │   ├── clip_retrieval_results.csv
│       │   └── clip_run_summary.txt
│       └── src/
│           └── clip_image_search.py
├── prompts/
│   ├── system_prompt.txt
│   └── rag_prompt_template.txt
├── docs/
│   ├── architecture.md
│   ├── data_flow.md
│   ├── evaluation.md
│   └── llm_evaluation_report.md
└── tests/
    ├── test_chunking.py
    ├── test_hybrid_search.py
    └── test_prompt_formatting.py
```

---

## Pipeline Workflow

### 1. SEC Data Ingestion

The ingestion layer reads SEC financial disclosure data from `txt.tsv` and joins it with company and filing metadata from `sub.tsv`.

It extracts fields such as:

- Company name
- CIK
- Filing form
- Fiscal year
- Fiscal period
- Filed date
- Disclosure tag
- Financial disclosure text

Command:

```bash
python main.py --task ingest
```

Example output:

```text
Created 24,552 chunks.
Saved chunks to data/processed/sec_note_chunks.csv
```

---

### 2. Text Cleaning and Chunking

The preprocessing layer cleans financial disclosure text by removing HTML artifacts, normalizing whitespace, and preparing text for retrieval.

The chunking logic creates overlapping chunks so that long disclosures can be searched effectively while preserving context.

Each chunk includes metadata such as:

- `chunk_id`
- `company_name`
- `form`
- `fiscal_year`
- `fiscal_period`
- `filed_date`
- `tag`
- `chunk_text`

This makes retrieval traceable and suitable for regulated financial use cases.

---

### 3. Embedding Creation and FAISS Indexing

The embedding layer converts financial text chunks into dense vector representations using a sentence-transformer model.

The FAISS index enables fast semantic search over thousands of financial disclosure chunks.

Command:

```bash
python main.py --task embeddings
```

Example output:

```text
Creating embeddings for 24,552 chunks using sentence-transformers/all-MiniLM-L6-v2
Saved embeddings to data/processed/embeddings.npy
Saved metadata to data/processed/chunk_metadata.csv
Saved FAISS index with 24,552 vectors to vector_store/faiss_index.faiss
```

---

### 4. BM25 Keyword Indexing

Financial filings often contain precise terminology such as:

- Revenue recognition
- Fair value measurements
- Leases
- Income taxes
- Debt
- Commitments and contingencies
- Significant accounting policies

Semantic search alone may miss exact keyword matches, so this project also builds a BM25 lexical search index.

Command:

```bash
python main.py --task bm25
```

Example output:

```text
Saved BM25 index to vector_store/bm25_index.pkl
```

---

### 5. Hybrid Retrieval

The retrieval layer combines:

- FAISS semantic search
- BM25 keyword search

Results are merged using **Reciprocal Rank Fusion**, which improves retrieval quality by balancing semantic relevance with exact keyword matching.

This is important for financial documents because the same concept may appear as a natural-language explanation, a formal accounting policy, or a specific XBRL tag.

---

### 6. Cross-Encoder Re-ranking

After hybrid retrieval, the pipeline applies a cross-encoder re-ranker to improve the order of retrieved chunks.

The re-ranker scores query-document pairs more precisely and helps reduce irrelevant context before the final answer or prompt is generated.

This improves retrieval precision and reduces the risk of sending weak or irrelevant chunks to the answer generation layer.

---

### 7. Metadata Filtering

The system supports metadata filtering to avoid mixing disclosures from different companies or filing types.

Example:

```bash
python main.py --task query --question "What does the filing say about revenue recognition?" --company "SEI INVESTMENTS CO" --form "10-Q"
```

This prevents cross-company contamination and improves trust in the retrieved evidence.

Supported filters include:

- Company name
- Filing form

The metadata structure can also be extended to support:

- Fiscal year
- Filed date
- Disclosure tag
- CIK

---

### 8. Answer Generation: Local Extractive Mode and LangChain LLM Mode

The pipeline supports two answer-generation modes.

#### Local Extractive Answer Mode

This mode does not require an API key. It generates a source-grounded answer from the top retrieved and re-ranked SEC filing chunk.

This mode is useful for:

- Local testing
- No-cost execution
- GitHub reproducibility
- Demonstrating retrieval quality without relying on external API access

#### LangChain + OpenAI Generation Mode

When `USE_LANGCHAIN_LLM=true` and an OpenAI API key with available quota is configured, the system uses LangChain to generate a final natural-language answer from the retrieved financial disclosure context.

The LangChain layer receives the final retrieved and re-ranked chunks from the custom retrieval pipeline. Retrieval is handled by:

```text
FAISS semantic search + BM25 keyword search + RRF + cross-encoder re-ranking
```

LangChain is used only for the answer-generation layer.

The prompt includes:

- User question
- Retrieved disclosure text
- Company name
- Filing form
- Fiscal year
- Filed date
- Disclosure tag
- Chunk ID
- Instructions to answer only from the retrieved context
- Instructions not to invent financial facts, figures, dates, or conclusions

This makes the system compatible with:

- OpenAI API
- Azure OpenAI
- Local LLMs
- LangChain chains
- LlamaIndex
- Hugging Face models

---

### 9. FastAPI Serving

The project includes a FastAPI application that exposes the retrieval and answer-generation pipeline through a `/ask` endpoint.

Run the API:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Example API request:

```json
{
  "question": "What does the filing say about revenue recognition?",
  "company_name": "SEI INVESTMENTS CO",
  "form": "10-Q"
}
```

Example API response includes:

- Question
- Answer
- Answer mode
- Applied filters
- Source list
- Retrieved chunks
- LLM prompt preview

---

## Example Query Result

Question:

```text
What does the filing say about revenue recognition?
```

Filter:

```json
{
  "company_name": "SEI INVESTMENTS CO",
  "form": "10-Q"
}
```

Top retrieved source:

| Field | Value |
|---|---|
| Company | SEI INVESTMENTS CO |
| Form | 10-Q |
| Fiscal Year | 2026 |
| Filed Date | 20260427 |
| Tag | RevenueFromContractWithCustomerPolicyTextBlock |
| Chunk ID | 8690 |
| Re-rank Score | 0.9888 |

Retrieved evidence:

```text
Revenue is recognized when the transfer of control of promised goods or services under the terms of a contract with customers are satisfied in an amount that reflects the consideration to which the Company expects to be entitled in exchange for those promised goods or services.
```

This demonstrates that the system correctly retrieves a relevant revenue recognition disclosure with source metadata.

---

## Example API Output

The `/ask` endpoint returns structured JSON.

Example local extractive mode output:

```json
{
  "question": "What does the filing say about revenue recognition?",
  "answer": "Revenue is recognized when the transfer of control of promised goods or services under the terms of a contract with customers is satisfied in an amount that reflects the consideration the company expects to receive. The filing also discusses principal versus agent assessment for third-party arrangements.\n\nSources: SEI INVESTMENTS CO, 10-Q, FY 2026, RevenueFromContractWithCustomerPolicyTextBlock, Chunk ID 8690.",
  "answer_mode": "local_extractive_answer",
  "filters": {
    "company_name": "SEI INVESTMENTS CO",
    "form": "10-Q"
  },
  "sources": [
    {
      "company_name": "SEI INVESTMENTS CO",
      "form": "10-Q",
      "fiscal_year": 2026,
      "filed_date": 20260427,
      "tag": "RevenueFromContractWithCustomerPolicyTextBlock",
      "chunk_id": 8690,
      "score": 0.9888
    }
  ],
  "retrieved_chunks": [],
  "llm_prompt_preview": "Question: What does the filing say about revenue recognition? ..."
}
```

If `USE_LANGCHAIN_LLM=true` and an OpenAI API key with available quota is configured, the response uses:

```text
answer_mode: langchain_llm_generation
```

If API quota is unavailable, the project can still run in local extractive mode:

```text
USE_LANGCHAIN_LLM=false
```

---

## Evaluation

The project includes an evaluation question set stored in:

```text
data/evaluation/evaluation_questions.csv
```

Example evaluation questions:

```text
What does the filing say about revenue recognition?
What are the main accounting policies mentioned?
What does the company disclose about fair value measurements?
What does the filing say about income taxes?
What does the company report about leases?
What does the filing say about debt or borrowings?
```

Run evaluation:

```bash
python main.py --task evaluate
```

Example output:

```text
Evaluation results saved to data/processed/evaluation_results.csv
```

The current evaluation checks whether retrieved chunks contain the expected financial topic. Future improvements can add RAGAS-based metrics such as faithfulness, answer relevancy, context precision, and context recall.

---

## LLM and Retrieval Evaluation Approach

The project includes a lightweight evaluation workflow for checking whether retrieved evidence is suitable for source-grounded answer generation.

Evaluation considers:

- **Context relevance**: whether the retrieved chunk directly addresses the question
- **Groundedness**: whether the answer can be supported by the retrieved context
- **Source traceability**: whether the response includes company, form, fiscal year, tag, and chunk ID
- **Citation correctness**: whether the answer points to the correct source metadata
- **Hallucination risk**: whether the answer introduces unsupported facts
- **Metadata filtering correctness**: whether company/form filters prevent cross-company context mixing

A detailed evaluation report is available at:

```text
docs/llm_evaluation_report.md
```

A sample evaluation output is included in:

```text
outputs/llm_evaluation_sample.csv
```

---

## LLaMA Factory LoRA Fine-tuning Demo

This repository includes a compact **LLaMA Factory + LoRA fine-tuning** experiment under:

```text
experiments/llama_factory_lora_finetuning/
```

The experiment uses financial RAG-style instruction data to fine-tune `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with LoRA adapters.

It includes:

- Alpaca-style financial instruction dataset
- LLaMA Factory dataset metadata
- TinyLlama LoRA SFT configuration
- Training command documentation
- Successful local CPU-based training run
- Training and evaluation metrics
- Qualitative evaluation sample

Training summary:

```text
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Method: LoRA supervised fine-tuning
Trainable parameters: 1,126,400
Training steps: 3
Train loss: 1.8529
Eval loss: 2.6567
Runtime: 1:22:59 on CPU
```

This experiment demonstrates a practical workflow for instruction data preparation, LoRA configuration, LLaMA Factory training, and evaluation for source-grounded financial question answering.

---

## CLIP Multimodal Retrieval Experiment

The repository includes a compact **CLIP-based multimodal retrieval** experiment under:

```text
experiments/clip_multimodal_retrieval/
```

The experiment demonstrates text-to-image retrieval for financial chart and document screenshot images using CLIP embeddings.

Implemented components:

- CLIP text embeddings
- CLIP image embeddings
- Similarity search
- Financial image metadata
- Sample financial chart/table images
- Retrieval output CSV

Example query:

```text
revenue trend chart
```

Top result:

```text
revenue_chart_sample.png
```

This experiment supports future multimodal document intelligence workflows such as chart retrieval, financial table image search, document screenshot retrieval, and multimodal RAG.

---

## Docker Deployment

The FastAPI RAG service can be containerized using Docker.

Build and run:

```bash
docker compose up --build
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

The Docker Compose setup mounts local `data/`, `vector_store/`, and `outputs/` folders into the container so the API can use generated FAISS and BM25 indexes without committing large artifacts to GitHub.

Stop the service:

```bash
docker compose down
```

---

## How to Run the Project

### 1. Create a virtual environment

Recommended Python version: **Python 3.11**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Configure dataset path

Update the SEC dataset path in:

```text
config/config.yaml
```

Example:

```yaml
paths:
  sec_notes_data: "C:/Users/md.w.ahmad/Downloads/LLMs/2026_04_notes"
```

### 4. Configure environment variables

Create a local `.env` file if you want optional LangChain + OpenAI answer generation.

Example `.env`:

```env
APP_ENV=local
SEC_NOTES_DATA_PATH=C:\Users\your_name\Downloads\LLMs\2026_04_notes
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_PATH=vector_store/faiss_index.faiss
BM25_STORE_PATH=vector_store/bm25_index.pkl
TOP_K_VECTOR=20
TOP_K_BM25=20
TOP_K_FINAL=5

OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
USE_LANGCHAIN_LLM=false
```

Use:

```env
USE_LANGCHAIN_LLM=false
```

for free local extractive mode.

Use:

```env
USE_LANGCHAIN_LLM=true
```

only if an OpenAI API key with available quota is configured.

The real `.env` file is ignored by Git and should never be pushed to GitHub.

---

### 5. Run the full pipeline

```powershell
python main.py --task ingest
python main.py --task embeddings
python main.py --task bm25
python main.py --task evaluate
```

### 6. Run a query

```powershell
python main.py --task query --question "What does the filing say about revenue recognition?" --company "SEI INVESTMENTS CO" --form "10-Q"
```

### 7. Run the API

```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

## Output Files

Generated full pipeline artifacts are created locally under:

```text
data/processed/
vector_store/
```

Important generated artifacts include:

```text
data/processed/sec_note_chunks.csv
data/processed/embeddings.npy
data/processed/chunk_metadata.csv
data/processed/evaluation_results.csv
vector_store/faiss_index.faiss
vector_store/bm25_index.pkl
```

These files are generated and may be large, so they are excluded from GitHub.

Small sample outputs are stored in:

```text
outputs/
```

Example sample outputs:

```text
outputs/api_response_sample.json
outputs/evaluation_results_sample.csv
outputs/sec_note_chunks_sample.csv
outputs/llm_evaluation_sample.csv
```

Experiment outputs are stored under:

```text
experiments/llama_factory_lora_finetuning/outputs/
experiments/clip_multimodal_retrieval/outputs/
```

Large generated model artifacts, checkpoints, and vector indexes are excluded from GitHub using `.gitignore`.

---

## Why Hybrid Search?

Financial documents contain both natural language and highly specific terminology. A purely semantic search system may retrieve similar-looking but incorrect passages, while a pure keyword search system may miss paraphrased disclosures.

This project combines both approaches:

| Retrieval Method | Strength |
|---|---|
| FAISS semantic search | Finds conceptually similar disclosure text |
| BM25 keyword search | Captures exact financial terminology and XBRL-like terms |
| RRF fusion | Combines both result lists robustly |
| Cross-encoder re-ranking | Improves final context relevance |
| LangChain answer generation | Converts retrieved evidence into a natural-language answer when API access is available |

This design is closer to real-world RAG systems used in financial document search, compliance workflows, research analysis, and regulated document intelligence.

---

## Current Status

Implemented:

- SEC notes ingestion
- Metadata-rich chunking
- FAISS semantic search
- BM25 keyword search
- Hybrid retrieval with Reciprocal Rank Fusion
- Cross-encoder re-ranking
- Company/form filtering
- FastAPI `/ask` endpoint
- Dockerized FastAPI deployment
- Local extractive source-grounded answer mode
- Optional LangChain + OpenAI answer-generation layer
- LLM-ready prompt preview
- Retrieval evaluation output
- LLM/RAG evaluation report
- LLaMA Factory + LoRA fine-tuning demo
- TinyLlama training evidence with train/eval metrics
- CLIP multimodal retrieval experiment
- Sample output files for GitHub

Planned improvements:

- Add RAGAS evaluation metrics
- Add stronger table-aware parsing
- Add Streamlit UI
- Add fiscal year and tag-level filtering
- Add more advanced source-cited generated answers
- Add Azure OpenAI support
- Add local LLM support through Ollama or Hugging Face models
- Extend CLIP experiment into full multimodal RAG

---

## Key Skills Demonstrated

This project demonstrates:

- Financial document intelligence
- Retrieval-Augmented Generation architecture
- SEC financial disclosure processing
- Data ingestion and preprocessing
- Metadata-rich chunking
- Embedding generation
- FAISS vector indexing
- BM25 lexical retrieval
- Hybrid search
- Reciprocal Rank Fusion
- Cross-encoder re-ranking
- LangChain-based answer generation
- OpenAI API integration
- LLM-ready prompt engineering
- FastAPI development
- Dockerized API deployment
- Source-grounded financial question answering
- Source-cited answer generation
- Retrieval evaluation
- LLM/RAG evaluation methodology
- LLaMA Factory + LoRA fine-tuning
- TinyLlama instruction fine-tuning workflow
- CLIP-based multimodal retrieval
- Modular Python engineering
- ML/NLP pipeline design
- Production-style RAG system architecture

---

## Future Improvements

Possible next upgrades:

1. **RAGAS Evaluation**  
   Add automated RAG metrics such as faithfulness, answer relevancy, context precision, and context recall.

2. **Table-Aware Processing**  
   Improve handling of numeric financial tables and structured disclosures.

3. **Source-Cited Generated Answers**  
   Return final answers with stronger inline citations showing company, filing form, fiscal year, tag, and chunk ID.

4. **UI Layer**  
   Add a Streamlit dashboard or lightweight frontend for interactive financial document search.

5. **Advanced Metadata Filtering**  
   Add filters for fiscal year, filed date, CIK, and disclosure tag.

6. **Azure OpenAI Support**  
   Add configuration for Azure OpenAI-based enterprise deployment.

7. **Local LLM Support**  
   Add optional local answer generation using Ollama or Hugging Face models.

8. **LangChain Extensions**  
   Add LangChain retrieval chains, memory-free QA chains, or tool-based document inspection workflows.

9. **Multimodal RAG Extension**  
   Extend the CLIP experiment into a full multimodal RAG workflow for financial charts, screenshots, and table images.

---

## Disclaimer

This project is for educational and portfolio purposes. It uses publicly available SEC financial disclosure data and does not provide investment, legal, accounting, or financial advice.

The system is designed to demonstrate financial document retrieval, RAG architecture, LangChain-based optional answer generation, LLaMA Factory fine-tuning, CLIP multimodal retrieval, and source-grounded document intelligence workflows.

---

## Author

**Md Wakil Ahmad**  
GitHub: https://github.com/Wakiloo7  
LinkedIn: https://www.linkedin.com/in/md-wakil-ahmad