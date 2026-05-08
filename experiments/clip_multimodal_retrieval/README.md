# CLIP Multimodal Retrieval Experiment

This experiment demonstrates a compact multimodal retrieval workflow using CLIP for financial document intelligence.

The goal is to retrieve relevant financial chart or document screenshot images using natural-language text queries.

## Purpose

This experiment extends the Financial Document Intelligence RAG Pipeline with a small multimodal component.

It demonstrates:

- Text-to-image retrieval
- CLIP image embeddings
- CLIP text embeddings
- Similarity search over financial chart/document screenshots
- Multimodal document intelligence workflow design

## Example Use Cases

- Search for a revenue chart using a text query
- Retrieve a financial table screenshot related to balance sheet information
- Retrieve a risk summary image from a document screenshot
- Extend text-only RAG toward multimodal RAG

## Folder Structure

experiments/clip_multimodal_retrieval/

README.md
requirements-clip.txt
data/image_metadata.csv
data/images/README.md
outputs/
src/clip_image_search.py

## Install Dependencies

pip install -r experiments\clip_multimodal_retrieval\requirements-clip.txt

## Add Images

Place sample images in:

experiments/clip_multimodal_retrieval/data/images/

Expected sample names:

- revenue_chart_sample.png
- risk_summary_sample.png
- balance_sheet_sample.png

You can use screenshots from financial charts, tables, dashboards, or generated demo images.

## Run Search

python experiments\clip_multimodal_retrieval\src\clip_image_search.py --query "revenue trend chart" --top-k 3

Output is saved to:

experiments/clip_multimodal_retrieval/outputs/clip_retrieval_results.csv

## Notes

This is a compact experiment for demonstrating multimodal retrieval concepts. It is not a production multimodal RAG system.

It supports future extensions such as:

- Financial chart understanding
- Multimodal RAG
- Document screenshot retrieval
- Table image retrieval
- CLIP-based visual search
