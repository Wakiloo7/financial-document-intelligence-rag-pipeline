from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


def load_model(model_name: str):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def embed_images(model, processor, image_paths):
    embeddings = []

    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors='pt')

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embeddings.append(image_features.squeeze(0).cpu().numpy())

    return np.vstack(embeddings)


def embed_text(model, processor, query: str):
    inputs = processor(text=[query], return_tensors='pt', padding=True)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.squeeze(0).cpu().numpy()


def search(query: str, top_k: int = 3):
    base_dir = Path('experiments/clip_multimodal_retrieval')
    image_dir = base_dir / 'data' / 'images'
    metadata_path = base_dir / 'data' / 'image_metadata.csv'
    output_path = base_dir / 'outputs' / 'clip_retrieval_results.csv'

    metadata = pd.read_csv(metadata_path)

    image_paths = []
    valid_rows = []

    for _, row in metadata.iterrows():
        image_path = image_dir / row['file_name']

        if image_path.exists():
            image_paths.append(image_path)
            valid_rows.append(row)

    if not image_paths:
        raise FileNotFoundError(
            'No image files found. Add sample images to experiments/clip_multimodal_retrieval/data/images/'
        )

    model_name = 'openai/clip-vit-base-patch32'
    model, processor = load_model(model_name)

    image_embeddings = embed_images(model, processor, image_paths)
    text_embedding = embed_text(model, processor, query)

    scores = image_embeddings @ text_embedding

    results = []
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    for rank, idx in enumerate(ranked_indices, start=1):
        row = valid_rows[idx]
        results.append(
            {
                'rank': rank,
                'query': query,
                'image_id': row['image_id'],
                'file_name': row['file_name'],
                'description': row['description'],
                'document_type': row['document_type'],
                'topic': row['topic'],
                'similarity_score': float(scores[idx]),
            }
        )

    results_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(results_df)
    print(f'Saved results to {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True)
    parser.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()

    search(query=args.query, top_k=args.top_k)


if __name__ == '__main__':
    main()
