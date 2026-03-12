import torch
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm

DATASET_ROOT = "LAION-final"
IMAGE_EMBEDDINGS_ROOT = "embeddings/image_embeddings/vit_b32_laion400m"
TEXT_EMBEDDINGS_ROOT = "embeddings/text_embeddings/vit_b32_laion400m"
OUTPUT_PATH = "confusing_classes.json"
TOP_K = 20

def load_text_embeddings(text_root):
    text_embeddings = {}
    text_dir = Path(text_root)
    for npy_file in sorted(text_dir.glob("*.npy")):
        class_id = int(npy_file.stem.split("_")[0])
        embedding = np.load(npy_file)
        text_embeddings[class_id] = embedding / np.linalg.norm(embedding)
    return text_embeddings

def get_animal_class_ids(dataset_root):
    class_ids = set()
    dataset_path = Path(dataset_root)
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            dir_name = class_dir.name
            if " " in dir_name:
                class_id = int(dir_name.split(" ")[0])
                class_ids.add(class_id)
    return sorted(class_ids)

def load_image_embeddings_for_class(image_root, dataset_root, class_id):
    dataset_path = Path(dataset_root)
    image_embeddings_path = Path(image_root)
    embeddings = []
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir() and class_dir.name.startswith(f"{class_id} "):
            class_embedding_dir = image_embeddings_path / class_dir.name
            if class_embedding_dir.exists():
                for split_dir in class_embedding_dir.iterdir():
                    if split_dir.is_dir():
                        for npy_file in split_dir.glob("*.npy"):
                            embedding = np.load(npy_file)
                            embeddings.append(embedding / np.linalg.norm(embedding))
            break
    return embeddings

def compute_confusing_classes(image_embeddings, text_embeddings, top_k):
    if not image_embeddings:
        return []
    text_ids = sorted(text_embeddings.keys())
    text_matrix = np.stack([text_embeddings[i] for i in text_ids])
    similarities = []
    for img_emb in image_embeddings:
        sim = np.dot(text_matrix, img_emb)
        similarities.append(sim)
    avg_similarities = np.mean(similarities, axis=0)
    top_indices = np.argsort(avg_similarities)[::-1][:top_k]
    return [text_ids[i] for i in top_indices]

def main():
    print("Loading text embeddings...")
    text_embeddings = load_text_embeddings(TEXT_EMBEDDINGS_ROOT)
    print(f"Loaded {len(text_embeddings)} text embeddings")
    
    print("Getting animal class IDs from dataset...")
    animal_class_ids = get_animal_class_ids(DATASET_ROOT)
    print(f"Found {len(animal_class_ids)} animal classes")
    
    confusing_classes = {}
    for class_id in tqdm(animal_class_ids, desc="Computing confusing classes"):
        image_embeddings = load_image_embeddings_for_class(
            IMAGE_EMBEDDINGS_ROOT, DATASET_ROOT, class_id
        )
        if image_embeddings:
            top_k_classes = compute_confusing_classes(
                image_embeddings, text_embeddings, TOP_K
            )
            confusing_classes[str(class_id)] = top_k_classes
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(confusing_classes, f)
    
    print(f"Saved confusing classes to {OUTPUT_PATH}")
    print(f"Total classes processed: {len(confusing_classes)}")

if __name__ == "__main__":
    main()
