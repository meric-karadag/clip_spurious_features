import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
import open_clip
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Model configurations from clip_load.md
MODEL_CONFIGS = {
    'vit_b16_laion400m': {
        'name': 'hf-hub:timm/vit_base_patch16_clip_224.laion400m_e31',
        'description': 'CLIP-ViT-B/16 LAION400M E31'
    },
    'vit_b16_laion2b': {
        'name': 'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        'description': 'CLIP-ViT-B/16 LAION2B S34B B88K'
    },
    'vit_b16_datacomp': {
        'name': 'hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K',
        'description': 'CLIP-ViT-B/16 DataComp1B XL S13B B90K'
    },
    'vit_b32_laion400m': {
        'name': 'hf-hub:timm/vit_base_patch32_clip_224.laion400m_e31',
        'description': 'CLIP-ViT-B/32 LAION400M E31'
    },
    'vit_b32_laion2b': {
        'name': 'hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
        'description': 'CLIP-ViT-B/32 LAION2B S34B B79K'
    },
    'vit_b32_datacomp': {
        'name': 'hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K',
        'description': 'CLIP-ViT-B/32 DataComp1B XL S13B B90K'
    },
    'vit_l14_laion400m': {
        'name': 'hf-hub:timm/vit_large_patch14_clip_224.laion400m_e31',
        'description': 'CLIP-ViT-L/14 LAION400M E31'
    },
    'vit_l14_laion2b': {
        'name': 'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
        'description': 'CLIP-ViT-L/14 LAION2B S32B B82K'
    },
    'vit_h14_laion2b': {
        'name': 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        'description': 'CLIP-ViT-H/14 LAION2B S32B B79K'
    },
    'vit_g14_laion2b': {
        'name': 'hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K',
        'description': 'CLIP-ViT-G/14 LAION2B S34B B88K'
    },
    'vit_bigg14_laion2b': {
        'name': 'hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
        'description': 'CLIP-ViT-bigG/14 LAION2B S39B B160K'
    },
    'convnext_b_laion400m': {
        'name': 'hf-hub:laion/CLIP-convnext_base-laion400M-s13B-b51K',
        'description': 'CLIP-ConvNext-B LAION400M S13B B51K'
    },
    'convnext_bw_laion2b': {
        'name': 'hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K',
        'description': 'CLIP-ConvNext-BW LAION2B S13B B82K'
    },
    # OpenAI CLIP models from transformers library
    'openai_vit_b16': {
        'name': 'openai/clip-vit-base-patch16',
        'description': 'OpenAI CLIP-ViT-B/16',
        'library': 'transformers'
    },
    'openai_vit_b32': {
        'name': 'openai/clip-vit-base-patch32',
        'description': 'OpenAI CLIP-ViT-B/32',
        'library': 'transformers'
    },
    'openai_vit_l14': {
        'name': 'openai/clip-vit-large-patch14',
        'description': 'OpenAI CLIP-ViT-L/14',
        'library': 'transformers'
    }
}


def load_clip_model(model_key, device):
    """Load CLIP model and preprocessing from configuration."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_key]
    print(f"Loading model: {config['description']}")
    print(f"Model identifier: {config['name']}")
    
    library = config.get('library', 'open_clip')
    
    if library == 'transformers':
        # Load OpenAI CLIP model from transformers
        model = CLIPModel.from_pretrained(config['name'])
        processor = CLIPProcessor.from_pretrained(config['name'])
        model = model.to(device)
        model.eval()
        
        # Create a preprocess function that uses the processor
        def preprocess(image):
            inputs = processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
        
        return model, preprocess, library
    else:
        # Load model from open_clip
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(config['name'])
        model = model.to(device)
        model.eval()
        
        return model, preprocess_val, library


def get_all_image_paths(dataset_root):
    """Recursively get all image paths from the dataset."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    
    dataset_path = Path(dataset_root)
    for file_path in dataset_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_paths.append(file_path)
    
    return image_paths


class ImageDataset(Dataset):
    """Custom dataset for loading images efficiently."""
    
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.preprocess(image)
            return image_tensor, str(img_path), True
        except Exception as e:
            # Return dummy tensor and error flag
            dummy_tensor = torch.zeros(3, 224, 224)  # Standard CLIP input size
            return dummy_tensor, str(img_path), False


def extract_and_save_embeddings(model, preprocess, image_paths, dataset_root, output_root, device, library='open_clip', batch_size=32, num_workers=4):
    """Extract image embeddings and save them in the mirrored directory structure."""
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)
    
    print(f"\nExtracting embeddings for {len(image_paths)} images...")
    print(f"Dataset root: {dataset_root}")
    print(f"Output root: {output_root}")
    print(f"Library: {library}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Process images with progress bar
    failed_images = []
    processed_count = 0
    
    with tqdm(total=len(image_paths), desc="Extracting embeddings", unit="img") as pbar:
        for batch_tensors, batch_paths, batch_valid in dataloader:
            # Filter out failed images
            valid_mask = batch_valid
            valid_tensors = batch_tensors[valid_mask].to(device)
            valid_paths = [path for path, valid in zip(batch_paths, batch_valid) if valid]
            
            # Track failed images
            for path, valid in zip(batch_paths, batch_valid):
                if not valid:
                    failed_images.append((path, "Failed to load image"))
            
            if len(valid_tensors) == 0:
                pbar.update(len(batch_paths))
                continue
            
            try:
                # Extract embeddings for the batch
                with torch.no_grad(), torch.cuda.amp.autocast():
                    if library == 'transformers':
                        # Use transformers CLIP model
                        image_features = model.get_image_features(pixel_values=valid_tensors)
                    else:
                        # Use open_clip model
                        image_features = model.encode_image(valid_tensors)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    image_features = image_features.cpu().numpy()
                
                # Save each embedding
                for idx, img_path_str in enumerate(valid_paths):
                    try:
                        img_path = Path(img_path_str)
                        embedding = image_features[idx]
                        
                        # Compute relative path and create output path
                        relative_path = img_path.relative_to(dataset_path)
                        output_file_path = output_path / relative_path.parent / f"{relative_path.stem}.npy"
                        
                        # Create output directory if it doesn't exist
                        output_file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Save embedding
                        np.save(output_file_path, embedding)
                        processed_count += 1
                    except Exception as e:
                        failed_images.append((img_path_str, str(e)))
                        tqdm.write(f"Failed to save {img_path_str}: {e}")
                
            except Exception as e:
                # If batch processing fails, mark all images in batch as failed
                for img_path_str in valid_paths:
                    failed_images.append((img_path_str, str(e)))
                tqdm.write(f"Failed to process batch: {e}")
            
            pbar.update(len(batch_paths))
    
    # Report results
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Failed: {len(failed_images)} images")
    
    if failed_images:
        print(f"\nFailed images:")
        for img_path, error in failed_images[:10]:  # Show first 10 failures
            print(f"  - {img_path}: {error}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Extract CLIP image embeddings from counteranimal dataset')
    parser.add_argument('--model', type=str, default='vit_b16_laion2b',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='CLIP model to use for embedding extraction')
    parser.add_argument('--dataset_root', type=str, default='LAION-final',
                       help='Root directory of the counteranimal dataset')
    parser.add_argument('--output_root', type=str, default='embeddings/image_embeddings',
                       help='Root directory for output embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing images (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker threads for data loading (default: 4)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Load model
    model, preprocess, library = load_clip_model(args.model, device)
    
    # Get all image paths
    print(f"\nScanning dataset directory: {args.dataset_root}")
    image_paths = get_all_image_paths(args.dataset_root)
    
    if not image_paths:
        print("No images found! Please check the dataset path.")
        return
    
    # Display dataset structure info
    class_folders = set()
    for img_path in image_paths:
        relative = Path(img_path).relative_to(args.dataset_root)
        if len(relative.parts) > 0:
            class_folders.add(relative.parts[0])
    
    print(f"Found {len(class_folders)} animal classes")
    print(f"Total images: {len(image_paths)}")
    
    # Create output path with model name
    output_path_with_model = Path(args.output_root) / args.model
    
    # Extract and save embeddings
    extract_and_save_embeddings(
        model=model,
        preprocess=preprocess,
        image_paths=image_paths,
        dataset_root=args.dataset_root,
        output_root=str(output_path_with_model),
        device=device,
        library=library,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"\nEmbeddings saved to: {output_path_with_model}")
    print(f"Model used: {MODEL_CONFIGS[args.model]['description']}")


if __name__ == '__main__':
    main()

