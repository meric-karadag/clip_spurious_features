import torch
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import open_clip
import argparse
from transformers import CLIPModel, CLIPTokenizer

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
    """Load CLIP model and tokenizer from configuration."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_key]
    print(f"Loading model: {config['description']}")
    print(f"Model identifier: {config['name']}")
    
    library = config.get('library', 'open_clip')
    
    if library == 'transformers':
        # Load OpenAI CLIP model from transformers
        model = CLIPModel.from_pretrained(config['name'])
        tokenizer = CLIPTokenizer.from_pretrained(config['name'])
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, library
    else:
        # Load model from open_clip
        model, _, _ = open_clip.create_model_and_transforms(config['name'])
        tokenizer = open_clip.get_tokenizer(config['name'])
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, library


def parse_imagenet_names(filepath):
    """Parse ImageNet class names from the text file.
    
    Format: {class_number}\t{class_name}
    Example: 0\ttench, Tinca tinca
    
    Returns:
        List of tuples: [(class_number, class_name), ...]
    """
    classes = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by tab
            parts = line.split('\t')
            if len(parts) >= 2:
                class_number = parts[0]
                class_name = parts[1]
                classes.append((class_number, class_name))
    
    return classes


def sanitize_filename(name):
    """Sanitize class name to be a valid filename."""
    # Replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name


def extract_and_save_text_embeddings(model, tokenizer, classes, output_root, device, library='open_clip', prompt_template="A photo of {}", batch_size=64, num_workers=0):
    """Extract text embeddings and save them.
    
    Args:
        model: CLIP model
        tokenizer: CLIP tokenizer
        classes: List of (class_number, class_name) tuples
        output_root: Root directory for output embeddings
        device: torch device
        library: Library used ('open_clip' or 'transformers')
        prompt_template: Template for text prompts (use {} as placeholder for class name)
        batch_size: Batch size for processing
        num_workers: Number of worker threads (usually not needed for text, default: 0)
    """
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracting text embeddings for {len(classes)} classes...")
    print(f"Output root: {output_root}")
    print(f"Library: {library}")
    print(f"Prompt template: '{prompt_template}'")
    print(f"Batch size: {batch_size}")
    if num_workers > 0:
        print(f"Num workers: {num_workers}")
    
    # Process in batches
    num_batches = (len(classes) + batch_size - 1) // batch_size
    failed_classes = []
    
    with tqdm(total=len(classes), desc="Extracting text embeddings", unit="class") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(classes))
            batch_classes = classes[start_idx:end_idx]
            
            # Create text prompts for the batch
            batch_texts = []
            batch_info = []
            
            for class_number, class_name in batch_classes:
                try:
                    text = prompt_template.format(class_name)
                    batch_texts.append(text)
                    batch_info.append((class_number, class_name))
                except Exception as e:
                    failed_classes.append((class_number, class_name, str(e)))
                    tqdm.write(f"Failed to create prompt for class {class_number}: {e}")
            
            if not batch_texts:
                pbar.update(len(batch_classes))
                continue
            
            try:
                # Tokenize texts
                if library == 'transformers':
                    # Use transformers tokenizer
                    text_tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
                else:
                    # Use open_clip tokenizer
                    text_tokens = tokenizer(batch_texts).to(device)
                
                # Extract embeddings
                with torch.no_grad(), torch.cuda.amp.autocast():
                    if library == 'transformers':
                        # Use transformers CLIP model
                        text_features = model.get_text_features(**text_tokens)
                    else:
                        # Use open_clip model
                        text_features = model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.cpu().numpy()
                
                # Save each embedding
                for idx, (class_number, class_name) in enumerate(batch_info):
                    try:
                        embedding = text_features[idx]
                        
                        # Create filename: {class_number}_{sanitized_class_name}.npy
                        sanitized_name = sanitize_filename(class_name)
                        filename = f"{class_number}_{sanitized_name}.npy"
                        output_file_path = output_path / filename
                        
                        # Save embedding
                        np.save(output_file_path, embedding)
                    except Exception as e:
                        failed_classes.append((class_number, class_name, str(e)))
                        tqdm.write(f"Failed to save embedding for class {class_number}: {e}")
                
            except Exception as e:
                # If batch processing fails, mark all classes in batch as failed
                for class_number, class_name in batch_info:
                    failed_classes.append((class_number, class_name, str(e)))
                tqdm.write(f"Failed to process batch: {e}")
            
            pbar.update(len(batch_classes))
    
    # Report results
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Successfully processed: {len(classes) - len(failed_classes)} classes")
    print(f"Failed: {len(failed_classes)} classes")
    
    if failed_classes:
        print(f"\nFailed classes:")
        for class_number, class_name, error in failed_classes[:10]:
            print(f"  - Class {class_number} ({class_name}): {error}")
        if len(failed_classes) > 10:
            print(f"  ... and {len(failed_classes) - 10} more")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Extract CLIP text embeddings from ImageNet class names')
    parser.add_argument('--model', type=str, default='vit_b16_laion2b',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='CLIP model to use for embedding extraction')
    parser.add_argument('--imagenet_file', type=str, default='LAION-final/imagenet_names.txt',
                       help='Path to ImageNet names file')
    parser.add_argument('--output_root', type=str, default='embeddings/text_embeddings',
                       help='Root directory for output embeddings')
    parser.add_argument('--prompt_template', type=str, default='A photo of {}',
                       help='Template for text prompts (use {} as placeholder for class name)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing texts (default: 64)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of worker threads for data loading (default: 0, not typically needed for text)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Load model
    model, tokenizer, library = load_clip_model(args.model, device)
    
    # Parse ImageNet class names
    print(f"\nParsing ImageNet class names from: {args.imagenet_file}")
    classes = parse_imagenet_names(args.imagenet_file)
    print(f"Found {len(classes)} classes")
    
    if not classes:
        print("No classes found! Please check the file path and format.")
        return
    
    # Display first few classes as examples
    print("\nFirst 5 classes:")
    for class_number, class_name in classes[:5]:
        print(f"  {class_number}: {class_name}")
    
    # Create output path with model name
    output_path_with_model = Path(args.output_root) / args.model
    
    # Extract and save embeddings
    extract_and_save_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        classes=classes,
        output_root=str(output_path_with_model),
        device=device,
        library=library,
        prompt_template=args.prompt_template,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"\nEmbeddings saved to: {output_path_with_model}")
    print(f"Model used: {MODEL_CONFIGS[args.model]['description']}")
    print(f"Total files created: {len(classes)}")


if __name__ == '__main__':
    main()

