"""
Extract 500 pairs of images from COCO 2017 test set containing the same classes.
Only annotations from the second image in each pair are kept as ground truth.

Requirements:
- pycocotools: pip install pycocotools
- COCO 2017 test set must be downloaded
"""

import os
import json
import random
import argparse
import shutil
import urllib.request
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Optional

try:
    from pycocotools.coco import COCO
except ImportError:
    print("Error: pycocotools not installed. Install it with: pip install pycocotools")
    exit(1)


def get_category_names(coco: COCO) -> Dict[int, str]:
    """Get mapping of category IDs to names."""
    return {cat['id']: cat['name'] for cat in coco.dataset['categories']}


def get_image_categories(coco: COCO, image_id: int) -> Set[int]:
    """Get all category IDs present in an image."""
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)
    return {ann['category_id'] for ann in annotations}


def get_images_by_categories(coco: COCO) -> Dict[frozenset, List[int]]:
    """Group images by their category sets."""
    images_by_categories = defaultdict(list)
    
    for image_id in coco.getImgIds():
        categories = get_image_categories(coco, image_id)
        if categories:  # Only include images with annotations
            cat_set = frozenset(categories)
            images_by_categories[cat_set].append(image_id)
    
    return images_by_categories


def download_image(url: str, output_path: Path) -> bool:
    """Download image from URL.
    
    Args:
        url: Image URL
        output_path: Path to save the image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as e:
        print(f"    Error downloading {url}: {e}")


def copy_image(src_path: Path, dst_path: Path) -> bool:
    """Copy image from local path.
    
    Args:
        src_path: Source image path
        dst_path: Destination image path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        shutil.copy2(src_path, dst_path)
    except Exception as e:
        print(f"    Error copying {src_path}: {e}")


def extract_image_pairs(
    coco: COCO,
    target_categories: List[int],
    num_pairs: int = 500,
    output_dir: str = './datasets/coco-cvpdl',
    split: str = 'val',
    image_dir: Optional[str] = None,
    use_coco_url: bool = True
) -> None:
    """
    Extract image pairs from COCO containing specified classes.
    
    Args:
        coco: COCO API object
        target_categories: List of category IDs to match
        num_pairs: Number of image pairs to extract
        output_dir: Output directory for pairs
        split: Dataset split to use
        image_dir: Directory containing COCO images (if not using COCO URL)
        use_coco_url: Whether to download images from COCO URL
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_path / 'images' / split
    annotations_dir = output_path / 'annotations' / split
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images with target categories
    target_cat_set = set(target_categories)
    candidate_images = []
    
    print(f"Finding images containing categories: {target_categories}")
    for image_id in coco.getImgIds():
        image_categories = get_image_categories(coco, image_id)
        # Check if image contains all target categories
        if target_cat_set.issubset(image_categories):
            candidate_images.append(image_id)
    
    print(f"Found {len(candidate_images)} images containing all target categories")
    
    if len(candidate_images) < num_pairs * 2:
        print(f"Warning: Only found {len(candidate_images)} candidate images, but need {num_pairs * 2} for {num_pairs} pairs")
        num_pairs = len(candidate_images) // 2
    
    # Randomly sample image pairs
    random.shuffle(candidate_images)
    image_pairs = []
    for i in range(0, min(len(candidate_images) - 1, num_pairs * 2), 2):
        image_pairs.append((candidate_images[i], candidate_images[i + 1]))
    
    # Process pairs
    coco_dataset = {
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
        'images': [],
        'annotations': [],
        'categories': coco.dataset['categories']
    }
    
    print(f"\nProcessing {len(image_pairs)} image pairs...")
    annotation_id = 1
    
    for pair_idx, (img1_id, img2_id) in enumerate(image_pairs):
        img1_info = coco.loadImgs(img1_id)[0]
        img2_info = coco.loadImgs(img2_id)[0]
        
        # Create pair filenames
        img1_new_name = f"{pair_idx:04d}_prompt.jpg"
        img2_new_name = f"{pair_idx:04d}_target.jpg"
        
        img1_dst = images_dir / img1_new_name
        img2_dst = images_dir / img2_new_name
        
        print(f"  [{pair_idx + 1}/{len(image_pairs)}] Processing pair {pair_idx}")
        
        # Download or copy images
        if use_coco_url:
            # Download from COCO URL
            retrieve_coco_image(img1_info, img1_dst)
            retrieve_coco_image(img2_info, img2_dst)
        else:
            # Copy from local directory
            if image_dir is None:
                print(f"    Error: image_dir not specified")
                raise ValueError("image_dir must be specified when use_coco_url is False")
            else:
                img1_src = Path(image_dir) / img1_info['file_name']
                img2_src = Path(image_dir) / img2_info['file_name']
                
                print(f"    Copying image 1: {img1_src}")
                copy_image(img1_src, img1_dst)
                print(f"    Copying image 2: {img2_src}")
                copy_image(img2_src, img2_dst)
        
        # Update image info with new filenames and IDs
        img1_copy = img1_info.copy()
        img1_copy['id'] = pair_idx * 2
        img1_copy['file_name'] = img1_new_name
        
        img2_copy = img2_info.copy()
        img2_copy['id'] = pair_idx * 2 + 1
        img2_copy['file_name'] = img2_new_name
        
        # coco_dataset['images'].append(img1_copy)
        coco_dataset['images'].append(img2_copy)

        # Only keep annotations for the second image
        ann_ids = coco.getAnnIds(imgIds=img2_id)
        annotations = coco.loadAnns(ann_ids)
        
        for ann in annotations:
            # Only keep annotations of target categories
            if ann['category_id'] not in target_categories:
                continue
            ann_copy = ann.copy()
            ann_copy['id'] = annotation_id
            ann_copy['image_id'] = pair_idx * 2 + 1  # Link to second image
            coco_dataset['annotations'].append(ann_copy)
            annotation_id += 1
    
    # Save annotations
    annotations_file = annotations_dir / f'instances_{split}.json'
    with open(annotations_file, 'w') as f:
        json.dump(coco_dataset, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"Output directory: {output_path}")
    print(f"Extracted {len(image_pairs)} pairs")
    print(f"Annotations saved to: {annotations_file}")

def retrieve_coco_image(img_info, img_dst):
    img_url = img_info.get('coco_url') or img_info.get('flickr_url')
    download_image(img_url, img_dst)


def main():
    parser = argparse.ArgumentParser(
        description='Extract COCO 2017 image pairs with matching classes'
    )
    parser.add_argument(
        '--annotation_file',
        type=str,
        required=True,
        help='Path to COCO annotation file (instances_test2017.json)'
    )
    parser.add_argument(
        '--categories',
        type=int,
        nargs='+',
        required=True,
        help='Category IDs to match (e.g., 1 18 for person and dog)'
    )
    parser.add_argument(
        '--num_pairs',
        type=int,
        default=500,
        help='Number of image pairs to extract (default: 500)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./datasets/coco-cvpdl',
        help='Output directory for extracted pairs (default: ./datasets/coco-cvpdl)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        help='Dataset split to use (e.g., train, val, test)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory containing COCO images (if not using COCO URL)'
    )
    parser.add_argument(
        '--use_local',
        action='store_true',
        help='Use local images instead of downloading from COCO URL'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Verify annotation file exists
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        exit(1)
    
    # Load COCO dataset
    print(f"Loading COCO dataset from: {args.annotation_file}")
    coco = COCO(args.annotation_file)
    
    # Print available categories
    categories = get_category_names(coco)
    print(f"\nAvailable categories ({len(categories)} total):")
    for cat_id, cat_name in sorted(categories.items()):
        print(f"  {cat_id}: {cat_name}")
    
    # Verify requested categories exist
    for cat_id in args.categories:
        if cat_id not in categories:
            print(f"Error: Category {cat_id} not found in dataset")
            exit(1)
    
    requested_categories = [categories[cid] for cid in args.categories]
    print(f"\nRequested categories: {args.categories} -> {requested_categories}")
    
    # Extract pairs
    extract_image_pairs(
        coco,
        args.categories,
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
        split=args.split,
        image_dir=args.image_dir,
        use_coco_url=not args.use_local
    )


if __name__ == '__main__':
    main()
