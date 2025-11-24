#!/usr/bin/env python3
"""
Fix Missing JSON Metadata Files

This script generates JSON metadata files for images that are missing them.
It analyzes each image and creates comprehensive metadata.
"""

import os
import json
from pathlib import Path
from PIL import Image
from datetime import datetime
import hashlib

# Import core modules
from core import ImageAnalyzer

def calculate_hash(file_path):
    """Calculate MD5 hash of file"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Error calculating hash: {e}")
        return ""

def create_metadata_for_image(image_path):
    """Create metadata JSON for an image file"""
    try:
        # Initialize analyzer
        analyzer = ImageAnalyzer()

        # Analyze image
        features = analyzer.analyze_image(image_path)

        # Get image properties
        with Image.open(image_path) as img:
            image_width, image_height = img.size
            image_format = img.format
            image_mode = img.mode

        # Get file info
        file_stats = os.stat(image_path)
        file_size = file_stats.st_size

        # Calculate hash
        image_hash = calculate_hash(image_path)

        # Extract timestamp from filename if possible
        filename = os.path.basename(image_path)
        # Format: image_YYYYMMDD_HHMMSS_mmm_hash.jpg
        parts = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').split('_')

        if len(parts) >= 4 and parts[0] == 'image':
            # Has timestamp in filename
            timestamp_str = f"{parts[1]}_{parts[2]}_{parts[3]}"
            try:
                download_time = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S")
            except:
                download_time = datetime.fromtimestamp(file_stats.st_mtime)
        else:
            # Use file modification time
            download_time = datetime.fromtimestamp(file_stats.st_mtime)
            timestamp_str = download_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Create comprehensive metadata
        metadata = {
            # Basic identifiers
            'filename': filename,
            'file_path': image_path,
            'image_id': timestamp_str,
            'md5_hash': image_hash,

            # Download info (best guess)
            'download_timestamp': download_time.isoformat(),
            'download_date': download_time.strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'unknown',  # Can't determine for old files
            'source_url': 'unknown',
            'source_category': 'unknown',

            # File properties
            'file_size_bytes': file_size,
            'file_size_kb': round(file_size / 1024, 2),

            # Image properties
            'image_properties': {
                'width': image_width,
                'height': image_height,
                'format': image_format,
                'mode': image_mode,
                'dimensions': f"{image_width}x{image_height}"
            },

            # Image features from analyzer
            'image_features': features,

            # Queryable attributes
            'queryable_attributes': {
                'brightness_level': 'bright' if features.get('brightness', 0) > 150 else 'dark',
                'image_quality': 'high' if features.get('contrast', 0) > 50 else 'medium',
                'sharpness_level': features.get('sharpness_level', 'unknown'),
                'color_diversity': features.get('color_diversity', 0),
                'aspect_ratio': features.get('aspect_ratio', 1.0),
                'megapixels': features.get('megapixels', 0)
            },

            # Processing info
            'metadata_created_by': 'fix_missing_json.py',
            'metadata_created_at': datetime.now().isoformat(),
            'note': 'Metadata generated retroactively for existing image'
        }

        return metadata

    except Exception as e:
        print(f"Error creating metadata for {image_path}: {e}")
        return None

def fix_missing_json_files(images_dir='./images'):
    """Find images without JSON files and create them"""

    print("="*70)
    print("FIX MISSING JSON METADATA FILES")
    print("="*70)
    print()

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    all_images = []
    for ext in image_extensions:
        all_images.extend(Path(images_dir).rglob(ext))

    # Filter out macOS metadata files
    all_images = [img for img in all_images if not img.name.startswith('._')]

    print(f"Found {len(all_images)} image files in {images_dir}")
    print()

    # Find images without JSON files
    missing_json = []
    for img_path in all_images:
        json_path = img_path.with_suffix('.json')
        if not json_path.exists():
            missing_json.append(img_path)

    print(f"Images without JSON metadata: {len(missing_json)}")
    print()

    if len(missing_json) == 0:
        print("✅ All images have JSON metadata files!")
        return

    # Ask for confirmation
    print("This will create JSON metadata files for all images that are missing them.")
    print()
    response = input("Do you want to proceed? (yes/no): ").strip().lower()

    if response != 'yes':
        print("Cancelled.")
        return

    print()
    print("Creating JSON metadata files...")
    print()

    # Process each image
    success_count = 0
    error_count = 0

    for idx, img_path in enumerate(missing_json, 1):
        print(f"[{idx}/{len(missing_json)}] Processing: {img_path.name}")

        # Create metadata
        metadata = create_metadata_for_image(str(img_path))

        if metadata:
            # Save JSON file
            json_path = img_path.with_suffix('.json')
            try:
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"  ✓ Created: {json_path.name}")
                success_count += 1
            except Exception as e:
                print(f"  ✗ Failed to save JSON: {e}")
                error_count += 1
        else:
            print(f"  ✗ Failed to create metadata")
            error_count += 1

        # Progress update every 10 images
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{len(missing_json)} ({success_count} success, {error_count} errors)")
            print()

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total images processed: {len(missing_json)}")
    print(f"✓ Successfully created: {success_count}")
    print(f"✗ Errors: {error_count}")
    print()

    if success_count > 0:
        print("✅ JSON metadata files have been created!")
        print()
        print("Next steps:")
        print("1. Go to 'Process & Embed' tab in the GUI")
        print("2. Click 'Process New Only' to create embeddings")
        print("3. Your images will now be searchable!")
    else:
        print("❌ No JSON files were created. Check the errors above.")

if __name__ == "__main__":
    fix_missing_json_files()
