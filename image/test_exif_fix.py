#!/usr/bin/env python3
"""
Test EXIF IFDRational fix

This script tests if images with EXIF data can be processed without errors.
"""

import os
import json
from pathlib import Path
from core import ImageAnalyzer

def test_image_analysis():
    """Test analyzing images with EXIF data"""

    print("="*70)
    print("Testing EXIF IFDRational Fix")
    print("="*70)
    print()

    # Get first few images
    images_dir = './images'
    image_files = list(Path(images_dir).glob('*.jpg'))[:5]

    if not image_files:
        print("No images found in ./images/")
        return

    print(f"Testing {len(image_files)} images...")
    print()

    analyzer = ImageAnalyzer()
    success_count = 0
    error_count = 0

    for img_path in image_files:
        print(f"Testing: {img_path.name}")

        try:
            # Analyze image
            features = analyzer.analyze_image(str(img_path))

            # Try to convert to JSON (this is where IFDRational would fail)
            json_str = json.dumps(features, indent=2)

            print(f"  ✓ Analysis successful")
            print(f"  ✓ JSON serialization successful")
            print(f"  - Features extracted: {len(features)} fields")

            # Show some features
            if 'brightness' in features:
                print(f"  - Brightness: {features['brightness']:.1f}")
            if 'contrast' in features:
                print(f"  - Contrast: {features['contrast']:.1f}")
            if 'has_exif' in features:
                print(f"  - Has EXIF: {features['has_exif']}")

            success_count += 1
            print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_count += 1
            print()

    print("="*70)
    print("Test Results")
    print("="*70)
    print(f"✓ Success: {success_count}")
    print(f"✗ Errors: {error_count}")
    print()

    if error_count == 0:
        print("✅ All tests passed! IFDRational issue is fixed.")
        print()
        print("You can now process images in the GUI:")
        print("  1. Go to 'Process & Embed' tab")
        print("  2. Click 'Process New Only'")
        print("  3. CLIP embeddings will be created successfully")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    test_image_analysis()
