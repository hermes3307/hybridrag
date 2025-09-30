#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify feature extraction is working correctly
"""

import os
import sys
import io
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_collector import FaceAnalyzer

def test_feature_extraction():
    """Test feature extraction on sample images"""
    print("🧪 Testing Feature Extraction")
    print("="*60)

    # Find sample image files
    faces_dir = Path("./faces")
    image_files = list(faces_dir.glob("*.jpg"))[:5]  # Test first 5 images

    if not image_files:
        print("❌ No image files found in ./faces directory")
        return

    print(f"📁 Found {len(image_files)} image files to test\n")

    # Create analyzer
    analyzer = FaceAnalyzer()

    # Test each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n📷 Testing image {i}: {image_path.name}")
        print("-" * 60)

        try:
            features = analyzer.estimate_basic_features(str(image_path))

            if features:
                print(f"   ✅ Features extracted successfully:")
                print(f"      🎂 Age Group: {features.get('estimated_age_group', 'N/A')}")
                print(f"      🎨 Skin Tone: {features.get('estimated_skin_tone', 'N/A')}")
                print(f"      📸 Quality: {features.get('image_quality', 'N/A')}")
                print(f"      💡 Brightness: {features.get('brightness', 0):.2f}")
                print(f"      🌈 Hue: {features.get('avg_hue', 0):.2f}")
                print(f"      🎨 Saturation: {features.get('avg_saturation', 0):.2f}")
            else:
                print(f"   ⚠️  No features extracted")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "="*60)
    print("✅ Feature extraction test completed!")

if __name__ == "__main__":
    test_feature_extraction()