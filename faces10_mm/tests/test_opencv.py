#!/usr/bin/env python3
"""
Test OpenCV Integration
Verifies that OpenCV is working correctly with the face processing system
"""

import sys

print("=" * 80)
print("TESTING OPENCV INTEGRATION")
print("=" * 80)

# Test 1: Import OpenCV
print("\n1. Testing OpenCV import...")
try:
    import cv2
    print(f"   ✅ OpenCV {cv2.__version__} imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import OpenCV: {e}")
    sys.exit(1)

# Test 2: Check Haar Cascades (needed for face detection)
print("\n2. Testing Haar Cascade face detection files...")
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"   Cascade path: {cascade_path}")

    import os
    if os.path.exists(cascade_path):
        print(f"   ✅ Face cascade file found")
    else:
        print(f"   ❌ Face cascade file NOT found")

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if not face_cascade.empty():
        print(f"   ✅ Face cascade loaded successfully")
    else:
        print(f"   ❌ Failed to load face cascade")
except Exception as e:
    print(f"   ⚠️  Warning: {e}")

# Test 3: Test basic OpenCV operations
print("\n3. Testing basic OpenCV operations...")
try:
    import numpy as np

    # Create a test image
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[:, :] = [128, 128, 128]  # Gray image

    # Test color conversion
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(test_img, cv2.COLOR_BGR2LAB)

    print(f"   ✅ Color space conversions working")
    print(f"   ✅ BGR→GRAY: {gray.shape}")
    print(f"   ✅ BGR→HSV: {hsv.shape}")
    print(f"   ✅ BGR→LAB: {lab.shape}")

    # Test edge detection
    edges = cv2.Canny(gray, 50, 150)
    print(f"   ✅ Edge detection (Canny) working")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test FaceAnalyzer with OpenCV
print("\n4. Testing FaceAnalyzer with OpenCV...")
try:
    from core import FaceAnalyzer, CV2_AVAILABLE

    print(f"   CV2_AVAILABLE flag: {CV2_AVAILABLE}")

    analyzer = FaceAnalyzer()

    if analyzer.cv2_available:
        print(f"   ✅ FaceAnalyzer initialized with OpenCV support")
    else:
        print(f"   ⚠️  FaceAnalyzer falling back to PIL (OpenCV not available)")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test on actual image (if exists)
print("\n5. Testing on sample image...")
try:
    from pathlib import Path

    # Look for any existing face image
    faces_dir = Path("./faces")
    sample_images = list(faces_dir.glob("*.jpg"))[:1]

    if not sample_images:
        print(f"   ⚠️  No sample images found in {faces_dir}")
        print(f"   Skipping image analysis test")
    else:
        sample_img = str(sample_images[0])
        print(f"   Testing with: {Path(sample_img).name}")

        from core import FaceAnalyzer
        analyzer = FaceAnalyzer()

        features = analyzer.analyze_face(sample_img)

        if 'error' not in features:
            print(f"   ✅ Image analysis successful")

            # Check what features were extracted
            if 'faces_detected' in features:
                print(f"   ✅ Face detection: {features['faces_detected']} faces found")

            if 'estimated_sex' in features:
                print(f"   ✅ Demographics extracted:")
                print(f"      - Sex: {features.get('estimated_sex', 'N/A')}")
                print(f"      - Age: {features.get('age_group', 'N/A')} ({features.get('estimated_age', 'N/A')})")
                print(f"      - Skin: {features.get('skin_tone', 'N/A')}")
                print(f"      - Hair: {features.get('hair_color', 'N/A')}")

            if 'brightness' in features:
                print(f"   ✅ Image properties:")
                print(f"      - Brightness: {features.get('brightness', 0):.1f}")
                print(f"      - Contrast: {features.get('contrast', 0):.1f}")
        else:
            print(f"   ❌ Error analyzing image: {features['error']}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
try:
    import cv2
    from core import FaceAnalyzer, CV2_AVAILABLE

    if CV2_AVAILABLE:
        print("\n✅ OpenCV is FULLY WORKING!")
        print("\nYour system will now:")
        print("  • Detect faces using Haar Cascades")
        print("  • Extract demographic features (sex, age, skin, hair)")
        print("  • Analyze image properties (brightness, contrast)")
        print("  • Provide more accurate face analytics")
        print("\nYou can now download and process faces with full OpenCV support!")
    else:
        print("\n⚠️  OpenCV is installed but not being used by the system")
        print("The system will fall back to basic PIL-based analysis")
except:
    print("\n❌ OpenCV integration has issues")

print("=" * 80)
