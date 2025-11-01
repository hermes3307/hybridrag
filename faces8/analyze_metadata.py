#!/usr/bin/env python3
"""Analyze metadata from downloaded face images"""

import json
import sys
from pathlib import Path
from collections import Counter

def analyze_metadata(directory="faces_final"):
    """Analyze JSON metadata files in directory"""

    dir_path = Path(directory)
    json_files = list(dir_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    print(f"=== METADATA ANALYSIS ===")
    print(f"Directory: {directory}")
    print(f"Total faces: {len(json_files)}\n")

    # Collect statistics
    sexes = []
    age_groups = []
    skin_tones = []
    hair_colors = []
    brightness = []
    face_counts = []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                attrs = data.get('queryable_attributes', {})
                features = data.get('face_features', {})

                sexes.append(attrs.get('sex', 'unknown'))
                age_groups.append(attrs.get('age_group', 'unknown'))
                skin_tones.append(attrs.get('skin_tone', 'unknown'))
                hair_colors.append(attrs.get('hair_color', 'unknown'))
                brightness.append(attrs.get('brightness_level', 'unknown'))
                face_counts.append(features.get('faces_detected', 0))
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # Print statistics
    print("SEX DISTRIBUTION:")
    for sex, count in Counter(sexes).most_common():
        pct = (count / len(json_files)) * 100
        print(f"  {sex:12s}: {count:3d} ({pct:5.1f}%)")

    print("\nAGE GROUP DISTRIBUTION:")
    for age, count in Counter(age_groups).most_common():
        pct = (count / len(json_files)) * 100
        print(f"  {age:12s}: {count:3d} ({pct:5.1f}%)")

    print("\nSKIN TONE DISTRIBUTION:")
    for tone, count in Counter(skin_tones).most_common():
        pct = (count / len(json_files)) * 100
        print(f"  {tone:12s}: {count:3d} ({pct:5.1f}%)")

    print("\nHAIR COLOR DISTRIBUTION:")
    for color, count in Counter(hair_colors).most_common():
        pct = (count / len(json_files)) * 100
        print(f"  {color:12s}: {count:3d} ({pct:5.1f}%)")

    print("\nBRIGHTNESS DISTRIBUTION:")
    for bright, count in Counter(brightness).most_common():
        pct = (count / len(json_files)) * 100
        print(f"  {bright:12s}: {count:3d} ({pct:5.1f}%)")

    print("\nFACE DETECTION:")
    for faces, count in Counter(face_counts).most_common():
        pct = (count / len(json_files)) * 100
        print(f"  {faces} face(s): {count:3d} ({pct:5.1f}%)")

    print("\n" + "="*50)

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "faces_final"
    analyze_metadata(directory)
