#!/usr/bin/env python3
"""
Fix Demographics - Simple version using DeepFace

This script re-analyzes face images with DeepFace to get accurate demographics.
"""

import sys
import os
import json
from pathlib import Path

print("Loading modules...")

# Import our system components first
from core import SystemConfig
from pgvector_db import PgVectorDatabaseManager

print("Initializing database...")

# Load configuration
config = SystemConfig.from_file("system_config.json")
db_manager = PgVectorDatabaseManager(config)

if not db_manager.initialize():
    print("✗ Failed to initialize database")
    sys.exit(1)

print("✓ Database initialized")

# Get a sample face to test
conn = db_manager.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT face_id, file_path FROM faces WHERE file_path IS NOT NULL LIMIT 1")
result = cursor.fetchone()
cursor.close()
db_manager.return_connection(conn)

if not result:
    print("No faces found")
    sys.exit(1)

face_id, file_path = result
print(f"\nTesting with: {file_path}")

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    sys.exit(1)

print("\nLoading DeepFace...")
try:
    from deepface import DeepFace
    print("✓ DeepFace loaded")
except ImportError:
    print("✗ DeepFace not installed")
    sys.exit(1)

print("\nAnalyzing image...")
try:
    result = DeepFace.analyze(
        img_path=file_path,
        actions=['age', 'gender', 'race'],
        enforce_detection=False,
        detector_backend='opencv',
        silent=True
    )

    if isinstance(result, list):
        result = result[0]

    print("\n" + "=" * 70)
    print("Analysis Results:")
    print("=" * 70)
    print(f"Age: {result.get('age', 'unknown')}")
    print(f"Gender: {result.get('gender', {})}")
    print(f"Race: {result.get('dominant_race', 'unknown')}")
    print(f"Race details: {result.get('race', {})}")
    print()

    # Determine gender
    gender_data = result.get('gender', {})
    if gender_data:
        is_woman = gender_data.get('Woman', 0) > gender_data.get('Man', 0)
        estimated_sex = 'female' if is_woman else 'male'
        confidence = max(gender_data.values())
        print(f"Estimated Sex: {estimated_sex} (confidence: {confidence:.1f}%)")

    print("\n✓ Test successful! DeepFace is working correctly.")
    print("\nTo process all images, run:")
    print("  python3 fix_demographics_batch.py")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
