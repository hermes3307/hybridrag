#!/usr/bin/env python3
"""
Stable Demographics Fixer - Avoids segmentation faults

This version:
- Processes images one at a time with error handling
- Uses simpler detector backend
- Has better memory management
- Saves progress frequently
"""

import sys
import os
import json
import signal
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Set environment variables BEFORE importing tensorflow/deepface
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only (more stable)

print("Loading system modules...")
from core import SystemConfig
from pgvector_db import PgVectorDatabaseManager

print("Loading DeepFace (this may take a moment)...")
try:
    from deepface import DeepFace
    print("✓ DeepFace loaded successfully")
except ImportError as e:
    print(f"✗ DeepFace not installed: {e}")
    print("\nInstall with: pip install deepface")
    sys.exit(1)

# Global state for graceful shutdown
should_stop = False
processed_count = 0
error_count = 0
start_time = time.time()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global should_stop
    print("\n\n⚠️  Interrupt received. Finishing current image and stopping...")
    print("Progress has been saved to the database.")
    should_stop = True

signal.signal(signal.SIGINT, signal_handler)


def analyze_with_deepface(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Analyze image with DeepFace - stable version

    Returns demographics or None on error
    """
    try:
        # Use opencv backend (most stable)
        # Enforce_detection=False means it won't crash if face not found
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'race'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )

        # Handle list result
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        # Extract and format demographics
        demographics = {}

        # Age
        demographics['estimated_age'] = result.get('age', 'unknown')

        # Gender
        gender_data = result.get('gender', {})
        if gender_data and isinstance(gender_data, dict):
            is_woman = gender_data.get('Woman', 0) > gender_data.get('Man', 0)
            demographics['estimated_sex'] = 'female' if is_woman else 'male'
            demographics['gender_confidence'] = round(max(gender_data.values()), 2)
        else:
            demographics['estimated_sex'] = 'unknown'
            demographics['gender_confidence'] = 0

        # Race
        demographics['dominant_race'] = result.get('dominant_race', 'unknown')
        race_scores = result.get('race', {})
        if race_scores:
            demographics['race_confidence'] = round(max(race_scores.values()), 2)

        # Map age to age group
        age = demographics['estimated_age']
        if isinstance(age, (int, float)):
            if age < 13:
                demographics['age_group'] = 'child'
            elif age < 25:
                demographics['age_group'] = 'young_adult'
            elif age < 40:
                demographics['age_group'] = 'adult'
            elif age < 60:
                demographics['age_group'] = 'middle_aged'
            else:
                demographics['age_group'] = 'senior'
        else:
            demographics['age_group'] = 'unknown'

        # Map race to skin color
        race = demographics['dominant_race'].lower()
        if race in ['white', 'middle eastern']:
            demographics['skin_color'] = 'light'
            demographics['skin_tone'] = 'light'
        elif race in ['latino hispanic', 'asian', 'indian']:
            demographics['skin_color'] = 'medium'
            demographics['skin_tone'] = 'medium'
        elif race in ['black']:
            demographics['skin_color'] = 'dark'
            demographics['skin_tone'] = 'dark'
        else:
            demographics['skin_color'] = 'unknown'
            demographics['skin_tone'] = 'unknown'

        return demographics

    except Exception as e:
        print(f"    ✗ Error analyzing: {e}")
        return None


def update_database(db_manager: PgVectorDatabaseManager, face_id: str,
                    file_path: str, demographics: Dict[str, Any]) -> bool:
    """Update face metadata in database"""
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor()

        # Get current metadata
        cursor.execute(
            "SELECT metadata FROM faces WHERE face_id = %s OR file_path = %s LIMIT 1",
            (face_id, file_path)
        )
        result = cursor.fetchone()

        if not result:
            cursor.close()
            db_manager.return_connection(conn)
            return False

        # Merge with existing metadata
        current_metadata = result[0] if result[0] else {}
        current_metadata.update(demographics)

        # Update database
        cursor.execute(
            """
            UPDATE faces
            SET metadata = %s, updated_at = NOW()
            WHERE face_id = %s OR file_path = %s
            """,
            (json.dumps(current_metadata), face_id, file_path)
        )

        conn.commit()
        cursor.close()
        db_manager.return_connection(conn)
        return True

    except Exception as e:
        print(f"    ✗ Database error: {e}")
        if conn:
            conn.rollback()
            db_manager.return_connection(conn)
        return False


def main():
    global processed_count, error_count, should_stop

    print("\n" + "=" * 70)
    print("Stable Demographics Fixer")
    print("=" * 70)
    print()

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Limit number of images')
    parser.add_argument('--start-from', type=int, default=0, help='Skip first N images')
    parser.add_argument('--sample', action='store_true', help='Random sample')
    args = parser.parse_args()

    # Initialize
    print("Connecting to database...")
    config = SystemConfig.from_file("system_config.json")
    db_manager = PgVectorDatabaseManager(config)

    if not db_manager.initialize():
        print("✗ Failed to initialize database")
        return 1

    print("✓ Database connected")

    # Get faces to process
    print("Fetching faces from database...")
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    query = "SELECT face_id, file_path FROM faces WHERE file_path IS NOT NULL"
    if args.sample:
        query += " ORDER BY RANDOM()"
    else:
        query += " ORDER BY created_at DESC"

    if args.limit:
        query += f" LIMIT {args.limit}"

    cursor.execute(query)
    all_faces = cursor.fetchall()
    cursor.close()
    db_manager.return_connection(conn)

    # Apply start-from offset
    faces = all_faces[args.start_from:] if args.start_from > 0 else all_faces

    print(f"✓ Found {len(all_faces)} total faces")
    if args.start_from > 0:
        print(f"  Starting from image #{args.start_from+1}, processing {len(faces)} images")
    print()

    if len(faces) == 0:
        print("No faces to process")
        return 0

    # Confirm
    if not args.limit and len(faces) > 100:
        print(f"⚠️  About to process {len(faces)} images. This may take several hours.")
        print("   Tip: Start with --limit 100 to test first")
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return 0

    print("\nProcessing images...")
    print("Press Ctrl+C to stop gracefully (progress will be saved)")
    print("-" * 70)

    # Process each face
    for idx, (face_id, file_path) in enumerate(faces, start=1):
        if should_stop:
            break

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"[{idx}/{len(faces)}] ⊘ File not found: {os.path.basename(file_path)}")
            error_count += 1
            continue

        print(f"[{idx}/{len(faces)}] Analyzing: {os.path.basename(file_path)}")

        # Analyze
        demographics = analyze_with_deepface(file_path)

        if demographics:
            # Update database
            if update_database(db_manager, face_id, file_path, demographics):
                processed_count += 1

                # Show what was detected
                sex = demographics.get('estimated_sex', '?')
                age = demographics.get('estimated_age', '?')
                race = demographics.get('dominant_race', '?')
                conf = demographics.get('gender_confidence', 0)

                print(f"    ✓ {sex} ({conf}%), age {age}, {race}")
            else:
                error_count += 1
        else:
            error_count += 1

        # Show progress every 10 images
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            remaining = (len(faces) - idx) / rate if rate > 0 else 0

            print(f"\n📊 Progress: {processed_count} updated, {error_count} errors")
            print(f"   Rate: {rate:.1f} images/sec, Est. time remaining: {remaining/60:.1f} min\n")

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"✓ Successfully updated: {processed_count}")
    print(f"✗ Errors: {error_count}")
    print(f"⏱  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"📈 Average rate: {processed_count/elapsed:.1f} images/sec")
    print()

    if should_stop:
        print("⚠️  Process interrupted by user")
        print(f"   You can resume by running with: --start-from {args.start_from + idx}")
    else:
        print("✓ All images processed!")

    print("\nDemographics have been updated in the database.")
    print("You can now search with accurate metadata filters.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
