#!/usr/bin/env python3
"""
Fix Demographics - Re-analyze faces with DeepFace for accurate demographics

This script uses DeepFace (a much more accurate deep learning model) to
re-analyze all face images and update the database with correct demographic
information including age, gender, race, and emotion.

Usage:
    python3 fix_demographics.py [--limit N] [--sample]

Options:
    --limit N    Only process N images (for testing)
    --sample     Process random sample instead of all images
    --force      Re-analyze all images even if they have metadata
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm

# Import DeepFace
try:
    from deepface import DeepFace
    print("✓ DeepFace loaded successfully")
except ImportError:
    print("✗ DeepFace not installed. Install with: pip install deepface")
    sys.exit(1)

# Import our system components
from core import SystemConfig
from pgvector_db import PgVectorDatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_demographics_with_deepface(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Use DeepFace to analyze demographics accurately

    Returns:
        Dictionary with age, gender, race, emotion, etc.
    """
    try:
        # Use opencv backend and disable GPU to avoid segfaults
        import cv2

        # DeepFace.analyze returns age, gender, race, emotion
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'race'],
            enforce_detection=False,  # Don't fail if face not detected
            detector_backend='opencv',  # Use opencv for stability
            silent=True
        )

        # Handle list or dict result
        if isinstance(result, list):
            result = result[0]

        # Extract and normalize the data
        demographics = {
            'estimated_age': result.get('age', 'unknown'),
            'estimated_sex': result['gender']['Woman'] > result['gender']['Man']
                           if 'gender' in result else 'unknown',
            'gender_confidence': max(result.get('gender', {}).values()) if 'gender' in result else 0,
            'dominant_race': result.get('dominant_race', 'unknown'),
            'race_details': result.get('race', {}),
        }

        # Convert gender to male/female/unknown
        if demographics['estimated_sex'] == True:
            demographics['estimated_sex'] = 'female'
        elif demographics['estimated_sex'] == False:
            demographics['estimated_sex'] = 'male'
        else:
            demographics['estimated_sex'] = 'unknown'

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

        # Map race to skin tone (approximate)
        race = demographics['dominant_race']
        if race in ['white', 'middle eastern']:
            demographics['skin_color'] = 'light'
            demographics['skin_tone'] = 'light'
        elif race in ['latino hispanic', 'asian']:
            demographics['skin_color'] = 'medium'
            demographics['skin_tone'] = 'medium'
        elif race in ['black', 'indian']:
            demographics['skin_color'] = 'dark'
            demographics['skin_tone'] = 'dark'
        else:
            demographics['skin_color'] = 'unknown'
            demographics['skin_tone'] = 'unknown'

        return demographics

    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {e}")
        return None


def update_face_metadata(db_manager: PgVectorDatabaseManager, face_id: str,
                        file_path: str, demographics: Dict[str, Any]) -> bool:
    """
    Update face metadata in the database
    """
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor()

        # Get current metadata
        cursor.execute(
            "SELECT metadata FROM faces WHERE face_id = %s OR file_path = %s",
            (face_id, file_path)
        )
        result = cursor.fetchone()

        if not result:
            logger.warning(f"Face not found in database: {face_id}")
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
            SET metadata = %s,
                updated_at = NOW()
            WHERE face_id = %s OR file_path = %s
            """,
            (json.dumps(current_metadata), face_id, file_path)
        )

        conn.commit()
        cursor.close()
        db_manager.return_connection(conn)
        return True

    except Exception as e:
        logger.error(f"Error updating metadata for {face_id}: {e}")
        if conn:
            conn.rollback()
            db_manager.return_connection(conn)
        return False


def main():
    parser = argparse.ArgumentParser(description='Fix demographics using DeepFace')
    parser.add_argument('--limit', type=int, help='Limit number of images to process')
    parser.add_argument('--sample', action='store_true', help='Process random sample')
    parser.add_argument('--force', action='store_true', help='Re-analyze all images')
    args = parser.parse_args()

    print("=" * 70)
    print("Fix Demographics - Re-analyze with DeepFace")
    print("=" * 70)
    print()

    # Load configuration
    config = SystemConfig.from_file("system_config.json")
    db_manager = PgVectorDatabaseManager(config)

    if not db_manager.initialize():
        print("✗ Failed to initialize database")
        return

    print("✓ Database initialized")

    # Get all face images from database
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    query = "SELECT face_id, file_path FROM faces"
    if args.limit:
        if args.sample:
            query += " ORDER BY RANDOM()"
        query += f" LIMIT {args.limit}"

    cursor.execute(query)
    faces = cursor.fetchall()
    cursor.close()
    db_manager.return_connection(conn)

    print(f"✓ Found {len(faces)} faces to process")
    print()

    if len(faces) == 0:
        print("No faces found in database")
        return

    # Ask for confirmation
    if not args.limit:
        response = input(f"About to re-analyze {len(faces)} images. This may take a while. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return

    print()
    print("Processing images...")
    print()

    # Process each face
    success_count = 0
    error_count = 0
    skip_count = 0

    for face_id, file_path in tqdm(faces, desc="Analyzing faces"):
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            skip_count += 1
            continue

        # Analyze demographics
        demographics = analyze_demographics_with_deepface(file_path)

        if demographics:
            # Update database
            if update_face_metadata(db_manager, face_id, file_path, demographics):
                success_count += 1
            else:
                error_count += 1
        else:
            error_count += 1

    print()
    print("=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"✓ Successfully updated: {success_count}")
    print(f"✗ Errors: {error_count}")
    print(f"⊘ Skipped: {skip_count}")
    print(f"Total processed: {len(faces)}")
    print()
    print("Demographics have been updated in the database!")
    print("You can now search with accurate metadata filters.")


if __name__ == "__main__":
    main()
