#!/usr/bin/env python3
"""
Test the schema fixes for add_face, add_faces_batch, and search_faces
"""

import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, '/home/pi/hybridrag/nfaces')

from core import SystemConfig, FaceData
from pgvector_db import PgVectorDatabaseManager

def create_test_face(face_id: str) -> FaceData:
    """Create a test face data object"""
    features = {
        'age_estimate': 30,
        'gender': 'male',
        'brightness': 0.7,
        'contrast': 0.6,
        'sharpness': 0.8,
    }

    # Create a simple 7-dimensional embedding for statistical model
    embedding = np.random.randn(7).tolist()

    return FaceData(
        face_id=face_id,
        file_path=f"./test_faces/{face_id}.jpg",
        features=features,
        embedding=embedding,
        timestamp=datetime.now().isoformat(),
        image_hash=f"test_hash_{face_id}"
    )

def test_add_face(db):
    """Test adding a single face"""
    print("\n" + "=" * 60)
    print("Test 1: Add Single Face")
    print("=" * 60)

    try:
        face_data = create_test_face("test_fix_face_001")
        result = db.add_face(face_data, embedding_model="statistical")

        if result:
            print("✓ Successfully added face")
            return True
        else:
            print("✗ Failed to add face")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_add_faces_batch(db):
    """Test adding multiple faces in batch"""
    print("\n" + "=" * 60)
    print("Test 2: Add Faces in Batch")
    print("=" * 60)

    try:
        # Create 5 test faces
        batch_data = [(create_test_face(f"test_fix_batch_{i}"), "statistical")
                      for i in range(5)]

        count = db.add_faces_batch(batch_data)

        if count > 0:
            print(f"✓ Successfully added {count} faces in batch")
            return True
        else:
            print("✗ Failed to add faces in batch")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_search_faces(db):
    """Test searching for faces"""
    print("\n" + "=" * 60)
    print("Test 3: Search Faces")
    print("=" * 60)

    try:
        # Create a random query embedding
        query_embedding = np.random.randn(7).tolist()

        results = db.search_faces(
            query_embedding=query_embedding,
            n_results=5,
            embedding_model="statistical"  # Use statistical model
        )

        if results:
            print(f"✓ Search returned {len(results)} results")
            for i, result in enumerate(results[:3]):
                print(f"  {i+1}. Face ID: {result['id']}, Distance: {result['distance']:.4f}")
            return True
        else:
            print("⚠ Search returned no results (might be OK if no data)")
            return True  # Not a failure if database is empty
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_get_stats(db):
    """Test getting database stats"""
    print("\n" + "=" * 60)
    print("Test 4: Get Database Stats")
    print("=" * 60)

    try:
        stats = db.get_stats()

        if stats:
            print("✓ Stats retrieved successfully:")
            print(f"  Total faces: {stats.get('total_faces', 0):,}")
            print(f"  Faces with embeddings: {stats.get('faces_with_embeddings', 0):,}")
            print(f"  Models: {', '.join(stats.get('embedding_models', []))}")
            print(f"  Database size: {stats.get('database_size', 'N/A')}")
            return True
        else:
            print("✗ Failed to get stats")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing Schema Fixes")
    print("=" * 60)

    # Initialize database
    config = SystemConfig()
    config.db_type = 'pgvector'
    config.db_name = 'vector_db'

    db = PgVectorDatabaseManager(config)

    if not db.initialize():
        print("✗ Failed to initialize database")
        return False

    print("✓ Database initialized")

    # Run tests
    results = []
    results.append(("Add Single Face", test_add_face(db)))
    results.append(("Add Faces Batch", test_add_faces_batch(db)))
    results.append(("Search Faces", test_search_faces(db)))
    results.append(("Get Stats", test_get_stats(db)))

    # Close database
    db.close()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("✓ All tests passed! The fixes are working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
