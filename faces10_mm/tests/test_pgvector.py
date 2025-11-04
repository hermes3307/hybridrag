#!/usr/bin/env python3
"""
Test script for pgvector database functionality

Tests:
1. Database connection
2. Adding faces with embeddings
3. Vector similarity search
4. Metadata filtering
5. Database statistics
"""

import sys
import logging
from datetime import datetime
import numpy as np

# Import necessary classes
from core import SystemConfig, FaceData
from pgvector_db import PgVectorDatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_face_data(face_id: str, embedding_dim: int = 7) -> FaceData:
    """Create a test FaceData object"""
    # Generate random embedding
    embedding = np.random.randn(embedding_dim).tolist()

    # Create features
    features = {
        'age_estimate': np.random.randint(18, 70),
        'gender': np.random.choice(['male', 'female']),
        'brightness': np.random.uniform(0.3, 0.9),
        'contrast': np.random.uniform(0.3, 0.9),
        'sharpness': np.random.uniform(0.5, 1.0),
        'has_glasses': np.random.choice([True, False]),
        'has_smile': np.random.choice([True, False])
    }

    # Create FaceData
    face_data = FaceData(
        face_id=face_id,
        file_path=f"./faces/{face_id}.jpg",
        features=features,
        embedding=embedding,
        timestamp=datetime.now().isoformat(),
        image_hash=f"hash_{face_id}"
    )

    return face_data

def test_database_connection():
    """Test 1: Database connection"""
    logger.info("=" * 60)
    logger.info("Test 1: Database Connection")
    logger.info("=" * 60)

    try:
        config = SystemConfig()
        config.db_type = "pgvector"
        config.db_name = "vector_db"

        db_manager = PgVectorDatabaseManager(config)

        if db_manager.initialize():
            logger.info("✓ Database connection successful")
            return db_manager
        else:
            logger.error("✗ Database connection failed")
            return None

    except Exception as e:
        logger.error(f"✗ Connection test failed: {e}")
        return None

def test_add_faces(db_manager):
    """Test 2: Adding faces to database"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Adding Faces")
    logger.info("=" * 60)

    try:
        # Add individual faces
        for i in range(5):
            face_data = create_test_face_data(f"test_face_{i}")
            if db_manager.add_face(face_data, embedding_model="statistical"):
                logger.info(f"✓ Added face {face_data.face_id}")
            else:
                logger.error(f"✗ Failed to add face {face_data.face_id}")

        # Test batch insert
        batch_faces = [(create_test_face_data(f"batch_face_{i}"), "statistical")
                       for i in range(5)]
        count = db_manager.add_faces_batch(batch_faces)
        logger.info(f"✓ Batch insert: {count} faces added")

        # Get total count
        total = db_manager.get_count()
        logger.info(f"✓ Total faces in database: {total}")

        return True

    except Exception as e:
        logger.error(f"✗ Add faces test failed: {e}")
        return False

def test_vector_search(db_manager):
    """Test 3: Vector similarity search"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Vector Similarity Search")
    logger.info("=" * 60)

    try:
        # Create query embedding
        query_embedding = np.random.randn(7).tolist()

        # Search for similar faces
        results = db_manager.search_faces(
            query_embedding=query_embedding,
            n_results=3,
            distance_metric='cosine'
        )

        logger.info(f"✓ Found {len(results)} similar faces")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. Face: {result['id']}, Distance: {result['distance']:.4f}")
            logger.info(f"      Gender: {result['metadata'].get('gender')}, "
                       f"Age: {result['metadata'].get('age_estimate')}")

        return True

    except Exception as e:
        logger.error(f"✗ Vector search test failed: {e}")
        return False

def test_metadata_search(db_manager):
    """Test 4: Metadata filtering"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Metadata Filtering")
    logger.info("=" * 60)

    try:
        # Search by metadata only
        results = db_manager.search_by_metadata(
            metadata_filter={'gender': 'female'},
            n_results=5
        )

        logger.info(f"✓ Found {len(results)} female faces")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. Face: {result['id']}, "
                       f"Age: {result['metadata'].get('age_estimate')}")

        # Hybrid search: vector + metadata
        query_embedding = np.random.randn(7).tolist()
        results = db_manager.search_faces(
            query_embedding=query_embedding,
            n_results=3,
            metadata_filter={'gender': 'male'}
        )

        logger.info(f"✓ Hybrid search: Found {len(results)} male faces")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. Face: {result['id']}, Distance: {result['distance']:.4f}")

        return True

    except Exception as e:
        logger.error(f"✗ Metadata search test failed: {e}")
        return False

def test_database_stats(db_manager):
    """Test 5: Database statistics"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Database Statistics")
    logger.info("=" * 60)

    try:
        stats = db_manager.get_stats()

        logger.info(f"✓ Total faces: {stats.get('total_faces')}")
        logger.info(f"✓ Faces with embeddings: {stats.get('faces_with_embeddings')}")
        logger.info(f"✓ Embedding models: {stats.get('embedding_models')}")
        logger.info(f"✓ Database size: {stats.get('database_size')}")

        # Get collection info
        info = db_manager.get_collection_info()
        logger.info(f"✓ Collection name: {info.get('name')}")
        logger.info(f"✓ Collection count: {info.get('count')}")
        logger.info(f"✓ Database path: {info.get('path')}")

        return True

    except Exception as e:
        logger.error(f"✗ Database stats test failed: {e}")
        return False

def test_duplicate_check(db_manager):
    """Test 6: Duplicate detection"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: Duplicate Detection")
    logger.info("=" * 60)

    try:
        # Check for existing hash
        is_duplicate = db_manager.check_duplicate("hash_test_face_0")
        logger.info(f"✓ Duplicate check for existing hash: {is_duplicate}")

        # Check for non-existing hash
        is_duplicate = db_manager.check_duplicate("nonexistent_hash")
        logger.info(f"✓ Duplicate check for new hash: {is_duplicate}")

        return True

    except Exception as e:
        logger.error(f"✗ Duplicate check test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting pgvector database tests...\n")

    # Test 1: Connection
    db_manager = test_database_connection()
    if not db_manager:
        logger.error("Cannot proceed without database connection")
        sys.exit(1)

    # Test 2: Add faces
    if not test_add_faces(db_manager):
        logger.error("Add faces test failed")

    # Test 3: Vector search
    if not test_vector_search(db_manager):
        logger.error("Vector search test failed")

    # Test 4: Metadata search
    if not test_metadata_search(db_manager):
        logger.error("Metadata search test failed")

    # Test 5: Database stats
    if not test_database_stats(db_manager):
        logger.error("Database stats test failed")

    # Test 6: Duplicate check
    if not test_duplicate_check(db_manager):
        logger.error("Duplicate check test failed")

    # Cleanup
    logger.info("\n" + "=" * 60)
    logger.info("Cleaning up test data...")
    logger.info("=" * 60)

    # Optional: Uncomment to clean up test data
    # if db_manager.reset_database():
    #     logger.info("✓ Database reset successful")
    # else:
    #     logger.error("✗ Database reset failed")

    db_manager.close()
    logger.info("\n✓ All tests completed!")

if __name__ == "__main__":
    main()
