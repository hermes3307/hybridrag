#!/usr/bin/env python3
"""
Test that the app can retrieve stats without errors
"""

import sys
sys.path.insert(0, '/home/pi/hybridrag/nfaces')

from core import SystemConfig
from pgvector_db import PgVectorDatabaseManager

def test_stats():
    print("=" * 60)
    print("Testing Database Stats After Fix")
    print("=" * 60)
    print()

    config = SystemConfig()
    config.db_type = 'pgvector'
    config.db_name = 'vector_db'

    db = PgVectorDatabaseManager(config)

    if not db.initialize():
        print("✗ Failed to initialize database")
        return False

    print("✓ Database initialized successfully")
    print()

    # Test get_stats
    print("Retrieving statistics...")
    stats = db.get_stats()

    if not stats:
        print("✗ Failed to retrieve stats")
        db.close()
        return False

    print("✓ Stats retrieved successfully!")
    print()
    print("Database Statistics:")
    print(f"  Total faces: {stats.get('total_faces', 0):,}")
    print(f"  Faces with embeddings: {stats.get('faces_with_embeddings', 0):,}")
    print(f"  Embedding models: {', '.join(stats.get('embedding_models', []))}")
    print(f"  Oldest face: {stats.get('oldest_face', 'N/A')}")
    print(f"  Newest face: {stats.get('newest_face', 'N/A')}")
    print(f"  Database size: {stats.get('database_size', 'N/A')}")
    print()

    db.close()

    print("=" * 60)
    print("✓ All tests passed! The fix is working correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_stats()
    sys.exit(0 if success else 1)
