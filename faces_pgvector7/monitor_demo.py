#!/usr/bin/env python3
"""
Demo script to showcase monitor capabilities
Shows how to use the DatabaseMonitor class programmatically
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from monitor import DatabaseMonitor
import time

def demo_monitor():
    """Demonstrate monitor capabilities"""
    print("=" * 70)
    print("pgvector Database Monitor - Demo")
    print("=" * 70)

    # Initialize monitor
    print("\n1. Initializing database monitor...")
    monitor = DatabaseMonitor()

    if not monitor.initialize():
        print("✗ Failed to initialize monitor")
        return False

    print("✓ Monitor initialized successfully")

    # Get vector count
    print("\n2. Getting vector count...")
    vector_count = monitor.get_vector_count()
    total_faces = monitor.get_total_faces()
    print(f"✓ Total faces: {total_faces}")
    print(f"✓ Vectors (with embeddings): {vector_count}")

    # Get connection info
    print("\n3. Getting connection information...")
    connections = monitor.get_connection_info()
    print(f"✓ Active connections: {len(connections)}")
    for conn in connections[:5]:  # Show first 5
        print(f"   - PID {conn['pid']}: {conn['user']} ({conn['state']})")

    # Get database stats
    print("\n4. Getting database statistics...")
    stats = monitor.get_database_stats()
    print(f"✓ Database size: {stats.get('database_size', 'N/A')}")
    print(f"✓ Table size: {stats.get('table_size', 'N/A')}")
    print(f"✓ Embedding models:")
    for model, count in stats.get('embedding_models', {}).items():
        print(f"   - {model}: {count}")

    # Get face list
    print("\n5. Getting recent faces...")
    faces = monitor.get_face_list(limit=5)
    print(f"✓ Retrieved {len(faces)} recent faces:")
    for face in faces:
        has_emb = "✓" if face['has_embedding'] else "✗"
        print(f"   {has_emb} {face['face_id'][:30]}... ({face['embedding_model']})")

    # Get detailed info for first face
    if faces:
        print("\n6. Getting detailed information for first face...")
        details = monitor.get_face_details(faces[0]['face_id'])
        if details:
            print(f"✓ Face ID: {details['face_id']}")
            print(f"  File path: {details['file_path']}")
            print(f"  Image hash: {details['image_hash'][:16]}...")
            print(f"  Age estimate: {details.get('age_estimate', 'N/A')}")
            print(f"  Gender: {details.get('gender', 'N/A')}")
            print(f"  Brightness: {details.get('brightness', 'N/A')}")
            print(f"  Embedding dimension: {details['embedding_dimension']}")
            print(f"  Has embedding: {details['has_embedding']}")

    # Simulate real-time monitoring
    print("\n7. Simulating real-time monitoring (5 seconds)...")
    print("   Watching vector count...")

    start_count = monitor.get_vector_count()
    for i in range(5):
        time.sleep(1)
        current_count = monitor.get_vector_count()
        if current_count != start_count:
            print(f"   ⚡ Vector count changed: {start_count} → {current_count}")
            start_count = current_count
        else:
            print(f"   - Check {i+1}/5: {current_count} vectors")

    # Cleanup
    print("\n8. Closing connections...")
    monitor.close()
    print("✓ Connections closed")

    print("\n" + "=" * 70)
    print("✓ Demo completed successfully!")
    print("=" * 70)
    print("\nTo launch the full GUI:")
    print("  python3 monitor.py")
    print("")

    return True


if __name__ == "__main__":
    try:
        success = demo_monitor()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n✗ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
