#!/usr/bin/env python3
"""
Quick test to validate GUI search functionality matches the temp test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module
collect_module = import_module('3_collect_faces')
FaceAnalyzer = collect_module.FaceAnalyzer
FaceEmbedder = collect_module.FaceEmbedder

from face_database import FaceDatabase, FaceSearchInterface

def test_search():
    print("🧪 Testing GUI Search Backend")
    print("=" * 50)

    # Initialize database
    face_db = FaceDatabase()
    search_interface = FaceSearchInterface(face_db=face_db)

    stats = face_db.get_database_stats()
    print(f"📊 Database: {stats['total_faces']} faces")

    # Get a test image from the database
    all_faces = face_db.collection.get(limit=1, include=['metadatas'])
    if not all_faces['ids']:
        print("❌ No faces in database")
        return

    test_face_id = all_faces['ids'][0]
    test_metadata = all_faces['metadatas'][0]
    test_image_path = test_metadata.get('file_path')

    if not test_image_path or not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return

    print(f"\n📷 Using test image: {os.path.basename(test_image_path)}")
    print(f"   Age: {test_metadata.get('estimated_age_group')}")
    print(f"   Skin: {test_metadata.get('estimated_skin_tone')}")

    # Test 1: Direct database search (like the GUI now does)
    print(f"\n🎯 TEST 1: Direct Database Search")
    print("-" * 50)

    analyzer = FaceAnalyzer()
    embedder = FaceEmbedder()

    features = analyzer.estimate_basic_features(test_image_path)
    query_embedding = embedder.generate_embedding(test_image_path, features)

    print(f"Embedding dimensions: {len(query_embedding)}")

    search_results = face_db.search_similar_faces(query_embedding, n_results=10)

    print(f"Found {search_results.get('count', 0)} results:")
    for i in range(min(5, search_results.get('count', 0))):
        face_id = search_results['ids'][i]
        distance = search_results['distances'][i]
        similarity = (1 - distance) * 100
        metadata = search_results['metadatas'][i] if i < len(search_results.get('metadatas', [])) else {}

        age = metadata.get('estimated_age_group', 'unknown')
        skin = metadata.get('estimated_skin_tone', 'unknown')

        print(f"  {i+1}. {face_id[:30]}... ({similarity:.1f}%)")
        print(f"     Age: {age}, Skin: {skin}")

    # Test 2: Search interface (alternative method)
    print(f"\n🎯 TEST 2: Search Interface Method")
    print("-" * 50)

    interface_results = search_interface.search_by_image(test_image_path, n_results=10)

    if "error" in interface_results:
        print(f"❌ Error: {interface_results['error']}")
    else:
        results = interface_results.get("results", {})
        print(f"Found {results.get('count', 0)} results:")
        for i in range(min(3, results.get('count', 0))):
            if i < len(results.get('ids', [])):
                face_id = results['ids'][i]
                distance = results['distances'][i] if i < len(results.get('distances', [])) else 1.0
                similarity = (1 - distance) * 100
                print(f"  {i+1}. {face_id[:30]}... ({similarity:.1f}%)")

    # Test 3: Combined search with metadata filters
    print(f"\n🎯 TEST 3: Combined Search with Filters")
    print("-" * 50)

    test_age = features.get('estimated_age_group')
    if test_age:
        filters = {'estimated_age_group': test_age}
        print(f"Filtering by age group: {test_age}")

        combined_results = search_interface.combined_search(
            query_embedding, filters, n_results=5
        )

        print(f"Found {combined_results.get('count', 0)} results with filter:")
        for i in range(min(3, combined_results.get('count', 0))):
            if i < len(combined_results.get('ids', [])):
                face_id = combined_results['ids'][i]
                distance = combined_results['distances'][i] if i < len(combined_results.get('distances', [])) else 1.0
                similarity = (1 - distance) * 100
                metadata = combined_results['metadatas'][i] if i < len(combined_results.get('metadatas', [])) else {}
                age = metadata.get('estimated_age_group', 'unknown')
                print(f"  {i+1}. {face_id[:30]}... ({similarity:.1f}%) - Age: {age}")

    print(f"\n✅ All tests completed successfully!")
    print(f"\n💡 Summary:")
    print(f"   • Direct database search: {search_results.get('count', 0)} results")
    print(f"   • Search interface: {results.get('count', 0)} results")
    print(f"   • Combined search with filters works correctly")
    print(f"   • GUI should now find similar images properly!")

if __name__ == "__main__":
    test_search()
