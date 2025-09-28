#!/usr/bin/env python3
"""
Test comparison quality with a new face download
"""

from face_collector import FaceCollector, process_faces
from face_database import FaceDatabase, FaceSearchInterface
import os

def main():
    print("🔍 Testing New Face Comparison Quality")
    print("="*50)

    # Initialize database
    face_db = FaceDatabase()
    search_interface = FaceSearchInterface(face_db)

    print(f"📊 Current database: {face_db.get_database_stats()['total_faces']} faces")

    # Download a new test face
    print("\n📥 Downloading new test face...")
    collector = FaceCollector(delay=1.0)
    new_face_files = collector.download_faces_batch(count=1, max_workers=1)

    if not new_face_files:
        print("❌ Failed to download test face")
        return

    test_face_path = new_face_files[0]
    print(f"✅ Downloaded test face: {os.path.basename(test_face_path)}")

    # Test comparison with existing faces
    print(f"\n🔍 Testing similarity search...")
    search_results = search_interface.search_by_image(test_face_path, n_results=10)

    if "error" in search_results:
        print(f"❌ Error: {search_results['error']}")
        return

    results = search_results["results"]
    query_features = search_results["query_features"]

    print(f"📋 Test face features:")
    print(f"   Age group: {query_features.get('estimated_age_group', 'unknown')}")
    print(f"   Skin tone: {query_features.get('estimated_skin_tone', 'unknown')}")
    print(f"   Quality: {query_features.get('image_quality', 'unknown')}")
    print(f"   Brightness: {query_features.get('brightness', 0):.1f}")

    print(f"\n🎯 Top {min(10, results['count'])} most similar faces:")
    for i in range(min(10, results['count'])):
        face_id = results['ids'][i]
        distance = results['distances'][i] if i < len(results['distances']) else 'N/A'
        similarity = (1 - distance) * 100 if isinstance(distance, (int, float)) else 'N/A'

        if i < len(results['metadatas']):
            metadata = results['metadatas'][i]
            age_group = metadata.get('estimated_age_group', 'unknown')
            skin_tone = metadata.get('estimated_skin_tone', 'unknown')

            print(f"   {i+1:2d}. {face_id[:20]}... (similarity: {similarity:.1f}%)")
            print(f"       Age: {age_group}, Skin: {skin_tone}")

    # Analyze quality of matches
    if results['distances']:
        distances = results['distances']
        similarities = [(1-d)*100 for d in distances]

        print(f"\n📈 Comparison Quality Analysis:")
        print(f"   Best match: {max(similarities):.1f}%")
        print(f"   Average similarity: {sum(similarities)/len(similarities):.1f}%")
        print(f"   Matches >90%: {sum(1 for s in similarities if s > 90)}")
        print(f"   Matches >80%: {sum(1 for s in similarities if s > 80)}")
        print(f"   Matches >70%: {sum(1 for s in similarities if s > 70)}")

    # Test feature-based search for same characteristics
    print(f"\n🔎 Testing feature-based search...")

    test_age = query_features.get('estimated_age_group')
    test_tone = query_features.get('estimated_skin_tone')

    if test_age:
        age_matches = face_db.search_by_features({"estimated_age_group": test_age}, n_results=20)
        print(f"   Same age group ({test_age}): {age_matches['count']} matches")

    if test_tone:
        tone_matches = face_db.search_by_features({"estimated_skin_tone": test_tone}, n_results=20)
        print(f"   Same skin tone ({test_tone}): {tone_matches['count']} matches")

    print(f"\n✅ Face comparison test completed!")

if __name__ == "__main__":
    main()