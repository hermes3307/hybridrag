#!/usr/bin/env python3
"""
Test script for face collection and search system
"""

import os
import json
from face_database import FaceDatabase, FaceSearchInterface
from face_collector import process_faces
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_face_system():
    """Test the complete face system"""
    print("🎭 Testing Face Collection and Search System")
    print("="*60)

    # Initialize database
    face_db = FaceDatabase()

    # Check if we have existing face data
    if os.path.exists("face_data.json"):
        print("📁 Loading existing face data...")
        with open("face_data.json", 'r') as f:
            face_data_dicts = json.load(f)

        # Convert back to FaceData objects
        from face_collector import FaceData
        face_data_list = []
        for data_dict in face_data_dicts:
            face_data = FaceData(**data_dict)
            face_data_list.append(face_data)

        print(f"✅ Loaded {len(face_data_list)} faces from file")

        # Add to database
        added_count = face_db.add_faces(face_data_list)
        print(f"✅ Added {added_count} faces to database")

    # Get database stats
    print("\n📊 Database Statistics:")
    stats = face_db.get_database_stats()
    print(f"Total faces: {stats['total_faces']}")

    if "age_group_distribution" in stats:
        print("Age group distribution:", stats["age_group_distribution"])
    if "skin_tone_distribution" in stats:
        print("Skin tone distribution:", stats["skin_tone_distribution"])
    if "quality_distribution" in stats:
        print("Quality distribution:", stats["quality_distribution"])

    # Test similarity search
    if stats['total_faces'] > 1:
        print("\n🔍 Testing similarity search...")

        # Get the first face and search for similar ones
        all_faces = face_db.collection.get(limit=1, include=["embeddings"])
        embeddings_list = all_faces.get("embeddings", [])
        print(f"Debug: embeddings_list type: {type(embeddings_list)}, length: {len(embeddings_list) if hasattr(embeddings_list, '__len__') else 'N/A'}")

        try:
            if len(embeddings_list) > 0:
                first_embedding = embeddings_list[0]

                results = face_db.search_similar_faces(first_embedding, n_results=3)
                print(f"Found {results['count']} similar faces:")

                for i in range(min(3, results['count'])):
                    face_id = results['ids'][i]
                    distance = results['distances'][i] if i < len(results['distances']) else 'N/A'
                    similarity = 1 - distance if isinstance(distance, float) else 'N/A'

                    print(f"  {i+1}. {face_id} (similarity: {similarity})")
            else:
                print("No embeddings found")
        except Exception as e:
            print(f"Could not access embeddings: {e}")

    # Test feature-based search
    print("\n🎯 Testing feature-based search...")

    # Search for faces with specific features
    feature_searches = [
        {"estimated_age_group": "adult"},
        {"estimated_skin_tone": "light"},
        {"image_quality": "high"}
    ]

    for filters in feature_searches:
        results = face_db.search_by_features(filters, n_results=5)
        filter_desc = ", ".join([f"{k}={v}" for k, v in filters.items()])
        print(f"Search '{filter_desc}': {results['count']} results")

    # Test duplicate detection
    print("\n🔍 Testing duplicate detection...")
    duplicates = face_db.find_duplicate_faces(similarity_threshold=0.95)
    print(f"Found {len(duplicates)} groups of similar faces")

    # Show available face files
    faces_dir = "./faces"
    if os.path.exists(faces_dir):
        face_files = [f for f in os.listdir(faces_dir) if f.endswith('.jpg')]
        print(f"\n📁 Available face files: {len(face_files)}")

        if face_files:
            print("Sample files:")
            for i, file in enumerate(face_files[:5]):
                print(f"  {i+1}. {file}")

            # Test image search with first available file
            if face_files:
                print(f"\n🖼️  Testing image search with {face_files[0]}...")
                search_interface = FaceSearchInterface(face_db)

                test_image_path = os.path.join(faces_dir, face_files[0])
                search_results = search_interface.search_by_image(test_image_path, n_results=3)

                if "error" not in search_results:
                    results = search_results["results"]
                    print(f"Found {results['count']} similar faces:")

                    for i in range(min(3, results['count'])):
                        face_id = results['ids'][i]
                        distance = results['distances'][i] if i < len(results['distances']) else 'N/A'
                        similarity = 1 - distance if isinstance(distance, float) else 'N/A'
                        print(f"  {i+1}. {face_id} (similarity: {similarity})")
                else:
                    print(f"Error in image search: {search_results['error']}")

    print("\n✅ Face system test completed!")

if __name__ == "__main__":
    test_face_system()