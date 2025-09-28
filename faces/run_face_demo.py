#!/usr/bin/env python3
"""
Face System Demo - Quick demonstration of the face collection and search system
"""

from face_collector import FaceCollector, process_faces
from face_database import FaceDatabase
import os

def main():
    print("🎭 Face Collection & Search Demo")
    print("="*50)

    # Step 1: Collect faces if needed
    faces_dir = "./faces"
    if not os.path.exists(faces_dir) or len(os.listdir(faces_dir)) < 5:
        print("📥 Collecting faces from ThisPersonDoesNotExist.com...")
        collector = FaceCollector(delay=1.5)
        face_files = collector.download_faces_batch(count=8, max_workers=2)

        if face_files:
            print("🔍 Processing faces and extracting features...")
            processed_faces = process_faces(face_files)

            # Save processed data
            from face_collector import save_face_data
            save_face_data(processed_faces)
            print(f"✅ Processed {len(processed_faces)} faces")
        else:
            print("❌ No faces collected")
            return

    # Step 2: Initialize database and add faces
    print("\n💾 Setting up vector database...")
    face_db = FaceDatabase()

    # Load existing face data
    if os.path.exists("face_data.json"):
        import json
        from face_collector import FaceData

        with open("face_data.json", 'r') as f:
            face_data_dicts = json.load(f)

        face_data_list = [FaceData(**data) for data in face_data_dicts]
        face_db.add_faces(face_data_list)
        print(f"✅ Added {len(face_data_list)} faces to database")

    # Step 3: Demonstrate searches
    print("\n🔍 Demonstrating search capabilities...")

    # Get database stats
    stats = face_db.get_database_stats()
    print(f"📊 Database: {stats['total_faces']} faces")

    # Feature-based search
    adult_results = face_db.search_by_features({"estimated_age_group": "adult"}, n_results=3)
    print(f"🎯 Adults found: {adult_results['count']}")

    light_results = face_db.search_by_features({"estimated_skin_tone": "light"}, n_results=3)
    print(f"🎨 Light skin tone: {light_results['count']}")

    # Similarity search
    if stats['total_faces'] > 0:
        all_faces = face_db.collection.get(limit=1, include=["embeddings"])
        if len(all_faces.get("embeddings", [])) > 0:
            first_embedding = all_faces["embeddings"][0]
            similar = face_db.search_similar_faces(first_embedding, n_results=3)
            print(f"🔗 Similar faces: {similar['count']}")

    print("\n✨ Demo completed!")
    print("\nTo explore further:")
    print("• Run 'python3 face_database.py' for interactive interface")
    print("• Run 'python3 test_face_system.py' for detailed testing")
    print("• Check README_face_system.md for full documentation")

if __name__ == "__main__":
    main()