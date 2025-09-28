#!/bin/bash

echo "💾 Step 4: Embedding Face Data into ChromaDB"
echo "============================================="

echo "🔄 Loading face data and storing embeddings in ChromaDB..."

# Create a simple script to load face data into ChromaDB
cat > temp_embed_faces.py << 'EOF'
import json
import os
from face_collector import FaceData
from face_database import FaceDatabase

def main():
    print("📂 Loading face data from face_data.json...")

    if not os.path.exists("face_data.json"):
        print("❌ face_data.json not found! Run step 3 first.")
        return

    # Load face data
    with open("face_data.json", 'r') as f:
        face_data_dicts = json.load(f)

    face_data_list = [FaceData(**data) for data in face_data_dicts]
    print(f"✅ Loaded {len(face_data_list)} face records")

    # Initialize database (clear existing faces collection)
    print("🗑️  Clearing existing faces collection...")
    face_db = FaceDatabase()

    try:
        face_db.client.delete_collection("faces")
        face_db._initialize_db()  # Recreate collection
        print("✅ Collection cleared and recreated")
    except Exception as e:
        print(f"ℹ️  Collection creation: {e}")

    # Add faces to database
    print("📥 Adding face embeddings to ChromaDB...")
    added_count = face_db.add_faces(face_data_list)
    print(f"✅ Added {added_count} face embeddings to database")

    # Get stats
    stats = face_db.get_database_stats()
    print(f"\n📊 Database statistics:")
    print(f"   Total faces: {stats['total_faces']}")
    print(f"   Age groups: {stats.get('age_group_distribution', {})}")
    print(f"   Skin tones: {stats.get('skin_tone_distribution', {})}")
    print(f"   Qualities: {stats.get('quality_distribution', {})}")

if __name__ == "__main__":
    main()
EOF

echo "🚀 Running embedding process..."
python3 temp_embed_faces.py

# Clean up temporary script
rm temp_embed_faces.py

echo ""
echo "📊 Database status after embedding:"
python3 run_chroma_info.py

echo ""
echo "📏 Storage usage:"
du -sh chroma_db/ faces/ face_data.json 2>/dev/null

echo ""
echo "✅ Step 4 completed!"
echo "📋 What was accomplished:"
echo "   • Loaded face data from JSON file"
echo "   • Created faces collection in ChromaDB"
echo "   • Stored 143-dimensional embeddings with metadata"
echo "   • Database now ready for semantic search"
echo ""
echo "➡️  Next: Run './5_inspect_database.sh' to examine the vector database in detail"