#!/bin/bash

echo "🔬 Step 5: Detailed Database and Vector Analysis"
echo "================================================"

echo "📊 Complete ChromaDB database information:"
python3 run_chroma_info.py

echo ""
echo "🔍 Detailed vector database analysis..."

# Create detailed inspection script
cat > temp_inspect_db.py << 'EOF'
import chromadb
import numpy as np
from face_database import FaceDatabase

def main():
    print("🔬 DETAILED VECTOR DATABASE INSPECTION")
    print("="*60)

    # Connect to database
    face_db = FaceDatabase()

    # Get collection details
    collection = face_db.collection
    total_count = collection.count()

    print(f"📁 Collection: {collection.name}")
    print(f"📄 Total vectors: {total_count:,}")
    print(f"🏷️  Metadata: {collection.metadata}")

    if total_count > 0:
        # Get sample data
        sample = collection.get(limit=min(5, total_count), include=["embeddings", "metadatas", "documents"])

        if sample["embeddings"]:
            embeddings = np.array(sample["embeddings"])

            print(f"\n🔢 VECTOR ANALYSIS:")
            print(f"   Vector dimensions: {embeddings.shape[1]}")
            print(f"   Vector data type: {embeddings.dtype}")
            print(f"   Memory per vector: {embeddings.nbytes // len(embeddings):,} bytes")
            print(f"   Total embedding memory: {embeddings.nbytes / (1024*1024):.2f} MB")

            # Statistical analysis
            print(f"\n📈 VECTOR STATISTICS:")
            print(f"   Min value: {np.min(embeddings):.6f}")
            print(f"   Max value: {np.max(embeddings):.6f}")
            print(f"   Mean value: {np.mean(embeddings):.6f}")
            print(f"   Standard deviation: {np.std(embeddings):.6f}")

            # Vector norms
            norms = np.linalg.norm(embeddings, axis=1)
            print(f"   Vector norms range: {np.min(norms):.6f} - {np.max(norms):.6f}")
            print(f"   Average norm: {np.mean(norms):.6f}")

        print(f"\n📋 SAMPLE METADATA:")
        if sample["metadatas"]:
            for i, metadata in enumerate(sample["metadatas"][:3]):
                print(f"   Sample {i+1}:")
                for key, value in metadata.items():
                    if key in ['estimated_age_group', 'estimated_skin_tone', 'image_quality']:
                        print(f"      {key}: {value}")

        print(f"\n📝 SAMPLE DOCUMENTS:")
        if sample["documents"]:
            for i, doc in enumerate(sample["documents"][:2]):
                print(f"   Doc {i+1}: {doc[:100]}...")

    # Analyze all collections
    client = chromadb.PersistentClient(path="./chroma_db")
    all_collections = client.list_collections()

    print(f"\n🗂️  ALL COLLECTIONS SUMMARY:")
    total_docs = 0
    for col in all_collections:
        count = col.count()
        total_docs += count
        print(f"   📁 {col.name}: {count:,} documents")

    print(f"\n📊 DATABASE TOTALS:")
    print(f"   Collections: {len(all_collections)}")
    print(f"   Total documents: {total_docs:,}")

if __name__ == "__main__":
    main()
EOF

python3 temp_inspect_db.py

# Clean up
rm temp_inspect_db.py

echo ""
echo "💽 Storage and file system analysis:"
echo "📏 Database directory contents:"
ls -la chroma_db/

echo ""
echo "📊 Storage usage breakdown:"
du -sh chroma_db/* 2>/dev/null | sort -hr

echo ""
echo "🗃️  Face images storage:"
if [ -d "faces" ]; then
    echo "   Face count: $(ls faces/*.jpg 2>/dev/null | wc -l)"
    echo "   Total size: $(du -sh faces/)"
    echo "   Average per face: $(python3 -c "import os; files=os.listdir('faces'); total=sum(os.path.getsize(f'faces/{f}') for f in files if f.endswith('.jpg')); print(f'{total//len(files)//1024}KB')" 2>/dev/null)"
fi

echo ""
echo "✅ Step 5 completed!"
echo "📋 What was analyzed:"
echo "   • Vector dimensions and data types"
echo "   • Statistical analysis of embeddings"
echo "   • Memory usage and storage optimization"
echo "   • Collection metadata and document samples"
echo "   • Complete database structure"
echo ""
echo "➡️  Next: Run './6_test_search.sh' to test semantic search with new faces"