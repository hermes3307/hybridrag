#!/bin/bash

echo "💾 Step 4: Embedding Face Data into ChromaDB"
echo "============================================="

echo "🔄 Scanning and embedding face images from ./faces directory..."

echo "🚀 Running embedding process with image scanning..."
python3 4_embed_faces.py --faces-dir ./faces --batch-size 50 --max-workers 4 --clear

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
