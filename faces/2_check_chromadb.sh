#!/bin/bash

echo "🔍 Step 2: ChromaDB Verification and Database Info"
echo "=================================================="

echo "📊 Checking ChromaDB installation and current database status..."
python3 run_chroma_info.py

echo ""
echo "🔬 Database file structure:"
if [ -d "./chroma_db" ]; then
    echo "✅ ChromaDB directory exists:"
    ls -la chroma_db/
    echo ""
    echo "📏 Database size:"
    du -sh chroma_db/
else
    echo "❌ ChromaDB directory not found!"
fi

echo ""
echo "✅ Step 2 completed!"
echo "📋 What was verified:"
echo "   • ChromaDB version and installation"
echo "   • Database collections and document counts"
echo "   • Database file structure and size"
echo ""
echo "➡️  Next: Run './3_collect_faces.sh' to start collecting face data"