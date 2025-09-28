#!/bin/bash

echo "🎭 Step 3: Face Data Collection from ThisPersonDoesNotExist.com"
echo "==============================================================="

echo "📥 Downloading synthetic faces and extracting features..."
echo "⏱️  This will take a few minutes (respectful rate limiting)..."

# Clean up any existing face data to start fresh
if [ -f "face_data.json" ]; then
    echo "🗑️  Removing existing face_data.json for fresh start..."
    rm face_data.json
fi

# Remove existing faces directory for clean collection
if [ -d "faces" ]; then
    echo "🗑️  Removing existing faces directory for fresh start..."
    rm -rf faces
fi

echo ""
echo "🚀 Starting face collection..."
python3 face_collector.py

echo ""
echo "📁 Face collection results:"
if [ -d "faces" ]; then
    echo "✅ Faces directory created:"
    ls -la faces/ | head -10
    echo ""
    echo "📊 Total faces collected: $(ls faces/*.jpg 2>/dev/null | wc -l)"
    echo "📏 Faces directory size: $(du -sh faces/)"
else
    echo "❌ Faces directory not found!"
fi

echo ""
if [ -f "face_data.json" ]; then
    echo "✅ Face data processed and saved:"
    echo "📏 Face data file size: $(du -sh face_data.json)"
    echo "🔢 Face records: $(python3 -c "import json; data=json.load(open('face_data.json')); print(len(data))")"
else
    echo "❌ Face data file not found!"
fi

echo ""
echo "✅ Step 3 completed!"
echo "📋 What was accomplished:"
echo "   • Downloaded synthetic faces from ThisPersonDoesNotExist.com"
echo "   • Extracted 143-dimensional embeddings for each face"
echo "   • Analyzed age groups, skin tones, and image quality"
echo "   • Saved processed data to face_data.json"
echo ""
echo "➡️  Next: Run './4_embed_to_chromadb.sh' to store embeddings in ChromaDB"