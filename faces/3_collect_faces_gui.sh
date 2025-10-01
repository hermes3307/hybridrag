#!/bin/bash

echo "🎭 Step 3: Face Data Collection from ThisPersonDoesNotExist.com (GUI)"
echo "======================================================================"

echo "📥 Starting GUI for downloading synthetic faces with metadata extraction..."

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
echo "🚀 Starting face collection GUI with JSON metadata generation..."
echo "   Each face will have:"
echo "   • Image file (.jpg)"
echo "   • Metadata file (.json) with queryable attributes"
echo ""
python3 3_collect_faces_gui.py

echo ""
echo "📁 Face collection results:"
if [ -d "faces" ]; then
    echo "✅ Faces directory created:"
    ls -la faces/ | head -10
    echo ""
    echo "📊 Total images collected: $(ls faces/*.jpg 2>/dev/null | wc -l)"
    echo "📊 Total JSON metadata files: $(ls faces/*.json 2>/dev/null | wc -l)"
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
echo "🔍 Sample JSON metadata structure:"
if [ -n "$(ls faces/*.json 2>/dev/null | head -1)" ]; then
    SAMPLE_JSON=$(ls faces/*.json 2>/dev/null | head -1)
    echo "   File: $(basename "$SAMPLE_JSON")"
    python3 -c "import json; import sys; data=json.load(open('$SAMPLE_JSON')); print('   Queryable attributes:', list(data.get('queryable_attributes', {}).keys()))"
fi

echo ""
echo "✅ Step 3 completed!"
echo "📋 What was accomplished:"
echo "   • Downloaded synthetic faces from ThisPersonDoesNotExist.com"
echo "   • Generated JSON metadata for each image with:"
echo "     - Face features (brightness, hue, saturation, quality)"
echo "     - Queryable attributes (age group, skin tone, image quality)"
echo "     - Download metadata (timestamp, hash, source)"
echo "     - Image properties (dimensions, format, file size)"
echo "   • Extracted embeddings for semantic search"
echo "   • Saved processed data to face_data.json"
echo ""
echo "➡️  Next: Run './4_embed_to_chromadb.sh' to store embeddings in ChromaDB"
