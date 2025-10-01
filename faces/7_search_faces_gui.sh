#!/bin/bash

echo "🔍 Face Search GUI Launcher"
echo "==========================="
echo ""

# Check if required files exist
required_files=("7_search_faces_gui.py" "face_database.py" "face_collector.py")

echo "🔍 Checking required files..."
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: $file not found!"
        echo "   Please ensure all required files are in the current directory"
        exit 1
    else
        echo "✅ Found: $file"
    fi
done

# Check if faces directory exists
if [ ! -d "./faces" ]; then
    echo "⚠️  Warning: ./faces directory not found"
    face_count=0
else
    face_count=$(find ./faces -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
    face_size=$(du -sh ./faces 2>/dev/null | cut -f1)
    echo "📂 Found faces directory with $face_count files ($face_size)"
fi

# Check ChromaDB status
echo ""
echo "🗄️  Checking ChromaDB status..."
if [ -d "./chroma_db" ]; then
    db_size=$(du -sh ./chroma_db 2>/dev/null | cut -f1)
    echo "✅ ChromaDB found ($db_size)"

    # Try to get database info
    if [ -f "5_inspect_database.py" ]; then
        echo "📊 Current database statistics:"
        python3 5_inspect_database.py 2>/dev/null | head -10
    fi
else
    echo "❌ ChromaDB not found - please run embedding process first"
    echo "   Run './4_embed_faces_gui.sh' to create the database"
    exit 1
fi

echo ""
echo "🚀 Starting Face Search GUI..."
echo "   - Database size: $(du -sh ./chroma_db 2>/dev/null | cut -f1)"
echo "   - Available disk space: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# Check Python dependencies
echo "🐍 Checking Python dependencies..."
python3 -c "
import sys
try:
    import tkinter
    print('✅ tkinter available')
except ImportError:
    print('❌ tkinter not available')
    sys.exit(1)

try:
    import numpy
    print('✅ numpy available')
except ImportError:
    print('❌ numpy not available - install with: pip install numpy')
    sys.exit(1)

try:
    import PIL
    print('✅ PIL available')
except ImportError:
    print('❌ PIL not available - install with: pip install Pillow')
    sys.exit(1)

try:
    import chromadb
    print('✅ chromadb available')
except ImportError:
    print('❌ chromadb not available - install with: pip install chromadb')
    sys.exit(1)

try:
    import requests
    print('✅ requests available')
except ImportError:
    print('⚠️  requests not available - random face download will not work')
    print('   Install with: pip install requests')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Missing required Python dependencies!"
    echo "   Please install missing packages and try again"
    exit 1
fi

echo ""
echo "✅ All dependencies satisfied"
echo "🎯 Launching Face Search GUI..."
echo ""

# Launch the GUI
python3 7_search_faces_gui.py

# Show completion message
echo ""
echo "🏁 Face Search GUI session ended"
echo ""

echo "💡 Tips:"
echo "   • Use semantic search to find similar faces"
echo "   • Apply metadata filters to narrow results"
echo "   • Try combined search for best results"
echo "   • Export results to JSON for further analysis"
echo ""
