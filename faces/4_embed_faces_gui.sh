#!/bin/bash

echo "🔮 Vector Embedding GUI Launcher"
echo "================================="
echo ""

# Check if required files exist
required_files=("4_embed_faces.py" "4_embed_faces_gui.py" "face_database.py")

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
    echo "   Creating faces directory..."
    mkdir -p ./faces
    echo "✅ Created: ./faces directory"
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
    if [ -f "2_database_info.py" ]; then
        echo "📊 Current database statistics:"
        python3 2_database_info.py 2>/dev/null | head -10
    fi
else
    echo "⚠️  ChromaDB not found - will be created during embedding process"
fi

echo ""
echo "🚀 Starting Vector Embedding GUI..."
echo "   - Total face files ready for processing: $(find ./faces -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)"
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
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Missing required Python dependencies!"
    echo "   Please install missing packages and try again"
    exit 1
fi

echo ""
echo "✅ All dependencies satisfied"
echo "🎯 Launching Vector Embedding GUI..."
echo ""

# Launch the GUI
python3 4_embed_faces_gui.py

# Show completion message
echo ""
echo "🏁 Vector Embedding GUI session ended"
echo ""

# Show final statistics if database exists
if [ -f "2_database_info.py" ] && [ -d "./chroma_db" ]; then
    echo "📊 Final database statistics:"
    python3 2_database_info.py 2>/dev/null
    echo ""
fi

echo "💡 Tips:"
echo "   • Use the GUI to process face images into vector embeddings"
echo "   • Monitor progress and file counts in real-time"
echo "   • Check 'Clear Existing Embeddings' to start fresh"
echo "   • Adjust batch size and workers for optimal performance"
echo ""
echo "➡️  Next steps:"
echo "   • Run './5_inspect_database.sh' to examine the database"
echo "   • Run './6_test_search.sh' to test similarity search"
echo ""
