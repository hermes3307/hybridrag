#!/bin/bash

echo "🔍 Face Similarity Search Test GUI Launcher"
echo "==========================================="

# Check if we're in the right directory
if [ ! -f "101.test_searchgui.py" ]; then
    echo "❌ Error: 101.test_searchgui.py not found in current directory"
    echo "   Please run this script from the faces directory"
    exit 1
fi

# Check required files
echo "🔍 Checking required files..."
required_files=(
    "101.test_searchgui.py"
    "face_database.py"
    "face_collector.py"
    "setup_chroma.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ Found: $file"
    else
        echo "❌ Missing: $file"
        missing_files=true
    fi
done

if [ "$missing_files" = true ]; then
    echo ""
    echo "❌ Some required files are missing. Please ensure all files are present."
    exit 1
fi

# Check if faces directory exists
if [ ! -d "faces" ]; then
    echo "📁 Creating faces directory..."
    mkdir -p faces
fi

# Check ChromaDB status
echo ""
echo "🗄️  Checking ChromaDB status..."
if [ -d "chroma_db" ]; then
    db_size=$(du -sh chroma_db | cut -f1)
    echo "✅ ChromaDB found ($db_size)"

    # Get basic database info
    echo "📊 Database statistics:"
    python3 -c "
try:
    from face_database import FaceDatabase
    db = FaceDatabase()
    stats = db.get_database_stats()
    print(f'   - Total faces: {stats.get(\"total_faces\", 0):,}')
    print(f'   - Collection: {stats.get(\"collection_name\", \"unknown\")}')
except Exception as e:
    print(f'   - Error accessing database: {e}')
" 2>/dev/null
else
    echo "⚠️  ChromaDB not found. You may need to run embedding first."
    echo "   Tip: Run './100.embedintoVectorgui.sh' to create embeddings"
fi

# Check available disk space
echo ""
echo "💽 Disk space check:"
available_space=$(df -h . | awk 'NR==2 {print $4}')
echo "   Available space: $available_space"

# Check Python dependencies
echo ""
echo "🐍 Checking Python dependencies..."
dependencies=(
    "tkinter"
    "PIL"
    "requests"
    "chromadb"
    "numpy"
)

python3 -c "
import sys
deps = ['tkinter', 'PIL', 'requests', 'chromadb', 'numpy']
missing = []
for dep in deps:
    try:
        if dep == 'PIL':
            import PIL
        else:
            __import__(dep)
        print(f'✅ {dep} available')
    except ImportError:
        print(f'❌ {dep} missing')
        missing.append(dep)

if missing:
    print()
    print('Missing dependencies. Install with:')
    for dep in missing:
        if dep == 'PIL':
            print('   pip install Pillow')
        else:
            print(f'   pip install {dep}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Missing Python dependencies. Please install them first."
    exit 1
fi

# Create test_images directory if it doesn't exist
if [ ! -d "test_images" ]; then
    echo ""
    echo "📁 Creating test_images directory..."
    mkdir -p test_images
fi

echo ""
echo "🚀 Starting Face Similarity Search Test GUI..."
echo "   - Download random test images from ThisPersonDoesNotExist"
echo "   - Select query images and find similar faces"
echo "   - View original files and search results"
echo "   - Export search results"
echo ""

# Check if display is available (for GUI)
if [ -z "$DISPLAY" ] && [ "$TERM_PROGRAM" != "vscode" ]; then
    echo "⚠️  Warning: No display detected. GUI may not work in headless environment."
    echo "   Make sure you're running this on a system with GUI support."
fi

echo "🎯 Launching GUI application..."
echo ""

# Launch the GUI
python3 101.test_searchgui.py

# Check exit code
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ Face Similarity Search Test GUI session ended successfully"
else
    echo "❌ Face Similarity Search Test GUI ended with error (exit code: $exit_code)"
fi

echo ""
echo "💡 Tips:"
echo "   • Use the Download tab to get random test images"
echo "   • Try the Similarity Search tab to find similar faces"
echo "   • Double-click images to view them in full size"
echo "   • Export results to JSON for further analysis"
echo ""
echo "➡️  Related tools:"
echo "   • Run './100.embedintoVectorgui.sh' to add more faces to database"
echo "   • Run './99.downbackground.gui.sh' to download more face images"
echo "   • Run './5_inspect_database.sh' to analyze the database"

exit $exit_code