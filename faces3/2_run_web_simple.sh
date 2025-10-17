#!/bin/bash

echo "🎭 Starting Integrated Face Processing Web Interface..."
echo "==========================================="
echo ""

# Check if Gradio is installed
if ! python3 -c "import gradio" 2>/dev/null; then
    echo "⚠️  Gradio not found. Installing..."
    pip3 install gradio
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Check dependencies
echo ""
echo "📦 Checking dependencies..."
python3 -c "
import sys
try:
    import gradio
    print('✅ Gradio installed')
except ImportError:
    print('❌ Gradio missing: pip install gradio')

try:
    import chromadb
    print('✅ ChromaDB installed')
except ImportError:
    print('❌ ChromaDB missing: pip install chromadb')

try:
    import cv2
    print('✅ OpenCV installed')
except ImportError:
    print('⚠️  OpenCV missing: pip install opencv-python (optional for camera)')

try:
    from PIL import Image
    print('✅ Pillow installed')
except ImportError:
    print('❌ Pillow missing: pip install Pillow')
"

echo ""
echo "🚀 Launching web interface..."
echo "📍 Access the interface at: http://localhost:7860"
echo "🌐 Or from network: http://$(hostname -I | awk '{print $1}'):7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the web interface
cd "$(dirname "$0")"
python3 web_interface.py
