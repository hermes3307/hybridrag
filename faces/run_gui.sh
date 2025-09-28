#!/bin/bash

echo "🎨 Launching Face Pipeline GUI..."
echo "================================="

# Check if we have tkinter available
python3 -c "import tkinter" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Tkinter is available"
    echo "🚀 Starting GUI application..."
    python3 face_pipeline_gui.py
else
    echo "❌ Tkinter not available"
    echo "📦 Installing tkinter..."
    # Try to install tkinter (varies by system)
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install python3-tk
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install tkinter
    elif command -v brew >/dev/null 2>&1; then
        # On macOS, tkinter should come with Python
        echo "⚠️  Tkinter should be included with Python on macOS"
        echo "Try running: python3 face_pipeline_gui.py"
    else
        echo "⚠️  Please install tkinter for your system"
        echo "Then run: python3 face_pipeline_gui.py"
    fi
fi