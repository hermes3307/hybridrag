#!/bin/bash
# Launch script for the unified Gradio face processing application

echo "🎭 Starting Unified Face Processing Application..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Checking dependencies..."
pip install --upgrade pip > /dev/null 2>&1

if ! python3 -c "import gradio" &> /dev/null; then
    echo "📥 Installing Gradio and dependencies..."
    pip install -r requirements.txt
else
    echo "✅ Dependencies already installed"
fi

# Check if database is configured
if [ ! -f ".env" ] && [ ! -f "system_config.json" ]; then
    echo ""
    echo "⚠️  No configuration found!"
    echo "Please configure the database in the Configuration tab when the app starts."
    echo ""
fi

# Launch the application
echo ""
echo "🚀 Launching application on http://localhost:7860"
echo "Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 app.py
