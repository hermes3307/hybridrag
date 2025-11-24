#!/bin/bash

# Qwen3 Web Interface Start Script
# This script starts the web-based interface for the Qwen3 chatbot

echo "Qwen3 Web Interface"
echo "=================="

# Check if virtual environment exists
VENV_PATH="/home/pi/qwen/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH"
    echo "Please create the virtual environment first:"
    echo "python3 -m venv $VENV_PATH"
    echo "source $VENV_PATH/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

echo "Starting web interface..."
echo "Access the interface at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo

# Start the web interface
python3 web_interface.py