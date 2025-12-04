#!/bin/bash
# Setup script for OCR Receipt Processing System

set -e

echo "=================================================="
echo "OCR Receipt Processing System - Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing Python dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

# Check for Tesseract
echo ""
echo "Checking for Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    tesseract_version=$(tesseract --version 2>&1 | head -n1)
    echo "✓ Tesseract found: $tesseract_version"
else
    echo "⚠ Tesseract not found (optional)"
    echo "  To install:"
    echo "    Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "    macOS: brew install tesseract"
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p temp output
echo "✓ Directories created"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo "  You can customize it for your needs"
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x test_receipt.py examples.sh
echo "✓ Scripts are executable"

# Summary
echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment (if not already):"
echo "   source venv/bin/activate"
echo ""
echo "2. Test OCR (no LLM, fast):"
echo "   python test_receipt.py your_receipt.jpg paddleocr ocr"
echo ""
echo "3. Test full pipeline (with LLM):"
echo "   python test_receipt.py your_receipt.jpg"
echo ""
echo "4. Start API server:"
echo "   python api.py"
echo ""
echo "5. Read the Quick Start guide:"
echo "   cat QUICKSTART.md"
echo ""
echo "=================================================="
echo ""
echo "Note: First run will download the Qwen3 model"
echo "      (~4GB for 7B model with 4-bit quantization)"
echo "=================================================="
