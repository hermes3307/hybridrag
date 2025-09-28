#!/bin/bash

echo "🚀 Step 1: ChromaDB Setup and Installation"
echo "=================================================="

echo "📦 Installing ChromaDB and dependencies..."
pip install chromadb numpy pillow requests beautifulsoup4

echo ""
echo "🔧 Running ChromaDB setup script..."
python3 setup_chroma.py

echo ""
echo "✅ Step 1 completed!"
echo "📋 What was accomplished:"
echo "   • ChromaDB installed and configured"
echo "   • Persistent database created in ./chroma_db/"
echo "   • Sample collection created for testing"
echo ""
echo "➡️  Next: Run './2_check_chromadb.sh' to verify the installation"