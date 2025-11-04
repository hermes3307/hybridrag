#!/bin/bash
# Face Processing System Launcher - Multi-Model Edition
#
# Launches the Face Processing GUI Application
#
# This application provides:
#   - Face image downloading from AI services
#   - Camera capture support
#   - Vector embedding generation with MULTIPLE models
#   - Multi-model similarity search with model selection
#   - Metadata filtering and comparison
#   - Real-time statistics and monitoring

echo "============================================"
echo "  Face Processing System"
echo "  Multi-Model Support Edition"
echo "============================================"
echo ""

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Display current configuration
MODELS="${EMBEDDING_MODELS:-facenet,arcface}"
DEFAULT_MODEL="${DEFAULT_SEARCH_MODEL:-facenet}"

echo "Configuration:"
echo "  Embedding Models: $MODELS"
echo "  Default Search Model: $DEFAULT_MODEL"
echo ""
echo "Starting application..."
echo ""
echo "Features:"
echo "  • Download AI-generated faces"
echo "  • Capture from camera"
echo "  • Multi-model embedding support"
echo "  • Search with model selection"
echo "  • Compare results across models"
echo "  • Filter by demographics"
echo ""
echo "Initializing..."
echo ""

python3 faces.py
