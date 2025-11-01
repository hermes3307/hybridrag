#!/bin/bash
################################################################################
# Quick Embedding Script - Interactive model selection
################################################################################
#
# This script allows you to choose the embedding model before running
#
# Usage:
#   ./run_embedding.sh
#
################################################################################

cd "$(dirname "$0")"

echo "================================================================================"
echo "🚀 EMBEDDING MANAGER - INTERACTIVE RUN"
echo "================================================================================"
echo ""

# Get current embedding model from environment or use default
CURRENT_MODEL="${EMBEDDING_MODEL:-statistical}"

echo "📦 EMBEDDING MODEL SELECTION"
echo "--------------------------------------------------------------------------------"
echo "Current model: $CURRENT_MODEL"
echo ""
echo "Available models:"
echo "  1) statistical  - Basic statistical features (always available, fast)"
echo "  2) facenet      - Deep learning model (requires facenet-pytorch)"
echo "  3) arcface      - State-of-the-art (requires insightface)"
echo "  4) deepface     - Multi-purpose deep learning"
echo "  5) vggface2     - Deep CNN model"
echo "  6) openface     - Lightweight deep learning"
echo ""
echo "Enter your choice (1-6), or press Enter to use current model [$CURRENT_MODEL]:"
read -r choice

echo ""
echo "⚙️  PARALLEL PROCESSING"
echo "--------------------------------------------------------------------------------"
echo "How many parallel workers? (1-8, default: 1)"
echo "  1 = Sequential (slower, safer)"
echo "  2-4 = Recommended for most systems"
echo "  5-8 = For powerful systems with many cores"
echo ""
echo "Enter number of workers (or press Enter for 1):"
read -r workers_choice

# Validate and set workers
if [[ -z "$workers_choice" ]]; then
    WORKERS=1
elif [[ "$workers_choice" =~ ^[1-8]$ ]]; then
    WORKERS="$workers_choice"
else
    echo "⚠️  Invalid choice. Using 1 worker (sequential)"
    WORKERS=1
fi

# Map choice to model name
case "$choice" in
    1)
        SELECTED_MODEL="statistical"
        ;;
    2)
        SELECTED_MODEL="facenet"
        ;;
    3)
        SELECTED_MODEL="arcface"
        ;;
    4)
        SELECTED_MODEL="deepface"
        ;;
    5)
        SELECTED_MODEL="vggface2"
        ;;
    6)
        SELECTED_MODEL="openface"
        ;;
    "")
        SELECTED_MODEL="$CURRENT_MODEL"
        echo "Using current model: $SELECTED_MODEL"
        ;;
    *)
        echo "❌ Invalid choice. Using current model: $CURRENT_MODEL"
        SELECTED_MODEL="$CURRENT_MODEL"
        ;;
esac

echo ""
echo "================================================================================"
echo "🎯 Configuration:"
echo "  Model: $SELECTED_MODEL"
echo "  Workers: $WORKERS"
echo "================================================================================"
echo ""
echo "Starting automatic embedding process..."
echo ""

# Run the CLI in auto-embed mode with selected model and workers
python3 embedding_manager_cli.py --model "$SELECTED_MODEL" --workers "$WORKERS" --auto-embed

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ Embedding completed successfully!"
    echo "================================================================================"
    exit 0
else
    echo ""
    echo "================================================================================"
    echo "❌ Embedding failed! Check the errors above."
    echo "================================================================================"
    exit 1
fi
