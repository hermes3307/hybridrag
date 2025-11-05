#!/bin/bash
################################################################################
# Quick Embedding Script - Multi-Model Support
################################################################################
#
# This script allows you to choose single or multiple embedding models
#
# Usage:
#   ./run_embedding.sh
#
################################################################################

cd "$(dirname "$0")"

echo "================================================================================"
echo "🚀 EMBEDDING MANAGER - MULTI-MODEL SUPPORT"
echo "================================================================================"
echo ""

# Get current embedding models from environment or use default
CURRENT_MODELS="${EMBEDDING_MODELS:-facenet,arcface}"

echo "📦 EMBEDDING MODEL SELECTION"
echo "--------------------------------------------------------------------------------"
echo "Current models: $CURRENT_MODELS"
echo ""
echo "Available models:"
echo "  1) statistical  - Basic statistical features (always available, fast)"
echo "  2) facenet      - FaceNet InceptionResnetV1 (requires facenet-pytorch)"
echo "  3) arcface      - ArcFace model (state-of-the-art accuracy)"
echo "  4) vggface2     - VGGFace2 deep CNN model"
echo "  5) insightface  - InsightFace model (very accurate)"
echo "  6) all          - Process with ALL available models"
echo "  7) multi        - Select multiple models (comma-separated)"
echo ""
echo "Enter your choice (1-7), or press Enter to use current models [$CURRENT_MODELS]:"
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

# Map choice to model name(s)
case "$choice" in
    1)
        SELECTED_MODELS="statistical"
        ;;
    2)
        SELECTED_MODELS="facenet"
        ;;
    3)
        SELECTED_MODELS="arcface"
        ;;
    4)
        SELECTED_MODELS="vggface2"
        ;;
    5)
        SELECTED_MODELS="insightface"
        ;;
    6)
        SELECTED_MODELS="statistical,facenet,arcface,vggface2,insightface"
        echo "🎯 Selected: ALL models"
        ;;
    7)
        echo ""
        echo "Enter models separated by commas (e.g., facenet,arcface,vggface2):"
        echo "Available: statistical, facenet, arcface, vggface2, insightface"
        read -r SELECTED_MODELS
        if [[ -z "$SELECTED_MODELS" ]]; then
            echo "⚠️  No models entered. Using current models: $CURRENT_MODELS"
            SELECTED_MODELS="$CURRENT_MODELS"
        fi
        ;;
    "")
        SELECTED_MODELS="$CURRENT_MODELS"
        echo "Using current models: $SELECTED_MODELS"
        ;;
    *)
        echo "❌ Invalid choice. Using current models: $CURRENT_MODELS"
        SELECTED_MODELS="$CURRENT_MODELS"
        ;;
esac

echo ""
echo "================================================================================"
echo "🎯 Configuration:"
echo "  Models: $SELECTED_MODELS"
echo "  Workers: $WORKERS"
echo "================================================================================"
echo ""

# Convert comma-separated models to array
IFS=',' read -ra MODEL_ARRAY <<< "$SELECTED_MODELS"

# Process each model
TOTAL_MODELS=${#MODEL_ARRAY[@]}
CURRENT=0
FAILED=0

for model in "${MODEL_ARRAY[@]}"; do
    # Trim whitespace
    model=$(echo "$model" | xargs)
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "================================================================================"
    echo "📊 Processing model $CURRENT/$TOTAL_MODELS: $model"
    echo "================================================================================"
    echo ""

    # Run the CLI in auto-embed mode with selected model and workers
    python3 embedding_manager_cli.py --model "$model" --workers "$WORKERS" --auto-embed

    # Check exit status
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Model '$model' completed successfully!"
    else
        echo ""
        echo "❌ Model '$model' failed!"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "================================================================================"
echo "📈 FINAL SUMMARY"
echo "================================================================================"
echo "  Total models: $TOTAL_MODELS"
echo "  Successful: $((TOTAL_MODELS - FAILED))"
echo "  Failed: $FAILED"
echo "================================================================================"

if [ $FAILED -eq 0 ]; then
    echo "✅ All models completed successfully!"
    exit 0
else
    echo "⚠️  Some models failed. Check the errors above."
    exit 1
fi
