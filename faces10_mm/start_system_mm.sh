#!/bin/bash
# Face Processing System Launcher - Multi-Model Edition
#
# Launches the Face Processing GUI Application with Multi-Model Support
#
# This application provides:
#   - Face image downloading from AI services
#   - Camera capture support
#   - Vector embedding generation with MULTIPLE models simultaneously
#   - Multi-model similarity search with model selection
#   - Cross-model comparison and analysis
#   - Metadata filtering and comparison
#   - Real-time statistics and monitoring
#
# Features:
#   - Supports multiple embedding models: FaceNet, ArcFace, VGGFace2, InsightFace, Statistical
#   - Each face can have embeddings from multiple models stored simultaneously
#   - Search across models or compare results between models
#   - Database uses multi-column vector storage (schema.sql)

echo "============================================"
echo "  Face Processing System"
echo "  Multi-Model Support Edition"
echo "============================================"
echo ""

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set multi-model mode
export SYSTEM_MODE="multimodel"
export USE_MULTIMODEL_SCHEMA="true"

# Display current configuration
MODELS="${EMBEDDING_MODELS:-facenet,arcface}"
DEFAULT_MODEL="${DEFAULT_SEARCH_MODEL:-facenet}"

echo "Configuration:"
echo "  System Mode: Multi-Model"
echo "  Embedding Models: $MODELS"
echo "  Default Search Model: $DEFAULT_MODEL"
echo "  Database: ${POSTGRES_DB:-vector_db}"
echo "  Table: faces (multi-column schema)"
echo ""

# Check PostgreSQL connection
echo "Checking database connection..."
if ! psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -c "SELECT 1" > /dev/null 2>&1; then
    echo "⚠️  Warning: Cannot connect to PostgreSQL database"
    echo "   Please ensure PostgreSQL is running and credentials are correct"
    echo ""
else
    echo "✓ Database connection successful"

    # Check if multi-model schema is deployed
    TABLE_CHECK=$(psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -tAc "SELECT column_name FROM information_schema.columns WHERE table_name='faces' AND column_name LIKE 'embedding_%' LIMIT 1" 2>/dev/null)

    if [ -z "$TABLE_CHECK" ]; then
        echo "⚠️  Warning: Multi-model schema not detected"
        echo "   Please run: psql -U postgres -d vector_db -f schema.sql"
        echo ""
    else
        echo "✓ Multi-model schema detected"

        # Show available model columns
        MODEL_COLS=$(psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -tAc "SELECT column_name FROM information_schema.columns WHERE table_name='faces' AND column_name LIKE 'embedding_%' ORDER BY column_name" 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
        echo "   Available embedding columns: $MODEL_COLS"
    fi
    echo ""
fi

echo "Starting application..."
echo ""
echo "Multi-Model Features:"
echo "  • Download AI-generated faces"
echo "  • Capture from camera"
echo "  • Generate embeddings with multiple models"
echo "  • Search with any available model"
echo "  • Compare results across models"
echo "  • Filter by demographics and metadata"
echo "  • View per-model statistics"
echo ""
echo "Initializing..."
echo ""

# Launch the GUI with multi-model flag
python3 faces.py --mode multimodel

