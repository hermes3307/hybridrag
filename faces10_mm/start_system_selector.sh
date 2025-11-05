#!/bin/bash
# Face Processing System Launcher - Mode Selector
#
# Interactive launcher that allows choosing between:
#   1. Multi-Model Mode: Use multiple embedding models simultaneously
#   2. Legacy Mode: Use single embedding model (backward compatibility)
#   3. Auto Mode: Detect schema and use appropriate mode

echo "============================================"
echo "  Face Processing System Launcher"
echo "============================================"
echo ""

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Function to detect which schema is active
detect_schema() {
    # Check if multi-model schema exists
    MULTIMODEL_CHECK=$(psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -tAc "SELECT COUNT(*) FROM information_schema.columns WHERE table_name='faces' AND column_name LIKE 'embedding_%'" 2>/dev/null)

    # Check if legacy schema exists
    LEGACY_CHECK=$(psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='faces_legacy'" 2>/dev/null)

    if [ "$MULTIMODEL_CHECK" -gt 0 ] 2>/dev/null; then
        echo "multimodel"
    elif [ "$LEGACY_CHECK" -gt 0 ] 2>/dev/null; then
        echo "legacy"
    else
        echo "none"
    fi
}

# Function to show schema info
show_schema_info() {
    local schema_type=$1

    if [ "$schema_type" = "multimodel" ]; then
        echo "  Current Schema: Multi-Model (faces table with multiple embedding columns)"
        MODEL_COLS=$(psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -tAc "SELECT column_name FROM information_schema.columns WHERE table_name='faces' AND column_name LIKE 'embedding_%' ORDER BY column_name" 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
        echo "  Available Models: $MODEL_COLS"

        FACE_COUNT=$(psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -tAc "SELECT COUNT(*) FROM faces" 2>/dev/null)
        echo "  Total Faces: $FACE_COUNT"
    elif [ "$schema_type" = "legacy" ]; then
        echo "  Current Schema: Legacy (faces_legacy table with single embedding column)"

        FACE_COUNT=$(psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -tAc "SELECT COUNT(*) FROM faces_legacy" 2>/dev/null)
        echo "  Total Faces: $FACE_COUNT"

        MODELS=$(psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -tAc "SELECT DISTINCT embedding_model FROM faces_legacy WHERE embedding_model IS NOT NULL" 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
        echo "  Models Used: $MODELS"
    else
        echo "  Current Schema: None detected (database needs initialization)"
    fi
}

# Check database connection
echo "Checking database connection..."
if ! psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -c "SELECT 1" > /dev/null 2>&1; then
    echo "⚠️  Warning: Cannot connect to PostgreSQL database"
    echo "   Host: ${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}"
    echo "   Database: ${POSTGRES_DB:-vector_db}"
    echo "   User: ${POSTGRES_USER:-postgres}"
    echo ""
    echo "Please ensure PostgreSQL is running and credentials are correct"
    exit 1
fi

echo "✓ Database connection successful"
echo ""

# Detect current schema
CURRENT_SCHEMA=$(detect_schema)
show_schema_info "$CURRENT_SCHEMA"
echo ""

# Display mode selection
echo "Select Mode:"
echo "  1) Multi-Model Mode (recommended for new setups)"
echo "     - Use multiple embedding models simultaneously"
echo "     - Store embeddings from FaceNet, ArcFace, VGGFace2, etc."
echo "     - Compare results across different models"
echo "     - Requires: schema.sql"
echo ""
echo "  2) Legacy Mode (backward compatibility)"
echo "     - Use single embedding model at a time"
echo "     - Compatible with older single-model systems"
echo "     - Requires: schema_legacy.sql"
echo ""
echo "  3) Auto Mode (detect from database)"
echo "     - Automatically detect and use appropriate mode"
echo "     - Current detection: $CURRENT_SCHEMA"
echo ""
echo "  4) Setup/Initialize Database"
echo "     - Deploy database schema"
echo "     - Choose between multi-model or legacy schema"
echo ""
echo "  0) Exit"
echo ""

# Get user choice
read -p "Enter choice [0-4]: " choice

case $choice in
    1)
        echo ""
        echo "Starting in Multi-Model Mode..."
        if [ "$CURRENT_SCHEMA" != "multimodel" ]; then
            echo "⚠️  Warning: Multi-model schema not detected"
            read -p "Deploy multi-model schema now? (y/n): " deploy
            if [ "$deploy" = "y" ] || [ "$deploy" = "Y" ]; then
                echo "Deploying schema.sql..."
                psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -f schema.sql
            fi
        fi
        ./start_system_mm.sh
        ;;

    2)
        echo ""
        echo "Starting in Legacy Mode..."
        if [ "$CURRENT_SCHEMA" != "legacy" ]; then
            echo "⚠️  Warning: Legacy schema not detected"
            read -p "Deploy legacy schema now? (y/n): " deploy
            if [ "$deploy" = "y" ] || [ "$deploy" = "Y" ]; then
                echo "Deploying schema_legacy.sql..."
                psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -f schema_legacy.sql
            fi
        fi
        export SYSTEM_MODE="legacy"
        export USE_LEGACY_SCHEMA="true"
        python3 faces.py --mode legacy
        ;;

    3)
        echo ""
        echo "Starting in Auto Mode (detected: $CURRENT_SCHEMA)..."
        if [ "$CURRENT_SCHEMA" = "multimodel" ]; then
            ./start_system_mm.sh
        elif [ "$CURRENT_SCHEMA" = "legacy" ]; then
            export SYSTEM_MODE="legacy"
            export USE_LEGACY_SCHEMA="true"
            python3 faces.py --mode legacy
        else
            echo "No schema detected. Please run setup first (option 4)"
            exit 1
        fi
        ;;

    4)
        echo ""
        echo "Database Setup"
        echo "=============="
        echo ""
        echo "Select schema to deploy:"
        echo "  1) Multi-Model Schema (schema.sql) - recommended"
        echo "  2) Legacy Schema (schema_legacy.sql)"
        echo "  3) Both schemas"
        echo ""
        read -p "Enter choice [1-3]: " schema_choice

        case $schema_choice in
            1)
                echo "Deploying multi-model schema..."
                psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -f schema.sql
                echo "✓ Multi-model schema deployed"
                ;;
            2)
                echo "Deploying legacy schema..."
                psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -f schema_legacy.sql
                echo "✓ Legacy schema deployed"
                ;;
            3)
                echo "Deploying both schemas..."
                psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -f schema.sql
                psql -h "${POSTGRES_HOST:-localhost}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-vector_db}" -f schema_legacy.sql
                echo "✓ Both schemas deployed"
                ;;
            *)
                echo "Invalid choice"
                exit 1
                ;;
        esac

        echo ""
        read -p "Start the application now? (y/n): " start_now
        if [ "$start_now" = "y" ] || [ "$start_now" = "Y" ]; then
            exec "$0"  # Restart this script
        fi
        ;;

    0)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
