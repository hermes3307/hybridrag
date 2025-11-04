#!/bin/bash
################################################################################
# Delete Embeddings by Model
################################################################################
#
# This script deletes embeddings from the database by embedding model
#
# Usage:
#   ./delete_embeddings_by_model.sh <model_name>              # Dry-run
#   ./delete_embeddings_by_model.sh <model_name> --confirm    # Actually delete
#
# Examples:
#   ./delete_embeddings_by_model.sh statistical              # Show what would be deleted
#   ./delete_embeddings_by_model.sh statistical --confirm    # Delete statistical embeddings
#   ./delete_embeddings_by_model.sh facenet --confirm        # Delete facenet embeddings
#
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Database configuration
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="vector_db"
DB_USER="postgres"
DB_PASS="postgres"

# Parse arguments
MODEL_NAME="$1"
DRY_RUN=true

if [ "$2" = "--confirm" ] || [ "$2" = "-c" ]; then
    DRY_RUN=false
fi

# Check if model name provided
if [ -z "$MODEL_NAME" ]; then
    echo -e "${RED}Error: Model name is required${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <model_name>              # Dry-run"
    echo "  $0 <model_name> --confirm    # Actually delete"
    echo ""
    echo "Examples:"
    echo "  $0 statistical"
    echo "  $0 statistical --confirm"
    echo "  $0 facenet --confirm"
    echo ""
    echo "Available models:"
    PGPASSWORD=$DB_PASS psql -h $DB_HOST -U $DB_USER -d $DB_NAME -t -c \
        "SELECT DISTINCT embedding_model FROM faces ORDER BY embedding_model;" 2>/dev/null | sed 's/^/ - /'
    exit 1
fi

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}🗑️  DELETE EMBEDDINGS BY MODEL${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Check database connection
echo -e "${YELLOW}📊 Checking database...${NC}"

# Get count for this model
COUNT=$(PGPASSWORD=$DB_PASS psql -h $DB_HOST -U $DB_USER -d $DB_NAME -t -c \
    "SELECT COUNT(*) FROM faces WHERE embedding_model = '$MODEL_NAME';" 2>/dev/null | xargs)

if [ -z "$COUNT" ] || [ "$COUNT" = "" ]; then
    echo -e "${RED}❌ Error: Could not connect to database${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}📊 STATISTICS${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Show all models
echo "Current database status:"
PGPASSWORD=$DB_PASS psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
    "SELECT embedding_model, COUNT(*) as count FROM faces GROUP BY embedding_model ORDER BY count DESC;" 2>/dev/null

echo ""

if [ "$COUNT" -eq 0 ]; then
    echo -e "${GREEN}✅ No embeddings found with model: ${YELLOW}$MODEL_NAME${NC}"
    echo ""
    echo "Nothing to delete."
    exit 0
fi

echo -e "Embeddings with model '${YELLOW}$MODEL_NAME${NC}': ${RED}$COUNT${NC}"
echo ""

# Show sample records
echo "Sample records (first 5):"
PGPASSWORD=$DB_PASS psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
    "SELECT face_id, embedding_model, timestamp FROM faces WHERE embedding_model = '$MODEL_NAME' LIMIT 5;" 2>/dev/null

echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}================================================================================${NC}"
    echo -e "${YELLOW}⚠️  DRY-RUN MODE - NO RECORDS DELETED${NC}"
    echo -e "${YELLOW}================================================================================${NC}"
    echo ""
    echo "This was a dry-run. No records were actually deleted."
    echo ""
    echo "To actually delete these ${RED}$COUNT${NC} records, run:"
    echo -e "  ${GREEN}$0 $MODEL_NAME --confirm${NC}"
    echo ""
else
    echo -e "${RED}================================================================================${NC}"
    echo -e "${RED}⚠️  DELETION MODE${NC}"
    echo -e "${RED}================================================================================${NC}"
    echo ""
    echo -e "${RED}WARNING: You are about to delete $COUNT embeddings!${NC}"
    echo -e "${RED}Model: $MODEL_NAME${NC}"
    echo -e "${RED}This action CANNOT be undone!${NC}"
    echo ""
    read -p "Type 'DELETE' to confirm: " CONFIRM

    if [ "$CONFIRM" != "DELETE" ]; then
        echo -e "${BLUE}Deletion cancelled.${NC}"
        exit 0
    fi

    echo ""
    echo -e "${RED}🗑️  Deleting records...${NC}"

    # Execute deletion
    RESULT=$(PGPASSWORD=$DB_PASS psql -h $DB_HOST -U $DB_USER -d $DB_NAME -t -c \
        "DELETE FROM faces WHERE embedding_model = '$MODEL_NAME';" 2>&1)

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully deleted $COUNT records${NC}"
        echo ""

        # Show updated stats
        echo "Updated database status:"
        PGPASSWORD=$DB_PASS psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c \
            "SELECT embedding_model, COUNT(*) as count FROM faces GROUP BY embedding_model ORDER BY count DESC;" 2>/dev/null
        echo ""
    else
        echo -e "${RED}❌ Error during deletion:${NC}"
        echo "$RESULT"
        exit 1
    fi
fi

echo -e "${BLUE}================================================================================${NC}"
