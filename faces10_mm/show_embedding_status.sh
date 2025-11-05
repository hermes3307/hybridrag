#!/bin/bash
################################################################################
# Embedding Status Display Script
################################################################################
#
# This script displays comprehensive embedding statistics for all models
#
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
DB_NAME="${POSTGRES_DB:-vector_db}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-postgres}"

# Clear screen
clear

echo ""
echo -e "${CYAN}================================================================================${NC}"
echo -e "${CYAN}                    MULTI-MODEL EMBEDDING STATUS                              ${NC}"
echo -e "${CYAN}================================================================================${NC}"
echo ""

# Get basic counts
STATS=$(PGPASSWORD=$DB_PASSWORD psql -h localhost -U $DB_USER -d $DB_NAME -t -A -F'|' -c "
SELECT
    COUNT(*) as total,
    COUNT(embedding_facenet) as facenet,
    COUNT(embedding_arcface) as arcface,
    COUNT(embedding_vggface2) as vggface2,
    COUNT(embedding_insightface) as insightface,
    COUNT(embedding_statistical) as statistical
FROM faces;
")

# Parse results
IFS='|' read -r TOTAL FACENET ARCFACE VGGFACE2 INSIGHTFACE STATISTICAL <<< "$STATS"

# Calculate percentages
if [ "$TOTAL" -gt 0 ]; then
    FACENET_PCT=$(echo "scale=1; $FACENET * 100 / $TOTAL" | bc)
    ARCFACE_PCT=$(echo "scale=1; $ARCFACE * 100 / $TOTAL" | bc)
    VGGFACE2_PCT=$(echo "scale=1; $VGGFACE2 * 100 / $TOTAL" | bc)
    INSIGHTFACE_PCT=$(echo "scale=1; $INSIGHTFACE * 100 / $TOTAL" | bc)
    STATISTICAL_PCT=$(echo "scale=1; $STATISTICAL * 100 / $TOTAL" | bc)
else
    FACENET_PCT=0
    ARCFACE_PCT=0
    VGGFACE2_PCT=0
    INSIGHTFACE_PCT=0
    STATISTICAL_PCT=0
fi

# Display overview
echo -e "${BLUE}📊 OVERVIEW${NC}"
echo -e "${BLUE}─────────────────────────────────────────────────────────────────────────────${NC}"
printf "%-30s %'10d\n" "Total Faces in Database:" "$TOTAL"
echo ""

# Function to create progress bar
create_progress_bar() {
    local percentage=$1
    local width=40
    local filled=$(echo "($percentage * $width) / 100" | bc)
    local empty=$((width - filled))

    printf "["
    for ((i=0; i<filled; i++)); do printf "█"; done
    for ((i=0; i<empty; i++)); do printf "░"; done
    printf "]"
}

# Function to get status emoji
get_status() {
    local count=$1
    local total=$2
    local pct=$(echo "scale=0; $count * 100 / $total" | bc)

    if [ "$count" -eq 0 ]; then
        echo "❌"
    elif [ "$pct" -eq 100 ]; then
        echo "✅"
    elif [ "$pct" -ge 75 ]; then
        echo "🟢"
    elif [ "$pct" -ge 25 ]; then
        echo "🟡"
    else
        echo "🔴"
    fi
}

# Display each model
echo -e "${GREEN}🎯 EMBEDDING MODELS${NC}"
echo -e "${GREEN}─────────────────────────────────────────────────────────────────────────────${NC}"

# FaceNet
STATUS=$(get_status $FACENET $TOTAL)
echo -e "${YELLOW}FaceNet${NC}"
printf "  %s Count: %'10d / %'d  (%5.1f%%)\n" "$STATUS" "$FACENET" "$TOTAL" "$FACENET_PCT"
printf "  Progress: "
create_progress_bar $FACENET_PCT
echo ""
REMAINING=$((TOTAL - FACENET))
if [ $REMAINING -gt 0 ]; then
    printf "  ${RED}⏳ Remaining: %'d faces${NC}\n" "$REMAINING"
fi
echo ""

# ArcFace
STATUS=$(get_status $ARCFACE $TOTAL)
echo -e "${YELLOW}ArcFace${NC}"
printf "  %s Count: %'10d / %'d  (%5.1f%%)\n" "$STATUS" "$ARCFACE" "$TOTAL" "$ARCFACE_PCT"
printf "  Progress: "
create_progress_bar $ARCFACE_PCT
echo ""
REMAINING=$((TOTAL - ARCFACE))
if [ $REMAINING -gt 0 ]; then
    printf "  ${RED}⏳ Remaining: %'d faces${NC}\n" "$REMAINING"
fi
echo ""

# VGGFace2
STATUS=$(get_status $VGGFACE2 $TOTAL)
echo -e "${YELLOW}VGGFace2${NC}"
printf "  %s Count: %'10d / %'d  (%5.1f%%)\n" "$STATUS" "$VGGFACE2" "$TOTAL" "$VGGFACE2_PCT"
printf "  Progress: "
create_progress_bar $VGGFACE2_PCT
echo ""
REMAINING=$((TOTAL - VGGFACE2))
if [ $REMAINING -gt 0 ]; then
    printf "  ${RED}⏳ Remaining: %'d faces${NC}\n" "$REMAINING"
fi
echo ""

# InsightFace
STATUS=$(get_status $INSIGHTFACE $TOTAL)
echo -e "${YELLOW}InsightFace${NC}"
printf "  %s Count: %'10d / %'d  (%5.1f%%)\n" "$STATUS" "$INSIGHTFACE" "$TOTAL" "$INSIGHTFACE_PCT"
printf "  Progress: "
create_progress_bar $INSIGHTFACE_PCT
echo ""
REMAINING=$((TOTAL - INSIGHTFACE))
if [ $REMAINING -gt 0 ]; then
    printf "  ${RED}⏳ Remaining: %'d faces${NC}\n" "$REMAINING"
fi
echo ""

# Statistical
STATUS=$(get_status $STATISTICAL $TOTAL)
echo -e "${YELLOW}Statistical${NC}"
printf "  %s Count: %'10d / %'d  (%5.1f%%)\n" "$STATUS" "$STATISTICAL" "$TOTAL" "$STATISTICAL_PCT"
printf "  Progress: "
create_progress_bar $STATISTICAL_PCT
echo ""
REMAINING=$((TOTAL - STATISTICAL))
if [ $REMAINING -gt 0 ]; then
    printf "  ${RED}⏳ Remaining: %'d faces${NC}\n" "$REMAINING"
fi
echo ""

# Multi-model statistics
echo -e "${MAGENTA}📈 MULTI-MODEL COVERAGE${NC}"
echo -e "${MAGENTA}─────────────────────────────────────────────────────────────────────────────${NC}"

MULTI_STATS=$(PGPASSWORD=$DB_PASSWORD psql -h localhost -U $DB_USER -d $DB_NAME -t -A -F'|' -c "
SELECT
    COUNT(*) FILTER (WHERE
        CASE WHEN embedding_facenet IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_arcface IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_vggface2 IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_insightface IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_statistical IS NOT NULL THEN 1 ELSE 0 END = 0
    ) as no_embeddings,
    COUNT(*) FILTER (WHERE
        CASE WHEN embedding_facenet IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_arcface IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_vggface2 IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_insightface IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_statistical IS NOT NULL THEN 1 ELSE 0 END = 1
    ) as one_model,
    COUNT(*) FILTER (WHERE
        CASE WHEN embedding_facenet IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_arcface IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_vggface2 IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_insightface IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_statistical IS NOT NULL THEN 1 ELSE 0 END = 2
    ) as two_models,
    COUNT(*) FILTER (WHERE
        CASE WHEN embedding_facenet IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_arcface IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_vggface2 IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_insightface IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_statistical IS NOT NULL THEN 1 ELSE 0 END >= 3
    ) as three_plus_models,
    COUNT(*) FILTER (WHERE
        CASE WHEN embedding_facenet IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_arcface IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_vggface2 IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_insightface IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN embedding_statistical IS NOT NULL THEN 1 ELSE 0 END = 5
    ) as all_five_models
FROM faces;
")

IFS='|' read -r NO_EMB ONE_MODEL TWO_MODELS THREE_PLUS ALL_FIVE <<< "$MULTI_STATS"

printf "%-35s %'10d\n" "Faces with no embeddings:" "$NO_EMB"
printf "%-35s %'10d\n" "Faces with 1 model:" "$ONE_MODEL"
printf "%-35s %'10d\n" "Faces with 2 models:" "$TWO_MODELS"
printf "%-35s %'10d\n" "Faces with 3+ models:" "$THREE_PLUS"
printf "%-35s %'10d ✨\n" "Faces with ALL 5 models:" "$ALL_FIVE"
echo ""

# Database size
echo -e "${CYAN}💾 DATABASE INFORMATION${NC}"
echo -e "${CYAN}─────────────────────────────────────────────────────────────────────────────${NC}"

DB_SIZE=$(PGPASSWORD=$DB_PASSWORD psql -h localhost -U $DB_USER -d $DB_NAME -t -A -c "
SELECT pg_size_pretty(pg_database_size('$DB_NAME'));
")

TABLE_SIZE=$(PGPASSWORD=$DB_PASSWORD psql -h localhost -U $DB_USER -d $DB_NAME -t -A -c "
SELECT pg_size_pretty(pg_total_relation_size('faces'));
")

INDEX_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h localhost -U $DB_USER -d $DB_NAME -t -A -c "
SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'faces';
")

printf "%-35s %s\n" "Total Database Size:" "$DB_SIZE"
printf "%-35s %s\n" "Faces Table Size:" "$TABLE_SIZE"
printf "%-35s %d\n" "Number of Indexes:" "$INDEX_COUNT"
echo ""

# Recent activity
echo -e "${BLUE}📅 RECENT ACTIVITY (Last 7 Days)${NC}"
echo -e "${BLUE}─────────────────────────────────────────────────────────────────────────────${NC}"

RECENT=$(PGPASSWORD=$DB_PASSWORD psql -h localhost -U $DB_USER -d $DB_NAME -t -A -F'|' -c "
SELECT
    TO_CHAR(DATE(created_at), 'YYYY-MM-DD') as date,
    COUNT(*) as count
FROM faces
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY DATE(created_at)
ORDER BY date DESC
LIMIT 7;
")

if [ -z "$RECENT" ]; then
    echo "  No recent activity"
else
    while IFS='|' read -r date count; do
        printf "  %s: %'10d faces\n" "$date" "$count"
    done <<< "$RECENT"
fi

echo ""
echo -e "${CYAN}================================================================================${NC}"
echo ""

# Quick commands reference
echo -e "${GREEN}💡 QUICK COMMANDS:${NC}"
echo "  Run embedding:     ./run_embedding.sh"
echo "  This status:       ./show_embedding_status.sh"
echo "  Watch live:        watch -n 5 ./show_embedding_status.sh"
echo ""
