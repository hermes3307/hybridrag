#!/bin/bash
################################################################################
# Database Backup Script
################################################################################
#
# This script creates a backup of your face recognition database
# before migration to multi-model schema
#
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================================"
echo "🔒 DATABASE BACKUP SCRIPT"
echo "================================================================================"
echo ""

# Configuration
DB_NAME="${POSTGRES_DB:-vector_db}"
DB_USER="${POSTGRES_USER:-postgres}"
BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/faces_backup_${TIMESTAMP}.sql"

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo -e "${BLUE}[INFO]${NC} Database: $DB_NAME"
echo -e "${BLUE}[INFO]${NC} Backup location: $BACKUP_FILE"
echo ""

# Check if faces table exists
echo -e "${BLUE}[INFO]${NC} Checking if faces table exists..."
TABLE_EXISTS=$(sudo -u postgres psql -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='faces';")

if [ "$TABLE_EXISTS" = "0" ]; then
    echo -e "${YELLOW}[WARNING]${NC} No 'faces' table found in database '$DB_NAME'"
    echo "Nothing to backup."
    exit 0
fi

# Get row count
echo -e "${BLUE}[INFO]${NC} Counting records..."
ROW_COUNT=$(sudo -u postgres psql -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM faces;")
echo -e "${GREEN}[SUCCESS]${NC} Found $ROW_COUNT records to backup"
echo ""

# Create full database dump
echo -e "${BLUE}[INFO]${NC} Creating full database backup..."
sudo -u postgres pg_dump "$DB_NAME" > "$BACKUP_FILE"
echo -e "${GREEN}[SUCCESS]${NC} Full database backup created: $BACKUP_FILE"

# Create table-only backup (just faces table)
TABLE_BACKUP_FILE="${BACKUP_DIR}/faces_table_only_${TIMESTAMP}.sql"
echo -e "${BLUE}[INFO]${NC} Creating faces table backup..."
sudo -u postgres pg_dump "$DB_NAME" -t faces > "$TABLE_BACKUP_FILE"
echo -e "${GREEN}[SUCCESS]${NC} Table backup created: $TABLE_BACKUP_FILE"

# Create CSV export for easy inspection
CSV_FILE="${BACKUP_DIR}/faces_data_${TIMESTAMP}.csv"
echo -e "${BLUE}[INFO]${NC} Exporting to CSV for inspection..."

# Use absolute path and ensure postgres can write to it
ABS_CSV_FILE="$(realpath "$CSV_FILE")"
CSV_TEMP="/tmp/faces_export_${TIMESTAMP}.csv"

# Export to /tmp first (postgres has access)
sudo -u postgres psql -d "$DB_NAME" -c "\COPY (SELECT id, face_id, file_path, timestamp, image_hash, embedding_model, age_estimate, gender, created_at FROM faces) TO '$CSV_TEMP' WITH CSV HEADER;" 2>/dev/null

# Move to final location and fix ownership
if [ -f "$CSV_TEMP" ]; then
    sudo mv "$CSV_TEMP" "$CSV_FILE"
    sudo chown $(whoami):$(whoami) "$CSV_FILE"
    echo -e "${GREEN}[SUCCESS]${NC} CSV export created: $CSV_FILE"
else
    echo -e "${YELLOW}[WARNING]${NC} CSV export skipped (not critical for backup)"
    CSV_FILE=""
fi

# Get backup file sizes
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
TABLE_SIZE=$(du -h "$TABLE_BACKUP_FILE" | cut -f1)
if [ -n "$CSV_FILE" ] && [ -f "$CSV_FILE" ]; then
    CSV_SIZE=$(du -h "$CSV_FILE" | cut -f1)
else
    CSV_SIZE="N/A"
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ BACKUP COMPLETED SUCCESSFULLY${NC}"
echo "================================================================================"
echo ""
echo "Backup Summary:"
echo "  Records backed up: $ROW_COUNT"
echo "  Full database backup: $BACKUP_FILE ($BACKUP_SIZE)"
echo "  Table-only backup: $TABLE_BACKUP_FILE ($TABLE_SIZE)"
if [ -n "$CSV_FILE" ] && [ -f "$CSV_FILE" ]; then
    echo "  CSV export: $CSV_FILE ($CSV_SIZE)"
else
    echo "  CSV export: Skipped (optional)"
fi
echo ""
echo "To restore from backup:"
echo "  Full restore: sudo -u postgres psql -d $DB_NAME < $BACKUP_FILE"
echo "  Table only: sudo -u postgres psql -d $DB_NAME < $TABLE_BACKUP_FILE"
echo ""
echo "================================================================================"
