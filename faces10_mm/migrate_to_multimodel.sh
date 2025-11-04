#!/bin/bash
################################################################################
# Migration to Multi-Model Schema
################################################################################
#
# This script safely migrates data from single-model to multi-model schema
#
# Steps:
#   1. Backup existing data
#   2. Rename old table to faces_old
#   3. Create new multi-model schema
#   4. Migrate data to appropriate embedding column
#   5. Verify migration
#
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "================================================================================"
echo "🔄 MIGRATION TO MULTI-MODEL SCHEMA"
echo "================================================================================"
echo ""

# Configuration
DB_NAME="${POSTGRES_DB:-vector_db}"
DB_USER="${POSTGRES_USER:-postgres}"

log_info "Database: $DB_NAME"
echo ""

# Step 1: Check if faces table exists
log_info "Step 1: Checking current database state..."
TABLE_EXISTS=$(sudo -u postgres psql -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='faces';")

if [ "$TABLE_EXISTS" = "0" ]; then
    log_warning "No 'faces' table found. Nothing to migrate."
    log_info "You can run ./install.sh to create a fresh multi-model schema."
    exit 0
fi

# Get current row count
ROW_COUNT=$(sudo -u postgres psql -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM faces;")
log_success "Found existing 'faces' table with $ROW_COUNT records"

# Get current schema info
HAS_EMBEDDING=$(sudo -u postgres psql -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM information_schema.columns WHERE table_name='faces' AND column_name='embedding';")
HAS_EMBEDDING_MODEL=$(sudo -u postgres psql -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM information_schema.columns WHERE table_name='faces' AND column_name='embedding_model';")

if [ "$HAS_EMBEDDING" = "1" ]; then
    log_info "Current schema: Single-model (has 'embedding' column)"
else
    log_warning "Current schema appears to already be multi-model"
    log_info "Column 'embedding' not found. You may already be using the new schema."
    exit 0
fi

echo ""
log_warning "⚠️  THIS WILL MODIFY YOUR DATABASE!"
log_warning "⚠️  Please ensure you have a backup before proceeding."
echo ""
read -p "Do you want to create a backup now? (RECOMMENDED - yes/no): " -r
echo ""

if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    log_info "Running backup script..."
    ./backup_database.sh
    echo ""
fi

echo ""
read -p "Continue with migration? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    log_info "Migration cancelled by user."
    exit 0
fi

# Step 2: Rename old table
log_info "Step 2: Renaming old table to 'faces_old'..."
sudo -u postgres psql -d "$DB_NAME" -c "ALTER TABLE IF EXISTS faces RENAME TO faces_old;" 2>/dev/null || true
sudo -u postgres psql -d "$DB_NAME" -c "DROP VIEW IF EXISTS faces_with_metadata CASCADE;" 2>/dev/null || true
log_success "Old table renamed to 'faces_old'"

# Step 3: Create new multi-model schema
log_info "Step 3: Creating new multi-model schema..."
if [ ! -f "schema.sql" ]; then
    log_error "schema.sql not found!"
    exit 1
fi

sudo -u postgres psql -d "$DB_NAME" -f schema.sql
log_success "New multi-model schema created"

# Step 4: Migrate data
log_info "Step 4: Migrating data from old table to new table..."

# Create migration query
cat > /tmp/migrate_query.sql <<'EOF'
-- Migrate data from faces_old to new faces table
-- Map old single embedding to appropriate column based on embedding_model

INSERT INTO faces (
    face_id,
    file_path,
    timestamp,
    image_hash,
    embedding_facenet,
    embedding_arcface,
    embedding_vggface2,
    embedding_insightface,
    embedding_statistical,
    models_processed,
    age_estimate,
    gender,
    brightness,
    contrast,
    sharpness,
    metadata,
    created_at,
    updated_at
)
SELECT
    face_id,
    file_path,
    timestamp,
    image_hash,
    -- Map embedding to correct column based on model name
    CASE WHEN embedding_model = 'facenet' THEN embedding ELSE NULL END as embedding_facenet,
    CASE WHEN embedding_model = 'arcface' THEN embedding ELSE NULL END as embedding_arcface,
    CASE WHEN embedding_model = 'vggface2' THEN embedding ELSE NULL END as embedding_vggface2,
    CASE WHEN embedding_model = 'insightface' THEN embedding ELSE NULL END as embedding_insightface,
    CASE WHEN embedding_model = 'statistical' THEN embedding ELSE NULL END as embedding_statistical,
    -- Set models_processed array
    ARRAY[embedding_model] as models_processed,
    age_estimate,
    gender,
    brightness,
    contrast,
    sharpness,
    metadata,
    created_at,
    updated_at
FROM faces_old;
EOF

sudo -u postgres psql -d "$DB_NAME" -f /tmp/migrate_query.sql
rm /tmp/migrate_query.sql

log_success "Data migration completed"

# Step 5: Verify migration
log_info "Step 5: Verifying migration..."

NEW_ROW_COUNT=$(sudo -u postgres psql -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM faces;")
OLD_ROW_COUNT=$(sudo -u postgres psql -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM faces_old;")

if [ "$NEW_ROW_COUNT" = "$OLD_ROW_COUNT" ]; then
    log_success "Row count verification: OK ($NEW_ROW_COUNT records)"
else
    log_error "Row count mismatch! Old: $OLD_ROW_COUNT, New: $NEW_ROW_COUNT"
    log_warning "Old table 'faces_old' is preserved for safety"
    exit 1
fi

# Check embedding distribution
log_info "Checking embedding distribution by model..."
sudo -u postgres psql -d "$DB_NAME" <<'EOF'
SELECT
    COUNT(*) FILTER (WHERE embedding_facenet IS NOT NULL) as facenet_count,
    COUNT(*) FILTER (WHERE embedding_arcface IS NOT NULL) as arcface_count,
    COUNT(*) FILTER (WHERE embedding_vggface2 IS NOT NULL) as vggface2_count,
    COUNT(*) FILTER (WHERE embedding_insightface IS NOT NULL) as insightface_count,
    COUNT(*) FILTER (WHERE embedding_statistical IS NOT NULL) as statistical_count
FROM faces;
EOF

log_success "Migration verification completed"

echo ""
echo "================================================================================"
log_success "✅ MIGRATION COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  Records migrated: $NEW_ROW_COUNT"
echo "  Old table: faces_old (preserved for safety)"
echo "  New table: faces (multi-model schema)"
echo ""
echo "Next steps:"
echo "  1. Verify your data in the new table"
echo "  2. Test searching with the migrated data"
echo "  3. Run additional models on existing faces:"
echo "     ./run_embedding.sh"
echo "  4. Once verified, you can drop the old table:"
echo "     sudo -u postgres psql -d $DB_NAME -c 'DROP TABLE faces_old;'"
echo ""
echo "To check statistics:"
echo "  sudo -u postgres psql -d $DB_NAME -c 'SELECT * FROM get_database_stats();'"
echo ""
echo "================================================================================"
