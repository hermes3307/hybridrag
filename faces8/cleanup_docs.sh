#!/bin/bash
################################################################################
# Documentation Cleanup Script
################################################################################
#
# This script organizes markdown documentation by:
# - Moving old/redundant docs to docs/archive/
# - Keeping essential documentation in main folder or docs/
# - Creating a clear documentation structure
#
# Usage:
#   ./cleanup_docs.sh              # Interactive mode
#   ./cleanup_docs.sh --dry-run    # Show what would be done
#
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --help)
            echo "Usage: $0 [--dry-run] [--help]"
            exit 0
            ;;
    esac
done

cd "$(dirname "$0")"

echo "================================================================================"
echo "📚 DOCUMENTATION CLEANUP"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}⚠️  DRY RUN MODE - No changes will be made${NC}"
    echo ""
fi

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN]${NC} $1"
    else
        echo -e "${GREEN}[EXECUTE]${NC} $1"
        eval "$1"
    fi
}

# Create backup
echo "📦 Step 1: Creating documentation backup..."
echo "--------------------------------------------------------------------------------"
BACKUP_FILE="../docs_backup_$(date +%Y%m%d_%H%M%S).tar.gz"

if [ "$DRY_RUN" = false ]; then
    tar -czf "$BACKUP_FILE" *.md *.txt 2>/dev/null || true
    if [ -f "$BACKUP_FILE" ]; then
        BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        echo -e "${GREEN}✅ Backup created: $BACKUP_FILE ($BACKUP_SIZE)${NC}"
    fi
else
    echo -e "${BLUE}[DRY RUN]${NC} Would create: $BACKUP_FILE"
fi
echo ""

# Create directories
echo "📁 Step 2: Creating directories..."
echo "--------------------------------------------------------------------------------"
run_cmd "mkdir -p docs/archive"
run_cmd "mkdir -p docs/guides"
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Archive old/temporary docs
echo "📦 Step 3: Archiving old/temporary documentation..."
echo "--------------------------------------------------------------------------------"

OLD_DOCS=(
    "BUGFIX_NOTES.md"
    "DELETION_SAFETY_CONFIRMATION.md"
    "METADATA_GENERATOR_CHANGELOG.md"
    "WHATS_NEW.md"
    "PERFORMANCE_RESULTS.md"
    "MONITOR_ARCHITECTURE.md"
)

for doc in "${OLD_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        run_cmd "mv '$doc' docs/archive/"
    fi
done

echo -e "${GREEN}✅ Old docs archived${NC}"
echo ""

# Archive duplicate/redundant docs
echo "🔄 Step 4: Archiving duplicate/redundant documentation..."
echo "--------------------------------------------------------------------------------"

DUPLICATE_DOCS=(
    "EMBEDDING_CLI_QUICKSTART.md"
    "EMBEDDING_MANAGEMENT_GUIDE.md"
    "README_EMBEDDING_SYSTEM.md"
    "QUICK_REFERENCE_10K.md"
    "QUICK_START_MONITOR.md"
    "MONITOR_SUMMARY.md"
    "SEARCH_README.md"
    "SHELL_SCRIPTS_README.md"
    "METADATA_GENERATOR_README.md"
    "METADATA_FIX.md"
    "METADATA_SOLUTION_SUMMARY.md"
    "ENHANCED_GUIDE.md"
    "JSON_GENERATION_GUIDE.md"
    "HOW_TO_FIX_METADATA.md"
    "DUPLICATE_REMOVAL_SUMMARY.txt"
    "EMBEDDING_CLI_SUMMARY.txt"
    "QUICK_START.txt"
    "README_INSTALL.txt"
)

for doc in "${DUPLICATE_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        run_cmd "mv '$doc' docs/archive/"
    fi
done

echo -e "${GREEN}✅ Duplicate docs archived${NC}"
echo ""

# Keep essential docs in main directory
echo "✅ Step 5: Essential documentation (keeping in main directory)..."
echo "--------------------------------------------------------------------------------"

ESSENTIAL_DOCS=(
    "INSTALLATION.md"
    "QUICK_START.md"
    "CLEANUP_SUMMARY.txt"
    "DOCS_CLEANUP_SUMMARY.txt"
    "QUICK_REFERENCE.txt"
    "requirements.txt"
)

echo "Essential docs to keep in main folder:"
for doc in "${ESSENTIAL_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo "  ✓ $doc"
    fi
done
echo ""

# Move feature guides to docs/guides/
echo "📖 Step 6: Moving feature guides to docs/guides/..."
echo "--------------------------------------------------------------------------------"

GUIDE_DOCS=(
    "PGVECTOR_README.md"
    "SPEEDUP_GUIDE.md"
    "FILE_CLEANUP_PLAN.md"
    "SEARCH_GUIDE.md"
    "UNDERSTANDING_THE_NUMBERS.md"
    "VECTOR_SQL_EXAMPLES.md"
    "EMBEDDING_CLI_README.md"
    "MONITOR_README.md"
    "BULK_DOWNLOAD_README.md"
    "REMOVE_DUPLICATES_GUIDE.md"
    "UBUNTU_10K_DOWNLOAD_GUIDE.md"
    "SHELL_SCRIPTS_GUIDE.md"
    "OPENCV_SETUP.md"
)

for doc in "${GUIDE_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        run_cmd "mv '$doc' docs/guides/"
    fi
done

echo -e "${GREEN}✅ Guides organized${NC}"
echo ""

# Summary
echo "================================================================================"
echo "📊 CLEANUP SUMMARY"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = false ]; then
    echo "📦 Backup: $BACKUP_FILE"
    echo ""
    echo "📁 Documentation structure:"
    echo "--------------------------------------------------------------------------------"
    echo ""
    echo "Main directory ($(ls -1 *.md 2>/dev/null | wc -l) essential docs):"
    ls -1 *.md 2>/dev/null | head -10 || echo "  (none)"
    echo ""
    echo "Feature guides ($(ls -1 docs/guides/*.md 2>/dev/null | wc -l) files):"
    ls -1 docs/guides/*.md 2>/dev/null | sed 's|docs/guides/||' | head -10 || echo "  (none)"
    echo ""
    echo "Archived docs ($(ls -1 docs/archive/*.md 2>/dev/null | wc -l) files):"
    ls -1 docs/archive/*.md 2>/dev/null | sed 's|docs/archive/||' | head -10 || echo "  (none)"
    echo ""
else
    echo -e "${YELLOW}⚠️  DRY RUN MODE - No changes were made${NC}"
    echo ""
fi

echo "================================================================================"
echo "✅ DOCUMENTATION CLEANUP COMPLETE!"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = false ]; then
    echo "📚 Documentation structure:"
    echo "  • Main folder: Essential docs (INSTALLATION.md, QUICK_START.md)"
    echo "  • docs/guides/: Feature-specific guides"
    echo "  • docs/archive/: Old/redundant documentation"
    echo ""
    echo "To read documentation:"
    echo "  cat QUICK_START.md              # Getting started"
    echo "  cat docs/guides/SPEEDUP_GUIDE.md   # Performance tips"
    echo "  ls docs/guides/                 # List all guides"
else
    echo "Run without --dry-run to perform cleanup:"
    echo "  ./cleanup_docs.sh"
fi

echo ""
