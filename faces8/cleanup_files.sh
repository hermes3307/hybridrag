#!/bin/bash
################################################################################
# File Cleanup Script
################################################################################
#
# This script organizes and cleans up the faces8 directory by:
# - Moving old/duplicate files to archive/
# - Moving test files to tests/
# - Renaming files for clarity
# - Creating a backup first
#
# Usage:
#   ./cleanup_files.sh              # Interactive mode with prompts
#   ./cleanup_files.sh --auto       # Automatic mode (no prompts)
#   ./cleanup_files.sh --dry-run    # Show what would be done
#
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DRY_RUN=false
AUTO_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --auto)
            AUTO_MODE=true
            ;;
        --help)
            echo "Usage: $0 [--dry-run] [--auto] [--help]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be done without making changes"
            echo "  --auto       Run automatically without prompts"
            echo "  --help       Show this help message"
            exit 0
            ;;
    esac
done

cd "$(dirname "$0")"

echo "================================================================================"
echo "🧹 FILE CLEANUP AND ORGANIZATION"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}⚠️  DRY RUN MODE - No changes will be made${NC}"
    echo ""
fi

# Function to execute or show command
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN]${NC} $1"
    else
        echo -e "${GREEN}[EXECUTE]${NC} $1"
        eval "$1"
    fi
}

# Step 1: Create backup
echo "📦 Step 1: Creating backup..."
echo "--------------------------------------------------------------------------------"
BACKUP_FILE="../faces8_backup_$(date +%Y%m%d_%H%M%S).tar.gz"

if [ "$DRY_RUN" = false ]; then
    tar -czf "$BACKUP_FILE" *.py *.sh *.md 2>/dev/null || true
    if [ -f "$BACKUP_FILE" ]; then
        BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        echo -e "${GREEN}✅ Backup created: $BACKUP_FILE ($BACKUP_SIZE)${NC}"
    else
        echo -e "${RED}❌ Backup failed!${NC}"
        exit 1
    fi
else
    echo -e "${BLUE}[DRY RUN]${NC} Would create: $BACKUP_FILE"
fi
echo ""

# Step 2: Create directories
echo "📁 Step 2: Creating directories..."
echo "--------------------------------------------------------------------------------"
run_cmd "mkdir -p archive"
run_cmd "mkdir -p tests"
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Step 3: Move old/duplicate files to archive
echo "🗃️  Step 3: Moving old/duplicate files to archive..."
echo "--------------------------------------------------------------------------------"

OLD_FILES=(
    "fix_demographics.py"
    "fix_demographics_simple.py"
    "monitor_demo.py"
    "embed.sh"
    "quick_download.sh"
    "download_with_metadata.sh"
)

for file in "${OLD_FILES[@]}"; do
    if [ -f "$file" ]; then
        run_cmd "mv '$file' archive/"
    else
        echo -e "${YELLOW}⚠️  File not found: $file${NC}"
    fi
done

echo -e "${GREEN}✅ Old files archived${NC}"
echo ""

# Step 4: Move test files
echo "🧪 Step 4: Moving test files to tests/ directory..."
echo "--------------------------------------------------------------------------------"

if ls test_*.py 1> /dev/null 2>&1; then
    for file in test_*.py; do
        if [ -f "$file" ]; then
            run_cmd "mv '$file' tests/"
        fi
    done
    echo -e "${GREEN}✅ Test files moved${NC}"
else
    echo -e "${YELLOW}⚠️  No test files found${NC}"
fi
echo ""

# Step 5: Rename files for clarity
echo "✏️  Step 5: Renaming files for clarity..."
echo "--------------------------------------------------------------------------------"

# Only rename if files exist and target doesn't exist
if [ -f "start.sh" ] && [ ! -f "start_system.sh" ]; then
    run_cmd "mv start.sh start_system.sh"
fi

if [ -f "check_status.sh" ] && [ ! -f "check_system_status.sh" ]; then
    run_cmd "mv check_status.sh check_system_status.sh"
fi

if [ -f "run_monitor.sh" ] && [ ! -f "start_monitor.sh" ]; then
    run_cmd "mv run_monitor.sh start_monitor.sh"
fi

echo -e "${GREEN}✅ Files renamed${NC}"
echo ""

# Step 6: Summary
echo "================================================================================"
echo "📊 CLEANUP SUMMARY"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = false ]; then
    echo "📦 Backup: $BACKUP_FILE"
    echo ""
    echo "📁 Current directory structure:"
    echo "--------------------------------------------------------------------------------"
    echo ""
    echo "Main Python files ($(ls -1 *.py 2>/dev/null | wc -l)):"
    ls -1 *.py 2>/dev/null | head -20 || echo "  (none)"
    echo ""
    echo "Main Shell scripts ($(ls -1 *.sh 2>/dev/null | wc -l)):"
    ls -1 *.sh 2>/dev/null | head -20 || echo "  (none)"
    echo ""
    echo "Archived files ($(ls -1 archive/*.{py,sh} 2>/dev/null | wc -l)):"
    ls -1 archive/*.{py,sh} 2>/dev/null | head -10 || echo "  (none)"
    echo ""
    echo "Test files ($(ls -1 tests/*.py 2>/dev/null | wc -l)):"
    ls -1 tests/*.py 2>/dev/null | head -10 || echo "  (none)"
    echo ""
else
    echo -e "${YELLOW}⚠️  DRY RUN MODE - No changes were made${NC}"
    echo ""
    echo "To actually perform cleanup, run:"
    echo "  ./cleanup_files.sh"
    echo ""
fi

echo "================================================================================"
echo "✅ CLEANUP COMPLETE!"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = false ]; then
    echo "Next steps:"
    echo "  1. Verify everything works: ./run_embedding.sh"
    echo "  2. Check archived files: ls -la archive/"
    echo "  3. After 1 month, if unused: rm -rf archive/"
    echo ""
    echo "To restore from backup if needed:"
    echo "  tar -xzf $BACKUP_FILE"
else
    echo "Run without --dry-run to actually perform cleanup:"
    echo "  ./cleanup_files.sh"
fi

echo ""
