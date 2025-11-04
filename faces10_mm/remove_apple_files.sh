#!/bin/bash
################################################################################
# Remove Apple Double Files (._* files)
################################################################################
#
# These are macOS metadata files that start with ._ prefix
# They are not needed on Linux and take up space
#
# Usage:
#   ./remove_apple_files.sh              # Dry-run (shows what would be deleted)
#   ./remove_apple_files.sh --confirm    # Actually delete files
#
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

FACES_DIR="/home/pi/faces"
DRY_RUN=true

# Parse arguments
if [ "$1" = "--confirm" ] || [ "$1" = "-c" ]; then
    DRY_RUN=false
fi

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}🗑️  REMOVE APPLE DOUBLE FILES${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

echo -e "${BLUE}📁 Scanning directory: $FACES_DIR${NC}"
echo ""

# Count Apple Double files
echo -e "${YELLOW}⏳ Finding Apple Double files (._*)...${NC}"

cd "$FACES_DIR" || exit 1

# Find all ._* files
APPLE_FILES=$(find . -maxdepth 1 -name "._*" -type f 2>/dev/null)
APPLE_COUNT=$(echo "$APPLE_FILES" | grep -c "^" 2>/dev/null || echo 0)

if [ "$APPLE_COUNT" -eq 0 ]; then
    echo -e "${GREEN}✅ No Apple Double files found!${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}=================================================================================${NC}"
echo -e "${BLUE}📊 STATISTICS${NC}"
echo -e "${BLUE}=================================================================================${NC}"
echo ""
echo -e "Apple Double Files Found:  ${YELLOW}$APPLE_COUNT${NC}"
echo -e "Type:                      macOS metadata files (._*)"
echo -e "Needed on Linux:           ${RED}NO${NC}"
echo -e "Safe to delete:            ${GREEN}YES${NC}"
echo ""

# Show sample files
echo -e "${BLUE}Sample files (first 10):${NC}"
echo "$APPLE_FILES" | head -10 | sed 's/^/  /'
if [ "$APPLE_COUNT" -gt 10 ]; then
    echo -e "  ${YELLOW}... and $((APPLE_COUNT - 10)) more files${NC}"
fi
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}=================================================================================${NC}"
    echo -e "${YELLOW}⚠️  DRY-RUN MODE - NO FILES DELETED${NC}"
    echo -e "${YELLOW}=================================================================================${NC}"
    echo ""
    echo "This was a dry-run. No files were actually deleted."
    echo ""
    echo "To actually delete these Apple Double files, run:"
    echo -e "  ${GREEN}./remove_apple_files.sh --confirm${NC}"
    echo ""
else
    echo -e "${RED}=================================================================================${NC}"
    echo -e "${RED}⚠️  DELETION MODE${NC}"
    echo -e "${RED}=================================================================================${NC}"
    echo ""
    echo -e "${YELLOW}About to delete $APPLE_COUNT Apple Double files.${NC}"
    echo ""
    read -p "Type 'DELETE' to confirm: " CONFIRM

    if [ "$CONFIRM" != "DELETE" ]; then
        echo -e "${BLUE}Deletion cancelled.${NC}"
        exit 0
    fi

    echo ""
    echo -e "${RED}🗑️  Deleting files...${NC}"

    DELETED=0
    ERRORS=0

    # Delete all ._* files
    while IFS= read -r file; do
        if [ -n "$file" ]; then
            if rm -f "$file" 2>/dev/null; then
                ((DELETED++))
                # Show progress every 1000 files
                if [ $((DELETED % 1000)) -eq 0 ]; then
                    echo "  Progress: $DELETED files deleted..."
                fi
            else
                ((ERRORS++))
            fi
        fi
    done <<< "$APPLE_FILES"

    echo ""
    echo -e "${BLUE}=================================================================================${NC}"
    echo -e "${BLUE}✅ DELETION COMPLETE${NC}"
    echo -e "${BLUE}=================================================================================${NC}"
    echo ""
    echo -e "Successfully deleted:  ${GREEN}$DELETED${NC} files"
    echo -e "Errors:               ${RED}$ERRORS${NC}"
    echo ""
    echo -e "${GREEN}Apple Double files removed!${NC}"
    echo ""
    echo "Run './check_status.sh' to see updated statistics."
    echo ""
fi

echo -e "${BLUE}================================================================================${NC}"
