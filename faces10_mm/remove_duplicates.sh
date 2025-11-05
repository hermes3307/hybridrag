#!/bin/bash
################################################################################
# Remove Duplicate Images Script
################################################################################
#
# This script removes duplicate image files, keeping only ONE copy of each
# unique image (identified by hash). It also removes the corresponding JSON
# files for deleted images.
#
# Usage:
#   ./remove_duplicates.sh              # Dry-run (shows what would be deleted)
#   ./remove_duplicates.sh --confirm    # Actually delete files
#   ./remove_duplicates.sh --stats      # Show duplicate statistics only
#
# Safety:
#   - Default is DRY-RUN mode (no files deleted)
#   - Creates backup list before deletion
#   - Keeps the OLDEST file for each hash
#
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
FACES_DIR="./faces"  # Use symlink to handle correctly
BACKUP_DIR="./duplicate_backups"
DRY_RUN=true

# Resolve symlink to real path
if [ -L "$FACES_DIR" ]; then
    FACES_DIR=$(readlink -f "$FACES_DIR")
fi

# Parse arguments
case "$1" in
    --confirm|-c)
        DRY_RUN=false
        ;;
    --stats|-s)
        STATS_ONLY=true
        ;;
    --help|-h)
        echo "Remove Duplicate Images Script"
        echo ""
        echo "Usage:"
        echo "  ./remove_duplicates.sh              Dry-run (shows what would be deleted)"
        echo "  ./remove_duplicates.sh --confirm    Actually delete duplicate files"
        echo "  ./remove_duplicates.sh --stats      Show statistics only"
        echo "  ./remove_duplicates.sh --help       Show this help"
        echo ""
        echo "How it works:"
        echo "  1. Groups files by hash (last part of filename)"
        echo "  2. For each hash with multiple files, keeps the OLDEST one"
        echo "  3. Deletes duplicate files and their JSON metadata"
        echo ""
        echo "Safety:"
        echo "  - Default mode is DRY-RUN (no deletion)"
        echo "  - Creates backup list of deleted files"
        echo "  - Always keeps at least one copy of each unique image"
        exit 0
        ;;
esac

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}🗑️  DUPLICATE IMAGE REMOVER${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Check if faces directory exists
if [ ! -d "$FACES_DIR" ]; then
    echo -e "${RED}Error: Faces directory not found: $FACES_DIR${NC}"
    exit 1
fi

echo -e "${CYAN}📁 Scanning directory: $FACES_DIR${NC}"
echo ""

# Create temporary file for analysis
TEMP_FILE=$(mktemp)
DUPLICATES_FILE=$(mktemp)

# Find all image files and extract hash
cd "$FACES_DIR" || exit 1

echo -e "${YELLOW}⏳ Analyzing files...${NC}"

# Get all .jpg files and extract their hashes
find . -maxdepth 1 -name "face_*.jpg" -type f | while read -r file; do
    # Extract hash (last part before .jpg)
    basename "$file" | sed 's/\.jpg$//' | awk -F'_' '{print $NF"|"$0".jpg"}'
done > "$TEMP_FILE"

# Count total files
TOTAL_FILES=$(wc -l < "$TEMP_FILE")

# Find duplicates (hashes that appear more than once)
awk -F'|' '{print $1}' "$TEMP_FILE" | sort | uniq -d > "$DUPLICATES_FILE"

DUPLICATE_HASHES=$(wc -l < "$DUPLICATES_FILE")

if [ "$DUPLICATE_HASHES" -eq 0 ]; then
    echo -e "${GREEN}✅ No duplicate images found!${NC}"
    echo -e "${GREEN}All $TOTAL_FILES images are unique.${NC}"
    rm -f "$TEMP_FILE" "$DUPLICATES_FILE"
    exit 0
fi

echo ""
echo -e "${BLUE}=================================================================================${NC}"
echo -e "${BLUE}📊 STATISTICS${NC}"
echo -e "${BLUE}=================================================================================${NC}"
echo ""

# Calculate statistics
UNIQUE_HASHES=$(awk -F'|' '{print $1}' "$TEMP_FILE" | sort -u | wc -l)
DUPLICATE_FILES=$((TOTAL_FILES - UNIQUE_HASHES))

echo -e "Total Image Files:        ${CYAN}$TOTAL_FILES${NC}"
echo -e "Unique Images (by hash):  ${GREEN}$UNIQUE_HASHES${NC}"
echo -e "Duplicate Files:          ${YELLOW}$DUPLICATE_FILES${NC}"
echo -e "Hashes with duplicates:   ${YELLOW}$DUPLICATE_HASHES${NC}"
echo ""

if [ "$STATS_ONLY" = true ]; then
    echo -e "${BLUE}=================================================================================${NC}"
    echo ""
    echo "To see which files would be deleted:"
    echo "  ./remove_duplicates.sh"
    echo ""
    echo "To actually delete duplicate files:"
    echo "  ./remove_duplicates.sh --confirm"
    echo ""
    rm -f "$TEMP_FILE" "$DUPLICATES_FILE"
    exit 0
fi

# Show detailed duplicate information
echo -e "${BLUE}=================================================================================${NC}"
echo -e "${BLUE}🔍 DUPLICATE FILES TO REMOVE${NC}"
echo -e "${BLUE}=================================================================================${NC}"
echo ""

FILES_TO_DELETE=()
FILES_TO_KEEP=()

# Process each hash that has duplicates
while read -r hash; do
    # Get all files with this hash, sorted by modification time (oldest first)
    FILES=$(grep "^${hash}|" "$TEMP_FILE" | cut -d'|' -f2 | while read -r fname; do
        echo "$(stat -c '%Y' "$fname") $fname"
    done | sort -n | awk '{print $2}')

    # Convert to array
    FILES_ARRAY=($FILES)

    # Keep the first (oldest) file
    KEEP_FILE="${FILES_ARRAY[0]}"
    FILES_TO_KEEP+=("$KEEP_FILE")

    echo -e "${GREEN}Hash: $hash${NC}"
    echo -e "  ${GREEN}✅ KEEP:${NC} $KEEP_FILE (oldest)"

    # Mark the rest for deletion
    for ((i=1; i<${#FILES_ARRAY[@]}; i++)); do
        DELETE_FILE="${FILES_ARRAY[i]}"
        FILES_TO_DELETE+=("$DELETE_FILE")
        echo -e "  ${RED}❌ DELETE:${NC} $DELETE_FILE"
    done
    echo ""
done < "$DUPLICATES_FILE"

# Summary
echo -e "${BLUE}=================================================================================${NC}"
echo -e "${BLUE}📋 SUMMARY${NC}"
echo -e "${BLUE}=================================================================================${NC}"
echo ""
echo -e "Files to KEEP:    ${GREEN}${#FILES_TO_KEEP[@]}${NC}"
echo -e "Files to DELETE:  ${RED}${#FILES_TO_DELETE[@]}${NC}"
echo -e "Space to save:    ${YELLOW}~$((${#FILES_TO_DELETE[@]} * 2))${NC} files (images + JSON)"
echo ""

# Execute deletion or show dry-run message
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}=================================================================================${NC}"
    echo -e "${YELLOW}⚠️  DRY-RUN MODE - NO FILES DELETED${NC}"
    echo -e "${YELLOW}=================================================================================${NC}"
    echo ""
    echo "This was a dry-run. No files were actually deleted."
    echo ""
    echo "To actually delete the duplicate files, run:"
    echo -e "  ${GREEN}./remove_duplicates.sh --confirm${NC}"
    echo ""
else
    echo -e "${RED}=================================================================================${NC}"
    echo -e "${RED}⚠️  DELETION MODE - FILES WILL BE DELETED${NC}"
    echo -e "${RED}=================================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Creating backup list...${NC}"

    # Create backup directory
    mkdir -p "$BACKUP_DIR"

    # Save list of files to be deleted
    BACKUP_FILE="$BACKUP_DIR/deleted_files_$(date +%Y%m%d_%H%M%S).txt"
    printf "%s\n" "${FILES_TO_DELETE[@]}" > "$BACKUP_FILE"

    echo -e "${GREEN}Backup list saved to: $BACKUP_FILE${NC}"
    echo ""

    # Ask for confirmation
    echo -e "${YELLOW}About to delete ${#FILES_TO_DELETE[@]} duplicate image files and their JSON files.${NC}"
    echo -e "${YELLOW}This action CANNOT be undone!${NC}"
    echo ""
    read -p "Type 'DELETE' to confirm: " CONFIRM

    if [ "$CONFIRM" != "DELETE" ]; then
        echo -e "${BLUE}Deletion cancelled.${NC}"
        rm -f "$TEMP_FILE" "$DUPLICATES_FILE"
        exit 0
    fi

    echo ""
    echo -e "${RED}🗑️  Deleting files...${NC}"
    echo ""

    DELETED_COUNT=0
    ERROR_COUNT=0

    for file in "${FILES_TO_DELETE[@]}"; do
        # Delete image file
        if rm -f "$file" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Deleted: $file"
            ((DELETED_COUNT++))

            # Delete corresponding JSON file
            JSON_FILE="${file%.jpg}.json"
            if [ -f "$JSON_FILE" ]; then
                if rm -f "$JSON_FILE" 2>/dev/null; then
                    echo -e "${GREEN}✓${NC} Deleted: $JSON_FILE"
                    ((DELETED_COUNT++))
                fi
            fi
        else
            echo -e "${RED}✗${NC} Failed to delete: $file"
            ((ERROR_COUNT++))
        fi
    done

    echo ""
    echo -e "${BLUE}=================================================================================${NC}"
    echo -e "${BLUE}✅ DELETION COMPLETE${NC}"
    echo -e "${BLUE}=================================================================================${NC}"
    echo ""
    echo -e "Successfully deleted: ${GREEN}$DELETED_COUNT${NC} files"
    echo -e "Errors:              ${RED}$ERROR_COUNT${NC}"
    echo -e "Backup list:         ${CYAN}$BACKUP_FILE${NC}"
    echo ""
    echo -e "${GREEN}Duplicate removal complete!${NC}"
    echo ""
    echo "Run './check_status.sh' to see updated statistics."
    echo ""
fi

# Cleanup
rm -f "$TEMP_FILE" "$DUPLICATES_FILE"

echo -e "${BLUE}================================================================================${NC}"
