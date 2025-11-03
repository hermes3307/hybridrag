#!/bin/bash
################################################################################
# Cleanup macOS Metadata Files
################################################################################
#
# This script removes macOS resource fork files (._*) that may cause errors
# during embedding processing
#
# Usage:
#   ./cleanup_macos_files.sh [directory]
#
################################################################################

# Get the faces directory
if [ -n "$1" ]; then
    FACES_DIR="$1"
else
    FACES_DIR="./faces"
fi

echo "================================================================================"
echo "🧹 CLEANUP MACOS METADATA FILES"
echo "================================================================================"
echo ""
echo "Target directory: $FACES_DIR"
echo ""

# Check if directory exists
if [ ! -d "$FACES_DIR" ]; then
    echo "❌ Error: Directory '$FACES_DIR' does not exist"
    exit 1
fi

# Count ._ files
echo "🔍 Scanning for macOS metadata files (._*)..."
FILE_COUNT=$(find "$FACES_DIR" -type f -name "._*" | wc -l)

echo "Found: $FILE_COUNT files"
echo ""

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "✅ No macOS metadata files found. Directory is clean!"
    exit 0
fi

# Show some examples
echo "Examples of files to be deleted:"
find "$FACES_DIR" -type f -name "._*" | head -5
if [ "$FILE_COUNT" -gt 5 ]; then
    echo "... and $((FILE_COUNT - 5)) more"
fi
echo ""

# Ask for confirmation
read -p "⚠️  Delete all $FILE_COUNT macOS metadata files? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled. No files were deleted."
    exit 0
fi

# Delete the files
echo ""
echo "🗑️  Deleting files..."
DELETED=0
while IFS= read -r file; do
    rm -f "$file"
    DELETED=$((DELETED + 1))
    if [ $((DELETED % 1000)) -eq 0 ]; then
        echo "  Deleted $DELETED files..."
    fi
done < <(find "$FACES_DIR" -type f -name "._*")

echo ""
echo "================================================================================"
echo "✅ Cleanup completed!"
echo "================================================================================"
echo "Deleted: $DELETED files"
echo ""
