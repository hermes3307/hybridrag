#!/bin/bash
################################################################################
# Delete All Duplicates - Automatic Script
################################################################################
#
# This script automatically deletes ALL duplicate files without prompting.
# Use with caution!
#
# Usage:
#   ./delete_all_duplicates.sh
#
################################################################################

echo "================================================================================"
echo "🗑️  AUTOMATIC DUPLICATE DELETION"
echo "================================================================================"
echo ""

# Run the duplicate removal with automatic confirmation
cd "$(dirname "$0")"

echo "📊 Checking for duplicates..."
echo ""

# First show stats
./remove_duplicates.sh --stats

echo ""
echo "⚠️  AUTOMATIC DELETION MODE"
echo ""
echo "This will automatically delete all duplicate files!"
echo "Press Ctrl+C within 5 seconds to cancel..."
echo ""

# Countdown
for i in 5 4 3 2 1; do
    echo "Proceeding in $i..."
    sleep 1
done

echo ""
echo "🚀 Starting deletion..."
echo ""

# Run with automatic confirmation by piping "DELETE" to stdin
echo "DELETE" | ./remove_duplicates.sh --confirm

echo ""
echo "✅ Done! Check status:"
echo "   ./check_status.sh"
echo ""
