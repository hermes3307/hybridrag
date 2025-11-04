#!/bin/bash
################################################################################
# Check Embedding Status - Quick status check
################################################################################
#
# This script shows current embedding statistics without making changes
#
# Usage:
#   ./check_status.sh
#
################################################################################

cd "$(dirname "$0")"

echo "================================================================================"
echo "📊 EMBEDDING STATUS CHECK"
echo "================================================================================"
echo ""

# Run the CLI in stats-only mode
python3 embedding_manager_cli.py --stats-only

exit $?
