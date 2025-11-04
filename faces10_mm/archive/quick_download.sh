#!/bin/bash
#
# Quick Download - Simple one-liner for fast downloads
# Usage: ./quick_download.sh [num_faces] [threads] [output_dir]
#

NUM=${1:-100}
THREADS=${2:-8}
OUTPUT=${3:-faces_quick}

echo "╔════════════════════════════════════════════════════════╗"
echo "║           QUICK FACE DOWNLOAD                          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📥 Downloading $NUM faces with $THREADS threads"
echo "📁 Output: $OUTPUT"
echo ""

python3 bulk_download_cli.py -n "$NUM" -t "$THREADS" -o "$OUTPUT"

echo ""
echo "✅ Done! Files saved to: $OUTPUT"
echo "📊 View files: ls -lh $OUTPUT/"
echo "📈 File count: $(ls $OUTPUT/*.jpg 2>/dev/null | wc -l) images"
