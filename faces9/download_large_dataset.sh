#!/bin/bash
#
# Large Dataset Download - Download 1000+ faces with optimal settings
# Usage: ./download_large_dataset.sh [num_faces] [output_dir]
#

NUM=${1:-1000}
OUTPUT=${2:-faces_dataset_large}
THREADS=16  # Use more threads for large downloads

echo "╔════════════════════════════════════════════════════════╗"
echo "║         LARGE DATASET DOWNLOAD (HIGH PERFORMANCE)      ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📥 Downloading $NUM faces"
echo "⚡ Using $THREADS threads for maximum speed"
echo "📁 Output: $OUTPUT"
echo ""
echo "⏱️  Estimated time: ~$((NUM / 2 / 60)) minutes"
echo ""

# Check available disk space
AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
REQUIRED=$((NUM * 550 / 1024))  # ~550KB per face

echo "💾 Disk space check:"
echo "   Available: ${AVAILABLE}GB"
echo "   Required: ~${REQUIRED}GB"
echo ""

if [ "$AVAILABLE" -lt "$REQUIRED" ]; then
    echo "⚠️  WARNING: Low disk space!"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
        echo "Cancelled."
        exit 1
    fi
fi

echo "Starting download..."
echo ""

python3 bulk_download_cli.py -n "$NUM" -t "$THREADS" -o "$OUTPUT"

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                  DOWNLOAD COMPLETE!                    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Statistics:"
echo "   Images: $(ls $OUTPUT/*.jpg 2>/dev/null | wc -l)"
echo "   Size: $(du -sh $OUTPUT 2>/dev/null | cut -f1)"
echo "   Location: $(pwd)/$OUTPUT"
echo ""
