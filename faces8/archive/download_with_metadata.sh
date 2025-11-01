#!/bin/bash
#
# Download with Metadata - Download faces with full JSON analysis
# Usage: ./download_with_metadata.sh [num_faces] [threads] [output_dir]
#

NUM=${1:-100}
THREADS=${2:-8}
OUTPUT=${3:-faces_metadata}

echo "╔════════════════════════════════════════════════════════╗"
echo "║      FACE DOWNLOAD WITH METADATA ANALYSIS              ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📥 Downloading $NUM faces with $THREADS threads"
echo "📁 Output: $OUTPUT"
echo "🔬 Generating JSON metadata with face analysis"
echo ""

python3 bulk_download_cli.py -n "$NUM" -t "$THREADS" -o "$OUTPUT" -m

echo ""
echo "✅ Done! Files saved to: $OUTPUT"
echo "📊 Image files: $(ls $OUTPUT/*.jpg 2>/dev/null | wc -l)"
echo "📄 JSON files: $(ls $OUTPUT/*.json 2>/dev/null | wc -l)"
echo ""
echo "Analyze metadata with:"
echo "  python3 analyze_metadata.py $OUTPUT"
