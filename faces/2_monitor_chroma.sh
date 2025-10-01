#!/bin/bash

echo "🔍 ChromaDB Vector Database Monitor"
echo "===================================="
echo ""
echo "📊 Starting real-time monitoring of ChromaDB..."
echo ""
echo "💡 This monitor shows:"
echo "   • Total embeddings in database"
echo "   • Processing rate (embeddings/second)"
echo "   • Data distribution (age, skin tone, quality)"
echo "   • Collection information"
echo "   • Real-time progress"
echo ""
echo "🎯 Usage:"
echo "   Press Ctrl+C to exit"
echo ""
echo "⚙️  Options:"
echo "   Default refresh: 1 second"
echo "   Custom refresh: ./monitor_chroma.sh 0.5  (for 0.5 seconds)"
echo ""
echo "Starting in 2 seconds..."
sleep 2

# Get refresh rate from argument or use default
REFRESH_RATE=${1:-1.0}

# Run the monitor
python3 2_monitor_chroma.py --refresh "$REFRESH_RATE"

echo ""
echo "✅ Monitoring session ended"
echo ""
