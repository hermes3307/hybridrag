#!/bin/bash
#
# Download 10,000 Faces - Quick Start Script for Ubuntu
#
# Usage: ./download_10k_faces.sh [threads] [output_directory]
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
THREADS=${1:-16}
OUTPUT=${2:-faces_10k}
NUM_FACES=10000

# Print banner
echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         DOWNLOAD 10,000 FACES - UBUNTU QUICK START         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# System check
echo -e "${YELLOW}[1/5] System Check${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CPU_CORES=$(nproc)
FREE_MEM=$(free -h | grep Mem | awk '{print $4}')
FREE_DISK=$(df -h . | tail -1 | awk '{print $4}')

echo "  CPU Cores:    $CPU_CORES"
echo "  Free Memory:  $FREE_MEM"
echo "  Free Disk:    $FREE_DISK"
echo "  Threads:      $THREADS"
echo "  Output Dir:   $OUTPUT"
echo ""

# Check disk space (need ~6 GB)
DISK_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$DISK_GB" -lt 7 ]; then
    echo -e "${RED}⚠️  WARNING: Low disk space!${NC}"
    echo "  Required: ~6 GB"
    echo "  Available: ${DISK_GB} GB"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

# Check if Python packages are installed
echo -e "${YELLOW}[2/5] Checking Dependencies${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

MISSING_DEPS=0

for pkg in requests rich psutil; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $pkg installed"
    else
        echo -e "  ${RED}✗${NC} $pkg NOT installed"
        MISSING_DEPS=1
    fi
done

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}Installing missing packages...${NC}"
    pip3 install requests rich psutil pillow opencv-python numpy
    echo ""
fi

# Estimate time and size
echo ""
echo -e "${YELLOW}[3/5] Download Estimates${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Calculate estimated time based on threads
if [ $THREADS -ge 16 ]; then
    EST_TIME="45-60 minutes"
    SPEED="~3.5 faces/sec"
elif [ $THREADS -ge 12 ]; then
    EST_TIME="55-75 minutes"
    SPEED="~2.8 faces/sec"
elif [ $THREADS -ge 8 ]; then
    EST_TIME="70-90 minutes"
    SPEED="~2.3 faces/sec"
else
    EST_TIME="90-120 minutes"
    SPEED="~1.5 faces/sec"
fi

echo "  Faces to download: $NUM_FACES"
echo "  Estimated time:    $EST_TIME"
echo "  Estimated speed:   $SPEED"
echo "  Estimated size:    ~5.5 GB"
echo ""

# Confirm before starting
echo -e "${YELLOW}[4/5] Confirmation${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  About to download $NUM_FACES faces"
echo "  Using $THREADS threads"
echo "  Output: $OUTPUT/"
echo ""

read -p "Start download? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Start download
echo ""
echo -e "${GREEN}[5/5] Starting Download${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

START_TIME=$(date +%s)

python3 bulk_download_cli.py -n $NUM_FACES -t $THREADS -o $OUTPUT

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

# Final summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    DOWNLOAD COMPLETE!                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

DOWNLOADED=$(ls $OUTPUT/*.jpg 2>/dev/null | wc -l)
SIZE=$(du -sh $OUTPUT 2>/dev/null | cut -f1)

echo "  ✅ Files downloaded: $DOWNLOADED"
echo "  📦 Total size:       $SIZE"
echo "  ⏱️  Total time:       ${MINUTES}m ${SECONDS}s"
echo "  📁 Location:         $(pwd)/$OUTPUT"
echo ""

# Suggest next steps
echo -e "${CYAN}Next Steps:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1. Generate metadata (optional):"
echo "     python3 generate_missing_metadata.py -d $OUTPUT -y"
echo ""
echo "  2. Analyze dataset:"
echo "     python3 analyze_metadata.py $OUTPUT"
echo ""
echo "  3. Check files:"
echo "     ls $OUTPUT/*.jpg | wc -l"
echo ""
echo -e "${GREEN}Done! 🎉${NC}"
echo ""
