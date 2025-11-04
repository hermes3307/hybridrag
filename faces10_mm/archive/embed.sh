#!/bin/bash
################################################################################
# Embedding Management CLI - Shell Wrapper
################################################################################
#
# This script provides easy access to the embedding management CLI
#
# Usage:
#   ./embed.sh              # Interactive mode (shows stats and asks)
#   ./embed.sh --stats      # Show statistics only
#   ./embed.sh --auto       # Auto-embed all pending images
#   ./embed.sh --help       # Show help
#
################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_SCRIPT="$SCRIPT_DIR/embedding_manager_cli.py"

# Check if CLI script exists
if [ ! -f "$CLI_SCRIPT" ]; then
    echo -e "${RED}Error: embedding_manager_cli.py not found!${NC}"
    echo "Expected location: $CLI_SCRIPT"
    exit 1
fi

# Parse arguments
case "$1" in
    --stats|stats|-s)
        echo -e "${BLUE}📊 Displaying statistics only...${NC}"
        python3 "$CLI_SCRIPT" --stats-only
        ;;
    --auto|auto|-a)
        echo -e "${GREEN}🚀 Auto-embedding all pending images...${NC}"
        python3 "$CLI_SCRIPT" --auto-embed
        ;;
    --facenet)
        echo -e "${GREEN}🚀 Auto-embedding with FaceNet model...${NC}"
        python3 "$CLI_SCRIPT" --model facenet --auto-embed
        ;;
    --arcface)
        echo -e "${GREEN}🚀 Auto-embedding with ArcFace model...${NC}"
        python3 "$CLI_SCRIPT" --model arcface --auto-embed
        ;;
    --statistical)
        echo -e "${GREEN}🚀 Auto-embedding with Statistical model...${NC}"
        python3 "$CLI_SCRIPT" --model statistical --auto-embed
        ;;
    --help|-h|help)
        echo -e "${BLUE}Embedding Management CLI - Shell Wrapper${NC}"
        echo ""
        echo "Usage:"
        echo "  ./embed.sh              Interactive mode (default)"
        echo "  ./embed.sh --stats      Show statistics only"
        echo "  ./embed.sh --auto       Auto-embed all pending images"
        echo "  ./embed.sh --facenet    Auto-embed with FaceNet model"
        echo "  ./embed.sh --arcface    Auto-embed with ArcFace model"
        echo "  ./embed.sh --statistical Auto-embed with Statistical model"
        echo "  ./embed.sh --help       Show this help"
        echo ""
        echo "For full CLI help:"
        echo "  python3 embedding_manager_cli.py --help"
        ;;
    "")
        echo -e "${BLUE}🎯 Starting interactive mode...${NC}"
        python3 "$CLI_SCRIPT"
        ;;
    *)
        echo -e "${YELLOW}Unknown option: $1${NC}"
        echo "Use './embed.sh --help' for usage information"
        exit 1
        ;;
esac

exit $?
