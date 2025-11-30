#!/bin/bash
################################################################################
# Batch Face Embedding Script
#
# Fast parallel embedding of all downloaded face images
# Supports multiple embedding models and configurable parallelism
#
# Usage:
#   ./embed_all.sh                    # Use config model, 4 workers
#   ./embed_all.sh facenet            # Use FaceNet model
#   ./embed_all.sh facenet 8          # Use FaceNet with 8 workers
#   ./embed_all.sh --list             # List available models
#   ./embed_all.sh --force facenet    # Re-embed all (including existing)
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL=""
WORKERS=4
FORCE_FLAG=""
CONFIG_FILE="system_config.json"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

show_usage() {
    cat << EOF
${GREEN}Batch Face Embedding Tool${NC}

${BLUE}Usage:${NC}
  $0 [OPTIONS] [MODEL] [WORKERS]

${BLUE}Arguments:${NC}
  MODEL          Embedding model (statistical, facenet, arcface, deepface, etc.)
  WORKERS        Number of parallel workers (default: 4)

${BLUE}Options:${NC}
  --list, -l         List available embedding models
  --force, -f        Re-embed all images (including already embedded)
  --config FILE      Use specific config file (default: system_config.json)
  --help, -h         Show this help message

${BLUE}Examples:${NC}
  # Embed all new images using model from config
  $0

  # Embed using FaceNet model
  $0 facenet

  # Use FaceNet with 8 parallel workers
  $0 facenet 8

  # Re-embed ALL images with ArcFace
  $0 --force arcface

  # List available models
  $0 --list

${BLUE}Available Models:${NC}
  statistical    Basic statistical features (always available, fast)
  facenet        Deep learning model (good accuracy)
  arcface        State-of-the-art (best accuracy)
  deepface       Multi-purpose framework
  vggface2       Deep CNN model
  openface       Lightweight model

${BLUE}Notes:${NC}
  - The script automatically skips already embedded images (use --force to override)
  - More workers = faster processing, but uses more CPU/memory
  - Recommended workers: 2-8 depending on your system
  - FaceNet/ArcFace require additional dependencies (see requirements.txt)

EOF
}

check_dependencies() {
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found. Please install Python 3"
        exit 1
    fi

    # Check if batch_embed.py exists
    if [ ! -f "batch_embed.py" ]; then
        print_error "batch_embed.py not found in current directory"
        exit 1
    fi

    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
}

activate_venv() {
    # Try to activate virtual environment if it exists
    if [ -d "venv" ]; then
        print_info "Activating virtual environment..."
        source venv/bin/activate
    elif [ -d "../venv" ]; then
        print_info "Activating virtual environment..."
        source ../venv/bin/activate
    fi
}

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --list|-l)
            check_dependencies
            activate_venv
            python3 batch_embed.py --list-models
            exit 0
            ;;
        --force|-f)
            FORCE_FLAG="--force"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            if [ -z "$MODEL" ]; then
                MODEL="$1"
            elif [ -z "$WORKERS" ] || [ "$WORKERS" = "4" ]; then
                WORKERS="$1"
            else
                print_error "Too many arguments"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

################################################################################
# Main Execution
################################################################################

print_header "BATCH FACE EMBEDDING"

# Check dependencies
check_dependencies

# Activate virtual environment if available
activate_venv

# Build command
CMD="python3 batch_embed.py --workers $WORKERS --config $CONFIG_FILE"

if [ -n "$MODEL" ]; then
    CMD="$CMD --model $MODEL"
fi

if [ -n "$FORCE_FLAG" ]; then
    CMD="$CMD $FORCE_FLAG"
fi

# Display settings
echo ""
print_info "Configuration:"
echo "  Config File: $CONFIG_FILE"
if [ -n "$MODEL" ]; then
    echo "  Model: $MODEL"
else
    echo "  Model: (from config file)"
fi
echo "  Workers: $WORKERS"
if [ -n "$FORCE_FLAG" ]; then
    echo "  Mode: Re-embed ALL images"
else
    echo "  Mode: Skip already embedded images"
fi
echo ""

# Confirm if force mode
if [ -n "$FORCE_FLAG" ]; then
    print_warning "You are about to re-embed ALL images (including existing ones)"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cancelled by user"
        exit 0
    fi
fi

# Run the embedding
print_info "Starting batch embedding..."
echo ""

if $CMD; then
    echo ""
    print_success "Batch embedding completed successfully!"
    exit 0
else
    EXIT_CODE=$?
    echo ""
    if [ $EXIT_CODE -eq 130 ]; then
        print_warning "Interrupted by user"
    else
        print_error "Batch embedding failed with exit code: $EXIT_CODE"
    fi
    exit $EXIT_CODE
fi
