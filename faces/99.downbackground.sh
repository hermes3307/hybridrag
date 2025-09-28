#!/bin/bash

# Background Face Download System Launcher
# Wrapper script for 99.downbackground.py with convenient options

set -e  # Exit on any error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/99.downbackground.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

# Function to show usage
show_usage() {
    print_header "🎭 Background Face Download System"
    echo "================================================"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "COMMANDS:"
    echo "  start           Start background downloading (default)"
    echo "  config          Create/edit configuration file"
    echo "  status          Show current download status"
    echo "  clean           Clean faces directory"
    echo "  help            Show this help message"
    echo
    echo "OPTIONS:"
    echo "  --faces-dir DIR     Faces storage directory (default: ./faces)"
    echo "  --delay SECONDS     Delay between downloads (default: 1.0)"
    echo "  --workers NUM       Number of worker threads (default: 3)"
    echo "  --limit NUM         Download limit (default: unlimited)"
    echo "  --no-duplicates     Disable duplicate checking"
    echo "  --config FILE       Configuration file path"
    echo
    echo "EXAMPLES:"
    echo "  $0                              # Start with default settings"
    echo "  $0 start --limit 100            # Download 100 faces"
    echo "  $0 start --faces-dir ./myfaces  # Use custom directory"
    echo "  $0 config                       # Create configuration file"
    echo "  $0 clean                        # Clean faces directory"
    echo
    echo "CONTROLS:"
    echo "  Ctrl+C             Stop downloading gracefully"
    echo "  SIGTERM            Graceful shutdown"
    echo
}

# Function to check Python script exists
check_python_script() {
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        print_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
}

# Function to check Python availability
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
}

# Function to create configuration
create_config() {
    print_info "Creating default configuration..."
    check_python_script
    python3 "$PYTHON_SCRIPT" --create-config
    print_success "Configuration created successfully"
}

# Function to show download status
show_status() {
    local faces_dir="${1:-./faces}"

    print_header "📊 Download Status"
    echo "===================="

    if [[ -d "$faces_dir" ]]; then
        local face_count=$(find "$faces_dir" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
        local dir_size=$(du -sh "$faces_dir" 2>/dev/null | cut -f1)

        print_info "Faces directory: $faces_dir"
        print_info "Total faces: $face_count"
        print_info "Directory size: $dir_size"

        if [[ $face_count -gt 0 ]]; then
            print_info "Latest faces:"
            find "$faces_dir" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | \
                head -5 | while read -r file; do
                echo "  - $(basename "$file")"
            done
        fi
    else
        print_warning "Faces directory does not exist: $faces_dir"
    fi

    # Check if download is currently running
    if pgrep -f "99.downbackground.py" > /dev/null; then
        print_info "Download process is currently running"
    else
        print_info "No download process currently running"
    fi
}

# Function to clean faces directory
clean_faces() {
    local faces_dir="${1:-./faces}"

    if [[ -d "$faces_dir" ]]; then
        local face_count=$(find "$faces_dir" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)

        if [[ $face_count -eq 0 ]]; then
            print_info "Faces directory is already empty"
            return
        fi

        print_warning "This will delete $face_count face images from $faces_dir"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            find "$faces_dir" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | xargs rm -f
            print_success "Cleaned $face_count face images"
        else
            print_info "Clean operation cancelled"
        fi
    else
        print_warning "Faces directory does not exist: $faces_dir"
    fi
}

# Function to start download
start_download() {
    local args=("$@")

    print_header "🚀 Starting Background Face Download"
    echo "====================================="

    check_python_script
    check_python

    # Show startup info
    print_info "Script: $PYTHON_SCRIPT"
    print_info "Arguments: ${args[*]}"
    print_info "Press Ctrl+C to stop downloading"
    echo

    # Start the Python script
    python3 "$PYTHON_SCRIPT" "${args[@]}"
}

# Function to handle cleanup on exit
cleanup() {
    print_info "Cleaning up background processes..."
    # Kill any background python processes for this script
    pkill -f "99.downbackground.py" 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Main script logic
main() {
    local command="start"
    local python_args=()

    # Parse command
    if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
        command="$1"
        shift
    fi

    # Handle commands
    case "$command" in
        "help"|"-h"|"--help")
            show_usage
            exit 0
            ;;
        "config")
            create_config
            exit 0
            ;;
        "status")
            # Extract faces directory from arguments if provided
            local faces_dir="./faces"
            while [[ $# -gt 0 ]]; do
                case $1 in
                    --faces-dir)
                        faces_dir="$2"
                        shift 2
                        ;;
                    *)
                        shift
                        ;;
                esac
            done
            show_status "$faces_dir"
            exit 0
            ;;
        "clean")
            # Extract faces directory from arguments if provided
            local faces_dir="./faces"
            while [[ $# -gt 0 ]]; do
                case $1 in
                    --faces-dir)
                        faces_dir="$2"
                        shift 2
                        ;;
                    *)
                        shift
                        ;;
                esac
            done
            clean_faces "$faces_dir"
            exit 0
            ;;
        "start")
            # Continue to start download with remaining arguments
            python_args=("$@")
            ;;
        *)
            # Treat unknown command as start with all arguments
            python_args=("$command" "$@")
            ;;
    esac

    # Start download
    start_download "${python_args[@]}"
}

# Run main function with all arguments
main "$@"