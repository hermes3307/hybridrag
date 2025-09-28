#!/bin/bash

# Background Face Download System - GUI Launcher
# Wrapper script for 99.downbackground.gui.py with system checks and setup

set -e  # Exit on any error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_GUI_SCRIPT="$SCRIPT_DIR/99.downbackground.gui.py"
PYTHON_CORE_SCRIPT="$SCRIPT_DIR/99.downbackground.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Function to show usage
show_usage() {
    print_header "🎭 Background Face Download System - GUI"
    echo "=============================================="
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "COMMANDS:"
    echo "  gui             Launch GUI interface (default)"
    echo "  check           Check system requirements"
    echo "  install-deps    Install Python dependencies"
    echo "  help            Show this help message"
    echo
    echo "OPTIONS:"
    echo "  --debug         Run with debug output"
    echo "  --no-check      Skip system requirements check"
    echo
    echo "EXAMPLES:"
    echo "  $0                      # Launch GUI"
    echo "  $0 gui                  # Launch GUI explicitly"
    echo "  $0 check                # Check requirements"
    echo "  $0 install-deps         # Install dependencies"
    echo
    echo "GUI FEATURES:"
    echo "  • Real-time download statistics"
    echo "  • Configuration management"
    echo "  • Start/Stop controls"
    echo "  • Duplicate checking toggle"
    echo "  • Download progress tracking"
    echo "  • Live log display"
    echo
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python availability
check_python() {
    print_step "Checking Python installation..."

    if ! command_exists python3; then
        print_error "Python 3 is required but not installed"
        echo
        echo "Please install Python 3:"
        echo "  • macOS: brew install python3"
        echo "  • Ubuntu/Debian: sudo apt-get install python3"
        echo "  • CentOS/RHEL: sudo yum install python3"
        return 1
    fi

    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_success "Python $python_version found"

    # Check if tkinter is available
    if ! python3 -c "import tkinter" 2>/dev/null; then
        print_error "tkinter (GUI toolkit) is not available"
        echo
        echo "Please install tkinter:"
        echo "  • macOS: Usually included with Python"
        echo "  • Ubuntu/Debian: sudo apt-get install python3-tk"
        echo "  • CentOS/RHEL: sudo yum install tkinter"
        return 1
    fi

    print_success "tkinter GUI toolkit available"
    return 0
}

# Function to check required Python modules
check_python_modules() {
    print_step "Checking Python modules..."

    local required_modules=("requests" "PIL" "numpy")
    local missing_modules=()

    for module in "${required_modules[@]}"; do
        if ! python3 -c "import $module" 2>/dev/null; then
            missing_modules+=("$module")
        fi
    done

    if [[ ${#missing_modules[@]} -gt 0 ]]; then
        print_warning "Missing Python modules: ${missing_modules[*]}"
        echo
        echo "Install missing modules with:"
        echo "  pip3 install ${missing_modules[*]}"
        echo
        echo "Or run: $0 install-deps"
        return 1
    fi

    print_success "All required Python modules available"
    return 0
}

# Function to check required files
check_required_files() {
    print_step "Checking required files..."

    local required_files=("$PYTHON_GUI_SCRIPT" "$PYTHON_CORE_SCRIPT")
    local missing_files=()

    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$(basename "$file")")
        fi
    done

    if [[ ${#missing_files[@]} -gt 0 ]]; then
        print_error "Missing required files: ${missing_files[*]}"
        echo
        echo "Please ensure all files are in the same directory:"
        echo "  • 99.downbackground.py      (Core download system)"
        echo "  • 99.downbackground.gui.py  (GUI interface)"
        echo "  • 99.downbackground.gui.sh  (This launcher)"
        return 1
    fi

    print_success "All required files found"
    return 0
}

# Function to install Python dependencies
install_dependencies() {
    print_header "🔧 Installing Python Dependencies"
    echo "=================================="

    local packages=("requests" "Pillow" "numpy")

    print_step "Installing required packages..."

    for package in "${packages[@]}"; do
        print_info "Installing $package..."
        if pip3 install "$package"; then
            print_success "$package installed successfully"
        else
            print_error "Failed to install $package"
            return 1
        fi
    done

    print_success "All dependencies installed successfully!"
    return 0
}

# Function to perform system check
system_check() {
    print_header "🔍 System Requirements Check"
    echo "============================="

    local checks_passed=0
    local total_checks=3

    # Check Python
    if check_python; then
        ((checks_passed++))
    fi

    # Check Python modules
    if check_python_modules; then
        ((checks_passed++))
    fi

    # Check required files
    if check_required_files; then
        ((checks_passed++))
    fi

    echo
    print_info "Check Results: $checks_passed/$total_checks passed"

    if [[ $checks_passed -eq $total_checks ]]; then
        print_success "✅ System is ready to run the GUI!"
        return 0
    else
        print_error "❌ System requirements not met"
        echo
        echo "To fix issues:"
        echo "  • Install missing dependencies: $0 install-deps"
        echo "  • Check the error messages above for specific solutions"
        return 1
    fi
}

# Function to launch GUI
launch_gui() {
    local debug_mode="$1"

    print_header "🚀 Launching GUI Interface"
    echo "=========================="

    print_step "Starting 99.downbackground.gui.py..."

    if [[ "$debug_mode" == "true" ]]; then
        print_info "Debug mode enabled"
        python3 "$PYTHON_GUI_SCRIPT" --debug
    else
        python3 "$PYTHON_GUI_SCRIPT"
    fi
}

# Function to show system info
show_system_info() {
    print_header "💻 System Information"
    echo "====================="

    echo "OS: $(uname -s) $(uname -r)"
    echo "Architecture: $(uname -m)"
    echo "Python: $(python3 --version 2>/dev/null || echo 'Not found')"
    echo "Script Directory: $SCRIPT_DIR"
    echo "Current Directory: $(pwd)"

    if [[ -f "$PYTHON_CORE_SCRIPT" ]]; then
        echo "Core Script: ✅ Found"
    else
        echo "Core Script: ❌ Missing"
    fi

    if [[ -f "$PYTHON_GUI_SCRIPT" ]]; then
        echo "GUI Script: ✅ Found"
    else
        echo "GUI Script: ❌ Missing"
    fi
}

# Function to handle cleanup on exit
cleanup() {
    print_info "Cleaning up..."
    # Kill any background python processes for GUI
    pkill -f "99.downbackground.gui.py" 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Main script logic
main() {
    local command="gui"
    local debug_mode="false"
    local skip_check="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            gui)
                command="gui"
                shift
                ;;
            check)
                command="check"
                shift
                ;;
            install-deps)
                command="install-deps"
                shift
                ;;
            help|-h|--help)
                command="help"
                shift
                ;;
            --debug)
                debug_mode="true"
                shift
                ;;
            --no-check)
                skip_check="true"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Handle commands
    case "$command" in
        "help")
            show_usage
            exit 0
            ;;
        "check")
            show_system_info
            echo
            system_check
            exit $?
            ;;
        "install-deps")
            install_dependencies
            exit $?
            ;;
        "gui")
            # Perform system check unless skipped
            if [[ "$skip_check" != "true" ]]; then
                if ! system_check; then
                    echo
                    print_error "System check failed. GUI cannot be launched."
                    echo
                    echo "Fix the issues above or use --no-check to skip validation"
                    exit 1
                fi
                echo
            fi

            # Launch GUI
            launch_gui "$debug_mode"
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Show startup banner
print_header "🎭 Background Face Downloader - GUI Launcher"
echo "============================================="
echo

# Run main function with all arguments
main "$@"