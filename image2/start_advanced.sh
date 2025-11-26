#!/bin/bash
# Image Search System - Advanced Startup Script
# This script provides options for setup, testing, and running the GUI

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Load environment variables from .env file
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set PGPASSWORD to avoid password prompts
export PGPASSWORD="${DB_PASSWORD:-postgres}"

# Functions
print_header() {
    echo ""
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC}  $1"
}

# Main menu
show_menu() {
    print_header "Image Search System - Control Panel"
    echo "Choose an option:"
    echo ""
    echo "  1) Start GUI Application"
    echo "  2) Setup Database"
    echo "  3) Install Dependencies"
    echo "  4) Test Download Sources"
    echo "  5) Test Downloader"
    echo "  6) Check System Status"
    echo "  7) View Logs"
    echo "  8) Clean Test Data"
    echo "  9) Exit"
    echo ""
    read -p "Enter option [1-9]: " choice
    echo ""
}

# Option 1: Start GUI
start_gui() {
    print_header "Starting GUI Application"

    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found!"
        print_info "Run option 3 (Install Dependencies) first"
        return 1
    fi

    source venv/bin/activate
    print_success "Virtual environment activated"

    if [ ! -f "image.py" ]; then
        print_error "image.py not found!"
        return 1
    fi

    print_info "Launching GUI..."
    python3 image.py
}

# Option 2: Setup Database
setup_database() {
    print_header "Database Setup"

    if ! command -v psql &> /dev/null; then
        print_error "PostgreSQL (psql) not found!"
        print_info "Please install PostgreSQL first"
        return 1
    fi

    print_info "Setting up database 'image_vector'..."

    # Check if database exists
    if psql -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-pi}" -lqt | cut -d \| -f 1 | grep -qw "${DB_NAME:-image_vector}"; then
        print_warning "Database '${DB_NAME:-image_vector}' already exists"
        read -p "Do you want to recreate it? (yes/no): " recreate
        if [ "$recreate" = "yes" ]; then
            print_info "Dropping existing database..."
            psql -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-pi}" -c "DROP DATABASE ${DB_NAME:-image_vector};"
        else
            print_info "Skipping database creation"
            return 0
        fi
    fi

    print_info "Creating database..."
    psql -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-pi}" -c "CREATE DATABASE ${DB_NAME:-image_vector};"
    print_success "Database created"

    print_info "Enabling pgvector extension..."
    psql -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-pi}" -d "${DB_NAME:-image_vector}" -c "CREATE EXTENSION IF NOT EXISTS vector;"
    print_success "pgvector enabled"

    if [ -f "schema.sql" ]; then
        print_info "Applying schema..."
        psql -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-pi}" -d "${DB_NAME:-image_vector}" -f schema.sql
        print_success "Schema applied"
    else
        print_warning "schema.sql not found - skipping schema setup"
    fi

    print_success "Database setup complete!"
}

# Option 3: Install Dependencies
install_dependencies() {
    print_header "Installing Dependencies"

    # Create venv if not exists
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_success "Virtual environment exists"
    fi

    source venv/bin/activate
    print_success "Virtual environment activated"

    print_info "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    print_success "pip upgraded"

    if [ -f "requirements.txt" ]; then
        print_info "Installing dependencies (this may take several minutes)..."
        pip install -r requirements.txt
        print_success "All dependencies installed!"
    else
        print_error "requirements.txt not found!"
        return 1
    fi
}

# Option 4: Test Download Sources
test_sources() {
    print_header "Testing Download Sources"

    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found! Run option 3 first."
        return 1
    fi

    source venv/bin/activate

    if [ -f "test_general_images.py" ]; then
        python3 test_general_images.py
    else
        print_error "test_general_images.py not found!"
    fi
}

# Option 5: Test Downloader
test_downloader() {
    print_header "Testing Image Downloader"

    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found! Run option 3 first."
        return 1
    fi

    source venv/bin/activate

    if [ -f "test_new_downloader.py" ]; then
        python3 test_new_downloader.py
    else
        print_error "test_new_downloader.py not found!"
    fi
}

# Option 6: Check System Status
check_status() {
    print_header "System Status Check"

    # Check venv
    if [ -d "venv" ]; then
        print_success "Virtual environment: OK"
    else
        print_error "Virtual environment: NOT FOUND"
    fi

    # Check database
    if command -v psql &> /dev/null; then
        if psql -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-pi}" -d "${DB_NAME:-image_vector}" -c "SELECT 1;" &> /dev/null 2>&1; then
            print_success "Database: OK (${DB_NAME:-image_vector} accessible)"

            # Get database stats
            COUNT=$(psql -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-pi}" -d "${DB_NAME:-image_vector}" -t -c "SELECT COUNT(*) FROM images;" 2>/dev/null | xargs)
            if [ ! -z "$COUNT" ]; then
                print_info "Total images in database: $COUNT"
            fi
        else
            print_warning "Database: image_vector not accessible"
        fi
    else
        print_warning "PostgreSQL: Not installed or not in PATH"
    fi

    # Check main files
    if [ -f "image.py" ]; then
        print_success "GUI application: OK (image.py found)"
    else
        print_warning "GUI application: image.py not found"
    fi

    if [ -f "core.py" ]; then
        print_success "Core module: OK (core.py found)"
    else
        print_error "Core module: core.py not found"
    fi

    if [ -f "schema.sql" ]; then
        print_success "Database schema: OK (schema.sql found)"
    else
        print_warning "Database schema: schema.sql not found"
    fi

    # Check images directory
    if [ -d "images" ]; then
        IMAGE_COUNT=$(find images -name "*.jpg" 2>/dev/null | wc -l)
        print_info "Downloaded images: $IMAGE_COUNT files in ./images/"
    else
        print_info "Images directory: Not created yet"
    fi

    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python: $PYTHON_VERSION"
    else
        print_error "Python 3 not found!"
    fi
}

# Option 7: View Logs
view_logs() {
    print_header "Recent Log Files"

    print_info "Looking for log files..."

    # Check for common log locations
    if [ -d "logs" ]; then
        print_success "Found logs directory"
        ls -lh logs/
    else
        print_info "No logs directory found"
    fi

    print_info "Recent downloaded images:"
    if [ -d "images" ]; then
        ls -lht images/*.jpg 2>/dev/null | head -10
    else
        print_info "No images directory"
    fi
}

# Option 8: Clean Test Data
clean_test_data() {
    print_header "Clean Test Data"

    print_warning "This will remove test downloads"
    read -p "Are you sure? (yes/no): " confirm

    if [ "$confirm" = "yes" ]; then
        if [ -d "test_downloads" ]; then
            print_info "Removing test_downloads directory..."
            rm -rf test_downloads
            print_success "Test data cleaned"
        else
            print_info "No test data found"
        fi
    else
        print_info "Cancelled"
    fi
}

# Main loop
while true; do
    show_menu

    case $choice in
        1)
            start_gui
            ;;
        2)
            setup_database
            ;;
        3)
            install_dependencies
            ;;
        4)
            test_sources
            ;;
        5)
            test_downloader
            ;;
        6)
            check_status
            ;;
        7)
            view_logs
            ;;
        8)
            clean_test_data
            ;;
        9)
            print_header "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid option. Please choose 1-9."
            ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
done
