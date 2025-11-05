#!/bin/bash

# Face Recognition System - PostgreSQL + pgvector Installation Script
# This script automates the installation and setup of all required components

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo "=========================================="
echo "Face Recognition System Installer"
echo "PostgreSQL + pgvector Setup"
echo "Multi-Model Support Edition"
echo "=========================================="
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration variables
DB_NAME="${POSTGRES_DB:-vector_db}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"

# Check if running on WSL
if grep -qi microsoft /proc/version; then
    log_info "Detected WSL environment"
    IS_WSL=true
else
    IS_WSL=false
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if PostgreSQL is running
check_postgres_running() {
    if sudo service postgresql status >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Step 1: Update system packages
log_info "Step 1: Updating system packages..."
sudo apt-get update -qq
log_success "System packages updated"

# Step 2: Install PostgreSQL
log_info "Step 2: Installing PostgreSQL..."
if command_exists psql; then
    log_warning "PostgreSQL is already installed"
    psql --version
else
    log_info "Installing PostgreSQL and dependencies..."
    sudo apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all
    log_success "PostgreSQL installed"
fi

# Step 3: Install build dependencies for pgvector
log_info "Step 3: Installing build dependencies..."
sudo apt-get install -y build-essential git
log_success "Build dependencies installed"

# Step 4: Install pgvector extension
log_info "Step 4: Installing pgvector extension..."
PGVECTOR_VERSION="v0.8.0"
TEMP_DIR="/tmp/pgvector_install_$$"

if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

log_info "Cloning pgvector repository (version $PGVECTOR_VERSION)..."
git clone --branch $PGVECTOR_VERSION https://github.com/pgvector/pgvector.git
cd pgvector

log_info "Building pgvector..."
make

log_info "Installing pgvector..."
sudo make install

cd "$SCRIPT_DIR"
rm -rf "$TEMP_DIR"
log_success "pgvector extension installed"

# Step 5: Start PostgreSQL service
log_info "Step 5: Starting PostgreSQL service..."
if check_postgres_running; then
    log_warning "PostgreSQL is already running"
else
    sudo service postgresql start
    sleep 2
    if check_postgres_running; then
        log_success "PostgreSQL service started"
    else
        log_error "Failed to start PostgreSQL service"
        exit 1
    fi
fi

# Step 6: Create database
log_info "Step 6: Creating database '$DB_NAME'..."
if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    log_warning "Database '$DB_NAME' already exists"
else
    sudo -u postgres psql -c "CREATE DATABASE $DB_NAME;"
    log_success "Database '$DB_NAME' created"
fi

# Step 7: Set PostgreSQL password
log_info "Step 7: Setting PostgreSQL password..."
sudo -u postgres psql -c "ALTER USER $DB_USER PASSWORD '$DB_PASSWORD';" 2>/dev/null || true
log_success "PostgreSQL password set"

# Step 8: Enable pgvector extension
log_info "Step 8: Enabling pgvector extension..."
if sudo -u postgres psql -d "$DB_NAME" -c "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';" | grep -q " 1"; then
    log_warning "pgvector extension already enabled"
else
    sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION vector;"
    log_success "pgvector extension enabled"
fi

# Step 9: Create database schema
log_info "Step 9: Creating database schema..."
if [ -f "schema.sql" ]; then
    # Check if faces table exists
    TABLE_EXISTS=$(sudo -u postgres psql -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='faces';")

    if [ "$TABLE_EXISTS" = "1" ]; then
        log_warning "Database schema already exists"
        read -p "Do you want to recreate the schema? This will DELETE ALL DATA! (yes/no): " -r
        echo
        if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log_warning "Recreating schema..."
            sudo -u postgres psql -d "$DB_NAME" -f schema.sql
            log_success "Database schema recreated"
        else
            log_info "Keeping existing schema"
        fi
    else
        sudo -u postgres psql -d "$DB_NAME" -f schema.sql
        log_success "Database schema created"
    fi
else
    log_error "schema.sql not found in current directory"
    exit 1
fi

# Step 10: Install Python system dependencies
log_info "Step 10: Installing Python system dependencies..."

# Install python3-tk for GUI support
if ! python3 -c "import tkinter" 2>/dev/null; then
    log_info "Installing python3-tk (required for GUI)..."
    sudo apt-get install -y python3-tk
    log_success "python3-tk installed"
else
    log_warning "python3-tk already installed"
fi

# Install python3-pil.imagetk for PIL/Tkinter integration
if ! python3 -c "from PIL import ImageTk" 2>/dev/null; then
    log_info "Installing python3-pil.imagetk (required for image display in GUI)..."
    sudo apt-get install -y python3-pil.imagetk
    log_success "python3-pil.imagetk installed"
else
    log_warning "python3-pil.imagetk already installed"
fi

# Check if pip3 is installed
if ! command_exists pip3 && ! python3 -m pip --version >/dev/null 2>&1; then
    log_warning "pip3 not found, installing python3-pip..."
    sudo apt-get install -y python3-pip
    log_success "pip3 installed"
fi

log_info "Step 11: Installing Python packages from requirements.txt..."
if [ -f "requirements.txt" ]; then
    # Check if we're in a virtual environment or need --break-system-packages
    if [ -z "$VIRTUAL_ENV" ]; then
        log_info "Installing with --break-system-packages flag..."
        # Use python3 -m pip for better compatibility
        if command_exists pip3; then
            pip3 install --break-system-packages -r requirements.txt
        else
            python3 -m pip install --break-system-packages -r requirements.txt
        fi
    else
        if command_exists pip3; then
            pip3 install -r requirements.txt
        else
            python3 -m pip install -r requirements.txt
        fi
    fi

    log_success "Python dependencies installed"
else
    log_error "requirements.txt not found"
    exit 1
fi

# Step 11: Create .env file if it doesn't exist
log_info "Step 11: Creating environment configuration..."
if [ -f ".env" ]; then
    log_warning ".env file already exists, creating .env.example instead"
    ENV_FILE=".env.example"
else
    ENV_FILE=".env"
fi

cat > "$ENV_FILE" << EOF
# PostgreSQL Database Configuration
POSTGRES_HOST=$DB_HOST
POSTGRES_PORT=$DB_PORT
POSTGRES_DB=$DB_NAME
POSTGRES_USER=$DB_USER
POSTGRES_PASSWORD=$DB_PASSWORD

# Application Settings
DB_TYPE=pgvector
FACES_DIR=./faces

# Multi-Model Configuration
# Supported models: facenet, arcface, vggface2, insightface, statistical
# Specify multiple models separated by commas to use multi-model support
EMBEDDING_MODELS=facenet,arcface
DEFAULT_SEARCH_MODEL=facenet

# Legacy single model support (deprecated, use EMBEDDING_MODELS instead)
# EMBEDDING_MODEL=statistical

# Connection Pool Settings
DB_MIN_CONNECTIONS=1
DB_MAX_CONNECTIONS=10

# Vector Search Settings
DEFAULT_SEARCH_LIMIT=10
DISTANCE_METRIC=cosine
EOF

log_success "Configuration file created: $ENV_FILE"

# Step 12: Create faces directory
log_info "Step 12: Creating faces directory..."
if [ ! -d "faces" ]; then
    mkdir -p faces
    log_success "Faces directory created"
else
    log_warning "Faces directory already exists"
fi

# Step 13: Test database connection
log_info "Step 13: Testing database connection..."
if [ -f "test_pgvector.py" ]; then
    log_info "Running connection test..."
    if python3 test_pgvector.py 2>&1 | grep -q "Test passed\|SUCCESS"; then
        log_success "Database connection test passed"
    else
        log_warning "Running basic connection test..."
        if sudo -u postgres psql -d "$DB_NAME" -c "SELECT version();" >/dev/null 2>&1; then
            log_success "Basic connection test passed"
        else
            log_error "Database connection test failed"
        fi
    fi
else
    log_info "Running basic connection test..."
    if sudo -u postgres psql -d "$DB_NAME" -c "SELECT version();" >/dev/null 2>&1; then
        log_success "Basic connection test passed"
    else
        log_error "Database connection test failed"
    fi
fi

# Step 14: Display installation summary
echo ""
echo "=========================================="
log_success "Installation completed successfully!"
echo "=========================================="
echo ""
echo "Database Configuration:"
echo "  Host: $DB_HOST"
echo "  Port: $DB_PORT"
echo "  Database: $DB_NAME"
echo "  User: $DB_USER"
echo "  Password: $DB_PASSWORD"
echo ""
echo "Configuration file: $ENV_FILE"
echo ""
echo "Next steps:"
echo "  1. Review and edit $ENV_FILE if needed"
echo "  2. Add face images to the 'faces' directory"
echo "  3. Run the application:"
echo "     python3 faces.py"
echo ""
echo "Testing:"
echo "  Run tests: python3 test_pgvector.py"
echo "  Inspect database: python3 inspect_database.py"
echo ""
echo "Useful commands:"
echo "  Start PostgreSQL: sudo service postgresql start"
echo "  Stop PostgreSQL:  sudo service postgresql stop"
echo "  Status:           sudo service postgresql status"
echo "  Connect to DB:    sudo -u postgres psql -d $DB_NAME"
echo ""
echo "=========================================="

# Optional: Open PostgreSQL CLI
read -p "Do you want to open PostgreSQL CLI to verify installation? (y/n): " -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Opening PostgreSQL CLI..."
    echo "Type '\dt' to list tables, '\q' to quit"
    sudo -u postgres psql -d "$DB_NAME"
fi

log_success "Setup complete! Your face recognition system is ready to use."
