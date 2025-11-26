#!/bin/bash
# Image Search System - Startup Script
# This script activates the virtual environment and launches the GUI

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Image Search System - Starting${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}Working directory: ${NC}$SCRIPT_DIR"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo -e "${YELLOW}Please create it first with:${NC}"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo -e "${GREEN}✓${NC} Virtual environment found"

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

# Check if main GUI file exists
if [ ! -f "image.py" ]; then
    echo -e "${RED}Error: image.py not found!${NC}"
    echo -e "${YELLOW}Please ensure the GUI application file exists.${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} GUI application found (image.py)"
echo ""

# Check database connection (optional)
echo -e "${YELLOW}Checking database connection...${NC}"
if command -v psql &> /dev/null; then
    # Load database credentials from .env file
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi

    # Use PGPASSWORD environment variable to avoid password prompt
    if PGPASSWORD="${DB_PASSWORD:-postgres}" psql -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-pi}" -d "${DB_NAME:-image_vector}" -c "SELECT 1;" &> /dev/null; then
        echo -e "${GREEN}✓${NC} Database '${DB_NAME:-image_vector}' is accessible"
    else
        echo -e "${YELLOW}⚠${NC}  Database '${DB_NAME:-image_vector}' not accessible"
        echo -e "${YELLOW}   The application will try to connect when it starts${NC}"
    fi
else
    echo -e "${YELLOW}⚠${NC}  psql not found - skipping database check"
fi
echo ""

# Check if required Python packages are installed
echo -e "${YELLOW}Checking Python dependencies...${NC}"
python3 -c "import requests, numpy, PIL, psycopg2" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Core dependencies installed"
else
    echo -e "${RED}✗${NC} Missing dependencies!"
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi
echo ""

# Display system info
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  System Information${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "Python: $(python3 --version)"
echo -e "Virtual env: $VIRTUAL_ENV"
echo -e "Working dir: $(pwd)"
echo ""

# Launch the GUI
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Launching GUI Application${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${GREEN}Starting image.py...${NC}"
echo -e "${YELLOW}(Press Ctrl+C to stop)${NC}"
echo ""

# Run the GUI application
python3 image.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo -e "${BLUE}=========================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Application exited normally${NC}"
else
    echo -e "${RED}Application exited with error code: $EXIT_CODE${NC}"
fi
echo -e "${BLUE}=========================================${NC}"

exit $EXIT_CODE
