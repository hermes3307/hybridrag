#!/bin/bash
#
# This script runs the download_group_photos.py script to fetch sample images
# for testing face detection and recognition.
#
# It ensures that the necessary dependencies are installed from requirements.txt
# and activates the virtual environment if it exists.

# Exit on any error
set -e

# --- Configuration ---
PYTHON_CMD="python3"
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
SCRIPT_TO_RUN="download_group_photos.py"

# --- Functions ---

# Function to print messages
log() {
    echo "[INFO] $1"
}

# Function to print error messages
error() {
    echo "[ERROR] $1" >&2
}

# --- Main Execution ---

# 1. Check if the Python script exists
if [ ! -f "$SCRIPT_TO_RUN" ]; then
    error "The script '$SCRIPT_TO_RUN' was not found."
    exit 1
fi

# 2. Activate virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    log "Activating Python virtual environment from '$VENV_DIR'..."
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    PYTHON_CMD="$VENV_DIR/bin/python"
else
    log "Virtual environment not found at '$VENV_DIR'. Using system Python '$PYTHON_CMD'."
fi

# 3. Check for and install dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    log "Checking and installing dependencies from '$REQUIREMENTS_FILE'..."
    "$PYTHON_CMD" -m pip install -r "$REQUIREMENTS_FILE"
else
    log "No '$REQUIREMENTS_FILE' found. Assuming dependencies are already installed."
fi

# 4. Run the script
log "Executing the '$SCRIPT_TO_RUN' script..."
"$PYTHON_CMD" "$SCRIPT_TO_RUN"

log "Script finished."
