#!/bin/bash
# Unix/Linux/Mac Shell Launcher for Face Recognition System
# Usage: ./run.sh [1-10] or just ./run.sh for menu

# Make script executable: chmod +x run.sh

if [ $# -eq 0 ]; then
    # No argument - show launcher menu
    python3 0_launcher.py
    exit 0
fi

# Map number to script
case "$1" in
    0)
        script="0_launcher.py"
        ;;
    1)
        script="1_setup_database.py"
        ;;
    2)
        script="2_database_info.py"
        ;;
    3)
        script="3_download_faces.py"
        ;;
    4)
        script="4_download_faces_gui.py"
        ;;
    5)
        script="5_embed_faces.py"
        ;;
    6)
        script="6_embed_faces_gui.py"
        ;;
    7)
        script="7_search_faces_gui.py"
        ;;
    8)
        script="8_validate_embeddings.py"
        ;;
    9)
        script="9_test_features.py"
        ;;
    10)
        script="10_complete_demo.py"
        ;;
    *)
        echo "Invalid component number: $1"
        echo "Usage: ./run.sh [0-10]"
        echo "  0 - Launcher menu"
        echo "  1 - Setup database"
        echo "  2 - Database info"
        echo "  3 - Download faces (CLI)"
        echo "  4 - Download faces (GUI)"
        echo "  5 - Embed faces (CLI)"
        echo "  6 - Embed faces (GUI)"
        echo "  7 - Search faces (GUI)"
        echo "  8 - Validate embeddings"
        echo "  9 - Test features"
        echo "  10 - Complete demo"
        exit 1
        ;;
esac

# Run the script
shift  # Remove first argument
python3 "$script" "$@"