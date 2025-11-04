#!/bin/bash
#
# Face Bulk Downloader - Convenient Shell Scripts
# Quick access to different download configurations
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║       HIGH-PERFORMANCE FACE BULK DOWNLOADER           ║"
    echo "║              Quick Start Menu                          ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print menu
print_menu() {
    echo -e "${GREEN}Available Options:${NC}"
    echo ""
    echo "  1) Quick Test (10 faces, no metadata)"
    echo "  2) Small Dataset (50 faces, no metadata)"
    echo "  3) Medium Dataset (100 faces, no metadata)"
    echo "  4) Large Dataset (500 faces, no metadata)"
    echo ""
    echo "  5) Quick Test with Metadata (10 faces + JSON)"
    echo "  6) Small Dataset with Metadata (50 faces + JSON)"
    echo "  7) Medium Dataset with Metadata (100 faces + JSON)"
    echo "  8) Large Dataset with Metadata (500 faces + JSON)"
    echo ""
    echo "  9) Custom Download (specify all parameters)"
    echo ""
    echo "  t) Test Download Speed"
    echo "  a) Analyze Metadata (from existing directory)"
    echo "  h) Show Help"
    echo "  q) Quit"
    echo ""
}

# Test download speed
test_speed() {
    echo -e "${YELLOW}Testing download speeds from both sources...${NC}"
    python3 test_download_speed.py
}

# Analyze metadata
analyze_metadata() {
    echo -e "${CYAN}Enter directory to analyze (default: faces_final):${NC}"
    read -r dir
    dir=${dir:-faces_final}

    if [ -d "$dir" ]; then
        echo -e "${YELLOW}Analyzing metadata in $dir...${NC}"
        python3 analyze_metadata.py "$dir"
    else
        echo -e "${RED}Directory $dir not found!${NC}"
    fi
}

# Show help
show_help() {
    echo -e "${CYAN}Command Line Usage:${NC}"
    echo ""
    echo "python3 bulk_download_cli.py [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n NUM         Number of faces to download (default: 100)"
    echo "  -t THREADS     Number of worker threads (default: 8)"
    echo "  -o OUTPUT      Output directory (default: faces_bulk)"
    echo "  -s SOURCE      Download source (thispersondoesnotexist/100k-faces)"
    echo "  -m, --metadata Generate JSON metadata files (slower)"
    echo "  --timeout SEC  Request timeout in seconds (default: 30)"
    echo ""
    echo "Examples:"
    echo "  python3 bulk_download_cli.py -n 100 -t 8"
    echo "  python3 bulk_download_cli.py -n 500 -t 16 -m"
    echo "  python3 bulk_download_cli.py -n 1000 -o dataset -m"
    echo ""
}

# Custom download
custom_download() {
    echo -e "${CYAN}Custom Download Configuration${NC}"
    echo ""

    echo -e "${YELLOW}Number of faces (default: 100):${NC}"
    read -r num
    num=${num:-100}

    echo -e "${YELLOW}Number of threads (default: 8):${NC}"
    read -r threads
    threads=${threads:-8}

    echo -e "${YELLOW}Output directory (default: faces_custom):${NC}"
    read -r output
    output=${output:-faces_custom}

    echo -e "${YELLOW}Generate metadata? (y/n, default: n):${NC}"
    read -r metadata

    echo -e "${YELLOW}Source (1=thispersondoesnotexist, 2=100k-faces, default: 1):${NC}"
    read -r source_choice
    source_choice=${source_choice:-1}

    if [ "$source_choice" = "2" ]; then
        source="100k-faces"
    else
        source="thispersondoesnotexist"
    fi

    # Build command
    cmd="python3 bulk_download_cli.py -n $num -t $threads -o $output -s $source"

    if [ "$metadata" = "y" ] || [ "$metadata" = "Y" ]; then
        cmd="$cmd -m"
    fi

    echo ""
    echo -e "${GREEN}Running: $cmd${NC}"
    echo ""

    eval "$cmd"
}

# Run download with parameters
run_download() {
    local num=$1
    local threads=$2
    local output=$3
    local metadata=$4

    echo -e "${GREEN}Downloading $num faces with $threads threads${NC}"
    echo -e "${CYAN}Output directory: $output${NC}"

    if [ "$metadata" = "yes" ]; then
        echo -e "${YELLOW}Generating JSON metadata files${NC}"
        python3 bulk_download_cli.py -n "$num" -t "$threads" -o "$output" -m
    else
        echo -e "${YELLOW}Images only (no metadata)${NC}"
        python3 bulk_download_cli.py -n "$num" -t "$threads" -o "$output"
    fi
}

# Main menu loop
main() {
    print_banner

    while true; do
        print_menu
        echo -e "${CYAN}Select an option:${NC}"
        read -r choice

        case $choice in
            1)
                run_download 10 4 "faces_quick_test" "no"
                ;;
            2)
                run_download 50 8 "faces_small" "no"
                ;;
            3)
                run_download 100 8 "faces_medium" "no"
                ;;
            4)
                run_download 500 16 "faces_large" "no"
                ;;
            5)
                run_download 10 4 "faces_quick_test_meta" "yes"
                ;;
            6)
                run_download 50 8 "faces_small_meta" "yes"
                ;;
            7)
                run_download 100 8 "faces_medium_meta" "yes"
                ;;
            8)
                run_download 500 16 "faces_large_meta" "yes"
                ;;
            9)
                custom_download
                ;;
            t|T)
                test_speed
                ;;
            a|A)
                analyze_metadata
                ;;
            h|H)
                show_help
                ;;
            q|Q)
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option. Please try again.${NC}"
                ;;
        esac

        echo ""
        echo -e "${YELLOW}Press Enter to continue...${NC}"
        read -r
        clear
        print_banner
    done
}

# Run main menu
main
