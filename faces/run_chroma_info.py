#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple ChromaDB Information Display Script
Shows database info without interactive menu
"""

import sys
import os
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add the current directory to Python path to import our setup module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from setup_chroma import initialize_chromadb, display_database_info, check_chromadb_installation

def main():
    """Main function to display ChromaDB information"""
    print("🔍 ChromaDB Database Information")
    print("="*50)

    # Check if ChromaDB is installed
    if not check_chromadb_installation():
        print("❌ ChromaDB not installed. Run setup_chroma.py first.")
        sys.exit(1)

    # Initialize ChromaDB
    client = initialize_chromadb()
    if not client:
        print("❌ Failed to initialize ChromaDB client")
        sys.exit(1)

    # Display database information
    display_database_info(client)

if __name__ == "__main__":
    main()