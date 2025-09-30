#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1️⃣  Setup ChromaDB Database
Installs ChromaDB and initializes the database
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the original setup_chroma
from setup_chroma import main

if __name__ == "__main__":
    print("="*70)
    print("1️⃣  SETUP CHROMADB DATABASE")
    print("="*70)
    print()
    main()