#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2️⃣  Database Information & Stats
Display comprehensive database information
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the original run_chroma_info
from run_chroma_info import main

if __name__ == "__main__":
    print("="*70)
    print("2️⃣  DATABASE INFORMATION & STATS")
    print("="*70)
    print()
    main()