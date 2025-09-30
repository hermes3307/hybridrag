#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8️⃣  Validate Embeddings
Check and validate face embeddings in the database
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the original validator
from validate_embeddings import main

if __name__ == "__main__":
    print("="*70)
    print("8️⃣  VALIDATE EMBEDDINGS")
    print("="*70)
    print()
    main()