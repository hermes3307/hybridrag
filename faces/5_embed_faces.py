#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5️⃣  Embed Faces into Vector DB (CLI)
Process face images and create embeddings - Command Line Interface
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the original embedder
import importlib.util

spec = importlib.util.spec_from_file_location("embedder", "100.embedintoVector.py")
embedder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embedder)

if __name__ == "__main__":
    print("="*70)
    print("5️⃣  EMBED FACES INTO VECTOR DB (CLI)")
    print("="*70)
    print()
    embedder.main()