#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3️⃣  Download Faces (CLI)
Download face images with metadata - Command Line Interface
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the original downloader
import importlib.util

spec = importlib.util.spec_from_file_location("downloader", "99.downbackground.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)

if __name__ == "__main__":
    print("="*70)
    print("3️⃣  DOWNLOAD FACES (CLI)")
    print("="*70)
    print()
    downloader.main()