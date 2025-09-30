#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4️⃣  Download Faces (GUI)
Download face images with metadata - Graphical Interface
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the original GUI
import importlib.util

spec = importlib.util.spec_from_file_location("download_gui", "99.downbackground.gui.py")
download_gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_gui)

if __name__ == "__main__":
    print("="*70)
    print("4️⃣  DOWNLOAD FACES (GUI)")
    print("="*70)
    print()
    download_gui.main()