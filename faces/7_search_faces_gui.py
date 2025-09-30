#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7️⃣  Unified Search Interface (GUI)
Search faces using semantic search, metadata filters, or both
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the unified search GUI
import importlib.util

spec = importlib.util.spec_from_file_location("search_gui", "102.unified_search_gui.py")
search_gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search_gui)

if __name__ == "__main__":
    print("="*70)
    print("7️⃣  UNIFIED SEARCH INTERFACE (GUI)")
    print("="*70)
    print()
    search_gui.main()