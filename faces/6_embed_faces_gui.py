#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6️⃣  Embed Faces into Vector DB (GUI)
Process face images and create embeddings - Graphical Interface
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the original GUI embedder
import importlib.util

spec = importlib.util.spec_from_file_location("embed_gui", "100.embedintoVectorgui.py")
embed_gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embed_gui)

if __name__ == "__main__":
    print("="*70)
    print("6️⃣  EMBED FACES INTO VECTOR DB (GUI)")
    print("="*70)
    print()
    embed_gui.main()