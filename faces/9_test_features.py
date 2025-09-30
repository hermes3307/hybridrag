#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9️⃣  Test Feature Extraction
Test and verify feature extraction on sample images
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the feature test
from test_feature_extraction import test_feature_extraction

if __name__ == "__main__":
    print("="*70)
    print("9️⃣  TEST FEATURE EXTRACTION")
    print("="*70)
    print()
    test_feature_extraction()