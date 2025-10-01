#!/usr/bin/env python3
"""
Face Collector Module - Compatibility Wrapper
Re-exports classes from 3_collect_faces.py for backward compatibility
"""

# Import all classes from the refactored module
from importlib.util import spec_from_file_location, module_from_spec
import os

# Load 3_collect_faces.py module
spec = spec_from_file_location("collect_faces", os.path.join(os.path.dirname(__file__), "3_collect_faces.py"))
collect_faces = module_from_spec(spec)
spec.loader.exec_module(collect_faces)

# Re-export the classes for backward compatibility
FaceData = collect_faces.FaceData
FaceAnalyzer = collect_faces.FaceAnalyzer
FaceEmbedder = collect_faces.FaceEmbedder
FaceCollector = getattr(collect_faces, 'FaceCollector', None)

# Re-export any functions that might be used
if hasattr(collect_faces, 'process_faces'):
    process_faces = collect_faces.process_faces

__all__ = ['FaceData', 'FaceAnalyzer', 'FaceEmbedder', 'FaceCollector', 'process_faces']
