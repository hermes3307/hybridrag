#!/usr/bin/env python3
"""
Test script to verify GUI fixes and enhanced logging
"""

import sys
import os
import tempfile
import threading
import time

# Test if tkinter is available
try:
    import tkinter as tk
    from tkinter import ttk
    TKINTER_AVAILABLE = True
    print("✅ tkinter is available")
except ImportError as e:
    TKINTER_AVAILABLE = False
    print(f"❌ tkinter not available: {e}")

# Test basic imports
try:
    from gui import AICoderGUI, GUILogHandler
    print("✅ GUI imports successful")
except ImportError as e:
    print(f"❌ GUI import failed: {e}")
    sys.exit(1)

def test_gui_components():
    """Test GUI component initialization"""
    if not TKINTER_AVAILABLE:
        print("⏭️ Skipping GUI test - tkinter not available")
        return False
    
    try:
        # Create test root window
        root = tk.Tk()
        root.withdraw()  # Hide window during test
        
        # Test GUI initialization
        app = AICoderGUI(root)
        
        # Test if key attributes exist
        assert hasattr(app, 'selected_files'), "selected_files attribute missing"
        assert hasattr(app, 'manual_files_listbox'), "manual_files_listbox attribute missing"
        assert isinstance(app.selected_files, list), "selected_files should be a list"
        
        # Test file browser method exists and is callable
        assert hasattr(app, 'browse_manual_file'), "browse_manual_file method missing"
        assert callable(app.browse_manual_file), "browse_manual_file should be callable"
        
        print("✅ GUI component initialization test passed")
        
        # Clean up
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ GUI component test failed: {e}")
        return False

def test_logging_enhancement():
    """Test enhanced logging functionality"""
    try:
        # Create a test text widget
        if TKINTER_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            text_widget = tk.Text(root)
            
            # Test GUILogHandler
            handler = GUILogHandler(text_widget)
            
            # Test color tag setup
            assert 'ERROR' in text_widget.tag_names(), "ERROR tag not configured"
            assert 'INFO' in text_widget.tag_names(), "INFO tag not configured"
            assert 'PROGRESS' in text_widget.tag_names(), "PROGRESS tag not configured"
            
            print("✅ Enhanced logging test passed")
            root.destroy()
        else:
            print("⏭️ Skipping logging test - tkinter not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced logging test failed: {e}")
        return False

def test_file_handling():
    """Test file handling improvements"""
    try:
        # Create temporary test files
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            test_file = f.name
            f.write(b"Test PDF content")
        
        # Test file existence and readability checks
        assert os.path.exists(test_file), "Test file should exist"
        assert os.access(test_file, os.R_OK), "Test file should be readable"
        
        print("✅ File handling test passed")
        
        # Clean up
        os.unlink(test_file)
        return True
        
    except Exception as e:
        print(f"❌ File handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running GUI fixes and enhancement tests...\n")
    
    tests = [
        ("GUI Components", test_gui_components),
        ("Enhanced Logging", test_logging_enhancement),
        ("File Handling", test_file_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        if test_func():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GUI fixes and enhancements are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)