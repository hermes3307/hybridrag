#!/usr/bin/env python3
"""
Simple script to run file_manager.py without OAuth2 authentication.
This will run the Google Drive Permission Manager in demo mode.
"""

import sys
import os

# Add the current directory to the path so we can import file_manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from file_manager import GoogleDrivePermissionManager

def main():
    print("=" * 60)
    print("  GOOGLE DRIVE PERMISSION MANAGER - NO OAUTH2 MODE")
    print("=" * 60)
    print()
    print("This will run the file manager in demo mode without requiring")
    print("OAuth2 authentication or Google API credentials.")
    print()
    print("Features available in demo mode:")
    print("- View sample shared files")
    print("- Simulate making files private")
    print("- Test the GUI interface")
    print()
    
    # Create the app with force_demo_mode=True
    app = GoogleDrivePermissionManager(force_demo_mode=True)
    
    print("Starting application in demo mode...")
    app.run()

if __name__ == "__main__":
    main() 