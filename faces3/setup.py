#!/usr/bin/env python3
"""
Setup Script for Integrated Face Processing System
Installs dependencies and initializes the system
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("🔧 Installing dependencies...")

    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")

    directories = [
        "./faces",
        "./chroma_db"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created: {directory}")

    print("✅ Directories created")

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")

    required_modules = [
        "chromadb",
        "numpy",
        "PIL",
        "requests",
        "cv2"
    ]

    failed_imports = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module}")
            failed_imports.append(module)

    if failed_imports:
        print(f"⚠️  Some modules failed to import: {failed_imports}")
        print("   The system will work with reduced functionality")
    else:
        print("✅ All modules imported successfully")

    return len(failed_imports) == 0

def initialize_database():
    """Initialize the ChromaDB database"""
    print("🗄️  Initializing database...")

    try:
        from core_backend import IntegratedFaceSystem
        system = IntegratedFaceSystem()
        if system.initialize():
            print("✅ Database initialized successfully")
            return True
        else:
            print("❌ Failed to initialize database")
            return False
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("🚀 INTEGRATED FACE PROCESSING SYSTEM SETUP")
    print("="*60)
    print()

    # Step 1: Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return False

    print()

    # Step 2: Create directories
    create_directories()
    print()

    # Step 3: Test imports
    imports_ok = test_imports()
    print()

    # Step 4: Initialize database
    if imports_ok:
        db_ok = initialize_database()
    else:
        print("⚠️  Skipping database initialization due to import failures")
        db_ok = False

    print()
    print("="*60)
    print("📋 SETUP SUMMARY")
    print("="*60)
    print(f"Dependencies: {'✅' if True else '❌'}")
    print(f"Directories: ✅")
    print(f"Imports: {'✅' if imports_ok else '⚠️'}")
    print(f"Database: {'✅' if db_ok else '❌'}")
    print()

    if imports_ok and db_ok:
        print("🎉 Setup completed successfully!")
        print("➡️  Run: python integrated_face_gui.py")
    else:
        print("⚠️  Setup completed with warnings")
        print("   Some features may not work properly")
        print("   Check the error messages above")

    print()
    return True

if __name__ == "__main__":
    main()