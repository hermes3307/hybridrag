#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Face Recognition System - Main Launcher
Quick access to all system components (1-10)
"""

import sys
import os
import io
import subprocess

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def print_banner():
    """Print system banner"""
    print("="*70)
    print("🎭 FACE RECOGNITION SYSTEM - LAUNCHER")
    print("="*70)
    print()

def print_menu():
    """Print main menu"""
    print("📋 AVAILABLE COMPONENTS:")
    print()

    print("  ⚙️  SETUP & CONFIGURATION")
    print("  1️⃣  - Setup ChromaDB Database")
    print("  2️⃣  - Database Information & Stats")
    print()

    print("  📥 DATA COLLECTION")
    print("  3️⃣  - Download Faces (CLI)")
    print("  4️⃣  - Download Faces (GUI)")
    print()

    print("  🔮 EMBEDDING & INDEXING")
    print("  5️⃣  - Embed Faces into Vector DB (CLI)")
    print("  6️⃣  - Embed Faces into Vector DB (GUI)")
    print()

    print("  🔍 SEARCH & QUERY")
    print("  7️⃣  - Unified Search Interface (GUI)")
    print()

    print("  🛠️  UTILITIES")
    print("  8️⃣  - Validate Embeddings")
    print("  9️⃣  - Test Feature Extraction")
    print("  🔟 - Complete Pipeline Demo")
    print()

    print("  0️⃣  - Exit")
    print()

def run_script(script_name: str, description: str):
    """Run a Python script"""
    print()
    print(f"🚀 Launching: {description}")
    print("-"*70)

    script_path = os.path.join(os.path.dirname(__file__), script_name)

    if not os.path.exists(script_path):
        print(f"❌ Error: {script_name} not found!")
        input("\nPress Enter to continue...")
        return

    try:
        # Run the script
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Script exited with error code: {e.returncode}")
    except KeyboardInterrupt:
        print("\n⚠️  Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running script: {e}")

    print()
    input("Press Enter to continue...")

def main():
    """Main launcher loop"""
    while True:
        # Clear screen (cross-platform)
        os.system('cls' if os.name == 'nt' else 'clear')

        print_banner()
        print_menu()

        choice = input("👉 Select component (0-10): ").strip()

        if choice == "0":
            print("\n👋 Goodbye!")
            break

        elif choice == "1":
            run_script("1_setup_database.py", "Setup ChromaDB Database")

        elif choice == "2":
            run_script("2_database_info.py", "Database Information & Stats")

        elif choice == "3":
            run_script("3_download_faces.py", "Download Faces (CLI)")

        elif choice == "4":
            run_script("4_download_faces_gui.py", "Download Faces (GUI)")

        elif choice == "5":
            run_script("5_embed_faces.py", "Embed Faces into Vector DB (CLI)")

        elif choice == "6":
            run_script("6_embed_faces_gui.py", "Embed Faces into Vector DB (GUI)")

        elif choice == "7":
            run_script("7_search_faces_gui.py", "Unified Search Interface (GUI)")

        elif choice == "8":
            run_script("8_validate_embeddings.py", "Validate Embeddings")

        elif choice == "9":
            run_script("9_test_features.py", "Test Feature Extraction")

        elif choice == "10":
            run_script("10_complete_demo.py", "Complete Pipeline Demo")

        else:
            print("\n❌ Invalid choice! Please select 0-10.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")
        sys.exit(1)