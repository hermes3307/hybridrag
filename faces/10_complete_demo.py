#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔟 Complete Pipeline Demo
Demonstrates the complete face recognition pipeline
"""

import sys
import os
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_step(step_num: int, description: str):
    """Print step header"""
    print()
    print("="*70)
    print(f"STEP {step_num}: {description}")
    print("="*70)
    print()

def wait_for_user():
    """Wait for user to continue"""
    input("\n👉 Press Enter to continue to next step...")

def main():
    """Run complete demo pipeline"""
    print("="*70)
    print("🔟 COMPLETE FACE RECOGNITION PIPELINE DEMO")
    print("="*70)
    print()
    print("This demo will guide you through the complete pipeline:")
    print("  1. Setup database")
    print("  2. Download sample faces")
    print("  3. Embed faces into vector database")
    print("  4. Perform similarity search")
    print("  5. View results")
    print()

    choice = input("🚀 Ready to start? (y/n): ").strip().lower()
    if choice != 'y':
        print("Demo cancelled.")
        return

    # Step 1: Check database
    print_step(1, "Check Database Setup")
    try:
        from setup_chroma import check_chromadb_installation
        if not check_chromadb_installation():
            print("❌ ChromaDB not installed!")
            print("👉 Please run: python 1_setup_database.py")
            return
        print("✅ Database ready!")
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return

    wait_for_user()

    # Step 2: Check for face files
    print_step(2, "Check Face Files")
    faces_dir = "./faces"
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)

    face_count = len([f for f in os.listdir(faces_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"📁 Found {face_count} face images in {faces_dir}")

    if face_count == 0:
        print("\n⚠️  No face images found!")
        print("👉 Please run: python 3_download_faces.py or python 4_download_faces_gui.py")
        return

    wait_for_user()

    # Step 3: Check embeddings
    print_step(3, "Check Vector Embeddings")
    try:
        from face_database import FaceDatabase
        db = FaceDatabase()
        stats = db.get_database_stats()
        total_faces = stats.get('total_faces', 0)

        print(f"🗄️  Database contains {total_faces} face embeddings")

        if total_faces == 0:
            print("\n⚠️  No embeddings found!")
            print("👉 Please run: python 5_embed_faces.py or python 6_embed_faces_gui.py")
            return
        elif total_faces < face_count:
            print(f"\n⚠️  Warning: {face_count - total_faces} faces not embedded yet")
            print("👉 Consider running: python 5_embed_faces.py")

        print("\n✅ Embeddings ready!")
    except Exception as e:
        print(f"❌ Error checking embeddings: {e}")
        return

    wait_for_user()

    # Step 4: Feature extraction test
    print_step(4, "Test Feature Extraction")
    try:
        from face_collector import FaceAnalyzer
        from pathlib import Path

        analyzer = FaceAnalyzer()
        test_images = list(Path(faces_dir).glob("*.jpg"))[:3]

        if test_images:
            print(f"Testing feature extraction on {len(test_images)} sample images...")
            print()

            for img_path in test_images:
                features = analyzer.estimate_basic_features(str(img_path))
                if features:
                    print(f"✅ {img_path.name}:")
                    print(f"   Age: {features.get('estimated_age_group', 'N/A')}")
                    print(f"   Skin: {features.get('estimated_skin_tone', 'N/A')}")
                    print(f"   Quality: {features.get('image_quality', 'N/A')}")
                    print()

            print("✅ Feature extraction working!")
        else:
            print("⚠️  No test images available")

    except Exception as e:
        print(f"⚠️  Feature extraction test skipped: {e}")

    wait_for_user()

    # Step 5: Search demo
    print_step(5, "Search Demonstration")
    print("The search interface allows you to:")
    print("  • 🧠 Semantic search - Find visually similar faces")
    print("  • 📋 Metadata search - Filter by attributes")
    print("  • 🔄 Combined search - Use both methods")
    print()
    print("👉 Launch search interface: python 7_search_faces_gui.py")
    print()

    launch = input("🚀 Launch search interface now? (y/n): ").strip().lower()
    if launch == 'y':
        try:
            import subprocess
            subprocess.run([sys.executable, "7_search_faces_gui.py"])
        except Exception as e:
            print(f"❌ Error launching search: {e}")

    # Summary
    print()
    print("="*70)
    print("🎉 DEMO COMPLETE!")
    print("="*70)
    print()
    print("📚 System Summary:")
    print(f"   • Face Images: {face_count}")
    print(f"   • Vector Embeddings: {total_faces}")
    print(f"   • Database: ChromaDB")
    print(f"   • Features: Age, Skin Tone, Quality, Brightness")
    print()
    print("🔧 Available Tools:")
    print("   0. python 0_launcher.py        - Main launcher menu")
    print("   1. python 1_setup_database.py  - Setup database")
    print("   2. python 2_database_info.py   - View database stats")
    print("   3. python 3_download_faces.py  - Download faces (CLI)")
    print("   4. python 4_download_faces_gui.py - Download faces (GUI)")
    print("   5. python 5_embed_faces.py     - Embed faces (CLI)")
    print("   6. python 6_embed_faces_gui.py - Embed faces (GUI)")
    print("   7. python 7_search_faces_gui.py - Search interface ⭐")
    print("   8. python 8_validate_embeddings.py - Validate data")
    print("   9. python 9_test_features.py   - Test features")
    print("  10. python 10_complete_demo.py  - This demo")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()