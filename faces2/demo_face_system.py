#!/usr/bin/env python3
"""
Face Processing System Demo
Demonstrates the integrated functionality without GUI requirements
"""

import os
import sys
import time
import json
from pathlib import Path

# Check dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
    print("✅ requests: Available")
except ImportError:
    REQUESTS_AVAILABLE = False
    print("❌ requests: Not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
    print("✅ PIL/Pillow: Available")
except ImportError:
    PIL_AVAILABLE = False
    print("❌ PIL/Pillow: Not available")

def show_banner():
    """Display application banner"""
    print("=" * 60)
    print("🎭 INTEGRATED FACE PROCESSING SYSTEM - DEMO")
    print("=" * 60)
    print()

def demo_config():
    """Demonstrate configuration management"""
    print("📋 CONFIGURATION DEMO")
    print("-" * 30)

    # Create sample config
    config = {
        "faces_dir": "./faces",
        "download_delay": 1.0,
        "batch_size": 50,
        "max_workers": 2
    }

    # Save config
    with open("demo_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Configuration saved to demo_config.json")
    print(f"   Faces directory: {config['faces_dir']}")
    print(f"   Download delay: {config['download_delay']}s")
    print(f"   Batch size: {config['batch_size']}")
    print()

def demo_directory_setup():
    """Demonstrate directory setup"""
    print("📁 DIRECTORY SETUP DEMO")
    print("-" * 30)

    directories = ["./faces", "./chroma_db", "./logs"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    print()

def demo_face_download():
    """Demonstrate face downloading"""
    print("📥 FACE DOWNLOAD DEMO")
    print("-" * 30)

    if not REQUESTS_AVAILABLE:
        print("❌ Cannot demo download - requests not available")
        print("   Install with: pip install requests")
        return

    print("🌐 Testing connection to ThisPersonDoesNotExist.com...")

    try:
        # Test download
        response = requests.get(
            "https://thispersondoesnotexist.com/",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10
        )
        response.raise_for_status()

        # Save test image
        timestamp = int(time.time())
        filename = f"demo_face_{timestamp}.jpg"
        filepath = os.path.join("./faces", filename)

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"✅ Successfully downloaded test face: {filename}")
        print(f"   File size: {len(response.content)} bytes")
        print(f"   Saved to: {filepath}")

    except Exception as e:
        print(f"❌ Download failed: {str(e)}")

    print()

def demo_image_analysis():
    """Demonstrate image analysis"""
    print("🔍 IMAGE ANALYSIS DEMO")
    print("-" * 30)

    # Find images in faces directory
    face_files = list(Path("./faces").glob("*.jpg"))

    if not face_files:
        print("📁 No images found for analysis")
        return

    if not PIL_AVAILABLE:
        print("❌ Cannot demo analysis - PIL not available")
        print("   Install with: pip install Pillow")
        return

    print(f"📁 Found {len(face_files)} images")

    total_size = 0
    dimensions = []

    for i, file_path in enumerate(face_files[:5]):  # Analyze first 5
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                dimensions.append((width, height))
                file_size = os.path.getsize(file_path)
                total_size += file_size

                print(f"   {file_path.name}: {width}x{height}, {file_size} bytes")

        except Exception as e:
            print(f"❌ Error analyzing {file_path.name}: {e}")

    if dimensions:
        avg_width = sum(d[0] for d in dimensions) / len(dimensions)
        avg_height = sum(d[1] for d in dimensions) / len(dimensions)

        print(f"\n📊 Analysis Summary:")
        print(f"   Average size: {avg_width:.0f} x {avg_height:.0f}")
        print(f"   Total size: {total_size / 1024:.1f} KB")

    print()

def demo_embedding_simulation():
    """Simulate embedding creation"""
    print("🧠 EMBEDDING SIMULATION DEMO")
    print("-" * 30)

    import random
    import hashlib

    # Simulate creating embeddings for faces
    face_files = list(Path("./faces").glob("*.jpg"))

    if not face_files:
        print("📁 No images found for embedding simulation")
        return

    embeddings_data = []

    for file_path in face_files[:3]:  # Process first 3
        # Simulate feature extraction
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()

        # Create random embedding (normally would be from ML model)
        embedding = [random.random() for _ in range(128)]

        # Simulate metadata
        metadata = {
            "file_path": str(file_path),
            "file_hash": file_hash[:8],
            "timestamp": time.time(),
            "embedding_size": len(embedding)
        }

        embeddings_data.append({
            "metadata": metadata,
            "embedding": embedding[:5] + ["..."] + embedding[-5:]  # Show truncated
        })

        print(f"✅ Created embedding for {file_path.name}")
        print(f"   Hash: {file_hash[:8]}")
        print(f"   Embedding size: {len(embedding)}")
        print(f"   Sample values: {[f'{x:.3f}' for x in embedding[:3]]}...")

    # Save embedding simulation data
    with open("demo_embeddings.json", "w") as f:
        json.dump(embeddings_data, f, indent=2)

    print(f"\n💾 Saved {len(embeddings_data)} embedding records to demo_embeddings.json")
    print()

def demo_search_simulation():
    """Simulate search functionality"""
    print("🔎 SEARCH SIMULATION DEMO")
    print("-" * 30)

    # Load embeddings if available
    if not os.path.exists("demo_embeddings.json"):
        print("❌ No embeddings found - run embedding demo first")
        return

    with open("demo_embeddings.json", "r") as f:
        embeddings_data = json.load(f)

    if not embeddings_data:
        print("❌ No embedding data available")
        return

    print(f"📊 Loaded {len(embeddings_data)} embeddings")

    # Simulate search query
    import random
    query_embedding = [random.random() for _ in range(128)]

    print("🔍 Simulating search with random query...")

    # Calculate simple similarity (cosine similarity simulation)
    results = []
    for item in embeddings_data:
        # Simulate distance calculation
        distance = random.random()  # Normally would be actual similarity
        results.append({
            "file_path": item["metadata"]["file_path"],
            "distance": distance,
            "metadata": item["metadata"]
        })

    # Sort by distance (lower = more similar)
    results.sort(key=lambda x: x["distance"])

    print(f"\n🎯 Search Results (top {min(3, len(results))}):")
    for i, result in enumerate(results[:3]):
        print(f"   {i+1}. {Path(result['file_path']).name}")
        print(f"      Distance: {result['distance']:.3f}")
        print(f"      Hash: {result['metadata']['file_hash']}")

    print()

def demo_statistics():
    """Show system statistics"""
    print("📊 SYSTEM STATISTICS DEMO")
    print("-" * 30)

    # Count files
    face_files = list(Path("./faces").glob("*.jpg"))
    config_files = list(Path(".").glob("*config*.json"))

    # Calculate sizes
    total_size = sum(f.stat().st_size for f in face_files if f.exists())

    # Show stats
    stats = {
        "faces_downloaded": len(face_files),
        "total_size_mb": total_size / (1024 * 1024),
        "config_files": len(config_files),
        "directories_created": len([d for d in ["./faces", "./chroma_db"] if os.path.exists(d)]),
        "demo_files_created": len(list(Path(".").glob("demo_*.json")))
    }

    for key, value in stats.items():
        if "size" in key:
            print(f"   {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")

    print()

def cleanup_demo():
    """Clean up demo files"""
    print("🧹 CLEANUP OPTIONS")
    print("-" * 30)

    demo_files = list(Path(".").glob("demo_*"))
    print(f"📁 Demo files created: {len(demo_files)}")

    for file_path in demo_files:
        print(f"   {file_path.name}")

    faces_files = list(Path("./faces").glob("*.jpg"))
    print(f"📸 Face images downloaded: {len(faces_files)}")

    print("\nTo clean up:")
    print("   rm demo_*.json")
    print("   rm -rf ./faces/*.jpg")
    print()

def main():
    """Main demonstration function"""
    show_banner()

    print("This demo showcases the integrated face processing system functionality")
    print("without requiring GUI components or user interaction.\n")

    # Run demonstrations
    demo_config()
    demo_directory_setup()
    demo_face_download()
    demo_image_analysis()
    demo_embedding_simulation()
    demo_search_simulation()
    demo_statistics()
    cleanup_demo()

    print("🎉 DEMO COMPLETED")
    print("=" * 30)
    print("The integrated face processing system includes:")
    print("✅ Configuration management")
    print("✅ Face downloading from web")
    print("✅ Image analysis and processing")
    print("✅ Vector embedding creation")
    print("✅ Similarity search functionality")
    print("✅ Statistics and monitoring")
    print()
    print("To use the full system:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Install tkinter: sudo apt-get install python3-tk")
    print("3. Run GUI: python3 integrated_face_gui.py")
    print("4. Or use console: python3 console_face_app.py")

if __name__ == "__main__":
    main()