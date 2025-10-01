#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2️⃣  Database Information & Stats
Display comprehensive database information
"""

import sys
import os
import io
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function to display ChromaDB information"""
    try:
        import chromadb

        print("=" * 70)
        print("2️⃣  DATABASE INFORMATION & STATS")
        print("=" * 70)
        print()
        print("📊 CHROMADB DATABASE INFORMATION")
        print("=" * 70)
        print()

        # Initialize client
        db_path = "./chroma_db"
        if not os.path.exists(db_path):
            print(f"❌ Database directory not found: {db_path}")
            return

        client = chromadb.PersistentClient(path=db_path)

        # Get all collections
        collections = client.list_collections()

        print(f"📁 Database Path: {os.path.abspath(db_path)}")
        print(f"📚 Total Collections: {len(collections)}")
        print()

        if not collections:
            print("⚠️  No collections found in database")
            return

        # Display collection information
        print("🗂️  COLLECTIONS:")
        print("-" * 70)

        total_vectors = 0
        for collection in collections:
            try:
                count = collection.count()
                total_vectors += count
                metadata = collection.metadata or {}

                print(f"\n📦 Collection: {collection.name}")
                print(f"   📊 Vector Count: {count:,}")

                if metadata:
                    print(f"   🏷️  Metadata:")
                    for key, value in metadata.items():
                        print(f"      • {key}: {value}")

                # Get sample data if available
                if count > 0:
                    try:
                        sample = collection.peek(limit=1)
                        if sample.get('embeddings') is not None and len(sample['embeddings']) > 0:
                            if sample['embeddings'][0] is not None:
                                dimensions = len(sample['embeddings'][0])
                                print(f"   📐 Vector Dimensions: {dimensions}")
                    except Exception as e:
                        print(f"   ⚠️  Could not get sample data: {e}")

            except Exception as e:
                print(f"\n📦 Collection: {collection.name}")
                print(f"   ❌ Error: {e}")

        print()
        print("-" * 70)
        print(f"🔢 TOTAL VECTORS ACROSS ALL COLLECTIONS: {total_vectors:,}")
        print()

        # Database size information
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(db_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)

            size_mb = total_size / (1024 * 1024)
            size_gb = size_mb / 1024

            print("💾 DATABASE SIZE:")
            print(f"   Total Size: {size_mb:.2f} MB ({size_gb:.3f} GB)")

            if total_vectors > 0:
                avg_size = (total_size / total_vectors) / 1024
                print(f"   Average per Vector: {avg_size:.2f} KB")
        except Exception as e:
            print(f"⚠️  Could not calculate database size: {e}")

        print()
        print("=" * 70)
        print(f"✅ Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

    except ImportError:
        print("❌ ChromaDB not installed. Install it with: pip install chromadb")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
