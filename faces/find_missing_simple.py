#!/usr/bin/env python3

import os
import chromadb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🔍 Finding missing files from database...")

    # Get face files from directory
    faces_dir = "faces"
    all_files = set()
    for f in os.listdir(faces_dir):
        if f.startswith("face_") and f.endswith(".jpg"):
            all_files.add(f)

    logger.info(f"📁 Found {len(all_files)} face files in directory")

    # Check ChromaDB
    try:
        client = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_collection("faces")

        # Get count from collection
        count = collection.count()
        logger.info(f"🗄️  Found {count} documents in ChromaDB")

        # Get a sample of IDs to understand the format
        sample_data = collection.get(limit=10)
        logger.info(f"📋 Sample IDs: {sample_data['ids'][:5] if sample_data['ids'] else 'None'}")

    except Exception as e:
        logger.error(f"❌ Error accessing ChromaDB: {e}")
        return

    # Simple calculation
    missing_count = len(all_files) - count
    logger.info(f"\n📊 SUMMARY:")
    logger.info(f"   📁 Total files: {len(all_files)}")
    logger.info(f"   🗄️  Total embeddings: {count}")
    logger.info(f"   ❌ Missing: {missing_count}")

    if missing_count > 0:
        logger.info(f"\n💡 To process the remaining {missing_count} files:")
        logger.info(f"   Run: python 100.embedintoVector.py --batch-size 50")
        logger.info(f"   Or use the GUI: ./100.embedintoVectorgui.sh")

if __name__ == "__main__":
    main()