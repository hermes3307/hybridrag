#!/usr/bin/env python3

import os
import chromadb
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_face_files():
    """Get all face files from the faces directory"""
    faces_dir = Path("faces")
    face_files = list(faces_dir.glob("face_*.jpg"))
    return sorted([f.name for f in face_files])

def get_embedded_files():
    """Get all files that have been embedded in ChromaDB"""
    try:
        client = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_collection("faces")

        # Get all IDs from the collection
        all_data = collection.get()
        embedded_files = []

        # Extract filenames from IDs (assuming IDs are based on filenames)
        for doc_id in all_data['ids']:
            # Convert back to filename format
            if doc_id.startswith('face_'):
                # Construct filename from ID
                embedded_files.append(doc_id + '.jpg')
            else:
                # Try to extract from metadata if available
                try:
                    metadata = collection.get(ids=[doc_id], include=['metadatas'])
                    if metadata['metadatas'] and 'filename' in metadata['metadatas'][0]:
                        embedded_files.append(metadata['metadatas'][0]['filename'])
                    else:
                        # Fallback: try to reconstruct from ID
                        embedded_files.append(doc_id + '.jpg')
                except:
                    pass

        return sorted(embedded_files)
    except Exception as e:
        logger.error(f"Error accessing ChromaDB: {e}")
        return []

def find_missing_files():
    """Find files that exist but are not embedded"""
    logger.info("🔍 Scanning face files...")
    face_files = get_face_files()
    logger.info(f"📁 Found {len(face_files)} face files")

    logger.info("🔍 Scanning embedded files...")
    embedded_files = get_embedded_files()
    logger.info(f"🗄️  Found {len(embedded_files)} embedded files")

    # Find missing files
    face_files_set = set(face_files)
    embedded_files_set = set(embedded_files)

    missing_files = face_files_set - embedded_files_set
    extra_embeddings = embedded_files_set - face_files_set

    logger.info("\n" + "="*60)
    logger.info("📊 VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"📁 Total face files: {len(face_files)}")
    logger.info(f"🗄️  Total embeddings: {len(embedded_files)}")
    logger.info(f"❌ Missing embeddings: {len(missing_files)}")
    logger.info(f"🔄 Extra embeddings: {len(extra_embeddings)}")

    if missing_files:
        logger.info(f"\n🚨 MISSING FILES ({len(missing_files)}):")
        for i, filename in enumerate(sorted(missing_files), 1):
            logger.info(f"  {i:4d}. {filename}")

    if extra_embeddings:
        logger.info(f"\n🔄 EXTRA EMBEDDINGS ({len(extra_embeddings)}):")
        for i, filename in enumerate(sorted(extra_embeddings), 1):
            logger.info(f"  {i:4d}. {filename}")

    return missing_files, extra_embeddings

def save_missing_files_list(missing_files):
    """Save missing files to a text file for processing"""
    if missing_files:
        with open("missing_files.txt", "w") as f:
            for filename in sorted(missing_files):
                f.write(f"{filename}\n")
        logger.info(f"💾 Saved missing files list to: missing_files.txt")

if __name__ == "__main__":
    logger.info("🔮 Face Embedding Validation Tool")
    logger.info("="*50)

    missing_files, extra_embeddings = find_missing_files()

    if missing_files:
        save_missing_files_list(missing_files)
        logger.info(f"\n💡 To process missing files, run:")
        logger.info(f"   python 100.embedintoVector.py --batch-size 50 --file-list missing_files.txt")
    else:
        logger.info("\n✅ All files are properly embedded!")

    logger.info("\n🎯 Validation complete!")