#!/usr/bin/env python3
"""
Enhanced Vector Embedding System for Face Data
Processes face images from ./faces directory into ChromaDB with progress tracking and duplicate removal
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

# Import existing modules
from face_collector import FaceData
from face_database import FaceDatabase

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingStats:
    """Statistics for the embedding process"""
    total_files: int = 0
    processed_files: int = 0
    successful_embeddings: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    start_time: float = 0
    elapsed_time: float = 0

    def get_rate(self) -> float:
        """Calculate processing rate per second"""
        if self.elapsed_time > 0:
            return self.processed_files / self.elapsed_time
        return 0.0

class VectorEmbeddingProcessor:
    """Enhanced processor for embedding faces into vector database"""

    def __init__(self, faces_dir: str = "./faces", batch_size: int = 50, max_workers: int = 4):
        self.faces_dir = faces_dir
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.stats = EmbeddingStats()
        self.running = True
        self.face_db: Optional[FaceDatabase] = None
        self.processed_hashes: Set[str] = set()

        # Ensure faces directory exists
        if not os.path.exists(faces_dir):
            raise ValueError(f"Faces directory not found: {faces_dir}")

    def get_file_hash(self, file_path: str) -> str:
        """Calculate hash of file for duplicate detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def scan_face_files(self) -> List[str]:
        """Scan faces directory for image files"""
        face_files = []
        supported_extensions = {'.jpg', '.jpeg', '.png'}

        for file_path in Path(self.faces_dir).iterdir():
            if file_path.suffix.lower() in supported_extensions:
                face_files.append(str(file_path))

        return sorted(face_files)

    def load_existing_embeddings(self) -> Set[str]:
        """Load existing embeddings to avoid duplicates"""
        existing_hashes = set()
        try:
            if self.face_db:
                # Get all existing face data from database
                collection = self.face_db.collection
                results = collection.get()
                if results and 'metadatas' in results:
                    for metadata in results['metadatas']:
                        if metadata and 'image_hash' in metadata:
                            existing_hashes.add(metadata['image_hash'])
                logger.info(f"Loaded {len(existing_hashes)} existing embeddings from database")
        except Exception as e:
            logger.warning(f"Could not load existing embeddings: {e}")

        return existing_hashes

    def extract_face_features(self, image_path: str) -> Dict[str, Any]:
        """Extract basic features from face image"""
        try:
            with Image.open(image_path) as img:
                # Get basic image info
                width, height = img.size
                format_type = img.format or 'Unknown'

                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Get basic color statistics
                img_array = np.array(img)

                features = {
                    'width': width,
                    'height': height,
                    'format': format_type,
                    'file_size': os.path.getsize(image_path),
                    'mean_brightness': float(np.mean(img_array)),
                    'std_brightness': float(np.std(img_array)),
                    'color_channels': img_array.shape[2] if len(img_array.shape) > 2 else 1
                }

                return features
        except Exception as e:
            logger.warning(f"Could not extract features from {image_path}: {e}")
            return {}

    def process_single_face(self, file_path: str, face_id: str) -> Optional[FaceData]:
        """Process a single face image"""
        try:
            # Calculate image hash
            image_hash = self.get_file_hash(file_path)
            if not image_hash:
                return None

            # Check if already processed
            if image_hash in self.processed_hashes:
                self.stats.duplicates_skipped += 1
                logger.info(f"Duplicate detected, skipping: {os.path.basename(file_path)}")
                return None

            # Extract features
            features = self.extract_face_features(file_path)
            if not features:
                return None

            # Create FaceData object
            face_data = FaceData(
                face_id=face_id,
                file_path=file_path,
                features=features,
                embedding=None,  # Will be generated by face_database
                timestamp=datetime.now().isoformat(),
                image_hash=image_hash
            )

            # Add to processed hashes
            self.processed_hashes.add(image_hash)

            return face_data

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats.errors += 1
            return None

    def process_batch(self, file_paths: List[str], start_idx: int) -> List[FaceData]:
        """Process a batch of face files"""
        face_data_list = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_file = {}
            for i, file_path in enumerate(file_paths):
                face_id = f"face_{start_idx + i:06d}"
                future = executor.submit(self.process_single_face, file_path, face_id)
                future_to_file[future] = file_path

            # Collect results
            for future in as_completed(future_to_file):
                if not self.running:
                    break

                file_path = future_to_file[future]
                try:
                    face_data = future.result()
                    if face_data:
                        face_data_list.append(face_data)
                        self.stats.successful_embeddings += 1
                        logger.info(f"Processed: {os.path.basename(file_path)} -> {face_data.face_id}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    self.stats.errors += 1
                finally:
                    self.stats.processed_files += 1

        return face_data_list

    def initialize_database(self, clear_existing: bool = False):
        """Initialize the face database"""
        try:
            self.face_db = FaceDatabase()

            if clear_existing:
                logger.info("Clearing existing faces collection...")
                try:
                    self.face_db.client.delete_collection("faces")
                    self.face_db._initialize_db()  # Recreate collection
                    logger.info("Collection cleared and recreated")
                except Exception as e:
                    logger.info(f"Collection creation: {e}")

            # Load existing embeddings for duplicate detection
            self.processed_hashes = self.load_existing_embeddings()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        self.stats.elapsed_time = time.time() - self.stats.start_time if self.stats.start_time > 0 else 0

        return {
            'total_files': self.stats.total_files,
            'processed_files': self.stats.processed_files,
            'successful_embeddings': self.stats.successful_embeddings,
            'duplicates_skipped': self.stats.duplicates_skipped,
            'errors': self.stats.errors,
            'elapsed_time': self.stats.elapsed_time,
            'processing_rate': self.stats.get_rate(),
            'progress_percentage': (self.stats.processed_files / max(self.stats.total_files, 1)) * 100,
            'remaining_files': max(0, self.stats.total_files - self.stats.processed_files)
        }

    def get_directory_info(self) -> Dict[str, Any]:
        """Get information about the faces directory"""
        try:
            total_size = 0
            file_count = 0

            for root, dirs, files in os.walk(self.faces_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                            file_count += 1
                        except OSError:
                            continue

            # Get available disk space
            free_space = shutil.disk_usage(self.faces_dir)[2]

            return {
                'total_files': file_count,
                'total_size_bytes': total_size,
                'free_space_bytes': free_space,
                'directory_path': self.faces_dir
            }
        except Exception as e:
            logger.error(f"Error getting directory info: {e}")
            return {}

    def start_embedding_process(self, clear_existing: bool = False) -> bool:
        """Start the embedding process"""
        try:
            logger.info("Starting vector embedding process...")
            self.stats.start_time = time.time()
            self.running = True

            # Initialize database
            self.initialize_database(clear_existing)

            # Scan for face files
            logger.info("Scanning for face files...")
            face_files = self.scan_face_files()
            self.stats.total_files = len(face_files)

            if self.stats.total_files == 0:
                logger.warning("No face files found in directory")
                return False

            logger.info(f"Found {self.stats.total_files} face files to process")

            # Process files in batches
            for i in range(0, len(face_files), self.batch_size):
                if not self.running:
                    break

                batch_files = face_files[i:i + self.batch_size]
                logger.info(f"Processing batch {i//self.batch_size + 1} ({len(batch_files)} files)...")

                # Process batch
                face_data_list = self.process_batch(batch_files, i)

                # Add to database
                if face_data_list and self.face_db:
                    try:
                        added_count = self.face_db.add_faces(face_data_list)
                        logger.info(f"Added {added_count} embeddings to database")
                    except Exception as e:
                        logger.error(f"Error adding batch to database: {e}")
                        self.stats.errors += len(face_data_list)

            # Final statistics
            final_stats = self.get_stats()
            logger.info("Embedding process completed!")
            logger.info(f"Total processed: {final_stats['processed_files']}")
            logger.info(f"Successful embeddings: {final_stats['successful_embeddings']}")
            logger.info(f"Duplicates skipped: {final_stats['duplicates_skipped']}")
            logger.info(f"Errors: {final_stats['errors']}")
            logger.info(f"Processing rate: {final_stats['processing_rate']:.2f} files/second")

            return True

        except Exception as e:
            logger.error(f"Embedding process failed: {e}")
            return False

    def stop_process(self):
        """Stop the embedding process"""
        self.running = False
        logger.info("Stopping embedding process...")

def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Embed face images into vector database")
    parser.add_argument("--faces-dir", default="./faces", help="Directory containing face images")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum worker threads")
    parser.add_argument("--clear", action="store_true", help="Clear existing embeddings")

    args = parser.parse_args()

    # Create processor
    processor = VectorEmbeddingProcessor(
        faces_dir=args.faces_dir,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )

    # Get directory info
    dir_info = processor.get_directory_info()
    print(f"\n📂 Directory Info:")
    print(f"   Path: {dir_info.get('directory_path', 'Unknown')}")
    print(f"   Files: {dir_info.get('total_files', 0):,}")
    print(f"   Size: {dir_info.get('total_size_bytes', 0) / (1024**3):.2f} GB")
    print(f"   Free space: {dir_info.get('free_space_bytes', 0) / (1024**3):.2f} GB")

    # Start processing
    success = processor.start_embedding_process(clear_existing=args.clear)

    if success:
        print("\n✅ Embedding process completed successfully!")
    else:
        print("\n❌ Embedding process failed!")

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())