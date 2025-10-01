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
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

# Set up logging first (before any imports that use logger)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from 3_collect_faces.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import classes from the actual existing module
try:
    # Try importing from 3_collect_faces module (without .py extension)
    import importlib.util
    spec = importlib.util.spec_from_file_location("collect_faces", "./3_collect_faces.py")
    collect_faces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collect_faces)
    FaceData = collect_faces.FaceData
    FaceAnalyzer = collect_faces.FaceAnalyzer
    FaceEmbedder = collect_faces.FaceEmbedder
except Exception as e:
    logger.error(f"Failed to import from 3_collect_faces.py: {e}")
    raise

# Import ChromaDB directly
import chromadb

@dataclass
class EmbeddingStats:
    """Statistics for the embedding process"""
    total_files: int = 0
    processed_files: int = 0
    successful_embeddings: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    metadata_loaded: int = 0
    metadata_missing: int = 0
    start_time: float = 0
    elapsed_time: float = 0

    def get_rate(self) -> float:
        """Calculate processing rate per second"""
        if self.elapsed_time > 0:
            return self.processed_files / self.elapsed_time
        return 0.0

class VectorEmbeddingProcessor:
    """Enhanced processor for embedding faces into vector database"""

    def __init__(self, faces_dir: str = "./faces", batch_size: int = 50, max_workers: int = 4,
                 db_path: str = "./chroma_db", collection_name: str = "faces"):
        self.faces_dir = faces_dir
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.db_path = db_path
        self.collection_name = collection_name
        self.stats = EmbeddingStats()
        self.running = True
        self.chroma_client = None
        self.collection = None
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
        """Scan faces directory for image files (including subdirectories)"""
        face_files = []
        supported_extensions = {'.jpg', '.jpeg', '.png'}

        faces_path = Path(self.faces_dir)

        # Check if faces_dir exists
        if not faces_path.exists():
            logger.error(f"Faces directory does not exist: {self.faces_dir}")
            return []

        # If faces_dir is a directory, scan it (non-recursively for performance)
        if faces_path.is_dir():
            for file_path in faces_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    face_files.append(str(file_path))

        logger.info(f"Found {len(face_files)} image files in {self.faces_dir}")
        return sorted(face_files)

    def load_existing_embeddings(self) -> Set[str]:
        """Load existing embeddings to avoid duplicates"""
        existing_hashes = set()
        try:
            if self.collection:
                # Get all existing face data from database
                results = self.collection.get()
                if results and 'metadatas' in results:
                    for metadata in results['metadatas']:
                        if metadata and 'image_hash' in metadata:
                            existing_hashes.add(metadata['image_hash'])
                logger.info(f"Loaded {len(existing_hashes)} existing embeddings from database")
        except Exception as e:
            logger.warning(f"Could not load existing embeddings: {e}")

        return existing_hashes

    def extract_face_features(self, image_path: str) -> Dict[str, Any]:
        """Extract comprehensive features from face image"""
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

                # Basic features
                features = {
                    'width': width,
                    'height': height,
                    'format': format_type,
                    'file_size': os.path.getsize(image_path),
                    'mean_brightness': float(np.mean(img_array)),
                    'std_brightness': float(np.std(img_array)),
                    'color_channels': img_array.shape[2] if len(img_array.shape) > 2 else 1
                }

                # Use FaceAnalyzer for advanced features
                try:
                    analyzer = FaceAnalyzer()
                    advanced_features = analyzer.estimate_basic_features(image_path)

                    # Merge advanced features
                    if advanced_features:
                        features.update(advanced_features)
                        logger.debug(f"Added advanced features: age={advanced_features.get('estimated_age_group')}, "
                                   f"skin_tone={advanced_features.get('estimated_skin_tone')}, "
                                   f"quality={advanced_features.get('image_quality')}")
                except Exception as e:
                    logger.debug(f"Could not extract advanced features: {e}")
                    # Add default values if analyzer fails
                    features.update({
                        'estimated_age_group': 'adult',
                        'estimated_skin_tone': 'medium',
                        'image_quality': 'medium'
                    })

                return features
        except Exception as e:
            logger.warning(f"Could not extract features from {image_path}: {e}")
            return {}

    def load_json_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Load JSON metadata file for an image"""
        try:
            # Get base filename without extension
            base_name = os.path.splitext(image_path)[0]
            json_path = f"{base_name}.json"

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.debug(f"Loaded metadata from {os.path.basename(json_path)}")
                return metadata
            else:
                logger.debug(f"No metadata file found: {os.path.basename(json_path)}")
                return None
        except Exception as e:
            logger.warning(f"Error loading metadata for {image_path}: {e}")
            return None

    def process_single_face(self, file_path: str, face_id: str) -> Optional[FaceData]:
        """Process a single face image with metadata"""
        try:
            # Calculate image hash
            image_hash = self.get_file_hash(file_path)
            if not image_hash:
                return None

            # Check if already processed
            if image_hash in self.processed_hashes:
                self.stats.duplicates_skipped += 1
                logger.info(f"⏭️  Duplicate detected, skipping: {os.path.basename(file_path)}")
                return None

            # Load JSON metadata
            json_metadata = self.load_json_metadata(file_path)

            # Extract features
            features = self.extract_face_features(file_path)
            if not features:
                return None

            # Merge JSON metadata into features
            if json_metadata:
                # Add important metadata fields
                features['source_metadata'] = {
                    'filename': json_metadata.get('filename'),
                    'md5_hash': json_metadata.get('md5_hash'),
                    'download_timestamp': json_metadata.get('download_timestamp'),
                    'download_date': json_metadata.get('download_date'),
                    'source_url': json_metadata.get('source_url'),
                    'file_size_kb': json_metadata.get('file_size_kb'),
                    'http_status_code': json_metadata.get('http_status_code')
                }

                # Add image properties
                if 'image_properties' in json_metadata:
                    features['json_image_properties'] = json_metadata['image_properties']

                # Add downloader config
                if 'downloader_config' in json_metadata:
                    features['downloader_config'] = json_metadata['downloader_config']

                self.stats.metadata_loaded += 1
                logger.info(f"📋 Loaded metadata for {os.path.basename(file_path)}")
            else:
                self.stats.metadata_missing += 1

            # Generate embedding using face embedder
            embedder = FaceEmbedder()
            embedding = embedder.generate_embedding(file_path, features)

            # Create FaceData object
            face_data = FaceData(
                face_id=face_id,
                file_path=file_path,
                features=features,
                embedding=embedding if embedding else None,
                timestamp=datetime.now().isoformat(),
                image_hash=image_hash
            )

            # Add to processed hashes
            self.processed_hashes.add(image_hash)

            return face_data

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
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

                        # Format embedding preview for logging
                        embedding_preview = ""
                        if face_data.embedding and len(face_data.embedding) > 0:
                            embedding_len = len(face_data.embedding)
                            if embedding_len >= 10:
                                preview_values = face_data.embedding[:5] + face_data.embedding[-5:]
                                embedding_preview = f"[{', '.join(f'{x:.4f}' for x in preview_values[:5])}, ..., {', '.join(f'{x:.4f}' for x in preview_values[5:])}] ({embedding_len}D)"
                            else:
                                embedding_preview = f"[{', '.join(f'{x:.4f}' for x in face_data.embedding)}] ({embedding_len}D)"
                        else:
                            embedding_preview = "[None] (0D)"

                        logger.info(f"Processed: {os.path.basename(file_path)} -> {face_data.face_id} | Embedding: {embedding_preview}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    self.stats.errors += 1
                finally:
                    self.stats.processed_files += 1

        return face_data_list

    def initialize_database(self, clear_existing: bool = False):
        """Initialize the face database"""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)

            if clear_existing:
                logger.info("Clearing existing faces collection...")
                try:
                    self.chroma_client.delete_collection(self.collection_name)
                    logger.info("Collection deleted")
                except Exception as e:
                    logger.debug(f"Collection deletion: {e}")

            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Face embeddings collection"}
            )
            logger.info(f"Collection '{self.collection_name}' initialized")

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
            'metadata_loaded': self.stats.metadata_loaded,
            'metadata_missing': self.stats.metadata_missing,
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
                if face_data_list and self.collection:
                    try:
                        # Prepare data for ChromaDB
                        ids = []
                        embeddings = []
                        metadatas = []
                        documents = []

                        for face_data in face_data_list:
                            if face_data.embedding:
                                ids.append(face_data.face_id)
                                embeddings.append(face_data.embedding)

                                # Prepare metadata (must be JSON-serializable)
                                metadata = {
                                    'image_hash': face_data.image_hash,
                                    'file_path': face_data.file_path,
                                    'timestamp': face_data.timestamp,
                                }

                                # Add basic features
                                if face_data.features:
                                    for key, value in face_data.features.items():
                                        # Skip nested dicts/objects, only add simple types
                                        if isinstance(value, (str, int, float, bool)):
                                            metadata[key] = value
                                        elif isinstance(value, dict):
                                            # Store as JSON string for nested dicts
                                            metadata[f'{key}_json'] = json.dumps(value)

                                metadatas.append(metadata)

                                # Document is the file name
                                documents.append(os.path.basename(face_data.file_path))

                        # Add to collection
                        if ids:
                            self.collection.add(
                                ids=ids,
                                embeddings=embeddings,
                                metadatas=metadatas,
                                documents=documents
                            )
                            logger.info(f"Added {len(ids)} embeddings to database")
                    except Exception as e:
                        logger.error(f"Error adding batch to database: {e}")
                        self.stats.errors += len(face_data_list)

            # Final statistics
            final_stats = self.get_stats()
            logger.info("=" * 60)
            logger.info("✅ Embedding process completed!")
            logger.info("=" * 60)
            logger.info(f"📊 Total processed: {final_stats['processed_files']}")
            logger.info(f"✅ Successful embeddings: {final_stats['successful_embeddings']}")
            logger.info(f"📋 Metadata loaded: {final_stats['metadata_loaded']}")
            logger.info(f"⚠️  Metadata missing: {final_stats['metadata_missing']}")
            logger.info(f"⏭️  Duplicates skipped: {final_stats['duplicates_skipped']}")
            logger.info(f"❌ Errors: {final_stats['errors']}")
            logger.info(f"⚡ Processing rate: {final_stats['processing_rate']:.2f} files/second")
            logger.info(f"⏱️  Total time: {final_stats['elapsed_time']:.2f} seconds")
            logger.info("=" * 60)

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