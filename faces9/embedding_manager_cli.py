#!/usr/bin/env python3
"""
Embedding Management CLI

A comprehensive CLI tool for managing face embeddings in the vector database.
Features:
- Display vector database statistics
- Count and match image files with JSON metadata
- Batch embed unembedded images with detailed progress
- Support for multiple embedding models
- Command-line arguments for automation
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import FaceEmbedder, FaceAnalyzer, SystemConfig, FaceData
from pgvector_db import PgVectorDatabaseManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingManagerCLI:
    """CLI Manager for Face Embeddings"""

    def __init__(self, faces_dir: str = None, embedding_model: str = None, quiet: bool = False, workers: int = 1):
        """Initialize the CLI manager"""
        self.config = SystemConfig()
        self.db_manager = PgVectorDatabaseManager(self.config)
        self.faces_dir = faces_dir or os.getenv('FACES_DIR', './faces')
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'statistical')
        self.quiet = quiet
        self.workers = max(1, workers)  # Ensure at least 1 worker
        self.lock = threading.Lock()  # For thread-safe progress updates

        # Initialize database
        if not self.quiet:
            print("🔌 Connecting to PostgreSQL database...")
        if not self.db_manager.initialize():
            if not self.quiet:
                print("❌ Failed to connect to database")
            sys.exit(1)
        if not self.quiet:
            print("✅ Database connection established\n")

    def get_database_stats(self) -> Dict:
        """Get statistics from the vector database"""
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            # Total vectors count
            cursor.execute("SELECT COUNT(*) FROM faces")
            total_vectors = cursor.fetchone()[0]

            # Vectors by embedding model
            cursor.execute("""
                SELECT embedding_model, COUNT(*)
                FROM faces
                GROUP BY embedding_model
                ORDER BY COUNT(*) DESC
            """)
            models = cursor.fetchall()

            # Vectors by date (last 7 days)
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*)
                FROM faces
                WHERE timestamp > NOW() - INTERVAL '7 days'
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """)
            recent_activity = cursor.fetchall()

            # Get all embedded face_ids and extract hashes for matching
            cursor.execute("SELECT face_id FROM faces")
            all_face_ids = cursor.fetchall()

            # Create a set of face_ids and also extract hash suffixes
            embedded_face_ids = {row[0] for row in all_face_ids}

            # Extract hash suffixes (last part after underscore) for fuzzy matching
            # Format: face_1761876049_d1548706 -> d1548706
            embedded_hashes = set()
            for face_id in embedded_face_ids:
                parts = face_id.split('_')
                if len(parts) >= 3:
                    embedded_hashes.add(parts[-1])  # Last part is the hash

            cursor.close()

            return {
                'total_vectors': total_vectors,
                'models': models,
                'recent_activity': recent_activity,
                'embedded_face_ids': embedded_face_ids,
                'embedded_hashes': embedded_hashes
            }

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'total_vectors': 0,
                'models': [],
                'recent_activity': [],
                'embedded_face_ids': set(),
                'embedded_hashes': set()
            }
        finally:
            if conn:
                self.db_manager.return_connection(conn)

    def scan_faces_directory(self) -> Dict:
        """Scan the faces directory and count files"""
        faces_path = Path(self.faces_dir)

        if not faces_path.exists():
            logger.error(f"Faces directory not found: {self.faces_dir}")
            return {
                'image_files': [],
                'json_files': [],
                'matched_pairs': [],
                'unmatched_images': [],
                'unmatched_jsons': []
            }

        # Find all image files (exclude macOS metadata files starting with ._ )
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = []
        for ext in image_extensions:
            # Filter out macOS metadata files (._*)
            image_files.extend([f for f in faces_path.glob(f'*{ext}') if not f.name.startswith('._')])

        # Find all JSON files (exclude macOS metadata files starting with ._ )
        json_files = [f for f in faces_path.glob('*.json') if not f.name.startswith('._')]

        # Create mapping of base names
        images_by_base = {}
        for img in image_files:
            base_name = img.stem  # filename without extension
            images_by_base[base_name] = img

        jsons_by_base = {}
        for json_file in json_files:
            base_name = json_file.stem
            jsons_by_base[base_name] = json_file

        # Find matched and unmatched files
        matched_pairs = []
        unmatched_images = []
        unmatched_jsons = []

        for base_name, img_path in images_by_base.items():
            if base_name in jsons_by_base:
                matched_pairs.append({
                    'image': img_path,
                    'json': jsons_by_base[base_name],
                    'base_name': base_name
                })
            else:
                unmatched_images.append(img_path)

        for base_name, json_path in jsons_by_base.items():
            if base_name not in images_by_base:
                unmatched_jsons.append(json_path)

        return {
            'image_files': image_files,
            'json_files': json_files,
            'matched_pairs': matched_pairs,
            'unmatched_images': unmatched_images,
            'unmatched_jsons': unmatched_jsons
        }

    def extract_hash_from_filename(self, filename: str) -> str:
        """
        Extract hash from filename
        Format: face_20251018_123838_826_4ba7ed60.jpg -> 4ba7ed60
        """
        try:
            # Get base name without extension
            base_name = Path(filename).stem
            # Extract the last part which is the hash
            parts = base_name.split('_')
            if len(parts) >= 4:
                return parts[-1]  # Last part is the hash
        except Exception as e:
            logger.warning(f"Could not extract hash from {filename}: {e}")
        return None

    def extract_face_id_from_filename(self, filename: str) -> str:
        """
        Extract face_id from filename
        Format: face_20251018_123838_826_4ba7ed60.jpg -> face_20251018_123838_826_4ba7ed60
        """
        try:
            # Remove file extension only, keep the full base name
            return Path(filename).stem
        except Exception as e:
            logger.warning(f"Could not extract face_id from {filename}: {e}")
        return None

    def get_unembedded_pairs(self, matched_pairs: List[Dict], db_stats: Dict) -> List[Dict]:
        """Filter matched pairs to find those not yet embedded"""
        unembedded = []
        embedded_hashes = db_stats.get('embedded_hashes', set())

        for pair in matched_pairs:
            # Extract hash from filename to check if already embedded
            file_hash = self.extract_hash_from_filename(pair['base_name'])

            # If hash not in database, this image is not embedded yet
            if file_hash and file_hash not in embedded_hashes:
                face_id = self.extract_face_id_from_filename(pair['base_name'])
                pair['face_id'] = face_id
                pair['hash'] = file_hash
                unembedded.append(pair)

        return unembedded

    def display_stats(self):
        """Display comprehensive statistics"""
        print("=" * 80)
        print("📊 EMBEDDING MANAGEMENT DASHBOARD")
        print("=" * 80)
        print()

        # Database statistics
        print("🗄️  DATABASE STATISTICS")
        print("-" * 80)
        db_stats = self.get_database_stats()

        print(f"Total Embedded Vectors: {db_stats['total_vectors']:,}")
        print(f"Unique Embeddings (by hash): {len(db_stats['embedded_hashes']):,}")
        print()

        if db_stats['models']:
            print("Embedding Models Used:")
            for model, count in db_stats['models']:
                percentage = (count / db_stats['total_vectors'] * 100) if db_stats['total_vectors'] > 0 else 0
                print(f"  • {model}: {count:,} vectors ({percentage:.1f}%)")
        print()

        if db_stats['recent_activity']:
            print("Recent Activity (Last 7 Days):")
            for date, count in db_stats['recent_activity']:
                print(f"  • {date}: {count:,} vectors")
        print()

        # File system statistics
        print("📁 FILE SYSTEM STATISTICS")
        print("-" * 80)
        scan_results = self.scan_faces_directory()

        total_images = len(scan_results['image_files'])
        total_jsons = len(scan_results['json_files'])
        matched_count = len(scan_results['matched_pairs'])
        unmatched_images_count = len(scan_results['unmatched_images'])
        unmatched_jsons_count = len(scan_results['unmatched_jsons'])

        # Count unique hashes in filesystem
        unique_file_hashes = set()
        for pair in scan_results['matched_pairs']:
            file_hash = self.extract_hash_from_filename(pair['base_name'])
            if file_hash:
                unique_file_hashes.add(file_hash)

        duplicate_files = total_images - len(unique_file_hashes)

        print(f"Faces Directory: {self.faces_dir}")
        print(f"Total Image Files: {total_images:,}")
        print(f"Unique Images (by hash): {len(unique_file_hashes):,}")
        if duplicate_files > 0:
            print(f"Duplicate Files (same content): {duplicate_files:,}")
        print(f"Total JSON Files: {total_jsons:,}")
        print()

        print(f"✅ Matched Pairs (Image + JSON): {matched_count:,}")
        print(f"⚠️  Unmatched Images (No JSON): {unmatched_images_count:,}")
        print(f"⚠️  Unmatched JSONs (No Image): {unmatched_jsons_count:,}")
        print()

        # Embedding status
        print("🎯 EMBEDDING STATUS")
        print("-" * 80)
        unembedded = self.get_unembedded_pairs(
            scan_results['matched_pairs'],
            db_stats
        )

        # Calculate based on unique images, not total files
        already_embedded = len(db_stats['embedded_hashes'])
        pending_unique = len(unique_file_hashes) - already_embedded

        print(f"Unique Images Available: {len(unique_file_hashes):,}")
        print(f"Already Embedded (Unique): {already_embedded:,}")
        print(f"Pending Embedding (Unique): {pending_unique:,}")

        if len(unique_file_hashes) > 0:
            progress_percentage = (already_embedded / len(unique_file_hashes)) * 100
            progress_bar = self.create_progress_bar(progress_percentage, width=50)
            print(f"\nProgress: {progress_bar} {progress_percentage:.1f}%")

        print()
        print("💡 NOTE: System deduplicates images by hash to avoid embedding duplicates")
        print()
        print("=" * 80)
        print()

        return scan_results, db_stats, unembedded

    def create_progress_bar(self, percentage: float, width: int = 50) -> str:
        """Create a text-based progress bar"""
        filled = int(width * percentage / 100)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    def process_single_image(self, pair: Dict, embedder: FaceEmbedder) -> Tuple[bool, Optional[FaceData], Optional[str]]:
        """Process a single image and return FaceData or error"""
        try:
            # Load metadata
            with open(pair['json'], 'r') as f:
                metadata = json.load(f)

            # Generate embedding
            image_path = str(pair['image'])

            # Extract features from metadata for statistical embedding
            features = {}
            if 'face_features' in metadata:
                features = metadata['face_features']

            embedding = embedder.create_embedding(image_path, features)

            if embedding is None or len(embedding) == 0:
                return False, None, "Failed to generate embedding"

            # Prepare features dict for FaceData
            face_features = {
                'brightness': metadata.get('face_features', {}).get('brightness', 0),
                'contrast': metadata.get('face_features', {}).get('contrast', 0),
                'sex': metadata.get('queryable_attributes', {}).get('sex', 'unknown'),
                'gender': metadata.get('queryable_attributes', {}).get('sex', 'unknown'),
                'age_group': metadata.get('queryable_attributes', {}).get('age_group', 'unknown'),
                'age_estimate': metadata.get('queryable_attributes', {}).get('estimated_age', 'unknown'),
                'skin_tone': metadata.get('queryable_attributes', {}).get('skin_tone', 'unknown'),
                'hair_color': metadata.get('queryable_attributes', {}).get('hair_color', 'unknown'),
                'faces_detected': metadata.get('face_features', {}).get('faces_detected', 0)
            }

            # Create FaceData object
            face_data = FaceData(
                face_id=pair['face_id'],
                file_path=image_path,
                features=face_features,
                embedding=embedding,
                timestamp=datetime.now().isoformat(),
                image_hash=pair.get('hash', '')
            )

            return True, face_data, None

        except Exception as e:
            return False, None, str(e)

    def embed_batch(self, unembedded_pairs: List[Dict]):
        """Embed a batch of images with detailed progress (supports parallel processing)"""
        if not unembedded_pairs:
            print("✅ All images are already embedded!")
            return

        total = len(unembedded_pairs)
        print(f"\n🚀 Starting batch embedding of {total:,} images...")
        print(f"📦 Embedding Model: {self.embedding_model}")
        print(f"⚙️  Workers: {self.workers} {'(parallel)' if self.workers > 1 else '(sequential)'}")
        print("-" * 80)
        print()

        # Initialize embedder
        try:
            embedder = FaceEmbedder(model_name=self.embedding_model)
            print(f"✅ Embedder initialized: {self.embedding_model}")
        except Exception as e:
            print(f"❌ Failed to initialize embedder: {e}")
            return

        # Statistics
        start_time = time.time()
        success_count = 0
        error_count = 0
        errors = []
        processed_count = 0

        print()
        print("Progress:")
        print("-" * 80)

        if self.workers == 1:
            # Sequential processing (original behavior)
            for idx, pair in enumerate(unembedded_pairs, 1):
                success, face_data, error = self.process_single_image(pair, embedder)

                if success:
                    # Store in database
                    self.db_manager.add_face(
                        face_data=face_data,
                        embedding_model=self.embedding_model
                    )
                    success_count += 1
                else:
                    error_count += 1
                    errors.append({
                        'file': pair['image'].name,
                        'error': error
                    })
                    logger.error(f"Error embedding {pair['image'].name}: {error}")

                # Display progress
                percentage = (idx / total) * 100
                progress_bar = self.create_progress_bar(percentage, width=40)
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                eta = avg_time * (total - idx)

                print(f"\r{progress_bar} {percentage:5.1f}% | "
                      f"{idx:,}/{total:,} | "
                      f"Success: {success_count:,} | "
                      f"Errors: {error_count:,} | "
                      f"ETA: {self.format_time(eta)}",
                      end='', flush=True)
        else:
            # Parallel processing with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Submit all tasks
                future_to_pair = {
                    executor.submit(self.process_single_image, pair, embedder): pair
                    for pair in unembedded_pairs
                }

                # Process completed tasks
                for future in as_completed(future_to_pair):
                    pair = future_to_pair[future]
                    processed_count += 1

                    try:
                        success, face_data, error = future.result()

                        if success:
                            # Store in database (thread-safe)
                            self.db_manager.add_face(
                                face_data=face_data,
                                embedding_model=self.embedding_model
                            )
                            with self.lock:
                                success_count += 1
                        else:
                            with self.lock:
                                error_count += 1
                                errors.append({
                                    'file': pair['image'].name,
                                    'error': error
                                })
                            logger.error(f"Error embedding {pair['image'].name}: {error}")

                    except Exception as e:
                        with self.lock:
                            error_count += 1
                            errors.append({
                                'file': pair['image'].name,
                                'error': str(e)
                            })
                        logger.error(f"Exception processing {pair['image'].name}: {e}")

                    # Display progress (thread-safe)
                    with self.lock:
                        percentage = (processed_count / total) * 100
                        progress_bar = self.create_progress_bar(percentage, width=40)
                        elapsed = time.time() - start_time
                        avg_time = elapsed / processed_count
                        eta = avg_time * (total - processed_count)

                        print(f"\r{progress_bar} {percentage:5.1f}% | "
                              f"{processed_count:,}/{total:,} | "
                              f"Success: {success_count:,} | "
                              f"Errors: {error_count:,} | "
                              f"ETA: {self.format_time(eta)}",
                              end='', flush=True)

        print()  # New line after progress
        print("-" * 80)

        # Final summary
        elapsed_time = time.time() - start_time
        print()
        print("=" * 80)
        print("📈 EMBEDDING SUMMARY")
        print("=" * 80)
        print(f"Total Processed: {total:,}")
        print(f"✅ Successfully Embedded: {success_count:,}")
        print(f"❌ Errors: {error_count:,}")
        print(f"⏱️  Total Time: {self.format_time(elapsed_time)}")
        if total > 0:
            print(f"⚡ Average Speed: {elapsed_time/total:.2f} seconds/image")
            if success_count > 0:
                print(f"⚡ Throughput: {success_count/elapsed_time:.2f} images/second")
        print()

        if errors:
            print("Errors encountered:")
            for err in errors[:10]:  # Show first 10 errors
                print(f"  • {err['file']}: {err['error']}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
            print()

        print("=" * 80)

    def format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def run(self, auto_embed: bool = False, stats_only: bool = False):
        """Main run method"""
        # Display statistics
        self.display_stats()

        if stats_only:
            return

        # Check for unembedded images
        db_stats = self.get_database_stats()
        file_stats = self.scan_faces_directory()

        # Get matched pairs and add hash and face_id
        matched_pairs = []
        for pair in file_stats['matched_pairs']:
            img_hash = self.extract_hash_from_filename(pair['image'].name)
            face_id = self.extract_face_id_from_filename(pair['image'].name)
            matched_pairs.append({
                'image': pair['image'],
                'json': pair['json'],
                'base_name': pair['base_name'],
                'hash': img_hash,
                'face_id': face_id
            })

        unembedded = self.get_unembedded_pairs(matched_pairs, db_stats)

        if not unembedded:
            print()
            print("=" * 80)
            print("✅ ALL IMAGES ARE ALREADY EMBEDDED!")
            print("=" * 80)
            return

        if auto_embed:
            # Auto-embed mode - no prompts
            self.embed_batch(unembedded)
        else:
            # Ask user if they want to embed
            print()
            print("❓ EMBED PENDING IMAGES?")
            print("-" * 80)
            print(f"Found {len(unembedded):,} images ready to be embedded.")
            print(f"Embedding model: {self.embedding_model}")
            print(f"Workers: {self.workers}")
            print()

            response = input("Do you want to embed these images now? [y/N]: ").strip().lower()
            should_embed = response in ['y', 'yes']

            if should_embed:
                self.embed_batch(unembedded)
            else:
                print("\n⏭️  Skipping embedding. Run again with --auto-embed to skip this prompt.")


def main():
    """Main entry point"""
    print("""
================================================================================
🔷 EMBEDDING MANAGEMENT CLI
================================================================================

Manage face embeddings in your vector database with ease!

Features:
  • View detailed database and file statistics
  • Identify images that need embedding
  • Batch embed images with progress tracking
  • Support for multiple embedding models
  • Parallel processing for faster embedding

Usage:
  # Interactive mode (asks before embedding)
  python embedding_manager_cli.py

  # Display statistics only
  python embedding_manager_cli.py --stats-only

  # Auto-embed without prompting
  python embedding_manager_cli.py --auto-embed

  # Use specific embedding model
  python embedding_manager_cli.py --model facenet --auto-embed

  # Use parallel processing with 4 workers
  python embedding_manager_cli.py --workers 4 --auto-embed

  # Use custom faces directory
  python embedding_manager_cli.py --faces-dir /path/to/faces --auto-embed

Available embedding models:
  - statistical (default, always available)
  - facenet (requires facenet-pytorch)
  - arcface (requires insightface)
  - deepface (requires deepface)
  - vggface2 (requires deepface)
  - openface (requires deepface)

================================================================================
""")

    parser = argparse.ArgumentParser(description='Manage face embeddings in vector database')

    parser.add_argument(
        '--faces-dir',
        type=str,
        help='Path to faces directory (default: from .env or ./faces)'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['statistical', 'facenet', 'arcface', 'deepface', 'vggface2', 'openface'],
        help='Embedding model to use (default: from .env or statistical)'
    )

    parser.add_argument(
        '--auto-embed',
        action='store_true',
        help='Automatically embed all pending images without prompting'
    )

    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Display statistics only, do not embed'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for embedding (default: 1, recommended: 2-4)'
    )

    args = parser.parse_args()

    try:
        cli = EmbeddingManagerCLI(
            faces_dir=args.faces_dir,
            embedding_model=args.model,
            quiet=args.quiet,
            workers=args.workers
        )
        cli.run(auto_embed=args.auto_embed, stats_only=args.stats_only)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        logger.exception("Fatal error in main")
        sys.exit(1)


if __name__ == "__main__":
    main()

