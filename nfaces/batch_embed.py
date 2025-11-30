#!/usr/bin/env python3
"""
Batch Embedding Tool for Face Images

Fast, parallel embedding of all face images in the faces directory.
Features:
- Multi-threaded parallel processing
- Choose embedding model (statistical, facenet, arcface, deepface, etc.)
- Progress tracking with ETA
- Resume capability (skip already embedded faces)
- Detailed statistics and reporting
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging

# Import core modules
from core import (
    SystemConfig,
    SystemStats,
    FaceProcessor,
    FaceAnalyzer,
    FaceEmbedder,
    AVAILABLE_MODELS
)
from pgvector_db import PgVectorDatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchEmbedder:
    """
    Batch embedding processor with parallel execution
    """

    def __init__(self, config: SystemConfig, embedding_model: str, max_workers: int = 4):
        self.config = config
        self.config.embedding_model = embedding_model
        self.max_workers = max_workers
        self.stats = SystemStats()

        # Initialize database
        self.db_manager = PgVectorDatabaseManager(self.config)
        if not self.db_manager.initialize():
            raise RuntimeError("Failed to initialize database")

        # Initialize processor
        self.processor = FaceProcessor(self.config, self.stats, self.db_manager)

        # Statistics
        self.total_files = 0
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.start_time = None

    def get_all_face_files(self) -> List[str]:
        """Get all face image files"""
        face_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            face_files.extend(Path(self.config.faces_dir).rglob(ext))

        # Filter out macOS metadata files
        face_files = [str(f) for f in face_files if not f.name.startswith('._')]
        return face_files

    def get_new_files_only(self) -> List[str]:
        """Get only files that haven't been embedded yet"""
        return self.processor.get_new_files_only()

    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file and return result"""
        result = {
            'file': os.path.basename(file_path),
            'success': False,
            'error': None,
            'duration': 0
        }

        start_time = time.time()

        try:
            success = self.processor.process_face_file(file_path)
            result['success'] = success
            result['duration'] = time.time() - start_time

            if success:
                self.processed_count += 1
            else:
                self.error_count += 1

        except Exception as e:
            result['error'] = str(e)
            result['duration'] = time.time() - start_time
            self.error_count += 1
            logger.error(f"Error processing {os.path.basename(file_path)}: {e}")

        return result

    def print_progress(self, current: int, total: int, elapsed: float):
        """Print progress bar with ETA"""
        if total == 0:
            return

        percentage = (current / total) * 100
        bar_length = 40
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        # Calculate ETA
        if current > 0:
            avg_time_per_file = elapsed / current
            remaining = total - current
            eta_seconds = avg_time_per_file * remaining
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "calculating..."

        # Get current stats
        stats = self.stats.get_stats()

        print(f"\r[{bar}] {current}/{total} ({percentage:.1f}%) | "
              f"Success: {self.processed_count} | "
              f"Errors: {self.error_count} | "
              f"ETA: {eta}", end='', flush=True)

    def embed_all(self, skip_existing: bool = True) -> Dict[str, Any]:
        """
        Embed all face images using parallel processing

        Args:
            skip_existing: If True, skip already embedded images

        Returns:
            Dictionary with embedding statistics
        """
        logger.info("=" * 80)
        logger.info("BATCH FACE EMBEDDING")
        logger.info("=" * 80)
        logger.info(f"Embedding Model: {self.config.embedding_model}")
        logger.info(f"Max Workers: {self.max_workers}")
        logger.info(f"Faces Directory: {self.config.faces_dir}")

        # Get files to process
        if skip_existing:
            logger.info("Scanning for new files (not yet in database)...")
            files_to_process = self.get_new_files_only()
        else:
            logger.info("Scanning for all face files...")
            files_to_process = self.get_all_face_files()

        self.total_files = len(files_to_process)

        if self.total_files == 0:
            logger.info("✅ No files to process!")
            return {
                'total': 0,
                'processed': 0,
                'errors': 0,
                'duration': 0
            }

        logger.info(f"Found {self.total_files} files to embed")
        logger.info("=" * 80)

        self.start_time = time.time()
        completed = 0

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path
                for file_path in files_to_process
            }

            # Process completed tasks
            for future in as_completed(future_to_file):
                result = future.result()
                completed += 1

                elapsed = time.time() - self.start_time
                self.print_progress(completed, self.total_files, elapsed)

        # Final newline after progress bar
        print()

        # Calculate final statistics
        total_duration = time.time() - self.start_time

        logger.info("=" * 80)
        logger.info("EMBEDDING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total Files: {self.total_files}")
        logger.info(f"Successfully Embedded: {self.processed_count}")
        logger.info(f"Errors: {self.error_count}")
        logger.info(f"Total Duration: {timedelta(seconds=int(total_duration))}")

        if self.processed_count > 0:
            avg_time = total_duration / self.processed_count
            rate = self.processed_count / total_duration
            logger.info(f"Average Time per Image: {avg_time:.2f}s")
            logger.info(f"Processing Rate: {rate:.2f} images/sec")

        # Get database statistics
        db_stats = self.db_manager.get_statistics()
        logger.info("=" * 80)
        logger.info("DATABASE STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total Faces in Database: {db_stats.get('total_faces', 0)}")
        logger.info(f"Embedding Model: {self.config.embedding_model}")
        logger.info("=" * 80)

        return {
            'total': self.total_files,
            'processed': self.processed_count,
            'errors': self.error_count,
            'duration': total_duration,
            'avg_time_per_image': total_duration / self.processed_count if self.processed_count > 0 else 0,
            'processing_rate': self.processed_count / total_duration if total_duration > 0 else 0
        }


def list_available_models():
    """List all available embedding models"""
    print("=" * 80)
    print("AVAILABLE EMBEDDING MODELS")
    print("=" * 80)

    for model_name, available in AVAILABLE_MODELS.items():
        status = "✅ Available" if available else "❌ Not installed"
        print(f"{model_name:15} - {status}")

    print("=" * 80)
    print("\nTo install missing models, see requirements.txt")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Batch embed all face images with parallel processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed all new images using configured model (from system_config.json)
  %(prog)s

  # Embed using FaceNet model with 8 parallel workers
  %(prog)s --model facenet --workers 8

  # Re-embed ALL images (including already embedded ones)
  %(prog)s --force --model arcface

  # List available models
  %(prog)s --list-models

  # Use specific config file
  %(prog)s --config my_config.json --model deepface

Available Models:
  - statistical  : Basic statistical features (always available, fast)
  - facenet      : Deep learning model (good accuracy, requires torch)
  - arcface      : State-of-the-art (best accuracy, requires insightface)
  - deepface     : Multi-purpose (requires deepface)
  - vggface2     : Deep CNN (requires deepface)
  - openface     : Lightweight (requires deepface)
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Embedding model to use (default: from config file)'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='system_config.json',
        help='Path to config file (default: system_config.json)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Re-embed all images, including already embedded ones'
    )

    parser.add_argument(
        '--list-models', '-l',
        action='store_true',
        help='List available embedding models and exit'
    )

    args = parser.parse_args()

    # List models and exit
    if args.list_models:
        list_available_models()
        return 0

    try:
        # Load configuration
        config = SystemConfig.from_file(args.config)

        # Override embedding model if specified
        if args.model:
            embedding_model = args.model
        else:
            embedding_model = config.embedding_model

        # Validate model
        if embedding_model not in AVAILABLE_MODELS:
            logger.error(f"Unknown embedding model: {embedding_model}")
            logger.info("Available models:")
            list_available_models()
            return 1

        if not AVAILABLE_MODELS[embedding_model]:
            logger.error(f"Model '{embedding_model}' is not installed")
            logger.info("Please install required dependencies (see requirements.txt)")
            return 1

        # Create batch embedder
        embedder = BatchEmbedder(
            config=config,
            embedding_model=embedding_model,
            max_workers=args.workers
        )

        # Run embedding
        skip_existing = not args.force
        results = embedder.embed_all(skip_existing=skip_existing)

        # Exit with appropriate code
        if results['errors'] > 0:
            logger.warning(f"Completed with {results['errors']} errors")
            return 1
        else:
            logger.info("✅ All images embedded successfully!")
            return 0

    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
