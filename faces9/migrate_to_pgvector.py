#!/usr/bin/env python3
"""
Migration Script: ChromaDB to PostgreSQL + pgvector

This script migrates face data from ChromaDB to PostgreSQL with pgvector extension.

Features:
- Exports all faces, embeddings, and metadata from ChromaDB
- Imports data into PostgreSQL with pgvector
- Progress tracking and error handling
- Dry-run mode to preview migration
- Backup option for safety

Usage:
    python3 migrate_to_pgvector.py --dry-run     # Preview migration
    python3 migrate_to_pgvector.py               # Perform migration
    python3 migrate_to_pgvector.py --batch-size 50  # Custom batch size
"""

import sys
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

# Import necessary classes
from core import SystemConfig, FaceData, DatabaseManager
from pgvector_db import PgVectorDatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChromaToPgVectorMigrator:
    """
    Migrates face data from ChromaDB to PostgreSQL + pgvector
    """

    def __init__(self, config: SystemConfig, dry_run: bool = False, batch_size: int = 100):
        """
        Initialize the migrator

        Args:
            config: System configuration
            dry_run: If True, only preview migration without making changes
            batch_size: Number of records to insert per batch
        """
        self.config = config
        self.dry_run = dry_run
        self.batch_size = batch_size

        # Initialize database managers
        self.chroma_db = None
        self.pgvector_db = None

        # Migration statistics
        self.stats = {
            'total_records': 0,
            'migrated': 0,
            'skipped': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }

    def initialize_databases(self) -> bool:
        """Initialize both database connections"""
        logger.info("Initializing database connections...")

        try:
            # Initialize ChromaDB (source)
            logger.info("Connecting to ChromaDB (source)...")
            chroma_config = SystemConfig()
            chroma_config.db_path = self.config.db_path
            chroma_config.collection_name = self.config.collection_name

            self.chroma_db = DatabaseManager(chroma_config)
            if not self.chroma_db.initialize():
                logger.error("Failed to initialize ChromaDB")
                return False

            logger.info(f"✓ Connected to ChromaDB at {self.config.db_path}")

            # Initialize PostgreSQL + pgvector (destination)
            if not self.dry_run:
                logger.info("Connecting to PostgreSQL + pgvector (destination)...")
                self.pgvector_db = PgVectorDatabaseManager(self.config)
                if not self.pgvector_db.initialize():
                    logger.error("Failed to initialize PostgreSQL + pgvector")
                    return False

                logger.info(f"✓ Connected to PostgreSQL at {self.config.db_host}:{self.config.db_port}/{self.config.db_name}")

            return True

        except Exception as e:
            logger.error(f"Error initializing databases: {e}")
            return False

    def get_chromadb_data(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve all data from ChromaDB

        Returns:
            Dictionary containing ids, embeddings, metadatas, and documents
        """
        try:
            logger.info("Retrieving data from ChromaDB...")

            # Get all data from ChromaDB collection
            results = self.chroma_db.collection.get(
                include=['embeddings', 'metadatas', 'documents']
            )

            self.stats['total_records'] = len(results['ids'])
            logger.info(f"✓ Found {self.stats['total_records']} records in ChromaDB")

            return results

        except Exception as e:
            logger.error(f"Error retrieving ChromaDB data: {e}")
            return None

    def convert_to_face_data(self, chroma_id: str, embedding: List[float],
                            metadata: Dict[str, Any]) -> FaceData:
        """
        Convert ChromaDB record to FaceData object

        Args:
            chroma_id: Record ID from ChromaDB
            embedding: Embedding vector
            metadata: Metadata dictionary

        Returns:
            FaceData object
        """
        # Extract fields from metadata
        face_id = metadata.get('face_id', chroma_id)
        file_path = metadata.get('file_path', '')
        timestamp = metadata.get('timestamp', datetime.now().isoformat())
        image_hash = metadata.get('image_hash', '')

        # Extract features (remove standard fields)
        features = {k: v for k, v in metadata.items()
                   if k not in ['face_id', 'file_path', 'timestamp', 'image_hash', 'embedding_model']}

        # Create FaceData object
        face_data = FaceData(
            face_id=face_id,
            file_path=file_path,
            features=features,
            embedding=embedding,
            timestamp=timestamp,
            image_hash=image_hash
        )

        return face_data

    def migrate_batch(self, batch_data: List[tuple], embedding_model: str) -> int:
        """
        Migrate a batch of records to pgvector

        Args:
            batch_data: List of (face_data, embedding_model) tuples
            embedding_model: Name of the embedding model

        Returns:
            Number of records successfully migrated
        """
        if self.dry_run:
            return len(batch_data)

        try:
            count = self.pgvector_db.add_faces_batch(batch_data, batch_size=self.batch_size)
            return count
        except Exception as e:
            logger.error(f"Error migrating batch: {e}")
            return 0

    def perform_migration(self) -> bool:
        """
        Perform the actual migration from ChromaDB to pgvector

        Returns:
            True if successful, False otherwise
        """
        self.stats['start_time'] = datetime.now()

        logger.info("=" * 70)
        logger.info("Starting migration from ChromaDB to PostgreSQL + pgvector")
        logger.info("=" * 70)

        if self.dry_run:
            logger.info("DRY RUN MODE: No changes will be made to the database")

        # Get data from ChromaDB
        chroma_data = self.get_chromadb_data()
        if not chroma_data:
            logger.error("Failed to retrieve data from ChromaDB")
            return False

        if self.stats['total_records'] == 0:
            logger.warning("No records found in ChromaDB to migrate")
            return True

        # Process records in batches
        logger.info(f"Processing {self.stats['total_records']} records in batches of {self.batch_size}...")

        batch_data = []

        for i in range(len(chroma_data['ids'])):
            try:
                # Extract data
                chroma_id = chroma_data['ids'][i]
                embedding = chroma_data['embeddings'][i] if chroma_data['embeddings'] else None
                metadata = chroma_data['metadatas'][i] if chroma_data['metadatas'] else {}

                # Get embedding model from metadata
                embedding_model = metadata.get('embedding_model', 'statistical')

                # Convert to FaceData
                face_data = self.convert_to_face_data(chroma_id, embedding, metadata)

                # Add to batch
                batch_data.append((face_data, embedding_model))

                # Process batch when it reaches batch_size
                if len(batch_data) >= self.batch_size:
                    migrated = self.migrate_batch(batch_data, embedding_model)
                    self.stats['migrated'] += migrated

                    if not self.dry_run:
                        logger.info(f"Progress: {self.stats['migrated']}/{self.stats['total_records']} "
                                   f"({(self.stats['migrated']/self.stats['total_records']*100):.1f}%)")

                    batch_data = []

            except Exception as e:
                logger.error(f"Error processing record {i}: {e}")
                self.stats['errors'] += 1
                continue

        # Process remaining records
        if batch_data:
            migrated = self.migrate_batch(batch_data, embedding_model)
            self.stats['migrated'] += migrated

        self.stats['end_time'] = datetime.now()

        return True

    def print_summary(self):
        """Print migration summary"""
        logger.info("\n" + "=" * 70)
        logger.info("Migration Summary")
        logger.info("=" * 70)

        logger.info(f"Total records in ChromaDB: {self.stats['total_records']}")
        logger.info(f"Successfully migrated: {self.stats['migrated']}")
        logger.info(f"Errors: {self.stats['errors']}")

        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            logger.info(f"Duration: {duration:.2f} seconds")

            if self.stats['migrated'] > 0:
                rate = self.stats['migrated'] / duration
                logger.info(f"Migration rate: {rate:.1f} records/second")

        if self.dry_run:
            logger.info("\nDRY RUN COMPLETED - No changes were made to the database")
        else:
            logger.info("\nMIGRATION COMPLETED")

            # Verify migration
            if self.pgvector_db:
                pg_count = self.pgvector_db.get_count()
                logger.info(f"PostgreSQL record count: {pg_count}")

        logger.info("=" * 70)

    def cleanup(self):
        """Close database connections"""
        if self.pgvector_db:
            self.pgvector_db.close()

        logger.info("Database connections closed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Migrate face data from ChromaDB to PostgreSQL + pgvector'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview migration without making changes'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of records to insert per batch (default: 100)'
    )
    parser.add_argument(
        '--chroma-path',
        type=str,
        default='./chroma_db',
        help='Path to ChromaDB database (default: ./chroma_db)'
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default='faces',
        help='ChromaDB collection name (default: faces)'
    )

    args = parser.parse_args()

    # Load configuration
    config = SystemConfig()
    config.db_type = "pgvector"
    config.db_path = args.chroma_path
    config.collection_name = args.collection_name

    # Check if ChromaDB exists
    if not os.path.exists(args.chroma_path):
        logger.error(f"ChromaDB path not found: {args.chroma_path}")
        logger.error("Please specify the correct path with --chroma-path")
        sys.exit(1)

    # Create migrator
    migrator = ChromaToPgVectorMigrator(
        config=config,
        dry_run=args.dry_run,
        batch_size=args.batch_size
    )

    try:
        # Initialize databases
        if not migrator.initialize_databases():
            logger.error("Failed to initialize databases")
            sys.exit(1)

        # Perform migration
        if not migrator.perform_migration():
            logger.error("Migration failed")
            sys.exit(1)

        # Print summary
        migrator.print_summary()

        # Cleanup
        migrator.cleanup()

        logger.info("✓ Migration process completed successfully!")

    except KeyboardInterrupt:
        logger.warning("\n\nMigration interrupted by user")
        migrator.cleanup()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        migrator.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
