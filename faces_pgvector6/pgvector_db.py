#!/usr/bin/env python3
"""
PostgreSQL + pgvector Database Manager for Face Recognition System

This module provides database operations using PostgreSQL with pgvector extension:
- Vector similarity search with cosine/L2/inner product distances
- Metadata filtering and hybrid search
- Efficient batch operations
- Connection pooling for performance

Dependencies:
- PostgreSQL 12+ with pgvector extension
- psycopg2-binary for database connectivity
- python-dotenv for configuration
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import execute_batch
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class PgVectorDatabaseManager:
    """
    PostgreSQL + pgvector Database Manager

    Manages all database operations for the face recognition system including:
    - Database initialization and connection pooling
    - Adding face embeddings with metadata
    - Vector similarity search (cosine, L2, inner product)
    - Metadata filtering and hybrid search
    - Duplicate detection via image hashing
    - Batch operations for efficiency
    """

    def __init__(self, config):
        """
        Initialize the database manager

        Args:
            config: SystemConfig object with database settings
        """
        self.config = config
        self.connection_pool = None
        self.initialized = False

        # Database connection parameters - prioritize env vars, then config
        self.db_params = {
            'host': os.getenv('POSTGRES_HOST') or getattr(config, 'db_host', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', getattr(config, 'db_port', 5432))),
            'database': os.getenv('POSTGRES_DB') or getattr(config, 'db_name', 'vector_db'),
            'user': os.getenv('POSTGRES_USER') or getattr(config, 'db_user', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD') or getattr(config, 'db_password', '')
        }

        # Connection pool settings
        self.min_connections = int(os.getenv('DB_MIN_CONNECTIONS', 1))
        self.max_connections = int(os.getenv('DB_MAX_CONNECTIONS', 10))

        # Vector settings
        self.embedding_dimension = 512  # Default dimension (matches schema)
        self.distance_metric = os.getenv('DISTANCE_METRIC', 'cosine')

    def initialize(self) -> bool:
        """
        Initialize PostgreSQL connection pool

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create connection pool
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                self.min_connections,
                self.max_connections,
                **self.db_params
            )

            if self.connection_pool:
                logger.info("PostgreSQL connection pool created successfully")

                # Test connection and verify pgvector extension
                conn = self.connection_pool.getconn()
                try:
                    cursor = conn.cursor()

                    # Check if pgvector extension is available
                    cursor.execute(
                        "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
                    )
                    if cursor.fetchone()[0] == 0:
                        logger.error("pgvector extension not found. Run: CREATE EXTENSION vector;")
                        return False

                    # Check if faces table exists
                    cursor.execute(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_name = 'faces'"
                    )
                    if cursor.fetchone()[0] == 0:
                        logger.error("faces table not found. Run schema.sql to create it.")
                        return False

                    cursor.close()
                    logger.info("Database initialized successfully")
                    self.initialized = True
                    return True

                finally:
                    self.connection_pool.putconn(conn)

            return False

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False

    def get_connection(self):
        """Get a connection from the pool"""
        if not self.connection_pool:
            raise Exception("Connection pool not initialized")
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """Return a connection to the pool"""
        if self.connection_pool:
            self.connection_pool.putconn(conn)

    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy types and other non-serializable types to JSON-serializable types

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _pad_embedding(self, embedding: List[float]) -> List[float]:
        """
        Pad embedding to match the expected dimension (512)

        Args:
            embedding: Original embedding vector

        Returns:
            Padded embedding vector
        """
        if len(embedding) < self.embedding_dimension:
            # Pad with zeros
            padded = list(embedding) + [0.0] * (self.embedding_dimension - len(embedding))
            return padded
        elif len(embedding) > self.embedding_dimension:
            # Truncate (shouldn't happen normally)
            logger.warning(f"Embedding dimension {len(embedding)} exceeds expected {self.embedding_dimension}, truncating")
            return embedding[:self.embedding_dimension]
        return embedding

    def add_face(self, face_data, embedding_model: str = "statistical") -> bool:
        """
        Add a face to the database

        Args:
            face_data: FaceData object containing face information
            embedding_model: Name of the embedding model used

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.warning("Database not initialized")
            return False

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Prepare embedding (pad if necessary)
            embedding = None
            if face_data.embedding:
                embedding = self._pad_embedding(face_data.embedding)

            # Extract metadata from features
            features = face_data.features

            # Prepare metadata JSONB (store all features for flexibility)
            # Convert numpy types to Python native types first
            features_serializable = self._convert_to_json_serializable(features)
            metadata_json = json.dumps(features_serializable)

            # Insert face data
            query = """
                INSERT INTO faces (
                    face_id, file_path, timestamp, image_hash, embedding_model,
                    embedding, age_estimate, gender, brightness, contrast, sharpness,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (face_id) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    timestamp = EXCLUDED.timestamp,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """

            cursor.execute(query, (
                face_data.face_id,
                face_data.file_path,
                face_data.timestamp,
                face_data.image_hash,
                embedding_model,
                embedding,
                features.get('age_estimate'),
                features.get('gender'),
                features.get('brightness'),
                features.get('contrast'),
                features.get('sharpness'),
                metadata_json
            ))

            conn.commit()
            cursor.close()
            logger.debug(f"Added face {face_data.face_id} to database")
            return True

        except Exception as e:
            logger.error(f"Error adding face to database: {e}")
            if conn:
                conn.rollback()
            return False

        finally:
            if conn:
                self.return_connection(conn)

    def add_faces_batch(self, face_data_list: List[Tuple], batch_size: int = 100) -> int:
        """
        Add multiple faces in batch for better performance

        Args:
            face_data_list: List of tuples (face_data, embedding_model)
            batch_size: Number of records to insert per batch

        Returns:
            int: Number of faces successfully added
        """
        if not self.initialized:
            logger.warning("Database not initialized")
            return 0

        conn = None
        added_count = 0

        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            query = """
                INSERT INTO faces (
                    face_id, file_path, timestamp, image_hash, embedding_model,
                    embedding, age_estimate, gender, brightness, contrast, sharpness,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (face_id) DO NOTHING
            """

            # Prepare data for batch insert
            batch_data = []
            for face_data, embedding_model in face_data_list:
                embedding = None
                if face_data.embedding:
                    embedding = self._pad_embedding(face_data.embedding)

                features = face_data.features
                features_serializable = self._convert_to_json_serializable(features)

                batch_data.append((
                    face_data.face_id,
                    face_data.file_path,
                    face_data.timestamp,
                    face_data.image_hash,
                    embedding_model,
                    embedding,
                    features_serializable.get('age_estimate'),
                    features_serializable.get('gender'),
                    features_serializable.get('brightness'),
                    features_serializable.get('contrast'),
                    features_serializable.get('sharpness'),
                    json.dumps(features_serializable)
                ))

            # Execute batch insert
            execute_batch(cursor, query, batch_data, page_size=batch_size)
            added_count = len(batch_data)

            conn.commit()
            cursor.close()
            logger.info(f"Added {added_count} faces in batch")
            return added_count

        except Exception as e:
            logger.error(f"Error in batch insert: {e}")
            if conn:
                conn.rollback()
            return 0

        finally:
            if conn:
                self.return_connection(conn)

    def search_faces(self, query_embedding: List[float], n_results: int = 10,
                    metadata_filter: Optional[Dict[str, Any]] = None,
                    distance_metric: str = 'cosine') -> List[Dict[str, Any]]:
        """
        Search for similar faces using vector similarity

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            metadata_filter: Optional metadata filters (e.g., {'gender': 'female'})
            distance_metric: Distance metric ('cosine', 'l2', 'inner_product')

        Returns:
            List of dictionaries containing face information and distances
        """
        if not self.initialized:
            return []

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Pad query embedding
            query_embedding = self._pad_embedding(query_embedding)

            # Select distance operator
            if distance_metric == 'cosine':
                distance_op = '<=>'
            elif distance_metric == 'l2':
                distance_op = '<->'
            elif distance_metric == 'inner_product':
                distance_op = '<#>'
            else:
                distance_op = '<=>'  # Default to cosine

            # Build query
            query = f"""
                SELECT
                    face_id, file_path, timestamp, image_hash,
                    embedding_model, age_estimate, gender, brightness,
                    contrast, sharpness, metadata,
                    embedding {distance_op} %s::vector AS distance
                FROM faces
                WHERE embedding IS NOT NULL
            """

            params = [query_embedding]

            # Add metadata filters if provided
            if metadata_filter:
                for key, value in metadata_filter.items():
                    if isinstance(value, dict):
                        # Handle operators like {'$gt': 25}
                        for op, val in value.items():
                            if op == '$gt':
                                query += f" AND (metadata->>%s)::float > %s"
                            elif op == '$lt':
                                query += f" AND (metadata->>%s)::float < %s"
                            elif op == '$gte':
                                query += f" AND (metadata->>%s)::float >= %s"
                            elif op == '$lte':
                                query += f" AND (metadata->>%s)::float <= %s"
                            params.extend([key, val])
                    else:
                        # Direct equality
                        query += f" AND metadata->>%s = %s"
                        params.extend([key, str(value)])

            query += f" ORDER BY distance LIMIT %s"
            params.append(n_results)

            cursor.execute(query, params)
            results = cursor.fetchall()

            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'id': row[0],  # face_id
                    'metadata': {
                        'face_id': row[0],
                        'file_path': row[1],
                        'timestamp': row[2],
                        'image_hash': row[3],
                        'embedding_model': row[4],
                        'age_estimate': row[5],
                        'gender': row[6],
                        'brightness': row[7],
                        'contrast': row[8],
                        'sharpness': row[9],
                        **(row[10] if row[10] else {})  # Additional metadata from JSONB
                    },
                    'distance': float(row[11])
                })

            cursor.close()
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching faces: {e}")
            return []

        finally:
            if conn:
                self.return_connection(conn)

    def search_by_metadata(self, metadata_filter: Dict[str, Any],
                          n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search faces by metadata only (no vector similarity)

        Args:
            metadata_filter: Metadata filters (e.g., {'gender': 'female', 'age_estimate': {'$gt': 25}})
            n_results: Number of results to return

        Returns:
            List of dictionaries containing face information
        """
        if not self.initialized:
            return []

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            query = """
                SELECT
                    face_id, file_path, timestamp, image_hash,
                    embedding_model, age_estimate, gender, brightness,
                    contrast, sharpness, metadata
                FROM faces
                WHERE 1=1
            """

            params = []

            # Add metadata filters
            for key, value in metadata_filter.items():
                if isinstance(value, dict):
                    # Handle operators
                    for op, val in value.items():
                        if op == '$gt':
                            query += f" AND (metadata->>%s)::float > %s"
                        elif op == '$lt':
                            query += f" AND (metadata->>%s)::float < %s"
                        elif op == '$gte':
                            query += f" AND (metadata->>%s)::float >= %s"
                        elif op == '$lte':
                            query += f" AND (metadata->>%s)::float <= %s"
                        params.extend([key, val])
                else:
                    query += f" AND metadata->>%s = %s"
                    params.extend([key, str(value)])

            query += f" ORDER BY created_at DESC LIMIT %s"
            params.append(n_results)

            cursor.execute(query, params)
            results = cursor.fetchall()

            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'id': row[0],
                    'metadata': {
                        'face_id': row[0],
                        'file_path': row[1],
                        'timestamp': row[2],
                        'image_hash': row[3],
                        'embedding_model': row[4],
                        'age_estimate': row[5],
                        'gender': row[6],
                        'brightness': row[7],
                        'contrast': row[8],
                        'sharpness': row[9],
                        **(row[10] if row[10] else {})
                    }
                })

            cursor.close()
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []

        finally:
            if conn:
                self.return_connection(conn)

    def check_duplicate(self, image_hash: str) -> bool:
        """
        Check if an image with the given hash already exists

        Args:
            image_hash: Hash of the image

        Returns:
            bool: True if duplicate exists, False otherwise
        """
        if not self.initialized:
            return False

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM faces WHERE image_hash = %s",
                (image_hash,)
            )
            count = cursor.fetchone()[0]
            cursor.close()

            return count > 0

        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False

        finally:
            if conn:
                self.return_connection(conn)

    def get_count(self) -> int:
        """
        Get total number of faces in database

        Returns:
            int: Total count of faces
        """
        if not self.initialized:
            return 0

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM faces")
            count = cursor.fetchone()[0]
            cursor.close()

            return count

        except Exception as e:
            logger.error(f"Error getting count: {e}")
            return 0

        finally:
            if conn:
                self.return_connection(conn)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary containing database statistics
        """
        if not self.initialized:
            return {}

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Use the helper function from schema
            cursor.execute("SELECT * FROM get_database_stats()")
            result = cursor.fetchone()

            if result:
                stats = {
                    'total_faces': result[0],
                    'faces_with_embeddings': result[1],
                    'embedding_models': result[2] if result[2] else [],
                    'oldest_face': result[3].isoformat() if result[3] else None,
                    'newest_face': result[4].isoformat() if result[4] else None,
                    'database_size': result[5]
                }
            else:
                stats = {}

            cursor.close()
            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

        finally:
            if conn:
                self.return_connection(conn)

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the database (compatibility with ChromaDB interface)

        Returns:
            Dictionary containing database information
        """
        if not self.initialized:
            return {}

        try:
            count = self.get_count()
            return {
                'name': self.db_params['database'],
                'count': count,
                'path': f"{self.db_params['host']}:{self.db_params['port']}/{self.db_params['database']}"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def delete_face(self, face_id: str) -> bool:
        """
        Delete a face from the database

        Args:
            face_id: ID of the face to delete

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            return False

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM faces WHERE face_id = %s", (face_id,))
            conn.commit()
            cursor.close()

            logger.info(f"Deleted face {face_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting face: {e}")
            if conn:
                conn.rollback()
            return False

        finally:
            if conn:
                self.return_connection(conn)

    def reset_database(self) -> bool:
        """
        Clear all data from the database

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            return False

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("TRUNCATE TABLE faces RESTART IDENTITY")
            conn.commit()
            cursor.close()

            logger.info("Database reset successfully")
            return True

        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            if conn:
                conn.rollback()
            return False

        finally:
            if conn:
                self.return_connection(conn)

    def close(self):
        """Close all database connections"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("All database connections closed")
