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
        import time
        try:
            start_time = time.time()
            logger.info(f"Attempting to connect to PostgreSQL at {self.db_params.get('host', 'localhost')}:{self.db_params.get('port', 5432)}")
            logger.info(f"Database: {self.db_params.get('database', 'N/A')}, User: {self.db_params.get('user', 'N/A')}")

            # Create connection pool
            pool_start = time.time()
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                self.min_connections,
                self.max_connections,
                **self.db_params
            )
            logger.info(f"✓ Connection pool created in {time.time() - pool_start:.2f}s")

            if self.connection_pool:
                # Test connection and verify pgvector extension
                logger.info("Testing database connection...")
                conn = self.connection_pool.getconn()
                try:
                    cursor = conn.cursor()

                    # Check if pgvector extension is available
                    ext_start = time.time()
                    cursor.execute(
                        "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
                    )
                    if cursor.fetchone()[0] == 0:
                        logger.error("✗ pgvector extension not found. Run: CREATE EXTENSION vector;")
                        return False
                    logger.info(f"✓ pgvector extension found ({time.time() - ext_start:.2f}s)")

                    # Check if faces table exists
                    table_start = time.time()
                    cursor.execute(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_name = 'faces'"
                    )
                    if cursor.fetchone()[0] == 0:
                        logger.error("✗ faces table not found. Run schema.sql to create it.")
                        return False
                    logger.info(f"✓ faces table found ({time.time() - table_start:.2f}s)")

                    cursor.close()
                    self.initialized = True

                    # Check and create indexes for better performance
                    logger.info("=" * 60)
                    logger.info("STARTING INDEX VERIFICATION")
                    logger.info("=" * 60)
                    logger.info("Checking if database indexes exist and are up-to-date...")
                    logger.info("This ensures optimal query performance for similarity search")
                    index_start = time.time()

                    # Call the index verification function
                    self._ensure_indexes()

                    logger.info("=" * 60)
                    logger.info(f"INDEX VERIFICATION COMPLETED in {time.time() - index_start:.2f}s")
                    logger.info("=" * 60)

                    logger.info(f"✓ Database initialized successfully (total: {time.time() - start_time:.2f}s)")
                    return True

                finally:
                    self.connection_pool.putconn(conn)

            return False

        except psycopg2.OperationalError as e:
            logger.error(f"✗ PostgreSQL connection failed: {e}")
            logger.error("  Please check:")
            logger.error("  1. PostgreSQL is running (systemctl status postgresql)")
            logger.error("  2. Connection parameters are correct")
            logger.error("  3. User has proper permissions")
            return False
        except Exception as e:
            logger.error(f"✗ Failed to initialize database: {e}")
            return False

    def _ensure_indexes(self):
        """Create indexes if they don't exist for optimal performance"""
        import time
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Get all existing indexes in one query (much faster than multiple checks)
            logger.info("→ Querying PostgreSQL for existing indexes on 'faces' table...")
            check_start = time.time()
            cursor.execute("""
                SELECT indexname, indexdef FROM pg_indexes
                WHERE tablename = 'faces'
                ORDER BY indexname
            """)
            existing_indexes_data = cursor.fetchall()
            existing_indexes = {row[0] for row in existing_indexes_data}
            check_time = time.time() - check_start

            logger.info(f"→ Query completed in {check_time:.3f}s")
            logger.info(f"→ Found {len(existing_indexes)} existing indexes:")
            logger.info("")

            # Show all existing indexes
            for idx_name, idx_def in existing_indexes_data:
                # Shorten the definition for readability
                short_def = idx_def.replace('public.faces', 'faces')
                if 'hnsw' in short_def.lower():
                    idx_type = "[VECTOR-HNSW]"
                elif 'gin' in short_def.lower():
                    idx_type = "[JSONB-GIN]"
                elif 'btree' in short_def.lower():
                    idx_type = "[BTREE]"
                else:
                    idx_type = "[OTHER]"
                logger.info(f"   {idx_type:15s} {idx_name}")

            logger.info("")
            logger.info("→ Checking which required indexes are missing...")

            # Define required indexes (multi-model support)
            # Note: Vector indexes are created in schema.sql, we just check metadata indexes here
            required_indexes = {
                'Metadata: sex': ['faces_metadata_sex_idx'],
                'Metadata: age_group': ['faces_metadata_age_group_idx'],
                'Timestamp': ['faces_timestamp_idx', 'idx_timestamp', 'idx_created_at']
            }

            indexes_to_create = []

            # Check if multi-model vector indexes exist (created by schema.sql)
            multimodel_indexes = [
                'idx_embedding_facenet_hnsw',
                'idx_embedding_arcface_hnsw',
                'idx_embedding_vggface2_hnsw',
                'idx_embedding_insightface_hnsw',
                'idx_embedding_statistical_hnsw'
            ]

            found_vector_indexes = sum(1 for idx in multimodel_indexes if idx in existing_indexes)
            if found_vector_indexes > 0:
                logger.info(f"   ✓ Found: {found_vector_indexes}/5 multi-model vector indexes")
            else:
                logger.info("   ⚠ No multi-model vector indexes found (run schema.sql to create them)")

            # Check metadata indexes
            if not any(idx in existing_indexes for idx in required_indexes['Metadata: sex']):
                indexes_to_create.append(('faces_metadata_sex_idx', """
                    CREATE INDEX IF NOT EXISTS faces_metadata_sex_idx
                    ON faces ((metadata->>'sex'))
                """))
                logger.info("   ✗ Missing: Metadata sex index")
            else:
                logger.info("   ✓ Found: Metadata sex index")

            if not any(idx in existing_indexes for idx in required_indexes['Metadata: age_group']):
                indexes_to_create.append(('faces_metadata_age_group_idx', """
                    CREATE INDEX IF NOT EXISTS faces_metadata_age_group_idx
                    ON faces ((metadata->>'age_group'))
                """))
                logger.info("   ✗ Missing: Metadata age_group index")
            else:
                logger.info("   ✓ Found: Metadata age_group index")

            if not any(idx in existing_indexes for idx in required_indexes['Timestamp']):
                indexes_to_create.append(('faces_timestamp_idx', """
                    CREATE INDEX IF NOT EXISTS faces_timestamp_idx
                    ON faces (timestamp)
                """))
                logger.info("   ✗ Missing: Timestamp index")
            else:
                logger.info("   ✓ Found: Timestamp index")

            logger.info("")

            # Create missing indexes
            if indexes_to_create:
                logger.info(f"→ Need to create {len(indexes_to_create)} missing index(es)")
                logger.info("")
                for idx_name, idx_sql in indexes_to_create:
                    create_start = time.time()
                    logger.info(f"   Creating {idx_name}...")
                    cursor.execute(idx_sql)
                    conn.commit()
                    logger.info(f"   ✓ Created {idx_name} in {time.time() - create_start:.2f}s")
                logger.info("")
            else:
                logger.info("→ ✓ All required indexes already exist - no action needed")
                logger.info("")

            cursor.close()

        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.return_connection(conn)

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
        Add a face to the database with multi-model support

        Args:
            face_data: FaceData object containing face information
            embedding_model: Name of the embedding model used (facenet, arcface, vggface2, insightface, statistical)

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

            # Extract age_estimate as integer only if it's numeric
            age_estimate_value = None
            age_val = features.get('age_estimate')
            if age_val is not None:
                try:
                    # Only use numeric values, skip strings like "60+", "40-60"
                    age_estimate_value = int(age_val)
                except (ValueError, TypeError):
                    # Keep as None for non-numeric values
                    age_estimate_value = None

            # Map model name to column name
            model_column_map = {
                'facenet': 'embedding_facenet',
                'arcface': 'embedding_arcface',
                'vggface2': 'embedding_vggface2',
                'insightface': 'embedding_insightface',
                'statistical': 'embedding_statistical'
            }

            embedding_column = model_column_map.get(embedding_model.lower(), 'embedding_statistical')

            # Check if face already exists
            cursor.execute("SELECT id, models_processed FROM faces WHERE face_id = %s", (face_data.face_id,))
            existing = cursor.fetchone()

            if existing:
                # Update existing face - add new embedding to appropriate column
                face_id_db, models_processed = existing

                # Update models_processed array
                if models_processed is None:
                    models_processed = []
                if embedding_model not in models_processed:
                    models_processed.append(embedding_model)

                query = f"""
                    UPDATE faces SET
                        file_path = %s,
                        timestamp = %s,
                        {embedding_column} = %s,
                        models_processed = %s,
                        age_estimate = %s,
                        gender = %s,
                        brightness = %s,
                        contrast = %s,
                        sharpness = %s,
                        metadata = %s,
                        updated_at = NOW()
                    WHERE face_id = %s
                """

                cursor.execute(query, (
                    face_data.file_path,
                    face_data.timestamp,
                    embedding,
                    models_processed,
                    age_estimate_value,
                    features.get('gender'),
                    features.get('brightness'),
                    features.get('contrast'),
                    features.get('sharpness'),
                    metadata_json,
                    face_data.face_id
                ))
            else:
                # Insert new face
                query = f"""
                    INSERT INTO faces (
                        face_id, file_path, timestamp, image_hash,
                        {embedding_column}, models_processed,
                        age_estimate, gender, brightness, contrast, sharpness,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                cursor.execute(query, (
                    face_data.face_id,
                    face_data.file_path,
                    face_data.timestamp,
                    face_data.image_hash,
                    embedding,
                    [embedding_model],
                    age_estimate_value,
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
        Add multiple faces in batch for better performance (multi-model support)

        Args:
            face_data_list: List of tuples (face_data, embedding_model)
            batch_size: Number of records to insert per batch

        Returns:
            int: Number of faces successfully added
        """
        if not self.initialized:
            logger.warning("Database not initialized")
            return 0

        # Use individual add_face for multi-model support
        # This ensures proper handling of model-specific columns
        added_count = 0
        for face_data, embedding_model in face_data_list:
            if self.add_face(face_data, embedding_model):
                added_count += 1

        logger.info(f"Added {added_count}/{len(face_data_list)} faces in batch")
        return added_count

    def search_faces(self, query_embedding: List[float], n_results: int = 10,
                    metadata_filter: Optional[Dict[str, Any]] = None,
                    distance_metric: str = 'cosine',
                    embedding_model: str = 'facenet') -> List[Dict[str, Any]]:
        """
        Search for similar faces using vector similarity (multi-model support)

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            metadata_filter: Optional metadata filters (e.g., {'gender': 'female'})
            distance_metric: Distance metric ('cosine', 'l2', 'inner_product')
            embedding_model: Model to use for search (facenet, arcface, vggface2, insightface, statistical)

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

            # Map model name to column name
            model_column_map = {
                'facenet': 'embedding_facenet',
                'arcface': 'embedding_arcface',
                'vggface2': 'embedding_vggface2',
                'insightface': 'embedding_insightface',
                'statistical': 'embedding_statistical'
            }

            embedding_column = model_column_map.get(embedding_model.lower(), 'embedding_facenet')

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
            # Note: For HNSW indexes, ordering by distance and using LIMIT is optimized
            query = f"""
                SELECT
                    face_id, file_path, timestamp, image_hash,
                    models_processed, age_estimate, gender, brightness,
                    contrast, sharpness, metadata,
                    {embedding_column} {distance_op} %s::vector AS distance
                FROM faces
                WHERE {embedding_column} IS NOT NULL
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
                # Merge dedicated columns with JSONB metadata
                # JSONB metadata takes precedence, but fall back to dedicated columns
                jsonb_metadata = row[10] if row[10] else {}

                metadata = {
                    'face_id': row[0],
                    'file_path': row[1],
                    'timestamp': row[2],
                    'image_hash': row[3],
                    'models_processed': row[4],  # Changed from embedding_model to models_processed
                    'age_estimate': row[5],
                    'gender': row[6],
                    'brightness': row[7],
                    'contrast': row[8],
                    'sharpness': row[9],
                }

                # Merge JSONB metadata
                metadata.update(jsonb_metadata)

                # Ensure estimated_sex exists (fallback to gender column)
                if not metadata.get('estimated_sex') and metadata.get('gender'):
                    metadata['estimated_sex'] = metadata['gender']

                # Ensure estimated_age exists (could be in age_estimate column or JSONB)
                if not metadata.get('estimated_age') and metadata.get('age_estimate'):
                    metadata['estimated_age'] = str(metadata['age_estimate'])

                formatted_results.append({
                    'id': row[0],
                    'metadata': metadata,
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

    def hybrid_search(self, query_embedding: List[float], metadata_filter: Dict[str, Any],
                     n_results: int = 10, embedding_model: str = 'facenet') -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and metadata filtering
        (Compatibility method - just calls search_faces with metadata_filter)

        Args:
            query_embedding: Query embedding vector
            metadata_filter: Metadata filters
            n_results: Number of results to return
            embedding_model: Model to use for search (facenet, arcface, vggface2, insightface, statistical)

        Returns:
            List of matching faces with distances
        """
        return self.search_faces(query_embedding, n_results, metadata_filter, embedding_model=embedding_model)

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
                    models_processed, age_estimate, gender, brightness,
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
                # Merge dedicated columns with JSONB metadata
                jsonb_metadata = row[10] if row[10] else {}

                metadata = {
                    'face_id': row[0],
                    'file_path': row[1],
                    'timestamp': row[2],
                    'image_hash': row[3],
                    'models_processed': row[4],  # Changed from embedding_model
                    'age_estimate': row[5],
                    'gender': row[6],
                    'brightness': row[7],
                    'contrast': row[8],
                    'sharpness': row[9],
                }

                # Merge JSONB metadata
                metadata.update(jsonb_metadata)

                # Ensure estimated_sex exists (fallback to gender column)
                if not metadata.get('estimated_sex') and metadata.get('gender'):
                    metadata['estimated_sex'] = metadata['gender']

                # Ensure estimated_age exists
                if not metadata.get('estimated_age') and metadata.get('age_estimate'):
                    metadata['estimated_age'] = str(metadata['age_estimate'])

                formatted_results.append({
                    'id': row[0],
                    'metadata': metadata,
                    'distance': 0.0  # Metadata search has no distance
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

    def hash_exists(self, image_hash: str) -> bool:
        """
        Check if an image with this hash already exists in the database
        (Alias for check_duplicate for compatibility with ChromaDB interface)

        Args:
            image_hash: Hash of the image

        Returns:
            bool: True if hash exists, False otherwise
        """
        return self.check_duplicate(image_hash)

    def check_embedding_model_mismatch(self, current_model: str) -> Dict[str, Any]:
        """
        Check if database has embeddings from different models

        Args:
            current_model: The current embedding model being used

        Returns:
            Dictionary with mismatch information:
            - has_mismatch: Boolean indicating if there's a model mismatch
            - models_found: Dictionary of model names and their counts
            - total_count: Total number of faces in database
            - current_model: The current model name
        """
        if not self.initialized:
            return {'has_mismatch': False, 'models_found': {}, 'total_count': 0}

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Check if this is a multi-model schema or legacy schema
            # Multi-model has multiple embedding_* columns, legacy has single embedding_model column
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='faces' AND column_name='embedding_model'
            """)
            has_legacy_column = cursor.fetchone() is not None

            if has_legacy_column:
                # Legacy schema: check embedding_model column
                cursor.execute("""
                    SELECT embedding_model, COUNT(*) as count
                    FROM faces
                    WHERE embedding_model IS NOT NULL
                    GROUP BY embedding_model
                """)

                results = cursor.fetchall()

                # Build model counts dictionary
                model_counts = {}
                total = 0
                for row in results:
                    model_name = row[0]
                    count = row[1]
                    model_counts[model_name] = count
                    total += count

                # Check for mismatch
                has_mismatch = False
                if current_model not in model_counts:
                    # Current model not in database at all
                    has_mismatch = len(model_counts) > 0
                elif len(model_counts) > 1:
                    # Multiple models in database
                    has_mismatch = True
                elif model_counts.get(current_model, 0) != total:
                    # Some entries don't have current model
                    has_mismatch = True

            else:
                # Multi-model schema: check models_processed array
                cursor.execute("""
                    SELECT DISTINCT unnest(models_processed) as model, COUNT(*) as count
                    FROM faces
                    WHERE models_processed IS NOT NULL AND array_length(models_processed, 1) > 0
                    GROUP BY model
                """)

                results = cursor.fetchall()

                # Build model counts dictionary
                model_counts = {}
                total = 0
                for row in results:
                    model_name = row[0]
                    count = row[1]
                    model_counts[model_name] = count
                    # Note: total is sum of all model embeddings, not unique faces
                    total += count

                # For multi-model schema, mismatch is not really applicable
                # since multiple models are expected. Return info but no mismatch.
                has_mismatch = False

            cursor.close()

            return {
                'has_mismatch': has_mismatch,
                'models_found': model_counts,
                'total_count': total,
                'current_model': current_model
            }

        except Exception as e:
            logger.error(f"Error checking model mismatch: {e}")
            return {
                'has_mismatch': False,
                'models_found': {},
                'total_count': 0,
                'current_model': current_model
            }

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

    def clear_all_data(self) -> bool:
        """
        Clear all data from the database
        (Alias for reset_database for compatibility with ChromaDB interface)

        Returns:
            bool: True if successful, False otherwise
        """
        return self.reset_database()

    def create_performance_indexes(self) -> bool:
        """
        Manually create/rebuild performance indexes.
        Useful for existing databases that don't have indexes yet.

        Returns:
            bool: True if successful
        """
        try:
            logger.info("Creating performance indexes...")
            self._ensure_indexes()
            logger.info("Performance indexes created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False

    def analyze_table_stats(self) -> Dict[str, Any]:
        """
        Analyze table statistics for query optimization

        Returns:
            Dictionary with table statistics
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Run ANALYZE to update statistics
            cursor.execute("ANALYZE faces")

            # Get index information
            cursor.execute("""
                SELECT
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE tablename = 'faces'
                ORDER BY indexname
            """)
            indexes = cursor.fetchall()

            # Get table size
            cursor.execute("""
                SELECT pg_size_pretty(pg_total_relation_size('faces'))
            """)
            table_size = cursor.fetchone()[0]

            cursor.close()

            return {
                'indexes': [{'name': idx[0], 'definition': idx[1]} for idx in indexes],
                'table_size': table_size,
                'analyzed': True
            }

        except Exception as e:
            logger.error(f"Failed to analyze table stats: {e}")
            return {'error': str(e)}
        finally:
            if conn:
                self.return_connection(conn)

    def reinitialize_schema(self) -> bool:
        """
        Completely reinitialize the database schema from scratch
        WARNING: This will DROP all existing data!

        Returns:
            bool: True if successful, False otherwise
        """
        conn = None
        try:
            logger.info("=" * 70)
            logger.info("REINITIALIZING VECTOR DATABASE SCHEMA")
            logger.info("=" * 70)
            logger.info("⚠️  WARNING: This will DROP all existing data!")

            conn = self.get_connection()
            cursor = conn.cursor()

            # Read and execute schema.sql
            schema_file = os.path.join(os.path.dirname(__file__), 'schema.sql')

            if not os.path.exists(schema_file):
                logger.error(f"Schema file not found: {schema_file}")
                return False

            logger.info(f"→ Reading schema from: {schema_file}")

            with open(schema_file, 'r') as f:
                schema_sql = f.read()

            logger.info("→ Executing schema SQL...")
            logger.info("")

            # Execute the schema
            cursor.execute(schema_sql)
            conn.commit()

            logger.info("✓ Schema created successfully")
            logger.info("")

            # Verify the schema was created
            logger.info("→ Verifying schema creation...")

            # Check if pgvector extension exists
            cursor.execute(
                "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
            )
            if cursor.fetchone()[0] == 0:
                logger.error("✗ pgvector extension not found after schema creation")
                return False
            logger.info("  ✓ pgvector extension verified")

            # Check if faces table exists
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'faces'"
            )
            if cursor.fetchone()[0] == 0:
                logger.error("✗ faces table not found after schema creation")
                return False
            logger.info("  ✓ faces table verified")

            # Check indexes
            cursor.execute("""
                SELECT COUNT(*) FROM pg_indexes
                WHERE tablename = 'faces'
            """)
            index_count = cursor.fetchone()[0]
            logger.info(f"  ✓ {index_count} indexes created")

            # Check functions
            cursor.execute("""
                SELECT COUNT(*) FROM pg_proc
                WHERE proname IN ('search_similar_faces', 'get_database_stats')
            """)
            function_count = cursor.fetchone()[0]
            logger.info(f"  ✓ {function_count} helper functions created")

            cursor.close()

            logger.info("")
            logger.info("=" * 70)
            logger.info("✓ DATABASE SCHEMA REINITIALIZED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info("")
            logger.info("The database is now ready for use with:")
            logger.info("  • Empty faces table")
            logger.info("  • All indexes created")
            logger.info("  • All helper functions available")
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"✗ Failed to reinitialize schema: {e}")
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
