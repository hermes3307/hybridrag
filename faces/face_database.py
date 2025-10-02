#!/usr/bin/env python3
"""
Face Database Integration with ChromaDB
Compatibility wrapper for monitoring and search operations
"""

import chromadb
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FaceDatabase:
    """ChromaDB integration for face embeddings and monitoring"""

    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "faces"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB client and collection with retry logic"""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Create persistent client with settings for concurrent access
                from chromadb.config import Settings

                self.client = chromadb.PersistentClient(
                    path=self.db_path,
                    settings=Settings(
                        allow_reset=False,
                        anonymized_telemetry=False
                    )
                )

                # Get or create collection
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Face embeddings collection"}
                )

                logger.info(f"Initialized face database: {self.collection_name}")
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Database init attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                    import time
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Error initializing database after {max_retries} attempts: {e}")
                    raise

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics with retry on lock"""
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                # Get basic collection info
                total_faces = self.collection.count()

                # Get sample data to analyze distributions
                stats = {
                    'total_faces': total_faces,
                    'age_group_distribution': {},
                    'skin_tone_distribution': {},
                    'quality_distribution': {}
                }

                if total_faces == 0:
                    return stats

                # Get all metadata for analysis (limit to reasonable amount)
                limit = min(total_faces, 10000)
                results = self.collection.get(limit=limit, include=['metadatas'])

                if results and 'metadatas' in results:
                    age_groups = {}
                    skin_tones = {}
                    qualities = {}

                    for metadata in results['metadatas']:
                        # Count age groups
                        age_group = metadata.get('estimated_age_group', 'unknown')
                        age_groups[age_group] = age_groups.get(age_group, 0) + 1

                        # Count skin tones
                        skin_tone = metadata.get('estimated_skin_tone', 'unknown')
                        skin_tones[skin_tone] = skin_tones.get(skin_tone, 0) + 1

                        # Count quality levels
                        quality = metadata.get('image_quality', 'unknown')
                        qualities[quality] = qualities.get(quality, 0) + 1

                    stats['age_group_distribution'] = age_groups
                    stats['skin_tone_distribution'] = skin_tones
                    stats['quality_distribution'] = qualities

                return stats

            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's a database lock error
                if 'locked' in error_msg or 'busy' in error_msg:
                    if attempt < max_retries - 1:
                        logger.debug(f"Database locked, retrying ({attempt + 1}/{max_retries})...")
                        import time
                        time.sleep(retry_delay)
                        continue

                logger.error(f"Error getting database stats: {e}")
                return {
                    'total_faces': 0,
                    'age_group_distribution': {},
                    'skin_tone_distribution': {},
                    'quality_distribution': {},
                    'error': str(e)
                }

        # If all retries failed
        return {
            'total_faces': 0,
            'age_group_distribution': {},
            'skin_tone_distribution': {},
            'quality_distribution': {},
            'error': 'Database locked after multiple retries'
        }

    def search_similar_faces(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Search for similar faces using embedding similarity"""
        try:
            logger.info(f"ChromaDB query - embedding dim: {len(query_embedding)}, n_results: {n_results}")

            # Check collection count
            count = self.collection.count()
            logger.info(f"Collection has {count} total vectors")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )

            logger.info(f"Query returned: {results.keys()}")
            logger.info(f"IDs: {len(results['ids'][0]) if results['ids'] and len(results['ids']) > 0 else 0}")

            return {
                "ids": results["ids"][0] if results["ids"] and len(results["ids"]) > 0 else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] and len(results["metadatas"]) > 0 else [],
                "documents": results["documents"][0] if results["documents"] and len(results["documents"]) > 0 else [],
                "distances": results["distances"][0] if results["distances"] and len(results["distances"]) > 0 else [],
                "count": len(results["ids"][0]) if results["ids"] and len(results["ids"]) > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"ids": [], "metadatas": [], "documents": [], "distances": [], "count": 0}

    def search_by_metadata(self, filters: Dict[str, Any], n_results: int = 10) -> Dict[str, Any]:
        """Search faces by metadata filters"""
        try:
            # Build where clause
            where_clause = {}
            for key, value in filters.items():
                if value is not None and value != "":
                    where_clause[key] = value

            if not where_clause:
                # No filters, return first n_results
                results = self.collection.get(limit=n_results, include=["metadatas"])
            else:
                # Query with filters
                results = self.collection.get(
                    where=where_clause,
                    limit=n_results,
                    include=["metadatas"]
                )

            return {
                "ids": results.get("ids", []),
                "metadatas": results.get("metadatas", []),
                "count": len(results.get("ids", []))
            }

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return {"ids": [], "metadatas": [], "count": 0}


class FaceSearchInterface:
    """Interface for advanced face search operations"""

    def __init__(self, face_db=None, db_path: str = "./chroma_db", collection_name: str = "faces"):
        """
        Initialize FaceSearchInterface.

        Args:
            face_db: Optional existing FaceDatabase instance to use
            db_path: Path to database (used if face_db not provided)
            collection_name: Collection name (used if face_db not provided)
        """
        if face_db is not None:
            self.face_db = face_db
        else:
            self.face_db = FaceDatabase(db_path, collection_name)

    def search_similar(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Search for similar faces"""
        return self.face_db.search_similar_faces(query_embedding, n_results)

    def search_by_image(self, image_path: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for similar faces by image path"""
        try:
            from PIL import Image
            import numpy as np

            logger.info(f"=== Vector Search Started ===")
            logger.info(f"Input image: {image_path}")
            logger.info(f"Requested results: {n_results}")

            # Load and process image
            logger.info("Loading and preprocessing image...")
            img = Image.open(image_path).convert('RGB')
            original_size = img.size
            img = img.resize((224, 224))
            image = np.array(img)
            logger.info(f"Image resized from {original_size} to (224, 224)")

            # Generate embedding using the SAME method as the database
            logger.info("Generating embedding vector using CNN features...")

            # Convert to grayscale
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2).astype(np.uint8)
            else:
                gray = image

            feature_vector = []

            # Histogram features (32 bins)
            hist, _ = np.histogram(gray, bins=32, range=[0, 256])
            feature_vector.extend(hist.tolist())
            logger.info(f"Added histogram features: {len(hist)} values")

            # Color moments for each channel (RGB)
            for channel in range(image.shape[2] if len(image.shape) == 3 else 1):
                ch = image[:,:,channel] if len(image.shape) == 3 else image
                feature_vector.extend([
                    float(np.mean(ch)),
                    float(np.std(ch)),
                    float(np.mean(np.power(ch - np.mean(ch), 3)))  # skewness
                ])
            logger.info(f"Added color moments: 9 values (3 channels × 3 moments)")

            # Texture features (7x7 grid = 49 patches × 2 features)
            patch_count = 0
            for i in range(0, gray.shape[0], 32):
                for j in range(0, gray.shape[1], 32):
                    patch = gray[i:i+32, j:j+32]
                    if patch.size > 0:
                        feature_vector.extend([
                            float(np.mean(patch)),
                            float(np.std(patch))
                        ])
                        patch_count += 1
            logger.info(f"Added texture features: {patch_count} patches × 2 = {patch_count * 2} values")

            # Add dummy extracted features (since we don't have metadata for query)
            feature_vector.extend([0.5, 0.5, 0.5, 0.5])
            logger.info(f"Added metadata features: 4 values")

            # Normalize to unit vector (SAME as database)
            feature_vector = np.array(feature_vector)
            if np.linalg.norm(feature_vector) > 0:
                feature_vector = feature_vector / np.linalg.norm(feature_vector)

            embedding = feature_vector.tolist()

            logger.info(f"Embedding vector size: {len(embedding)}")
            logger.info(f"Embedding stats - min: {min(embedding):.4f}, max: {max(embedding):.4f}, mean: {sum(embedding)/len(embedding):.4f}")

            # Search with embedding
            logger.info("Querying ChromaDB with embedding vector...")
            results = self.face_db.search_similar_faces(embedding, n_results)

            logger.info(f"ChromaDB returned {results.get('count', 0)} results")
            logger.info(f"Result structure: {list(results.keys())}")

            if results.get('count', 0) > 0:
                logger.info(f"Top 3 distances: {results.get('distances', [])[:3]}")
                logger.info(f"Top 3 IDs: {results.get('ids', [])[:3]}")

            logger.info("=== Vector Search Completed ===")

            return {"results": results}

        except Exception as e:
            logger.error(f"Error in search_by_image: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

    def search_by_features(self, filters: Dict[str, Any], n_results: int = 10) -> Dict[str, Any]:
        """Search by feature filters"""
        return self.face_db.search_by_metadata(filters, n_results)

    def combined_search(self, query_embedding: List[float], filters: Dict[str, Any],
                       n_results: int = 5) -> Dict[str, Any]:
        """Combined semantic and metadata search"""
        try:
            # Build where clause
            where_clause = {}
            for key, value in filters.items():
                if value is not None and value != "":
                    where_clause[key] = value

            # Query with both embedding and filters
            if where_clause:
                results = self.face_db.collection.query(
                    query_embeddings=[query_embedding],
                    where=where_clause,
                    n_results=n_results,
                    include=["metadatas", "documents", "distances"]
                )
            else:
                results = self.face_db.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["metadatas", "documents", "distances"]
                )

            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "count": len(results["ids"][0]) if results["ids"] else 0
            }

        except Exception as e:
            logger.error(f"Error in combined search: {e}")
            return {"ids": [], "metadatas": [], "documents": [], "distances": [], "count": 0}
