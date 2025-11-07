import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import os

logger = logging.getLogger(__name__)

class PgVectorDatabaseManager:
    def __init__(self, config):
        self.config = config
        self.conn = None

    def initialize(self):
        try:
            self.conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                dbname=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
            return True
        except psycopg2.Error as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            return False

    def close(self):
        if self.conn:
            self.conn.close()

    def get_connection(self):
        return self.conn

    def return_connection(self, conn):
        pass

    def reinitialize_schema(self):
        try:
            # Get absolute path to schema.sql (in the same directory as this file)
            schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

            if not os.path.exists(schema_path):
                logger.error(f"Schema file not found: {schema_path}")
                return False

            with self.conn.cursor() as cursor:
                with open(schema_path, "r") as f:
                    sql_content = f.read()
                    cursor.execute(sql_content)
            self.conn.commit()
            logger.info("Schema reinitialized successfully")
            return True
        except FileNotFoundError as e:
            logger.error(f"Schema file not found: {e}")
            return False
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error reinitializing schema: {e}")
            self.conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Unexpected error reinitializing schema: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def get_collection_info(self):
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM images")
                return cursor.fetchone()
        except psycopg2.Error as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def hash_exists(self, image_hash):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM images WHERE image_hash = %s", (image_hash,))
                return cursor.fetchone() is not None
        except psycopg2.Error as e:
            logger.error(f"Error checking hash: {e}")
            return False

    def add_image(self, image_data):
        """Add image metadata (without embeddings)"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO images (image_id, file_path, timestamp, image_hash, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (image_id) DO NOTHING""",
                    (
                        image_data.image_id,
                        image_data.file_path,
                        image_data.timestamp,
                        image_data.image_hash,
                        psycopg2.extras.Json(image_data.features)
                    )
                )
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error adding image: {e}")
            self.conn.rollback()
            return False

    def add_embedding(self, image_id, model_name, embedding):
        """Add an embedding for an image"""
        try:
            # Determine embedding dimension and which column to use
            embedding_dim = len(embedding)

            if embedding_dim == 512:
                column_name = 'embedding_512'
            elif embedding_dim == 1024:
                column_name = 'embedding_1024'
            else:
                # Pad or truncate to 512
                if embedding_dim < 512:
                    embedding = embedding + [0.0] * (512 - embedding_dim)
                else:
                    embedding = embedding[:512]
                column_name = 'embedding_512'
                embedding_dim = 512

            with self.conn.cursor() as cursor:
                cursor.execute(
                    f"""INSERT INTO image_embeddings (image_id, embedding_model, embedding_dimension, {column_name})
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (image_id, embedding_model)
                    DO UPDATE SET {column_name} = EXCLUDED.{column_name}, embedding_dimension = EXCLUDED.embedding_dimension""",
                    (image_id, model_name, embedding_dim, embedding)
                )
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error adding embedding: {e}")
            self.conn.rollback()
            return False

    def search_images(self, embedding, model_name, limit=10):
        """Search for similar images using a specific model"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Convert embedding to string format for pgvector
                embedding_str = str(embedding)
                
                cursor.execute(
                    "SELECT * FROM search_similar_images(%s, %s, %s)",
                    (embedding_str, model_name, limit)
                )
                return cursor.fetchall()
        except psycopg2.Error as e:
            logger.error(f"Error searching images: {e}")
            return []

    def hybrid_search(self, embedding, model_name, metadata_filter, limit=10):
        """Search using both vector similarity and metadata filters"""
        try:
            # For now, just use vector search with the specified model
            # Metadata filtering can be added as post-processing or in SQL WHERE clause
            return self.search_images(embedding, model_name, limit)
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    def multi_embedding_search(self, clip_emb, yolo_emb, resnet_emb, limit=10,
                               clip_weight=0.5, yolo_weight=0.25, resnet_weight=0.25):
        """Search using multiple embeddings with weighted fusion"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT * FROM search_multi_embedding(%s, %s, %s, %s, %s, %s, %s)""",
                    (clip_emb, yolo_emb, resnet_emb, clip_weight, yolo_weight, resnet_weight, limit)
                )
                return cursor.fetchall()
        except psycopg2.Error as e:
            logger.error(f"Error in multi-embedding search: {e}")
            return []

    def text_to_image_search(self, text_query, limit=10):
        """Search images using text query via CLIP"""
        try:
            # Generate CLIP text embedding
            from transformers import CLIPProcessor, CLIPModel
            import torch

            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            inputs = processor(text=[text_query], return_tensors="pt", padding=True)
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
            text_embedding = text_features.cpu().numpy().flatten().tolist()

            # Search using CLIP model
            return self.search_images(text_embedding, 'clip', limit)
        except Exception as e:
            logger.error(f"Error in text-to-image search: {e}")
            return []

    def check_embedding_model_mismatch(self, current_model=None):
        """Check if there are embedding model mismatches in database

        Args:
            current_model: The currently configured embedding model (optional)

        Returns:
            dict: Mismatch information including:
                - has_mismatch: bool
                - current_model: str
                - models_found: dict of {model_name: count}
                - total_count: int
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get all embedding models in database
                cursor.execute("""
                    SELECT embedding_model, COUNT(*) as count
                    FROM image_embeddings
                    GROUP BY embedding_model
                """)
                results = cursor.fetchall()
                models_found = {row['embedding_model']: row['count'] for row in results}

                # Get total count
                cursor.execute("SELECT COUNT(*) as total FROM image_embeddings")
                total_count = cursor.fetchone()['total']

                # Check for mismatch
                has_mismatch = False
                if current_model and models_found:
                    # Mismatch if there are models different from current_model
                    has_mismatch = any(model != current_model for model in models_found.keys())

                return {
                    'has_mismatch': has_mismatch,
                    'current_model': current_model,
                    'models_found': models_found,
                    'total_count': total_count
                }
        except Exception as e:
            logger.error(f"Error checking model mismatch: {e}")
            return {
                'has_mismatch': False,
                'current_model': current_model,
                'models_found': {},
                'total_count': 0
            }

    def get_stats(self):
        """Get database statistics"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get total images count
                cursor.execute("SELECT COUNT(*) as count FROM images")
                total_images = cursor.fetchone()['count']

                # Get database size
                cursor.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as size
                """)
                db_size = cursor.fetchone()['size']

                return {
                    'total_images': total_images,
                    'database_size': db_size
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'total_images': 0,
                'database_size': 'Unknown'
            }

