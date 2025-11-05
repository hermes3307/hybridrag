import psycopg2
from psycopg2.extras import RealDictCursor
import logging

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
            with self.conn.cursor() as cursor:
                with open("schema.sql", "r") as f:
                    cursor.execute(f.read())
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error reinitializing schema: {e}")
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

    def add_image(self, image_data, embedding_model):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO images (image_id, file_path, timestamp, image_hash, embedding_model, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (
                        image_data.image_id,
                        image_data.file_path,
                        image_data.timestamp,
                        image_data.image_hash,
                        embedding_model,
                        image_data.embedding,
                        psycopg2.extras.Json(image_data.features)
                    )
                )
            self.conn.commit()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error adding image: {e}")
            self.conn.rollback()
            return False

    def search_images(self, embedding, limit):
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM search_similar_images(%s, %s)", (embedding, limit))
                return cursor.fetchall()
        except psycopg2.Error as e:
            logger.error(f"Error searching images: {e}")
            return []

    def mixed_search(self, clip_embedding, yolo_embedding, action_embedding, limit):
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        i.image_id,
                        i.file_path,
                        i.metadata,
                        (i.embedding <-> %s) * 0.33 + (i.embedding <-> %s) * 0.33 + (i.embedding <-> %s) * 0.33 AS distance
                    FROM images i
                    WHERE i.embedding_model = 'clip' OR i.embedding_model = 'yolo' OR i.embedding_model = 'action'
                    ORDER BY distance
                    LIMIT %s
                """, (clip_embedding, yolo_embedding, action_embedding, limit))
                return cursor.fetchall()
        except psycopg2.Error as e:
            logger.error(f"Error in mixed search: {e}")
            return []

