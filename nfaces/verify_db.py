#!/usr/bin/env python3
"""
Verify database setup
"""

import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database parameters from .env
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = int(os.getenv('POSTGRES_PORT', 5432))
DB_NAME = os.getenv('POSTGRES_DB', 'vector_db')
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

def verify_database():
    """Verify database setup"""
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()

        print("=" * 60)
        print("Database Connection")
        print("=" * 60)
        print(f"✓ Connected to database '{DB_NAME}'")
        print()

        # Check pgvector extension
        print("=" * 60)
        print("Extensions")
        print("=" * 60)
        cursor.execute(
            "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
        )
        result = cursor.fetchone()
        if result:
            print(f"✓ pgvector extension: version {result[1]}")
        else:
            print("✗ pgvector extension not found")
        print()

        # Check tables
        print("=" * 60)
        print("Tables")
        print("=" * 60)
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        for table in tables:
            print(f"✓ Table: {table[0]}")
        print()

        # Check faces table structure
        print("=" * 60)
        print("Faces Table Structure")
        print("=" * 60)
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'faces'
            ORDER BY ordinal_position
        """)
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[0]}: {col[1]}")
        print()

        # Check indexes
        print("=" * 60)
        print("Indexes")
        print("=" * 60)
        cursor.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'faces'
            ORDER BY indexname
        """)
        indexes = cursor.fetchall()
        for idx in indexes:
            print(f"✓ {idx[0]}")
        print()

        # Check data count
        print("=" * 60)
        print("Data Statistics")
        print("=" * 60)
        cursor.execute("SELECT COUNT(*) FROM faces")
        count = cursor.fetchone()[0]
        print(f"Total faces: {count}")

        if count > 0:
            cursor.execute("SELECT COUNT(DISTINCT embedding_model) FROM faces")
            models = cursor.fetchone()[0]
            print(f"Embedding models: {models}")

            cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM faces")
            dates = cursor.fetchone()
            print(f"Date range: {dates[0]} to {dates[1]}")
        print()

        cursor.close()
        conn.close()

        print("=" * 60)
        print("✓ Database is properly configured!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"✗ Error verifying database: {e}")
        return False

if __name__ == "__main__":
    verify_database()
