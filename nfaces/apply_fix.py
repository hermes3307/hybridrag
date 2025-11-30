#!/usr/bin/env python3
"""
Apply the fix for the get_database_stats() function
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

def apply_fix():
    """Apply the SQL fix"""
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
        print("Applying fix for get_database_stats() function")
        print("=" * 60)
        print()

        # Read and execute the fix
        with open('fix_stats_function.sql', 'r') as f:
            fix_sql = f.read()

        print("Executing SQL fix...")
        cursor.execute(fix_sql)
        conn.commit()
        print("✓ Function updated successfully")
        print()

        # Test the function
        print("Testing the updated function...")
        cursor.execute("SELECT * FROM get_database_stats()")
        result = cursor.fetchone()

        if result:
            print("✓ Function works! Results:")
            print(f"  Total faces: {result[0]}")
            print(f"  Faces with embeddings: {result[1]}")
            print(f"  Embedding models: {result[2]}")
            print(f"  Oldest face: {result[3]}")
            print(f"  Newest face: {result[4]}")
            print(f"  Database size: {result[5]}")
        else:
            print("✗ Function returned no results")

        cursor.close()
        conn.close()

        print()
        print("=" * 60)
        print("✓ Fix applied successfully!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"✗ Error applying fix: {e}")
        if conn:
            conn.rollback()
        return False

if __name__ == "__main__":
    apply_fix()
