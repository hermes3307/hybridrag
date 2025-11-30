#!/usr/bin/env python3
"""
Database setup script
Creates database and runs schema without requiring sudo
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
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

def create_database():
    """Create the database if it doesn't exist"""
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database='postgres',
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (DB_NAME,)
        )
        exists = cursor.fetchone()

        if not exists:
            print(f"Creating database '{DB_NAME}'...")
            cursor.execute(f'CREATE DATABASE {DB_NAME}')
            print(f"✓ Database '{DB_NAME}' created successfully")
        else:
            print(f"Database '{DB_NAME}' already exists")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return False

def run_schema():
    """Run the schema.sql file"""
    try:
        # Connect to the target database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # Read and execute schema file
        print(f"Running schema.sql...")
        with open('schema.sql', 'r') as f:
            schema_sql = f.read()

        cursor.execute(schema_sql)
        conn.commit()

        print("✓ Schema created successfully")

        # Verify tables were created
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        print(f"✓ Created tables: {', '.join([t[0] for t in tables])}")

        # Verify pgvector extension
        cursor.execute(
            "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
        )
        if cursor.fetchone()[0] > 0:
            print("✓ pgvector extension is enabled")
        else:
            print("✗ pgvector extension not found")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"✗ Error running schema: {e}")
        if conn:
            conn.rollback()
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("Database Setup Script")
    print("=" * 60)
    print(f"Host: {DB_HOST}")
    print(f"Port: {DB_PORT}")
    print(f"Database: {DB_NAME}")
    print(f"User: {DB_USER}")
    print("=" * 60)
    print()

    # Step 1: Create database
    if not create_database():
        print("✗ Failed to create database. Please check your PostgreSQL configuration.")
        return False

    print()

    # Step 2: Run schema
    if not run_schema():
        print("✗ Failed to run schema. Please check schema.sql file.")
        return False

    print()
    print("=" * 60)
    print("✓ Database setup completed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    main()
