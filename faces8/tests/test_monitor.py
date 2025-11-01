#!/usr/bin/env python3
"""
Quick test script to verify monitor can connect to database
Run this before launching the GUI to verify configuration
"""

import sys
import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

def test_connection():
    """Test database connection"""
    print("=" * 60)
    print("pgvector Database Monitor - Connection Test")
    print("=" * 60)

    # Get connection parameters
    db_params = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'vector_db'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', '')
    }

    print("\nConfiguration:")
    print(f"  Host: {db_params['host']}")
    print(f"  Port: {db_params['port']}")
    print(f"  Database: {db_params['database']}")
    print(f"  User: {db_params['user']}")
    print(f"  Password: {'*' * len(db_params['password']) if db_params['password'] else '(empty)'}")

    # Test connection
    print("\nTesting connection...")
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Get PostgreSQL version
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"✓ Connected successfully!")
        print(f"  PostgreSQL version: {version[:80]}...")

        # Check for pgvector extension
        cursor.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'")
        has_pgvector = cursor.fetchone()[0] > 0

        if has_pgvector:
            print("✓ pgvector extension found")
        else:
            print("✗ pgvector extension NOT found")
            print("  Run: CREATE EXTENSION vector;")
            return False

        # Check for faces table
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'faces'"
        )
        has_faces_table = cursor.fetchone()[0] > 0

        if has_faces_table:
            print("✓ faces table found")

            # Get table stats
            cursor.execute("SELECT COUNT(*) FROM faces")
            total_faces = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM faces WHERE embedding IS NOT NULL")
            total_vectors = cursor.fetchone()[0]

            print(f"  Total faces: {total_faces}")
            print(f"  Total vectors: {total_vectors}")
        else:
            print("✗ faces table NOT found")
            print("  Run schema.sql to create the table")
            return False

        # Check active connections
        cursor.execute(
            "SELECT COUNT(*) FROM pg_stat_activity WHERE datname = %s",
            (db_params['database'],)
        )
        active_connections = cursor.fetchone()[0]
        print(f"  Active connections: {active_connections}")

        cursor.close()
        conn.close()

        print("\n" + "=" * 60)
        print("✓ All checks passed! You can now run the monitor.")
        print("=" * 60)
        print("\nTo start the monitor:")
        print("  python3 monitor.py")
        print("  or")
        print("  ./run_monitor.sh")
        print("")

        return True

    except psycopg2.OperationalError as e:
        print(f"✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if PostgreSQL is running:")
        print("   sudo systemctl status postgresql")
        print("2. Verify your .env file configuration")
        print("3. Test connection manually:")
        print(f"   psql -h {db_params['host']} -U {db_params['user']} -d {db_params['database']}")
        return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
