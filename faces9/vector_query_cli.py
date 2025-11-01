#!/usr/bin/env python3
"""
CLI Tool for pgvector Database Queries

A terminal-based interface for querying vectors and metadata.
Perfect for quick searches and scripting.

Usage:
    python3 vector_query_cli.py
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()


class VectorQueryCLI:
    """Command-line interface for vector queries"""

    def __init__(self):
        self.db_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'vector_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }
        self.conn = None

    def connect(self) -> bool:
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(**self.db_params)
            print(f"✓ Connected to {self.db_params['database']}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✓ Connection closed")

    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        """Execute query and return results"""
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                cursor.close()
                return columns, rows
            else:
                self.conn.commit()
                cursor.close()
                return [], []

        except Exception as e:
            print(f"✗ Query error: {e}")
            return [], []

    def show_stats(self):
        """Show database statistics"""
        print("\n" + "=" * 70)
        print("DATABASE STATISTICS")
        print("=" * 70)

        # Total faces
        cols, rows = self.execute_query("SELECT COUNT(*) FROM faces")
        if rows:
            print(f"Total Faces: {rows[0][0]}")

        # Vectors
        cols, rows = self.execute_query(
            "SELECT COUNT(*) FROM faces WHERE embedding IS NOT NULL"
        )
        if rows:
            print(f"Vectors (with embeddings): {rows[0][0]}")

        # Models
        cols, rows = self.execute_query("""
            SELECT embedding_model, COUNT(*)
            FROM faces
            WHERE embedding_model IS NOT NULL
            GROUP BY embedding_model
        """)
        if rows:
            print("\nEmbedding Models:")
            for row in rows:
                print(f"  {row[0]}: {row[1]}")

        # Date range
        cols, rows = self.execute_query(
            "SELECT MIN(created_at), MAX(created_at) FROM faces"
        )
        if rows and rows[0][0]:
            print(f"\nDate Range:")
            print(f"  Oldest: {rows[0][0]}")
            print(f"  Newest: {rows[0][1]}")

        # Database size
        cols, rows = self.execute_query(
            "SELECT pg_size_pretty(pg_database_size(%s))",
            (self.db_params['database'],)
        )
        if rows:
            print(f"\nDatabase Size: {rows[0][0]}")

        print("=" * 70)

    def search_metadata(self):
        """Interactive metadata search"""
        print("\n" + "=" * 70)
        print("METADATA SEARCH")
        print("=" * 70)

        filters = []
        params = []

        # Gender
        gender = input("\nGender (male/female/unknown/Enter to skip): ").strip()
        if gender:
            filters.append("(gender = %s OR metadata->>'estimated_sex' = %s)")
            params.extend([gender, gender])

        # Age range
        age_min = input("Minimum age (Enter to skip): ").strip()
        if age_min:
            filters.append("(age_estimate >= %s OR (metadata->>'age_estimate')::float >= %s)")
            params.extend([float(age_min), float(age_min)])

        age_max = input("Maximum age (Enter to skip): ").strip()
        if age_max:
            filters.append("(age_estimate <= %s OR (metadata->>'age_estimate')::float <= %s)")
            params.extend([float(age_max), float(age_max)])

        # Brightness
        bright_min = input("Minimum brightness 0-255 (Enter to skip): ").strip()
        if bright_min:
            filters.append("brightness >= %s")
            params.append(float(bright_min))

        bright_max = input("Maximum brightness 0-255 (Enter to skip): ").strip()
        if bright_max:
            filters.append("brightness <= %s")
            params.append(float(bright_max))

        # Limit
        limit = input("Maximum results [default 20]: ").strip() or "20"

        # Build query
        query = """
            SELECT face_id, file_path, gender, age_estimate, brightness,
                   created_at
            FROM faces
            WHERE 1=1
        """

        if filters:
            query += " AND " + " AND ".join(filters)

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(int(limit))

        # Execute
        print("\n🔍 Searching...")
        cols, rows = self.execute_query(query, tuple(params))

        if not rows:
            print("✗ No results found")
            return

        # Display results
        print(f"\n✓ Found {len(rows)} results:\n")
        print(f"{'#':<4} {'Face ID':<35} {'Gender':<10} {'Age':<8} {'Brightness':<10} {'Created'}")
        print("-" * 100)

        for idx, row in enumerate(rows, 1):
            face_id = row[0][:32] + "..." if len(row[0]) > 32 else row[0]
            gender = row[2] or 'N/A'
            age = str(row[3]) if row[3] else 'N/A'
            brightness = f"{row[4]:.1f}" if row[4] else 'N/A'
            created = row[5].strftime('%Y-%m-%d %H:%M') if row[5] else 'N/A'

            print(f"{idx:<4} {face_id:<35} {gender:<10} {age:<8} {brightness:<10} {created}")

        # Ask for details
        detail = input("\nEnter row number to see details (or Enter to skip): ").strip()
        if detail and detail.isdigit() and 1 <= int(detail) <= len(rows):
            self.show_face_details(rows[int(detail) - 1][0])

    def similarity_search(self):
        """Interactive similarity search"""
        print("\n" + "=" * 70)
        print("VECTOR SIMILARITY SEARCH")
        print("=" * 70)

        # Get random face for query
        print("\n🎲 Picking a random face for similarity search...")

        cols, rows = self.execute_query("""
            SELECT face_id, file_path, embedding
            FROM faces
            WHERE embedding IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 1
        """)

        if not rows:
            print("✗ No faces with embeddings found")
            return

        query_face_id = rows[0][0]
        query_embedding = rows[0][2]

        print(f"✓ Query face: {query_face_id}")

        # Distance metric
        print("\nDistance metrics:")
        print("  1. Cosine (default)")
        print("  2. L2 (Euclidean)")
        print("  3. Inner Product")

        metric_choice = input("Select metric [1-3, default 1]: ").strip() or "1"

        metric_map = {
            '1': ('cosine', '<=>'),
            '2': ('l2', '<->'),
            '3': ('inner_product', '<#>')
        }

        metric_name, metric_op = metric_map.get(metric_choice, metric_map['1'])

        # Limit
        limit = input("Number of results [default 10]: ").strip() or "10"

        # Execute similarity search
        print(f"\n🔍 Finding similar faces using {metric_name} distance...")

        query = f"""
            SELECT face_id, file_path, gender, brightness,
                   embedding {metric_op} %s::vector AS distance
            FROM faces
            WHERE embedding IS NOT NULL
            ORDER BY distance
            LIMIT %s
        """

        cols, rows = self.execute_query(query, (query_embedding, int(limit)))

        if not rows:
            print("✗ No results found")
            return

        # Display results
        print(f"\n✓ Found {len(rows)} similar faces:\n")
        print(f"{'#':<4} {'Face ID':<35} {'Gender':<10} {'Brightness':<10} {'Distance'}")
        print("-" * 90)

        for idx, row in enumerate(rows, 1):
            face_id = row[0][:32] + "..." if len(row[0]) > 32 else row[0]
            gender = row[2] or 'N/A'
            brightness = f"{row[3]:.1f}" if row[3] else 'N/A'
            distance = f"{row[4]:.6f}"

            marker = "🎯" if idx == 1 else "  "
            print(f"{marker} {idx:<2} {face_id:<35} {gender:<10} {brightness:<10} {distance}")

        # Ask for details
        detail = input("\nEnter row number to see details (or Enter to skip): ").strip()
        if detail and detail.isdigit() and 1 <= int(detail) <= len(rows):
            self.show_face_details(rows[int(detail) - 1][0])

    def show_face_details(self, face_id: str):
        """Show detailed face information"""
        query = """
            SELECT face_id, file_path, timestamp, image_hash, embedding_model,
                   age_estimate, gender, brightness, contrast, sharpness,
                   metadata, created_at
            FROM faces
            WHERE face_id = %s
        """

        cols, rows = self.execute_query(query, (face_id,))

        if not rows:
            print("✗ Face not found")
            return

        row = rows[0]

        print("\n" + "=" * 70)
        print("FACE DETAILS")
        print("=" * 70)

        print(f"\nFace ID: {row[0]}")
        print(f"File Path: {row[1]}")
        print(f"Image Hash: {row[3]}")
        print(f"Embedding Model: {row[4]}")
        print(f"Created: {row[11]}")

        print(f"\n--- Attributes ---")
        print(f"Gender: {row[6] or 'N/A'}")
        print(f"Age: {row[5] or 'N/A'}")
        print(f"Brightness: {row[7]:.2f}" if row[7] else "Brightness: N/A")
        print(f"Contrast: {row[8]:.2f}" if row[8] else "Contrast: N/A")
        print(f"Sharpness: {row[9]:.2f}" if row[9] else "Sharpness: N/A")

        if row[10]:  # metadata
            print(f"\n--- Full Metadata ---")
            metadata = row[10]
            for key, value in metadata.items():
                if not isinstance(value, dict):
                    print(f"{key}: {value}")

        print("=" * 70)

    def custom_query(self):
        """Execute custom SQL query"""
        print("\n" + "=" * 70)
        print("CUSTOM SQL QUERY")
        print("=" * 70)

        print("\nEnter your SQL query (end with semicolon and press Enter twice):")
        print("Example: SELECT COUNT(*) FROM faces WHERE gender = 'female';")

        lines = []
        while True:
            line = input()
            if not line and lines:
                break
            lines.append(line)

        query = "\n".join(lines).strip()

        if not query:
            print("✗ Empty query")
            return

        print("\n🔍 Executing query...")

        cols, rows = self.execute_query(query)

        if not cols:
            print("✓ Query executed successfully (no results to display)")
            return

        # Display results in table format
        print(f"\n✓ Query returned {len(rows)} rows:\n")

        # Print headers
        header = " | ".join(f"{col:<20}" for col in cols)
        print(header)
        print("-" * len(header))

        # Print rows
        for row in rows:
            row_str = " | ".join(f"{str(val):<20}" for val in row)
            print(row_str)

        print()

    def quick_queries(self):
        """Show quick query menu"""
        print("\n" + "=" * 70)
        print("QUICK QUERIES")
        print("=" * 70)

        queries = [
            ("Vector count", "SELECT COUNT(*) FROM faces WHERE embedding IS NOT NULL"),
            ("Gender distribution", "SELECT gender, COUNT(*) FROM faces GROUP BY gender"),
            ("Avg brightness by gender", "SELECT gender, AVG(brightness) FROM faces WHERE gender IS NOT NULL GROUP BY gender"),
            ("Recent 10 faces", "SELECT face_id, gender, brightness, created_at FROM faces ORDER BY created_at DESC LIMIT 10"),
            ("Brightest faces", "SELECT face_id, gender, brightness FROM faces ORDER BY brightness DESC LIMIT 10"),
            ("Darkest faces", "SELECT face_id, gender, brightness FROM faces ORDER BY brightness ASC LIMIT 10"),
        ]

        for idx, (name, _) in enumerate(queries, 1):
            print(f"{idx}. {name}")

        choice = input("\nSelect query [1-6]: ").strip()

        if not choice.isdigit() or not 1 <= int(choice) <= len(queries):
            print("✗ Invalid choice")
            return

        name, query = queries[int(choice) - 1]

        print(f"\n🔍 Executing: {name}...")

        cols, rows = self.execute_query(query)

        if not rows:
            print("✗ No results")
            return

        # Display results
        print(f"\n✓ Results:\n")

        # Print headers
        header = " | ".join(f"{col:<20}" for col in cols)
        print(header)
        print("-" * len(header))

        # Print rows
        for row in rows:
            row_str = " | ".join(f"{str(val):<20}" for val in row)
            print(row_str)

        print()

    def main_menu(self):
        """Show main menu"""
        while True:
            print("\n" + "=" * 70)
            print("PGVECTOR QUERY CLI - Main Menu")
            print("=" * 70)

            print("\n1. 📊 Show Statistics")
            print("2. 🔍 Metadata Search")
            print("3. 🎯 Vector Similarity Search")
            print("4. ⚡ Quick Queries")
            print("5. 💻 Custom SQL Query")
            print("6. ❌ Exit")

            choice = input("\nSelect option [1-6]: ").strip()

            if choice == '1':
                self.show_stats()
            elif choice == '2':
                self.search_metadata()
            elif choice == '3':
                self.similarity_search()
            elif choice == '4':
                self.quick_queries()
            elif choice == '5':
                self.custom_query()
            elif choice == '6':
                print("\n👋 Goodbye!")
                break
            else:
                print("✗ Invalid choice. Please select 1-6.")

    def run(self):
        """Run the CLI"""
        print("=" * 70)
        print("pgvector Database Query CLI")
        print("=" * 70)

        if not self.connect():
            return

        try:
            self.main_menu()
        except KeyboardInterrupt:
            print("\n\n✗ Interrupted by user")
        except Exception as e:
            print(f"\n✗ Error: {e}")
        finally:
            self.close()


def main():
    """Main entry point"""
    cli = VectorQueryCLI()
    cli.run()


if __name__ == "__main__":
    main()
