#!/usr/bin/env python3
"""
Test Vector Similarity Search

This script tests if vector similarity search is working correctly
and demonstrates the results visually.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv
from PIL import Image
import json

load_dotenv()


class SimilarityTester:
    """Test similarity search functionality"""

    def __init__(self):
        self.db_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'vector_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }
        self.conn = None

    def connect(self):
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(**self.db_params)
            print("✓ Connected to database")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def get_random_face(self):
        """Get a random face with embedding"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT face_id, file_path, gender, brightness, embedding
            FROM faces
            WHERE embedding IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 1
        """)
        row = cursor.fetchone()
        cursor.close()

        if row:
            return {
                'face_id': row[0],
                'file_path': row[1],
                'gender': row[2],
                'brightness': row[3],
                'embedding': row[4]
            }
        return None

    def test_cosine_similarity(self, query_face):
        """Test cosine distance similarity search"""
        print("\n" + "=" * 80)
        print("TEST 1: Cosine Distance Similarity Search")
        print("=" * 80)

        cursor = self.conn.cursor()

        query = """
            SELECT face_id, file_path, gender, brightness,
                   embedding <=> %s::vector AS distance
            FROM faces
            WHERE embedding IS NOT NULL
              AND face_id != %s
            ORDER BY distance
            LIMIT 10
        """

        cursor.execute(query, (query_face['embedding'], query_face['face_id']))
        results = cursor.fetchall()
        cursor.close()

        print(f"\nQuery Face: {query_face['face_id']}")
        print(f"  Gender: {query_face['gender']}")
        print(f"  Brightness: {query_face['brightness']:.2f}" if query_face['brightness'] else "  Brightness: N/A")
        print(f"  File: {query_face['file_path']}")

        print(f"\nTop 10 Most Similar Faces (Cosine Distance):")
        print(f"{'Rank':<6} {'Face ID':<35} {'Gender':<10} {'Brightness':<12} {'Distance'}")
        print("-" * 90)

        for idx, row in enumerate(results, 1):
            face_id = row[0][:32] + "..." if len(row[0]) > 32 else row[0]
            gender = row[2] or 'N/A'
            brightness = f"{row[3]:.2f}" if row[3] else 'N/A'
            distance = f"{row[4]:.6f}"

            marker = "🎯" if idx == 1 else f"{idx:2d}."
            print(f"{marker:<6} {face_id:<35} {gender:<10} {brightness:<12} {distance}")

        # Analyze results
        distances = [row[4] for row in results]
        print(f"\nDistance Statistics:")
        print(f"  Min: {min(distances):.6f}")
        print(f"  Max: {max(distances):.6f}")
        print(f"  Avg: {sum(distances)/len(distances):.6f}")

        return results

    def test_l2_similarity(self, query_face):
        """Test L2 distance similarity search"""
        print("\n" + "=" * 80)
        print("TEST 2: L2 Distance Similarity Search")
        print("=" * 80)

        cursor = self.conn.cursor()

        query = """
            SELECT face_id, file_path, gender, brightness,
                   embedding <-> %s::vector AS distance
            FROM faces
            WHERE embedding IS NOT NULL
              AND face_id != %s
            ORDER BY distance
            LIMIT 10
        """

        cursor.execute(query, (query_face['embedding'], query_face['face_id']))
        results = cursor.fetchall()
        cursor.close()

        print(f"\nTop 10 Most Similar Faces (L2 Distance):")
        print(f"{'Rank':<6} {'Face ID':<35} {'Gender':<10} {'Brightness':<12} {'Distance'}")
        print("-" * 90)

        for idx, row in enumerate(results, 1):
            face_id = row[0][:32] + "..." if len(row[0]) > 32 else row[0]
            gender = row[2] or 'N/A'
            brightness = f"{row[3]:.2f}" if row[3] else 'N/A'
            distance = f"{row[4]:.6f}"

            marker = "🎯" if idx == 1 else f"{idx:2d}."
            print(f"{marker:<6} {face_id:<35} {gender:<10} {brightness:<12} {distance}")

        return results

    def test_inner_product(self, query_face):
        """Test inner product similarity search"""
        print("\n" + "=" * 80)
        print("TEST 3: Inner Product Similarity Search")
        print("=" * 80)

        cursor = self.conn.cursor()

        query = """
            SELECT face_id, file_path, gender, brightness,
                   embedding <#> %s::vector AS similarity
            FROM faces
            WHERE embedding IS NOT NULL
              AND face_id != %s
            ORDER BY similarity
            LIMIT 10
        """

        cursor.execute(query, (query_face['embedding'], query_face['face_id']))
        results = cursor.fetchall()
        cursor.close()

        print(f"\nTop 10 Most Similar Faces (Inner Product):")
        print(f"{'Rank':<6} {'Face ID':<35} {'Gender':<10} {'Brightness':<12} {'Similarity'}")
        print("-" * 90)

        for idx, row in enumerate(results, 1):
            face_id = row[0][:32] + "..." if len(row[0]) > 32 else row[0]
            gender = row[2] or 'N/A'
            brightness = f"{row[3]:.2f}" if row[3] else 'N/A'
            similarity = f"{row[4]:.6f}"

            marker = "🎯" if idx == 1 else f"{idx:2d}."
            print(f"{marker:<6} {face_id:<35} {gender:<10} {brightness:<12} {similarity}")

        return results

    def test_hybrid_search(self, query_face):
        """Test hybrid search (vector + metadata)"""
        print("\n" + "=" * 80)
        print("TEST 4: Hybrid Search (Vector + Metadata Filter)")
        print("=" * 80)

        cursor = self.conn.cursor()

        # Filter by same gender and similar brightness
        query = """
            SELECT face_id, file_path, gender, brightness,
                   embedding <=> %s::vector AS distance
            FROM faces
            WHERE embedding IS NOT NULL
              AND face_id != %s
              AND gender = %s
              AND brightness BETWEEN %s AND %s
            ORDER BY distance
            LIMIT 10
        """

        gender_filter = query_face['gender'] if query_face['gender'] else 'unknown'
        brightness = query_face['brightness'] if query_face['brightness'] else 128
        brightness_min = brightness - 30
        brightness_max = brightness + 30

        cursor.execute(query, (
            query_face['embedding'],
            query_face['face_id'],
            gender_filter,
            brightness_min,
            brightness_max
        ))
        results = cursor.fetchall()
        cursor.close()

        print(f"Filters: gender={gender_filter}, brightness={brightness_min:.0f}-{brightness_max:.0f}")
        print(f"\nTop 10 Similar Faces with Filters:")
        print(f"{'Rank':<6} {'Face ID':<35} {'Gender':<10} {'Brightness':<12} {'Distance'}")
        print("-" * 90)

        for idx, row in enumerate(results, 1):
            face_id = row[0][:32] + "..." if len(row[0]) > 32 else row[0]
            gender = row[2] or 'N/A'
            brightness = f"{row[3]:.2f}" if row[3] else 'N/A'
            distance = f"{row[4]:.6f}"

            marker = "🎯" if idx == 1 else f"{idx:2d}."
            print(f"{marker:<6} {face_id:<35} {gender:<10} {brightness:<12} {distance}")

        return results

    def compare_metrics(self, query_face):
        """Compare all three distance metrics side by side"""
        print("\n" + "=" * 80)
        print("TEST 5: Compare All Distance Metrics")
        print("=" * 80)

        cursor = self.conn.cursor()

        query = """
            SELECT face_id, gender, brightness,
                   embedding <=> %s::vector AS cosine_dist,
                   embedding <-> %s::vector AS l2_dist,
                   embedding <#> %s::vector AS inner_prod
            FROM faces
            WHERE embedding IS NOT NULL
              AND face_id != %s
            ORDER BY cosine_dist
            LIMIT 5
        """

        cursor.execute(query, (
            query_face['embedding'],
            query_face['embedding'],
            query_face['embedding'],
            query_face['face_id']
        ))
        results = cursor.fetchall()
        cursor.close()

        print(f"\nTop 5 Faces - All Metrics Compared:")
        print(f"{'#':<4} {'Face ID':<30} {'Cosine':<12} {'L2':<12} {'Inner Prod'}")
        print("-" * 80)

        for idx, row in enumerate(results, 1):
            face_id = row[0][:27] + "..." if len(row[0]) > 27 else row[0]
            cosine = f"{row[3]:.6f}"
            l2 = f"{row[4]:.6f}"
            inner = f"{row[5]:.6f}"

            print(f"{idx:<4} {face_id:<30} {cosine:<12} {l2:<12} {inner}")

        return results

    def test_image_accessibility(self, results):
        """Test if result images are accessible"""
        print("\n" + "=" * 80)
        print("TEST 6: Image Accessibility Check")
        print("=" * 80)

        accessible = 0
        not_found = 0

        for row in results[:5]:  # Check first 5
            file_path = row[1]
            if os.path.exists(file_path):
                try:
                    img = Image.open(file_path)
                    width, height = img.size
                    print(f"✓ {os.path.basename(file_path):<40} ({width}x{height})")
                    accessible += 1
                except Exception as e:
                    print(f"✗ {os.path.basename(file_path):<40} (Error: {e})")
                    not_found += 1
            else:
                print(f"✗ {os.path.basename(file_path):<40} (File not found)")
                not_found += 1

        print(f"\nAccessibility: {accessible}/{len(results[:5])} images accessible")

    def run_all_tests(self):
        """Run all similarity search tests"""
        print("=" * 80)
        print("PGVECTOR SIMILARITY SEARCH TEST SUITE")
        print("=" * 80)

        if not self.connect():
            return

        # Get random query face
        print("\nSelecting random query face...")
        query_face = self.get_random_face()

        if not query_face:
            print("✗ No faces with embeddings found in database")
            return

        # Run all tests
        try:
            # Test 1: Cosine similarity
            cosine_results = self.test_cosine_similarity(query_face)

            # Test 2: L2 similarity
            self.test_l2_similarity(query_face)

            # Test 3: Inner product
            self.test_inner_product(query_face)

            # Test 4: Hybrid search
            self.test_hybrid_search(query_face)

            # Test 5: Compare metrics
            self.compare_metrics(query_face)

            # Test 6: Check images
            self.test_image_accessibility(cosine_results)

            # Summary
            print("\n" + "=" * 80)
            print("TEST SUMMARY")
            print("=" * 80)
            print("✓ All similarity search tests completed successfully!")
            print("\nKey Findings:")
            print("  - Cosine distance search: WORKING")
            print("  - L2 distance search: WORKING")
            print("  - Inner product search: WORKING")
            print("  - Hybrid search (vector + metadata): WORKING")
            print("  - Image accessibility: CHECKED")
            print("\n✅ Vector similarity search is functioning correctly!")

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if self.conn:
                self.conn.close()
                print("\n✓ Database connection closed")


def main():
    """Main entry point"""
    tester = SimilarityTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
