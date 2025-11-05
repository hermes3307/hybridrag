#!/usr/bin/env python3
"""
Test script for metadata search functionality
"""

from core import SystemConfig
from pgvector_db import PgVectorDatabaseManager

# Initialize system
config = SystemConfig.from_file("system_config.json")
db_manager = PgVectorDatabaseManager(config)

if not db_manager.initialize():
    print("Failed to initialize database")
    exit(1)

print("Database initialized successfully")
print()

# Test 1: Search for male faces
print("=" * 60)
print("Test 1: Search for male faces")
print("=" * 60)
metadata_filter = {'estimated_sex': 'male'}
results = db_manager.search_by_metadata(metadata_filter, n_results=5)

print(f"Found {len(results)} results")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"  Face ID: {result['id']}")
    print(f"  File: {result['metadata'].get('file_path', 'Unknown')}")
    print(f"  Sex: {result['metadata'].get('estimated_sex', 'Unknown')}")
    print(f"  Age: {result['metadata'].get('age_group', 'Unknown')}")
    print(f"  Distance: {result.get('distance', 'N/A')}")

# Test 2: Search for female faces
print("\n" + "=" * 60)
print("Test 2: Search for female faces")
print("=" * 60)
metadata_filter = {'estimated_sex': 'female'}
results = db_manager.search_by_metadata(metadata_filter, n_results=5)

print(f"Found {len(results)} results")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"  Face ID: {result['id']}")
    print(f"  File: {result['metadata'].get('file_path', 'Unknown')}")
    print(f"  Sex: {result['metadata'].get('estimated_sex', 'Unknown')}")
    print(f"  Age: {result['metadata'].get('age_group', 'Unknown')}")
    print(f"  Distance: {result.get('distance', 'N/A')}")

# Test 3: Combined filter - male + blonde hair
print("\n" + "=" * 60)
print("Test 3: Combined filter - male + blonde hair")
print("=" * 60)
metadata_filter = {
    'estimated_sex': 'male',
    'hair_color': 'blonde'
}
results = db_manager.search_by_metadata(metadata_filter, n_results=5)

print(f"Found {len(results)} results")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"  Face ID: {result['id']}")
    print(f"  File: {result['metadata'].get('file_path', 'Unknown')}")
    print(f"  Sex: {result['metadata'].get('estimated_sex', 'Unknown')}")
    print(f"  Hair: {result['metadata'].get('hair_color', 'Unknown')}")
    print(f"  Distance: {result.get('distance', 'N/A')}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
