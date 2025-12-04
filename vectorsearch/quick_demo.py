#!/usr/bin/env python3
"""
Quick Demo - Enhanced Vector Store

Shows practical usage in under 50 lines.
"""

from enhanced_vectorstore import EnhancedVectorStore
import numpy as np

print("=" * 70)
print("  QUICK DEMO: Enhanced In-Memory Vector Store")
print("=" * 70)

# 1. Create vector store with HNSW index
print("\n1. Creating vector store with HNSW index...")
store = EnhancedVectorStore(
    dimension=8,
    index_type="hnsw",
    metric="cosine"
)
print(f"   {store}")

# 2. Insert documents with metadata and tags
print("\n2. Inserting documents...")
documents = [
    {"text": "Python machine learning tutorial", "category": "tech", "tags": ["tutorial", "ml"]},
    {"text": "Advanced JavaScript frameworks", "category": "tech", "tags": ["tutorial", "js"]},
    {"text": "Football championship highlights", "category": "sports", "tags": ["sports", "video"]},
    {"text": "Italian pasta recipes", "category": "food", "tags": ["recipe", "italian"]},
    {"text": "Yoga and meditation guide", "category": "health", "tags": ["wellness", "tutorial"]},
    {"text": "Deep learning with PyTorch", "category": "tech", "tags": ["ml", "tutorial", "advanced"]},
    {"text": "Basketball training tips", "category": "sports", "tags": ["sports", "tutorial"]},
    {"text": "French bakery techniques", "category": "food", "tags": ["recipe", "french"]},
]

for i, doc in enumerate(documents):
    # Simulate embeddings with random vectors
    embedding = np.random.randn(8).tolist()

    store.insert(
        embedding,
        vector_id=f"doc_{i}",
        metadata={'text': doc['text'], 'category': doc['category']},
        tags=doc['tags'],
        normalize=True
    )
    print(f"   ✓ Inserted: {doc['text'][:40]}")

# 3. Basic search
print("\n3. Basic search - finding similar documents...")
query = np.random.randn(8).tolist()
results = store.search(query, top_k=3)

print(f"\n   Top 3 similar documents:")
for rank, (doc_id, distance, vector) in enumerate(results, 1):
    print(f"   {rank}. {vector.metadata['text'][:50]}")
    print(f"      Distance: {distance:.4f}, Tags: {vector.tags}")

# 4. Tag-based search
print("\n4. Tag-based search - finding tutorials...")
results = store.search_by_tags(query, tags=['tutorial'], top_k=3)

print(f"\n   Top 3 tutorials:")
for rank, (doc_id, distance, vector) in enumerate(results, 1):
    print(f"   {rank}. {vector.metadata['text'][:50]}")

# 5. Metadata filtering
print("\n5. Metadata filtering - finding tech documents...")
results = store.search_by_metadata(query, {'category': 'tech'}, top_k=3)

print(f"\n   Top 3 tech documents:")
for rank, (doc_id, distance, vector) in enumerate(results, 1):
    print(f"   {rank}. {vector.metadata['text'][:50]}")

# 6. Update a document
print("\n6. Updating a document...")
print(f"   Original: {store.get('doc_0').metadata}")
store.update('doc_0', metadata={'text': 'Python ML tutorial - UPDATED', 'category': 'tech', 'featured': True})
store.update('doc_0', add_tags=['featured'])
print(f"   Updated:  {store.get('doc_0').metadata}")
print(f"   Tags:     {store.get('doc_0').tags}")
print(f"   Version:  {store.get('doc_0').version}")

# 7. Get statistics
print("\n7. Vector store statistics...")
stats = store.get_stats()
print(f"   Total vectors: {stats['num_vectors']}")
print(f"   Memory usage: {stats['memory']['total_mb']:.3f} MB")
print(f"   Unique tags: {stats['vectors']['unique_tags']}")
print(f"   Index type: {stats['type']}")
print(f"   HNSW layers: {stats['graph']['num_layers']}")

# 8. Save and load
print("\n8. Testing persistence...")
store.save('demo_store.json')
print(f"   ✓ Saved to demo_store.json")

loaded = EnhancedVectorStore.load('demo_store.json')
print(f"   ✓ Loaded: {loaded}")
print(f"   ✓ All {loaded.size()} vectors restored with metadata and tags")

# 9. Compare with original simple store
print("\n9. Also available: Simple Vector Store (HNSW only)...")
from simple_vectorstore import SimpleVectorStore

simple = SimpleVectorStore(dimension=8)
for i in range(8):
    simple.insert(np.random.randn(8).tolist(), metadata={'doc': f'doc_{i}'})

results = simple.search(query, top_k=3, metric='cosine')
print(f"   Simple store: {simple}")
print(f"   Search works: {len(results)} results found")

print("\n" + "=" * 70)
print("  DEMO COMPLETE!")
print("=" * 70)
print("\nKey Features Demonstrated:")
print("  ✓ Multiple index types (HNSW, Flat)")
print("  ✓ Rich metadata and tags")
print("  ✓ Tag-based filtering")
print("  ✓ Metadata filtering")
print("  ✓ Update operations with versioning")
print("  ✓ Comprehensive statistics")
print("  ✓ Save/Load persistence")
print("\nFiles:")
print("  - enhanced_vectorstore.py  (1,258 lines - full features)")
print("  - simple_vectorstore.py    (528 lines - basic HNSW)")
print("  - test_enhanced.py         (comprehensive tests)")
print("  - ENHANCED_FEATURES.md     (complete documentation)")
print("  - COMPARISON.md            (simple vs enhanced)")
