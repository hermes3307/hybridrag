#!/usr/bin/env python3
"""
Example usage of SimpleVectorStore.

This script demonstrates:
- Creating a vector store
- Inserting vectors
- Searching for similar vectors
- Deleting vectors
- Saving and loading the store
"""

from simple_vectorstore import SimpleVectorStore, Vector


def main():
    print("=== Simple Vector Store Demo ===\n")

    # Create a vector store with 3-dimensional vectors
    print("1. Creating vector store (dimension=3)")
    store = SimpleVectorStore(dimension=3)
    print(f"   {store}\n")

    # Insert some vectors
    print("2. Inserting vectors")
    id1 = store.insert([1.0, 0.0, 0.0], metadata={'name': 'vector1', 'category': 'A'})
    print(f"   Inserted: {id1}")

    id2 = store.insert([0.0, 1.0, 0.0], metadata={'name': 'vector2', 'category': 'B'})
    print(f"   Inserted: {id2}")

    id3 = store.insert([0.0, 0.0, 1.0], metadata={'name': 'vector3', 'category': 'A'})
    print(f"   Inserted: {id3}")

    id4 = store.insert([0.7, 0.7, 0.0], metadata={'name': 'vector4', 'category': 'C'})
    print(f"   Inserted: {id4}")

    id5 = store.insert([0.5, 0.5, 0.7], metadata={'name': 'vector5', 'category': 'A'})
    print(f"   Inserted: {id5}")

    print(f"\n   Store now contains {store.size()} vectors\n")

    # Search for similar vectors (cosine similarity)
    print("3. Searching for vectors similar to [1.0, 0.5, 0.0] using cosine similarity")
    query = [1.0, 0.5, 0.0]
    results = store.search(query, top_k=3, metric='cosine')

    for i, (vec_id, score, vector) in enumerate(results, 1):
        print(f"   {i}. ID: {vec_id}, Score: {score:.4f}, Metadata: {vector.metadata}")

    # Search using Euclidean distance
    print("\n4. Searching for vectors similar to [0.0, 1.0, 1.0] using Euclidean distance")
    query2 = [0.0, 1.0, 1.0]
    results = store.search(query2, top_k=3, metric='euclidean')

    for i, (vec_id, score, vector) in enumerate(results, 1):
        # Note: score is negative distance for sorting purposes
        print(f"   {i}. ID: {vec_id}, Distance: {-score:.4f}, Metadata: {vector.metadata}")

    # Get a specific vector
    print(f"\n5. Retrieving vector by ID: {id2}")
    vec = store.get(id2)
    if vec:
        print(f"   Found: {vec}")
        print(f"   Data: {vec.data}")
        print(f"   Metadata: {vec.metadata}")

    # Delete a vector
    print(f"\n6. Deleting vector: {id3}")
    deleted = store.delete(id3)
    print(f"   Deleted: {deleted}")
    print(f"   Store now contains {store.size()} vectors")

    # Try to delete non-existent vector
    deleted = store.delete("non_existent")
    print(f"   Trying to delete 'non_existent': {deleted}")

    # Save the store
    print("\n7. Saving vector store to file")
    store.save('vectorstore.json')
    print("   Saved to vectorstore.json")

    # Load the store
    print("\n8. Loading vector store from file")
    loaded_store = SimpleVectorStore.load('vectorstore.json')
    print(f"   {loaded_store}")
    print(f"   Loaded {loaded_store.size()} vectors")

    # Verify loaded store works
    print("\n9. Testing loaded store with search")
    results = loaded_store.search([1.0, 1.0, 0.0], top_k=2, metric='cosine')
    for i, (vec_id, score, vector) in enumerate(results, 1):
        print(f"   {i}. ID: {vec_id}, Score: {score:.4f}")

    # Vector operations
    print("\n10. Vector operations demo")
    v1 = Vector([3.0, 4.0, 0.0])
    v2 = Vector([1.0, 0.0, 0.0])

    print(f"   Vector 1: {v1.data}")
    print(f"   Vector 2: {v2.data}")
    print(f"   Magnitude of v1: {v1.magnitude():.4f}")
    print(f"   Dot product: {v1.dot(v2):.4f}")
    print(f"   Cosine similarity: {v1.cosine_similarity(v2):.4f}")
    print(f"   Euclidean distance: {v1.euclidean_distance(v2):.4f}")

    normalized = v1.normalize()
    print(f"   Normalized v1: {normalized.data}")
    print(f"   Magnitude of normalized: {normalized.magnitude():.4f}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
