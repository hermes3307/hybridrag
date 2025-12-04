#!/usr/bin/env python3
"""
HNSW Index Search Examples

This script demonstrates various search scenarios using HNSW index:
- Document similarity search
- Image embedding search
- Parameter tuning
- Search quality analysis
"""

import numpy as np
from simple_vectorstore import SimpleVectorStore, Vector
import time


def create_document_embeddings():
    """Create sample document embeddings with categories."""
    documents = [
        # Technology documents (cluster 1)
        {"id": "tech_1", "text": "Python programming language", "category": "technology", "embedding": [0.9, 0.1, 0.0, 0.1, 0.8, 0.2]},
        {"id": "tech_2", "text": "Machine learning algorithms", "category": "technology", "embedding": [0.8, 0.2, 0.1, 0.2, 0.9, 0.1]},
        {"id": "tech_3", "text": "Neural networks deep learning", "category": "technology", "embedding": [0.85, 0.15, 0.05, 0.15, 0.85, 0.15]},
        {"id": "tech_4", "text": "JavaScript web development", "category": "technology", "embedding": [0.88, 0.12, 0.02, 0.08, 0.75, 0.25]},
        {"id": "tech_5", "text": "Database systems SQL", "category": "technology", "embedding": [0.82, 0.18, 0.08, 0.18, 0.72, 0.28]},

        # Sports documents (cluster 2)
        {"id": "sport_1", "text": "Football championship game", "category": "sports", "embedding": [0.1, 0.9, 0.1, 0.8, 0.1, 0.2]},
        {"id": "sport_2", "text": "Basketball tournament finals", "category": "sports", "embedding": [0.15, 0.85, 0.12, 0.82, 0.15, 0.18]},
        {"id": "sport_3", "text": "Tennis grand slam", "category": "sports", "embedding": [0.12, 0.88, 0.08, 0.85, 0.12, 0.22]},
        {"id": "sport_4", "text": "Olympic games athletics", "category": "sports", "embedding": [0.08, 0.92, 0.15, 0.88, 0.08, 0.25]},
        {"id": "sport_5", "text": "Soccer world cup", "category": "sports", "embedding": [0.1, 0.9, 0.11, 0.87, 0.1, 0.21]},

        # Food documents (cluster 3)
        {"id": "food_1", "text": "Italian pasta recipes", "category": "food", "embedding": [0.2, 0.1, 0.9, 0.1, 0.15, 0.85]},
        {"id": "food_2", "text": "Japanese sushi cuisine", "category": "food", "embedding": [0.18, 0.12, 0.88, 0.12, 0.18, 0.82]},
        {"id": "food_3", "text": "French bakery desserts", "category": "food", "embedding": [0.22, 0.08, 0.92, 0.08, 0.12, 0.88]},
        {"id": "food_4", "text": "Mexican tacos burrito", "category": "food", "embedding": [0.25, 0.15, 0.85, 0.15, 0.2, 0.8]},
        {"id": "food_5", "text": "Chinese dim sum", "category": "food", "embedding": [0.19, 0.11, 0.89, 0.11, 0.16, 0.84]},

        # Health documents (cluster 4)
        {"id": "health_1", "text": "Yoga meditation wellness", "category": "health", "embedding": [0.3, 0.3, 0.3, 0.7, 0.2, 0.5]},
        {"id": "health_2", "text": "Fitness exercise training", "category": "health", "embedding": [0.35, 0.35, 0.25, 0.75, 0.25, 0.45]},
        {"id": "health_3", "text": "Nutrition healthy diet", "category": "health", "embedding": [0.28, 0.32, 0.32, 0.72, 0.22, 0.52]},
        {"id": "health_4", "text": "Mental health therapy", "category": "health", "embedding": [0.32, 0.28, 0.28, 0.78, 0.18, 0.48]},
        {"id": "health_5", "text": "Sleep quality improvement", "category": "health", "embedding": [0.3, 0.3, 0.3, 0.7, 0.2, 0.5]},
    ]

    # Normalize embeddings
    for doc in documents:
        embedding = np.array(doc["embedding"])
        doc["embedding"] = (embedding / np.linalg.norm(embedding)).tolist()

    return documents


def example_1_basic_search():
    """Example 1: Basic HNSW search with document embeddings."""
    print("=" * 70)
    print("Example 1: Basic Document Similarity Search with HNSW")
    print("=" * 70)
    print()

    # Create vector store
    dimension = 6
    store = SimpleVectorStore(dimension=dimension, M=16, ef_construction=200, ef_search=50)

    # Load documents
    documents = create_document_embeddings()
    print(f"Loading {len(documents)} documents into HNSW index...")

    for doc in documents:
        store.insert(
            doc["embedding"],
            vector_id=doc["id"],
            metadata={"text": doc["text"], "category": doc["category"]}
        )

    print(f"Index built: {store}")
    print()

    # Example queries
    queries = [
        {
            "name": "Technology Query",
            "text": "artificial intelligence and coding",
            "embedding": [0.87, 0.13, 0.05, 0.15, 0.83, 0.17]
        },
        {
            "name": "Sports Query",
            "text": "soccer and basketball",
            "embedding": [0.12, 0.88, 0.13, 0.85, 0.12, 0.20]
        },
        {
            "name": "Food Query",
            "text": "cooking and recipes",
            "embedding": [0.20, 0.10, 0.90, 0.10, 0.15, 0.85]
        }
    ]

    # Normalize query embeddings
    for query in queries:
        embedding = np.array(query["embedding"])
        query["embedding"] = (embedding / np.linalg.norm(embedding)).tolist()

    # Perform searches
    for query in queries:
        print(f"Query: {query['name']} - '{query['text']}'")
        print("-" * 70)

        results = store.search(query["embedding"], top_k=5, metric='cosine')

        for rank, (vec_id, score, vector) in enumerate(results, 1):
            print(f"  {rank}. [{vector.metadata['category']:10s}] "
                  f"Score: {score:.4f} | {vector.metadata['text']}")
        print()


def example_2_parameter_comparison():
    """Example 2: Compare search results with different ef_search values."""
    print("=" * 70)
    print("Example 2: HNSW Parameter Tuning (ef_search)")
    print("=" * 70)
    print()

    dimension = 6
    store = SimpleVectorStore(dimension=dimension, M=16, ef_construction=200, ef_search=50)

    # Load documents
    documents = create_document_embeddings()
    for doc in documents:
        store.insert(doc["embedding"], vector_id=doc["id"],
                    metadata={"text": doc["text"], "category": doc["category"]})

    # Query
    query_embedding = [0.87, 0.13, 0.05, 0.15, 0.83, 0.17]
    query_embedding = (np.array(query_embedding) / np.linalg.norm(query_embedding)).tolist()

    print("Query: 'Programming and AI technology'")
    print()

    # Test different ef_search values
    ef_values = [10, 30, 50, 100]

    for ef in ef_values:
        store.set_ef_search(ef)

        print(f"ef_search = {ef}")
        print("-" * 70)

        start_time = time.time()
        results = store.search(query_embedding, top_k=5, metric='cosine')
        search_time = (time.time() - start_time) * 1000

        for rank, (vec_id, score, vector) in enumerate(results, 1):
            print(f"  {rank}. Score: {score:.4f} | {vector.metadata['text'][:40]}")

        print(f"  Search time: {search_time:.3f} ms")
        print()


def example_3_metric_comparison():
    """Example 3: Compare cosine similarity vs euclidean distance."""
    print("=" * 70)
    print("Example 3: Distance Metric Comparison")
    print("=" * 70)
    print()

    dimension = 6
    store = SimpleVectorStore(dimension=dimension, M=16, ef_construction=200, ef_search=50)

    # Load documents
    documents = create_document_embeddings()
    for doc in documents:
        store.insert(doc["embedding"], vector_id=doc["id"],
                    metadata={"text": doc["text"], "category": doc["category"]})

    # Query
    query_embedding = [0.20, 0.10, 0.90, 0.10, 0.15, 0.85]
    query_embedding = (np.array(query_embedding) / np.linalg.norm(query_embedding)).tolist()

    print("Query: 'Food and cooking'")
    print()

    # Cosine similarity
    print("Using Cosine Similarity:")
    print("-" * 70)
    results = store.search(query_embedding, top_k=5, metric='cosine')
    for rank, (vec_id, score, vector) in enumerate(results, 1):
        print(f"  {rank}. Score: {score:.4f} | {vector.metadata['text']}")
    print()

    # Euclidean distance
    print("Using Euclidean Distance:")
    print("-" * 70)
    results = store.search(query_embedding, top_k=5, metric='euclidean')
    for rank, (vec_id, score, vector) in enumerate(results, 1):
        # Score is negative distance for sorting
        print(f"  {rank}. Distance: {-score:.4f} | {vector.metadata['text']}")
    print()


def example_4_incremental_search():
    """Example 4: Search with incremental index building."""
    print("=" * 70)
    print("Example 4: Incremental Index Building and Search")
    print("=" * 70)
    print()

    dimension = 6
    store = SimpleVectorStore(dimension=dimension, M=16, ef_construction=200, ef_search=50)

    documents = create_document_embeddings()

    # Query that we'll use throughout
    query_embedding = [0.87, 0.13, 0.05, 0.15, 0.83, 0.17]
    query_embedding = (np.array(query_embedding) / np.linalg.norm(query_embedding)).tolist()

    print("Building index incrementally and searching at each step...")
    print()

    # Add documents in batches and search
    batch_sizes = [5, 10, 15, 20]

    for batch_size in batch_sizes:
        # Add documents up to batch_size
        for doc in documents[:batch_size]:
            if store.get(doc["id"]) is None:  # Don't add duplicates
                store.insert(doc["embedding"], vector_id=doc["id"],
                           metadata={"text": doc["text"], "category": doc["category"]})

        print(f"Index size: {store.size()} documents")
        print("-" * 70)

        results = store.search(query_embedding, top_k=3, metric='cosine')

        for rank, (vec_id, score, vector) in enumerate(results, 1):
            print(f"  {rank}. [{vector.metadata['category']:10s}] "
                  f"Score: {score:.4f} | {vector.metadata['text'][:35]}")
        print()


def example_5_filtered_search():
    """Example 5: Search with post-filtering by metadata."""
    print("=" * 70)
    print("Example 5: Metadata-Filtered Search")
    print("=" * 70)
    print()

    dimension = 6
    store = SimpleVectorStore(dimension=dimension, M=16, ef_construction=200, ef_search=50)

    # Load documents
    documents = create_document_embeddings()
    for doc in documents:
        store.insert(doc["embedding"], vector_id=doc["id"],
                    metadata={"text": doc["text"], "category": doc["category"]})

    # Query
    query_embedding = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    query_embedding = (np.array(query_embedding) / np.linalg.norm(query_embedding)).tolist()

    print("Query: 'General mixed content'")
    print()

    # Search all
    print("All Results:")
    print("-" * 70)
    results = store.search(query_embedding, top_k=10, metric='cosine')
    for rank, (vec_id, score, vector) in enumerate(results, 1):
        print(f"  {rank}. [{vector.metadata['category']:10s}] "
              f"Score: {score:.4f} | {vector.metadata['text'][:35]}")
    print()

    # Filter by category
    categories = ["technology", "sports", "food"]

    for category in categories:
        print(f"Filtered Results - Category: {category}")
        print("-" * 70)

        # Get more results and filter
        all_results = store.search(query_embedding, top_k=20, metric='cosine')
        filtered = [(vid, score, vec) for vid, score, vec in all_results
                   if vec.metadata['category'] == category][:3]

        for rank, (vec_id, score, vector) in enumerate(filtered, 1):
            print(f"  {rank}. Score: {score:.4f} | {vector.metadata['text']}")
        print()


def example_6_graph_analysis():
    """Example 6: Analyze HNSW graph structure."""
    print("=" * 70)
    print("Example 6: HNSW Graph Structure Analysis")
    print("=" * 70)
    print()

    dimension = 6
    store = SimpleVectorStore(dimension=dimension, M=16, ef_construction=200, ef_search=50)

    # Load documents
    documents = create_document_embeddings()
    for doc in documents:
        store.insert(doc["embedding"], vector_id=doc["id"],
                    metadata={"text": doc["text"], "category": doc["category"]})

    print(f"Total vectors: {store.size()}")
    print(f"Dimension: {store.dimension}")
    print(f"M parameter: {store.M}")
    print()

    print("HNSW Graph Layers:")
    print("-" * 70)

    for layer in sorted(store.index.graph.keys(), reverse=True):
        nodes = store.index.graph[layer]
        total_connections = sum(len(connections) for connections in nodes.values())
        avg_connections = total_connections / len(nodes) if nodes else 0

        print(f"  Layer {layer}: {len(nodes):2d} nodes, "
              f"{total_connections:3d} connections, "
              f"avg degree: {avg_connections:.1f}")

    print()
    print(f"Entry point: {store.index.entry_point}")
    print(f"Max layer: {store.index.max_layer}")

    # Show connections for a specific node
    if store.index.entry_point:
        print()
        print(f"Connections for entry point '{store.index.entry_point}':")
        print("-" * 70)

        for layer in sorted(store.index.graph.keys(), reverse=True):
            if store.index.entry_point in store.index.graph[layer]:
                connections = store.index.graph[layer][store.index.entry_point]
                print(f"  Layer {layer}: {len(connections)} connections -> {list(connections)[:5]}")


def example_7_save_load_search():
    """Example 7: Save, load, and search."""
    print("=" * 70)
    print("Example 7: Persistence - Save, Load, and Search")
    print("=" * 70)
    print()

    dimension = 6

    # Create and populate store
    print("Creating original store...")
    store = SimpleVectorStore(dimension=dimension, M=16, ef_construction=200, ef_search=50)

    documents = create_document_embeddings()
    for doc in documents:
        store.insert(doc["embedding"], vector_id=doc["id"],
                    metadata={"text": doc["text"], "category": doc["category"]})

    print(f"Original store: {store.size()} vectors")

    # Search before saving
    query_embedding = [0.87, 0.13, 0.05, 0.15, 0.83, 0.17]
    query_embedding = (np.array(query_embedding) / np.linalg.norm(query_embedding)).tolist()

    print("\nSearching original store:")
    print("-" * 70)
    results_before = store.search(query_embedding, top_k=3, metric='cosine')
    for rank, (vec_id, score, vector) in enumerate(results_before, 1):
        print(f"  {rank}. Score: {score:.4f} | {vector.metadata['text'][:40]}")

    # Save
    print("\nSaving to 'hnsw_index.json'...")
    store.save('hnsw_index.json')
    print("Saved!")

    # Load
    print("\nLoading from 'hnsw_index.json'...")
    loaded_store = SimpleVectorStore.load('hnsw_index.json')
    print(f"Loaded store: {loaded_store.size()} vectors")

    # Search after loading
    print("\nSearching loaded store:")
    print("-" * 70)
    results_after = loaded_store.search(query_embedding, top_k=3, metric='cosine')
    for rank, (vec_id, score, vector) in enumerate(results_after, 1):
        print(f"  {rank}. Score: {score:.4f} | {vector.metadata['text'][:40]}")

    # Verify results are identical
    print()
    if results_before == results_after:
        print("✓ Results are identical before and after save/load!")
    else:
        print("✗ Results differ (this may happen due to floating point precision)")


def main():
    """Run all HNSW search examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "HNSW INDEX SEARCH EXAMPLES" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    examples = [
        example_1_basic_search,
        example_2_parameter_comparison,
        example_3_metric_comparison,
        example_4_incremental_search,
        example_5_filtered_search,
        example_6_graph_analysis,
        example_7_save_load_search,
    ]

    for i, example in enumerate(examples, 1):
        example()
        if i < len(examples):
            print("\n" + "▪" * 70 + "\n")

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
