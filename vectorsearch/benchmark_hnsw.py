#!/usr/bin/env python3
"""
Benchmark script to demonstrate HNSW index performance.

Compares:
- HNSW indexed search
- Linear search performance
- Impact of different HNSW parameters
"""

import numpy as np
import time
from simple_vectorstore import SimpleVectorStore, Vector


def generate_random_vectors(num_vectors: int, dimension: int):
    """Generate random normalized vectors."""
    vectors = []
    for i in range(num_vectors):
        data = np.random.randn(dimension).astype(np.float32)
        # Normalize
        data = data / np.linalg.norm(data)
        vectors.append(data.tolist())
    return vectors


def linear_search(vectors, query, top_k, metric='cosine'):
    """Perform linear search through all vectors."""
    query_vec = Vector(query)
    results = []

    for i, vec_data in enumerate(vectors):
        vec = Vector(vec_data, f"vec_{i}")
        if metric == 'cosine':
            score = query_vec.cosine_similarity(vec)
            results.append((score, i))
        else:  # euclidean
            dist = query_vec.euclidean_distance(vec)
            results.append((-dist, i))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def benchmark_hnsw():
    """Run HNSW benchmark tests."""
    print("=== HNSW Index Benchmark ===\n")

    # Test parameters
    dimension = 128
    num_vectors = 1000
    num_queries = 50
    top_k = 10

    print(f"Configuration:")
    print(f"  Dimension: {dimension}")
    print(f"  Vectors: {num_vectors}")
    print(f"  Queries: {num_queries}")
    print(f"  Top-K: {top_k}\n")

    # Generate random vectors
    print("Generating random vectors...")
    vectors = generate_random_vectors(num_vectors, dimension)
    query_vectors = generate_random_vectors(num_queries, dimension)
    print("Done!\n")

    # Test 1: HNSW Index Performance
    print("=" * 60)
    print("Test 1: HNSW Index Performance")
    print("=" * 60)

    # Build HNSW index
    print("\nBuilding HNSW index (M=16, ef_construction=200)...")
    store = SimpleVectorStore(dimension, M=16, ef_construction=200, ef_search=50)

    start_time = time.time()
    for i, vec in enumerate(vectors):
        store.insert(vec, f"vec_{i}")
    build_time = time.time() - start_time

    print(f"Index built in {build_time:.3f} seconds")
    print(f"Average insertion time: {(build_time/num_vectors)*1000:.3f} ms/vector\n")

    # Search with HNSW
    print("Searching with HNSW index...")
    start_time = time.time()
    hnsw_results = []
    for query in query_vectors:
        results = store.search(query, top_k=top_k, metric='cosine')
        hnsw_results.append(results)
    hnsw_search_time = time.time() - start_time

    print(f"HNSW search completed in {hnsw_search_time:.3f} seconds")
    print(f"Average query time: {(hnsw_search_time/num_queries)*1000:.3f} ms/query\n")

    # Test 2: Linear Search Performance
    print("=" * 60)
    print("Test 2: Linear Search Performance (Baseline)")
    print("=" * 60)

    print("\nSearching with linear scan...")
    start_time = time.time()
    linear_results = []
    for query in query_vectors:
        results = linear_search(vectors, query, top_k, metric='cosine')
        linear_results.append(results)
    linear_search_time = time.time() - start_time

    print(f"Linear search completed in {linear_search_time:.3f} seconds")
    print(f"Average query time: {(linear_search_time/num_queries)*1000:.3f} ms/query\n")

    # Performance comparison
    print("=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    speedup = linear_search_time / hnsw_search_time
    print(f"\nHNSW Speedup: {speedup:.2f}x faster than linear search")
    print(f"Time saved per query: {((linear_search_time - hnsw_search_time)/num_queries)*1000:.3f} ms\n")

    # Test 3: Impact of ef_search parameter
    print("=" * 60)
    print("Test 3: Impact of ef_search Parameter")
    print("=" * 60)

    ef_values = [10, 50, 100, 200]
    print("\nTesting different ef_search values...\n")

    for ef in ef_values:
        store.set_ef_search(ef)
        start_time = time.time()
        for query in query_vectors:
            store.search(query, top_k=top_k, metric='cosine')
        search_time = time.time() - start_time

        avg_time = (search_time / num_queries) * 1000
        print(f"ef_search={ef:3d}: {avg_time:6.3f} ms/query")

    print("\nNote: Higher ef_search = better accuracy but slower search\n")

    # Test 4: Different M parameters
    print("=" * 60)
    print("Test 4: Impact of M Parameter (Build Time)")
    print("=" * 60)

    m_values = [4, 8, 16, 32]
    test_size = 200  # Smaller dataset for faster testing
    test_vectors = vectors[:test_size]

    print(f"\nBuilding indices with different M values ({test_size} vectors)...\n")

    for m in m_values:
        store_temp = SimpleVectorStore(dimension, M=m, ef_construction=200, ef_search=50)
        start_time = time.time()
        for i, vec in enumerate(test_vectors):
            store_temp.insert(vec, f"vec_{i}")
        build_time = time.time() - start_time

        avg_build = (build_time / test_size) * 1000
        print(f"M={m:2d}: Build time={build_time:.3f}s, Avg={avg_build:.3f} ms/vector")

    print("\nNote: Higher M = better search quality but slower build time\n")

    # Test 5: Memory usage
    print("=" * 60)
    print("Test 5: Graph Structure Analysis")
    print("=" * 60)

    print(f"\nHNSW Graph Statistics:")
    print(f"  Total nodes: {len(store.index.vectors)}")
    print(f"  Max layer: {store.index.max_layer}")
    print(f"  Entry point: {store.index.entry_point}")

    total_edges = 0
    for layer, nodes in store.index.graph.items():
        edges_in_layer = sum(len(connections) for connections in nodes.values())
        print(f"  Layer {layer}: {len(nodes)} nodes, {edges_in_layer} edges")
        total_edges += edges_in_layer

    print(f"  Total edges: {total_edges}")
    print(f"  Avg degree: {total_edges / len(store.index.vectors):.2f}\n")

    print("=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_hnsw()
