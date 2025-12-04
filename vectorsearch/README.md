# Simple Vector Store with HNSW Index

A Python implementation of an in-memory vector database with HNSW (Hierarchical Navigable Small World) indexing for fast approximate nearest neighbor search.

## Features

- **Vector Type**: Custom vector class with common operations
  - Magnitude calculation
  - Normalization
  - Dot product
  - Cosine similarity
  - Euclidean distance

- **HNSW Index**: Fast approximate nearest neighbor search
  - Hierarchical graph structure
  - Logarithmic search complexity
  - Configurable accuracy/speed tradeoff

- **Operations**:
  - Insert: Add vectors with optional metadata
  - Delete: Remove vectors and repair graph connections
  - Search: Find top-k similar vectors (cosine or Euclidean)
  - Get: Retrieve vector by ID

- **Persistence**: Save/load to JSON files

## Quick Start

```python
from simple_vectorstore import SimpleVectorStore

# Create a vector store with 128-dimensional vectors
store = SimpleVectorStore(dimension=128)

# Insert vectors
id1 = store.insert([0.1, 0.2, ...], metadata={'name': 'vector1'})
id2 = store.insert([0.3, 0.4, ...], metadata={'name': 'vector2'})

# Search for similar vectors
query = [0.15, 0.25, ...]
results = store.search(query, top_k=5, metric='cosine')

for vec_id, score, vector in results:
    print(f"ID: {vec_id}, Score: {score:.4f}, Metadata: {vector.metadata}")

# Delete a vector
store.delete(id1)

# Save to file
store.save('vectorstore.json')

# Load from file
loaded_store = SimpleVectorStore.load('vectorstore.json')
```

## HNSW Parameters

The vector store supports several parameters to tune performance:

- **M** (default: 16): Maximum number of connections per node
  - Higher M = better search quality, slower build time
  - Typical values: 4-64

- **ef_construction** (default: 200): Construction time accuracy
  - Higher = better index quality, slower build time
  - Typical values: 100-500

- **ef_search** (default: 50): Search time accuracy
  - Higher = better search accuracy, slower queries
  - Can be adjusted at runtime with `set_ef_search()`
  - Typical values: 10-200

```python
# Create store with custom parameters
store = SimpleVectorStore(
    dimension=128,
    M=16,
    ef_construction=200,
    ef_search=50
)

# Adjust search accuracy at runtime
store.set_ef_search(100)  # Higher accuracy
```

## Performance

Based on benchmark with 1000 vectors (dimension=128):

- **Build Time**: ~4ms per vector (M=16)
- **Search Time**: ~2.4ms per query (ef_search=50)
- **Speedup**: 2.57x faster than linear search
- **Scalability**: Search time grows logarithmically with dataset size

Performance comparison by ef_search:
- ef_search=10: 1.5ms/query (fastest)
- ef_search=50: 2.7ms/query (balanced)
- ef_search=100: 3.9ms/query (accurate)

## Distance Metrics

Two distance metrics are supported:

- **Cosine Similarity**: Measures angle between vectors (best for normalized vectors)
- **Euclidean Distance**: Measures straight-line distance (best for spatial data)

```python
# Cosine similarity search
results = store.search(query, top_k=5, metric='cosine')

# Euclidean distance search
results = store.search(query, top_k=5, metric='euclidean')
```

## Examples

Run the example script:
```bash
python3 example_usage.py
```

Run the benchmark:
```bash
python3 benchmark_hnsw.py
```

## Implementation Details

### HNSW Algorithm

The implementation uses a hierarchical graph structure:

1. **Layered Graph**: Multiple layers with decreasing node count
2. **Layer Assignment**: Nodes randomly assigned to max layer
3. **Insertion**:
   - Search from top layer down
   - Connect to M nearest neighbors at each layer
   - Prune connections to maintain graph quality
4. **Search**:
   - Start from entry point at top layer
   - Greedily traverse to nearest neighbor
   - Descend layers and refine search
   - Return top-k results from layer 0

### Graph Structure

- Layer 0: Contains all vectors with M0 (2*M) connections each
- Higher layers: Contain subset of vectors with M connections each
- Entry point: Highest-level node for starting searches
- Bidirectional edges for efficient traversal

## Files

- `simple_vectorstore.py`: Main implementation
- `example_usage.py`: Basic usage demonstration
- `benchmark_hnsw.py`: Performance benchmarks
- `README.md`: This file

## Requirements

- Python 3.7+
- NumPy

Install dependencies:
```bash
pip install numpy
```

## Limitations

- In-memory only (no disk-based storage beyond JSON export)
- Approximate search (not exact nearest neighbors)
- No batch operations
- Single-threaded

## Future Enhancements

Potential improvements:
- Batch insertion/search
- More sophisticated pruning heuristics
- Additional distance metrics
- Parallel search
- Compressed vector storage
- Incremental persistence

## References

- Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. IEEE transactions on pattern analysis and machine intelligence.
