import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import json
import heapq
import random
import math


class Vector:
    """Simple vector type with basic operations."""

    def __init__(self, data: List[float], vector_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Initialize a vector.

        Args:
            data: List of float values representing the vector
            vector_id: Optional unique identifier for the vector
            metadata: Optional metadata dictionary
        """
        self.data = np.array(data, dtype=np.float32)
        self.id = vector_id
        self.metadata = metadata or {}

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Vector(id={self.id}, dim={len(self)}, data={self.data[:3]}...)"

    def magnitude(self) -> float:
        """Calculate the magnitude (L2 norm) of the vector."""
        return float(np.linalg.norm(self.data))

    def normalize(self) -> 'Vector':
        """Return a normalized copy of the vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vector(self.data.copy(), self.id, self.metadata.copy())
        return Vector(self.data / mag, self.id, self.metadata.copy())

    def dot(self, other: 'Vector') -> float:
        """Calculate dot product with another vector."""
        return float(np.dot(self.data, other.data))

    def cosine_similarity(self, other: 'Vector') -> float:
        """Calculate cosine similarity with another vector."""
        mag_product = self.magnitude() * other.magnitude()
        if mag_product == 0:
            return 0.0
        return self.dot(other) / mag_product

    def euclidean_distance(self, other: 'Vector') -> float:
        """Calculate Euclidean distance to another vector."""
        return float(np.linalg.norm(self.data - other.data))


class HNSWIndex:
    """HNSW (Hierarchical Navigable Small World) index for fast approximate nearest neighbor search."""

    def __init__(self, dimension: int, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        """
        Initialize HNSW index.

        Args:
            dimension: Vector dimension
            M: Max number of bidirectional connections per node (except layer 0)
            ef_construction: Size of dynamic candidate list during construction
            ef_search: Size of dynamic candidate list during search
        """
        self.dimension = dimension
        self.M = M
        self.M0 = M * 2  # Max connections at layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1.0 / math.log(2.0)  # Normalization factor for level assignment

        # Graph structure: {layer: {node_id: set(connected_node_ids)}}
        self.graph: Dict[int, Dict[str, Set[str]]] = {}
        self.vectors: Dict[str, Vector] = {}
        self.node_levels: Dict[str, int] = {}  # Max level for each node
        self.entry_point: Optional[str] = None
        self.max_layer = -1

    def _get_random_level(self) -> int:
        """Randomly determine the level for a new node."""
        return int(-math.log(random.uniform(0, 1)) * self.ml)

    def _distance(self, vec1: Vector, vec2: Vector, metric: str = 'euclidean') -> float:
        """Calculate distance between two vectors."""
        if metric == 'cosine':
            return 1.0 - vec1.cosine_similarity(vec2)  # Convert similarity to distance
        else:  # euclidean
            return vec1.euclidean_distance(vec2)

    def _search_layer(self, query: Vector, entry_points: Set[str], num_closest: int,
                      layer: int, metric: str = 'euclidean') -> List[Tuple[float, str]]:
        """
        Search for nearest neighbors at a specific layer using greedy search.

        Args:
            query: Query vector
            entry_points: Starting points for the search
            num_closest: Number of closest neighbors to find
            layer: Layer to search
            metric: Distance metric

        Returns:
            List of (distance, node_id) tuples
        """
        visited = set(entry_points)
        candidates = []  # Min heap: (distance, node_id)
        w = []  # Max heap: (-distance, node_id) for nearest neighbors

        for point in entry_points:
            if point not in self.vectors:
                continue
            dist = self._distance(query, self.vectors[point], metric)
            heapq.heappush(candidates, (dist, point))
            heapq.heappush(w, (-dist, point))

        while candidates:
            current_dist, current = heapq.heappop(candidates)

            # If current is further than furthest in w, stop
            if current_dist > -w[0][0]:
                break

            # Check neighbors at this layer
            neighbors = self.graph.get(layer, {}).get(current, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor not in self.vectors:
                        continue

                    dist = self._distance(query, self.vectors[neighbor], metric)

                    if dist < -w[0][0] or len(w) < num_closest:
                        heapq.heappush(candidates, (dist, neighbor))
                        heapq.heappush(w, (-dist, neighbor))

                        # Keep only num_closest nearest
                        if len(w) > num_closest:
                            heapq.heappop(w)

        # Convert max heap back to normal distances
        return [(-dist, node_id) for dist, node_id in w]

    def _get_neighbors(self, candidates: List[Tuple[float, str]], M: int) -> List[str]:
        """
        Select M neighbors from candidates using heuristic.

        Args:
            candidates: List of (distance, node_id) tuples
            M: Number of neighbors to select

        Returns:
            List of selected node IDs
        """
        # Sort by distance and take M closest
        candidates.sort(key=lambda x: x[0])
        return [node_id for _, node_id in candidates[:M]]

    def insert(self, vector: Vector, metric: str = 'euclidean'):
        """
        Insert a vector into the HNSW index.

        Args:
            vector: Vector to insert
            metric: Distance metric ('cosine' or 'euclidean')
        """
        node_id = vector.id
        self.vectors[node_id] = vector

        # Determine level for new node
        level = self._get_random_level()
        self.node_levels[node_id] = level

        if self.entry_point is None:
            # First node
            self.entry_point = node_id
            self.max_layer = level
            # Initialize graph layers
            for lc in range(level + 1):
                if lc not in self.graph:
                    self.graph[lc] = {}
                self.graph[lc][node_id] = set()
            return

        # Find nearest neighbors at all layers
        nearest = [self.entry_point]

        # Search from top layer to level+1
        for lc in range(self.max_layer, level, -1):
            nearest = self._search_layer(vector, set(nearest), 1, lc, metric)
            nearest = [node_id for _, node_id in nearest]

        # Insert at layers from level down to 0
        for lc in range(level, -1, -1):
            if lc not in self.graph:
                self.graph[lc] = {}

            # Find ef_construction nearest neighbors at this layer
            candidates = self._search_layer(vector, set(nearest), self.ef_construction, lc, metric)

            # Select M neighbors
            M = self.M0 if lc == 0 else self.M
            neighbors = self._get_neighbors(candidates, M)

            # Add bidirectional connections
            self.graph[lc][node_id] = set()
            for neighbor in neighbors:
                self.graph[lc][node_id].add(neighbor)
                # Ensure neighbor exists in graph at this layer
                if neighbor not in self.graph[lc]:
                    self.graph[lc][neighbor] = set()
                self.graph[lc][neighbor].add(node_id)

                # Prune neighbor's connections if needed
                max_conn = self.M0 if lc == 0 else self.M
                if len(self.graph[lc][neighbor]) > max_conn:
                    # Prune to M connections
                    neighbor_vec = self.vectors[neighbor]
                    neighbor_candidates = [
                        (self._distance(neighbor_vec, self.vectors[conn], metric), conn)
                        for conn in self.graph[lc][neighbor]
                    ]
                    pruned = self._get_neighbors(neighbor_candidates, max_conn)
                    self.graph[lc][neighbor] = set(pruned)

            nearest = neighbors

        # Update entry point if necessary
        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = node_id

    def search(self, query: Vector, top_k: int, metric: str = 'euclidean') -> List[Tuple[float, str]]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vector
            top_k: Number of results to return
            metric: Distance metric ('cosine' or 'euclidean')

        Returns:
            List of (distance, node_id) tuples sorted by distance
        """
        if self.entry_point is None:
            return []

        # Start from entry point and search down
        nearest = [self.entry_point]

        # Search from top layer down to layer 1
        for lc in range(self.max_layer, 0, -1):
            nearest = self._search_layer(query, set(nearest), 1, lc, metric)
            nearest = [node_id for _, node_id in nearest]

        # Search at layer 0 with ef_search
        candidates = self._search_layer(query, set(nearest), max(self.ef_search, top_k), 0, metric)

        # Sort and return top_k
        candidates.sort(key=lambda x: x[0])
        return candidates[:top_k]

    def delete(self, vector_id: str):
        """
        Delete a vector from the index.

        Args:
            vector_id: ID of the vector to delete
        """
        if vector_id not in self.vectors:
            return False

        level = self.node_levels[vector_id]

        # Remove connections at all layers
        for lc in range(level + 1):
            if lc in self.graph and vector_id in self.graph[lc]:
                # Remove connections from neighbors
                for neighbor in self.graph[lc][vector_id]:
                    if neighbor in self.graph[lc]:
                        self.graph[lc][neighbor].discard(vector_id)

                # Remove node from graph
                del self.graph[lc][vector_id]

        # Remove vector and level info
        del self.vectors[vector_id]
        del self.node_levels[vector_id]

        # Update entry point if necessary
        if vector_id == self.entry_point:
            if self.vectors:
                # Find new entry point (node with highest level)
                self.entry_point = max(self.node_levels.keys(), key=lambda k: self.node_levels[k])
                self.max_layer = self.node_levels[self.entry_point]
            else:
                self.entry_point = None
                self.max_layer = -1

        return True


class SimpleVectorStore:
    """Simple in-memory vector store with HNSW index for fast search."""

    def __init__(self, dimension: int, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        """
        Initialize the vector store with HNSW index.

        Args:
            dimension: Expected dimension of vectors
            M: HNSW parameter - max connections per node
            ef_construction: HNSW parameter - construction time accuracy
            ef_search: HNSW parameter - search time accuracy
        """
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = HNSWIndex(dimension, M, ef_construction, ef_search)
        self._next_id = 0

    def _generate_id(self) -> str:
        """Generate a unique ID for a vector."""
        new_id = f"vec_{self._next_id}"
        self._next_id += 1
        return new_id

    def insert(self, data: List[float], vector_id: Optional[str] = None,
               metadata: Optional[Dict] = None, metric: str = 'euclidean') -> str:
        """
        Insert a vector into the store.

        Args:
            data: Vector data as list of floats
            vector_id: Optional ID for the vector (auto-generated if not provided)
            metadata: Optional metadata dictionary
            metric: Distance metric for indexing ('cosine' or 'euclidean')

        Returns:
            The ID of the inserted vector

        Raises:
            ValueError: If vector dimension doesn't match store dimension
        """
        if len(data) != self.dimension:
            raise ValueError(f"Vector dimension {len(data)} doesn't match store dimension {self.dimension}")

        if vector_id is None:
            vector_id = self._generate_id()

        if vector_id in self.index.vectors:
            raise ValueError(f"Vector with ID {vector_id} already exists")

        vector = Vector(data, vector_id, metadata)
        self.index.insert(vector, metric)
        return vector_id

    def update(self, vector_id: str, data: Optional[List[float]] = None,
               metadata: Optional[Dict] = None, metric: str = 'euclidean') -> bool:
        """
        Update a vector's data and/or metadata.

        Args:
            vector_id: ID of the vector to update
            data: Optional new vector data (if None, keeps existing data)
            metadata: Optional new metadata (if None, keeps existing metadata)
            metric: Distance metric for re-indexing if data is updated

        Returns:
            True if vector was updated, False if not found

        Raises:
            ValueError: If new data dimension doesn't match store dimension
        """
        if vector_id not in self.index.vectors:
            return False

        existing_vector = self.index.vectors[vector_id]

        # If only updating metadata, no need to rebuild index
        if data is None and metadata is not None:
            existing_vector.metadata = metadata
            return True

        # If updating data, need to re-index
        if data is not None:
            if len(data) != self.dimension:
                raise ValueError(f"Vector dimension {len(data)} doesn't match store dimension {self.dimension}")

            # Delete old vector and re-insert with new data
            self.index.delete(vector_id)

            # Merge metadata if provided, otherwise keep existing
            new_metadata = metadata if metadata is not None else existing_vector.metadata

            # Re-insert with new data
            vector = Vector(data, vector_id, new_metadata)
            self.index.insert(vector, metric)

        return True

    def batch_insert(self, vectors: List[Tuple[List[float], Optional[str], Optional[Dict]]],
                     metric: str = 'euclidean', show_progress: bool = False) -> List[str]:
        """
        Insert multiple vectors efficiently in batch.

        Args:
            vectors: List of tuples (data, vector_id, metadata)
            metric: Distance metric for indexing
            show_progress: Print progress updates for large batches

        Returns:
            List of IDs for inserted vectors

        Raises:
            ValueError: If any vector dimension doesn't match
        """
        inserted_ids = []
        total = len(vectors)

        for idx, (data, vector_id, metadata) in enumerate(vectors):
            if len(data) != self.dimension:
                raise ValueError(f"Vector {idx} dimension {len(data)} doesn't match store dimension {self.dimension}")

            vec_id = self.insert(data, vector_id, metadata, metric)
            inserted_ids.append(vec_id)

            if show_progress and (idx + 1) % 1000 == 0:
                print(f"Inserted {idx + 1}/{total} vectors...")

        if show_progress and total >= 1000:
            print(f"Batch insert complete: {total} vectors inserted")

        return inserted_ids

    def batch_update(self, updates: List[Tuple[str, Optional[List[float]], Optional[Dict]]],
                     metric: str = 'euclidean', show_progress: bool = False) -> int:
        """
        Update multiple vectors efficiently in batch.

        Args:
            updates: List of tuples (vector_id, data, metadata)
            metric: Distance metric for re-indexing
            show_progress: Print progress updates for large batches

        Returns:
            Number of vectors successfully updated
        """
        updated_count = 0
        total = len(updates)

        for idx, (vector_id, data, metadata) in enumerate(updates):
            if self.update(vector_id, data, metadata, metric):
                updated_count += 1

            if show_progress and (idx + 1) % 1000 == 0:
                print(f"Updated {idx + 1}/{total} vectors...")

        if show_progress and total >= 1000:
            print(f"Batch update complete: {updated_count}/{total} vectors updated")

        return updated_count

    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector from the store.

        Args:
            vector_id: ID of the vector to delete

        Returns:
            True if vector was deleted, False if not found
        """
        return self.index.delete(vector_id)

    def get(self, vector_id: str) -> Optional[Vector]:
        """
        Retrieve a vector by ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            The vector if found, None otherwise
        """
        return self.index.vectors.get(vector_id)

    def search(self, query: List[float], top_k: int = 5,
               metric: str = 'euclidean') -> List[Tuple[str, float, Vector]]:
        """
        Search for similar vectors using HNSW index.

        Args:
            query: Query vector as list of floats
            top_k: Number of results to return
            metric: Distance metric ('cosine' or 'euclidean')

        Returns:
            List of tuples (vector_id, similarity_score, vector) sorted by similarity

        Raises:
            ValueError: If query dimension doesn't match or invalid metric
        """
        if len(query) != self.dimension:
            raise ValueError(f"Query dimension {len(query)} doesn't match store dimension {self.dimension}")

        if metric not in ['cosine', 'euclidean']:
            raise ValueError(f"Invalid metric '{metric}'. Use 'cosine' or 'euclidean'")

        query_vector = Vector(query)
        results = self.index.search(query_vector, top_k, metric)

        # Convert distances to similarity scores
        output = []
        for dist, node_id in results:
            vector = self.index.vectors[node_id]
            if metric == 'cosine':
                score = 1.0 - dist  # Convert distance back to similarity
            else:  # euclidean
                score = -dist  # Negative distance for sorting compatibility
            output.append((node_id, score, vector))

        return output

    def size(self) -> int:
        """Return the number of vectors in the store."""
        return len(self.index.vectors)

    def get_memory_info(self) -> Dict:
        """
        Get approximate memory usage information for the vector store.

        Returns:
            Dictionary with memory usage statistics
        """
        import sys

        # Calculate vector data memory
        vector_memory = sum(vec.data.nbytes for vec in self.index.vectors.values())

        # Calculate graph structure memory (approximate)
        graph_connections = sum(
            len(connections)
            for layer_nodes in self.index.graph.values()
            for connections in layer_nodes.values()
        )

        # Approximate memory per connection (pointer + overhead)
        connection_memory = graph_connections * sys.getsizeof("")

        # Metadata memory (approximate)
        metadata_memory = sum(
            sys.getsizeof(str(vec.metadata))
            for vec in self.index.vectors.values()
        )

        total_memory = vector_memory + connection_memory + metadata_memory

        return {
            'total_bytes': total_memory,
            'total_mb': total_memory / (1024 * 1024),
            'vector_data_bytes': vector_memory,
            'vector_data_mb': vector_memory / (1024 * 1024),
            'graph_connections': graph_connections,
            'graph_memory_bytes': connection_memory,
            'metadata_memory_bytes': metadata_memory,
            'num_vectors': len(self.index.vectors),
            'num_layers': len(self.index.graph),
            'dimension': self.dimension
        }

    def clear(self):
        """Remove all vectors from the store."""
        self.index = HNSWIndex(self.dimension, self.M, self.ef_construction, self.ef_search)
        self._next_id = 0

    def save(self, filepath: str):
        """
        Save the vector store to a JSON file.

        Args:
            filepath: Path to save the store
        """
        # Convert graph structure to serializable format
        graph_data = {}
        for layer, nodes in self.index.graph.items():
            graph_data[str(layer)] = {
                node_id: list(connections)
                for node_id, connections in nodes.items()
            }

        data = {
            'dimension': self.dimension,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'next_id': self._next_id,
            'entry_point': self.index.entry_point,
            'max_layer': self.index.max_layer,
            'node_levels': self.index.node_levels,
            'graph': graph_data,
            'vectors': [
                {
                    'id': vec.id,
                    'data': vec.data.tolist(),
                    'metadata': vec.metadata
                }
                for vec in self.index.vectors.values()
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'SimpleVectorStore':
        """
        Load a vector store from a JSON file.

        Args:
            filepath: Path to load the store from

        Returns:
            Loaded SimpleVectorStore instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        store = cls(
            data['dimension'],
            data.get('M', 16),
            data.get('ef_construction', 200),
            data.get('ef_search', 50)
        )
        store._next_id = data['next_id']

        # Restore index state
        store.index.entry_point = data.get('entry_point')
        store.index.max_layer = data.get('max_layer', -1)
        store.index.node_levels = data.get('node_levels', {})

        # Restore vectors
        for vec_data in data['vectors']:
            vector = Vector(vec_data['data'], vec_data['id'], vec_data['metadata'])
            store.index.vectors[vector.id] = vector

        # Restore graph structure
        if 'graph' in data:
            for layer_str, nodes in data['graph'].items():
                layer = int(layer_str)
                store.index.graph[layer] = {
                    node_id: set(connections)
                    for node_id, connections in nodes.items()
                }

        return store

    def set_ef_search(self, ef_search: int):
        """
        Update the ef_search parameter for search quality/speed tradeoff.

        Args:
            ef_search: New ef_search value (higher = better accuracy, slower search)
        """
        self.ef_search = ef_search
        self.index.ef_search = ef_search

    def __repr__(self):
        return f"SimpleVectorStore(dimension={self.dimension}, size={self.size()}, index=HNSW, M={self.M})"
