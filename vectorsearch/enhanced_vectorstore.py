#!/usr/bin/env python3
"""
Enhanced Vector Store with Multiple Index Types

This module provides:
- Enhanced Vector class with validation, versioning, and rich metadata
- Abstract base class for vector indexes
- Multiple index implementations (Flat, HNSW, IVF)
- Advanced search capabilities
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from abc import ABC, abstractmethod
import json
import heapq
import random
import math
import time
from datetime import datetime
from enum import Enum


class DistanceMetric(Enum):
    """Supported distance metrics."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class VectorStatus(Enum):
    """Vector status."""
    ACTIVE = "active"
    DELETED = "deleted"
    ARCHIVED = "archived"


class EnhancedVector:
    """
    Enhanced vector class with rich metadata and validation.

    Features:
    - Automatic normalization
    - Versioning support
    - Timestamps
    - Tags and categories
    - Custom metadata
    - Validation
    """

    def __init__(
        self,
        data: Union[List[float], np.ndarray],
        vector_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        normalize: bool = False,
        validate: bool = True
    ):
        """
        Initialize an enhanced vector.

        Args:
            data: Vector data
            vector_id: Unique identifier
            metadata: Custom metadata dictionary
            tags: List of tags for categorization
            normalize: Whether to normalize the vector
            validate: Whether to validate input
        """
        # Convert to numpy array
        self.data = np.array(data, dtype=np.float32)

        if validate:
            self._validate()

        if normalize:
            self._normalize_inplace()

        self.id = vector_id
        self.metadata = metadata or {}
        self.tags = tags or []

        # Automatic fields
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.version = 1
        self.status = VectorStatus.ACTIVE
        self._hash = None

    def _validate(self):
        """Validate vector data."""
        if len(self.data) == 0:
            raise ValueError("Vector cannot be empty")

        if not np.all(np.isfinite(self.data)):
            raise ValueError("Vector contains invalid values (NaN or Inf)")

        if np.all(self.data == 0):
            raise ValueError("Vector cannot be all zeros")

    def _normalize_inplace(self):
        """Normalize vector in place."""
        norm = np.linalg.norm(self.data)
        if norm > 0:
            self.data /= norm

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"EnhancedVector(id={self.id}, dim={len(self)}, version={self.version}, tags={self.tags})"

    def __hash__(self):
        """Compute hash for vector (cached)."""
        if self._hash is None:
            self._hash = hash((self.id, self.data.tobytes()))
        return self._hash

    def __eq__(self, other):
        """Check equality."""
        if not isinstance(other, EnhancedVector):
            return False
        return self.id == other.id and np.array_equal(self.data, other.data)

    # Vector operations
    def magnitude(self) -> float:
        """Calculate L2 norm."""
        return float(np.linalg.norm(self.data))

    def normalize(self) -> 'EnhancedVector':
        """Return normalized copy."""
        normalized_data = self.data / self.magnitude() if self.magnitude() > 0 else self.data.copy()
        vec = EnhancedVector(
            normalized_data,
            self.id,
            self.metadata.copy(),
            self.tags.copy(),
            validate=False
        )
        vec.created_at = self.created_at
        vec.updated_at = self.updated_at
        vec.version = self.version
        vec.status = self.status
        return vec

    def dot(self, other: 'EnhancedVector') -> float:
        """Dot product."""
        return float(np.dot(self.data, other.data))

    def cosine_similarity(self, other: 'EnhancedVector') -> float:
        """Cosine similarity."""
        mag_product = self.magnitude() * other.magnitude()
        if mag_product == 0:
            return 0.0
        return self.dot(other) / mag_product

    def euclidean_distance(self, other: 'EnhancedVector') -> float:
        """Euclidean distance."""
        return float(np.linalg.norm(self.data - other.data))

    def manhattan_distance(self, other: 'EnhancedVector') -> float:
        """Manhattan (L1) distance."""
        return float(np.sum(np.abs(self.data - other.data)))

    def distance(self, other: 'EnhancedVector', metric: DistanceMetric) -> float:
        """Calculate distance using specified metric."""
        if metric == DistanceMetric.EUCLIDEAN:
            return self.euclidean_distance(other)
        elif metric == DistanceMetric.COSINE:
            return 1.0 - self.cosine_similarity(other)
        elif metric == DistanceMetric.DOT_PRODUCT:
            return -self.dot(other)  # Negative for max -> min
        elif metric == DistanceMetric.MANHATTAN:
            return self.manhattan_distance(other)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Metadata operations
    def add_tag(self, tag: str):
        """Add a tag."""
        if tag not in self.tags:
            self.tags.append(tag)
            self._update_timestamp()

    def remove_tag(self, tag: str):
        """Remove a tag."""
        if tag in self.tags:
            self.tags.remove(tag)
            self._update_timestamp()

    def has_tag(self, tag: str) -> bool:
        """Check if vector has tag."""
        return tag in self.tags

    def update_metadata(self, metadata: Dict[str, Any], merge: bool = True):
        """Update metadata."""
        if merge:
            self.metadata.update(metadata)
        else:
            self.metadata = metadata
        self._update_timestamp()

    def _update_timestamp(self):
        """Update timestamp and increment version."""
        self.updated_at = datetime.now().isoformat()
        self.version += 1
        self._hash = None  # Invalidate hash cache

    def update_data(self, data: Union[List[float], np.ndarray], normalize: bool = False):
        """Update vector data."""
        self.data = np.array(data, dtype=np.float32)
        self._validate()
        if normalize:
            self._normalize_inplace()
        self._update_timestamp()

    def mark_deleted(self):
        """Mark vector as deleted."""
        self.status = VectorStatus.DELETED
        self._update_timestamp()

    def mark_archived(self):
        """Mark vector as archived."""
        self.status = VectorStatus.ARCHIVED
        self._update_timestamp()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'data': self.data.tolist(),
            'metadata': self.metadata,
            'tags': self.tags,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'version': self.version,
            'status': self.status.value
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EnhancedVector':
        """Create from dictionary."""
        vec = cls(
            data['data'],
            data.get('id'),
            data.get('metadata'),
            data.get('tags'),
            validate=False
        )
        vec.created_at = data.get('created_at', vec.created_at)
        vec.updated_at = data.get('updated_at', vec.updated_at)
        vec.version = data.get('version', 1)
        vec.status = VectorStatus(data.get('status', 'active'))
        return vec


class VectorIndex(ABC):
    """Abstract base class for vector indexes."""

    def __init__(self, dimension: int, metric: DistanceMetric = DistanceMetric.EUCLIDEAN):
        """
        Initialize index.

        Args:
            dimension: Vector dimension
            metric: Distance metric to use
        """
        self.dimension = dimension
        self.metric = metric
        self.vectors: Dict[str, EnhancedVector] = {}

    @abstractmethod
    def insert(self, vector: EnhancedVector):
        """Insert a vector into the index."""
        pass

    @abstractmethod
    def search(self, query: EnhancedVector, top_k: int) -> List[Tuple[float, str]]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vector
            top_k: Number of results

        Returns:
            List of (distance, vector_id) tuples
        """
        pass

    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        """Delete a vector from the index."""
        pass

    def get(self, vector_id: str) -> Optional[EnhancedVector]:
        """Get vector by ID."""
        return self.vectors.get(vector_id)

    def size(self) -> int:
        """Get number of vectors."""
        return len(self.vectors)

    @abstractmethod
    def get_index_stats(self) -> Dict:
        """Get index statistics."""
        pass


class FlatIndex(VectorIndex):
    """
    Flat (brute-force) index for exact nearest neighbor search.

    Guarantees 100% recall but O(n) search time.
    Best for small datasets or when exact results are required.
    """

    def __init__(self, dimension: int, metric: DistanceMetric = DistanceMetric.EUCLIDEAN):
        super().__init__(dimension, metric)
        self.vector_matrix = None
        self.vector_ids = []
        self._needs_rebuild = False

    def insert(self, vector: EnhancedVector):
        """Insert vector."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match index dimension {self.dimension}")

        self.vectors[vector.id] = vector
        self.vector_ids.append(vector.id)
        self._needs_rebuild = True

    def _rebuild_matrix(self):
        """Rebuild the vector matrix."""
        if not self.vectors:
            self.vector_matrix = None
            self.vector_ids = []
            return

        self.vector_ids = list(self.vectors.keys())
        self.vector_matrix = np.vstack([self.vectors[vid].data for vid in self.vector_ids])
        self._needs_rebuild = False

    def search(self, query: EnhancedVector, top_k: int) -> List[Tuple[float, str]]:
        """Brute-force search."""
        if not self.vectors:
            return []

        if self._needs_rebuild:
            self._rebuild_matrix()

        # Compute distances to all vectors
        if self.metric == DistanceMetric.EUCLIDEAN:
            distances = np.linalg.norm(self.vector_matrix - query.data, axis=1)
        elif self.metric == DistanceMetric.COSINE:
            # Cosine distance = 1 - cosine similarity
            similarities = np.dot(self.vector_matrix, query.data) / (
                np.linalg.norm(self.vector_matrix, axis=1) * np.linalg.norm(query.data)
            )
            distances = 1.0 - similarities
        elif self.metric == DistanceMetric.DOT_PRODUCT:
            distances = -np.dot(self.vector_matrix, query.data)
        elif self.metric == DistanceMetric.MANHATTAN:
            distances = np.sum(np.abs(self.vector_matrix - query.data), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Get top-k
        top_k = min(top_k, len(distances))
        if top_k == 0:
            return []

        if top_k >= len(distances):
            # Return all sorted
            sorted_indices = np.argsort(distances)
            return [(float(distances[i]), self.vector_ids[i]) for i in sorted_indices]
        else:
            # Use argpartition for partial sort
            top_indices = np.argpartition(distances, top_k - 1)[:top_k]
            top_indices = top_indices[np.argsort(distances[top_indices])]
            return [(float(distances[i]), self.vector_ids[i]) for i in top_indices]

    def delete(self, vector_id: str) -> bool:
        """Delete vector."""
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            self._needs_rebuild = True
            return True
        return False

    def get_index_stats(self) -> Dict:
        """Get statistics."""
        return {
            'type': 'FlatIndex',
            'dimension': self.dimension,
            'metric': self.metric.value,
            'num_vectors': len(self.vectors),
            'search_complexity': 'O(n)',
            'recall': '100%'
        }


class HNSWIndex(VectorIndex):
    """
    Enhanced HNSW (Hierarchical Navigable Small World) index.

    Features:
    - Fast approximate nearest neighbor search
    - Configurable accuracy/speed tradeoff
    - Incremental updates
    - Graph statistics
    """

    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_M: int = None,
        max_M0: int = None
    ):
        """
        Initialize HNSW index.

        Args:
            dimension: Vector dimension
            metric: Distance metric
            M: Max connections per node (except layer 0)
            ef_construction: Construction time accuracy parameter
            ef_search: Search time accuracy parameter
            max_M: Maximum M value (default: M)
            max_M0: Maximum connections at layer 0 (default: M * 2)
        """
        super().__init__(dimension, metric)
        self.M = M
        self.M0 = max_M0 or (M * 2)
        self.max_M = max_M or M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1.0 / math.log(2.0)

        self.graph: Dict[int, Dict[str, Set[str]]] = {}
        self.node_levels: Dict[str, int] = {}
        self.entry_point: Optional[str] = None
        self.max_layer = -1

        # Statistics
        self.insert_count = 0
        self.search_count = 0
        self.total_search_time = 0.0

    def _get_random_level(self) -> int:
        """Randomly determine level for new node."""
        return int(-math.log(random.uniform(0, 1)) * self.ml)

    def _distance(self, vec1: EnhancedVector, vec2: EnhancedVector) -> float:
        """Calculate distance between vectors."""
        return vec1.distance(vec2, self.metric)

    def _search_layer(
        self,
        query: EnhancedVector,
        entry_points: Set[str],
        num_closest: int,
        layer: int
    ) -> List[Tuple[float, str]]:
        """Search for nearest neighbors at a specific layer."""
        visited = set(entry_points)
        candidates = []
        w = []

        for point in entry_points:
            if point not in self.vectors:
                continue
            dist = self._distance(query, self.vectors[point])
            heapq.heappush(candidates, (dist, point))
            heapq.heappush(w, (-dist, point))

        while candidates:
            current_dist, current = heapq.heappop(candidates)

            if current_dist > -w[0][0]:
                break

            neighbors = self.graph.get(layer, {}).get(current, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor not in self.vectors:
                        continue

                    dist = self._distance(query, self.vectors[neighbor])

                    if dist < -w[0][0] or len(w) < num_closest:
                        heapq.heappush(candidates, (dist, neighbor))
                        heapq.heappush(w, (-dist, neighbor))

                        if len(w) > num_closest:
                            heapq.heappop(w)

        return [(-dist, node_id) for dist, node_id in w]

    def _get_neighbors(self, candidates: List[Tuple[float, str]], M: int) -> List[str]:
        """Select M neighbors using heuristic."""
        candidates.sort(key=lambda x: x[0])
        return [node_id for _, node_id in candidates[:M]]

    def insert(self, vector: EnhancedVector):
        """Insert vector into HNSW index."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match index dimension {self.dimension}")

        node_id = vector.id
        self.vectors[node_id] = vector
        level = self._get_random_level()
        self.node_levels[node_id] = level

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = level
            for lc in range(level + 1):
                if lc not in self.graph:
                    self.graph[lc] = {}
                self.graph[lc][node_id] = set()
            self.insert_count += 1
            return

        nearest = [self.entry_point]

        for lc in range(self.max_layer, level, -1):
            nearest = self._search_layer(vector, set(nearest), 1, lc)
            nearest = [node_id for _, node_id in nearest]

        for lc in range(level, -1, -1):
            if lc not in self.graph:
                self.graph[lc] = {}

            candidates = self._search_layer(vector, set(nearest), self.ef_construction, lc)
            M = self.M0 if lc == 0 else self.M
            neighbors = self._get_neighbors(candidates, M)

            self.graph[lc][node_id] = set()
            for neighbor in neighbors:
                self.graph[lc][node_id].add(neighbor)
                if neighbor not in self.graph[lc]:
                    self.graph[lc][neighbor] = set()
                self.graph[lc][neighbor].add(node_id)

                max_conn = self.M0 if lc == 0 else self.M
                if len(self.graph[lc][neighbor]) > max_conn:
                    neighbor_vec = self.vectors[neighbor]
                    neighbor_candidates = [
                        (self._distance(neighbor_vec, self.vectors[conn]), conn)
                        for conn in self.graph[lc][neighbor]
                    ]
                    pruned = self._get_neighbors(neighbor_candidates, max_conn)
                    self.graph[lc][neighbor] = set(pruned)

            nearest = neighbors

        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = node_id

        self.insert_count += 1

    def search(self, query: EnhancedVector, top_k: int) -> List[Tuple[float, str]]:
        """Search for k nearest neighbors."""
        if self.entry_point is None:
            return []

        start_time = time.time()

        nearest = [self.entry_point]

        for lc in range(self.max_layer, 0, -1):
            nearest = self._search_layer(query, set(nearest), 1, lc)
            nearest = [node_id for _, node_id in nearest]

        candidates = self._search_layer(query, set(nearest), max(self.ef_search, top_k), 0)
        candidates.sort(key=lambda x: x[0])

        self.search_count += 1
        self.total_search_time += (time.time() - start_time)

        return candidates[:top_k]

    def delete(self, vector_id: str) -> bool:
        """Delete vector from index."""
        if vector_id not in self.vectors:
            return False

        level = self.node_levels[vector_id]

        for lc in range(level + 1):
            if lc in self.graph and vector_id in self.graph[lc]:
                for neighbor in self.graph[lc][vector_id]:
                    if neighbor in self.graph[lc]:
                        self.graph[lc][neighbor].discard(vector_id)
                del self.graph[lc][vector_id]

        del self.vectors[vector_id]
        del self.node_levels[vector_id]

        if vector_id == self.entry_point:
            if self.vectors:
                self.entry_point = max(self.node_levels.keys(), key=lambda k: self.node_levels[k])
                self.max_layer = self.node_levels[self.entry_point]
            else:
                self.entry_point = None
                self.max_layer = -1

        return True

    def set_ef_search(self, ef_search: int):
        """Update ef_search parameter."""
        self.ef_search = ef_search

    def get_index_stats(self) -> Dict:
        """Get detailed index statistics."""
        if not self.vectors:
            return {
                'type': 'HNSWIndex',
                'dimension': self.dimension,
                'metric': self.metric.value,
                'num_vectors': 0
            }

        # Calculate graph statistics
        layer_stats = {}
        for layer in sorted(self.graph.keys(), reverse=True):
            nodes = self.graph[layer]
            total_connections = sum(len(connections) for connections in nodes.values())
            avg_connections = total_connections / len(nodes) if nodes else 0

            layer_stats[f'layer_{layer}'] = {
                'nodes': len(nodes),
                'total_connections': total_connections,
                'avg_connections': round(avg_connections, 2)
            }

        avg_search_time = (self.total_search_time / self.search_count * 1000) if self.search_count > 0 else 0

        return {
            'type': 'HNSWIndex',
            'dimension': self.dimension,
            'metric': self.metric.value,
            'num_vectors': len(self.vectors),
            'parameters': {
                'M': self.M,
                'M0': self.M0,
                'ef_construction': self.ef_construction,
                'ef_search': self.ef_search
            },
            'graph': {
                'max_layer': self.max_layer,
                'num_layers': len(self.graph),
                'entry_point': self.entry_point,
                'layers': layer_stats
            },
            'performance': {
                'total_inserts': self.insert_count,
                'total_searches': self.search_count,
                'avg_search_time_ms': round(avg_search_time, 3)
            },
            'search_complexity': 'O(log n)',
            'recall': '~95-99%'
        }


class IVFIndex(VectorIndex):
    """
    IVF (Inverted File) index for large-scale vector search.

    Partitions vectors into clusters and searches only relevant clusters.
    Offers good balance between speed and accuracy for large datasets.
    """

    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
        num_clusters: int = 100,
        nprobe: int = 10
    ):
        """
        Initialize IVF index.

        Args:
            dimension: Vector dimension
            metric: Distance metric
            num_clusters: Number of clusters (inverted lists)
            nprobe: Number of clusters to search
        """
        super().__init__(dimension, metric)
        self.num_clusters = num_clusters
        self.nprobe = nprobe
        self.centroids: Optional[np.ndarray] = None
        self.inverted_lists: Dict[int, List[str]] = {i: [] for i in range(num_clusters)}
        self.is_trained = False
        self.training_size = 0

    def train(self, training_vectors: List[EnhancedVector]):
        """
        Train the index by computing cluster centroids.

        Args:
            training_vectors: Vectors to use for training
        """
        if len(training_vectors) < self.num_clusters:
            raise ValueError(f"Need at least {self.num_clusters} vectors for training")

        # Extract data
        data = np.vstack([v.data for v in training_vectors])

        # Simple k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        kmeans.fit(data)

        self.centroids = kmeans.cluster_centers_
        self.is_trained = True
        self.training_size = len(training_vectors)

    def _assign_to_cluster(self, vector: EnhancedVector) -> int:
        """Assign vector to nearest cluster."""
        if not self.is_trained:
            raise RuntimeError("Index must be trained before inserting vectors")

        # Find nearest centroid
        distances = np.linalg.norm(self.centroids - vector.data, axis=1)
        return int(np.argmin(distances))

    def insert(self, vector: EnhancedVector):
        """Insert vector into index."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match index dimension {self.dimension}")

        if not self.is_trained:
            raise RuntimeError("Index must be trained before inserting vectors")

        cluster_id = self._assign_to_cluster(vector)
        self.vectors[vector.id] = vector
        self.inverted_lists[cluster_id].append(vector.id)

    def search(self, query: EnhancedVector, top_k: int) -> List[Tuple[float, str]]:
        """Search for k nearest neighbors."""
        if not self.is_trained or not self.vectors:
            return []

        # Find nprobe nearest centroids
        centroid_distances = np.linalg.norm(self.centroids - query.data, axis=1)
        nearest_clusters = np.argpartition(centroid_distances, self.nprobe)[:self.nprobe]

        # Search in selected clusters
        candidates = []
        for cluster_id in nearest_clusters:
            for vector_id in self.inverted_lists[cluster_id]:
                if vector_id in self.vectors:
                    dist = self.vectors[vector_id].distance(query, self.metric)
                    candidates.append((dist, vector_id))

        # Return top-k
        candidates.sort(key=lambda x: x[0])
        return candidates[:top_k]

    def delete(self, vector_id: str) -> bool:
        """Delete vector from index."""
        if vector_id not in self.vectors:
            return False

        # Find and remove from inverted list
        for cluster_id, vec_list in self.inverted_lists.items():
            if vector_id in vec_list:
                vec_list.remove(vector_id)
                break

        del self.vectors[vector_id]
        return True

    def get_index_stats(self) -> Dict:
        """Get index statistics."""
        cluster_sizes = {f'cluster_{i}': len(vecs) for i, vecs in self.inverted_lists.items() if vecs}
        avg_cluster_size = sum(cluster_sizes.values()) / len(cluster_sizes) if cluster_sizes else 0

        return {
            'type': 'IVFIndex',
            'dimension': self.dimension,
            'metric': self.metric.value,
            'num_vectors': len(self.vectors),
            'parameters': {
                'num_clusters': self.num_clusters,
                'nprobe': self.nprobe
            },
            'training': {
                'is_trained': self.is_trained,
                'training_size': self.training_size
            },
            'clusters': {
                'non_empty_clusters': len(cluster_sizes),
                'avg_cluster_size': round(avg_cluster_size, 2),
                'max_cluster_size': max(cluster_sizes.values()) if cluster_sizes else 0,
                'min_cluster_size': min(cluster_sizes.values()) if cluster_sizes else 0
            },
            'search_complexity': f'O(nprobe * n/num_clusters)',
            'recall': f'~{min(100, self.nprobe * 100 // self.num_clusters)}%'
        }

class EnhancedVectorStore:
    """
    Enhanced vector store with multiple index type support.

    Features:
    - Multiple index types (Flat, HNSW, IVF)
    - Rich vector metadata
    - Advanced search capabilities
    - Batch operations
    - Statistics and monitoring
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "hnsw",
        metric: Union[str, DistanceMetric] = "euclidean",
        index_params: Optional[Dict] = None
    ):
        """
        Initialize enhanced vector store.

        Args:
            dimension: Vector dimension
            index_type: Type of index ("flat", "hnsw", "ivf")
            metric: Distance metric
            index_params: Index-specific parameters
        """
        self.dimension = dimension

        # Convert metric string to enum
        if isinstance(metric, str):
            metric = DistanceMetric(metric.lower())
        self.metric = metric

        # Create index
        index_params = index_params or {}
        if index_type.lower() == "flat":
            self.index = FlatIndex(dimension, metric)
        elif index_type.lower() == "hnsw":
            self.index = HNSWIndex(
                dimension,
                metric,
                M=index_params.get('M', 16),
                ef_construction=index_params.get('ef_construction', 200),
                ef_search=index_params.get('ef_search', 50)
            )
        elif index_type.lower() == "ivf":
            self.index = IVFIndex(
                dimension,
                metric,
                num_clusters=index_params.get('num_clusters', 100),
                nprobe=index_params.get('nprobe', 10)
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.index_type = index_type.lower()
        self._next_id = 0

    def _generate_id(self) -> str:
        """Generate unique ID."""
        new_id = f"vec_{self._next_id}"
        self._next_id += 1
        return new_id

    def insert(
        self,
        data: Union[List[float], np.ndarray],
        vector_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        normalize: bool = False
    ) -> str:
        """
        Insert a vector.

        Args:
            data: Vector data
            vector_id: Optional ID
            metadata: Optional metadata
            tags: Optional tags
            normalize: Whether to normalize

        Returns:
            Vector ID
        """
        if vector_id is None:
            vector_id = self._generate_id()

        if vector_id in self.index.vectors:
            raise ValueError(f"Vector {vector_id} already exists")

        vector = EnhancedVector(data, vector_id, metadata, tags, normalize)
        self.index.insert(vector)
        return vector_id

    def batch_insert(
        self,
        vectors: List[Tuple],
        show_progress: bool = False
    ) -> List[str]:
        """
        Batch insert vectors.

        Args:
            vectors: List of (data, vector_id, metadata, tags, normalize) tuples
            show_progress: Show progress

        Returns:
            List of inserted IDs
        """
        inserted_ids = []
        total = len(vectors)

        for idx, vec_tuple in enumerate(vectors):
            # Unpack with defaults
            data = vec_tuple[0]
            vector_id = vec_tuple[1] if len(vec_tuple) > 1 else None
            metadata = vec_tuple[2] if len(vec_tuple) > 2 else None
            tags = vec_tuple[3] if len(vec_tuple) > 3 else None
            normalize = vec_tuple[4] if len(vec_tuple) > 4 else False

            vec_id = self.insert(data, vector_id, metadata, tags, normalize)
            inserted_ids.append(vec_id)

            if show_progress and (idx + 1) % 1000 == 0:
                print(f"Inserted {idx + 1}/{total} vectors...")

        if show_progress and total >= 1000:
            print(f"Batch insert complete: {total} vectors")

        return inserted_ids

    def update(
        self,
        vector_id: str,
        data: Optional[Union[List[float], np.ndarray]] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
        normalize: bool = False
    ) -> bool:
        """
        Update a vector.

        Args:
            vector_id: Vector ID
            data: New data (None to keep existing)
            metadata: New metadata (None to keep existing)
            tags: Replace tags (None to keep existing)
            add_tags: Tags to add
            remove_tags: Tags to remove
            normalize: Normalize new data

        Returns:
            True if updated
        """
        vector = self.index.get(vector_id)
        if vector is None:
            return False

        # Update data
        if data is not None:
            vector.update_data(data, normalize)
            # Re-index
            self.index.delete(vector_id)
            self.index.insert(vector)

        # Update metadata
        if metadata is not None:
            vector.update_metadata(metadata)

        # Update tags
        if tags is not None:
            vector.tags = tags
            vector._update_timestamp()

        if add_tags:
            for tag in add_tags:
                vector.add_tag(tag)

        if remove_tags:
            for tag in remove_tags:
                vector.remove_tag(tag)

        return True

    def batch_update(
        self,
        updates: List[Tuple],
        show_progress: bool = False
    ) -> int:
        """
        Batch update vectors.

        Args:
            updates: List of (vector_id, data, metadata, ...) tuples
            show_progress: Show progress

        Returns:
            Number of updated vectors
        """
        updated_count = 0
        total = len(updates)

        for idx, update_tuple in enumerate(updates):
            vector_id = update_tuple[0]
            data = update_tuple[1] if len(update_tuple) > 1 else None
            metadata = update_tuple[2] if len(update_tuple) > 2 else None

            if self.update(vector_id, data, metadata):
                updated_count += 1

            if show_progress and (idx + 1) % 1000 == 0:
                print(f"Updated {idx + 1}/{total} vectors...")

        if show_progress and total >= 1000:
            print(f"Batch update complete: {updated_count}/{total} vectors")

        return updated_count

    def search(
        self,
        query: Union[List[float], np.ndarray, EnhancedVector],
        top_k: int = 5,
        filter_func: Optional[callable] = None
    ) -> List[Tuple[str, float, EnhancedVector]]:
        """
        Search for similar vectors.

        Args:
            query: Query vector
            top_k: Number of results
            filter_func: Optional filter function (takes EnhancedVector, returns bool)

        Returns:
            List of (vector_id, distance, vector) tuples
        """
        if isinstance(query, (list, np.ndarray)):
            query = EnhancedVector(query, validate=False)

        results = self.index.search(query, top_k * 10 if filter_func else top_k)

        # Apply filter if provided
        if filter_func:
            filtered = []
            for dist, vec_id in results:
                vector = self.index.vectors[vec_id]
                if filter_func(vector):
                    filtered.append((vec_id, dist, vector))
                if len(filtered) >= top_k:
                    break
            return filtered
        else:
            return [(vec_id, dist, self.index.vectors[vec_id]) for dist, vec_id in results]

    def search_by_tags(
        self,
        query: Union[List[float], np.ndarray],
        tags: List[str],
        top_k: int = 5,
        match_all: bool = False
    ) -> List[Tuple[str, float, EnhancedVector]]:
        """
        Search with tag filtering.

        Args:
            query: Query vector
            tags: Tags to filter by
            top_k: Number of results
            match_all: If True, vector must have all tags; if False, any tag

        Returns:
            List of (vector_id, distance, vector) tuples
        """
        if match_all:
            filter_func = lambda v: all(tag in v.tags for tag in tags)
        else:
            filter_func = lambda v: any(tag in v.tags for tag in tags)

        return self.search(query, top_k, filter_func)

    def search_by_metadata(
        self,
        query: Union[List[float], np.ndarray],
        metadata_filter: Dict[str, Any],
        top_k: int = 5
    ) -> List[Tuple[str, float, EnhancedVector]]:
        """
        Search with metadata filtering.

        Args:
            query: Query vector
            metadata_filter: Metadata key-value pairs to match
            top_k: Number of results

        Returns:
            List of (vector_id, distance, vector) tuples
        """
        filter_func = lambda v: all(
            v.metadata.get(k) == val for k, val in metadata_filter.items()
        )

        return self.search(query, top_k, filter_func)

    def get(self, vector_id: str) -> Optional[EnhancedVector]:
        """Get vector by ID."""
        return self.index.get(vector_id)

    def delete(self, vector_id: str) -> bool:
        """Delete vector."""
        return self.index.delete(vector_id)

    def size(self) -> int:
        """Get number of vectors."""
        return self.index.size()

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        index_stats = self.index.get_index_stats()

        # Memory usage
        import sys
        vector_memory = sum(v.data.nbytes for v in self.index.vectors.values())
        metadata_memory = sum(
            sys.getsizeof(str(v.metadata)) + sys.getsizeof(str(v.tags))
            for v in self.index.vectors.values()
        )

        # Tag statistics
        all_tags = set()
        for v in self.index.vectors.values():
            all_tags.update(v.tags)

        return {
            **index_stats,
            'memory': {
                'vector_data_mb': vector_memory / (1024 * 1024),
                'metadata_mb': metadata_memory / (1024 * 1024),
                'total_mb': (vector_memory + metadata_memory) / (1024 * 1024)
            },
            'vectors': {
                'total': len(self.index.vectors),
                'unique_tags': len(all_tags),
                'avg_tags_per_vector': sum(len(v.tags) for v in self.index.vectors.values()) / max(1, len(self.index.vectors))
            }
        }

    def save(self, filepath: str):
        """Save to JSON file."""
        data = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric.value,
            'next_id': self._next_id,
            'vectors': [v.to_dict() for v in self.index.vectors.values()]
        }

        # Add index-specific data
        if self.index_type == 'hnsw':
            graph_data = {}
            for layer, nodes in self.index.graph.items():
                graph_data[str(layer)] = {
                    node_id: list(connections)
                    for node_id, connections in nodes.items()
                }

            data['hnsw_data'] = {
                'M': self.index.M,
                'M0': self.index.M0,
                'ef_construction': self.index.ef_construction,
                'ef_search': self.index.ef_search,
                'entry_point': self.index.entry_point,
                'max_layer': self.index.max_layer,
                'node_levels': self.index.node_levels,
                'graph': graph_data
            }
        elif self.index_type == 'ivf':
            data['ivf_data'] = {
                'num_clusters': self.index.num_clusters,
                'nprobe': self.index.nprobe,
                'is_trained': self.index.is_trained,
                'centroids': self.index.centroids.tolist() if self.index.centroids is not None else None,
                'inverted_lists': {str(k): v for k, v in self.index.inverted_lists.items()}
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'EnhancedVectorStore':
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Create store
        index_params = {}
        if data['index_type'] == 'hnsw' and 'hnsw_data' in data:
            hnsw_data = data['hnsw_data']
            index_params = {
                'M': hnsw_data.get('M', 16),
                'ef_construction': hnsw_data.get('ef_construction', 200),
                'ef_search': hnsw_data.get('ef_search', 50)
            }
        elif data['index_type'] == 'ivf' and 'ivf_data' in data:
            ivf_data = data['ivf_data']
            index_params = {
                'num_clusters': ivf_data.get('num_clusters', 100),
                'nprobe': ivf_data.get('nprobe', 10)
            }

        store = cls(
            data['dimension'],
            data['index_type'],
            data['metric'],
            index_params
        )
        store._next_id = data.get('next_id', 0)

        # Restore vectors
        for vec_data in data['vectors']:
            vector = EnhancedVector.from_dict(vec_data)
            store.index.vectors[vector.id] = vector

        # Restore index-specific data
        if data['index_type'] == 'hnsw' and 'hnsw_data' in data:
            hnsw_data = data['hnsw_data']
            store.index.entry_point = hnsw_data.get('entry_point')
            store.index.max_layer = hnsw_data.get('max_layer', -1)
            store.index.node_levels = hnsw_data.get('node_levels', {})

            if 'graph' in hnsw_data:
                for layer_str, nodes in hnsw_data['graph'].items():
                    layer = int(layer_str)
                    store.index.graph[layer] = {
                        node_id: set(connections)
                        for node_id, connections in nodes.items()
                    }
        elif data['index_type'] == 'ivf' and 'ivf_data' in data:
            ivf_data = data['ivf_data']
            store.index.is_trained = ivf_data.get('is_trained', False)
            if ivf_data.get('centroids'):
                store.index.centroids = np.array(ivf_data['centroids'], dtype=np.float32)
            if ivf_data.get('inverted_lists'):
                store.index.inverted_lists = {
                    int(k): v for k, v in ivf_data['inverted_lists'].items()
                }

        return store

    def __repr__(self):
        return f"EnhancedVectorStore(dimension={self.dimension}, index={self.index_type}, size={self.size()}, metric={self.metric.value})"
