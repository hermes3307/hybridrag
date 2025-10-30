#!/usr/bin/env python3
"""
Advanced Search Engine for Face Database

Provides enhanced search capabilities:
1. Multiple value filters (OR logic)
2. Range queries (age, brightness, etc.)
3. Text-based natural language search
4. Demographic-only search (no image needed)
5. Complex boolean queries
6. Saved search templates
7. Search result export
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from pathlib import Path
import csv


@dataclass
class SearchQuery:
    """Advanced search query structure"""
    # Vector search (optional)
    query_image: Optional[str] = None

    # Demographic filters
    sex: List[str] = None  # Can be multiple: ['male', 'female']
    age_groups: List[str] = None  # ['young_adult', 'adult']
    age_range: Tuple[int, int] = None  # (25, 35) for age 25-35
    skin_tones: List[str] = None
    skin_colors: List[str] = None
    hair_colors: List[str] = None

    # Image property filters
    brightness_range: Tuple[float, float] = None  # (0, 255)
    contrast_range: Tuple[float, float] = None
    has_face: Optional[bool] = None
    min_image_quality: Optional[str] = None  # 'high', 'medium'

    # System filters
    embedding_models: List[str] = None

    # Query options
    n_results: int = 10
    search_mode: str = 'hybrid'  # 'vector', 'metadata', 'hybrid'
    sort_by: Optional[str] = None  # 'distance', 'timestamp', etc.


class AdvancedSearchEngine:
    """
    Advanced search engine with enhanced filtering capabilities
    """

    def __init__(self, system):
        """Initialize with IntegratedFaceSystem instance"""
        self.system = system
        self.saved_queries = {}
        self._load_saved_queries()

    def search(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """
        Execute advanced search with complex filters
        """
        # Build ChromaDB where clause for metadata filters
        where_clause = self._build_where_clause(query)

        results = []

        if query.search_mode == 'metadata':
            # Metadata-only search
            results = self._search_metadata(where_clause, query.n_results)

        elif query.search_mode == 'vector' and query.query_image:
            # Vector similarity search
            results = self._search_vector(query.query_image, where_clause, query.n_results)

        elif query.search_mode == 'hybrid' and query.query_image:
            # Hybrid search (vector + metadata)
            results = self._search_hybrid(query.query_image, where_clause, query.n_results)

        else:
            raise ValueError("Invalid search mode or missing query image")

        # Apply post-filtering for range queries (ChromaDB doesn't support range natively)
        results = self._apply_range_filters(results, query)

        # Sort results if requested
        if query.sort_by:
            results = self._sort_results(results, query.sort_by)

        return results

    def _convert_where_to_pgvector(self, where_clause: Optional[Dict]) -> Dict[str, Any]:
        """
        Convert ChromaDB where clause to pgvector metadata filter

        Note: pgvector has limited support for complex queries.
        This method handles basic conversions and simplifies complex ones.
        """
        if not where_clause:
            return {}

        # Handle simple single condition
        if not isinstance(where_clause, dict):
            return {}

        metadata_filter = {}

        # Handle $and operator
        if '$and' in where_clause:
            conditions = where_clause['$and']
            for condition in conditions:
                metadata_filter.update(self._convert_single_condition(condition))
            return metadata_filter

        # Handle single condition
        return self._convert_single_condition(where_clause)

    def _convert_single_condition(self, condition: Dict) -> Dict[str, Any]:
        """Convert a single condition from ChromaDB to pgvector format"""
        converted = {}

        for key, value in condition.items():
            if isinstance(value, dict):
                # Handle operators
                if '$in' in value:
                    # pgvector doesn't support $in directly
                    # Use the first value as a workaround
                    # For proper OR support, queries should be run separately
                    values = value['$in']
                    if values:
                        converted[key] = values[0]
                elif '$ne' in value:
                    # Skip $ne for now (not supported in basic pgvector search)
                    pass
                else:
                    # Pass through operators like $gt, $lt, etc.
                    converted[key] = value
            else:
                # Direct value
                converted[key] = value

        return converted

    def _build_where_clause(self, query: SearchQuery) -> Dict[str, Any]:
        """
        Build ChromaDB where clause from query

        ChromaDB supports:
        - Exact match: {'key': 'value'}
        - OR logic: {'key': {'$in': ['val1', 'val2']}}
        - AND logic: {'$and': [{...}, {...}]}
        - NOT logic: {'key': {'$ne': 'value'}}
        """
        conditions = []

        # Sex filter (OR logic for multiple values)
        if query.sex:
            if len(query.sex) == 1:
                conditions.append({'estimated_sex': query.sex[0]})
            else:
                conditions.append({'estimated_sex': {'$in': query.sex}})

        # Age group filter (OR logic)
        if query.age_groups:
            if len(query.age_groups) == 1:
                conditions.append({'age_group': query.age_groups[0]})
            else:
                conditions.append({'age_group': {'$in': query.age_groups}})

        # Skin tone filter (OR logic)
        if query.skin_tones:
            if len(query.skin_tones) == 1:
                conditions.append({'skin_tone': query.skin_tones[0]})
            else:
                conditions.append({'skin_tone': {'$in': query.skin_tones}})

        # Skin color filter (OR logic)
        if query.skin_colors:
            if len(query.skin_colors) == 1:
                conditions.append({'skin_color': query.skin_colors[0]})
            else:
                conditions.append({'skin_color': {'$in': query.skin_colors}})

        # Hair color filter (OR logic)
        if query.hair_colors:
            if len(query.hair_colors) == 1:
                conditions.append({'hair_color': query.hair_colors[0]})
            else:
                conditions.append({'hair_color': {'$in': query.hair_colors}})

        # Has face filter
        if query.has_face is not None:
            conditions.append({'has_face': query.has_face})

        # Image quality filter
        if query.min_image_quality:
            if query.min_image_quality == 'high':
                conditions.append({'image_quality': 'high'})
            # 'medium' includes both high and medium

        # Embedding model filter
        if query.embedding_models:
            if len(query.embedding_models) == 1:
                conditions.append({'embedding_model': query.embedding_models[0]})
            else:
                conditions.append({'embedding_model': {'$in': query.embedding_models}})

        # Combine all conditions with AND logic
        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {'$and': conditions}

    def _apply_range_filters(self, results: List[Dict[str, Any]], query: SearchQuery) -> List[Dict[str, Any]]:
        """
        Apply range filters that ChromaDB doesn't support natively
        """
        filtered = []

        for result in results:
            metadata = result.get('metadata', {})

            # Age range filter
            if query.age_range:
                age_str = metadata.get('estimated_age', '')
                if not self._age_in_range(age_str, query.age_range):
                    continue

            # Brightness range filter
            if query.brightness_range:
                brightness = metadata.get('brightness', -1)
                if brightness < query.brightness_range[0] or brightness > query.brightness_range[1]:
                    continue

            # Contrast range filter
            if query.contrast_range:
                contrast = metadata.get('contrast', -1)
                if contrast < query.contrast_range[0] or contrast > query.contrast_range[1]:
                    continue

            filtered.append(result)

        return filtered

    def _age_in_range(self, age_str: str, age_range: Tuple[int, int]) -> bool:
        """Check if age string is within range"""
        # age_str is like '25-40', '18-25', '60+', etc.
        if not age_str:
            return False

        min_age, max_age = age_range

        if '+' in age_str:
            # Senior: '60+'
            age_min = int(age_str.replace('+', ''))
            return age_min <= max_age
        elif '-' in age_str:
            # Range: '25-40'
            parts = age_str.split('-')
            age_min, age_max = int(parts[0]), int(parts[1])
            # Check if ranges overlap
            return not (age_max < min_age or age_min > max_age)

        return False

    def _search_metadata(self, where_clause: Optional[Dict], n_results: int) -> List[Dict[str, Any]]:
        """Metadata-only search"""
        # Check if using pgvector or ChromaDB
        if hasattr(self.system.db_manager, 'collection'):
            # ChromaDB backend
            if not where_clause:
                # Get all results if no filter
                results = self.system.db_manager.collection.get(limit=n_results)
            else:
                results = self.system.db_manager.collection.get(
                    where=where_clause,
                    limit=n_results
                )

            return [
                {
                    'id': results['ids'][i],
                    'metadata': results['metadatas'][i],
                    'distance': 0.0  # No distance for metadata-only
                }
                for i in range(len(results['ids']))
            ]
        else:
            # pgvector backend
            metadata_filter = self._convert_where_to_pgvector(where_clause) if where_clause else {}
            return self.system.db_manager.search_by_metadata(metadata_filter, n_results)

    def _search_vector(self, query_image: str, where_clause: Optional[Dict], n_results: int) -> List[Dict[str, Any]]:
        """Vector similarity search"""
        from core import FaceAnalyzer, FaceEmbedder

        # Create embedding for query image
        analyzer = FaceAnalyzer()
        embedder = FaceEmbedder(model_name=self.system.config.embedding_model)

        features = analyzer.analyze_face(query_image)
        embedding = embedder.create_embedding(query_image, features)

        # Convert where_clause for pgvector if needed
        if hasattr(self.system.db_manager, 'collection'):
            # ChromaDB backend
            return self.system.db_manager.search_faces(embedding, n_results, where_clause)
        else:
            # pgvector backend
            metadata_filter = self._convert_where_to_pgvector(where_clause) if where_clause else None
            return self.system.db_manager.search_faces(embedding, n_results, metadata_filter)

    def _search_hybrid(self, query_image: str, where_clause: Optional[Dict], n_results: int) -> List[Dict[str, Any]]:
        """Hybrid search (vector + metadata)"""
        return self._search_vector(query_image, where_clause, n_results)

    def _sort_results(self, results: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
        """Sort results by specified field"""
        if sort_by == 'distance':
            return sorted(results, key=lambda x: x.get('distance', 0))
        elif sort_by == 'timestamp':
            return sorted(results, key=lambda x: x.get('metadata', {}).get('timestamp', ''), reverse=True)
        elif sort_by == 'brightness':
            return sorted(results, key=lambda x: x.get('metadata', {}).get('brightness', 0), reverse=True)
        return results

    def text_search(self, text_query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Natural language text search

        Examples:
        - "blonde female"
        - "young adult male with dark hair"
        - "bright images of females"
        - "seniors with gray hair"
        """
        query = self._parse_text_query(text_query)
        query.n_results = n_results
        return self.search(query)

    def _parse_text_query(self, text: str) -> SearchQuery:
        """Parse natural language query into SearchQuery"""
        text = text.lower()
        query = SearchQuery()
        query.search_mode = 'metadata'

        # Sex detection
        if 'male' in text and 'female' not in text:
            query.sex = ['male']
        elif 'female' in text:
            query.sex = ['female']

        # Age group detection
        age_keywords = {
            'child': ['child', 'kid', 'young child'],
            'young_adult': ['young adult', 'young', 'teenager', 'teen'],
            'adult': ['adult'],
            'middle_aged': ['middle aged', 'middle-aged', 'middle age'],
            'senior': ['senior', 'elderly', 'old']
        }

        for age_group, keywords in age_keywords.items():
            if any(kw in text for kw in keywords):
                if not query.age_groups:
                    query.age_groups = []
                query.age_groups.append(age_group)

        # Hair color detection
        hair_colors = ['blonde', 'black', 'brown', 'red', 'gray', 'grey']
        for color in hair_colors:
            if color in text:
                if not query.hair_colors:
                    query.hair_colors = []
                if color == 'grey':
                    color = 'gray'
                query.hair_colors.append(color)

        # Skin tone detection
        skin_keywords = {
            'light': ['light skin', 'pale', 'fair'],
            'dark': ['dark skin', 'dark complexion'],
            'medium': ['medium skin', 'tan']
        }

        for tone, keywords in skin_keywords.items():
            if any(kw in text for kw in keywords):
                if not query.skin_colors:
                    query.skin_colors = []
                query.skin_colors.append(tone)

        # Brightness detection
        if 'bright' in text or 'well-lit' in text:
            query.brightness_range = (150, 255)
        elif 'dark' in text and 'hair' not in text and 'skin' not in text:
            query.brightness_range = (0, 150)

        # Quality detection
        if 'high quality' in text or 'good quality' in text:
            query.min_image_quality = 'high'

        return query

    def save_query(self, name: str, query: SearchQuery):
        """Save a query for later reuse"""
        self.saved_queries[name] = query
        self._persist_saved_queries()

    def load_query(self, name: str) -> Optional[SearchQuery]:
        """Load a saved query"""
        return self.saved_queries.get(name)

    def list_saved_queries(self) -> List[str]:
        """List all saved query names"""
        return list(self.saved_queries.keys())

    def _persist_saved_queries(self):
        """Save queries to disk"""
        queries_file = Path("saved_searches.json")
        with open(queries_file, 'w') as f:
            # Convert SearchQuery objects to dicts
            data = {name: query.__dict__ for name, query in self.saved_queries.items()}
            json.dump(data, f, indent=2)

    def _load_saved_queries(self):
        """Load queries from disk"""
        queries_file = Path("saved_searches.json")
        if queries_file.exists():
            with open(queries_file, 'r') as f:
                data = json.load(f)
                self.saved_queries = {name: SearchQuery(**query_dict)
                                     for name, query_dict in data.items()}

    def export_results(self, results: List[Dict[str, Any]], output_file: str, format: str = 'json'):
        """
        Export search results

        Formats: 'json', 'csv', 'txt'
        """
        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

        elif format == 'csv':
            if not results:
                return

            with open(output_file, 'w', newline='') as f:
                # Get all metadata keys
                all_keys = set()
                for r in results:
                    all_keys.update(r.get('metadata', {}).keys())

                fieldnames = ['id', 'distance'] + sorted(all_keys)
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                writer.writeheader()
                for result in results:
                    row = {'id': result.get('id'), 'distance': result.get('distance', 0)}
                    row.update(result.get('metadata', {}))
                    writer.writerow(row)

        elif format == 'txt':
            with open(output_file, 'w') as f:
                for i, result in enumerate(results, 1):
                    f.write(f"Result {i}:\n")
                    f.write(f"  ID: {result.get('id')}\n")
                    f.write(f"  Distance: {result.get('distance', 0):.4f}\n")
                    metadata = result.get('metadata', {})
                    f.write(f"  File: {metadata.get('file_path', 'N/A')}\n")
                    f.write(f"  Sex: {metadata.get('estimated_sex', 'N/A')}\n")
                    f.write(f"  Age: {metadata.get('estimated_age', 'N/A')}\n")
                    f.write(f"  Hair: {metadata.get('hair_color', 'N/A')}\n")
                    f.write(f"  Skin: {metadata.get('skin_tone', 'N/A')}\n")
                    f.write("\n")

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics by demographics"""
        # Check if using pgvector or ChromaDB
        if hasattr(self.system.db_manager, 'collection'):
            # ChromaDB backend
            all_records = self.system.db_manager.collection.get(limit=100000)

            if not all_records or not all_records.get('metadatas'):
                return {}

            stats = {
                'total_count': len(all_records['metadatas']),
                'sex_distribution': {},
                'age_distribution': {},
                'skin_tone_distribution': {},
                'hair_color_distribution': {}
            }

            for metadata in all_records['metadatas']:
                # Sex
                sex = metadata.get('estimated_sex', 'unknown')
                stats['sex_distribution'][sex] = stats['sex_distribution'].get(sex, 0) + 1

                # Age
                age = metadata.get('age_group', 'unknown')
                stats['age_distribution'][age] = stats['age_distribution'].get(age, 0) + 1

                # Skin
                skin = metadata.get('skin_tone', 'unknown')
                stats['skin_tone_distribution'][skin] = stats['skin_tone_distribution'].get(skin, 0) + 1

                # Hair
                hair = metadata.get('hair_color', 'unknown')
                stats['hair_color_distribution'][hair] = stats['hair_color_distribution'].get(hair, 0) + 1

            return stats
        else:
            # pgvector backend - get all records
            all_records = self.system.db_manager.search_by_metadata({}, n_results=100000)

            if not all_records:
                return {}

            stats = {
                'total_count': len(all_records),
                'sex_distribution': {},
                'age_distribution': {},
                'skin_tone_distribution': {},
                'hair_color_distribution': {}
            }

            for record in all_records:
                metadata = record.get('metadata', {})

                # Sex
                sex = metadata.get('estimated_sex', 'unknown')
                stats['sex_distribution'][sex] = stats['sex_distribution'].get(sex, 0) + 1

                # Age
                age = metadata.get('age_group', 'unknown')
                stats['age_distribution'][age] = stats['age_distribution'].get(age, 0) + 1

                # Skin
                skin = metadata.get('skin_tone', 'unknown')
                stats['skin_tone_distribution'][skin] = stats['skin_tone_distribution'].get(skin, 0) + 1

                # Hair
                hair = metadata.get('hair_color', 'unknown')
                stats['hair_color_distribution'][hair] = stats['hair_color_distribution'].get(hair, 0) + 1

            return stats


def main():
    """Example usage"""
    from core import IntegratedFaceSystem

    # Initialize system
    system = IntegratedFaceSystem()
    if not system.initialize():
        print("Failed to initialize system")
        return

    # Create advanced search engine
    search_engine = AdvancedSearchEngine(system)

    # Example 1: Search for blonde females
    print("Example 1: Search for blonde females")
    query = SearchQuery(
        sex=['female'],
        hair_colors=['blonde'],
        n_results=5,
        search_mode='metadata'
    )
    results = search_engine.search(query)
    print(f"Found {len(results)} results")

    # Example 2: Text-based search
    print("\nExample 2: Text search - 'young adult males with dark hair'")
    results = search_engine.text_search("young adult males with dark hair", n_results=5)
    print(f"Found {len(results)} results")

    # Example 3: Complex query with multiple filters
    print("\nExample 3: Complex query")
    query = SearchQuery(
        sex=['male', 'female'],  # OR logic: male OR female
        age_groups=['young_adult', 'adult'],  # young_adult OR adult
        hair_colors=['brown', 'black'],  # brown OR black hair
        brightness_range=(100, 200),  # medium brightness
        has_face=True,
        n_results=10,
        search_mode='metadata'
    )
    results = search_engine.search(query)
    print(f"Found {len(results)} results")

    # Example 4: Save query for later
    search_engine.save_query("blonde_females", SearchQuery(
        sex=['female'],
        hair_colors=['blonde'],
        n_results=10,
        search_mode='metadata'
    ))

    # Example 5: Export results
    if results:
        search_engine.export_results(results, "search_results.json", format='json')
        search_engine.export_results(results, "search_results.csv", format='csv')
        print("\nResults exported to search_results.json and search_results.csv")

    # Show statistics
    stats = search_engine.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"Total records: {stats.get('total_count', 0)}")


if __name__ == "__main__":
    main()
