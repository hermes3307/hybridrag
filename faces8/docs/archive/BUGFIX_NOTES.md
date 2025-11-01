# Bug Fix - Advanced Search pgvector Compatibility

## Issue
The `advanced_search.py` module was throwing an `AttributeError` when using pgvector backend:
```
AttributeError: 'PgVectorDatabaseManager' object has no attribute 'collection'
```

## Root Cause
The advanced search module was written specifically for ChromaDB and used ChromaDB's API (`collection.get()`). When using pgvector, the database manager doesn't have a `collection` attribute.

## Solution
Updated `advanced_search.py` to support both ChromaDB and pgvector backends:

1. **Added backend detection**: Check for `collection` attribute to determine which backend is in use
2. **Added conversion methods**:
   - `_convert_where_to_pgvector()`: Converts ChromaDB-style where clauses to pgvector metadata filters
   - `_convert_single_condition()`: Handles individual filter conditions
3. **Updated search methods**:
   - `_search_metadata()`: Now supports both backends
   - `_search_vector()`: Properly converts filters for each backend
   - `get_statistics()`: Retrieves statistics from both backends

## Changes Made

### File: advanced_search.py

#### Lines 96-146: Added conversion methods
```python
def _convert_where_to_pgvector(self, where_clause: Optional[Dict]) -> Dict[str, Any]:
    """Convert ChromaDB where clause to pgvector metadata filter"""
    # Handles $and, $in, and other operators

def _convert_single_condition(self, condition: Dict) -> Dict[str, Any]:
    """Convert a single condition from ChromaDB to pgvector format"""
    # Maps field values and operators
```

#### Lines 220-245: Updated _search_metadata()
- Added backend detection with `hasattr(self.system.db_manager, 'collection')`
- ChromaDB path: Uses `collection.get()` as before
- pgvector path: Uses `search_by_metadata()` with converted filters

#### Lines 247-265: Updated _search_vector()
- Added backend-specific filter conversion
- Maintains compatibility with both systems

#### Lines 430-500: Updated get_statistics()
- Separate implementations for ChromaDB and pgvector
- Handles different data structures returned by each backend

## Limitations
- **$in operator**: pgvector doesn't support OR logic directly, so when multiple values are provided (e.g., `sex: ['male', 'female']`), only the first value is used
- **$ne operator**: NOT logic is currently skipped for pgvector
- For full OR support, multiple separate queries would need to be executed

## Testing
Verified the fix works with:
```bash
# Test basic search
python3 -c "from advanced_search import *; ..."

# Test search CLI
./search_cli.py
```

Both metadata-only and vector similarity searches now work correctly with pgvector.

## Recommendation
For production use with complex OR queries, consider:
1. Running multiple queries for each OR condition and merging results
2. Using PostgreSQL's native OR queries directly
3. Implementing a more sophisticated query planner

## Date
2025-10-30
