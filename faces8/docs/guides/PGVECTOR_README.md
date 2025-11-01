# PostgreSQL + pgvector Implementation Guide

This document describes the pgvector implementation for the Face Recognition System.

## Overview

The system now supports both ChromaDB (legacy) and PostgreSQL + pgvector for vector storage. PostgreSQL + pgvector provides:

- **Better scalability** - Production-ready database system
- **Advanced querying** - Full SQL support with metadata filtering
- **Concurrent access** - Multiple users/processes can access simultaneously
- **ACID compliance** - Data integrity and reliability
- **Flexible metadata** - JSONB support for complex metadata queries

---

## Architecture

### Database Schema

```sql
-- Main faces table
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(255) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    image_hash VARCHAR(64) NOT NULL,
    embedding_model VARCHAR(50) NOT NULL,
    embedding vector(512),  -- pgvector type

    -- Metadata columns
    age_estimate INTEGER,
    gender VARCHAR(20),
    brightness FLOAT,
    contrast FLOAT,
    sharpness FLOAT,

    -- Flexible JSONB metadata
    metadata JSONB,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Vector Indexes

The system uses **HNSW (Hierarchical Navigable Small World)** index for fast approximate nearest neighbor search:

```sql
CREATE INDEX idx_embedding_hnsw_cosine ON faces
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Distance Metrics

- **Cosine distance** (`<=>`) - Recommended for face embeddings
- **L2 distance** (`<->`) - Euclidean distance
- **Inner product** (`<#>`) - Negative inner product

---

## Configuration

### Environment Variables (.env)

```bash
# PostgreSQL Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Application Settings
DB_TYPE=pgvector
EMBEDDING_MODEL=statistical
FACES_DIR=./faces

# Connection Pool Settings
DB_MIN_CONNECTIONS=1
DB_MAX_CONNECTIONS=10

# Vector Search Settings
DEFAULT_SEARCH_LIMIT=10
DISTANCE_METRIC=cosine
```

### System Configuration (core.py)

```python
config = SystemConfig()
config.db_type = "pgvector"  # or "chromadb"
config.db_host = "localhost"
config.db_port = 5432
config.db_name = "vector_db"
config.db_user = "postgres"
config.db_password = "postgres"
```

---

## Installation & Setup

### 1. Install PostgreSQL and pgvector

```bash
# Update package list
sudo apt-get update

# Install PostgreSQL
sudo apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all

# Install build dependencies
sudo apt-get install -y build-essential git

# Clone and install pgvector
cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 2. Start PostgreSQL

```bash
sudo service postgresql start
sudo service postgresql status
```

### 3. Create Database and Enable pgvector

```bash
# Create database
sudo -u postgres psql -c "CREATE DATABASE vector_db;"

# Enable pgvector extension
sudo -u postgres psql -d vector_db -c "CREATE EXTENSION vector;"

# Set password (optional)
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
```

### 4. Create Schema

```bash
sudo -u postgres psql -d vector_db -f schema.sql
```

### 5. Install Python Dependencies

```bash
pip3 install --break-system-packages psycopg2-binary python-dotenv
```

### 6. Configure Environment

Edit `.env` file with your database credentials.

---

## Usage

### Initialize System with pgvector

```python
from core import IntegratedFaceSystem, SystemConfig

# Create configuration
config = SystemConfig()
config.db_type = "pgvector"

# Initialize system
system = IntegratedFaceSystem()
if system.initialize():
    print("System initialized with pgvector!")
```

### Add Faces

```python
# Add single face
face_data = FaceData(...)
system.db_manager.add_face(face_data, embedding_model="statistical")

# Add multiple faces (batch)
faces = [(face_data1, "statistical"), (face_data2, "statistical")]
count = system.db_manager.add_faces_batch(faces)
```

### Search Faces

```python
# Vector similarity search
results = system.db_manager.search_faces(
    query_embedding=embedding,
    n_results=10,
    distance_metric='cosine'
)

# Hybrid search (vector + metadata)
results = system.db_manager.search_faces(
    query_embedding=embedding,
    n_results=10,
    metadata_filter={'gender': 'female', 'age_estimate': {'$gt': 25}}
)

# Metadata-only search
results = system.db_manager.search_by_metadata(
    metadata_filter={'gender': 'male'},
    n_results=10
)
```

### Get Statistics

```python
# Database statistics
stats = system.db_manager.get_stats()
print(f"Total faces: {stats['total_faces']}")
print(f"Database size: {stats['database_size']}")

# Collection info
info = system.db_manager.get_collection_info()
print(f"Database: {info['name']}")
print(f"Count: {info['count']}")
```

---

## Migration from ChromaDB

### Preview Migration (Dry Run)

```bash
python3 migrate_to_pgvector.py --dry-run
```

### Perform Migration

```bash
# Default settings
python3 migrate_to_pgvector.py

# Custom batch size
python3 migrate_to_pgvector.py --batch-size 50

# Custom ChromaDB path
python3 migrate_to_pgvector.py --chroma-path ./my_chroma_db
```

### Migration Options

```
--dry-run              Preview migration without making changes
--batch-size N         Number of records per batch (default: 100)
--chroma-path PATH     Path to ChromaDB database
--collection-name NAME ChromaDB collection name (default: faces)
```

---

## Testing

### Run Test Suite

```bash
python3 test_pgvector.py
```

Tests include:
1. Database connection
2. Adding faces (individual and batch)
3. Vector similarity search
4. Metadata filtering
5. Hybrid search
6. Database statistics
7. Duplicate detection

---

## Performance Tuning

### Index Parameters

```sql
-- HNSW index tuning
CREATE INDEX idx_embedding_hnsw ON faces
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,              -- Higher = better recall, more memory
    ef_construction = 64 -- Higher = better index quality, slower build
);

-- IVFFlat index (alternative)
CREATE INDEX idx_embedding_ivfflat ON faces
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- Adjust based on dataset size
```

### Connection Pooling

Adjust in `.env`:
```
DB_MIN_CONNECTIONS=1
DB_MAX_CONNECTIONS=10
```

### Batch Operations

Use batch insert for better performance:
```python
db_manager.add_faces_batch(face_list, batch_size=100)
```

---

## File Structure

```
faces_pgvector6/
├── core.py                    # Core backend (updated for pgvector)
├── pgvector_db.py            # PgVectorDatabaseManager class
├── schema.sql                # Database schema
├── migrate_to_pgvector.py    # Migration script
├── test_pgvector.py          # Test suite
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
├── faces.py                  # GUI application
└── PGVECTOR_README.md        # This file
```

---

## Troubleshooting

### Connection Error: "no password supplied"

1. Set PostgreSQL password:
   ```bash
   sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
   ```

2. Update `.env` file:
   ```
   POSTGRES_PASSWORD=postgres
   ```

### Error: "pgvector extension not found"

Run in PostgreSQL:
```sql
CREATE EXTENSION vector;
```

### Error: "Object of type bool is not JSON serializable"

This has been fixed in `pgvector_db.py` with `_convert_to_json_serializable()` method.

### Slow Queries

1. Check if indexes exist:
   ```sql
   SELECT indexname FROM pg_indexes WHERE tablename = 'faces';
   ```

2. Rebuild indexes if needed:
   ```sql
   REINDEX INDEX idx_embedding_hnsw_cosine;
   ```

3. Run VACUUM and ANALYZE:
   ```bash
   sudo -u postgres psql -d vector_db -c "VACUUM ANALYZE faces;"
   ```

---

## Advanced Queries

### Find Similar Faces with Distance Threshold

```python
results = db_manager.search_faces(
    query_embedding=embedding,
    n_results=100,
    distance_metric='cosine'
)
# Filter by distance threshold in Python
filtered = [r for r in results if r['distance'] < 0.5]
```

### Complex Metadata Queries

```python
# Age range + gender
results = db_manager.search_faces(
    query_embedding=embedding,
    metadata_filter={
        'gender': 'female',
        'age_estimate': {'$gte': 20, '$lte': 30}
    }
)
```

### Get All Faces for Specific Model

```sql
SELECT face_id, file_path
FROM faces
WHERE embedding_model = 'statistical';
```

---

## Backup & Maintenance

### Backup Database

```bash
# Full database backup
sudo -u postgres pg_dump vector_db > backup_$(date +%Y%m%d).sql

# Compressed backup
sudo -u postgres pg_dump vector_db | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore Database

```bash
# Restore from backup
sudo -u postgres psql vector_db < backup_20251030.sql
```

### Regular Maintenance

```bash
# Vacuum and analyze
sudo -u postgres psql -d vector_db -c "VACUUM ANALYZE faces;"

# Check database size
sudo -u postgres psql -d vector_db -c "SELECT pg_size_pretty(pg_database_size('vector_db'));"
```

---

## References

- **pgvector GitHub**: https://github.com/pgvector/pgvector
- **PostgreSQL Documentation**: https://www.postgresql.org/docs/
- **HNSW Algorithm**: https://arxiv.org/abs/1603.09320

---

## Support

For issues or questions:
1. Check this documentation
2. Review test_pgvector.py for examples
3. Check PostgreSQL logs: `/var/log/postgresql/`
4. Verify configuration in `.env` file

---

**Last Updated**: 2025-10-30
**Version**: 1.0
