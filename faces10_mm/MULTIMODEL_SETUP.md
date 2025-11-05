# Multi-Model Setup Guide

This guide explains how to use the multi-model face recognition system with support for multiple embedding models.

## Overview

The system now supports two operational modes:

1. **Multi-Model Mode** (recommended): Store and search embeddings from multiple models simultaneously
2. **Legacy Mode**: Single embedding model for backward compatibility with older setups

## Quick Start

### Option 1: Interactive Launcher (Recommended)

```bash
./start_system_selector.sh
```

This interactive script will:
- Detect your current database schema
- Show available modes
- Guide you through setup if needed
- Launch the application in your chosen mode

### Option 2: Direct Launch

**Multi-Model Mode:**
```bash
./start_system_mm.sh
```

**Legacy Mode:**
```bash
./start_system.sh
```

**With Command-Line Arguments:**
```bash
python3 faces.py --mode multimodel  # Multi-model mode
python3 faces.py --mode legacy      # Legacy mode
python3 faces.py --mode auto        # Auto-detect from environment
```

## Database Setup

### Multi-Model Schema (schema.sql)

The multi-model schema stores embeddings from different models in separate columns:

```sql
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(255) UNIQUE NOT NULL,

    -- Multiple embedding columns
    embedding_facenet vector(512),
    embedding_arcface vector(512),
    embedding_vggface2 vector(512),
    embedding_insightface vector(512),
    embedding_statistical vector(512),

    -- Track which models have been processed
    models_processed TEXT[],

    -- Metadata...
);
```

**Deploy:**
```bash
psql -U postgres -d vector_db -f schema.sql
```

### Legacy Schema (schema_legacy.sql)

The legacy schema uses a single embedding column with a model identifier:

```sql
CREATE TABLE faces_legacy (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(255) UNIQUE NOT NULL,

    -- Single embedding column
    embedding vector(512),
    embedding_model VARCHAR(50),  -- e.g., 'facenet', 'arcface'

    -- Metadata...
);
```

**Deploy:**
```bash
psql -U postgres -d vector_db -f schema_legacy.sql
```

### Both Schemas (Side-by-Side)

You can deploy both schemas to maintain compatibility:

```bash
psql -U postgres -d vector_db -f schema.sql
psql -U postgres -d vector_db -f schema_legacy.sql
```

This creates both `faces` (multi-model) and `faces_legacy` (single-model) tables.

## Configuration

### Environment Variables (.env)

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Multi-Model Configuration
EMBEDDING_MODELS=facenet,arcface          # Comma-separated list
DEFAULT_SEARCH_MODEL=facenet              # Default for searches

# System Mode
SYSTEM_MODE=multimodel                    # multimodel, legacy, or auto
```

### Available Embedding Models

- **facenet**: FaceNet (InceptionResnetV1) - Fast and accurate
- **arcface**: ArcFace - State-of-the-art accuracy
- **vggface2**: VGGFace2 - Good balance
- **insightface**: InsightFace - High performance
- **statistical**: Statistical features - Fast, no deep learning required

## Features by Mode

### Multi-Model Mode Features

✓ Store embeddings from multiple models per face
✓ Search using any available model
✓ Compare results across different models
✓ Per-model statistics and analytics
✓ Independent model indexes for fast search
✓ Future-proof: add new models without migration

### Legacy Mode Features

✓ Single embedding per face
✓ Backward compatible with older systems
✓ Lower storage requirements
✓ Simpler schema and queries
✓ Good for single-model workflows

## Usage Examples

### Multi-Model Mode Workflow

1. **Start the system:**
   ```bash
   ./start_system_mm.sh
   ```

2. **Download faces:**
   - Use the "Download Faces" tab
   - Download AI-generated faces or capture from camera

3. **Generate embeddings:**
   - Go to "Process & Embed" tab
   - Select multiple models (e.g., facenet, arcface)
   - Process faces to generate embeddings for all selected models

4. **Search with specific model:**
   - Go to "Search Faces" tab
   - Select a search model (facenet, arcface, etc.)
   - Upload query image
   - View results from that specific model

5. **Compare across models:**
   - Search with different models
   - Compare which model gives better results for your use case

### Legacy Mode Workflow

1. **Start the system:**
   ```bash
   ./start_system.sh
   ```

2. **Configure single model:**
   - Go to Configuration tab
   - Select one embedding model

3. **Process and search:**
   - Download faces
   - Process with selected model
   - Search using the same model

## Migration

### From Legacy to Multi-Model

If you have existing data in legacy mode and want to migrate:

1. **Deploy multi-model schema:**
   ```bash
   psql -U postgres -d vector_db -f schema.sql
   ```

2. **Run migration script:**
   ```bash
   python3 migrate_to_multimodel.sh
   ```

This will copy your data and preserve embeddings.

### From Multi-Model to Legacy

To extract a single model's embeddings to legacy format:

```sql
-- Use the built-in migration function
SELECT migrate_multimodel_to_legacy('facenet');
```

This extracts FaceNet embeddings from the multi-model table to legacy format.

## Performance Considerations

### Multi-Model Mode

- **Storage**: ~5x more storage (one embedding per model)
- **Indexing**: Separate HNSW index per model
- **Search Speed**: Same as single model (searches one index at a time)
- **Processing**: Can process multiple models in parallel

### Legacy Mode

- **Storage**: 1x baseline storage
- **Indexing**: Single HNSW index
- **Search Speed**: Fast
- **Processing**: One model at a time

## Troubleshooting

### "Multi-model schema not detected"

**Solution:** Deploy the schema:
```bash
psql -U postgres -d vector_db -f schema.sql
```

### "Cannot connect to PostgreSQL"

**Solution:** Check PostgreSQL is running:
```bash
systemctl status postgresql
# or
./check_postgres.sh
```

### "Model not available"

**Solution:** Install the required model libraries:
```bash
pip install facenet-pytorch  # For FaceNet
pip install insightface      # For ArcFace/InsightFace
pip install deepface         # For VGGFace2 and others
```

### Mode confusion

**Check current mode:**
```bash
echo $SYSTEM_MODE
```

**Reset mode:**
```bash
export SYSTEM_MODE=multimodel
# or edit .env file
```

## File Reference

- `start_system_selector.sh` - Interactive mode selector
- `start_system_mm.sh` - Direct multi-model launcher
- `start_system.sh` - Legacy/original launcher
- `schema.sql` - Multi-model database schema
- `schema_legacy.sql` - Legacy single-model schema
- `.env` - Environment configuration
- `faces.py` - Main GUI application (supports --mode argument)

## Advanced Usage

### Custom Model Configuration

Edit `.env`:
```bash
# Use only specific models
EMBEDDING_MODELS=arcface,insightface

# Set default search model
DEFAULT_SEARCH_MODEL=arcface
```

### Programmatic Mode Selection

```python
import os

# Set mode before importing
os.environ['SYSTEM_MODE'] = 'multimodel'
os.environ['EMBEDDING_MODELS'] = 'facenet,arcface,vggface2'

# Then import and use
from faces import IntegratedFaceGUI
app = IntegratedFaceGUI()
app.run()
```

### Database Statistics

```bash
# Multi-model statistics
psql -U postgres -d vector_db -c "SELECT * FROM get_database_stats();"

# Legacy statistics
psql -U postgres -d vector_db -c "SELECT * FROM get_database_stats_legacy();"
```

## Support

For issues or questions:
1. Check this documentation
2. Review logs in the GUI
3. Check PostgreSQL logs
4. Verify schema deployment: `\dt` in psql

## Best Practices

1. **Start with multi-model mode** for new projects (more flexible)
2. **Use legacy mode** only for backward compatibility
3. **Test models** before processing large datasets
4. **Monitor storage** when using multiple models
5. **Backup database** before migrations
6. **Use appropriate models** for your use case:
   - Speed: statistical, facenet
   - Accuracy: arcface, insightface
   - Balance: vggface2

## Summary

| Feature | Multi-Model | Legacy |
|---------|-------------|--------|
| Multiple models per face | ✓ | ✗ |
| Storage per face | 5x | 1x |
| Search speed | Fast | Fast |
| Flexibility | High | Low |
| Schema | schema.sql | schema_legacy.sql |
| Launcher | start_system_mm.sh | start_system.sh |
| Recommended for | New projects | Backward compatibility |
