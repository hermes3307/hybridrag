# Face Processing System - Operating Modes

## Quick Reference

### Launch Commands

| Mode | Command | Description |
|------|---------|-------------|
| **Interactive** | `./start_system_selector.sh` | Choose mode interactively |
| **Multi-Model** | `./start_system_mm.sh` | Launch with multiple embedding models |
| **Legacy** | `./start_system.sh` | Launch with single embedding model |
| **CLI Multi-Model** | `python3 faces.py --mode multimodel` | Direct multi-model launch |
| **CLI Legacy** | `python3 faces.py --mode legacy` | Direct legacy launch |
| **CLI Auto** | `python3 faces.py --mode auto` | Auto-detect mode |

## System Modes

### 1. Multi-Model Mode

**Best for:** New projects, research, model comparison

**Features:**
- Store embeddings from multiple models (FaceNet, ArcFace, VGGFace2, etc.)
- Search using any model independently
- Compare results across models
- Add new models without data migration

**Database:** Uses `faces` table with multiple embedding columns (`embedding_facenet`, `embedding_arcface`, etc.)

**Schema:** `schema.sql`

**Configuration (.env):**
```bash
SYSTEM_MODE=multimodel
EMBEDDING_MODELS=facenet,arcface
DEFAULT_SEARCH_MODEL=facenet
```

### 2. Legacy Mode

**Best for:** Backward compatibility, single-model workflows, lower storage

**Features:**
- Store one embedding per face
- Single model at a time
- Lower storage requirements
- Simpler schema

**Database:** Uses `faces_legacy` table with single embedding column

**Schema:** `schema_legacy.sql`

**Configuration (.env):**
```bash
SYSTEM_MODE=legacy
EMBEDDING_MODEL=facenet
```

### 3. Auto Mode

**Best for:** Automatic schema detection

**Features:**
- Automatically detects which schema is deployed
- Uses multi-model if `faces` table has multiple embedding columns
- Falls back to legacy if `faces_legacy` table exists

**Configuration (.env):**
```bash
SYSTEM_MODE=auto
```

## File Overview

### Launcher Scripts

| File | Purpose |
|------|---------|
| `start_system_selector.sh` | Interactive mode selector with setup wizard |
| `start_system_mm.sh` | Multi-model launcher |
| `start_system.sh` | Legacy/original launcher |

### Database Schemas

| File | Purpose |
|------|---------|
| `schema.sql` | Multi-model schema (multiple embedding columns) |
| `schema_legacy.sql` | Legacy schema (single embedding column) |

### Configuration

| File | Purpose |
|------|---------|
| `.env` | Environment configuration for database and models |

### Documentation

| File | Purpose |
|------|---------|
| `MULTIMODEL_SETUP.md` | Complete setup guide |
| `SYSTEM_MODES.md` | This file - quick reference |

## Database Schema Comparison

### Multi-Model Schema (schema.sql)

```sql
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(255) UNIQUE,

    -- Multiple embedding columns
    embedding_facenet vector(512),
    embedding_arcface vector(512),
    embedding_vggface2 vector(512),
    embedding_insightface vector(512),
    embedding_statistical vector(512),

    -- Track processed models
    models_processed TEXT[],

    -- Metadata
    age_estimate INTEGER,
    gender VARCHAR(20),
    -- ... more metadata
);
```

### Legacy Schema (schema_legacy.sql)

```sql
CREATE TABLE faces_legacy (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(255) UNIQUE,

    -- Single embedding column
    embedding vector(512),
    embedding_model VARCHAR(50),  -- 'facenet', 'arcface', etc.

    -- Metadata
    age_estimate INTEGER,
    gender VARCHAR(20),
    -- ... more metadata
);
```

## Setup Steps

### New Installation (Multi-Model)

1. Deploy schema:
   ```bash
   psql -U postgres -d vector_db -f schema.sql
   ```

2. Configure .env:
   ```bash
   SYSTEM_MODE=multimodel
   EMBEDDING_MODELS=facenet,arcface
   ```

3. Launch:
   ```bash
   ./start_system_mm.sh
   ```

### New Installation (Legacy)

1. Deploy schema:
   ```bash
   psql -U postgres -d vector_db -f schema_legacy.sql
   ```

2. Configure .env:
   ```bash
   SYSTEM_MODE=legacy
   EMBEDDING_MODEL=facenet
   ```

3. Launch:
   ```bash
   ./start_system.sh
   ```

### Interactive Setup (Recommended for First Time)

1. Launch selector:
   ```bash
   ./start_system_selector.sh
   ```

2. Choose option 4 (Setup/Initialize Database)

3. Select schema type

4. Launch application

## Migration Between Modes

### From Legacy to Multi-Model

1. Backup existing data:
   ```bash
   ./backup_database.sh
   ```

2. Deploy multi-model schema:
   ```bash
   psql -U postgres -d vector_db -f schema.sql
   ```

3. Run migration:
   ```bash
   ./migrate_to_multimodel.sh
   ```

### From Multi-Model to Legacy

Extract specific model embeddings:

```sql
SELECT migrate_multimodel_to_legacy('facenet');
```

This SQL function copies data from the `faces` table to `faces_legacy` table,
extracting only the specified model's embeddings.

## Environment Variables

### Common Variables

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# System Mode
SYSTEM_MODE=multimodel  # multimodel, legacy, or auto
```

### Multi-Model Specific

```bash
# Comma-separated list of models
EMBEDDING_MODELS=facenet,arcface,vggface2

# Default model for searches
DEFAULT_SEARCH_MODEL=facenet
```

### Legacy Specific

```bash
# Single model
EMBEDDING_MODEL=facenet
```

## Available Embedding Models

| Model | Speed | Accuracy | Library | Notes |
|-------|-------|----------|---------|-------|
| **statistical** | ⚡⚡⚡ | ⭐⭐ | Built-in | No deep learning, always available |
| **facenet** | ⚡⚡⚡ | ⭐⭐⭐ | facenet-pytorch | Fast and accurate |
| **arcface** | ⚡⚡ | ⭐⭐⭐⭐ | insightface | State-of-the-art accuracy |
| **vggface2** | ⚡⚡ | ⭐⭐⭐ | deepface | Good balance |
| **insightface** | ⚡⚡ | ⭐⭐⭐⭐ | insightface | High performance |

## Checking Current Mode

### From Command Line

```bash
echo $SYSTEM_MODE
```

### From Database

```bash
# Check for multi-model schema
psql -U postgres -d vector_db -c "SELECT column_name FROM information_schema.columns WHERE table_name='faces' AND column_name LIKE 'embedding_%';"

# Check for legacy schema
psql -U postgres -d vector_db -c "\dt faces_legacy"
```

### From Application

The window title shows the current mode:
- "Face Processing System - Multi-Model Mode"
- "Face Processing System - Legacy Mode"
- "Face Processing System - Auto Mode"

## GUI Changes for Multi-Model Support

The GUI now supports mode-aware operation:

1. **Window title** shows current mode
2. **Command-line arguments** for programmatic mode selection
3. **Environment variables** for configuration
4. **Compatible with both schemas** automatically

## Troubleshooting

### Issue: "Schema not detected"

**Solution:** Deploy the appropriate schema:
```bash
./start_system_selector.sh  # Then choose option 4
```

### Issue: Mode confusion

**Check .env file:**
```bash
cat .env | grep SYSTEM_MODE
```

**Reset mode:**
```bash
export SYSTEM_MODE=multimodel
```

### Issue: Model not available

**Install required libraries:**
```bash
pip install facenet-pytorch insightface deepface
```

**Check available models:**
```python
from core import check_embedding_models
print(check_embedding_models())
```

## Summary

| Aspect | Multi-Model | Legacy |
|--------|-------------|--------|
| **Embeddings per face** | Multiple (1 per model) | Single |
| **Storage** | ~5x base | 1x base |
| **Flexibility** | High | Low |
| **Schema file** | schema.sql | schema_legacy.sql |
| **Table name** | faces | faces_legacy |
| **Launcher** | start_system_mm.sh | start_system.sh |
| **Use case** | New projects, research | Backward compatibility |

## Next Steps

1. **Read full documentation:** See `MULTIMODEL_SETUP.md`
2. **Choose your mode:** Based on your needs
3. **Deploy schema:** Using appropriate .sql file
4. **Configure:** Edit .env file
5. **Launch:** Use appropriate launcher script
6. **Process faces:** Generate embeddings
7. **Search:** Query with your chosen model(s)

---

For detailed information, see `MULTIMODEL_SETUP.md`
