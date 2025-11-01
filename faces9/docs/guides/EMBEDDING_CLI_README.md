# Embedding Management CLI

A comprehensive command-line interface for managing face embeddings in the PostgreSQL vector database.

## Features

✅ **Database Statistics**
- Total embedded vectors count
- Breakdown by embedding model
- Recent activity (last 7 days)
- Embedded face IDs tracking

✅ **File System Analysis**
- Count total image files
- Count total JSON metadata files
- Match image-JSON pairs
- Identify unmatched files

✅ **Smart Embedding Status**
- Track already embedded images
- Identify pending embeddings
- Display progress percentage
- Visual progress bar

✅ **Batch Embedding**
- Process multiple images efficiently
- Real-time progress display
- Detailed ETA and speed metrics
- Error tracking and reporting

✅ **Multiple Embedding Models**
- Statistical (default, always available)
- FaceNet (requires facenet-pytorch)
- ArcFace (requires insightface)
- DeepFace (requires deepface)
- VGGFace2 (requires deepface)
- OpenFace (requires deepface)

## Installation

### Prerequisites

1. **PostgreSQL with pgvector extension**
   ```bash
   # Install PostgreSQL
   sudo apt-get install postgresql postgresql-contrib

   # Install pgvector extension
   cd /tmp
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   sudo make install

   # Enable extension
   psql -U postgres -d vector_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

2. **Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment configuration**

   Create or edit `.env` file:
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

## Usage

### Basic Commands

#### 1. Display Statistics and Prompt for Embedding (Interactive)
```bash
python embedding_manager_cli.py
```

This will:
- Show database statistics
- Count and analyze files
- Display embedding status
- Ask if you want to embed pending images

#### 2. Display Statistics Only
```bash
python embedding_manager_cli.py --stats-only
```

Perfect for monitoring without making changes.

#### 3. Auto-Embed All Pending Images
```bash
python embedding_manager_cli.py --auto-embed
```

Automatically embeds all pending images without prompting.

### Advanced Options

#### Use Specific Embedding Model
```bash
# Use FaceNet model
python embedding_manager_cli.py --model facenet --auto-embed

# Use ArcFace model
python embedding_manager_cli.py --model arcface --auto-embed

# Use Statistical model (default)
python embedding_manager_cli.py --model statistical --auto-embed
```

#### Custom Faces Directory
```bash
python embedding_manager_cli.py --faces-dir /path/to/faces --auto-embed
```

#### Quiet Mode (Reduced Output)
```bash
python embedding_manager_cli.py --quiet --auto-embed
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--faces-dir PATH` | Path to faces directory | From `.env` or `./faces` |
| `--model MODEL` | Embedding model to use | From `.env` or `statistical` |
| `--auto-embed` | Auto-embed without prompting | `False` (interactive) |
| `--stats-only` | Display statistics only | `False` |
| `--quiet` | Reduce output verbosity | `False` |
| `-h, --help` | Show help message | - |

### Available Models

| Model | Dimensions | Requirements | Speed | Accuracy |
|-------|-----------|--------------|-------|----------|
| **statistical** | 512 | None (built-in) | ⚡⚡⚡ Fast | ⭐⭐ Basic |
| **facenet** | 512 | `facenet-pytorch` | ⚡⚡ Medium | ⭐⭐⭐ Good |
| **arcface** | 512 | `insightface` | ⚡ Slow | ⭐⭐⭐⭐ Excellent |
| **deepface** | 4096 | `deepface` | ⚡ Slow | ⭐⭐⭐⭐ Excellent |
| **vggface2** | 2622 | `deepface` | ⚡ Slow | ⭐⭐⭐ Good |
| **openface** | 128 | `deepface` | ⚡⚡ Medium | ⭐⭐ Basic |

## Output Examples

### Statistics Display

```
================================================================================
📊 EMBEDDING MANAGEMENT DASHBOARD
================================================================================

🗄️  DATABASE STATISTICS
--------------------------------------------------------------------------------
Total Embedded Vectors: 29,135

Embedding Models Used:
  • facenet: 29,135 vectors (100.0%)

Recent Activity (Last 7 Days):
  • 2025-11-01: 1 vectors
  • 2025-10-31: 29,134 vectors

📁 FILE SYSTEM STATISTICS
--------------------------------------------------------------------------------
Faces Directory: ./faces
Total Image Files: 58,660
Total JSON Files: 58,660

✅ Matched Pairs (Image + JSON): 58,660
⚠️  Unmatched Images (No JSON): 0
⚠️  Unmatched JSONs (No Image): 0

🎯 EMBEDDING STATUS
--------------------------------------------------------------------------------
Already Embedded: 58,063
Pending Embedding: 597

Progress: [█████████████████████████████████████████████████░] 99.0%

================================================================================
```

### Embedding Progress

```
🚀 Starting batch embedding of 597 images...
📦 Embedding Model: statistical
--------------------------------------------------------------------------------

✅ Embedder initialized: statistical

Progress:
--------------------------------------------------------------------------------
[████████████████████████████████████████] 100.0% | 597/597 | Success: 597 | Errors: 0 | ETA: 0s
--------------------------------------------------------------------------------

================================================================================
📈 EMBEDDING SUMMARY
================================================================================
Total Processed: 597
✅ Successfully Embedded: 597
❌ Errors: 0
⏱️  Total Time: 5m 23s
⚡ Average Speed: 0.54 seconds/image

================================================================================
```

## File Structure

The CLI expects the following file structure in the faces directory:

```
faces/
├── face_20251018_123838_826_4ba7ed60.jpg
├── face_20251018_123838_826_4ba7ed60.json
├── face_20251018_123840_711_887772e4.jpg
├── face_20251018_123840_711_887772e4.json
└── ...
```

Each image must have a corresponding JSON metadata file with the same base name.

### JSON Metadata Structure

```json
{
  "filename": "face_20251018_123838_826_4ba7ed60.jpg",
  "face_id": "20251018_123838_826",
  "md5_hash": "4ba7ed6062c2e873c55d3c24e80bf0d5",
  "face_features": {
    "brightness": 82.49,
    "contrast": 49.96,
    "faces_detected": 1,
    "skin_tone": "medium",
    "hair_color": "brown",
    "age_group": "senior",
    "estimated_sex": "male"
  },
  "queryable_attributes": {
    "sex": "male",
    "age_group": "senior",
    "skin_tone": "medium"
  }
}
```

## Database Schema

The CLI stores embeddings in PostgreSQL with the following structure:

```sql
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(255) UNIQUE NOT NULL,
    embedding vector(512),  -- 512-dimensional vector
    metadata JSONB,
    embedding_model VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for fast similarity search
CREATE INDEX faces_embedding_idx
ON faces USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

## Automation Examples

### Cron Job for Nightly Embedding

Add to crontab:
```bash
# Run every night at 2 AM
0 2 * * * cd /home/pi/hybridrag/faces8 && python3 embedding_manager_cli.py --auto-embed --model facenet >> /var/log/embeddings.log 2>&1
```

### Shell Script for Batch Processing

```bash
#!/bin/bash
# embed_all.sh

cd /home/pi/hybridrag/faces8

echo "Starting embedding process..."
python3 embedding_manager_cli.py --auto-embed --model facenet

if [ $? -eq 0 ]; then
    echo "✅ Embedding completed successfully"
else
    echo "❌ Embedding failed"
    exit 1
fi
```

### Python Script Integration

```python
import subprocess

# Run embedding with specific model
result = subprocess.run(
    ['python3', 'embedding_manager_cli.py', '--auto-embed', '--model', 'facenet'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("Embedding successful")
else:
    print(f"Error: {result.stderr}")
```

## Troubleshooting

### Connection Errors

```bash
# Test database connection
PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c "SELECT COUNT(*) FROM faces;"

# Check if pgvector extension is installed
PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Model Loading Issues

```bash
# Check if required packages are installed
pip list | grep -E "facenet|insightface|deepface"

# Test model import
python3 -c "from facenet_pytorch import InceptionResnetV1; print('FaceNet OK')"
```

### File Permission Issues

```bash
# Check faces directory permissions
ls -la /home/pi/faces

# Fix permissions if needed
chmod -R 755 /home/pi/faces
```

## Performance Tips

1. **Use Statistical Model for Speed**: If you need fast processing, use the default statistical model
2. **Batch Processing**: The CLI automatically processes in batches for efficiency
3. **Connection Pooling**: Database connections are pooled for better performance
4. **Index Optimization**: Ensure HNSW indexes are created for fast similarity search

## API Key Support

The system is designed to work with various AI face generation services. You can configure API keys in the `.env` file or through environment variables.

## Related Files

- `core.py` - Core face processing and embedding functionality
- `pgvector_db.py` - PostgreSQL database manager
- `.env` - Environment configuration
- `schema.sql` - Database schema

## License

See the main project LICENSE file.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in the console output
3. Check PostgreSQL logs: `/var/log/postgresql/`
4. Review the error summary in the embedding output
