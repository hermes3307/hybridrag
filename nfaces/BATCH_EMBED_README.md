# Batch Face Embedding Tool

Fast, parallel embedding of all face images with multi-threading support.

## Features

✅ **Parallel Processing** - Multi-threaded embedding for maximum speed
✅ **Model Selection** - Choose from multiple embedding models
✅ **Smart Resume** - Automatically skip already embedded images
✅ **Progress Tracking** - Real-time progress bar with ETA
✅ **Detailed Statistics** - Processing rate, duration, success/error counts
✅ **Error Handling** - Continue processing even if some images fail

## Quick Start

### Basic Usage

```bash
# Embed all new images using model from config
./embed_all.sh

# Use specific embedding model
./embed_all.sh facenet

# Use 8 parallel workers for faster processing
./embed_all.sh facenet 8

# List available models
./embed_all.sh --list
```

### Advanced Usage

```bash
# Re-embed ALL images (including already embedded)
./embed_all.sh --force arcface

# Use custom config file
./embed_all.sh --config my_config.json facenet

# Use Python directly for more control
python3 batch_embed.py --model facenet --workers 8 --force
```

## Available Embedding Models

| Model | Accuracy | Speed | Dependencies | Notes |
|-------|----------|-------|--------------|-------|
| **statistical** | ⭐⭐ | ⚡⚡⚡⚡⚡ | None | Always available, good baseline |
| **facenet** | ⭐⭐⭐⭐ | ⚡⚡⚡ | torch, facenet-pytorch | Recommended for most uses |
| **arcface** | ⭐⭐⭐⭐⭐ | ⚡⚡ | insightface, onnxruntime | Best accuracy |
| **deepface** | ⭐⭐⭐⭐ | ⚡⚡ | deepface | Multi-purpose framework |
| **vggface2** | ⭐⭐⭐⭐ | ⚡⚡ | deepface | Deep CNN |
| **openface** | ⭐⭐⭐ | ⚡⚡⚡ | deepface | Lightweight |

## Installation

### Install Dependencies

**For statistical model only (no extra deps):**
```bash
# Already included in base requirements
```

**For FaceNet (recommended):**
```bash
pip3 install facenet-pytorch torch torchvision
```

**For ArcFace (best accuracy):**
```bash
pip3 install insightface onnxruntime
```

**For DeepFace models:**
```bash
pip3 install deepface
```

**Or install all models:**
```bash
pip3 install -r requirements.txt
```

## Command-Line Options

### Shell Script (`embed_all.sh`)

```bash
./embed_all.sh [OPTIONS] [MODEL] [WORKERS]

Options:
  --list, -l         List available embedding models
  --force, -f        Re-embed all images (including already embedded)
  --config FILE      Use specific config file
  --help, -h         Show help message

Arguments:
  MODEL              Embedding model to use
  WORKERS            Number of parallel workers (default: 4)
```

### Python Script (`batch_embed.py`)

```bash
python3 batch_embed.py [OPTIONS]

Options:
  -m, --model MODEL        Embedding model
  -w, --workers N          Number of workers (default: 4)
  -c, --config FILE        Config file path
  -f, --force              Re-embed all images
  -l, --list-models        List available models
```

## Examples

### Example 1: First Time Embedding

```bash
# Embed all downloaded images with FaceNet using 4 workers
./embed_all.sh facenet

Output:
══════════════════════════════════════════════════════════════════════════════
BATCH FACE EMBEDDING
══════════════════════════════════════════════════════════════════════════════
Embedding Model: facenet
Max Workers: 4
Faces Directory: ./faces
Found 1000 files to embed
══════════════════════════════════════════════════════════════════════════════
[████████████████████████████████████████] 1000/1000 (100.0%) | Success: 998 | Errors: 2 | ETA: 00:00:00
══════════════════════════════════════════════════════════════════════════════
EMBEDDING COMPLETE
══════════════════════════════════════════════════════════════════════════════
Total Files: 1000
Successfully Embedded: 998
Errors: 2
Total Duration: 0:15:30
Average Time per Image: 0.93s
Processing Rate: 1.08 images/sec
```

### Example 2: Speed Comparison

```bash
# Test with different worker counts
./embed_all.sh statistical 2   # 2 workers
./embed_all.sh statistical 4   # 4 workers (default)
./embed_all.sh statistical 8   # 8 workers (faster on multi-core)
```

### Example 3: Model Comparison

```bash
# Compare different models (re-embed with each)
./embed_all.sh --force statistical  # Fast but basic
./embed_all.sh --force facenet      # Balanced (recommended)
./embed_all.sh --force arcface      # Best accuracy
```

### Example 4: Resume Interrupted Embedding

```bash
# If embedding was interrupted, just run again
# It will automatically skip already embedded images
./embed_all.sh facenet
```

## Performance Tips

### Optimal Worker Count

- **2 workers**: Low-end systems, Raspberry Pi
- **4 workers**: Standard desktops (default)
- **8 workers**: High-end systems, servers
- **16+ workers**: Very powerful systems with many cores

**Rule of thumb**: Use `CPU cores - 1` for optimal performance

### Speed Benchmarks (approximate)

On a typical desktop (4 cores):

| Model | Workers | Images/sec | Time for 1000 images |
|-------|---------|------------|---------------------|
| statistical | 4 | 10-15 | ~1-2 min |
| facenet | 4 | 1-2 | ~10-15 min |
| arcface | 4 | 0.5-1 | ~15-30 min |

### Memory Considerations

- **Statistical**: ~100MB per worker
- **FaceNet**: ~500MB per worker
- **ArcFace**: ~1GB per worker

Reduce workers if you run out of memory.

## Workflow Integration

### Typical Workflow

```bash
# 1. Download faces (using app or downloader)
./run_app.sh
# (download some faces)

# 2. Batch embed all downloaded faces
./embed_all.sh facenet 4

# 3. Verify embeddings
python3 -c "
from pgvector_db import PgVectorDatabaseManager
from core import SystemConfig
config = SystemConfig.from_file('system_config.json')
db = PgVectorDatabaseManager(config)
stats = db.get_statistics()
print(f'Total faces in DB: {stats[\"total_faces\"]}')
"

# 4. Start searching!
./run_app.sh
```

### Automated Pipeline

```bash
#!/bin/bash
# automated_pipeline.sh - Download and embed in batches

# Download 100 faces
python3 -c "from core import *; system = IntegratedFaceSystem(); ..."

# Embed immediately
./embed_all.sh facenet 4

# Repeat as needed
```

## Troubleshooting

### Issue: "Model not available"

```bash
# List installed models
./embed_all.sh --list

# Install missing model (example for FaceNet)
pip3 install facenet-pytorch torch torchvision
```

### Issue: Out of memory

```bash
# Reduce number of workers
./embed_all.sh facenet 2  # Instead of 4
```

### Issue: Some images fail to embed

- Check the error messages in the output
- Failed images are logged but don't stop the process
- Common causes: corrupted images, unsupported formats
- The script continues with other images

### Issue: Database connection error

```bash
# Verify database is running
psql -h localhost -U postgres -d vector_db -c "SELECT COUNT(*) FROM faces;"

# Check .env file
cat .env

# Test connection
python3 verify_db.py
```

### Issue: Very slow processing

**Possible causes:**
1. Using too many workers (context switching overhead)
2. Slow disk I/O (faces on network drive, slow SD card)
3. CPU-intensive model with limited cores

**Solutions:**
```bash
# Try fewer workers
./embed_all.sh facenet 2

# Use faster model
./embed_all.sh statistical 4

# Check system resources
htop  # Monitor CPU/memory usage during embedding
```

## Technical Details

### How It Works

1. **Discovery**: Scans faces directory for all image files
2. **Filtering**: Checks database to find unembed images (unless --force)
3. **Parallel Processing**: Distributes work across multiple threads
4. **Face Analysis**: Extracts facial features from each image
5. **Embedding**: Creates vector embedding using selected model
6. **Storage**: Saves embedding to PostgreSQL with pgvector
7. **Progress**: Updates real-time progress bar with statistics

### Database Schema

Embeddings are stored in the `faces` table:
```sql
face_id         TEXT PRIMARY KEY
file_path       TEXT
features        JSONB              -- Facial features
embedding       VECTOR(512)        -- Vector embedding
timestamp       TIMESTAMP
image_hash      TEXT UNIQUE
embedding_model TEXT               -- Model used
```

### Thread Safety

- Each thread has its own database connection
- Thread-safe statistics tracking with locks
- No race conditions on file processing
- Safe to interrupt (Ctrl+C) at any time

## Best Practices

1. **Start with statistical model** to test your pipeline
2. **Use --list** to check available models before starting
3. **Test with small batch first** (move some images to test folder)
4. **Monitor the first few minutes** to ensure no errors
5. **Use appropriate worker count** for your system
6. **Don't run multiple instances** on same database simultaneously
7. **Backup database** before re-embedding (--force)

## See Also

- `requirements.txt` - Python dependencies
- `PGVECTOR_README.md` - Database setup and usage
- `README_UNIFIED_APP.md` - Main application guide
- `SEARCH_GUIDE.md` - Search functionality

## Support

For issues or questions:
1. Check this README
2. Review error messages in console output
3. Check system requirements in requirements.txt
4. Verify database connection with verify_db.py
