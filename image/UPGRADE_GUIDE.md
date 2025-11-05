# Image Search System - Multi-Embedding Upgrade Guide

## Overview
Your image search system has been significantly enhanced with:
1. **Multiple embedding models** per image (CLIP, YOLO, ResNet, Statistical)
2. **Diverse AI-generated image sources** (faces, artwork, animals, objects)
3. **Multi-embedding fusion search** for better accuracy
4. **Text-to-image search** via CLIP
5. **Updated database schema** supporting multiple embeddings

## What's New

### 1. New AI Image Sources
The downloader now supports diverse AI-generated content beyond faces:

| Source | Category | Description |
|--------|----------|-------------|
| `thispersondoesnotexist` | Faces | High quality AI-generated faces (1024x1024) |
| `100k-faces` | Faces | 100K AI-generated faces dataset |
| `thisartworkdoesnotexist` | Artwork | AI-generated artwork and paintings |
| `thiscatdoesnotexist` | Animals | GAN-generated cat images |
| `thishorsedoesnotexist` | Animals | AI-generated horse images |
| `unrealperson` | Mixed | Diverse AI-generated people and objects |

**Usage:**
```python
# In system_config.json or programmatically
config.download_source = 'thisartworkdoesnotexist'  # or any source above
```

### 2. Multiple Embedding Models

The system now generates **3-4 embeddings per image**:

| Model | Dimensions | Purpose | Best For |
|-------|------------|---------|----------|
| **CLIP** | 512 | Image-text similarity | Text queries, semantic search |
| **YOLO** | 80→512 | Object detection | Finding objects/things in images |
| **ResNet** | 2048→512 | Deep visual features | Visual similarity, texture, style |
| **Statistical** | 512 | Color/brightness stats | Fallback, basic similarity |

### 3. New Database Schema

**Before:**
- Single `images` table with one embedding per image

**After:**
- `images` table: Image metadata only
- `image_embeddings` table: Multiple embeddings per image
  - Supports 512-dim, 1024-dim embeddings
  - One row per (image, model) combination

### 4. Enhanced Search Capabilities

#### Single Model Search
```python
# Search using CLIP
results = db_manager.search_images(
    embedding=clip_embedding,
    model_name='clip',
    limit=10
)
```

#### Multi-Embedding Fusion Search
```python
# Combine all three models with custom weights
results = db_manager.multi_embedding_search(
    clip_emb=clip_embedding,
    yolo_emb=yolo_embedding,
    resnet_emb=resnet_embedding,
    clip_weight=0.5,     # 50% weight on semantic similarity
    yolo_weight=0.25,    # 25% weight on objects
    resnet_weight=0.25,  # 25% weight on visual features
    limit=10
)
```

#### Text-to-Image Search
```python
# Search images using natural language
results = db_manager.text_to_image_search(
    text_query="a beautiful sunset over mountains",
    limit=10
)
```

## Migration Steps

### Step 1: Backup Existing Data
```bash
# Backup your PostgreSQL database (if you had one with old name)
pg_dump -U postgres -d vector_db > backup_before_upgrade.sql

# Backup images directory
cp -r ./images ./images_backup
```

### Step 2: Create New Database and Apply Schema
```bash
# Create the new image_vector database
psql -U postgres -c "CREATE DATABASE image_vector;"

# Enable pgvector extension
psql -U postgres -d image_vector -c "CREATE EXTENSION vector;"

# Apply the new schema
psql -U postgres -d image_vector -f schema.sql
```

**Note:** We're using a new database name `image_vector` to avoid conflicts with your existing `vector_db`.

### Step 3: Install Required Dependencies
```bash
# Core dependencies
pip install torch torchvision transformers pillow numpy

# For CLIP
pip install transformers

# For YOLO
pip install ultralytics

# For ResNet (included in torchvision)
# Already installed with torch

# Database
pip install psycopg2-binary
```

### Step 4: Re-process Existing Images

If you have existing images that need new embeddings:

```python
from core import IntegratedImageSystem

# Initialize system
system = IntegratedImageSystem()
system.initialize()

# Process all images with multiple embeddings
system.processor.process_all_images(
    progress_callback=lambda cur, total, msg: print(f"{msg}")
)
```

This will:
- Generate CLIP, YOLO, ResNet, and Statistical embeddings for each image
- Store all embeddings in the new `image_embeddings` table
- Take approximately 1-5 seconds per image (depending on GPU availability)

## Configuration Changes

### system_config.json

The config file now supports:

```json
{
  "images_dir": "./images",
  "db_host": "localhost",
  "db_port": 5432,
  "db_name": "image_vector",
  "db_user": "postgres",
  "db_password": "postgres",
  "download_delay": 1.0,
  "max_workers": 2,
  "batch_size": 50,
  "embedding_model": "clip",
  "download_source": "thisartworkdoesnotexist"
}
```

**Note:** The `embedding_model` parameter is now used only for backward compatibility. The system automatically generates all available embeddings.

## Code Examples

### Example 1: Download Diverse Images
```python
from core import IntegratedImageSystem

system = IntegratedImageSystem()
system.initialize()

# Download artwork
system.config.download_source = 'thisartworkdoesnotexist'
for i in range(10):
    file_path = system.downloader.download_image()
    print(f"Downloaded artwork: {file_path}")

# Download cats
system.config.download_source = 'thiscatdoesnotexist'
for i in range(10):
    file_path = system.downloader.download_image()
    print(f"Downloaded cat: {file_path}")
```

### Example 2: Process and Search
```python
from core import IntegratedImageSystem
from PIL import Image

system = IntegratedImageSystem()
system.initialize()

# Process an image
image_path = "./images/sample.jpg"
system.processor.process_image_file(image_path)

# Search by text
results = system.db_manager.text_to_image_search(
    text_query="cat sitting on a chair",
    limit=5
)

for result in results:
    print(f"Match: {result['file_path']}, Distance: {result['distance']}")
```

### Example 3: Multi-Model Search
```python
# Generate embeddings for a query image
from core import ImageEmbedder, ImageAnalyzer

query_image = "./query.jpg"
analyzer = ImageAnalyzer()
features = analyzer.analyze_image(query_image)

# Create embeddings with each model
clip_embedder = ImageEmbedder('clip')
yolo_embedder = ImageEmbedder('yolo')
resnet_embedder = ImageEmbedder('resnet')

clip_emb = clip_embedder.create_embedding(query_image, features)
yolo_emb = yolo_embedder.create_embedding(query_image, features)
resnet_emb = resnet_embedder.create_embedding(query_image, features)

# Search with fusion
results = system.db_manager.multi_embedding_search(
    clip_emb=clip_emb,
    yolo_emb=yolo_emb,
    resnet_emb=resnet_emb,
    limit=10
)
```

## Performance Considerations

### Embedding Generation Speed
- **CLIP**: ~0.5-1s per image (CPU), ~0.1s (GPU)
- **YOLO**: ~0.3-0.8s per image (CPU), ~0.05s (GPU)
- **ResNet**: ~0.2-0.5s per image (CPU), ~0.03s (GPU)
- **Statistical**: ~0.01s per image

**Total per image: ~1-3s (CPU), ~0.2-0.5s (GPU)**

### Storage Requirements
- **Each image**: ~2KB metadata
- **Each embedding**: ~2KB (512-dim) or ~4KB (1024-dim)
- **4 embeddings per image**: ~8KB
- **Per 1000 images**: ~10MB

### Search Performance
- **Single model search**: <50ms for 10K images
- **Multi-embedding fusion**: <100ms for 10K images
- **Text-to-image**: <150ms (includes text encoding)

## Troubleshooting

### Issue: Model loading fails
```
Error: CLIP requires: pip install torch transformers
```
**Solution:** Install required dependencies:
```bash
pip install torch transformers ultralytics
```

### Issue: CUDA out of memory
**Solution:** Use CPU or reduce batch size:
```python
import torch
# Force CPU usage
device = torch.device('cpu')
```

### Issue: Slow embedding generation
**Solution:**
1. Use GPU if available
2. Process in batches
3. Disable models you don't need:
```python
# In ImageProcessor __init__, comment out unwanted models:
# self.embedders['yolo'] = None  # Disable YOLO
```

### Issue: Database connection error
```
Error connecting to PostgreSQL: could not connect to server
```
**Solution:** Ensure PostgreSQL is running and image_vector database exists:
```bash
# Start PostgreSQL
sudo systemctl start postgresql

# Create database and install pgvector
psql -U postgres -c "CREATE DATABASE image_vector;"
psql -U postgres -d image_vector -c "CREATE EXTENSION vector;"
```

## API Reference Updates

### ImageProcessor
- **New:** Automatically generates multiple embeddings
- **New method:** `process_image_file(file_path, callback)` - Process single image with all models

### PgVectorDatabaseManager
- **Changed:** `add_image(image_data)` - No longer takes embedding_model parameter
- **New:** `add_embedding(image_id, model_name, embedding)` - Add individual embeddings
- **Changed:** `search_images(embedding, model_name, limit)` - Now requires model_name
- **New:** `multi_embedding_search(clip_emb, yolo_emb, resnet_emb, ...)` - Fusion search
- **New:** `text_to_image_search(text_query, limit)` - Natural language search

### ImageDownloader
- **New sources:** Added 4 new AI image sources
- **New method:** `_download_from_generic_source(url)` - Generic downloader

### ImageEmbedder
- **New model:** ResNet support via `_init_resnet()` and `_embed_resnet()`

## Next Steps

1. **Backup your data** (see Step 1 above)
2. **Apply database schema** (see Step 2)
3. **Install dependencies** (see Step 3)
4. **Re-process images** if you have existing ones (see Step 4)
5. **Test the new features** (see Examples)

## Support

- Check model availability: `from core import AVAILABLE_MODELS; print(AVAILABLE_MODELS)`
- View database stats: `db_manager.get_collection_info()`
- Check system status: `system.get_system_status()`

## Rollback Procedure

If you need to revert to old system:

1. Drop the new database (optional):
```bash
psql -U postgres -c "DROP DATABASE image_vector;"
```

2. Your old vector_db database remains untouched and can still be used

3. Restore images if needed:
```bash
rm -rf ./images
mv ./images_backup ./images
```

**Note:** The new `image_vector` database is completely separate from your existing `vector_db`, so there's no conflict.
