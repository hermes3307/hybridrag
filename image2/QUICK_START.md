# Quick Start Guide - Enhanced Image Search System

## 🚀 What's New

Your image search system now has:
- ✅ **Multiple embeddings** per image (CLIP + YOLO + ResNet)
- ✅ **Diverse AI images** (faces, artwork, animals, objects)
- ✅ **Text-to-image search** ("find me a sunset")
- ✅ **Multi-model fusion** for better accuracy

## ⚡ Quick Setup

### 1. Install Dependencies
```bash
pip install torch torchvision transformers pillow numpy psycopg2-binary ultralytics
```

### 2. Setup Database
```bash
# Start PostgreSQL
sudo systemctl start postgresql

# Create database and apply schema
psql -U postgres -c "CREATE DATABASE image_vector;"
psql -U postgres -d image_vector -c "CREATE EXTENSION vector;"
psql -U postgres -d image_vector -f schema.sql
```

### 3. Run the System
```python
from core import IntegratedImageSystem

# Initialize
system = IntegratedImageSystem()
system.initialize()

# Download diverse images
system.config.download_source = 'thisartworkdoesnotexist'
system.downloader.download_image()  # Downloads AI artwork

system.config.download_source = 'thiscatdoesnotexist'
system.downloader.download_image()  # Downloads AI cats
```

## 🎯 Common Use Cases

### 1. Download and Process Images

```python
from core import IntegratedImageSystem

system = IntegratedImageSystem()
system.initialize()

# Download 10 AI-generated artworks
system.config.download_source = 'thisartworkdoesnotexist'
for i in range(10):
    file_path = system.downloader.download_image()
    if file_path:
        # Automatically generates CLIP, YOLO, ResNet embeddings
        system.processor.process_image_file(file_path)
        print(f"✓ Processed: {file_path}")
```

### 2. Search by Text (Natural Language)

```python
# Search using natural language
results = system.db_manager.text_to_image_search(
    text_query="a cat sitting on a chair",
    limit=5
)

for result in results:
    print(f"Match: {result['file_path']}")
    print(f"Distance: {result['distance']:.3f}")
```

### 3. Search by Image (Similar Images)

```python
from core import ImageEmbedder, ImageAnalyzer

# Analyze query image
query_image = "./my_query_image.jpg"
analyzer = ImageAnalyzer()
features = analyzer.analyze_image(query_image)

# Generate CLIP embedding
embedder = ImageEmbedder('clip')
query_embedding = embedder.create_embedding(query_image, features)

# Search for similar images
results = system.db_manager.search_images(
    embedding=query_embedding,
    model_name='clip',
    limit=10
)
```

### 4. Multi-Model Fusion Search (Best Accuracy)

```python
from core import ImageEmbedder, ImageAnalyzer

query_image = "./query.jpg"
analyzer = ImageAnalyzer()
features = analyzer.analyze_image(query_image)

# Generate all embeddings
clip_emb = ImageEmbedder('clip').create_embedding(query_image, features)
yolo_emb = ImageEmbedder('yolo').create_embedding(query_image, features)
resnet_emb = ImageEmbedder('resnet').create_embedding(query_image, features)

# Fusion search with custom weights
results = system.db_manager.multi_embedding_search(
    clip_emb=clip_emb,
    yolo_emb=yolo_emb,
    resnet_emb=resnet_emb,
    clip_weight=0.5,    # 50% semantic
    yolo_weight=0.25,   # 25% objects
    resnet_weight=0.25, # 25% visual features
    limit=10
)
```

## 🎨 Available Image Sources

Change the download source anytime (all from Picsum Photos - high-quality real images):

```python
# General images
system.config.download_source = 'picsum_general'    # 1024x768 general photos
system.config.download_source = 'picsum_square'     # 1024x1024 square photos
system.config.download_source = 'picsum_portrait'   # 768x1024 portrait photos

# Landscape images
system.config.download_source = 'picsum_landscape'  # 1920x1080 landscape photos (default)
system.config.download_source = 'picsum_hd'         # 2560x1440 HD landscape photos
```

**Note:** All sources use Picsum Photos, which provides random high-quality real-world images from Unsplash's collection - including landscapes, architecture, nature, people, objects, and more!

## 🔍 Search Strategies

### When to Use Each Model

| Model | Use When You Want | Example Query |
|-------|------------------|---------------|
| **CLIP** | Semantic/conceptual search | "sunset", "happy people" |
| **YOLO** | Object-based search | Images with "cars", "dogs" |
| **ResNet** | Visual similarity | Similar colors, textures, style |
| **Fusion** | Best overall results | Complex queries, high precision |

### Search Examples

```python
# Text-based (CLIP)
results = system.db_manager.text_to_image_search("red sports car")

# Object-based (YOLO)
yolo_emb = ImageEmbedder('yolo').create_embedding(query_img, features)
results = system.db_manager.search_images(yolo_emb, 'yolo', limit=10)

# Visual similarity (ResNet)
resnet_emb = ImageEmbedder('resnet').create_embedding(query_img, features)
results = system.db_manager.search_images(resnet_emb, 'resnet', limit=10)

# Best results (Fusion)
results = system.db_manager.multi_embedding_search(
    clip_emb, yolo_emb, resnet_emb, limit=10
)
```

## 🛠️ Configuration

Edit `system_config.json`:

```json
{
  "images_dir": "./images",
  "download_source": "thisartworkdoesnotexist",
  "download_delay": 1.0,
  "db_host": "localhost",
  "db_port": 5432,
  "db_name": "image_vector",
  "db_user": "postgres",
  "db_password": "postgres"
}
```

## 📊 Check System Status

```python
# Check available models
from core import AVAILABLE_MODELS
print("Available models:", AVAILABLE_MODELS)

# Get database stats
stats = system.db_manager.get_collection_info()
print(f"Total images: {stats['count']}")

# Get system status
status = system.get_system_status()
print(status)
```

## 🎓 Model Comparison

| Feature | CLIP | YOLO | ResNet | Statistical |
|---------|------|------|--------|-------------|
| **Speed** | Medium | Medium | Fast | Very Fast |
| **Accuracy** | High | High | High | Low |
| **Text Search** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Object Detection** | ❌ No | ✅ Yes | ❌ No | ❌ No |
| **Visual Features** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **GPU Recommended** | Yes | Yes | Yes | No |
| **Best For** | Text queries | Objects | Visual similarity | Fallback |

## ⚡ Performance Tips

1. **Use GPU** if available:
   ```python
   import torch
   print("CUDA available:", torch.cuda.is_available())
   ```

2. **Process in batches**:
   ```python
   system.processor.process_new_images_only(
       progress_callback=lambda c, t, m: print(f"{c}/{t}")
   )
   ```

3. **Disable unused models**:
   ```python
   # Edit core.py ImageProcessor.__init__
   # Comment out models you don't need
   ```

## 🐛 Troubleshooting

### Models not loading?
```bash
pip install --upgrade torch transformers ultralytics
```

### Database connection error?
```bash
sudo systemctl start postgresql
psql -U postgres -d image_vector -c "CREATE EXTENSION vector;"
```

### Out of memory?
```python
# Use CPU instead of GPU
import torch
torch.set_num_threads(1)
```

## 📚 Advanced Usage

### Custom Fusion Weights

Adjust weights based on your use case:

```python
# For text-heavy queries (emphasize CLIP)
results = system.db_manager.multi_embedding_search(
    clip_emb, yolo_emb, resnet_emb,
    clip_weight=0.7, yolo_weight=0.15, resnet_weight=0.15
)

# For object-focused search (emphasize YOLO)
results = system.db_manager.multi_embedding_search(
    clip_emb, yolo_emb, resnet_emb,
    clip_weight=0.3, yolo_weight=0.5, resnet_weight=0.2
)

# For visual similarity (emphasize ResNet)
results = system.db_manager.multi_embedding_search(
    clip_emb, yolo_emb, resnet_emb,
    clip_weight=0.2, yolo_weight=0.2, resnet_weight=0.6
)
```

### Batch Processing

```python
# Process only new images (skip existing)
stats = system.processor.process_new_images_only(
    progress_callback=lambda cur, total, msg: print(f"Progress: {cur}/{total}")
)
print(f"Processed: {stats['processed']}, Errors: {stats['errors']}")
```

## 🎉 That's It!

You're ready to use the enhanced multi-embedding image search system with diverse AI-generated images!

**Next Steps:**
- Read [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md) for detailed migration info
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Browse [CODE_REFERENCE.md](CODE_REFERENCE.md) for API details
