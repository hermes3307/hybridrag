# CODE REFERENCE - DETAILED IMPLEMENTATION GUIDE

## 1. DATA MODELS & CLASSES

### 1.1 ImageData (core.py, lines 102-113)
```python
@dataclass
class ImageData:
    """Data class for image information"""
    image_id: str
    file_path: str
    features: Dict[str, Any]
    embedding: Optional[List[float]]
    timestamp: str
    image_hash: str
```
**Used by**: ImageProcessor when storing images in database

---

### 1.2 SystemConfig (core.py, lines 116-154)
```python
@dataclass
class SystemConfig:
    images_dir: str = "./images"
    
    # PostgreSQL + pgvector settings
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "vector_db"
    db_user: str = "postgres"
    db_password: str = "postgres"
    
    # Application settings
    download_delay: float = 1.0
    max_workers: int = 2
    batch_size: int = 50
    embedding_model: str = "statistical"  # KEY: selects embedding
    download_source: str = "thispersondoesnotexist"
```
**Key Field**: `embedding_model` - controls which embedding is created
**Loaded from**: system_config.json or created with defaults

---

### 1.3 SystemStats (core.py, lines 155-293)
Thread-safe statistics tracking. Key methods:
- `increment_download_success(elapsed_time)` - Track download metrics
- `increment_embed_success(elapsed_time)` - Track embedding metrics
- `get_stats()` - Get comprehensive statistics dict

---

## 2. MAIN PROCESSING CLASSES

### 2.1 ImageAnalyzer - Feature Extraction

**Initialization**:
```python
analyzer = ImageAnalyzer()
```

**Main Method**:
```python
def analyze_image(self, image_path: str) -> Dict[str, Any]:
    """
    Returns dict with:
    - Image properties: width, height, format, mode
    - EXIF data: camera, GPS, dates
    - File metadata: size, timestamps
    - Quality metrics: sharpness, noise, edge density
    - Color analysis: brightness, contrast, saturation
    """
```

**Output Example**:
```python
{
    'width': 1024,
    'height': 1024,
    'mode': 'RGB',
    'brightness': 128.5,
    'contrast': 45.2,
    'sharpness_score': 234.5,
    'noise_level': 12.3,
    'edge_density': 0.15,
    'megapixels': 1.05,
    'aspect_ratio': 1.0,
    'has_exif': True,
    'gps_latitude': 37.7749,
    'gps_longitude': -122.4194,
    # ... many more fields
}
```

---

### 2.2 ImageEmbedder - Vector Generation

**Initialization with model selection**:
```python
# Use default statistical model
embedder = ImageEmbedder()

# Use specific model
embedder = ImageEmbedder(model_name="clip")  # or "yolo", "action"
```

**Embedding Creation**:
```python
def create_embedding(self, image_path: str, features: Dict[str, Any]) -> List[float]:
    """
    Returns float list of dimensions depending on model:
    - Statistical: 512 dimensions
    - CLIP: 512 dimensions
    - YOLO: 80 dimensions
    - Action: 15 dimensions
    """
```

**Model-Specific Methods**:
```python
def _embed_clip(self, image_path: str) -> List[float]:
    # Vision-language embedding using CLIPModel
    
def _embed_yolo(self, image_path: str) -> List[float]:
    # Object detection: counts objects per COCO class (80 total)
    
def _embed_action(self, image_path: str) -> List[float]:
    # Action recognition: one-hot encoded 15 actions
    
def _create_statistical_embedding(self, img_array: np.ndarray, features: Dict) -> np.ndarray:
    # Pixel statistics: mean, std, histogram per RGB channel
    # Input image as numpy array, features dict with brightness/contrast
    # Returns normalized 512D vector
```

---

### 2.3 ImageDownloader - Image Acquisition

**Initialization**:
```python
downloader = ImageDownloader(config, stats)
```

**Download Single Image**:
```python
def download_image(self) -> Optional[str]:
    """
    Returns: file_path of downloaded image or None on failure
    
    Process:
    1. Load existing hashes (duplicate detection)
    2. Download image from source
    3. Calculate MD5 hash
    4. Check if already downloaded
    5. Save to images/ directory
    6. Analyze and create metadata
    7. Track statistics
    """
```

**Download Sources**:
```python
DOWNLOAD_SOURCES = {
    'thispersondoesnotexist': {
        'name': 'ThisPersonDoesNotExist.com',
        'url': 'https://thispersondoesnotexist.com/',
    },
    '100k-faces': {
        'name': '100K AI Faces',
        'url': 'https://100k-faces.vercel.app/api/random-image',
    }
}
```

**Background Operations**:
```python
# Load hashes in background thread
downloader.start_background_hash_loading(
    progress_callback=lambda current, total, msg: print(f"{current}/{total}: {msg}"),
    completion_callback=lambda count, elapsed: print(f"Loaded {count} in {elapsed}s")
)

# Start continuous download loop
downloader.start_download_loop(callback=on_image_downloaded)

# Stop the loop
downloader.stop_download_loop()
```

---

### 2.4 ImageProcessor - Batch Processing

**Initialization**:
```python
processor = ImageProcessor(config, stats, db_manager)
```

**Key Methods**:
```python
def get_new_files_only(self) -> List[str]:
    """Returns list of unprocessed image files"""
    
def process_new_images_only(self, callback=None, progress_callback=None) -> Dict[str, int]:
    """
    Process images not in database
    
    Returns:
    {
        'total_files': 100,
        'processed': 95,
        'skipped': 0,
        'errors': 5
    }
    """

def process_all_images(self, callback=None, progress_callback=None):
    """Process all images in images/ directory"""
```

**Callback Usage**:
```python
def on_image_processed(image_data):
    print(f"Processed: {image_data.image_id}")

def on_progress(current, total, message):
    print(f"Progress: {current}/{total} - {message}")

processor.process_new_images_only(
    callback=on_image_processed,
    progress_callback=on_progress
)
```

---

### 2.5 IntegratedImageSystem - Main Orchestrator

**Initialization & Setup**:
```python
# Create system
system = IntegratedImageSystem(config_file="system_config.json")

# Initialize (connects to database)
if system.initialize():
    print("System ready")
else:
    print("Failed to initialize")

# Access components
downloader = system.downloader       # ImageDownloader instance
processor = system.processor         # ImageProcessor instance
db = system.db_manager              # PgVectorDatabaseManager instance
stats = system.stats                # SystemStats instance
```

**Getting System Status**:
```python
status = system.get_system_status()
# Returns:
# {
#     'database': {'count': 1000},
#     'statistics': {
#         'download_attempts': 1500,
#         'download_success': 1200,
#         'embed_success': 1100,
#         'search_queries': 250
#     },
#     'config': {...config fields...},
#     'images_directory': './images',
#     'images_count': 1200
# }
```

---

## 3. DATABASE OPERATIONS

### 3.1 PgVectorDatabaseManager (pgvector_db.py)

**Connection**:
```python
db = PgVectorDatabaseManager(config)
if db.initialize():
    conn = db.get_connection()
else:
    print("Connection failed")

db.close()
```

**Adding Images**:
```python
def add_image(self, image_data, embedding_model):
    """
    Args:
        image_data: ImageData instance with embedding, features, etc.
        embedding_model: "statistical", "clip", "yolo", "action"
    
    Returns: True on success, False on failure
    """
```

**Searching by Vector**:
```python
def search_images(self, embedding, limit):
    """
    Args:
        embedding: List[float] - vector embedding (must match model dimension)
        limit: int - max results to return
    
    Returns: List[Dict] with keys: image_id, file_path, distance, metadata
    """
```

**Metadata Search**:
```python
def search_by_metadata(self, metadata_filter, limit):
    """
    Args:
        metadata_filter: Dict with filters like {'brightness_level': 'bright'}
        limit: int
    
    Returns: List[Dict] matching metadata criteria
    """
```

**Hybrid Search**:
```python
def hybrid_search(self, embedding, metadata_filter, limit):
    """Vector search + metadata filter combined"""
```

**Mixed Model Search** (EXPERIMENTAL):
```python
def mixed_search(self, clip_embedding, yolo_embedding, action_embedding, limit):
    """
    Attempts weighted combination of multiple model embeddings
    Current implementation uses fixed weights: 0.33 each
    
    NOTE: This is slow because embeddings aren't pre-stored per-model
    """
```

---

## 4. DATABASE SCHEMA

### 4.1 Images Table Structure

```sql
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255) UNIQUE NOT NULL,          -- Timestamp identifier
    file_path TEXT NOT NULL,                         -- Local file path
    timestamp TIMESTAMP NOT NULL,                    -- Download/process time
    image_hash VARCHAR(64) NOT NULL,                 -- MD5 for dedup
    embedding_model VARCHAR(50) NOT NULL,           -- Which model created this
    
    embedding vector(512),                          -- THE KEY COLUMN
                                                    -- Stores vector embedding
                                                    -- Fixed 512 dimensions
    
    brightness FLOAT,                               -- Quality metrics
    contrast FLOAT,
    sharpness FLOAT,
    
    metadata JSONB,                                 -- Flexible storage
                                                    -- Contains all analysis results
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- CRITICAL INDEXES
CREATE INDEX idx_embedding_hnsw_cosine ON images
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
-- HNSW: Hierarchical Navigable Small World
-- Fast approximate nearest neighbor search
-- m=16: more connections = better recall
-- ef_construction=64: higher = better index quality but slower build
```

### 4.2 Search Function

```sql
CREATE OR REPLACE FUNCTION search_similar_images(
    query_embedding vector(512),
    limit_count INTEGER DEFAULT 10,
    distance_threshold FLOAT DEFAULT 1.0
)
RETURNS TABLE (
    image_id VARCHAR(255),
    file_path TEXT,
    distance FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.image_id,
        i.file_path,
        i.embedding <=> query_embedding AS distance,    -- Cosine distance
        i.metadata
    FROM images i
    WHERE i.embedding IS NOT NULL
        AND i.embedding <=> query_embedding < distance_threshold
    ORDER BY i.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;
```

---

## 5. GUI COMPONENTS (image.py)

### 5.1 Tab Structure

```python
class IntegratedImageGUI:
    def __init__(self):
        self.notebook = ttk.Notebook(...)
        
        # Tab 0: Overview
        self.overview_frame = ttk.Frame(...)
        self.notebook.add(self.overview_frame, text="System Overview")
        self.create_overview_tab()
        
        # Tab 1: Download
        self.download_frame = ttk.Frame(...)
        self.notebook.add(self.download_frame, text="Download Images")
        self.create_download_tab()
        
        # Tab 2: Process
        self.process_frame = ttk.Frame(...)
        self.notebook.add(self.process_frame, text="Process & Embed")
        self.create_process_tab()
        
        # Tab 3: Search
        self.search_frame = ttk.Frame(...)
        self.notebook.add(self.search_frame, text="Search Images")
        self.create_search_tab()
        
        # Tab 4: Configuration
        self.config_frame = ttk.Frame(...)
        self.notebook.add(self.config_frame, text="Configuration")
        self.create_config_tab()
```

### 5.2 Search Implementation (image.py, lines 1980-2082)

**Search Mode Selection**:
```python
search_mode = self.search_mode_var.get()
# Options: "metadata", "vector", "hybrid", "mixed"
```

**Vector Search Path**:
```python
# 1. Get image to search
image_path = self.search_image_var.get()

# 2. Create embedding with CURRENT configured model
analyzer = ImageAnalyzer()
embedder = ImageEmbedder(model_name=self.system.config.embedding_model)
features = analyzer.analyze_image(image_path)
embedding = embedder.create_embedding(image_path, features)

# 3. Search database
results = self.system.db_manager.search_images(embedding, num_results)

# 4. Display results
self.display_search_results(results)
```

**Mixed Search Path** (SLOW - on-the-fly embeddings):
```python
# Create embeddings for ALL models at query time
clip_embedder = ImageEmbedder(model_name="clip")
yolo_embedder = ImageEmbedder(model_name="yolo")
action_embedder = ImageEmbedder(model_name="action")

clip_embedding = clip_embedder.create_embedding(image_path, features)
yolo_embedding = yolo_embedder.create_embedding(image_path, features)
action_embedding = action_embedder.create_embedding(image_path, features)

# Search with weighted combination (NOT pre-computed - SLOW!)
results = self.system.db_manager.mixed_search(
    clip_embedding, yolo_embedding, action_embedding, limit
)
```

---

## 6. CONFIGURATION & ENVIRONMENT

### 6.1 System Configuration File

**File**: `system_config.json`
```json
{
  "images_dir": "./images",
  "db_host": "localhost",
  "db_port": 5432,
  "db_name": "vector_db",
  "db_user": "postgres",
  "db_password": "postgres",
  "download_delay": 1.0,
  "max_workers": 2,
  "batch_size": 50,
  "embedding_model": "statistical",
  "download_source": "thispersondoesnotexist",
  "config_file": "system_config.json"
}
```

### 6.2 Environment Variables

**File**: `.env`
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_images
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
DB_TYPE=pgvector
EMBEDDING_MODEL=statistical
IMAGES_DIR=./images
DB_MIN_CONNECTIONS=1
DB_MAX_CONNECTIONS=10
DEFAULT_SEARCH_LIMIT=10
DISTANCE_METRIC=cosine
```

---

## 7. IMPORTANT CODE PATTERNS

### 7.1 Model Selection Pattern

```python
# In core.py - ImageEmbedder initialization
def __init__(self, model_name: str = "statistical"):
    self.model_name = model_name
    self._initialize_model()  # Loads model based on name

# Usage:
embedder_clip = ImageEmbedder(model_name="clip")
embedder_yolo = ImageEmbedder(model_name="yolo")
embedder_stat = ImageEmbedder()  # Defaults to statistical
```

### 7.2 Error Handling & Fallback

```python
# If model loading fails, falls back to statistical
def _initialize_model(self):
    try:
        if self.model_name == "clip":
            self._init_clip()
        # ... other models
    except Exception as e:
        logger.error(f"Failed to initialize {self.model_name}")
        self.model_name = "statistical"  # Fallback
        self.embedding_size = 512
```

### 7.3 Thread Safety

```python
# All statistics use thread locks
class SystemStats:
    def __init__(self):
        self.lock = threading.Lock()
    
    def increment_download_success(self):
        with self.lock:
            self.download_success += 1
```

### 7.4 Callback Pattern for Progress

```python
# Generic callback pattern used throughout
def process_images(progress_callback=None):
    for idx, file_path in enumerate(files, 1):
        if progress_callback:
            progress_callback(idx, total, f"Processing {idx}/{total}")
        # ... do work ...

# Usage:
def on_progress(current, total, message):
    print(f"{current}/{total}: {message}")

processor.process_new_images_only(progress_callback=on_progress)
```

---

## 8. CRITICAL LIMITATIONS & ISSUES

### 8.1 Missing process_image_file() Method

**Problem**: ImageProcessor calls `self.process_image_file()` but it's not defined.
**Location**: core.py lines 1285, 1316 call the method but definition is missing
**Impact**: Critical - actual image processing won't work
**Fix Required**: Implement this method to:
1. Analyze image with ImageAnalyzer
2. Create embedding with ImageEmbedder
3. Build ImageData object
4. Store in database with db_manager.add_image()

### 8.2 Single-Model Database Limitation

**Problem**: Schema has only ONE `embedding vector(512)` column
**Impact**: Can't store multiple embeddings per image
**Current Workaround**: "Mixed" search creates embeddings on-the-fly (SLOW)
**Fix Required**: 
- Add separate columns: embedding_clip, embedding_yolo, embedding_action
- Update add_image() to accept multiple embeddings
- Update search methods to select appropriate column

### 8.3 Model Mismatch Handling

**Problem**: If database has embeddings from Model A, can't search with Model B
**Check**: `db_manager.check_embedding_model_mismatch()`
**Current Solution**: Prevent search, ask user to re-embed all data
**Issue**: Re-embedding is slow for large datasets

---

## 9. DATA FLOW DIAGRAMS

### Download Flow
```
User clicks "Download"
    → ImageDownloader.download_image()
        → Load hashes (duplicate detection)
        → HTTP request to source
        → Calculate MD5 hash
        → Check if duplicate
        → Save to images/
        → ImageAnalyzer.analyze_image()
        → Build metadata JSON
        → Return file_path
    → GUI updates preview + stats
```

### Processing Flow
```
User clicks "Process & Embed"
    → ImageProcessor.process_new_images_only()
        → For each new file:
            → ImageAnalyzer.analyze_image()
            → ImageEmbedder.create_embedding(model_name)
            → Build ImageData object
            → db_manager.add_image()
                → INSERT INTO images (...)
                → INSERT embedding vector
            → Stats tracking
        → Return summary stats
    → GUI updates progress bar + results
```

### Search Flow
```
User selects image + clicks "Search"
    → image.py: search_images()
        → Check model mismatch
        → ImageAnalyzer.analyze_image(query_image)
        → ImageEmbedder.create_embedding(query_image)
        → db_manager.search_images(embedding)
            → SQL: SELECT * FROM search_similar_images(...)
            → HNSW index finds nearest neighbors
            → ORDER BY distance
            → LIMIT results
        → db_manager returns: List[Dict] with distances
        → GUI displays results with thumbnails
```

---

## 10. TESTING & DEBUGGING

### Check Embedding Models Availability
```python
from core import check_embedding_models
status = check_embedding_models()
# Returns: {'clip': True, 'yolo': False, 'action': True, 'statistical': True}
```

### Test Database Connection
```python
from core import IntegratedImageSystem
system = IntegratedImageSystem()
if system.initialize():
    status = system.get_system_status()
    print(f"Connected. DB has {status['database']['count']} images")
else:
    print("Connection failed")
```

### Test Image Embedding
```python
from core import ImageAnalyzer, ImageEmbedder
analyzer = ImageAnalyzer()
embedder = ImageEmbedder(model_name="clip")

features = analyzer.analyze_image("test.jpg")
embedding = embedder.create_embedding("test.jpg", features)
print(f"Embedding shape: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
```

---

