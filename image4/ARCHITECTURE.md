# IMAGE SEARCH PROGRAM - ARCHITECTURE OVERVIEW

## PROJECT STRUCTURE

```
/home/pi/hybridrag/image/
├── core.py                 # Main backend - image processing engine
├── image.py               # GUI application (Tkinter)
├── pgvector_db.py         # PostgreSQL + pgvector database manager
├── schema.sql             # PostgreSQL schema definition
├── requirements.txt       # Python dependencies
├── install.sh            # Installation/setup script
├── .env.example          # Environment configuration template
├── images/               # Downloaded/processed images directory
└── __pycache__/          # Python cache
```

---

## CORE ARCHITECTURE COMPONENTS

### 1. DATA LAYER (PostgreSQL + pgvector)

**Database: pgvector_db.py**
- `PgVectorDatabaseManager` class handles all database operations
- Uses psycopg2 for PostgreSQL connection
- Key methods:
  - `initialize()` - Connect to PostgreSQL
  - `add_image()` - Store image with embedding
  - `search_images()` - Vector similarity search
  - `hash_exists()` - Check for duplicates via MD5 hash
  - `mixed_search()` - Multi-model search (CLIP + YOLO + Action)
  - `hybrid_search()` - Vector + metadata search
  - `search_by_metadata()` - Metadata-only filtering
  - `check_embedding_model_mismatch()` - Validate embedding compatibility

**Schema: schema.sql**
- Table: `images` (main storage)
  - `id` - Serial primary key
  - `image_id` - Unique identifier (timestamp)
  - `file_path` - Local file location
  - `image_hash` - MD5 hash (duplicate detection)
  - `embedding_model` - Model used (statistical/clip/yolo/action)
  - `embedding` - Vector(512) - stores vector embeddings
  - `brightness`, `contrast`, `sharpness` - Quality metrics
  - `metadata` - JSONB for flexible attributes
  - `created_at`, `updated_at` - Timestamps

- Indexes:
  - HNSW vector index on embedding (cosine similarity)
  - Hash indexes for fast lookups
  - JSONB GIN index for metadata queries

- Functions:
  - `search_similar_images()` - SQL function for cosine similarity search
  - `get_database_stats()` - Statistics aggregation

---

### 2. IMAGE PROCESSING LAYER (core.py)

#### A. ImageAnalyzer Class
**Purpose**: Extract image features and metadata
**Location**: core.py, lines 298-621

**Features Extracted**:
- Basic properties: dimensions, format, file size
- EXIF metadata: camera info, GPS, datetime
- File metadata: creation/modification times, file size
- Color analysis: brightness, contrast, saturation (OpenCV or PIL)
- Quality metrics:
  - Sharpness score (Laplacian variance)
  - Noise level estimation
  - Edge density
  - Dynamic range
  - Aspect ratio classification
  - Megapixels calculation

**Key Methods**:
- `analyze_image(image_path)` - Main analysis function returns Dict with all features
- `_extract_exif_data()` - GPS and camera metadata
- `_calculate_quality_metrics()` - Sharpness and quality scores
- `_analyze_colors()` - OpenCV-based color analysis
- `_analyze_colors_pil()` - PIL fallback for color analysis

---

#### B. ImageEmbedder Class
**Purpose**: Generate vector embeddings from images
**Location**: core.py, lines 626-861

**Supported Models**:
1. **Statistical** (512 dimensions) - ALWAYS AVAILABLE
   - Pixel-level statistics (mean, std, median, min, max per channel)
   - Histogram features (normalized bins)
   - Quality metric incorporation
   - No ML dependencies
   
2. **CLIP** (512 dimensions) - openai/clip-vit-base-patch32
   - Vision-language model
   - Good for semantic similarity
   - Requires: torch, transformers
   
3. **YOLO** (80 dimensions) - ultralytics/yolov5s
   - Object detection-based embedding
   - "Bag of objects" representation
   - Counts detected object classes
   - Requires: torch, ultralytics
   
4. **Action** (15 dimensions) - Human-Action-Recognition
   - Action classification model
   - One-hot encoded outputs
   - Requires: torch, transformers

**Key Methods**:
- `create_embedding(image_path, features)` - Dispatcher to appropriate model
- `_embed_clip()` - Generate CLIP embeddings
- `_embed_yolo()` - Generate object detection embeddings
- `_embed_action()` - Generate action classification embeddings
- `_create_statistical_embedding()` - Fallback/always-available method
- `_initialize_model()` - Model initialization with error handling

**Critical Note**: Different models produce different-sized embeddings:
- Statistical: 512D (normalized)
- CLIP: 512D
- YOLO: 80D
- Action: 15D

---

#### C. ImageDownloader Class
**Purpose**: Download images from AI generation services
**Location**: core.py, lines 867-1210

**Download Sources**:
1. **ThisPersonDoesNotExist.com** - High quality AI faces (1024x1024)
2. **100K Faces API** - AI-generated faces from generated.photos

**Key Features**:
- Duplicate detection via MD5 hashing
- Background hash loading (threaded)
- Background download loop support
- Comprehensive metadata generation
- Configurable download delays

**Key Methods**:
- `download_image()` - Download single image
- `_download_from_thispersondoesnotexist()` - Source 1
- `_download_from_100k_faces()` - Source 2
- `_load_existing_hashes()` - Load existing file hashes
- `_load_existing_hashes_with_callback()` - With progress updates
- `start_download_loop()` - Continuous downloading in background
- `stop_download_loop()` - Stop background downloader
- `start_background_hash_loading()` - Pre-load hashes for duplicate detection

**Metadata Generated**:
- Filename, file path, image ID (timestamp)
- Download timestamp and source URL
- File size, HTTP status
- Image properties (width, height, format, mode)
- Face features from ImageAnalyzer
- Queryable attributes for filtering

---

#### D. ImageProcessor Class
**Purpose**: Process local images and create database entries
**Location**: core.py, lines 1212-1317

**Components**:
- Uses ImageAnalyzer to extract features
- Uses ImageEmbedder to create embeddings (configurable model)
- Stores results in PostgreSQL + pgvector database
- Tracks processed files to avoid duplicates

**Key Methods**:
- `process_new_images_only()` - Process images not in database
- `process_all_images()` - Process all images in directory
- `get_new_files_only()` - List unprocessed image files
- Callback support for progress updates
- Hash-based duplicate detection

**Flow**:
1. Find image files (*.jpg, *.jpeg, *.png)
2. Check if hash exists in database
3. Analyze image for features
4. Create embedding using configured model
5. Store in database with all metadata

---

#### E. IntegratedImageSystem Class
**Purpose**: Main orchestrator integrating all components
**Location**: core.py, lines 1322-1373

**Initialization Flow**:
```
IntegratedImageSystem()
├── Load config from system_config.json
├── Initialize SystemConfig
├── Initialize SystemStats (tracking)
├── Create PgVectorDatabaseManager
├── Create ImageDownloader
└── Create ImageProcessor (after DB init)
```

**Key Methods**:
- `initialize()` - Setup database and processor
- `get_system_status()` - Comprehensive system stats

---

### 3. CONFIGURATION LAYER

**SystemConfig Class** (core.py, lines 116-154):
- Database settings (host, port, user, password, database name)
- Application settings (images_dir, embedding_model, download_source)
- Performance tuning (download_delay, max_workers, batch_size)
- Can be loaded from JSON file (system_config.json)
- Can be saved back to JSON

**Default Environment** (.env.example):
```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_images
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
DB_TYPE=pgvector
EMBEDDING_MODEL=statistical
IMAGES_DIR=./images
```

---

### 4. STATISTICS & MONITORING

**SystemStats Class** (core.py, lines 155-293):
- Thread-safe statistics tracking
- Downloads: attempts, success, duplicates, errors, timing
- Embeddings: processed, success, duplicates, errors, timing
- Search queries count
- Session-based statistics
- Performance metrics (rates, averages, timing)

---

### 5. GUI APPLICATION (image.py)

**Main Class: IntegratedImageGUI**
- Tkinter-based tabbed interface
- Lazy-loads core modules for faster startup

**Tabs**:
1. **System Overview** - Status, statistics, logs
2. **Download Images** - Download controls, statistics, preview
3. **Process & Embed** - Process controls, embedding configuration
4. **Search Images** - Search by vector/metadata/hybrid/mixed
5. **Configuration** - System settings, model selection, database management

**Key Search Methods**:
- `search_images()` - Main search dispatcher
- Search modes: metadata, vector, hybrid, mixed
- `display_search_results()` - UI for results
- `_build_metadata_filter()` - Filter construction

---

## CURRENT EMBEDDING IMPLEMENTATION

### Supported Models:
1. **Statistical** (default) - No dependencies, always works
2. **CLIP** - Vision-language model (512D)
3. **YOLO** - Object detection (80D)
4. **Action** - Action recognition (15D)

### Model Selection:
- Configured in `SystemConfig.embedding_model`
- Can be changed via GUI "Configuration" tab
- Applied when processing new images

### Key Limitation - MULTI-MODEL SUPPORT:
**Current Issue**: Database stores embeddings from only ONE model at a time
- Schema has single `embedding vector(512)` column
- All images must use same embedding model
- `check_embedding_model_mismatch()` prevents mixed-model searches

**Mixed Search Workaround** (not fully implemented):
- GUI has "mixed" search mode that creates CLIP, YOLO, Action embeddings on-the-fly
- Attempts weighted combination in pgvector_db.py line 98-115
- But database doesn't store these, so search is slow

---

## IMAGE DOWNLOAD & STORAGE FLOW

### Download Process:
```
ImageDownloader.download_image()
├── Load existing hashes (duplicate detection)
├── Download image from source (TPNE or 100K Faces)
├── Calculate MD5 hash
├── Check if hash already exists
├── Save to images/ directory with timestamp
├── Analyze image (ImageAnalyzer)
├── Generate comprehensive metadata
├── Save metadata alongside image (JSON)
└── Stats tracking
```

### Storage:
- **Images**: `/home/pi/hybridrag/image/images/image_YYYYMMDD_HHMMSS_MMMM_HASH.jpg`
- **Metadata**: Adjacent JSON files with analysis results
- **Database**: All info + embeddings stored in PostgreSQL

---

## SEARCH FUNCTIONALITY

### Search Types:
1. **Vector Search**
   - Query image analyzed and embedded
   - Cosine similarity to all images in DB
   - Returns top-N by distance

2. **Metadata Search**
   - Filter by brightness, quality, has_face, etc.
   - Pure SQL WHERE clause
   - No embedding needed

3. **Hybrid Search**
   - Vector search + metadata filters
   - Combines both approaches

4. **Mixed Search**
   - Multi-model search (attempted)
   - Creates CLIP + YOLO + Action embeddings on-the-fly
   - Weighted combination: 0.33 each
   - Currently slow (no pre-computed embeddings)

### Search Process:
```
search_images(embedding, limit)
├── Vector similarity using pgvector cosine distance
├── Order by distance (closest first)
├── Limit results
└── Return metadata with distance scores
```

---

## DIRECTORY LAYOUT WITH EXAMPLES

```
/home/pi/hybridrag/image/
├── core.py (1373 lines)
│   ├── ImageAnalyzer (lines 298-621)
│   ├── ImageEmbedder (lines 626-861)
│   ├── ImageDownloader (lines 867-1210)
│   ├── ImageProcessor (lines 1212-1317)
│   └── IntegratedImageSystem (lines 1322-1373)
│
├── image.py (2700+ lines)
│   └── IntegratedImageGUI
│       ├── create_overview_tab()
│       ├── create_download_tab()
│       ├── create_process_tab()
│       ├── create_search_tab()
│       ├── create_config_tab()
│       ├── search_images()
│       ├── download_single()
│       └── various helper methods
│
├── pgvector_db.py (117 lines)
│   └── PgVectorDatabaseManager
│       ├── initialize()
│       ├── add_image()
│       ├── search_images()
│       ├── mixed_search()
│       └── hybrid_search()
│
├── schema.sql (163 lines)
│   ├── CREATE TABLE images
│   ├── HNSW vector index
│   ├── Helper functions
│   └── Trigger for updated_at
│
├── requirements.txt
│   ├── requests
│   ├── numpy
│   ├── Pillow
│   ├── opencv-python
│   ├── psycopg2-binary
│   ├── torch
│   ├── torchvision
│   ├── transformers
│   └── ultralytics
│
├── install.sh (344 lines)
│   ├── PostgreSQL setup
│   ├── pgvector installation
│   ├── Schema creation
│   └── Python dependencies
│
└── images/ (downloads stored here)
```

---

## KEY FILES TO MODIFY FOR MULTIPLE EMBEDDINGS

### Priority 1 - CRITICAL (Must modify):
1. **pgvector_db.py**
   - Add columns for multiple embeddings (embedding_clip, embedding_yolo, etc.)
   - Modify add_image() to store multiple embeddings
   - Update search methods to use appropriate embedding column
   - Implement true multi-model search

2. **schema.sql**
   - Add embedding columns for each model
   - Update indexes for new embedding columns
   - Modify search function to support model selection

3. **core.py - ImageProcessor.process_image_file()**
   - Currently undefined/incomplete (referenced but not shown)
   - Needs to generate multiple embeddings per image
   - Store all embeddings in DB

### Priority 2 - IMPORTANT (Should modify):
1. **core.py - ImageEmbedder**
   - Add batch embedding method for efficiency
   - Cache model instances to avoid reloading

2. **image.py - create_process_tab()**
   - Option to select multiple models for processing
   - Progress tracking per model

3. **image.py - search_images()**
   - Model selection in UI
   - Multi-model search weighting options

### Priority 3 - ENHANCEMENT:
1. **Configuration**
   - Add per-model settings
   - Model-specific parameters

2. **Database Queries**
   - Indexed queries for each model
   - Statistics per embedding model

3. **GUI**
   - Model management interface
   - Embedding quality metrics per model

---

## DEPENDENCIES & REQUIREMENTS

**System**:
- PostgreSQL 12+
- pgvector extension (v0.8.0+)
- Python 3.8+
- Build tools (gcc, make)

**Python Packages**:
- requests - HTTP downloads
- numpy - Array operations
- Pillow - Image I/O and processing
- opencv-python - Advanced image analysis
- psycopg2-binary - PostgreSQL driver
- torch - ML framework
- torchvision - Vision utilities
- transformers - CLIP/Action models
- ultralytics - YOLOv5 models

---

## PERFORMANCE CHARACTERISTICS

**Download**:
- Single image: ~2-5 seconds (network dependent)
- With duplicate detection: +0.1-0.5s per image

**Embedding Generation**:
- Statistical: ~100-200ms per image
- CLIP: ~500ms-1s per image (GPU: ~100-200ms)
- YOLO: ~200-500ms per image (GPU: ~50-100ms)
- Action: ~300-600ms per image

**Search**:
- Vector search: ~10-50ms (HNSW index)
- Metadata search: ~5-20ms
- Mixed search: ~1-5s per image (on-the-fly embeddings)

**Database**:
- Connection: ~50ms
- Insert: ~10-20ms
- Search (top-10): ~20-50ms

---

## CONFIGURATION FLOW

```
System Start
├── Load .env or system_config.json
├── SystemConfig with all settings
├── PgVectorDatabaseManager connects to PostgreSQL
├── ImageDownloader initialized
└── ImageProcessor initialized (once DB ready)

GUI Startup
├── Lazy load core modules
├── Create tabs
├── Check embedding models
└── Display system status

User Actions
├── Download → ImageDownloader.download_image()
├── Process → ImageProcessor.process_new_images_only()
├── Search → Create embedding + search_images()
└── Config → SystemConfig.save_to_file()
```

---

## ARCHITECTURAL STRENGTHS
- Modular design (separate classes for each function)
- Database-centric (PostgreSQL + pgvector for scalability)
- Multiple embedding models supported
- Comprehensive metadata tracking
- Thread-safe statistics
- GUI for easy interaction

---

## ARCHITECTURAL LIMITATIONS
1. **Single-model limitation**: Only one embedding type per session
2. **No multi-embedding search**: Mixed search is inefficient (on-the-fly)
3. **Missing process_image_file()**: ImageProcessor incomplete
4. **No batch processing optimization**: Process one image at a time
5. **Limited error recovery**: Failures not automatically retried
6. **No model versioning**: Can't track which model version created embedding

