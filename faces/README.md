# 🎭 Face Recognition System - Complete Guide

## Quick Start (3 Steps)

```bash
# 1. Setup database
python 1_setup_database.py

# 2. Download and embed faces
python 4_download_faces_gui.py    # Download faces with GUI
python 6_embed_faces_gui.py       # Embed with metadata

# 3. Search faces
python 7_search_faces_gui.py      # Search with unified interface
```

## 🚀 Easy Access - Launcher Menu

```bash
python 0_launcher.py
```

The launcher provides a menu to access all components (1-10).

## 📋 System Components (0-10)

### 0️⃣  Main Launcher
**File:** `0_launcher.py`
**Purpose:** Interactive menu to launch any component
**Usage:** `python 0_launcher.py`

---

### 1️⃣  Setup ChromaDB Database
**File:** `1_setup_database.py`
**Purpose:** Install ChromaDB and initialize database
**Usage:** `python 1_setup_database.py`
**Run:** Once during initial setup

**What it does:**
- Installs ChromaDB package
- Creates database directory
- Initializes collections
- Verifies installation

---

### 2️⃣  Database Information & Stats
**File:** `2_database_info.py`
**Purpose:** View database statistics and collection info
**Usage:** `python 2_database_info.py`
**Run:** Anytime to check database status

**Shows:**
- Total collections
- Number of vectors per collection
- Vector dimensions
- Storage size
- Face feature distributions (age, skin tone, quality)
- Sample data

---

### 3️⃣  Download Faces (CLI)
**File:** `3_download_faces.py`
**Purpose:** Download face images with metadata - Command Line
**Usage:** `python 3_download_faces.py [options]`

**Options:**
```bash
--count 100           # Download 100 faces
--faces-dir ./faces   # Output directory
--delay 0.001         # Delay between downloads (seconds)
--max-workers 100     # Parallel downloads
```

**Features:**
- Downloads from ThisPersonDoesNotExist
- Saves images with unique hash-based names
- Creates JSON metadata for each image
- Duplicate detection
- Progress tracking

---

### 4️⃣  Download Faces (GUI)
**File:** `4_download_faces_gui.py`
**Purpose:** Download face images with metadata - Graphical Interface
**Usage:** `python 4_download_faces_gui.py`

**Features:**
- Visual progress tracking
- Real-time statistics
- Pause/resume downloads
- Duplicate detection
- Metadata preview
- Easy configuration

**GUI Sections:**
- Download controls (count, delay, workers)
- Progress bar and statistics
- Downloaded files list
- Error logging

---

### 5️⃣  Embed Faces into Vector DB (CLI)
**File:** `5_embed_faces.py`
**Purpose:** Process images and create vector embeddings - Command Line
**Usage:** `python 5_embed_faces.py [options]`

**Options:**
```bash
--faces-dir ./faces   # Input directory
--batch-size 50       # Process in batches
--max-workers 4       # Parallel processing
--clear               # Clear existing embeddings
```

**Process:**
1. Scans faces directory
2. Extracts features (age, skin tone, quality, brightness)
3. Loads JSON metadata
4. Generates embeddings
5. Stores in vector database

**Features Extracted:**
- Age group (young_adult, adult, mature_adult)
- Skin tone (light, medium, dark)
- Image quality (high, medium, low)
- Brightness statistics
- Color analysis

---

### 6️⃣  Embed Faces into Vector DB (GUI)
**File:** `6_embed_faces_gui.py`
**Purpose:** Process images and create embeddings - Graphical Interface
**Usage:** `python 6_embed_faces_gui.py`

**Features:**
- Visual progress tracking
- Real-time statistics
- Metadata loading status
- Feature extraction preview
- Database collection management
- Completion statistics

**GUI Sections:**
- Configuration (directory, batch size, workers)
- Directory information
- Embedding statistics
- Processing log
- Database info button

---

### 7️⃣  Unified Search Interface (GUI) ⭐
**File:** `7_search_faces_gui.py`
**Purpose:** Search faces with semantic, metadata, or combined search
**Usage:** `python 7_search_faces_gui.py`

**Search Modes:**
- **🧠 Semantic Search** - Find visually similar faces
- **📋 Metadata Search** - Filter by attributes
- **🔄 Combined Search** - Use both methods

**Filters:**
- Age group
- Skin tone
- Image quality
- Brightness range
- Date range

**Query Sources:**
- Select from file
- Download random face
- Paste from clipboard

**Results Display:**
- Rank and similarity %
- Filename and metadata
- Sortable columns
- View full image
- View complete metadata
- Export results as JSON

**Key Features:**
- Single temp file (auto-cleanup)
- Resizable panels
- Real-time preview
- Rich metadata display

---

### 8️⃣  Validate Embeddings
**File:** `8_validate_embeddings.py`
**Purpose:** Verify embedding quality and consistency
**Usage:** `python 8_validate_embeddings.py`

**Checks:**
- Embedding dimensions
- Vector quality
- Metadata completeness
- File existence
- Duplicate detection
- Distribution analysis

---

### 9️⃣  Test Feature Extraction
**File:** `9_test_features.py`
**Purpose:** Test feature extraction on sample images
**Usage:** `python 9_test_features.py`

**Tests:**
- Age estimation
- Skin tone detection
- Quality assessment
- Brightness analysis
- Color statistics

**Output:**
- Feature values for sample images
- Distribution statistics
- Visual verification

---

### 🔟 Complete Pipeline Demo
**File:** `10_complete_demo.py`
**Purpose:** Demonstrates complete pipeline workflow
**Usage:** `python 10_complete_demo.py`

**Demo Steps:**
1. Check database setup
2. Verify face images
3. Check embeddings
4. Test feature extraction
5. Launch search interface
6. Show system summary

---

## 📂 File Organization

### Main Components (0-10)
```
0_launcher.py              - Main launcher menu
1_setup_database.py        - Database setup
2_database_info.py         - Database stats
3_download_faces.py        - Download CLI
4_download_faces_gui.py    - Download GUI
5_embed_faces.py           - Embedding CLI
6_embed_faces_gui.py       - Embedding GUI
7_search_faces_gui.py      - Search GUI ⭐
8_validate_embeddings.py   - Validation
9_test_features.py         - Feature testing
10_complete_demo.py        - Complete demo
```

### Core Libraries (Used by all components)
```
face_collector.py          - Face collection and feature extraction
face_database.py           - Database operations and search
setup_chroma.py            - ChromaDB setup utilities
run_chroma_info.py         - Database info utilities
```

### Legacy Files (Still supported, but use numbered versions)
```
99.downbackground.py           → Use 3_download_faces.py
99.downbackground.gui.py       → Use 4_download_faces_gui.py
100.embedintoVector.py         → Use 5_embed_faces.py
100.embedintoVectorgui.py      → Use 6_embed_faces_gui.py
101.test_searchgui.py          → Use 7_search_faces_gui.py (deprecated)
102.unified_search_gui.py      → Use 7_search_faces_gui.py
```

---

## 🎯 Common Workflows

### First Time Setup
```bash
# 1. Setup database
python 1_setup_database.py

# 2. Download faces
python 4_download_faces_gui.py  # Download 100-1000 faces

# 3. Embed faces
python 6_embed_faces_gui.py     # Select "Clear Existing Embeddings"

# 4. Verify
python 2_database_info.py       # Check stats

# 5. Search
python 7_search_faces_gui.py    # Start searching!
```

### Adding More Faces
```bash
# Download more faces
python 3_download_faces.py --count 500

# Embed new faces (don't clear!)
python 5_embed_faces.py         # Skips duplicates automatically

# Verify
python 2_database_info.py
```

### Re-embedding with Updated Features
```bash
# Re-embed with new feature extraction
python 5_embed_faces.py --clear

# Or use GUI
python 6_embed_faces_gui.py     # Check "Clear Existing Embeddings"
```

### Searching Faces

**Semantic Search (Find Similar):**
```bash
python 7_search_faces_gui.py
# 1. Load query image
# 2. Select "Semantic Search Only"
# 3. Click SEARCH
```

**Metadata Search (Browse by Attributes):**
```bash
python 7_search_faces_gui.py
# 1. Select "Metadata Search Only"
# 2. Set filters (age, skin, quality)
# 3. Click SEARCH
```

**Combined Search (Best Results):**
```bash
python 7_search_faces_gui.py
# 1. Load query image
# 2. Select "Combined Search"
# 3. Set filters
# 4. Click SEARCH
```

---

## 🔧 Troubleshooting

### Database Not Found
```bash
# Solution: Setup database
python 1_setup_database.py
```

### No Embeddings
```bash
# Solution: Embed faces first
python 6_embed_faces_gui.py
```

### Features Show "Unknown"
```bash
# Solution: Re-embed with feature extraction
python 5_embed_faces.py --clear
```

### Temp Files Not Cleaning
```bash
# Solution: Use new search interface
python 7_search_faces_gui.py
# Has automatic cleanup!
```

### Unicode/Encoding Errors (Windows)
All numbered scripts (1-10) handle Windows encoding automatically.

---

## 📊 Performance Tips

**For Fast Downloads:**
- Use CLI: `python 3_download_faces.py --max-workers 100`
- Higher workers = faster (but may hit rate limits)

**For Fast Embedding:**
- Use batches: `--batch-size 100`
- More workers: `--max-workers 8`

**For Best Search Results:**
- Use Combined Search mode
- Start with 10-20 results
- Adjust similarity threshold
- Add specific filters

**Database Size:**
- 100 faces = ~5-10 MB
- 1,000 faces = ~50-100 MB
- 10,000 faces = ~500 MB - 1 GB

---

## 🆘 Getting Help

**Check System Status:**
```bash
python 2_database_info.py       # Database info
python 8_validate_embeddings.py # Validation
python 9_test_features.py       # Feature test
```

**Run Complete Demo:**
```bash
python 10_complete_demo.py      # Step-by-step guide
```

**Use Launcher Menu:**
```bash
python 0_launcher.py            # Interactive menu
```

---

## 📚 Additional Resources

- [UNIFIED_SEARCH_GUIDE.md](UNIFIED_SEARCH_GUIDE.md) - Detailed search guide
- [FIXES_SUMMARY.md](FIXES_SUMMARY.md) - Recent bug fixes
- Individual script help: `python X_script.py --help`

---

## 🎉 Quick Reference

| Task | Command | Time |
|------|---------|------|
| Setup | `python 1_setup_database.py` | 1 min |
| Download 100 faces | `python 3_download_faces.py --count 100` | 2-5 min |
| Embed faces | `python 5_embed_faces.py` | 2-10 min |
| Search | `python 7_search_faces_gui.py` | Instant |
| Check stats | `python 2_database_info.py` | 5 sec |

**Recommended Starting Point:** `python 0_launcher.py` 🚀