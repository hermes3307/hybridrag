# 🎭 Unified Face Processing Application

A modern, web-based GUI application for complete face processing workflows: **Download → Embed → Index → Search**

All operations are unified in a single, intuitive interface powered by **Gradio**.

---

## ✨ Features

### 🎯 Unified Interface
- **Single Application**: All operations in one place
- **Modern Web UI**: Clean, responsive Gradio interface
- **Real-time Statistics**: Live updates of system status
- **Progress Tracking**: Visual progress bars for long operations

### 🔄 Complete Pipeline
1. **📥 Download**: Fetch face images from multiple sources
2. **⚙️ Process**: Create vector embeddings using various models
3. **💾 Index**: Store in PostgreSQL with pgvector (HNSW index)
4. **🔍 Search**: Find similar faces using vector similarity and metadata filters

### 🚀 Key Capabilities
- **Multiple Download Sources**:
  - ThisPersonDoesNotExist (AI-generated faces)
  - 100k-faces dataset
- **Flexible Embedding Models**:
  - Statistical (default)
  - FaceNet
  - ArcFace
  - DeepFace
  - VGGFace2
  - OpenFace
- **Advanced Search**:
  - Vector similarity search (cosine, L2, inner product)
  - Metadata filtering (demographics, image properties)
  - Hybrid search (combined vector + metadata)
- **Configuration Panel**:
  - Database connection settings
  - Application preferences
  - Model selection

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL 12+ with pgvector extension
- 2GB+ RAM recommended

### Installation

1. **Install system dependencies** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv postgresql
```

2. **Install pgvector extension**:
```bash
sudo apt-get install postgresql-server-dev-all
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

3. **Setup database**:
```bash
sudo -u postgres psql
CREATE DATABASE vector_db;
\c vector_db
CREATE EXTENSION vector;
\q
```

4. **Install Python dependencies**:
```bash
# Option 1: Using the launch script (recommended)
./run_app.sh

# Option 2: Manual installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Launch Application

**Simple method** (recommended):
```bash
./run_app.sh
```

**Manual method**:
```bash
source venv/bin/activate
python3 app.py
```

The application will start on `http://localhost:7860`

---

## 📖 User Guide

### 1. Initial Setup

When you first launch the app:

1. Navigate to the **⚙️ Configuration** tab
2. Enter your PostgreSQL connection details:
   - Host: `localhost` (default)
   - Port: `5432` (default)
   - Database: `vector_db`
   - User: `postgres`
   - Password: (your postgres password)
3. Click **🔌 Test Connection** to verify
4. Configure application settings:
   - Faces directory (where images are stored)
   - Embedding model (start with "statistical")
   - Batch size and workers
5. Click **💾 Save Configuration**

### 2. Download & Process Pipeline

Navigate to **📥 Download & Process** tab:

#### Download Faces
1. Select source: "ThisPersonDoesNotExist" or "100k-faces"
2. Set number of faces to download (1-100)
3. Adjust delay between downloads (0.5-5 seconds)
4. Click **⬇️ Start Download**
5. Monitor progress and statistics

#### Process & Embed
1. Set batch size (10-200, default: 50)
2. Set worker threads (1-8, default: 4)
3. Check "Process new files only" to skip already processed images
4. Click **⚙️ Start Processing**
5. Watch real-time progress

#### Complete Pipeline (One-Click)
- Click **🚀 Run Complete Pipeline** to automatically:
  1. Download the specified number of faces
  2. Process and embed them into vectors
  3. Store them in the database with HNSW index

### 3. Search for Similar Faces

Navigate to **🔍 Search Faces** tab:

#### Upload Query Image
- **Option 1**: Click **📁 Upload Image** and select a face image
- **Option 2**: Click **📷 Use Webcam** to capture from camera
- **Option 3**: Drag and drop an image

#### Configure Search
1. **Number of results**: Set how many similar faces to return (1-50)
2. **Search mode**:
   - **Vector Search Only**: Pure similarity search (fastest)
   - **Metadata Filter Only**: Filter by demographics/properties
   - **Hybrid Search**: Combine vector similarity + metadata (most powerful)

#### Apply Metadata Filters (Optional)
Fine-tune your search with filters:
- **Demographics**: Sex, Age Group, Skin Tone, Hair Color
- **Image Properties**: Brightness, Quality

#### Execute Search
1. Click **🔍 Search**
2. View results in the gallery grid
3. Each result shows the similarity distance

### 4. Monitor System Statistics

At the top of any tab, view real-time statistics:
- **Database Status**: Connection health
- **Total Faces**: Number of faces in database
- **Download Stats**: Success/error counts
- **Embedding Stats**: Processing metrics
- **Search Count**: Total searches performed

Click **🔄 Refresh Stats** to update manually.

---

## 🏗️ Architecture

### Technology Stack
- **Frontend**: Gradio 4.0+ (Web-based UI)
- **Backend**: Python 3.8+
- **Database**: PostgreSQL 12+ with pgvector extension
- **Vector Index**: HNSW (via pgvector)
- **Image Processing**: OpenCV, Pillow
- **Embeddings**: Multiple models supported

### Application Structure
```
app.py                    # Main Gradio application (unified GUI)
├── UnifiedFaceApp       # Application controller class
├── Download & Process   # Tab 1: Pipeline operations
├── Search Faces        # Tab 2: Search interface
└── Configuration       # Tab 3: Settings panel

Backend Modules:
├── core.py              # Face processing logic
├── pgvector_db.py       # Database operations
└── advanced_search.py   # Search engine
```

### Data Flow
```
1. Download → faces/*.jpg
2. Process → Vector embeddings (512-dim)
3. Index → PostgreSQL + pgvector (HNSW)
4. Search → Query → Results
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
EMBEDDING_MODEL=statistical
FACES_DIR=./faces
```

### System Config (`system_config.json`)
```json
{
  "db_host": "localhost",
  "db_port": 5432,
  "db_name": "vector_db",
  "db_user": "postgres",
  "db_password": "your_password",
  "faces_dir": "./faces",
  "embedding_model": "statistical",
  "download_source": "thispersondoesnotexist",
  "batch_size": 50,
  "max_workers": 4,
  "download_delay": 1.0
}
```

Configuration can be edited:
- Via the **⚙️ Configuration** tab in the GUI
- Manually editing `system_config.json`
- Using environment variables in `.env`

---

## 🔧 Advanced Usage

### Embedding Models

The application supports multiple embedding models:

| Model | Description | Accuracy | Speed | Dependencies |
|-------|-------------|----------|-------|--------------|
| **statistical** | Default, no dependencies | Medium | ⚡⚡⚡ | None (built-in) |
| **facenet** | FaceNet model | High | ⚡⚡ | facenet-pytorch |
| **arcface** | ArcFace/InsightFace | Very High | ⚡ | insightface |
| **deepface** | DeepFace framework | High | ⚡ | deepface |
| **vggface2** | VGGFace2 model | High | ⚡⚡ | keras |
| **openface** | OpenFace model | Medium | ⚡⚡ | openface |

To use advanced models, install additional dependencies:
```bash
pip install facenet-pytorch insightface deepface
```

### Search Modes Explained

#### Vector Search Only
- Uses only visual similarity
- Fastest search method
- Best for: "Find faces that look like this"
- Distance metric: cosine/L2/inner product

#### Metadata Filter Only
- Uses only demographic/image filters
- No similarity calculation
- Best for: "Find all males with dark skin tone"

#### Hybrid Search (Recommended)
- Combines vector similarity + metadata filters
- Most powerful and flexible
- Best for: "Find similar faces that are female with blonde hair"
- Filters are applied BEFORE similarity ranking

### Performance Tuning

**Download Performance**:
- Increase delay for rate-limited sources
- Decrease delay for local/unlimited sources

**Processing Performance**:
- Increase batch size for faster processing (uses more RAM)
- Increase workers for multi-core CPUs
- Use "statistical" model for fastest embeddings

**Search Performance**:
- Use "Vector Search Only" for fastest results
- Enable metadata filters only when needed
- Reduce top_k for faster queries

---

## 📊 Database Schema

The application uses a single `faces` table:

```sql
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(50) UNIQUE,
    file_path TEXT,
    embedding vector(512),  -- pgvector type
    features JSONB,         -- Metadata
    image_hash VARCHAR(32),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_embedding ON faces USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_features ON faces USING gin (features);
```

Metadata stored in `features` JSONB:
- **Demographics**: sex, age_group, skin_tone, hair_color
- **Image Properties**: brightness, contrast, quality, dimensions
- **Face Detection**: faces_detected, has_face
- **System**: face_id, timestamp, image_hash, embedding_model

---

## 🆚 Comparison: Old vs New GUI

| Feature | Old GUI (faces.py) | New GUI (app.py) |
|---------|-------------------|------------------|
| Framework | Tkinter (desktop) | Gradio (web) |
| Interface | 5 separate tabs | 3 unified tabs |
| Access | Local desktop only | Web browser (local/remote) |
| Deployment | Desktop app | Web app (localhost or cloud) |
| Styling | Basic tkinter | Modern, responsive web UI |
| Mobile Support | ❌ No | ✅ Yes (responsive) |
| Configuration | Separate tab, complex | Unified settings panel |
| Statistics | Per-tab display | Global, always visible |
| Progress Tracking | Text logs | Visual progress bars |
| Image Upload | File dialog only | Drag-drop, webcam, file |
| Results Display | Custom canvas | Built-in gallery |
| Installation | Requires tkinter | pip install only |
| Customization | Limited | CSS, themes available |

---

## 🐛 Troubleshooting

### Application won't start
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Verify Gradio installation
python3 -c "import gradio; print(gradio.__version__)"

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Database connection fails
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify database exists
sudo -u postgres psql -c "\l" | grep vector_db

# Test connection manually
psql -h localhost -U postgres -d vector_db
```

### pgvector extension not found
```bash
# Verify extension is installed
sudo -u postgres psql -d vector_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Check extension is loaded
sudo -u postgres psql -d vector_db -c "\dx" | grep vector
```

### Slow download speeds
- Increase `download_delay` to avoid rate limiting
- Check internet connection
- Try alternative download source

### Out of memory during processing
- Reduce `batch_size` (e.g., from 50 to 20)
- Reduce `max_workers` (e.g., from 4 to 2)
- Close other applications

### Search returns no results
- Verify database has faces: Check statistics at top
- Try "Metadata Filter Only" mode with loose filters
- Check query image is a valid face photo
- Review database logs for errors

---

## 🔐 Security Considerations

### Production Deployment

If deploying to a public server:

1. **Enable authentication**:
```python
app.launch(
    auth=("username", "password"),  # Add login
    server_name="0.0.0.0",
    server_port=7860
)
```

2. **Use environment variables for secrets**:
```bash
# Never commit passwords to git
export POSTGRES_PASSWORD="secure_password"
```

3. **Enable HTTPS**:
```bash
# Use reverse proxy (nginx) with SSL
# Or use Gradio's share=True for temporary public links
```

4. **Limit access**:
```python
# Bind to localhost only for local-only access
app.launch(server_name="127.0.0.1")
```

---

## 📚 Additional Resources

### Related Files
- `faces.py` - Legacy tkinter GUI (still available)
- `monitor.py` - Database monitoring tool
- `search_cli.py` - Command-line search interface
- `vector_query_cli.py` - Interactive CLI queries

### Documentation
- `README.md` - Project overview
- `SEARCH_GUIDE.md` - Detailed search documentation
- `ARCHITECTURE.md` - System architecture guide

### External Resources
- [Gradio Documentation](https://gradio.app/docs/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

## 🤝 Contributing

To add new features to the unified app:

1. **New download source**: Add to `FaceDownloader` in `core.py`
2. **New embedding model**: Add to `FaceEmbedder` in `core.py`
3. **New search filter**: Extend metadata in `advanced_search.py`
4. **UI improvements**: Modify `app.py` Gradio components

---

## 📝 License

This project follows the same license as the main repository.

---

## 🎯 Next Steps

After launching the application:

1. ✅ Configure database connection
2. ✅ Download sample faces (start with 10-20)
3. ✅ Process and embed the faces
4. ✅ Try different search modes
5. ✅ Experiment with metadata filters
6. ✅ Adjust performance settings

---

## 💡 Tips & Best Practices

### For Best Results
- Start with small batches (10-20 faces) to test
- Use "statistical" model initially (fastest, no dependencies)
- Enable "Process new files only" to avoid re-processing
- Use "Hybrid Search" for most accurate results
- Apply metadata filters to narrow down large datasets

### Performance Optimization
- **Small dataset (<1000 faces)**: Use defaults
- **Medium dataset (1000-10000 faces)**: Increase batch_size to 100, workers to 6
- **Large dataset (>10000 faces)**: Consider using advanced embedding models, tune PostgreSQL

### Workflow Recommendations
1. Download in batches (e.g., 50 at a time)
2. Process immediately after downloading
3. Test search with a few queries
4. Iterate and adjust settings

---

**Enjoy your unified face processing experience! 🎭**
