# Face Processing System

A streamlined GUI application for face image processing with AI-generated faces, vector embeddings, and similarity search.

## Overview

This system provides a complete workflow for building and querying a face recognition database:
- Download AI-generated face images from web services
- Capture faces from camera
- Analyze faces for features and demographics
- Create vector embeddings using multiple models
- Search for similar faces using vector similarity
- Filter by metadata (age, sex, skin tone, hair color, etc.)

## Features

- **Face Acquisition**: Download from AI services (ThisPersonDoesNotExist, 100K Faces) or capture from camera
- **Multi-Model Embeddings**: Support for Statistical, FaceNet, ArcFace, DeepFace, VGGFace2, and OpenFace
- **Advanced Search**: Vector similarity, metadata filtering, and hybrid search
- **Demographic Analysis**: Automatic estimation of age, sex, skin tone, and hair color
- **Real-time Monitoring**: Live statistics, thumbnails, and progress tracking
- **Simple Interface**: Single GUI application with intuitive tabbed interface

## Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   ./1_run_integrated_gui.sh
   ```

   Or directly:
   ```bash
   python3 integrated_face_gui.py
   ```

### Basic Workflow

1. **Start the application** - The system initializes automatically
2. **Download Faces** - Go to "Download Faces" tab and click "Start Download"
3. **Process Faces** - Go to "Process & Embed" tab and click "Process All Faces"
4. **Search Faces** - Go to "Search Faces" tab, select an image, and search

## File Structure

```
faces4/
├── integrated_face_gui.py   # Main GUI application
├── core_backend.py          # Core backend functionality
├── 1_run_integrated_gui.sh  # Application launcher
├── requirements.txt         # Python dependencies
├── setup.py                # Setup script
├── benchmark_performance.py # Performance testing
├── README.md               # This file
├── system_config.json      # Configuration (auto-created)
├── faces/                  # Face images directory
└── chroma_db/             # ChromaDB vector database
```

## User Interface

The application has five main tabs:

### 1. System Overview
- View system status and statistics
- Monitor download/processing rates
- Track database size and uptime

### 2. Download Faces
- Download AI-generated faces automatically
- Download single faces on demand
- Capture faces from camera
- View real-time download statistics and thumbnails

### 3. Process & Embed
- Process all faces or only new ones
- Select embedding model (Statistical, FaceNet, ArcFace, etc.)
- Monitor processing progress with statistics
- View processed face thumbnails

### 4. Search Faces
- Search by image (upload or camera capture)
- Three search modes:
  - **Vector**: Pure similarity search
  - **Metadata**: Filter by demographics and properties
  - **Hybrid**: Combine vector similarity + metadata
- Filter by: sex, age group, skin tone/color, hair color, brightness, quality

### 5. Configuration
- Configure database and storage paths
- Select embedding model
- Check dependencies and model availability
- Re-embed data when changing models
- Manage system settings

## Configuration

Configuration is stored in `system_config.json`:

```json
{
  "faces_dir": "./faces",
  "db_path": "./chroma_db",
  "collection_name": "faces",
  "download_delay": 1.0,
  "max_workers": 4,
  "batch_size": 50,
  "embedding_model": "statistical",
  "download_source": "thispersondoesnotexist"
}
```

### Key Settings

- **faces_dir**: Directory for face images
- **db_path**: ChromaDB database location
- **embedding_model**: Model for creating embeddings
  - `statistical` (default, always available)
  - `facenet` (requires: `pip install facenet-pytorch torch torchvision`)
  - `arcface` (requires: `pip install insightface onnxruntime`)
  - `deepface` (requires: `pip install deepface`)
  - `vggface2` (requires: `pip install deepface`)
  - `openface` (requires: `pip install deepface`)
- **download_source**: Face download source
  - `thispersondoesnotexist` (1024x1024 high quality)
  - `100k-faces` (generated.photos dataset)

## Dependencies

### Required
- **chromadb**: Vector database
- **numpy**: Numerical operations
- **Pillow**: Image processing
- **requests**: HTTP downloads
- **tkinter**: GUI (usually included with Python)

### Optional
- **opencv-python**: Advanced image analysis and camera support
- **facenet-pytorch, torch, torchvision**: FaceNet model
- **insightface, onnxruntime**: ArcFace model
- **deepface**: DeepFace, VGGFace2, OpenFace models

Install optional dependencies as needed:
```bash
# For camera support and advanced analysis
pip install opencv-python

# For FaceNet
pip install facenet-pytorch torch torchvision

# For ArcFace
pip install insightface onnxruntime

# For DeepFace/VGGFace2/OpenFace
pip install deepface
```

## Architecture

### Core Components

**Data Classes**
- `FaceData`: Face information container
- `SystemConfig`: System configuration
- `SystemStats`: Statistics tracking

**Analysis & Embedding**
- `FaceAnalyzer`: Extracts features and demographics from images
- `FaceEmbedder`: Creates vector embeddings using various models

**Database & Storage**
- `DatabaseManager`: ChromaDB operations and vector search
- `FaceDownloader`: Downloads AI-generated faces
- `FaceProcessor`: Processes images and creates embeddings

**Integration**
- `IntegratedFaceSystem`: Main system orchestrator
- `IntegratedFaceGUI`: GUI application

### Processing Pipeline

```
Download → Analyze → Embed → Store → Search
   ↓         ↓         ↓       ↓        ↓
AI Face   Features  Vector  ChromaDB  Results
```

## Advanced Usage

### Programmatic API

Use the backend directly without GUI:

```python
from core_backend import IntegratedFaceSystem

# Initialize system
system = IntegratedFaceSystem()
system.initialize()

# Download and process a face
file_path = system.downloader.download_face()
system.processor.process_face_file(file_path)

# Search for similar faces
from core_backend import FaceAnalyzer, FaceEmbedder

analyzer = FaceAnalyzer()
embedder = FaceEmbedder(model_name='statistical')

features = analyzer.analyze_face('query_image.jpg')
embedding = embedder.create_embedding('query_image.jpg', features)

results = system.db_manager.search_faces(embedding, n_results=10)
```

### Batch Processing

Process existing images in a directory:

1. Copy images to `faces/` directory
2. Open "Process & Embed" tab
3. Click "Process All Faces" or "Process New Only"

### Model Migration

When changing embedding models:

1. Go to "Configuration" tab
2. Select new embedding model
3. Click "Re-embed All Data"
4. Wait for processing to complete

**Important**: Searching with mismatched models produces incorrect results. The system will warn you and block searches if model mismatch is detected.

## Troubleshooting

### Common Issues

**ChromaDB Errors**
```bash
pip install --upgrade chromadb
```

**OpenCV/Camera Issues**
```bash
pip install opencv-python
```

**Permission Errors**
- Ensure write permissions for `faces/` and `chroma_db/` directories

**Model Mismatch Warning**
- Click "Re-embed All Data" in Configuration tab to rebuild with current model

**Network Download Failures**
- Check internet connection
- Try different download source in Configuration
- Increase download delay

### Performance Tips

1. **For faster downloads**: Decrease download delay (but respect server limits)
2. **For faster processing**: Increase max workers (if you have CPU cores)
3. **For large databases**: Use more specific metadata filters when searching
4. **For accuracy**: Use deep learning models (FaceNet, ArcFace) instead of statistical

## Development

The codebase is organized for easy extension:

- **core_backend.py**: Modify to add new models, features, or database operations
- **integrated_face_gui.py**: Modify to add new tabs or UI features
- Clean section organization with clear comments
- Comprehensive docstrings for all classes and methods

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- AI-generated faces from ThisPersonDoesNotExist.com and generated.photos
- Vector storage powered by ChromaDB
- GUI built with Python tkinter
