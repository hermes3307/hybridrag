# Face Collection and Semantic Search System

A comprehensive system to download synthetic face images from ThisPersonDoesNotExist.com, extract features, generate embeddings, and perform semantic search using ChromaDB.

## 🎯 Features

- **Face Collection**: Downloads synthetic faces from ThisPersonDoesNotExist.com
- **Feature Extraction**: Age group, skin tone, image quality estimation
- **Face Embeddings**: 143-dimensional feature vectors for similarity search
- **ChromaDB Integration**: Persistent vector storage for semantic search
- **Similarity Search**: Find similar faces using vector embeddings
- **Feature-based Search**: Search by age group, skin tone, quality
- **Duplicate Detection**: Find similar faces in the database
- **Interactive Interface**: Command-line interface for database operations

## 📁 Files

```
face_collector.py      # Face download and feature extraction
face_database.py       # ChromaDB integration and search
test_face_system.py    # Automated testing script
setup_chroma.py        # ChromaDB setup (from previous system)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install chromadb numpy pillow requests
# Optional: pip install opencv-python (for advanced features)
```

### 2. Collect Faces and Test System

```bash
python3 test_face_system.py
```

This will:
- Load any existing face data
- Add faces to ChromaDB
- Test similarity search
- Test feature-based search
- Test duplicate detection
- Test image-based search

### 3. Interactive Database Management

```bash
python3 face_database.py
```

Interactive menu options:
1. Search by image file
2. Search by features
3. Show database stats
4. Find duplicate faces
5. Exit

## 📊 System Capabilities

### Face Features Extracted

- **Age Groups**: young_adult, adult, mature_adult
- **Skin Tones**: light, medium, dark
- **Image Quality**: high, medium, low
- **Technical Features**: brightness, hue, saturation, image size

### Search Options

1. **Similarity Search**: Find faces similar to a given face
2. **Feature Search**: Filter by age, skin tone, quality
3. **Image Search**: Upload an image to find similar faces
4. **Duplicate Detection**: Find potential duplicate faces

## 🔧 Technical Details

### Face Embeddings

- **Dimensions**: 143 features per face
- **Components**:
  - Histogram features (32 bins)
  - Color moments (HSV channels)
  - Texture features (patch-based)
  - Extracted metadata features

### Database Schema

ChromaDB collection with:
- **IDs**: Unique face identifiers
- **Embeddings**: 143D feature vectors
- **Metadata**: Face features and file paths
- **Documents**: Text descriptions for search

### Performance

- **Face Download**: ~2 seconds per face (respectful rate limiting)
- **Feature Extraction**: ~50ms per face
- **Search**: Sub-second for thousands of faces
- **Storage**: ~16MB for database + face images

## 📈 Example Results

### Database Statistics
```
Total faces: 9
Age group distribution: {'adult': 7, 'mature_adult': 2}
Skin tone distribution: {'light': 8, 'medium': 1}
Quality distribution: {'high': 9}
```

### Similarity Search Results
```
Found 3 similar faces:
  1. face_0001 (similarity: 1.0)
  2. face_0004 (similarity: 0.971)
  3. face_0009 (similarity: 0.934)
```

## 🛠️ Advanced Usage

### Manual Face Collection

```python
from face_collector import FaceCollector, process_faces

# Download faces
collector = FaceCollector(delay=2.0)
face_files = collector.download_faces_batch(count=20)

# Process and extract features
processed_faces = process_faces(face_files)
```

### Database Operations

```python
from face_database import FaceDatabase

# Initialize database
face_db = FaceDatabase()

# Add faces
face_db.add_faces(processed_faces)

# Search by features
results = face_db.search_by_features({
    "estimated_age_group": "adult",
    "estimated_skin_tone": "light"
})

# Similarity search
similar = face_db.search_similar_faces(embedding_vector, n_results=5)
```

### Image-based Search

```python
from face_database import FaceSearchInterface

search_interface = FaceSearchInterface(face_db)
results = search_interface.search_by_image("path/to/image.jpg")
```

## 🔍 Search Examples

### By Age Group
```bash
# Find all adult faces
estimated_age_group: adult

# Results: All faces classified as adults
```

### By Skin Tone
```bash
# Find light skin tone faces
estimated_skin_tone: light

# Results: Faces with light skin tone classification
```

### By Similarity
```bash
# Find faces similar to a specific image
# Upload: face_0001.jpg
# Results: Top 5 most similar faces with similarity scores
```

## 📂 Data Storage

### Face Images
- **Location**: `./faces/` directory
- **Format**: JPG files (1024x1024)
- **Naming**: `face_{id}_{hash}.jpg`

### Database
- **Location**: `./chroma_db/` directory
- **Type**: ChromaDB persistent storage
- **Backup**: `face_data.json` for processed features

## 🎮 Interactive Features

The interactive interface (`face_database.py`) provides:

- **Real-time search**: Instant similarity and feature searches
- **Database statistics**: Live metrics and distributions
- **Duplicate detection**: Find similar faces automatically
- **File upload**: Search using your own images

## ⚡ Performance Tips

1. **Batch Processing**: Process multiple faces at once
2. **Rate Limiting**: Respect ThisPersonDoesNotExist.com (2s delay)
3. **Storage**: Keep face images for re-processing
4. **Indexing**: ChromaDB automatically indexes embeddings

## 🔧 Troubleshooting

### OpenCV Not Available
- System falls back to PIL for image processing
- Some advanced features may be simplified
- Install OpenCV for full functionality: `pip install opencv-python`

### Slow Downloads
- Increase delay between requests if needed
- Check internet connection
- ThisPersonDoesNotExist.com may have rate limits

### Memory Usage
- Large face collections may use significant RAM
- Consider processing in smaller batches
- ChromaDB handles persistence automatically

## 🎯 Use Cases

- **Face Recognition Research**: Study face similarity algorithms
- **Computer Vision**: Train and test face detection models
- **Data Analysis**: Analyze face feature distributions
- **Educational**: Learn about embeddings and similarity search
- **Prototyping**: Build face-based applications

## 🔮 Future Enhancements

- **Advanced Models**: FaceNet, ArcFace embeddings
- **Real-time Processing**: Webcam face search
- **Clustering**: Automatic face grouping
- **API Interface**: REST API for remote access
- **Web Interface**: Browser-based face search

---

The system is designed to be educational and respectful, using only synthetic faces that don't represent real people. Perfect for learning about vector databases, embeddings, and semantic search!