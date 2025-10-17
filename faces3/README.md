# Integrated Face Processing System

A unified GUI application that combines face downloading, embedding, and search functionality into a single interface.

## Features

- **Face Download**: Download synthetic faces from ThisPersonDoesNotExist.com
- **Face Processing**: Analyze and create embeddings for face images
- **Vector Search**: Search for similar faces using ChromaDB vector database
- **Integrated GUI**: Single application with tabbed interface for all functions
- **Real-time Statistics**: Monitor download rates, processing progress, and system status

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Installation

1. **Clone or copy the files to your desired directory**

2. **Run the setup script**:
   ```bash
   python setup.py
   ```

   This will:
   - Install all required dependencies
   - Create necessary directories
   - Initialize the ChromaDB database
   - Test all imports

3. **Alternative manual installation**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

1. **Launch the application**:
   ```bash
   python integrated_face_gui.py
   ```

2. **System Overview Tab**: View system status and statistics

3. **Download Faces Tab**:
   - Configure download settings
   - Start/stop automatic downloading
   - Download single faces

4. **Process & Embed Tab**:
   - Process downloaded faces into vector embeddings
   - Monitor processing progress

5. **Search Faces Tab**:
   - Search for similar faces using an image
   - View search results with thumbnails

6. **Configuration Tab**:
   - Configure database settings
   - Check dependencies status
   - Reset database if needed

### Workflow

1. **Initialize System**: The system automatically initializes when started
2. **Download Faces**: Use the Download tab to collect face images
3. **Process Faces**: Use the Process tab to create embeddings for the images
4. **Search Faces**: Use the Search tab to find similar faces

## File Structure

```
faces2/
├── core_backend.py          # Core backend functionality
├── integrated_face_gui.py   # Main GUI application
├── requirements.txt         # Python dependencies
├── setup.py                # Setup and installation script
├── README.md               # This file
├── system_config.json      # Configuration file (created automatically)
├── faces/                  # Downloaded face images
└── chroma_db/             # ChromaDB vector database
```

## Configuration

The system uses `system_config.json` for configuration. Key settings:

- `faces_dir`: Directory for storing downloaded face images (default: "./faces")
- `db_path`: ChromaDB database path (default: "./chroma_db")
- `collection_name`: Database collection name (default: "faces")
- `download_delay`: Delay between downloads in seconds (default: 1.0)
- `max_workers`: Number of processing workers (default: 2)
- `batch_size`: Batch size for processing (default: 50)

## Dependencies

- **chromadb**: Vector database for embeddings
- **numpy**: Numerical computations
- **Pillow (PIL)**: Image processing
- **requests**: HTTP requests for downloading
- **opencv-python**: Advanced image analysis (optional)
- **tkinter**: GUI framework (usually included with Python)

## Architecture

### Core Components

1. **IntegratedFaceSystem**: Main system coordinator
2. **DatabaseManager**: ChromaDB integration
3. **FaceDownloader**: Image downloading from web
4. **FaceProcessor**: Image analysis and embedding creation
5. **FaceAnalyzer**: Extract features from images
6. **FaceEmbedder**: Create vector embeddings

### GUI Components

- **System Overview**: Status monitoring and statistics
- **Download Control**: Face downloading interface
- **Processing Control**: Embedding creation interface
- **Search Interface**: Similar face search
- **Configuration**: System settings and database management

## Troubleshooting

### Common Issues

1. **ChromaDB Installation Issues**:
   ```bash
   pip install --upgrade chromadb
   ```

2. **OpenCV Issues**:
   ```bash
   pip install opencv-python-headless
   ```

3. **Permission Issues**:
   - Ensure write permissions for faces/ and chroma_db/ directories

4. **Network Issues**:
   - Check internet connection for face downloading
   - Some networks may block ThisPersonDoesNotExist.com

### Error Messages

- **"System not initialized"**: Run setup.py or check dependencies
- **"Failed to initialize database"**: Check ChromaDB installation
- **"No results found"**: Database may be empty, process some faces first

## Performance Tips

1. **Download Settings**:
   - Increase delay between downloads to reduce server load
   - Use fewer workers on slower systems

2. **Processing Settings**:
   - Adjust batch size based on available memory
   - Reduce max workers on slower systems

3. **Database**:
   - The database grows with each processed face
   - Consider periodic cleanup of old data

## Advanced Usage

### Batch Processing

Process existing images:
1. Copy images to the faces/ directory
2. Use the "Process All Faces" button
3. Wait for completion

### Search Optimization

For better search results:
1. Process more face images
2. Use high-quality search images
3. Adjust number of results based on database size

## Development

### Adding Features

The modular architecture allows easy extension:

1. **Backend**: Extend core_backend.py classes
2. **GUI**: Add tabs to integrated_face_gui.py
3. **Configuration**: Update SystemConfig dataclass

### API Usage

Core backend can be used independently:

```python
from core_backend import IntegratedFaceSystem

system = IntegratedFaceSystem()
system.initialize()

# Download a face
file_path = system.downloader.download_face()

# Process the face
system.processor.process_face_file(file_path)

# Search for similar faces
results = system.db_manager.search_faces(embedding, n_results=10)
```

## License

This project is provided as-is for educational and research purposes.

## Credits

- Uses synthetic faces from ThisPersonDoesNotExist.com
- Built with ChromaDB for vector storage
- GUI created with tkinter