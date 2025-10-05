# 🎭 Integrated Face Processing System - Web Interface

Complete web-based interface for face downloading, processing, and search with camera support.

## 🚀 Quick Start

### Run the Web Interface

```bash
cd faces2
./2_run_web_simple.sh
```

Then open your browser to: **http://localhost:7860**

## 📦 Installation

### Required Dependencies

```bash
pip3 install gradio chromadb Pillow opencv-python --break-system-packages
```

Or install individually:
- `pip3 install gradio --break-system-packages` - Web framework
- `pip3 install chromadb --break-system-packages` - Vector database
- `pip3 install Pillow --break-system-packages` - Image processing
- `pip3 install opencv-python --break-system-packages` - Camera support (optional)

### Verify Installation

```bash
python3 -c "import gradio; print('Gradio:', gradio.__version__)"
python3 -c "import chromadb; print('ChromaDB installed')"
python3 -c "import cv2; print('OpenCV installed')"
```

## 🌟 Features

### 1. 📊 System Overview Tab
- Real-time system status
- Database statistics
- Download/embed/search metrics
- Configuration display

### 2. ⬇️ Download Faces Tab
**Download Options:**
- **📥 Download Single** - Download one face at a time
- **▶️ Start Download** - Download multiple faces (1-100)
- **⏹️ Stop Download** - Stop downloading

**Live Webcam Capture:**
- **📹 Live Webcam Preview** - Real-time webcam stream
- **📸 Capture from Webcam** - Save current webcam frame
- Shows live preview before capture

**Features:**
- Live preview of captured images
- Real-time batch download gallery
- Progress tracking with download rate
- Status messages for each operation
- Automatic JSON metadata generation

### 3. ⚙️ Process & Embed Tab
- **🔄 Process All Faces** - Create embeddings for all images
- **Duplicate Detection** - Automatically skips duplicate images
- Batch processing
- Progress feedback
- Error reporting
- Duplicate count tracking

### 4. 🔍 Search Faces Tab
**Query Input:**
- **Upload Image** - Drag & drop or browse
- **📹 Live Webcam** - Real-time webcam preview
- **📸 Use Webcam Image** - Capture from live webcam stream
- **Query Preview** - See your selected image in real-time

**Search Modes:**
- **Vector** - Image similarity search only
- **Metadata** - Filter by demographics only
- **Hybrid** - Combine image + metadata filters

**Metadata Filters:**
- **Sex**: male, female, unknown
- **Age Group**: child, young_adult, adult, middle_aged, senior
- **Skin Tone**: very_light, light, medium, tan, brown, dark
- **Skin Color**: light, medium, dark
- **Hair Color**: black, dark_brown, brown, blonde, red, gray, etc.

**Results:**
- Gallery view with thumbnails
- Distance scores
- Metadata captions
- 1-50 results

### 5. ⚙️ Configuration Tab
**Database:**
- Initialize vector database
- Set path and collection name

**Download Directory:**
- Initialize download folder
- View file count and size

**System Settings:**
- Download delay (0.1-10s)
- Batch size (1-200)
- Max workers (1-8)
- Save/Load configuration

## 📸 Camera Features

### Download Tab - Live Webcam:
1. Enable your webcam in the **"Live Webcam Capture"** section
2. See live preview of webcam feed
3. Click **"📸 Capture from Webcam"** when ready
4. Image automatically saved to download directory
5. JSON metadata auto-generated
6. Status shows saved filename

### Search Tab - Live Webcam:
1. Enable webcam in **"Live Webcam for Search"** section
2. See real-time webcam preview
3. Click **"📸 Use Webcam Image"** to capture
4. Image automatically loads into query field
5. Preview appears on right side
6. Click **"🔍 Search Faces"** to search

**Note:** Webcam requires browser camera permissions. OpenCV optional for advanced features.

## 🌐 Access Options

### Local Access
```
http://localhost:7860
```

### Network Access (from other devices)
```
http://YOUR_IP:7860
```

Find your IP:
```bash
# macOS/Linux
hostname -I

# macOS alternative
ifconfig | grep "inet "
```

### Share Link (optional)
Edit `web_interface.py` line 718:
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True  # Enable public share link
)
```

## 🔧 Configuration Files

### system_config.json
Stores system settings:
```json
{
  "faces_dir": "./faces",
  "db_path": "./chroma_db",
  "collection_name": "faces",
  "download_delay": 1.0,
  "batch_size": 50,
  "max_workers": 2
}
```

### Directory Structure
```
faces2/
├── web_interface.py          # Web application
├── core_backend.py            # Backend logic
├── 2_run_web_simple.sh       # Launch script
├── system_config.json        # Configuration
├── faces/                    # Downloaded images
│   ├── face_*.jpg           # Face images
│   ├── face_*.json          # Metadata files
│   └── temp/                # Temporary query images
└── chroma_db/               # Vector database
```

## 🔍 Duplicate Detection

The system automatically detects and skips duplicate images during the embedding process:

- **How it works:** Each image is hashed (MD5) and checked against the database
- **When duplicates are found:** They are skipped and counted separately
- **Benefits:**
  - Saves processing time
  - Prevents database bloat
  - Maintains clean dataset
- **Statistics:** View duplicate count in system status

### Viewing Duplicate Stats:
1. Go to **System Overview** tab
2. Check **"embed_duplicates"** counter
3. See total duplicates skipped

## 🎯 Common Use Cases

### Download 50 Faces
1. Go to **Download Faces** tab
2. Set slider to 50
3. Click **"▶️ Start Download"**
4. Wait for completion
5. Click **"⏹️ Stop Download"** to stop early

### Capture Your Photo and Search (Live Webcam)
1. Go to **Search Faces** tab
2. Enable webcam in **"Live Webcam for Search"** section
3. See yourself in live preview
4. Click **"📸 Use Webcam Image"** when ready
5. Your photo appears in query field and preview
6. Select search mode (Vector/Metadata/Hybrid)
7. Click **"🔍 Search Faces"**
8. View similar faces in gallery

### Search by Demographics
1. Go to **Search Faces** tab
2. Select **"Metadata"** mode
3. Set filters (e.g., female, young_adult, blonde hair)
4. Click **"🔍 Search Faces"**
5. Results match your criteria

### Hybrid Search
1. Upload/capture query image
2. Select **"Hybrid"** mode
3. Set metadata filters
4. Click **"🔍 Search Faces"**
5. Results are visually similar AND match filters

## 🐛 Troubleshooting

### Port Already in Use
Change port in `web_interface.py`:
```python
demo.launch(server_port=7861)  # Change from 7860
```

### Camera Not Working
- Install OpenCV: `pip3 install opencv-python --break-system-packages`
- Grant camera permissions in System Settings
- Try different camera index: `cv2.VideoCapture(1)` instead of `0`

### Database Errors
1. Go to **Configuration** tab
2. Click **"🗄️ Initialize Vector Database"**
3. Check ChromaDB is installed: `pip3 list | grep chromadb`

### Module Not Found
```bash
pip3 install gradio chromadb Pillow opencv-python --break-system-packages
```

## 📊 Performance Tips

- **Download Delay**: Set 0.5-1.0s to avoid rate limiting
- **Batch Size**: Use 50-100 for processing
- **Max Workers**: Set to CPU cores - 1 (e.g., 4 for 8-core)
- **Search Results**: Start with 10-20 for faster results

## 🔒 Security Notes

- Web interface runs on local network by default
- Use `share=False` (default) to keep it private
- Don't expose to internet without authentication
- Camera access requires user permission

## 📝 Comparison: GUI vs Web

| Feature | GUI (Tkinter) | Web (Gradio) |
|---------|---------------|--------------|
| Access | Desktop only | Any browser |
| Platform | macOS/Windows/Linux | Universal |
| Mobile | ❌ | ✅ |
| Remote Access | ❌ | ✅ |
| Deployment | Desktop app | Web server |
| Camera | OpenCV popup | Instant capture |
| Preview | Thumbnails | Gallery view |
| Sharing | ❌ | ✅ (share links) |

## 🆚 When to Use Which?

### Use GUI (1_run_integrated_gui.sh) when:
- Running on local desktop
- Want traditional desktop app
- Need detailed control windows
- Prefer keyboard shortcuts

### Use Web (2_run_web_simple.sh) when:
- Need remote access
- Using mobile/tablet
- Want to share with others
- Deploying to server
- Prefer modern web UI

## 📞 Support

Check status:
```bash
python3 web_interface.py
```

Test imports:
```bash
python3 -c "from web_interface import WebInterface; print('OK')"
```

## 🎉 Enjoy!

Access at: **http://localhost:7860**

All GUI features available in your browser! 🚀
