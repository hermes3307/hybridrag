# Face Pipeline GUI Application

A graphical user interface for the Face Vector Database Pipeline that allows you to run each step through an easy-to-use tabbed interface.

## 🎨 Features

- **Tabbed Interface**: Each pipeline step has its own tab
- **Real-time Output**: See command output as it runs
- **Progress Tracking**: Visual indicators for long-running operations
- **Status Overview**: Monitor pipeline status and system info
- **Quick Actions**: Easy access to common tasks
- **No Code Changes**: Uses existing shell scripts unchanged

## 🚀 Quick Start

### Launch the GUI:
```bash
./run_gui.sh
# OR
python3 face_pipeline_gui.py
```

### Requirements:
- Python 3 with tkinter (usually included)
- All existing pipeline scripts must be present
- Same dependencies as command-line version

## 📋 GUI Tabs

### 1. Setup ChromaDB
- Install ChromaDB and dependencies
- Create initial database structure
- One-click setup with progress feedback

### 2. Verify Database
- Check ChromaDB installation
- View database collections and size
- Confirm everything is working

### 3. Collect Faces
- Download synthetic faces from ThisPersonDoesNotExist.com
- Extract features and generate embeddings
- Progress bar for long operations
- Real-time download feedback

### 4. Embed to DB
- Load face data into ChromaDB
- Create searchable vector collection
- Display database statistics

### 5. Inspect DB
- Detailed vector database analysis
- Storage usage and optimization info
- Collection metadata viewer

### 6. Test Search
- Download test face and run similarity search
- Performance metrics and accuracy results
- Custom image search (browse your own images)

### 📊 Overview Tab
- **Pipeline Status**: Current state of all components
- **Quick Actions**:
  - Refresh status
  - Open faces folder
  - Launch interactive search
- **System Information**: Working directory, Python version, database size

## 🎯 GUI Advantages

### Vs Command Line:
- **Visual Progress**: See what's happening in real-time
- **Error Handling**: Clear error messages and status
- **Easier Navigation**: Click between steps instead of remembering commands
- **Beginner Friendly**: No need to remember shell commands

### Features:
- **Real-time Output**: Command output streams live to GUI
- **Status Tracking**: Always know what's completed/pending
- **Quick Access**: Jump to any step without prerequisites
- **File Integration**: Browse images, open folders directly

## 💡 How It Works

The GUI is a **wrapper around existing scripts** - it doesn't change any core functionality:

1. **Calls Shell Scripts**: Each tab runs the corresponding `.sh` script
2. **Streams Output**: Shows real-time command output in scrollable text areas
3. **Thread Safety**: Runs commands in background threads to keep GUI responsive
4. **Status Monitoring**: Checks file system for pipeline status

## 🔧 Usage Tips

### First Time Setup:
1. Open **Step 1** tab and click "Setup ChromaDB"
2. Wait for completion, then move to **Step 2**
3. Continue through steps 3-6 in order

### Resuming Work:
- Check **Overview** tab to see current status
- Jump to any step that needs to be run
- Use "Refresh Status" to update information

### Troubleshooting:
- Output tabs show detailed error messages
- Overview tab shows what's missing
- Each step can be re-run if needed

## 📁 File Structure

```
face_pipeline_gui.py    # Main GUI application
run_gui.sh             # GUI launcher script
README_GUI.md          # This documentation

# Existing files (unchanged):
1_setup_chromadb.sh    # Step 1 script
2_check_chromadb.sh    # Step 2 script
3_collect_faces.sh     # Step 3 script
4_embed_to_chromadb.sh # Step 4 script
5_inspect_database.sh  # Step 5 script
6_test_search.sh       # Step 6 script
```

## 🎨 GUI Components

### Main Window:
- **900x700 pixels** - optimal for most screens
- **Tabbed interface** - easy navigation
- **Status bar** - current operation status
- **Resizable** - adjust to your preference

### Each Tab Contains:
- **Title and description** - what the step does
- **Information panel** - detailed explanation
- **Run button** - start the operation
- **Output area** - scrollable command output
- **Progress indicators** - where applicable

### Overview Tab Special Features:
- **Pipeline status** - visual checklist
- **Quick actions** - common operations
- **System info** - environment details
- **Real-time updates** - auto-refresh capability

## 🔍 Behind the Scenes

The GUI uses:
- **tkinter** - Python's built-in GUI framework
- **subprocess** - to run shell scripts
- **threading** - for responsive interface
- **real-time streaming** - live command output

No changes to existing pipeline code - it's a pure interface layer!

## 🚀 Ready to Use!

Launch with: `python3 face_pipeline_gui.py`

The GUI provides an intuitive way to run the complete face vector database pipeline without memorizing commands or dealing with terminal output.