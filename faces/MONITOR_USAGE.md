# 🔍 ChromaDB Monitor - Usage Guide

## Overview
Real-time graphical monitoring system for ChromaDB vector database during embedding operations.

## Features
✨ **Beautiful Terminal UI** with live updates
📊 **Real-time Statistics** - Processing rates, totals, progress
📈 **Visual Progress Bars** - See completion percentage
🎨 **Data Distribution** - Age groups, skin tones, quality metrics
⚡ **Performance Metrics** - Instant and average processing rates
🗄️ **Collection Info** - Database details and dimensions

## Quick Start

### Method 1: Using Shell Script (Recommended)
```bash
# Start with default 1-second refresh
./monitor_chroma.sh

# Start with custom refresh rate (0.5 seconds)
./monitor_chroma.sh 0.5

# Start with slower refresh (2 seconds)
./monitor_chroma.sh 2
```

### Method 2: Using Python Directly
```bash
# Default 1-second refresh
python3 monitor_chroma.py

# Custom refresh rate
python3 monitor_chroma.py --refresh 0.5

# See all options
python3 monitor_chroma.py --help
```

## Monitoring During Embedding

### Option A: Two Terminal Windows
**Terminal 1** - Run embedding:
```bash
./4_embed_to_chromadb.sh
```

**Terminal 2** - Monitor progress:
```bash
./monitor_chroma.sh
```

### Option B: Background Embedding + Monitoring
```bash
# Start embedding in background
./4_embed_to_chromadb.sh > embedding.log 2>&1 &

# Monitor in foreground
./monitor_chroma.sh
```

## Display Sections

### 📈 Processing Statistics
- **Total Embeddings**: Current count in database
- **Average Rate**: Processing speed (embeddings/second)
- **Current Rate**: Instant processing speed
- **Estimated Time**: ETA for completion

### 📊 Overall Progress
- Visual progress bar
- Percentage complete
- Files processed vs total files

### 🗄️ Collection Info
- Collection name
- Document count
- Embedding dimension
- Database path

### 📊 Data Distribution
- **Age Groups**: Distribution across age categories
- **Skin Tones**: Distribution across skin tone categories
- **Quality**: Distribution across quality levels

## Controls
- **Ctrl+C**: Exit monitor
- Auto-refresh based on configured rate

## Refresh Rate Guide
- `0.5s`: Fast updates, higher CPU usage (for quick operations)
- `1.0s`: **Default** - Good balance
- `2.0s`: Slower updates, lower CPU usage (for long operations)

## Examples

### Monitor with fast refresh during active embedding
```bash
./monitor_chroma.sh 0.5
```

### Monitor with slow refresh for overnight processing
```bash
./monitor_chroma.sh 2
```

### Check database status quickly
```bash
# Press Ctrl+C after viewing
./monitor_chroma.sh
```

## Tips
1. **Use two terminals** for best experience - one for embedding, one for monitoring
2. **Adjust refresh rate** based on your processing speed
3. **Lower refresh rate** (1-2s) saves CPU during long operations
4. **Monitor shows all existing data** even if embedding is not currently running

## Troubleshooting

### Monitor shows no data
- Make sure ChromaDB database exists (`./chroma_db` directory)
- Run embedding first: `./4_embed_to_chromadb.sh`

### Rich library not found
The script will automatically install it, or manually:
```bash
pip3 install rich
```

### Display looks broken
- Make sure terminal is at least 100 characters wide
- Use a terminal that supports Unicode and colors

## Technical Details
- **Language**: Python 3
- **Dependencies**: rich, face_database, chromadb
- **Database**: ChromaDB (local storage)
- **Update Method**: Polling at specified refresh rate
- **Thread-safe**: Yes

---

**Created**: 2025-10-01
**Purpose**: Monitor ChromaDB vector database embedding progress in real-time
