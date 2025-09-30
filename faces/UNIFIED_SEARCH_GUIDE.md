# Unified Face Search Interface Guide

## Overview

The new **Unified Search Interface** ([102.unified_search_gui.py](102.unified_search_gui.py)) provides a comprehensive face search system with:

✅ **Semantic Search** - Find visually similar faces
✅ **Metadata Search** - Filter by face attributes
✅ **Combined Search** - Use both methods together
✅ **Single Temp File** - No more temp file clutter
✅ **Rich Metadata Display** - View all face information
✅ **Export Results** - Save search results as JSON

## Key Features

### 1. Three Search Modes

**🧠 Semantic Search Only**
- Finds faces visually similar to your query image
- Uses deep learning embeddings
- Adjustable similarity threshold
- Configurable number of results

**📋 Metadata Search Only**
- Search based on face attributes without a query image
- Filter by age group, skin tone, quality, brightness, date
- Perfect for browsing your collection

**🔄 Combined Search (Recommended)**
- Best of both worlds!
- Finds visually similar faces AND applies metadata filters
- Example: "Find similar faces that are high quality and recent"

### 2. Advanced Metadata Filters

**Age Group Filter:**
- Young Adult
- Adult
- Mature Adult

**Skin Tone Filter:**
- Light
- Medium
- Dark

**Image Quality Filter:**
- High (sharp, clear images)
- Medium
- Low (blurry images)

**Brightness Range:**
- Adjustable min/max brightness (0-255)
- Find darker or brighter faces

**Date Range:**
- Filter by download date
- Format: YYYY-MM-DD
- Find recently downloaded faces

### 3. Query Image Sources

**📁 Select from File**
- Browse your local files
- Supports: JPG, PNG, BMP, GIF, TIFF

**🌐 Download Random Face**
- Instantly download a random face from ThisPersonDoesNotExist
- Perfect for testing

**📋 Paste from Clipboard**
- Copy an image anywhere, paste it here
- Quick and convenient

### 4. Single Temp File Management

**Problem Solved:** Old interface created many temp files that cluttered your system.

**New Solution:**
- Only ONE temp file at a time
- Automatically cleaned up when you:
  - Download a new image
  - Paste from clipboard
  - Close the application
- No more leftover files!

### 5. Rich Results Display

Results show:
- **Rank** - Search result position
- **Match %** - Similarity percentage (semantic search)
- **Filename** - Image filename
- **Age** - Estimated age group
- **Skin** - Estimated skin tone
- **Quality** - Image quality assessment
- **Brightness** - Average brightness value
- **Date** - Download date

### 6. Result Actions

**👁️ View Image** - Open full-size image
**📋 View Metadata** - See complete JSON metadata
**💾 Export Results** - Save as JSON file
**🗑️ Clear Results** - Clear result list

## How to Use

### Basic Semantic Search

1. Launch the app:
   ```bash
   python 102.unified_search_gui.py
   ```

2. Go to **"📷 Query Image"** tab
3. Click **"🌐 Download Random Face"** or **"📁 Select from File"**
4. Go to **"⚙️ Search Mode"** tab
5. Select **"🧠 Semantic Search Only"**
6. Adjust **"Number of results"** (default: 10)
7. Click **"🔍 SEARCH"**

### Metadata-Only Search

1. Go to **"⚙️ Search Mode"** tab
2. Select **"📋 Metadata Search Only"**
3. Go to **"🔍 Filters"** tab
4. Select filters (e.g., Age: Adult, Quality: High)
5. Click **"🔍 SEARCH"**

No query image needed!

### Combined Search (Most Powerful)

1. Select a query image
2. Go to **"⚙️ Search Mode"** tab
3. Select **"🔄 Combined Search"**
4. Go to **"🔍 Filters"** tab
5. Add any filters you want
6. Click **"🔍 SEARCH"**

Example: "Find faces similar to this image, but only high-quality ones downloaded recently"

## Search Examples

### Example 1: Find High-Quality Similar Faces

```
1. Load query image
2. Mode: Combined Search
3. Filters:
   - Quality: High
   - Brightness: 100-200 (well-lit)
4. Number of results: 20
5. SEARCH
```

### Example 2: Browse Recent Adult Faces

```
1. Mode: Metadata Search Only
2. Filters:
   - Age Group: Adult
   - Date Range: 2025-09-01 to 2025-09-30
3. Number of results: 50
4. SEARCH
```

### Example 3: Find Similar Dark Tones

```
1. Load query image (darker skin tone)
2. Mode: Combined Search
3. Filters:
   - Skin Tone: Dark
4. Min Similarity: 70%
5. SEARCH
```

## Export Results

Search results can be exported as JSON containing:

```json
{
  "search_mode": "combined",
  "search_timestamp": "2025-09-30T14:30:00",
  "query_image": "path/to/query.jpg",
  "num_results": 10,
  "filters": {
    "estimated_age_group": "adult",
    "image_quality": "high"
  },
  "results": [
    {
      "rank": 1,
      "face_id": "face_000123",
      "similarity": 95.5,
      "metadata": { ... }
    }
  ]
}
```

## Tips & Tricks

### For Best Semantic Search Results:

1. Use clear, well-lit query images
2. Start with 10-20 results, expand if needed
3. Lower similarity threshold to get more results
4. Use Combined mode to filter out unwanted matches

### For Efficient Metadata Search:

1. Start broad, then add more filters
2. Use date ranges to search recent additions
3. Combine multiple filters for specific needs
4. Remember: quality filter is very useful!

### Temp File Management:

- ✅ Only ONE temp file exists at a time
- ✅ Auto-cleanup on download/paste
- ✅ Auto-cleanup on app close
- ✅ Manual cleanup with "🗑️ Clear All" button

## Comparison with Old Interface

| Feature | Old (101) | New (102) |
|---------|-----------|-----------|
| Search Modes | Semantic only | Semantic, Metadata, Combined |
| Metadata Filters | None | Age, Skin, Quality, Brightness, Date |
| Temp Files | Many files | Single file |
| Cleanup | Manual | Automatic |
| Query Sources | File, Download | File, Download, Clipboard |
| Results Display | Basic | Full metadata table |
| Export | Basic | Enhanced with filters |
| UI Layout | Tabs | Resizable panels |

## Troubleshooting

**"No results found"**
- Check your filters aren't too restrictive
- Lower the similarity threshold
- Try metadata-only search first
- Verify database has faces embedded

**"Query image file not found"**
- Image may have been deleted
- Download a new random face
- Select a different file

**"Database initialization failed"**
- Run embedding first: `python 100.embedintoVector.py`
- Or use GUI: `python 100.embedintoVectorgui.py`
- Check ChromaDB is installed

**Temp file not cleaning up**
- Use "🗑️ Clear All" button
- Close and reopen app
- Check permissions on temp directory

## Performance Notes

- **Semantic search**: ~0.5-2 seconds for 10-50 results
- **Metadata search**: ~1-5 seconds depending on database size
- **Combined search**: Slightly slower (searches more, then filters)

Database with 100-1000 faces: Very fast
Database with 10,000+ faces: May take a few seconds

## Next Steps

1. Try all three search modes
2. Experiment with different filters
3. Export results for analysis
4. Provide feedback for improvements!

## Related Files

- [100.embedintoVector.py](100.embedintoVector.py) - Embed faces with metadata
- [100.embedintoVectorgui.py](100.embedintoVectorgui.py) - GUI for embedding
- [101.test_searchgui.py](101.test_searchgui.py) - Old search interface (deprecated)
- [102.unified_search_gui.py](102.unified_search_gui.py) - **New unified interface** ⭐