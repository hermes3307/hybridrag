# Generate Missing Metadata for Face Images

A tool to retroactively generate JSON metadata files for face images that don't have them.

## 📝 Overview

This script scans a directory for JPG/JPEG files and generates comprehensive JSON metadata for any images missing their corresponding `.json` files. It performs the same face analysis as the bulk downloader, making it perfect for adding metadata to existing face image collections.

## 🚀 Quick Start

```bash
# Generate metadata for ./faces directory (default)
python3 generate_missing_metadata.py

# Generate metadata for specific directory
python3 generate_missing_metadata.py -d faces_quick

# Auto-confirm without prompts (for automation)
python3 generate_missing_metadata.py -d faces_quick -y

# Regenerate all metadata (overwrite existing JSON)
python3 generate_missing_metadata.py -d faces_quick --force -y

# Dry run - see what would be done
python3 generate_missing_metadata.py -d faces_quick --dry-run
```

## 🎯 Usage

### Basic Usage

```bash
python3 generate_missing_metadata.py [OPTIONS]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--directory DIR` | `-d DIR` | Directory containing face images (default: `./faces`) |
| `--force` | `-f` | Regenerate metadata even if JSON already exists |
| `--yes` | `-y` | Auto-confirm without prompting (for automation) |
| `--dry-run` | | Show what would be done without creating files |

### Examples

**1. Generate missing metadata for default directory:**
```bash
python3 generate_missing_metadata.py
```

**2. Generate for specific directory with auto-confirm:**
```bash
python3 generate_missing_metadata.py -d faces_dataset -y
```

**3. Regenerate all metadata (overwrite existing):**
```bash
python3 generate_missing_metadata.py -d faces_old --force -y
```

**4. Check what would be generated:**
```bash
python3 generate_missing_metadata.py -d faces_test --dry-run
```

**5. Batch process multiple directories:**
```bash
for dir in faces_*; do
    python3 generate_missing_metadata.py -d "$dir" -y
done
```

## 📊 What Gets Generated

Each JSON file includes:

### ✅ Basic Information
- Filename and file path
- Face ID (extracted from filename or timestamp)
- MD5 hash (for deduplication)
- Download timestamp (generation time)
- File size (bytes and KB)

### ✅ Image Properties
- Dimensions (width × height)
- Format (JPEG, PNG, etc.)
- Color mode (RGB, RGBA, etc.)

### ✅ Face Analysis (if available)
- **Demographics**: sex, age group, estimated age
- **Physical Features**: skin tone, skin color, hair color
- **Image Quality**: brightness, contrast, saturation
- **Face Detection**: number of faces, bounding boxes
- **Dominant Colors**: BGR color values

### ✅ Queryable Attributes
- Brightness level (bright/dark)
- Image quality (high/medium)
- Has face (true/false)
- Face count
- All demographic attributes

### ✅ Metadata Info
- Generation timestamp
- Generator script name
- Retroactive flag (indicates post-generation)

## 📄 Sample JSON Output

```json
{
  "filename": "face_20251101_220814_034_3a2c124f.jpg",
  "file_path": "faces_quick/face_20251101_220814_034_3a2c124f.jpg",
  "face_id": "20251101_220814_034",
  "md5_hash": "3a2c124f60ec6a7f0a8c17fff3bd41e8",
  "download_timestamp": "2025-11-01T22:16:02.398397",
  "download_date": "20251101 22:08:14",
  "source_url": "unknown (generated retroactively)",
  "file_size_kb": 521.73,
  "image_properties": {
    "width": 1024,
    "height": 1024,
    "format": "JPEG",
    "mode": "RGB"
  },
  "face_features": {
    "brightness": 140.37,
    "contrast": 36.73,
    "faces_detected": 1,
    "face_regions": [[154, 183, 721, 721]],
    "skin_tone": "medium",
    "hair_color": "light_gray",
    "age_group": "senior",
    "estimated_age": "60+",
    "estimated_sex": "male"
  },
  "queryable_attributes": {
    "brightness_level": "dark",
    "image_quality": "medium",
    "has_face": true,
    "face_count": 1,
    "sex": "male",
    "age_group": "senior",
    "skin_tone": "medium"
  },
  "metadata_generated": {
    "generated_at": "2025-11-01T22:16:02.398397",
    "generator": "generate_missing_metadata.py",
    "retroactive": true
  }
}
```

## 🎬 Example Session

```bash
$ python3 generate_missing_metadata.py -d faces_quick -y

✓ Face analyzer initialized

Scanning directory: /home/pi/hybridrag/faces8/faces_quick

Found:
  Total images: 106
  Already have JSON: 0
  Need metadata: 106

Generating metadata...

  Processing images... ━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:22

============================================================
METADATA GENERATION COMPLETE!
============================================================

 Total Images Found              106
 Already Had JSON                  0
 ✅ Metadata Generated           106
 ❌ Errors                         0

Directory: /home/pi/hybridrag/faces8/faces_quick

Final counts:
  JPG files: 106
  JSON files: 106

✓ All images now have metadata!
============================================================
```

## ⚙️ How It Works

1. **Scan Directory**: Finds all `.jpg` and `.jpeg` files
2. **Check Existing**: Skips images that already have `.json` files (unless `--force` used)
3. **Extract Info**: Parses filename for timestamp or uses file modification time
4. **Calculate Hash**: Computes MD5 hash of image file
5. **Analyze Image**: Opens image to get dimensions, format, etc.
6. **Face Detection**: If available, analyzes face features, demographics
7. **Generate JSON**: Creates comprehensive metadata file
8. **Save**: Writes JSON file with same base name as image

## 🔄 Integration

### Use with Analyze Tool

```bash
# Generate metadata
python3 generate_missing_metadata.py -d faces_dataset -y

# Analyze the metadata
python3 analyze_metadata.py faces_dataset
```

### Use with Bulk Downloader

```bash
# Download images without metadata (faster)
python3 bulk_download_cli.py -n 500 -o dataset

# Generate metadata later
python3 generate_missing_metadata.py -d dataset -y
```

## 📈 Performance

**Test Results: 106 faces**
- Time: ~22 seconds
- Speed: ~4.8 faces/second
- Success Rate: 100%
- Errors: 0

**Performance Factors:**
- Face analysis is CPU-intensive
- Faster on SSD than HDD
- Scales linearly with image count

**Estimated Times:**
- 100 images: ~20-25 seconds
- 500 images: ~2 minutes
- 1000 images: ~4 minutes
- 5000 images: ~20 minutes

## 🛡️ Smart Features

### ✅ Duplicate Detection
- Skips images that already have JSON (unless `--force`)
- Shows count of already-processed images
- Safe to run multiple times

### ✅ Filename Parsing
- Extracts face ID from filename if formatted correctly
- Falls back to file modification time
- Format: `face_YYYYMMDD_HHMMSS_mmm_HASH.jpg`

### ✅ Error Handling
- Continues on individual file errors
- Reports error count at end
- Does not fail entire batch on single error

### ✅ Progress Tracking
- Real-time progress bar
- Shows processing speed
- Displays final statistics

### ✅ Non-Interactive Mode
- Use `-y` flag for automation
- No prompts, runs silently
- Perfect for scripts and cron jobs

## 🚨 Troubleshooting

### Issue: "Face analyzer not available"

**Solution:**
```bash
# Install required packages
pip3 install opencv-python pillow numpy
```

The script will still work but metadata will be basic (no face analysis).

### Issue: "Directory not found"

**Solution:**
Check the directory path:
```bash
ls -d faces_directory
python3 generate_missing_metadata.py -d ./faces_directory
```

### Issue: Some JSON files are empty

**Solution:**
Regenerate with `--force`:
```bash
python3 generate_missing_metadata.py -d directory --force -y
```

### Issue: Slow processing

**Causes & Solutions:**
- Face analysis is CPU-intensive (normal)
- HDD slower than SSD (use SSD if available)
- Large images take longer (resize if needed)

## 🔧 Advanced Usage

### Batch Process Multiple Directories

```bash
#!/bin/bash
for dir in faces_batch_*; do
    echo "Processing $dir..."
    python3 generate_missing_metadata.py -d "$dir" -y
    echo "Done with $dir"
    echo ""
done
```

### Cron Job (Daily Metadata Generation)

```bash
# Add to crontab: generate metadata for new images daily at 2 AM
0 2 * * * cd /path/to/faces8 && python3 generate_missing_metadata.py -d ./faces -y >> metadata_gen.log 2>&1
```

### Selective Regeneration

```bash
# Delete all JSON files in directory
rm faces_directory/*.json

# Regenerate all metadata
python3 generate_missing_metadata.py -d faces_directory -y
```

### Integration with Download Scripts

```bash
# Download without metadata (faster)
./quick_download.sh 1000

# Generate metadata afterwards
python3 generate_missing_metadata.py -d faces_quick -y
```

## 📋 Requirements

### Required
- Python 3.8+
- `rich` library (for CLI display)
- `Pillow` (PIL) for image processing

### Optional (for face analysis)
- `opencv-python` - Face detection
- `numpy` - Numerical processing
- `core.py` - Face analyzer module

### Installation

```bash
pip3 install rich pillow opencv-python numpy
```

## 🎯 Use Cases

**1. Retroactive Metadata for Old Collections**
```bash
python3 generate_missing_metadata.py -d old_faces_archive -y
```

**2. Adding Analysis to Downloaded Images**
```bash
# Downloaded from another source
python3 generate_missing_metadata.py -d external_dataset -y
```

**3. Rebuilding Corrupted Metadata**
```bash
# Delete corrupted JSON files first
rm faces_dir/*.json
# Regenerate all
python3 generate_missing_metadata.py -d faces_dir -y
```

**4. Research Dataset Preparation**
```bash
# Generate metadata for analysis
python3 generate_missing_metadata.py -d research_faces -y
# Analyze demographics
python3 analyze_metadata.py research_faces
```

## 📝 Notes

- **Default directory**: `./faces` (same as original system)
- **JSON format**: Compatible with `bulk_download_cli.py` output
- **Face ID**: Extracted from filename or file timestamp
- **Source URL**: Marked as "unknown (generated retroactively)"
- **Safe to run**: Skips existing JSON by default
- **Force mode**: Use `--force` to regenerate all

## 🔗 Related Tools

- **`bulk_download_cli.py`** - Download faces with metadata
- **`analyze_metadata.py`** - Analyze metadata statistics
- **`download_faces.sh`** - Interactive download menu
- **`quick_download.sh`** - Fast download without metadata

## 💡 Tips

1. **Run dry-run first**: Use `--dry-run` to preview
2. **Use -y for automation**: Skips prompts
3. **Check before force**: `--force` overwrites all JSON
4. **Batch large directories**: Process in chunks if needed
5. **Verify results**: Use `analyze_metadata.py` to check

---

**Ready to generate metadata? Run the script!** 🚀

```bash
python3 generate_missing_metadata.py -d your_directory -y
```
