# Shell Scripts for Face Bulk Downloader

Easy-to-use shell scripts for downloading face images in bulk.

## 📜 Available Scripts

### 1. Interactive Menu (`download_faces.sh`)

**The easiest way to get started!**

```bash
./download_faces.sh
```

Features:
- ✅ Interactive menu with preset configurations
- ✅ Quick test, small, medium, and large dataset options
- ✅ With or without metadata generation
- ✅ Custom download wizard
- ✅ Built-in speed testing
- ✅ Metadata analysis tool
- ✅ Color-coded output

**Menu Options:**
1. Quick Test (10 faces, no metadata)
2. Small Dataset (50 faces, no metadata)
3. Medium Dataset (100 faces, no metadata)
4. Large Dataset (500 faces, no metadata)
5. Quick Test with Metadata (10 faces + JSON)
6. Small Dataset with Metadata (50 faces + JSON)
7. Medium Dataset with Metadata (100 faces + JSON)
8. Large Dataset with Metadata (500 faces + JSON)
9. Custom Download (specify all parameters)
- t) Test Download Speed
- a) Analyze Metadata
- h) Show Help
- q) Quit

---

### 2. Quick Download (`quick_download.sh`)

**Fast and simple - just specify how many faces you want!**

```bash
# Download 100 faces (default)
./quick_download.sh

# Download 500 faces with 12 threads
./quick_download.sh 500 12

# Download 200 faces, 8 threads, custom output
./quick_download.sh 200 8 my_faces
```

**Usage:**
```bash
./quick_download.sh [num_faces] [threads] [output_dir]
```

**Parameters:**
- `num_faces`: Number of faces to download (default: 100)
- `threads`: Number of worker threads (default: 8)
- `output_dir`: Output directory name (default: faces_quick)

**Examples:**
```bash
./quick_download.sh 50              # 50 faces, 8 threads, faces_quick/
./quick_download.sh 200 16          # 200 faces, 16 threads, faces_quick/
./quick_download.sh 100 8 dataset   # 100 faces, 8 threads, dataset/
```

---

### 3. Download with Metadata (`download_with_metadata.sh`)

**Download faces with full JSON metadata and face analysis!**

```bash
# Download 100 faces with metadata (default)
./download_with_metadata.sh

# Download 200 faces with 12 threads
./download_with_metadata.sh 200 12

# Download 500 faces, custom output
./download_with_metadata.sh 500 8 research_data
```

**Usage:**
```bash
./download_with_metadata.sh [num_faces] [threads] [output_dir]
```

**Parameters:**
- `num_faces`: Number of faces to download (default: 100)
- `threads`: Number of worker threads (default: 8)
- `output_dir`: Output directory name (default: faces_metadata)

**What you get:**
- ✅ JPG images (1024x1024)
- ✅ JSON metadata files with:
  - Demographics (age, gender, skin tone)
  - Face features (brightness, contrast)
  - Face detection results
  - HTTP headers and download info

**Examples:**
```bash
./download_with_metadata.sh 50                    # Basic usage
./download_with_metadata.sh 200 16                # More faces, faster
./download_with_metadata.sh 100 8 research_v1     # Custom directory
```

---

### 4. Large Dataset Download (`download_large_dataset.sh`)

**Optimized for downloading 1000+ faces with high performance!**

```bash
# Download 1000 faces (default)
./download_large_dataset.sh

# Download 5000 faces
./download_large_dataset.sh 5000

# Download 2000 faces, custom output
./download_large_dataset.sh 2000 faces_2k
```

**Usage:**
```bash
./download_large_dataset.sh [num_faces] [output_dir]
```

**Parameters:**
- `num_faces`: Number of faces to download (default: 1000)
- `output_dir`: Output directory name (default: faces_dataset_large)

**Features:**
- ✅ Automatically uses 16 threads for maximum speed
- ✅ Disk space check before download
- ✅ Estimated time calculation
- ✅ Progress tracking
- ✅ Final statistics summary

**Examples:**
```bash
./download_large_dataset.sh 1000                  # 1K faces
./download_large_dataset.sh 5000                  # 5K faces
./download_large_dataset.sh 10000 faces_10k       # 10K faces
```

**Performance:**
- 1000 faces: ~7-8 minutes
- 5000 faces: ~35-40 minutes
- 10000 faces: ~70-75 minutes

---

## 🚀 Quick Start Examples

### Scenario 1: Just Testing
```bash
# Quick test with 10 faces
./quick_download.sh 10
```

### Scenario 2: Building a Dataset
```bash
# Download 500 faces with metadata for research
./download_with_metadata.sh 500 12 research_dataset
```

### Scenario 3: Large Scale Collection
```bash
# Download 5000 faces for machine learning
./download_large_dataset.sh 5000 ml_training_data
```

### Scenario 4: Interactive Exploration
```bash
# Use the interactive menu
./download_faces.sh
# Then select option 7 (Medium Dataset with Metadata)
```

---

## 📊 Comparison Table

| Script | Best For | Speed | Metadata | Complexity |
|--------|----------|-------|----------|------------|
| `download_faces.sh` | Beginners, exploration | Medium | Optional | Easy (menu) |
| `quick_download.sh` | Quick tests, simple needs | Fast | No | Very Easy |
| `download_with_metadata.sh` | Research, analysis | Medium | Yes | Easy |
| `download_large_dataset.sh` | Large datasets | Fastest | No | Easy |

---

## 🎯 Recommended Workflows

### For Beginners
```bash
# Start with the interactive menu
./download_faces.sh
```

### For Quick Downloads
```bash
# Just download 100 faces fast
./quick_download.sh 100
```

### For Research/Analysis
```bash
# Download with full metadata
./download_with_metadata.sh 200 8 study_2024
# Then analyze
python3 analyze_metadata.py study_2024
```

### For Large Datasets
```bash
# Download 5000 faces
./download_large_dataset.sh 5000 training_set
```

---

## 🔧 Advanced Usage

### Combining Scripts

**Test speed first, then download:**
```bash
# Test which source is faster
./download_faces.sh   # Select option 't'

# Then do a large download
./download_large_dataset.sh 1000
```

**Download and analyze:**
```bash
# Download with metadata
./download_with_metadata.sh 500 12 dataset_v1

# Analyze the results
python3 analyze_metadata.py dataset_v1
```

**Batch processing:**
```bash
# Download multiple datasets
for i in {1..5}; do
    ./quick_download.sh 100 8 "batch_$i"
    sleep 30  # Cooldown between batches
done
```

---

## 📁 Output Structure

All scripts create organized output directories:

```
faces_quick/           (from quick_download.sh)
├── face_*.jpg

faces_metadata/        (from download_with_metadata.sh)
├── face_*.jpg
└── face_*.json

faces_dataset_large/   (from download_large_dataset.sh)
├── face_*.jpg
```

---

## ⚙️ Script Parameters Summary

### Thread Recommendations
- **4 threads**: Conservative, low CPU (~20%)
- **8 threads**: Balanced, optimal for most cases (~17%)
- **16 threads**: Maximum speed, higher CPU (~40%)
- **32 threads**: Overkill, not recommended

### Download Sizes
- 10 faces ≈ 5 MB
- 50 faces ≈ 25 MB
- 100 faces ≈ 50 MB
- 500 faces ≈ 250 MB
- 1000 faces ≈ 500 MB
- 5000 faces ≈ 2.5 GB

### Download Times (approximate, 8 threads)
- 10 faces: ~5 seconds
- 50 faces: ~20 seconds
- 100 faces: ~45 seconds
- 500 faces: ~4 minutes
- 1000 faces: ~8 minutes

---

## 🛠️ Troubleshooting

### Script won't run
```bash
# Make sure scripts are executable
chmod +x *.sh
```

### Slow downloads
```bash
# Test speed first
./download_faces.sh   # Option 't'

# Use more threads
./quick_download.sh 100 16
```

### Out of disk space
```bash
# Check available space
df -h

# The large dataset script checks automatically
./download_large_dataset.sh 5000
```

### Want to stop a download
```bash
# Press Ctrl+C to stop
# Already downloaded files are preserved
```

---

## 📖 Direct Python Usage

If you prefer direct command-line control:

```bash
# Basic download
python3 bulk_download_cli.py -n 100

# With metadata
python3 bulk_download_cli.py -n 100 -m

# Full options
python3 bulk_download_cli.py -n 500 -t 16 -o dataset -m -s thispersondoesnotexist
```

**See:** `python3 bulk_download_cli.py --help`

---

## 🎓 Tips & Best Practices

1. **Start small**: Test with 10-50 faces before large downloads
2. **Use metadata**: For research/analysis, always use `-m` flag
3. **Monitor resources**: Check CPU/memory during first run
4. **Batch downloads**: For 10K+ faces, download in batches of 1000
5. **Disk space**: Ensure ~1GB free per 1000 faces
6. **Network**: Wired connection recommended for large datasets

---

## 📞 Quick Reference

```bash
# Interactive menu
./download_faces.sh

# Quick download
./quick_download.sh [num] [threads] [output]

# With metadata
./download_with_metadata.sh [num] [threads] [output]

# Large dataset
./download_large_dataset.sh [num] [output]

# Test speed
python3 test_download_speed.py

# Analyze data
python3 analyze_metadata.py [directory]

# Direct control
python3 bulk_download_cli.py --help
```

---

**Ready to download? Pick a script and run it! 🚀**
