# JSON Metadata Generation Guide

## ✅ **YES! You Have JSON Generation Tools**

---

## 🎯 **3 Tools for JSON Generation**

### **1. Download with JSON** (for NEW images)

#### Shell Script:
```bash
./download_with_metadata.sh [num_faces] [threads] [output_dir]
```

#### Python Script:
```bash
python3 bulk_download_cli.py -n NUM -t THREADS -o OUTPUT -m
```

**What it does:**
- Downloads face images from AI services
- Automatically generates JSON metadata
- Analyzes face features (brightness, contrast, demographics)

**Examples:**
```bash
# Download 100 faces with JSON
./download_with_metadata.sh 100

# Download 1000 faces with 16 threads
./download_with_metadata.sh 1000 16 my_faces

# Python version
python3 bulk_download_cli.py -n 500 -t 8 -o faces -m
```

---

### **2. Generate JSON for EXISTING Images** (retroactive)

#### Script:
```bash
python3 generate_missing_metadata.py -d DIRECTORY
```

**What it does:**
- Scans directory for images WITHOUT JSON files
- Generates JSON metadata for each image
- Analyzes face features and demographics
- Creates complete metadata files

**Examples:**
```bash
# Check what would be generated (dry-run)
python3 generate_missing_metadata.py -d /home/pi/faces --dry-run

# Generate JSON for missing files
python3 generate_missing_metadata.py -d /home/pi/faces

# Auto-confirm without prompting
python3 generate_missing_metadata.py -d /home/pi/faces -y

# Force regenerate ALL JSON (even if exists)
python3 generate_missing_metadata.py -d /home/pi/faces --force -y
```

**Options:**
- `-d, --directory` - Directory with images (default: ./faces)
- `-f, --force` - Regenerate even if JSON exists
- `-y, --yes` - Auto-confirm without prompting
- `--dry-run` - Show what would be done

---

### **3. Analyze Existing JSON Files**

#### Script:
```bash
python3 analyze_metadata.py [directory]
```

**What it does:**
- Reads existing JSON files
- Shows statistics and analysis
- Reports metadata quality

**Example:**
```bash
python3 analyze_metadata.py /home/pi/faces
```

---

## 📄 **JSON File Structure**

Each image gets a corresponding JSON file:

```
face_20251101_123456_789_abc123.jpg   ← Image
face_20251101_123456_789_abc123.json  ← Metadata
```

### JSON Contents:

```json
{
  "filename": "face_20251101_123456_789_abc123.jpg",
  "file_path": "./faces/face_20251101_123456_789_abc123.jpg",
  "face_id": "20251101_123456_789",
  "md5_hash": "abc123def456...",
  "download_timestamp": "2025-11-01T12:34:56",
  "file_size_bytes": 605605,
  "file_size_kb": 591.41,

  "image_properties": {
    "width": 1024,
    "height": 1024,
    "format": "JPEG",
    "mode": "RGB",
    "dimensions": "1024x1024"
  },

  "face_features": {
    "brightness": 82.49,
    "contrast": 49.96,
    "saturation_mean": 72.53,
    "faces_detected": 1,
    "face_regions": [[159, 203, 701, 701]],
    "skin_tone": "medium",
    "skin_color": "medium",
    "hair_color": "brown",
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
    "estimated_age": "60+",
    "skin_tone": "medium",
    "hair_color": "brown"
  }
}
```

---

## 🎯 **Common Use Cases**

### Scenario 1: Download New Images with JSON
```bash
# Download 500 faces with metadata
./download_with_metadata.sh 500 8 my_faces

# Check results
ls my_faces/*.jpg | wc -l    # Count images
ls my_faces/*.json | wc -l   # Count JSON files
```

---

### Scenario 2: Generate JSON for Existing Images

**You have images but no JSON files:**

```bash
# Step 1: Check how many are missing JSON
python3 generate_missing_metadata.py -d /home/pi/faces --dry-run

# Step 2: Generate the JSON files
python3 generate_missing_metadata.py -d /home/pi/faces -y

# Step 3: Verify
ls /home/pi/faces/*.json | wc -l
```

---

### Scenario 3: Regenerate ALL JSON Files

**You want to regenerate JSON with updated analysis:**

```bash
# Backup existing JSON (optional)
mkdir json_backup
cp /home/pi/faces/*.json json_backup/

# Regenerate all JSON files
python3 generate_missing_metadata.py -d /home/pi/faces --force -y

# Check results
python3 analyze_metadata.py /home/pi/faces
```

---

### Scenario 4: Check Current Status

```bash
# Count images
find /home/pi/faces -name "*.jpg" | wc -l

# Count JSON files
find /home/pi/faces -name "*.json" | wc -l

# Analyze metadata
python3 analyze_metadata.py /home/pi/faces
```

---

## 📊 **Current Status Check**

### Check Your Faces Directory:

```bash
# Current status
cd /home/pi/faces

echo "Images: $(ls *.jpg 2>/dev/null | wc -l)"
echo "JSON files: $(ls *.json 2>/dev/null | wc -l)"

# Missing JSON files
python3 generate_missing_metadata.py -d /home/pi/faces --dry-run
```

### Your Current Status:
```
Total images:  29,730
Total JSON:    29,730
Missing JSON:  0
Status:        ✅ All images have JSON!
```

---

## 🚀 **Quick Commands**

### Generate JSON for images without it:
```bash
python3 generate_missing_metadata.py -d /home/pi/faces -y
```

### Download new images with JSON:
```bash
./download_with_metadata.sh 100
```

### Analyze existing JSON:
```bash
python3 analyze_metadata.py /home/pi/faces
```

### Force regenerate ALL JSON:
```bash
python3 generate_missing_metadata.py -d /home/pi/faces --force -y
```

---

## ⚙️ **Advanced Options**

### Dry Run (See What Would Happen):
```bash
python3 generate_missing_metadata.py -d /home/pi/faces --dry-run
```

### Process Specific Directory:
```bash
python3 generate_missing_metadata.py -d /path/to/images -y
```

### Regenerate with Force:
```bash
python3 generate_missing_metadata.py -d /home/pi/faces --force -y
```

---

## 📈 **Performance**

### Generation Speed:
- **Basic metadata**: ~0.1 seconds/image
- **With face analysis**: ~0.5-1 second/image
- **1000 images**: ~8-16 minutes

### Batch Processing:
```bash
# Process large directory
python3 generate_missing_metadata.py -d /home/pi/faces -y

# Progress shown in real-time
# ████████████████████████ 85% | 850/1000 | 2m 30s remaining
```

---

## 🛠️ **Troubleshooting**

### "Module not found" errors:
```bash
pip install Pillow rich
```

### "Face analysis not available":
```bash
# Optional - installs face analysis dependencies
pip install opencv-python numpy
```

### Permission errors:
```bash
chmod +x generate_missing_metadata.py
chmod 644 /home/pi/faces/*
```

### JSON validation:
```bash
# Test if JSON is valid
python3 -m json.tool /home/pi/faces/face_*.json | head -20
```

---

## 📁 **File Locations**

All scripts in: `/home/pi/hybridrag/faces8/`

- `generate_missing_metadata.py` - Generate JSON for existing images
- `download_with_metadata.sh` - Download with JSON
- `bulk_download_cli.py` - Python download script
- `analyze_metadata.py` - Analyze JSON files

---

## ✅ **Summary**

**You have 3 tools:**

1. **Download + JSON**: `./download_with_metadata.sh`
2. **Generate JSON**: `python3 generate_missing_metadata.py`
3. **Analyze JSON**: `python3 analyze_metadata.py`

**Your current status:**
- ✅ 29,730 images with JSON files
- ✅ All images have metadata
- ✅ Ready to use!

---

**Created by Claude Code** 🤖
