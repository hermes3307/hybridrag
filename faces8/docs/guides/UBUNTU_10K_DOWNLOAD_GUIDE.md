# Download 10,000 Faces on Ubuntu - Complete Guide

## 🚀 Quick Start (TL;DR)

```bash
# Option 1: Fast download without metadata (recommended for 10K)
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k

# Option 2: With metadata (slower but includes face analysis)
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k -m

# Option 3: Download fast, add metadata later
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k
python3 generate_missing_metadata.py -d faces_10k -y
```

---

## 📋 Prerequisites

### 1. Check System Requirements

```bash
# Check CPU cores
nproc
# Recommended: 8+ cores for 16 threads

# Check available RAM
free -h
# Recommended: 4+ GB free

# Check disk space
df -h .
# Required: ~6 GB for 10,000 faces
```

### 2. Install Dependencies (if not already installed)

```bash
# Update package list
sudo apt update

# Install Python 3 and pip
sudo apt install python3 python3-pip -y

# Install required Python packages
pip3 install requests rich psutil pillow opencv-python numpy
```

### 3. Verify Installation

```bash
# Check Python version
python3 --version
# Should be 3.8 or higher

# Verify packages
pip3 list | grep -E "(requests|rich|psutil)"
```

---

## 🎯 Step-by-Step Instructions

### Method 1: Fast Download (No Metadata) - RECOMMENDED

**Best for:** Just getting 10,000 face images quickly

```bash
# Navigate to directory
cd /home/pi/hybridrag/faces8

# Run bulk download
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k

# Expected time: ~60-75 minutes
# Expected size: ~5.5 GB
```

**Command breakdown:**
- `-n 10000` = Download 10,000 faces
- `-t 16` = Use 16 worker threads (adjust based on your CPU)
- `-o faces_10k` = Output to `faces_10k/` directory

---

### Method 2: Download with Metadata - COMPREHENSIVE

**Best for:** Need face analysis data (age, gender, skin tone, etc.)

```bash
# Download with metadata generation
python3 bulk_download_cli.py -n 10000 -t 12 -o faces_10k_meta -m

# Expected time: ~90-120 minutes
# Expected size: ~5.5 GB (images) + ~18 MB (JSON)
```

**Note:** Use fewer threads (12 instead of 16) because face analysis is CPU-intensive.

---

### Method 3: Two-Stage Process - FLEXIBLE

**Best for:** Maximum flexibility and efficiency

**Stage 1: Download images fast**
```bash
# Download 10,000 images without metadata
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k

# Time: ~60-75 minutes
```

**Stage 2: Generate metadata later**
```bash
# Generate metadata for all downloaded images
python3 generate_missing_metadata.py -d faces_10k -y

# Time: ~35-40 minutes for 10,000 images
```

**Total time:** ~95-115 minutes (similar to Method 2)
**Advantage:** Can download first, then decide if you need metadata

---

## ⚙️ Optimization for Ubuntu

### 1. Determine Optimal Thread Count

```bash
# Get CPU core count
CORES=$(nproc)
echo "CPU Cores: $CORES"

# Recommended thread count
THREADS=$((CORES * 2))
echo "Recommended threads: $THREADS"
```

**Thread Guidelines:**
- 4 cores → 8-12 threads
- 8 cores → 12-16 threads
- 16 cores → 16-24 threads
- 32+ cores → 24-32 threads

### 2. Check Network Speed

```bash
# Install speedtest (if not installed)
sudo apt install speedtest-cli -y

# Test your network
speedtest-cli

# Recommended: 10+ Mbps for smooth downloading
```

### 3. Optimize for SSD vs HDD

**If using SSD (recommended):**
```bash
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k
```

**If using HDD (slower):**
```bash
# Use fewer threads to reduce I/O contention
python3 bulk_download_cli.py -n 10000 -t 8 -o faces_10k
```

---

## 🖥️ Running in Background

### Option 1: Using nohup

```bash
# Start download in background
nohup python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k > download.log 2>&1 &

# Get process ID
echo $!

# Monitor progress
tail -f download.log

# Check if still running
ps aux | grep bulk_download
```

### Option 2: Using screen

```bash
# Install screen
sudo apt install screen -y

# Start screen session
screen -S face_download

# Run download
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k

# Detach: Press Ctrl+A then D

# Reattach later
screen -r face_download

# List sessions
screen -ls
```

### Option 3: Using tmux

```bash
# Install tmux
sudo apt install tmux -y

# Start tmux session
tmux new -s face_download

# Run download
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k

# Detach: Press Ctrl+B then D

# Reattach later
tmux attach -t face_download
```

---

## 📊 Monitoring Progress

### Real-time Monitoring

```bash
# In another terminal, watch file count
watch -n 5 'ls faces_10k/*.jpg 2>/dev/null | wc -l'

# Check directory size
watch -n 10 'du -sh faces_10k'

# Monitor system resources
htop
# or
top
```

### Check Download Stats

```bash
# Count downloaded files
ls faces_10k/*.jpg | wc -l

# Total size
du -sh faces_10k/

# Average file size
du -sh faces_10k/ | awk '{print $1}'
```

---

## 🔧 Advanced Usage

### 1. Batch Download in Chunks

**Download in 10 batches of 1000:**

```bash
#!/bin/bash
for i in {1..10}; do
    echo "=== Batch $i/10 ==="
    python3 bulk_download_cli.py -n 1000 -t 16 -o faces_batch_$i
    echo "Batch $i complete. Waiting 60 seconds..."
    sleep 60
done

echo "All batches complete!"
```

**Advantages:**
- Can stop and resume
- Organize into smaller directories
- Avoid overwhelming the source server

### 2. Download with Rate Limiting

```bash
# Use fewer threads to be gentler on the server
python3 bulk_download_cli.py -n 10000 -t 8 -o faces_10k
```

### 3. Download Different Sources

```bash
# Use 100k-faces source (alternative)
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k -s 100k-faces

# Note: thispersondoesnotexist is usually faster
```

---

## 📈 Expected Performance

### Download Speed Estimates

Based on test results with 8 threads:

| Configuration | Speed | Time for 10K | Notes |
|---------------|-------|--------------|-------|
| 8 threads, no metadata | 2.3 faces/s | ~72 min | Balanced |
| 16 threads, no metadata | ~3.5 faces/s | ~48 min | Fast |
| 24 threads, no metadata | ~4.0 faces/s | ~42 min | Maximum |
| 12 threads, with metadata | ~1.5 faces/s | ~111 min | Slower |

**Factors affecting speed:**
- Network speed
- CPU cores
- Disk type (SSD vs HDD)
- Server response time
- Duplicate rate

### Resource Usage

**Expected resource usage (16 threads):**
```
CPU: 30-50%
Memory: 2-4 GB
Network: 1-3 Mbps
Disk I/O: 5-10 MB/s
```

---

## 🛡️ Error Handling

### Common Issues

**1. Network timeout:**
```bash
# Increase timeout
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k --timeout 60
```

**2. Disk full:**
```bash
# Check before starting
df -h .

# Clean up space if needed
rm -rf old_directory/
```

**3. Too many duplicates:**
```bash
# This is normal, the script handles it automatically
# Duplicates are skipped, download continues
```

**4. Process killed:**
```bash
# Check memory
free -h

# Reduce threads if out of memory
python3 bulk_download_cli.py -n 10000 -t 8 -o faces_10k
```

---

## 📁 Post-Download Tasks

### 1. Verify Download

```bash
# Count files
TOTAL=$(ls faces_10k/*.jpg 2>/dev/null | wc -l)
echo "Downloaded: $TOTAL faces"

# Check size
du -sh faces_10k/

# Verify file integrity
file faces_10k/*.jpg | head -10
```

### 2. Generate Metadata (if not done during download)

```bash
# Generate JSON metadata for all images
python3 generate_missing_metadata.py -d faces_10k -y

# Expected time: ~35-40 minutes
```

### 3. Analyze Dataset

```bash
# Analyze metadata statistics
python3 analyze_metadata.py faces_10k

# Shows demographics, age distribution, etc.
```

### 4. Organize Files

```bash
# Move to final location
mv faces_10k /path/to/final/location/

# Or compress for backup
tar -czf faces_10k.tar.gz faces_10k/
```

---

## 🎬 Complete Example Session

```bash
# 1. Check system
echo "=== System Check ==="
echo "CPU Cores: $(nproc)"
echo "Free Memory: $(free -h | grep Mem | awk '{print $4}')"
echo "Free Disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# 2. Navigate to directory
cd /home/pi/hybridrag/faces8

# 3. Test download speed (optional)
echo "=== Testing Download Speed ==="
python3 test_download_speed.py
echo ""

# 4. Start download in screen session
echo "=== Starting Download ==="
screen -dmS face_download bash -c "python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k 2>&1 | tee download.log"

echo "Download started in background!"
echo "Monitor with: screen -r face_download"
echo "Or check log: tail -f download.log"
echo ""

# 5. Monitor progress (in another terminal)
# watch -n 5 'ls faces_10k/*.jpg 2>/dev/null | wc -l'
```

---

## 🔥 Quick Reference Commands

```bash
# Start download
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k

# Start in background
nohup python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k > download.log 2>&1 &

# Check progress
ls faces_10k/*.jpg | wc -l

# Monitor live
watch -n 5 'ls faces_10k/*.jpg 2>/dev/null | wc -l'

# Generate metadata later
python3 generate_missing_metadata.py -d faces_10k -y

# Analyze dataset
python3 analyze_metadata.py faces_10k
```

---

## 📝 Summary

### Recommended Command for Ubuntu

```bash
# Best balance of speed and reliability
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k
```

### Expected Results

```
✅ Downloaded: 10,000+ faces (allowing for duplicates)
📦 Total Size: ~5.5 GB
⏱️  Time: ~48-72 minutes (depending on system)
💾 Location: ./faces_10k/
```

### Next Steps

1. **Verify**: Check file count and size
2. **Metadata**: Generate JSON metadata if needed
3. **Analyze**: Run analysis on demographics
4. **Backup**: Compress or move to final location

---

## 💡 Pro Tips

1. **Use SSD** for faster I/O
2. **Wired connection** faster than WiFi
3. **Run overnight** for large downloads
4. **Use screen/tmux** to prevent disconnection
5. **Monitor logs** to catch issues early
6. **Start small** (test with 100 first)
7. **Check disk space** before starting
8. **Use 16 threads** on 8-core systems
9. **Download first, metadata later** for flexibility
10. **Keep terminal open** or use background process

---

**Ready to download 10,000 faces? Start with the quick command above!** 🚀
