# Quick Start Guide - Bulk Face Downloader

## TL;DR - Get Started in 30 Seconds

```bash
# 1. Test download speed (optional)
python3 test_download_speed.py

# 2. Download 100 faces with 8 threads
python3 bulk_download_cli.py

# 3. Download more with custom settings
python3 bulk_download_cli.py -n 500 -t 16 -o my_faces
```

## What's New?

### ✨ High-Performance CLI Bulk Downloader

A brand new command-line tool optimized for downloading large batches of face images with:

- **Multi-threading**: 8 threads by default (configurable)
- **Live Dashboard**: Real-time progress bars, statistics, and system monitoring
- **Smart Features**: Automatic deduplication, error handling, retry logic
- **Resource Monitoring**: CPU, Memory, Disk, Network tracking
- **Fast Performance**: Up to 0.5+ faces/second (vs 0.4 faces/2s in original)

### 📊 Download Speed Analysis

The `test_download_speed.py` script benchmarks both sources:

**Results:**
- **thispersondoesnotexist.com** ⚡ FASTEST
  - 281.8 KB/s average speed
  - 100% success rate
  - 2.02s average per image

- **100k-faces.vercel.app**
  - 145.5 KB/s average speed
  - 80% success rate
  - 3.48s average per image

## System Information

Your system has:
- **CPU Cores**: 8
- **Memory**: 15 GB total, 12 GB available
- **Python**: 3.12.3
- **Packages**: requests, rich, tqdm, psutil (all installed ✓)

## Usage Examples

### Basic Usage

```bash
# Download 100 faces (default)
python3 bulk_download_cli.py

# Download 50 faces with 4 threads
python3 bulk_download_cli.py -n 50 -t 4

# Download 1000 faces with maximum threads
python3 bulk_download_cli.py -n 1000 -t 16
```

### Advanced Usage

```bash
# Use alternative source
python3 bulk_download_cli.py -s 100k-faces -n 200

# Custom output directory
python3 bulk_download_cli.py -o dataset_faces -n 500

# Increase timeout for slow connections
python3 bulk_download_cli.py -n 100 --timeout 60

# Maximum performance (for your 8-core system)
python3 bulk_download_cli.py -n 1000 -t 16 -o bulk_dataset
```

### Real Example Output

```
Starting bulk download...
Source: thispersondoesnotexist
Threads: 4
Target: 20 downloads
Output: faces_test

╭──────────────────────────────────────╮
│   🚀 HIGH-PERFORMANCE FACE DOWNLOADER   │
╰──────────────────────────────────────╯
╭──────────────────────────────────────╮
│ Downloading faces... ████████ 100%  │
╰──────────────────────────────────────╯

📈 Statistics
✅ Successful Downloads      21
🔄 Total Attempts            35
🔁 Duplicates                11
❌ Errors                     1
📦 Total Downloaded      11.20 MB
⚡ Avg Speed            122.0 KB/s
⏱️  Avg Time/Download      2.16s
📊 Downloads/Second        0.44

⚙️  System Resources
🔧 Active Threads          5/5
💻 CPU Usage             16.3%
🧠 Memory Usage     2.5/15.5 GB
💾 Disk Free           125 GB
```

## Performance Tips

### Optimal Settings for Your System

```bash
# Conservative (low resource usage)
python3 bulk_download_cli.py -n 100 -t 4

# Balanced (recommended)
python3 bulk_download_cli.py -n 500 -t 8

# Aggressive (maximum speed)
python3 bulk_download_cli.py -n 1000 -t 16

# Extreme (use with caution)
python3 bulk_download_cli.py -n 5000 -t 32
```

### Thread Guidelines

Your system has **8 CPU cores**:
- **4 threads**: Conservative, low CPU usage (~20-30%)
- **8 threads**: Optimal for balanced performance (~40-50%)
- **16 threads**: Maximum throughput (~60-80%)
- **32 threads**: Overkill, may cause throttling

## File Output

### Naming Convention

Files are saved with descriptive names:
```
face_20251101_213940_019_86d3e846.jpg
     │        │         │   └─ MD5 hash (first 8 chars)
     │        │         └───── Milliseconds
     │        └─────────────── Time (HHMMSS)
     └──────────────────────── Date (YYYYMMDD)
```

### Directory Structure

```
faces_bulk/
├── face_20251101_213939_723_7cad9ccf.jpg  (597 KB)
├── face_20251101_213940_019_86d3e846.jpg  (538 KB)
├── face_20251101_214013_507_9a020c5e.jpg  (536 KB)
└── ... (more files)
```

## Common Use Cases

### 1. Quick Dataset Creation

```bash
# Download 100 faces for testing
python3 bulk_download_cli.py -n 100 -o test_dataset
```

### 2. Large Dataset Collection

```bash
# Download 5000 faces over time
python3 bulk_download_cli.py -n 5000 -t 12 -o large_dataset
```

### 3. Daily Automated Downloads

```bash
# Add to cron: download 100 faces daily
0 2 * * * cd /home/pi/hybridrag/faces8 && python3 bulk_download_cli.py -n 100 -o daily_$(date +\%Y\%m\%d)
```

### 4. Batch Processing

```bash
# Download in multiple batches
for i in {1..10}; do
    python3 bulk_download_cli.py -n 100 -o batch_$i -t 8
    sleep 60  # 1-minute cooldown
done
```

## Monitoring & Debugging

### Real-Time System Monitoring

```bash
# In another terminal, monitor resources
watch -n 1 'free -h && echo && top -bn1 | head -20'
```

### Check Download Progress

```bash
# Count downloaded files
ls faces_bulk/*.jpg | wc -l

# Check total size
du -sh faces_bulk/

# View latest downloads
ls -lht faces_bulk/ | head -10
```

### Network Monitoring

```bash
# Monitor network usage
iftop -i eth0  # or your network interface

# Check network speed
speedtest-cli
```

## Troubleshooting Quick Fixes

### Downloads Too Slow?
```bash
# Increase threads
python3 bulk_download_cli.py -n 100 -t 16

# Use faster source
python3 bulk_download_cli.py -s thispersondoesnotexist
```

### Too Many Errors?
```bash
# Increase timeout
python3 bulk_download_cli.py -n 100 --timeout 60

# Reduce threads
python3 bulk_download_cli.py -n 100 -t 4
```

### High CPU Usage?
```bash
# Reduce threads
python3 bulk_download_cli.py -n 100 -t 2
```

### Running Out of Disk Space?
```bash
# Check available space
df -h

# Download to different location
python3 bulk_download_cli.py -o /mnt/external/faces -n 100
```

## Files Created

1. **test_download_speed.py** - Speed benchmarking tool
2. **bulk_download_cli.py** - Main CLI downloader
3. **BULK_DOWNLOAD_README.md** - Comprehensive documentation
4. **QUICK_START.md** - This file (quick reference)

## Next Steps

### Option 1: Start Downloading
```bash
python3 bulk_download_cli.py -n 100
```

### Option 2: Test Speed First
```bash
python3 test_download_speed.py
```

### Option 3: Large Dataset Collection
```bash
# Download 1000 faces with optimal settings
python3 bulk_download_cli.py -n 1000 -t 8 -o dataset_1k
```

### Option 4: Integration with Main System
```bash
# Download to faces directory used by main GUI
python3 bulk_download_cli.py -n 500 -o faces -t 8
```

## Performance Estimates

Based on test results with **4 threads** downloading **21 faces in 47.8s**:

| Faces | Threads | Est. Time | Est. Size |
|-------|---------|-----------|-----------|
| 100 | 4 | ~4 min | ~50 MB |
| 100 | 8 | ~2 min | ~50 MB |
| 500 | 8 | ~12 min | ~250 MB |
| 1000 | 8 | ~25 min | ~500 MB |
| 1000 | 16 | ~15 min | ~500 MB |
| 5000 | 16 | ~75 min | ~2.5 GB |

*Note: Actual times vary based on network speed, duplicates, and errors*

## Key Features Comparison

| Feature | Original faces.py | New CLI Tool |
|---------|------------------|--------------|
| Interface | GUI (Tkinter) | CLI (Rich) |
| Threads | 1 | 1-32 (configurable) |
| Speed | ~1 face/2.5s | ~0.5 face/s |
| Monitoring | Basic | Comprehensive |
| Resources | High (GUI) | Low (CLI only) |
| Scalability | Limited | Excellent |
| Automation | Manual | Script-friendly |

## Get Help

```bash
# Show all options
python3 bulk_download_cli.py --help

# Read full documentation
cat BULK_DOWNLOAD_README.md

# Check system resources
free -h && df -h && nproc
```

---

**Ready to download? Let's go! 🚀**

```bash
python3 bulk_download_cli.py -n 100 -t 8
```
