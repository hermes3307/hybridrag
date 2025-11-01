# High-Performance CLI Face Bulk Downloader

A highly optimized command-line tool for downloading face images in bulk with real-time graphical status display, multi-threading support, and comprehensive resource monitoring.

## Features

✨ **High Performance**
- Multi-threaded concurrent downloads (configurable threads)
- Automatic duplicate detection using MD5 hashing
- Optimized for maximum throughput
- Memory-efficient streaming downloads

📊 **Rich CLI Graphics**
- Real-time progress bars with ETA
- Live statistics dashboard
- System resource monitoring (CPU, Memory, Disk, Network)
- Download speed and rate tracking
- Error breakdown and reporting

🎯 **Smart Download Management**
- Automatic retry on failures
- Timeout handling
- Connection pooling
- Deduplication to avoid waste

🔧 **Flexible Configuration**
- Multiple download sources
- Configurable thread count
- Custom output directories
- Adjustable timeouts

## Installation

### Prerequisites

```bash
# Python 3.8+ required
python3 --version

# Install required packages
pip3 install requests rich psutil
```

## Usage

### Basic Usage

```bash
# Download 100 faces (default settings)
python3 bulk_download_cli.py

# Download with custom number
python3 bulk_download_cli.py -n 500

# Use more threads for faster downloads
python3 bulk_download_cli.py -n 1000 -t 16
```

### Advanced Options

```bash
# All options
python3 bulk_download_cli.py -n <num> -t <threads> -o <output> -s <source> --timeout <seconds>

# Examples:
python3 bulk_download_cli.py -n 500 -t 12 -o my_faces
python3 bulk_download_cli.py -s 100k-faces -n 200 -t 8
python3 bulk_download_cli.py -n 1000 -t 16 --timeout 60
```

### Command-Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--num` | `-n` | 100 | Number of faces to download |
| `--threads` | `-t` | 8 | Number of worker threads |
| `--output` | `-o` | faces_bulk | Output directory path |
| `--source` | `-s` | thispersondoesnotexist | Download source |
| `--timeout` | | 30 | Request timeout in seconds |

### Available Sources

1. **thispersondoesnotexist** (Recommended - Fastest)
   - URL: https://thispersondoesnotexist.com/
   - Speed: ~282 KB/s average
   - Success Rate: 100%
   - Image Size: ~500-600 KB

2. **100k-faces**
   - URL: https://100k-faces.vercel.app/
   - Speed: ~146 KB/s average
   - Success Rate: ~80%
   - Image Size: ~500 KB

## Performance Benchmarks

### Download Speed Test Results

```
thispersondoesnotexist.com:
  Average Time:  2.02s
  Average Size:  568.9 KB
  Average Speed: 281.8 KB/s
  Success Rate:  100%

100k-faces.vercel.app:
  Average Time:  3.48s
  Average Size:  506.0 KB
  Average Speed: 145.5 KB/s
  Success Rate:  80%

⚡ WINNER: thispersondoesnotexist.com
```

### System Requirements

**Recommended Configuration:**
- CPU: 4+ cores (8 threads optimal)
- RAM: 2 GB minimum, 4 GB recommended
- Network: Stable broadband (10+ Mbps)
- Disk: 1 GB free space per 1000 images

**Thread Recommendations:**
- 4 cores: 4-8 threads
- 8 cores: 8-16 threads
- 16+ cores: 16-32 threads

### Performance Metrics

Test run with 20 faces, 4 threads:
```
✅ Successfully Downloaded:  21 faces
📦 Total Size:              11.20 MB
⏱️  Total Time:              47.8s
⚡ Average Speed:           122.0 KB/s
📊 Download Rate:           0.44 faces/sec
```

Estimated performance for 1000 faces (8 threads):
- Time: ~25-30 minutes
- Size: ~500-600 MB
- Speed: ~0.5-0.7 faces/sec

## Features Deep Dive

### 1. Real-Time Dashboard

The CLI displays a live dashboard with:

```
🚀 HIGH-PERFORMANCE FACE DOWNLOADER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Downloading faces... ████████░░ 80% 800/1000

📈 Statistics
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric                ┃         Value ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━┫
┃ ✅ Successful        ┃           800 ┃
┃ 🔄 Total Attempts    ┃         1,250 ┃
┃ 🔁 Duplicates        ┃           420 ┃
┃ ❌ Errors            ┃            30 ┃
┃ 📦 Total Downloaded  ┃      450.5 MB ┃
┃ ⚡ Avg Speed         ┃    180.5 KB/s ┃
┃ 📊 Downloads/Second  ┃          0.65 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┛

⚙️  System Resources
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Resource        ┃            Usage ┃
┣━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━┫
┃ 🔧 Threads      ┃              9/9 ┃
┃ 💻 CPU          ┃            35.2% ┃
┃ 🧠 Memory       ┃    18.5% (2.8GB) ┃
┃ 💾 Disk Free    ┃          125.3GB ┃
┃ 📡 Network Sent ┃          2.5 MB  ┃
┃ 📥 Network Recv ┃        450.8 MB  ┃
┗━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━┛
```

### 2. Duplicate Detection

- MD5 hash computed for each image
- Automatic deduplication
- Prevents wasted bandwidth
- Saves disk space

### 3. Error Handling

Robust error handling with detailed reporting:
- Connection timeouts
- Network errors
- HTTP errors (with status codes)
- Automatic retry mechanism

### 4. Resource Monitoring

Real-time monitoring of:
- Active threads
- CPU usage percentage
- Memory consumption (used/total)
- Disk space available
- Network I/O (sent/received)

## Output Format

### File Naming Convention

```
face_YYYYMMDD_HHMMSS_mmm_HASH.jpg
```

Example:
```
face_20251101_213940_019_86d3e846.jpg
     │        │         │   │
     │        │         │   └─ First 8 chars of MD5 hash
     │        │         └───── Milliseconds
     │        └─────────────── Timestamp (HHMMSS)
     └──────────────────────── Date (YYYYMMDD)
```

### Directory Structure

```
faces_bulk/
├── face_20251101_213939_723_7cad9ccf.jpg
├── face_20251101_213940_019_86d3e846.jpg
├── face_20251101_214013_507_9a020c5e.jpg
└── ...
```

## Troubleshooting

### Issue: Downloads are slow

**Solutions:**
1. Increase thread count: `-t 16`
2. Check network connection
3. Try different source: `-s thispersondoesnotexist`
4. Increase timeout: `--timeout 60`

### Issue: High CPU usage

**Solutions:**
1. Reduce thread count: `-t 4`
2. Add delays between requests (modify code)
3. Monitor with `htop` or `top`

### Issue: Memory errors

**Solutions:**
1. Reduce thread count
2. Download in smaller batches
3. Check available RAM: `free -h`
4. Close other applications

### Issue: Connection timeouts

**Solutions:**
1. Increase timeout: `--timeout 60`
2. Check network stability
3. Reduce concurrent threads
4. Try different source

### Issue: Duplicates

This is normal! Sources may serve the same image multiple times. The tool automatically skips duplicates.

## Performance Optimization Tips

### 1. Optimal Thread Count

```bash
# Get CPU core count
nproc

# Recommended: 1-2x CPU cores
python3 bulk_download_cli.py -t $(nproc)
python3 bulk_download_cli.py -t $(($(nproc) * 2))
```

### 2. Network Optimization

- Use wired connection over WiFi
- Close bandwidth-heavy applications
- Check network speed: `speedtest-cli`

### 3. Disk Optimization

- Use SSD over HDD
- Ensure sufficient free space
- Use fast I/O filesystem (ext4, xfs)

### 4. System Resources

```bash
# Monitor resources while downloading
watch -n 1 'free -h && echo && df -h && echo && ps aux | grep bulk_download'
```

## Integration with Face Processing System

The downloaded faces can be directly used with the main face processing system:

```bash
# 1. Download faces in bulk
python3 bulk_download_cli.py -n 1000 -o faces_data

# 2. Process with main system (if using faces.py)
# The main GUI can process from faces_data/ directory
```

## Comparison with Original Downloader

| Feature | Original (faces.py) | Bulk CLI |
|---------|-------------------|----------|
| Interface | GUI (Tkinter) | CLI (Rich) |
| Threading | Single thread loop | Multi-threaded |
| Speed | ~1 face/2.5s | ~0.5 face/s (8 threads) |
| Monitoring | Basic stats | Full system metrics |
| Deduplication | Yes | Yes |
| Metadata | JSON files | Filenames only |
| Resource Usage | Higher (GUI) | Lower (CLI) |
| Scalability | Limited | Excellent |

## Advanced Usage

### Batch Processing Script

```bash
#!/bin/bash
# Download faces in batches

for i in {1..10}; do
    echo "Batch $i/10"
    python3 bulk_download_cli.py -n 100 -o batch_$i -t 8
    sleep 30  # Cool-down between batches
done
```

### Integration with cron

```bash
# Download 100 faces daily at 2 AM
0 2 * * * cd /home/pi/hybridrag/faces8 && python3 bulk_download_cli.py -n 100 -o daily_faces
```

### Custom Processing Pipeline

```python
import os
from pathlib import Path

# After download, process files
faces_dir = Path("faces_bulk")
for img_file in faces_dir.glob("*.jpg"):
    # Your custom processing
    print(f"Processing {img_file}")
```

## License

Same as parent project.

## Contributing

Contributions welcome! Areas for improvement:
- Additional download sources
- Resume capability
- Async/await implementation
- GPU-accelerated processing
- Database integration

## Support

For issues or questions:
1. Check troubleshooting section
2. Review system requirements
3. Test network connectivity
4. Check logs/error output

---

**Happy downloading! 🚀**
