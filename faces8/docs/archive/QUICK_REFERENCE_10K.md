# 10,000 Faces Download - Quick Reference Card

## 🚀 Fastest Way to Download 10K Faces on Ubuntu

### One-Line Commands

```bash
# Method 1: Automated script (EASIEST)
./download_10k_faces.sh

# Method 2: Direct Python command
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k

# Method 3: Background process
nohup python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k > download.log 2>&1 &
```

---

## 📊 Quick Specs

| Item | Value |
|------|-------|
| **Faces** | 10,000 |
| **Size** | ~5.5 GB |
| **Time** | 45-75 min |
| **Threads** | 16 (recommended) |
| **Speed** | ~2.3-3.5 faces/sec |

---

## 🎯 Command Options

```bash
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k [OPTIONS]

Options:
  -n 10000          Number of faces
  -t 16             Worker threads (adjust for your CPU)
  -o faces_10k      Output directory
  -m                Add metadata (slower)
  -s SOURCE         Source (thispersondoesnotexist/100k-faces)
  --timeout 60      Request timeout
```

---

## 💻 Thread Recommendations

| CPU Cores | Threads | Speed |
|-----------|---------|-------|
| 4 cores | 8 | Slow |
| 8 cores | 12-16 | Good |
| 16 cores | 16-24 | Fast |
| 32+ cores | 24-32 | Fastest |

**Find your cores:** `nproc`

---

## 🔧 Installation (Ubuntu)

```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip -y
pip3 install requests rich psutil pillow opencv-python numpy

# Verify
python3 --version
pip3 list | grep rich
```

---

## 📈 Monitor Progress

```bash
# Count files (live update)
watch -n 5 'ls faces_10k/*.jpg 2>/dev/null | wc -l'

# Check size
watch -n 10 'du -sh faces_10k'

# System resources
htop
```

---

## 🖥️ Background Execution

### Using nohup
```bash
nohup python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k > download.log 2>&1 &
tail -f download.log
```

### Using screen
```bash
screen -S download
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k
# Ctrl+A then D to detach
screen -r download  # reattach
```

### Using tmux
```bash
tmux new -s download
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k
# Ctrl+B then D to detach
tmux attach -t download  # reattach
```

---

## 📝 Post-Download

```bash
# Count files
ls faces_10k/*.jpg | wc -l

# Check size
du -sh faces_10k/

# Generate metadata
python3 generate_missing_metadata.py -d faces_10k -y

# Analyze data
python3 analyze_metadata.py faces_10k
```

---

## ⚡ Performance Tweaks

```bash
# Maximum speed (16+ threads)
python3 bulk_download_cli.py -n 10000 -t 24 -o faces_10k

# Balanced (12 threads)
python3 bulk_download_cli.py -n 10000 -t 12 -o faces_10k

# Conservative (8 threads)
python3 bulk_download_cli.py -n 10000 -t 8 -o faces_10k
```

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Too slow | Increase threads: `-t 24` |
| Out of memory | Decrease threads: `-t 8` |
| Network timeout | Increase timeout: `--timeout 60` |
| Disk full | Check: `df -h .` |

---

## 📋 Checklist

- [ ] Check disk space: `df -h .` (need ~6 GB)
- [ ] Check RAM: `free -h` (need ~4 GB free)
- [ ] Install packages: `pip3 install requests rich psutil`
- [ ] Navigate to directory: `cd /home/pi/hybridrag/faces8`
- [ ] Run download: `./download_10k_faces.sh` OR `python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k`
- [ ] Monitor progress: `watch -n 5 'ls faces_10k/*.jpg | wc -l'`
- [ ] Verify completion: `ls faces_10k/*.jpg | wc -l`
- [ ] (Optional) Generate metadata: `python3 generate_missing_metadata.py -d faces_10k -y`

---

## 🎬 Complete Example

```bash
# 1. Check system
nproc              # CPU cores
free -h            # Memory
df -h .            # Disk space

# 2. Start download
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k

# 3. Wait ~60 minutes

# 4. Verify
ls faces_10k/*.jpg | wc -l    # Should be ~10,000+
du -sh faces_10k/              # Should be ~5.5 GB

# 5. Generate metadata (optional)
python3 generate_missing_metadata.py -d faces_10k -y

# 6. Analyze
python3 analyze_metadata.py faces_10k
```

---

## 🔥 TL;DR - Just Run This

```bash
./download_10k_faces.sh
```

**Or:**

```bash
python3 bulk_download_cli.py -n 10000 -t 16 -o faces_10k
```

**Expected:** ~60 minutes, ~5.5 GB, ~10,000 faces

---

**Need help?** See full guide: `UBUNTU_10K_DOWNLOAD_GUIDE.md`
