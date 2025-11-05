# Embedding CLI - Quick Start Guide

## 🚀 Quick Start (30 seconds)

### 1. View Current Status
```bash
python3 embedding_manager_cli.py --stats-only
```

### 2. Embed All Pending Images (Interactive)
```bash
python3 embedding_manager_cli.py
```
Press `y` when prompted to start embedding.

### 3. Embed All Pending Images (Automatic)
```bash
python3 embedding_manager_cli.py --auto-embed
```

---

## 📋 Common Commands

### Statistics Only
```bash
python3 embedding_manager_cli.py --stats-only
```
Shows: Database stats, file counts, embedding progress

### Interactive Embedding
```bash
python3 embedding_manager_cli.py
```
Shows stats and asks if you want to embed

### Auto Embedding (No Prompt)
```bash
python3 embedding_manager_cli.py --auto-embed
```
Automatically embeds all pending images

### Use Specific Model
```bash
# FaceNet (best balance of speed and accuracy)
python3 embedding_manager_cli.py --model facenet --auto-embed

# Statistical (fastest)
python3 embedding_manager_cli.py --model statistical --auto-embed

# ArcFace (most accurate)
python3 embedding_manager_cli.py --model arcface --auto-embed
```

### Custom Directory
```bash
python3 embedding_manager_cli.py --faces-dir /path/to/faces --auto-embed
```

---

## 📊 What You'll See

### Dashboard Output
```
================================================================================
📊 EMBEDDING MANAGEMENT DASHBOARD
================================================================================

🗄️  DATABASE STATISTICS
Total Embedded Vectors: 29,135
Embedding Models Used:
  • facenet: 29,135 vectors (100.0%)

📁 FILE SYSTEM STATISTICS
Total Image Files: 58,660
Total JSON Files: 58,660
✅ Matched Pairs (Image + JSON): 58,660

🎯 EMBEDDING STATUS
Already Embedded: 58,063
Pending Embedding: 597
Progress: [█████████████████████████████████████████████████░] 99.0%
```

### Embedding Progress
```
[████████████████████████████████████████] 100.0% | 597/597 |
Success: 597 | Errors: 0 | ETA: 0s

✅ Successfully Embedded: 597
⏱️  Total Time: 5m 23s
⚡ Average Speed: 0.54 seconds/image
```

---

## 🔧 Configuration

Edit `.env` file:
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

EMBEDDING_MODEL=statistical
FACES_DIR=./faces
```

---

## ❓ FAQ

**Q: How do I check if my database is connected?**
```bash
PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c "SELECT COUNT(*) FROM faces;"
```

**Q: What embedding model should I use?**
- `statistical` - Fast, good for testing (default)
- `facenet` - Good balance of speed and accuracy (recommended)
- `arcface` - Best accuracy, slower

**Q: How long does embedding take?**
- Statistical: ~0.1-0.5 seconds per image
- FaceNet: ~0.5-1.5 seconds per image
- ArcFace: ~1-3 seconds per image

**Q: Can I stop and resume?**
Yes! The CLI tracks what's already embedded. Press Ctrl+C to stop, then run again to resume.

**Q: What if I have errors?**
The CLI shows error summary at the end. Most common issues:
- Invalid image files
- Missing JSON metadata
- Corrupted files

---

## 🎯 Pro Tips

1. **Use `--stats-only` first** to see what needs embedding
2. **Test with small batches** using `--stats-only`, then `--auto-embed`
3. **Use `--quiet`** for cron jobs
4. **Monitor logs** to track progress over time

---

## 📞 Need Help?

Run with `--help` for full options:
```bash
python3 embedding_manager_cli.py --help
```

See full documentation:
```bash
cat EMBEDDING_CLI_README.md
```

---

**Created by Claude Code** 🤖
