# Shell Scripts Guide - Embedding Management

## 🎯 Quick Reference

Three simple shell scripts for easy embedding management:

---

## 📜 Shell Scripts

### 1. **embed.sh** - Main Interactive Script

The primary script with multiple modes.

#### Usage:
```bash
# Interactive mode (asks before embedding)
./embed.sh

# Show statistics only
./embed.sh --stats

# Auto-embed all pending images
./embed.sh --auto

# Use specific model
./embed.sh --facenet      # Use FaceNet model
./embed.sh --arcface      # Use ArcFace model
./embed.sh --statistical  # Use Statistical model

# Show help
./embed.sh --help
```

---

### 2. **check_status.sh** - Quick Status Check

Shows current embedding statistics without making any changes.

#### Usage:
```bash
./check_status.sh
```

#### Output:
- Total embedded vectors
- Embedding models used
- Recent activity
- File counts
- Embedding progress

---

### 3. **run_embedding.sh** - Auto-Run Script

Automatically embeds all pending images without prompting.

#### Usage:
```bash
./run_embedding.sh
```

Perfect for:
- Cron jobs
- Automation scripts
- Quick batch processing

---

## 🚀 Common Workflows

### Quick Status Check
```bash
./check_status.sh
```

### Interactive Embedding
```bash
./embed.sh
```
Then type `y` when asked.

### Automatic Embedding
```bash
./run_embedding.sh
```

### Check Before Embedding
```bash
# First check status
./check_status.sh

# Then auto-embed if needed
./run_embedding.sh
```

---

## 🔧 Advanced Usage

### One-Liner Status Check
```bash
./embed.sh --stats
```

### One-Liner Auto-Embed
```bash
./embed.sh --auto
```

### Specific Model Embedding
```bash
# Fast embedding
./embed.sh --statistical

# Balanced (recommended)
./embed.sh --facenet

# Most accurate
./embed.sh --arcface
```

---

## 📊 Script Comparison

| Script | Mode | Prompts User | Use Case |
|--------|------|--------------|----------|
| `embed.sh` | Interactive | ✅ Yes | General use |
| `embed.sh --stats` | Read-only | ❌ No | Quick check |
| `embed.sh --auto` | Auto-run | ❌ No | Automation |
| `check_status.sh` | Read-only | ❌ No | Status only |
| `run_embedding.sh` | Auto-run | ❌ No | Batch processing |

---

## 🤖 Automation Examples

### Cron Job (Run Daily at 2 AM)
```bash
# Edit crontab
crontab -e

# Add this line:
0 2 * * * cd /home/pi/hybridrag/faces8 && ./run_embedding.sh >> /var/log/embeddings.log 2>&1
```

### Systemd Timer
Create `/etc/systemd/system/embedding.service`:
```ini
[Unit]
Description=Embedding Management Service

[Service]
Type=oneshot
WorkingDirectory=/home/pi/hybridrag/faces8
ExecStart=/home/pi/hybridrag/faces8/run_embedding.sh
User=pi

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/embedding.timer`:
```ini
[Unit]
Description=Run embedding daily

[Timer]
OnCalendar=daily
OnCalendar=02:00

[Install]
WantedBy=timers.target
```

Enable:
```bash
sudo systemctl enable embedding.timer
sudo systemctl start embedding.timer
```

### Shell Script Integration
```bash
#!/bin/bash

cd /home/pi/hybridrag/faces8

# Check status first
echo "Checking status..."
./check_status.sh

# Run embedding if needed
echo "Starting embedding..."
./run_embedding.sh

if [ $? -eq 0 ]; then
    echo "✅ Success!"
else
    echo "❌ Failed!"
    exit 1
fi
```

---

## 🛠️ Troubleshooting

### Script Not Executable
```bash
chmod +x embed.sh
chmod +x check_status.sh
chmod +x run_embedding.sh
```

### Python Not Found
```bash
# Check Python installation
which python3

# If not found, install:
sudo apt-get install python3
```

### Database Connection Error
```bash
# Test database connection
PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c "SELECT 1;"
```

### Permission Issues
```bash
# Fix file permissions
chmod +x *.sh
chmod 644 *.py
```

---

## 📁 File Locations

All scripts are located in:
```
/home/pi/hybridrag/faces8/
```

Files:
- `embed.sh` - Main interactive script
- `check_status.sh` - Status checker
- `run_embedding.sh` - Auto-run script
- `embedding_manager_cli.py` - Python CLI (called by shell scripts)

---

## 💡 Pro Tips

1. **Always check status first**:
   ```bash
   ./check_status.sh
   ```

2. **Use interactive mode when testing**:
   ```bash
   ./embed.sh
   ```

3. **Use auto mode for production**:
   ```bash
   ./run_embedding.sh
   ```

4. **Check logs for errors**:
   ```bash
   ./run_embedding.sh 2>&1 | tee embedding.log
   ```

5. **Test database connection**:
   ```bash
   PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c "SELECT COUNT(*) FROM faces;"
   ```

---

## 🎯 Quick Start (30 Seconds)

```bash
# 1. Check current status
./check_status.sh

# 2. Run embedding
./run_embedding.sh

# 3. Verify completion
./check_status.sh
```

---

## 📞 Getting Help

### Shell Script Help
```bash
./embed.sh --help
```

### Full CLI Help
```bash
python3 embedding_manager_cli.py --help
```

### Documentation
```bash
cat EMBEDDING_CLI_README.md
cat EMBEDDING_CLI_QUICKSTART.md
```

---

**Created with ❤️ by Claude Code**
