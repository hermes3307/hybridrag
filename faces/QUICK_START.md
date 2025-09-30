# 🚀 Quick Start Guide

## Simplest Way to Start

### Option 1: Interactive Menu (Recommended)
```bash
python 0_launcher.py
```
Then select 1-10 from the menu!

### Option 2: Shell Scripts
**Windows:**
```cmd
run.bat          # Show menu
run.bat 7        # Run component 7 directly
```

**Linux/Mac:**
```bash
./run.sh         # Show menu
./run.sh 7       # Run component 7 directly
```

### Option 3: Direct Commands
```bash
python 1_setup_database.py
python 4_download_faces_gui.py
python 6_embed_faces_gui.py
python 7_search_faces_gui.py
```

---

## 3-Minute Setup

```bash
# Step 1: Setup (1 minute)
python 1_setup_database.py

# Step 2: Get faces (1 minute)
python 4_download_faces_gui.py
# Click "Download" -> Enter 100 -> Wait

# Step 3: Embed (1 minute)
python 6_embed_faces_gui.py
# Check "Clear Existing" -> Click "Start Embedding"

# Done! Now search:
python 7_search_faces_gui.py
```

---

## Component Numbers Quick Reference

| # | Component | When to Use |
|---|-----------|-------------|
| **0** | Launcher Menu | When you want a menu |
| **1** | Setup Database | First time only |
| **2** | Database Info | Check what you have |
| **3** | Download (CLI) | Batch downloads |
| **4** | Download (GUI) | Visual downloads |
| **5** | Embed (CLI) | Batch embedding |
| **6** | Embed (GUI) | Visual embedding ⭐ |
| **7** | Search (GUI) | Find faces! ⭐⭐⭐ |
| **8** | Validate | Check quality |
| **9** | Test Features | Verify extraction |
| **10** | Complete Demo | Learn the system |

⭐ = Recommended for beginners
⭐⭐⭐ = Most used

---

## Most Common Commands

### Check what you have:
```bash
python 2_database_info.py
```

### Download 100 more faces:
```bash
python 3_download_faces.py --count 100
```

### Embed new faces:
```bash
python 5_embed_faces.py
```

### Search faces:
```bash
python 7_search_faces_gui.py
```

---

## Shell Script Examples

**Windows:**
```cmd
run.bat 1        # Setup
run.bat 4        # Download GUI
run.bat 6        # Embed GUI
run.bat 7        # Search GUI
run.bat 2        # Check stats
```

**Linux/Mac:**
```bash
./run.sh 1       # Setup
./run.sh 4       # Download GUI
./run.sh 6       # Embed GUI
./run.sh 7       # Search GUI
./run.sh 2       # Check stats
```

---

## Troubleshooting One-Liners

```bash
# Database not found?
python 1_setup_database.py

# No faces?
python 4_download_faces_gui.py

# No embeddings?
python 6_embed_faces_gui.py

# Features show "unknown"?
python 5_embed_faces.py --clear

# Just show me everything:
python 0_launcher.py
```

---

## Daily Usage Pattern

**First time (once):**
```bash
run.bat 1        # or python 1_setup_database.py
```

**Regular workflow:**
```bash
run.bat 4        # Download faces
run.bat 6        # Embed faces
run.bat 7        # Search faces
```

**Check status:**
```bash
run.bat 2        # Database info
```

---

## Single Command for Everything

Want to see the whole pipeline? Just run:
```bash
python 10_complete_demo.py
```

Or:
```bash
run.bat 10       # Windows
./run.sh 10      # Linux/Mac
```

---

## Help Commands

```bash
# Component-specific help
python 3_download_faces.py --help
python 5_embed_faces.py --help

# System help
python 0_launcher.py
python 10_complete_demo.py
```

---

## File Quick Reference

```
Core Components:
├── 0_launcher.py              ← Start here!
├── 1_setup_database.py        ← Run once
├── 4_download_faces_gui.py    ← Get faces
├── 6_embed_faces_gui.py       ← Process faces
└── 7_search_faces_gui.py      ← Search faces ⭐

Shell Launchers:
├── run.bat                    ← Windows
└── run.sh                     ← Linux/Mac

Documentation:
├── QUICK_START.md             ← You are here!
├── README.md                  ← Full guide
└── UNIFIED_SEARCH_GUIDE.md    ← Search details
```

---

## Remember

**Most used commands:**
1. `python 0_launcher.py` - Menu for everything
2. `python 7_search_faces_gui.py` - Search interface
3. `python 2_database_info.py` - Check status

**Or with shell scripts:**
1. `run.bat` - Menu (Windows)
2. `./run.sh` - Menu (Linux/Mac)

That's it! 🎉