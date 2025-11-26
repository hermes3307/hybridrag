# 🚀 Startup Scripts Guide

## Available Scripts

### 1. `start.sh` - Simple Launcher (Recommended for Daily Use)
Quick and simple script to launch the GUI application.

**Usage:**
```bash
./start.sh
```

**What it does:**
- ✅ Activates virtual environment automatically
- ✅ Checks if dependencies are installed
- ✅ Verifies database connection
- ✅ Launches the GUI application
- ✅ Shows colored status messages

**Perfect for:** Daily use when everything is already set up.

---

### 2. `start_advanced.sh` - Interactive Control Panel
Full-featured interactive menu with setup and testing options.

**Usage:**
```bash
./start_advanced.sh
```

**Features:**
```
1) Start GUI Application       - Launch the main app
2) Setup Database              - Create and configure database
3) Install Dependencies        - Install all Python packages
4) Test Download Sources       - Check if image sources work
5) Test Downloader             - Test downloading images
6) Check System Status         - View system health
7) View Logs                   - See recent activity
8) Clean Test Data             - Remove test files
9) Exit                        - Close the menu
```

**Perfect for:** Initial setup, troubleshooting, and maintenance.

---

### 3. `setup_database.sh` - Database Setup Only
Standalone script for database initialization.

**Usage:**
```bash
./setup_database.sh
```

**What it does:**
- Creates `image_vector` database
- Enables pgvector extension
- Applies schema from schema.sql
- Interactive confirmation for existing databases

---

## 📋 First-Time Setup (Complete Workflow)

### Option A: Using Advanced Script (Easiest)
```bash
# Run the interactive menu
./start_advanced.sh

# Then follow these steps in order:
# 1. Choose option 3 - Install Dependencies
# 2. Choose option 2 - Setup Database
# 3. Choose option 6 - Check System Status (verify everything is OK)
# 4. Choose option 1 - Start GUI Application
```

### Option B: Using Individual Scripts
```bash
# 1. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Setup database
./setup_database.sh

# 3. Start the application
./start.sh
```

---

## 🎯 Common Tasks

### Daily Use
```bash
# Just run the simple launcher
./start.sh
```

### Check if Everything Works
```bash
./start_advanced.sh
# Choose option 6 - Check System Status
```

### Test Image Downloads
```bash
./start_advanced.sh
# Choose option 4 - Test Download Sources
# Choose option 5 - Test Downloader
```

### Reinstall Dependencies
```bash
./start_advanced.sh
# Choose option 3 - Install Dependencies
```

### Reset Database
```bash
./start_advanced.sh
# Choose option 2 - Setup Database
# Answer "yes" when asked to recreate
```

---

## 🔧 Script Details

### start.sh
**Location:** `/home/pi/hybridrag/image/start.sh`

**Pre-checks:**
- Virtual environment exists
- image.py file exists
- Database is accessible (optional check)
- Core dependencies are installed

**Output:**
- Color-coded status messages
- System information display
- Clean error messages

**Exit codes:**
- 0: Success
- 1: Error (missing venv, missing files, etc.)

---

### start_advanced.sh
**Location:** `/home/pi/hybridrag/image/start_advanced.sh`

**Interactive Menu Options:**

| Option | Description | Requirements |
|--------|-------------|--------------|
| 1 | Start GUI | venv + dependencies |
| 2 | Setup Database | PostgreSQL installed |
| 3 | Install Dependencies | Python 3 + pip |
| 4 | Test Sources | venv + dependencies |
| 5 | Test Downloader | venv + dependencies |
| 6 | Check Status | None |
| 7 | View Logs | None |
| 8 | Clean Test Data | None |
| 9 | Exit | None |

**Features:**
- Color-coded output
- Progress indicators
- Error handling
- Interactive confirmations
- Loop menu (keeps running until you exit)

---

### setup_database.sh
**Location:** `/home/pi/hybridrag/image/setup_database.sh`

**Requirements:**
- PostgreSQL installed and running
- User has postgres access
- schema.sql file present

**Safety Features:**
- Checks if database already exists
- Asks for confirmation before dropping
- Verifies table creation
- Shows created tables

---

## 🐛 Troubleshooting

### Script won't run (Permission denied)
```bash
chmod +x start.sh
chmod +x start_advanced.sh
chmod +x setup_database.sh
```

### Virtual environment not found
```bash
# Create it manually
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Database connection failed
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Start it if needed
sudo systemctl start postgresql

# Try creating database manually
psql -U postgres -c "CREATE DATABASE image_vector;"
```

### Dependencies installation fails
```bash
# Upgrade pip first
source venv/bin/activate
pip install --upgrade pip

# Install dependencies one by one
pip install requests numpy pillow
pip install psycopg2-binary
pip install torch torchvision
pip install transformers ultralytics
```

### GUI doesn't start
```bash
# Check if image.py exists
ls -la image.py

# Check for Python errors
source venv/bin/activate
python3 image.py
```

---

## 💡 Tips

### Running in Background
```bash
# Start GUI in background
./start.sh &

# View the process
ps aux | grep image.py

# Stop it
pkill -f image.py
```

### Auto-start on Boot (systemd)
```bash
# Create a systemd service
sudo nano /etc/systemd/system/image-search.service

# Add:
[Unit]
Description=Image Search System
After=postgresql.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/hybridrag/image
ExecStart=/home/pi/hybridrag/image/start.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable image-search
sudo systemctl start image-search
```

### Create Desktop Shortcut
```bash
# Create .desktop file
cat > ~/Desktop/ImageSearch.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=Image Search
Comment=Multi-Embedding Image Search System
Exec=/home/pi/hybridrag/image/start.sh
Icon=image-viewer
Terminal=true
Categories=Graphics;Utility;
EOF

chmod +x ~/Desktop/ImageSearch.desktop
```

---

## 📊 Status Indicators

The scripts use color-coded indicators:

- ✅ **Green** - Success, everything OK
- ⚠️  **Yellow** - Warning, optional issue
- ❌ **Red** - Error, needs attention
- ℹ️  **Cyan** - Information, FYI
- 🔵 **Blue** - Headers and titles

---

## 🎉 Quick Reference

| Task | Command |
|------|---------|
| Start GUI | `./start.sh` |
| Full menu | `./start_advanced.sh` |
| Setup database | `./setup_database.sh` |
| Install deps | `./start_advanced.sh` → option 3 |
| Test downloads | `./start_advanced.sh` → option 4 |
| Check status | `./start_advanced.sh` → option 6 |
| Clean tests | `./start_advanced.sh` → option 8 |

---

**For daily use:** Just run `./start.sh` and you're good to go! 🚀
