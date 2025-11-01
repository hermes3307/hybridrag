# File Cleanup and Organization Plan

## Current Status
- **Total Scripts**: 42 files (.sh and .py)
- **Issue**: Too many similar/duplicate files, unclear naming
- **Goal**: Clean, organized structure with clear naming

---

## 📋 CLEANUP RECOMMENDATIONS

### ❌ Files to DELETE (Old/Duplicate/Temporary)

```bash
# Old/superseded fix scripts (keep only stable version)
rm fix_demographics.py                    # Superseded by fix_demographics_stable.py
rm fix_demographics_simple.py             # Superseded by fix_demographics_stable.py

# Duplicate monitor files (keep only enhanced version)
rm monitor_demo.py                        # Demo version, not needed
rm test_monitor.py                        # Test file, move to tests/ if needed

# Old embedding script (superseded by embedding_manager_cli.py)
rm embed.sh                               # Old version, use run_embedding.sh instead

# Redundant download scripts (keep essential ones)
rm quick_download.sh                      # Superseded by download_faces.sh
rm download_with_metadata.sh              # Superseded by bulk_download_cli.py

# Test files (move to tests/ directory instead of deleting)
# We'll create a tests/ directory for these
```

---

## 📝 Files to RENAME (Unclear Names)

### Current → New Name (with reason)

```bash
# More descriptive names
mv start.sh              → start_system.sh           # Clearer purpose
mv check_status.sh       → check_system_status.sh   # More specific
mv embed.sh              → (DELETE - old version)
mv run_monitor.sh        → start_monitor.sh         # Consistent naming
```

---

## 📁 PROPOSED DIRECTORY STRUCTURE

```
faces8/
├── 📂 core/                          # Core system files
│   ├── core.py                       # Main face processing
│   ├── pgvector_db.py                # Database management
│   └── faces.py                      # GUI application
│
├── 📂 cli/                           # Command-line tools
│   ├── embedding_manager_cli.py      # Embedding management ⭐
│   ├── search_cli.py                 # Search interface
│   ├── vector_query_cli.py           # Vector queries
│   ├── advanced_search.py            # Advanced search
│   ├── bulk_download_cli.py          # Bulk downloads
│   └── inspect_database.py           # DB inspector
│
├── 📂 scripts/                       # Shell scripts
│   ├── run_embedding.sh              # Main embedding runner ⭐
│   ├── start_system.sh               # System starter
│   ├── start_monitor.sh              # Monitor starter
│   ├── check_system_status.sh        # Status checker ⭐
│   ├── install.sh                    # Installation
│   └── db_manage.sh                  # Database management
│
├── 📂 download/                      # Download scripts
│   ├── download_faces.sh             # General download
│   ├── download_10k_faces.sh         # 10K dataset
│   └── download_large_dataset.sh     # Large datasets
│
├── 📂 maintenance/                   # Cleanup/maintenance
│   ├── remove_duplicates.sh          # Remove duplicate images
│   ├── delete_duplicates_python.py   # Python duplicate remover
│   ├── delete_all_duplicates.sh      # Batch delete
│   ├── remove_apple_files.sh         # macOS cleanup ⭐
│   ├── delete_embeddings_by_model.sh # Delete embeddings
│   ├── fix_demographics_stable.py    # Fix demographics ⭐
│   ├── generate_missing_metadata.py  # Generate metadata
│   └── analyze_metadata.py           # Analyze metadata
│
├── 📂 monitoring/                    # Monitoring tools
│   ├── monitor_enhanced.py           # Enhanced monitor ⭐
│   ├── monitor.py                    # Basic monitor
│   └── search_examples.py            # Search examples
│
├── 📂 migration/                     # Database migration
│   └── migrate_to_pgvector.py        # pgvector migration
│
├── 📂 tests/                         # Test files
│   ├── test_download_speed.py
│   ├── test_metadata_search.py
│   ├── test_opencv.py
│   └── test_pgvector.py
│
├── 📂 docs/                          # Documentation
│   ├── SPEEDUP_GUIDE.md
│   ├── README.md
│   └── ... (other .md files)
│
└── 📂 deprecated/                    # Old files (archive)
    └── (moved old files here)
```

---

## 🚀 SIMPLIFIED STRUCTURE (Recommended)

If you want to keep it simple without subdirectories:

### Keep These Essential Files:

#### **Core Python Files** (10 files)
```
core.py                      # Main system
pgvector_db.py               # Database
faces.py                     # GUI
embedding_manager_cli.py     # Embeddings ⭐ MAIN TOOL
search_cli.py                # Search
vector_query_cli.py          # Vector queries
advanced_search.py           # Advanced search
monitor_enhanced.py          # Monitoring
inspect_database.py          # DB inspector
migrate_to_pgvector.py       # Migration
```

#### **Essential Shell Scripts** (8 files)
```
run_embedding.sh             # ⭐ MAIN EMBEDDING SCRIPT
start_system.sh              # Start system
check_system_status.sh       # Check status
install.sh                   # Installation
db_manage.sh                 # Database management
download_faces.sh            # Download faces
download_10k_faces.sh        # Download 10K dataset
remove_apple_files.sh        # macOS cleanup
```

#### **Maintenance Tools** (6 files)
```
remove_duplicates.sh
delete_duplicates_python.py
delete_embeddings_by_model.sh
fix_demographics_stable.py
generate_missing_metadata.py
analyze_metadata.py
```

#### **Move to archive/** (Everything else)

---

## 📊 CLEANUP COMMANDS

### Option 1: Safe Cleanup (Move to archive)

```bash
# Create archive directory
mkdir -p archive

# Move old/duplicate files
mv fix_demographics.py archive/
mv fix_demographics_simple.py archive/
mv monitor_demo.py archive/
mv test_monitor.py archive/
mv embed.sh archive/
mv quick_download.sh archive/
mv download_with_metadata.sh archive/

# Move test files to tests directory
mkdir -p tests
mv test_*.py tests/

echo "✅ Files archived safely!"
```

### Option 2: Aggressive Cleanup (Delete)

```bash
# ⚠️ WARNING: This permanently deletes files!
# Make a backup first: tar -czf backup.tar.gz *.py *.sh

# Delete old versions
rm -f fix_demographics.py fix_demographics_simple.py
rm -f monitor_demo.py test_monitor.py
rm -f embed.sh quick_download.sh download_with_metadata.sh

# Move test files
mkdir -p tests
mv test_*.py tests/

echo "✅ Cleanup complete!"
```

### Option 3: Rename Important Files

```bash
# Rename for clarity
mv start.sh start_system.sh
mv check_status.sh check_system_status.sh
mv run_monitor.sh start_monitor.sh

echo "✅ Files renamed!"
```

---

## 🎯 RECOMMENDED ACTION PLAN

### Step 1: Create Backup
```bash
cd /home/pi/hybridrag/faces8
tar -czf ../faces8_backup_$(date +%Y%m%d).tar.gz *.py *.sh *.md
echo "✅ Backup created!"
```

### Step 2: Safe Cleanup
```bash
# Create directories
mkdir -p archive tests

# Move old/duplicate files to archive
mv fix_demographics.py fix_demographics_simple.py archive/
mv monitor_demo.py archive/
mv embed.sh quick_download.sh download_with_metadata.sh archive/

# Move test files
mv test_*.py tests/

# Rename key files
mv start.sh start_system.sh
mv check_status.sh check_system_status.sh

echo "✅ Cleanup complete!"
```

### Step 3: Verify
```bash
# List main files
ls -1 *.py *.sh

# Should see clean list of ~24 files
```

---

## 📱 QUICK REFERENCE AFTER CLEANUP

### Most Used Commands:

```bash
# Embedding (MAIN TOOL)
./run_embedding.sh                    # Interactive embedding with model/worker selection

# System Management
./start_system.sh                     # Start the system
./check_system_status.sh              # Check status
./install.sh                          # Install dependencies

# Download
./download_faces.sh                   # Download faces
./download_10k_faces.sh               # Download 10K dataset

# Maintenance
./remove_apple_files.sh               # Remove macOS junk files
./remove_duplicates.sh                # Remove duplicate images
python3 fix_demographics_stable.py    # Fix demographics

# Database
./db_manage.sh                        # Database management
python3 inspect_database.py           # Inspect database

# Search
python3 search_cli.py                 # Search interface
python3 vector_query_cli.py           # Vector queries

# Monitoring
python3 monitor_enhanced.py           # Enhanced monitoring
```

---

## 🎨 File Naming Conventions (Going Forward)

- **Scripts that start something**: `start_*.sh`
- **Scripts that run a process**: `run_*.sh`
- **Scripts for checking/status**: `check_*.sh`
- **Python CLI tools**: `*_cli.py`
- **Python utilities**: `*_manager.py`, `*_helper.py`
- **Test files**: `test_*.py` (in tests/ directory)

---

## ✅ FINAL RESULT

### Before: 42 files (confusing)
### After: ~24 files (organized)

**Reduction**: 43% fewer files
**Benefit**: Easier to find what you need!

---

## 🚨 IMPORTANT NOTES

1. **Always backup before cleanup**: `tar -czf backup.tar.gz *.py *.sh`
2. **Test after cleanup**: Run `./run_embedding.sh` to verify
3. **Keep archive/ directory**: Don't delete immediately
4. **Update documentation**: If you share this project
5. **Git commit**: If using git, commit after cleanup

---

## 💡 RECOMMENDATIONS

### Immediate Actions (Do Now):
1. ✅ Create backup
2. ✅ Move old fix_demographics*.py to archive
3. ✅ Move test files to tests/
4. ✅ Rename start.sh → start_system.sh
5. ✅ Delete embed.sh (superseded by run_embedding.sh)

### Nice to Have (Optional):
- Create subdirectories (core/, cli/, scripts/, etc.)
- Write README.md with file descriptions
- Create a main launcher script

### Keep Monitoring:
- After 1 month, if archive/ files unused → delete
- Document which scripts you actually use
- Remove any you never touch
