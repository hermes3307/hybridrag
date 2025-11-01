# Remove Duplicates Script - User Guide

## 🎯 Purpose

Remove duplicate image files from your faces directory, keeping only ONE copy of each unique image (identified by hash).

---

## 📋 Quick Start

### Step 1: Check Statistics
```bash
./remove_duplicates.sh --stats
```

Shows how many duplicates you have without taking any action.

### Step 2: Dry-Run (See What Would Be Deleted)
```bash
./remove_duplicates.sh
```

Shows exactly which files would be deleted (NO FILES ARE DELETED).

### Step 3: Actually Delete Duplicates
```bash
./remove_duplicates.sh --confirm
```

Deletes duplicate files after confirmation.

---

## 🛡️ Safety Features

✅ **Default is Dry-Run** - No files deleted unless you use `--confirm`
✅ **Keeps Oldest File** - For each duplicate set, keeps the oldest copy
✅ **Backup List Created** - Saves list of deleted files before deletion
✅ **Confirmation Required** - Must type "DELETE" to confirm
✅ **Deletes JSON Too** - Removes corresponding JSON metadata files

---

## 📊 How It Works

### 1. Identifies Duplicates by Hash
```
face_20251025_102649_294_d1548706.jpg  ← Hash: d1548706
face_20251026_153012_789_d1548706.jpg  ← Same hash = duplicate
face_20251027_091234_567_d1548706.jpg  ← Same hash = duplicate
```

### 2. Keeps Oldest File
```
✅ KEEP:   face_20251025_102649_294_d1548706.jpg  (oldest)
❌ DELETE: face_20251026_153012_789_d1548706.jpg
❌ DELETE: face_20251027_091234_567_d1548706.jpg
```

### 3. Deletes Both Image and JSON
```
Deleted: face_20251026_153012_789_d1548706.jpg
Deleted: face_20251026_153012_789_d1548706.json
```

---

## 💻 Command Options

| Command | Description | Safe? |
|---------|-------------|-------|
| `./remove_duplicates.sh` | Dry-run (shows what would be deleted) | ✅ Yes |
| `./remove_duplicates.sh --stats` | Show statistics only | ✅ Yes |
| `./remove_duplicates.sh --confirm` | Actually delete files | ⚠️  Deletes files |
| `./remove_duplicates.sh --help` | Show help message | ✅ Yes |

---

## 📈 Example Output

### Statistics Mode
```
$ ./remove_duplicates.sh --stats

================================================================================
📊 STATISTICS
================================================================================

Total Image Files:        58,660
Unique Images (by hash):  29,730
Duplicate Files:          28,930
Hashes with duplicates:   14,465

To see which files would be deleted:
  ./remove_duplicates.sh

To actually delete duplicate files:
  ./remove_duplicates.sh --confirm
```

### Dry-Run Mode
```
$ ./remove_duplicates.sh

================================================================================
🔍 DUPLICATE FILES TO REMOVE
================================================================================

Hash: d1548706
  ✅ KEEP: face_20251025_102649_294_d1548706.jpg (oldest)
  ❌ DELETE: face_20251026_153012_789_d1548706.jpg
  ❌ DELETE: face_20251027_091234_567_d1548706.jpg

Hash: b487536e
  ✅ KEEP: face_20251025_103015_412_b487536e.jpg (oldest)
  ❌ DELETE: face_20251026_154521_987_b487536e.jpg

================================================================================
📋 SUMMARY
================================================================================

Files to KEEP:    29,730
Files to DELETE:  28,930
Space to save:    ~57,860 files (images + JSON)

================================================================================
⚠️  DRY-RUN MODE - NO FILES DELETED
================================================================================

This was a dry-run. No files were actually deleted.

To actually delete the duplicate files, run:
  ./remove_duplicates.sh --confirm
```

### Deletion Mode
```
$ ./remove_duplicates.sh --confirm

================================================================================
⚠️  DELETION MODE - FILES WILL BE DELETED
================================================================================

Creating backup list...
Backup list saved to: ./duplicate_backups/deleted_files_20251101_234530.txt

About to delete 28,930 duplicate image files and their JSON files.
This action CANNOT be undone!

Type 'DELETE' to confirm: DELETE

🗑️  Deleting files...

✓ Deleted: face_20251026_153012_789_d1548706.jpg
✓ Deleted: face_20251026_153012_789_d1548706.json
✓ Deleted: face_20251027_091234_567_d1548706.jpg
✓ Deleted: face_20251027_091234_567_d1548706.json
...

================================================================================
✅ DELETION COMPLETE
================================================================================

Successfully deleted: 57,860 files
Errors:              0
Backup list:         ./duplicate_backups/deleted_files_20251101_234530.txt

Duplicate removal complete!

Run './check_status.sh' to see updated statistics.
```

---

## 🔄 After Deletion

### Check Updated Statistics
```bash
./check_status.sh
```

Expected results:
- Total Image Files: 29,730 (reduced from 58,660)
- Unique Images: 29,730 (same)
- Duplicate Files: 0 (all removed)

---

## 💾 Backup & Recovery

### Backup List Location
```
./duplicate_backups/deleted_files_YYYYMMDD_HHMMSS.txt
```

This file contains the full paths of all deleted files.

### View Deleted Files
```bash
cat ./duplicate_backups/deleted_files_*.txt
```

### Note About Recovery
⚠️ Files are **permanently deleted**. There is NO automatic recovery.
The backup list only shows what was deleted, it doesn't restore files.

---

## ⚠️ Important Notes

1. **No Undo** - Once deleted, files cannot be recovered (unless you have system backups)
2. **Database Safe** - Deleting duplicate files does NOT affect embedded vectors
3. **Keep One Copy** - Always keeps at least one copy of each unique image
4. **Oldest Preserved** - Keeps the oldest file by modification time
5. **Test First** - Always run without `--confirm` first to see what will happen

---

## 🎓 Decision Guide

### Should I Delete Duplicates?

**Reasons to DELETE:**
- ✅ Save disk space (~57,860 files)
- ✅ Cleaner file system
- ✅ Faster directory operations
- ✅ Database already handles deduplication

**Reasons to KEEP:**
- ⚠️ Want multiple copies for backup
- ⚠️ Different timestamps might be important
- ⚠️ Unsure about which files to keep

**Recommendation:**
It's **safe to delete** duplicates since the database already stores unique embeddings. The filesystem duplicates serve no functional purpose.

---

## 📝 Step-by-Step Workflow

### Conservative Approach (Recommended)
```bash
# 1. Check statistics
./remove_duplicates.sh --stats

# 2. See exactly what would be deleted
./remove_duplicates.sh > duplicate_review.txt

# 3. Review the file
less duplicate_review.txt

# 4. If satisfied, delete
./remove_duplicates.sh --confirm

# 5. Verify results
./check_status.sh
```

### Quick Approach
```bash
# Delete directly (still requires confirmation)
./remove_duplicates.sh --confirm
```

---

## 🔍 Verification

### Before Deletion
```bash
# Count files
find /home/pi/faces -name "*.jpg" | wc -l
# Output: 58,660

# Check disk usage
du -sh /home/pi/faces
# Output: ~15 GB
```

### After Deletion
```bash
# Count files
find /home/pi/faces -name "*.jpg" | wc -l
# Output: 29,730

# Check disk usage
du -sh /home/pi/faces
# Output: ~7.5 GB

# Space saved: ~7.5 GB
```

---

## 🚨 Troubleshooting

### "No duplicate images found"
✅ Good! You have no duplicates to remove.

### "Permission denied"
```bash
# Make script executable
chmod +x remove_duplicates.sh

# Or run with bash
bash remove_duplicates.sh
```

### Script hangs on large datasets
⏳ Be patient. Analyzing 60,000+ files takes time (1-2 minutes).

### Want to see progress
```bash
# Run with verbose mode (if implemented)
# Or watch file count decrease
watch -n 5 'find /home/pi/faces -name "*.jpg" | wc -l'
```

---

## ✅ Best Practices

1. **Test First** - Always run without `--confirm` first
2. **Backup** - Have system backups before mass deletion
3. **Off-Hours** - Run during low-usage times for large datasets
4. **Verify** - Check statistics before and after
5. **Keep Logs** - Save output for records

---

## 📞 Support

### Get Help
```bash
./remove_duplicates.sh --help
```

### Check Results
```bash
./check_status.sh
```

### View Documentation
```bash
cat REMOVE_DUPLICATES_GUIDE.md
```

---

**Script Location:** `/home/pi/hybridrag/faces8/remove_duplicates.sh`

**Backup Location:** `/home/pi/hybridrag/faces8/duplicate_backups/`

---

**Created by Claude Code** 🤖
**Handle with care!** ⚠️
