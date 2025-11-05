# Migration Guide: Single-Model → Multi-Model Schema

## Overview

This guide walks you through safely migrating from the old single-model schema to the new multi-model schema **without losing any data**.

---

## 🔍 Understanding the Changes

### Old Schema (Single-Model):
```sql
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(255) UNIQUE,
    embedding vector(512),           -- Single embedding
    embedding_model VARCHAR(50),     -- Model name as text
    ...
);
```

### New Schema (Multi-Model):
```sql
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(255) UNIQUE,
    embedding_facenet vector(512),      -- Separate columns
    embedding_arcface vector(512),      -- for each model
    embedding_vggface2 vector(512),
    embedding_insightface vector(512),
    embedding_statistical vector(512),
    models_processed TEXT[],            -- Array of models
    ...
);
```

---

## ⚠️ Important Information

### What Gets Migrated:
✅ All face records
✅ All metadata (age, gender, brightness, etc.)
✅ All embeddings (mapped to correct model column)
✅ File paths and timestamps
✅ JSONB metadata

### What Changes:
- Old table renamed to `faces_old` (preserved for safety)
- New table created with multi-model columns
- Embeddings mapped based on `embedding_model` field

### What You Need:
- Existing database with `faces` table
- ~2x disk space temporarily (for old + new table)
- 5-10 minutes downtime

---

## 🚀 Migration Methods

### Method 1: Automated Migration (RECOMMENDED)
Use the migration script - handles everything automatically.

### Method 2: Manual Backup & Fresh Install
Backup data, fresh install, re-embed all faces.

---

## Method 1: Automated Migration

### Step 1: Check Current State
```bash
cd /home/pi/hybridrag/faces10_mm

# Check if faces table exists
sudo -u postgres psql -d vector_db -c "\d faces"

# Count records
sudo -u postgres psql -d vector_db -c "SELECT COUNT(*) FROM faces;"

# Check which models you have
sudo -u postgres psql -d vector_db -c "SELECT embedding_model, COUNT(*) FROM faces GROUP BY embedding_model;"
```

### Step 2: Run Backup Script
```bash
./backup_database.sh
```

This creates:
- Full database backup: `backups/faces_backup_YYYYMMDD_HHMMSS.sql`
- Table-only backup: `backups/faces_table_only_YYYYMMDD_HHMMSS.sql`
- CSV export: `backups/faces_data_YYYYMMDD_HHMMSS.csv`

### Step 3: Run Migration Script
```bash
./migrate_to_multimodel.sh
```

The script will:
1. ✅ Check current database state
2. ✅ Offer to create backup
3. ✅ Ask for confirmation
4. ✅ Rename `faces` → `faces_old`
5. ✅ Create new multi-model schema
6. ✅ Migrate all data
7. ✅ Verify record counts
8. ✅ Show embedding distribution

### Step 4: Verify Migration
```bash
# Check new table
sudo -u postgres psql -d vector_db -c "SELECT COUNT(*) FROM faces;"

# Check statistics
sudo -u postgres psql -d vector_db -c "SELECT * FROM get_database_stats();"

# See which models are in use
sudo -u postgres psql -d vector_db -c "SELECT models_processed, COUNT(*) FROM faces GROUP BY models_processed;"
```

### Step 5: Test Search
```bash
# Test with your existing model
sudo -u postgres psql -d vector_db <<EOF
SELECT face_id, file_path
FROM search_similar_faces(
    (SELECT embedding_facenet FROM faces LIMIT 1),
    'facenet',
    10,
    1.0
)
LIMIT 5;
EOF
```

### Step 6: Clean Up (After Verification)
Once you've verified everything works:
```bash
# Drop old table
sudo -u postgres psql -d vector_db -c "DROP TABLE faces_old;"

# Or keep it for safety
echo "Keep faces_old table for a few days before dropping"
```

---

## Method 2: Manual Backup & Fresh Install

### Step 1: Export Face Metadata
```bash
sudo -u postgres psql -d vector_db -c "\COPY (SELECT face_id, file_path, embedding_model FROM faces) TO 'face_list.csv' WITH CSV HEADER;"
```

### Step 2: Backup Database
```bash
sudo -u postgres pg_dump vector_db > full_backup.sql
```

### Step 3: Fresh Install
```bash
./install.sh
# Answer "yes" to recreate schema
```

### Step 4: Re-embed All Faces
```bash
./run_embedding.sh
# Choose models you want
# This will re-process all images
```

---

## 📊 Migration Examples

### Example 1: 1,000 faces with FaceNet
**Before:**
```sql
faces table:
- 1,000 rows
- embedding_model = 'facenet'
- Single embedding column
```

**After:**
```sql
faces table:
- 1,000 rows
- embedding_facenet populated
- embedding_arcface NULL
- models_processed = ['facenet']
```

### Example 2: Mixed Models
**Before:**
```sql
500 rows with embedding_model = 'facenet'
500 rows with embedding_model = 'statistical'
```

**After:**
```sql
500 rows: embedding_facenet populated, models_processed = ['facenet']
500 rows: embedding_statistical populated, models_processed = ['statistical']
```

---

## 🔧 Troubleshooting

### Issue: Migration script says "Nothing to migrate"
**Cause:** No `faces` table exists or already migrated
**Solution:** Check if you're in the right database

### Issue: Row count mismatch after migration
**Cause:** Some records may have issues
**Solution:** Check migration logs, old table is preserved as `faces_old`

### Issue: Search not working after migration
**Cause:** Model name mismatch
**Solution:** Check which column is populated:
```sql
SELECT
    COUNT(*) FILTER (WHERE embedding_facenet IS NOT NULL) as facenet,
    COUNT(*) FILTER (WHERE embedding_arcface IS NOT NULL) as arcface,
    COUNT(*) FILTER (WHERE embedding_statistical IS NOT NULL) as statistical
FROM faces;
```

### Issue: Out of disk space
**Cause:** Need 2x space for old + new table
**Solution:** Free up space or use external drive

---

## 🎯 Post-Migration: Adding More Models

After migration, you can add more models to existing faces:

```bash
# Run embedding with additional models
./run_embedding.sh
# Choose option 7 (multi)
# Enter: arcface,vggface2
```

This will:
- Keep existing embeddings (e.g., facenet)
- Add new embeddings (arcface, vggface2)
- Update `models_processed` array

---

## 📈 Performance Impact

### Storage:
- Old schema: ~2KB per face (1 embedding)
- New schema: ~2-10KB per face (1-5 embeddings)

### Search Speed:
- Same speed (each model has its own index)
- Can search specific model for fastest results

### Migration Time:
- 1,000 faces: ~1 minute
- 10,000 faces: ~5 minutes
- 100,000 faces: ~30 minutes

---

## 🔄 Rollback Plan

If something goes wrong:

### Rollback Method 1: Restore from Backup
```bash
# Stop application
# Drop new table
sudo -u postgres psql -d vector_db -c "DROP TABLE faces;"

# Restore from backup
sudo -u postgres psql -d vector_db < backups/faces_backup_YYYYMMDD_HHMMSS.sql
```

### Rollback Method 2: Rename Old Table Back
```bash
# Drop new table
sudo -u postgres psql -d vector_db -c "DROP TABLE faces;"

# Rename old table back
sudo -u postgres psql -d vector_db -c "ALTER TABLE faces_old RENAME TO faces;"
```

---

## ✅ Verification Checklist

After migration, verify:

- [ ] Row count matches (old count = new count)
- [ ] Sample search returns expected results
- [ ] All metadata fields preserved
- [ ] File paths are correct
- [ ] Embeddings mapped to correct column
- [ ] Application can search faces
- [ ] `get_database_stats()` shows correct data

---

## 📞 Support

If you encounter issues:

1. Check `backups/` directory for your backup files
2. Review migration script output
3. Check PostgreSQL logs: `sudo tail -f /var/log/postgresql/postgresql-*.log`
4. Verify old table still exists: `sudo -u postgres psql -d vector_db -c "\dt"`

---

## 🎉 Success!

Once migration is complete:

1. ✅ All data preserved
2. ✅ Ready for multi-model embeddings
3. ✅ Can search with specific models
4. ✅ Can compare model results

**Next:** Run `./run_embedding.sh` to add more models!

---

## Quick Command Reference

```bash
# Backup
./backup_database.sh

# Migrate
./migrate_to_multimodel.sh

# Verify
sudo -u postgres psql -d vector_db -c "SELECT * FROM get_database_stats();"

# Add more models
./run_embedding.sh

# Drop old table (after verification)
sudo -u postgres psql -d vector_db -c "DROP TABLE faces_old;"
```
