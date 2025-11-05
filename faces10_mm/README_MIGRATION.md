# Quick Migration Reference

## ⚡ Quick Start: Safe Migration

### For Existing Database Users

If you have an existing database with face data and want to upgrade to multi-model support:

```bash
# 1. Backup your data (creates backups/ directory)
./backup_database.sh

# 2. Migrate to multi-model schema
./migrate_to_multimodel.sh

# 3. Verify migration worked
sudo -u postgres psql -d vector_db -c "SELECT * FROM get_database_stats();"

# 4. Add more models to existing faces
./run_embedding.sh
```

---

## 🆕 For Fresh Install

If you DON'T have existing data:

```bash
# Just run install
./install.sh

# Then embed your faces
./run_embedding.sh
```

---

## 📋 What Each Script Does

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `backup_database.sh` | Backup existing data | Before any changes |
| `migrate_to_multimodel.sh` | Migrate to new schema | Upgrade existing DB |
| `install.sh` | Fresh database setup | New installation |
| `run_embedding.sh` | Generate embeddings | After install/migration |
| `start_system.sh` | Start application | Run the system |

---

## 🛡️ Safety Features

### Backup Script:
- Creates 3 types of backups
- Full database dump (.sql)
- Table-only backup (.sql)
- CSV export (for inspection)

### Migration Script:
- Preserves old table as `faces_old`
- Verifies row counts
- Can rollback if issues occur
- Shows distribution of models

---

## ⚠️ Important Notes

### DO NOT run `install.sh` if you have existing data!
It will **DELETE** your database. Use migration instead.

### Safe Upgrade Path:
```
Existing DB → backup_database.sh → migrate_to_multimodel.sh → Verify → Done ✅
```

### Fresh Install Path:
```
No DB → install.sh → run_embedding.sh → start_system.sh ✅
```

---

## 🔍 Quick Checks

### Do I have existing data?
```bash
sudo -u postgres psql -d vector_db -c "SELECT COUNT(*) FROM faces;"
```

### What model am I using?
```bash
sudo -u postgres psql -d vector_db -c "SELECT DISTINCT embedding_model FROM faces;"
```

### Check migration was successful:
```bash
sudo -u postgres psql -d vector_db -c "\dt"
# Should show: faces, faces_old
```

---

## 📞 Help

- **Full guide:** `MIGRATION_GUIDE.md`
- **Multi-model info:** `MULTIMODEL_SETUP_SUMMARY.md`
- **Quick start:** `QUICK_START.md`

---

## 🎯 TL;DR

**Have data?** → `backup_database.sh` → `migrate_to_multimodel.sh`

**No data?** → `install.sh`

**Both cases:** → `run_embedding.sh` → `start_system.sh`
