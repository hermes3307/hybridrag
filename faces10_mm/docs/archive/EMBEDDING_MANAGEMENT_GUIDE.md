# Embedding Management Guide

## 🎯 Managing Embeddings by Model

You have multiple tools to manage embeddings in your database.

---

## 📊 **View All Embedding Models**

### Python Script (Recommended):
```bash
python3 manage_embeddings.py list
```

### Shell Script:
```bash
./delete_embeddings_by_model.sh
```
(Shows available models when run without arguments)

### Direct SQL:
```bash
PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c \
  "SELECT embedding_model, COUNT(*) FROM faces GROUP BY embedding_model;"
```

---

## 🗑️ **Delete Embeddings by Model**

### Method 1: Python Script (Recommended)

#### Dry-run (see what would be deleted):
```bash
python3 manage_embeddings.py delete statistical
python3 manage_embeddings.py delete facenet
```

#### Actually delete:
```bash
echo "DELETE" | python3 manage_embeddings.py delete statistical --confirm
echo "DELETE" | python3 manage_embeddings.py delete facenet --confirm
```

---

### Method 2: Shell Script

#### Dry-run:
```bash
./delete_embeddings_by_model.sh statistical
./delete_embeddings_by_model.sh facenet
```

#### Actually delete:
```bash
./delete_embeddings_by_model.sh statistical --confirm
./delete_embeddings_by_model.sh facenet --confirm
```

---

### Method 3: Direct SQL

```bash
# Check count first
PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c \
  "SELECT COUNT(*) FROM faces WHERE embedding_model = 'statistical';"

# Delete
PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c \
  "DELETE FROM faces WHERE embedding_model = 'statistical';"
```

---

## 📋 **Common Use Cases**

### Delete All Statistical Embeddings
```bash
# Safe way (with confirmation)
./delete_embeddings_by_model.sh statistical --confirm

# Or with Python
echo "DELETE" | python3 manage_embeddings.py delete statistical --confirm
```

### Delete All FaceNet Embeddings
```bash
./delete_embeddings_by_model.sh facenet --confirm
```

### Keep Only One Model
```bash
# Example: Keep facenet, delete all others

# Delete statistical
./delete_embeddings_by_model.sh statistical --confirm

# Delete arcface (if exists)
./delete_embeddings_by_model.sh arcface --confirm

# Check what's left
python3 manage_embeddings.py list
```

---

## 🔍 **Check What You Have**

### Current Status:
```bash
python3 manage_embeddings.py list
```

Expected output:
```
+-------------+--------+------------------+------------------+
| Model       | Count  | First Added      | Last Added       |
+-------------+--------+------------------+------------------+
| facenet     | 29,135 | 2025-10-31 11:00 | 2025-11-01 21:39 |
| statistical | 1      | 2025-11-02 00:22 | 2025-11-02 00:22 |
+-------------+--------+------------------+------------------+

Total embeddings: 29,136
```

---

## ⚠️ **Important Notes**

### About Deletion:
1. ✅ **Safe**: Deleting by model only removes database records
2. ✅ **Files preserved**: Original image and JSON files are NOT deleted
3. ⚠️  **Irreversible**: Database records cannot be recovered after deletion
4. ⚠️  **Re-embeddable**: You can re-embed the images later

### Which Model to Keep:
- **facenet**: Good accuracy, balanced speed (RECOMMENDED)
- **statistical**: Fast but less accurate
- **arcface**: Most accurate but slower

### Recommendation:
**Keep facenet** embeddings (29,135) and delete statistical (1).

---

## 🚀 **Quick Commands**

### List models:
```bash
python3 manage_embeddings.py list
```

### Delete statistical model:
```bash
# Dry-run first
python3 manage_embeddings.py delete statistical

# Actually delete
echo "DELETE" | python3 manage_embeddings.py delete statistical --confirm
```

### Verify deletion:
```bash
python3 manage_embeddings.py list
```

---

## 📊 **Your Current Situation**

You have:
- **facenet**: 29,135 embeddings ✅ (KEEP THIS)
- **statistical**: 1 embedding (test record, can delete)

Recommendation:
```bash
# Delete the test statistical embedding
echo "DELETE" | python3 manage_embeddings.py delete statistical --confirm

# Verify
python3 manage_embeddings.py list
```

After this, you'll have 29,135 clean facenet embeddings! 🎉

---

## 🛠️ **Troubleshooting**

### "No module named psycopg2"
```bash
pip install psycopg2-binary
```

### "Permission denied"
```bash
chmod +x manage_embeddings.py
chmod +x delete_embeddings_by_model.sh
```

### Can't connect to database
```bash
# Test connection
PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c "SELECT 1;"
```

---

## 📁 **Files**

- `manage_embeddings.py` - Python management script
- `delete_embeddings_by_model.sh` - Shell script
- `EMBEDDING_MANAGEMENT_GUIDE.md` - This guide

---

**Created by Claude Code** 🤖
