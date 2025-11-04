# Multi-Model Face Recognition - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Install System
```bash
cd /home/pi/hybridrag/faces10_mm
./install.sh
```

### 2. Configure Models
Edit `.env` file:
```bash
EMBEDDING_MODELS=facenet,arcface
DEFAULT_SEARCH_MODEL=facenet
```

### 3. Run Embedding
```bash
./run_embedding.sh
# Choose: 6 (all models) or 7 (custom)
```

### 4. Start Application
```bash
./start_system.sh
```

---

## 📊 Available Models

| # | Model | Best For | Speed |
|---|-------|----------|-------|
| 1 | statistical | Testing | ⚡⚡⚡ |
| 2 | facenet | General use | ⚡⚡ |
| 3 | arcface | Accuracy | ⚡ |
| 4 | vggface2 | Balanced | ⚡⚡ |
| 5 | insightface | Best accuracy | ⚡ |

---

## 🔍 Common Tasks

### Embed with 2 models:
```bash
./run_embedding.sh
> 7
> facenet,arcface
```

### Embed with all models:
```bash
./run_embedding.sh
> 6
```

### Check database stats:
```bash
sudo -u postgres psql -d vector_db -c "SELECT * FROM get_database_stats();"
```

### Search with specific model:
```sql
SELECT * FROM search_similar_faces(
    '[...]'::vector(512),
    'facenet',
    10,
    0.8
);
```

---

## ⚙️ Configuration

### .env file:
```bash
# Models to use (comma-separated)
EMBEDDING_MODELS=facenet,arcface

# Default model for searching
DEFAULT_SEARCH_MODEL=facenet

# Database settings
POSTGRES_HOST=localhost
POSTGRES_DB=vector_db
```

---

## 📁 Database Schema

### Table Structure:
```
faces
├── id (primary key)
├── face_id (unique)
├── file_path
├── embedding_facenet      ← FaceNet model
├── embedding_arcface      ← ArcFace model
├── embedding_vggface2     ← VGGFace2 model
├── embedding_insightface  ← InsightFace model
├── embedding_statistical  ← Statistical model
├── models_processed       ← Array of processed models
└── metadata (jsonb)
```

---

## 🛠️ Troubleshooting

### Models not installed?
```bash
source venv/bin/activate
pip install facenet-pytorch torch torchvision
```

### Database connection failed?
```bash
sudo service postgresql start
sudo service postgresql status
```

### Slow performance?
- Use fewer models
- Reduce parallel workers
- Search with specific model instead of all

---

## 📈 Performance Tips

1. **Start with 2 models** (facenet + arcface)
2. **Use parallel workers** (2-4 for most systems)
3. **Search specific model** for faster results
4. **Use all models** only when comparing

---

## 🎯 Recommended Workflow

1. **Initial setup:** Use facenet (fast, accurate)
2. **Add arcface:** For better accuracy
3. **Compare results:** See which works best
4. **Choose default:** Set in .env file

---

For detailed information, see: `MULTIMODEL_SETUP_SUMMARY.md`
