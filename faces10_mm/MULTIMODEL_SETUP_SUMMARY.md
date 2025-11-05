# Multi-Model Face Recognition System - Setup Summary

## Overview
This document summarizes the multi-model support implementation in the faces10_mm system. The system now supports running multiple face recognition models simultaneously and storing their embeddings separately for comparison and optimal accuracy.

## Date: 2025-11-04

---

## What is Multi-Model Support?

Multi-model support allows you to:
1. **Generate embeddings** from multiple models (FaceNet, ArcFace, VGGFace2, InsightFace, Statistical)
2. **Store embeddings separately** in dedicated database columns
3. **Search using specific models** or compare results across all models
4. **Evaluate model performance** by comparing accuracy and results

---

## Changes Made

### 1. Database Schema (schema.sql)

#### Before:
- Single `embedding` column (512 dimensions)
- Single `embedding_model` field (text)
- One HNSW index for all embeddings

#### After:
- **5 separate embedding columns:**
  - `embedding_facenet` - FaceNet InceptionResnetV1
  - `embedding_arcface` - ArcFace model
  - `embedding_vggface2` - VGGFace2 deep CNN
  - `embedding_insightface` - InsightFace model
  - `embedding_statistical` - Statistical features

- **New field:**
  - `models_processed` (TEXT[]) - Array tracking which models have been applied

- **5 separate HNSW indexes:**
  - One index per embedding column for optimal search performance

#### New Functions:

**`search_similar_faces(query_embedding, model_name, limit_count, distance_threshold)`**
- Search using a specific model
- Parameters:
  - `query_embedding`: vector(512) - The face embedding to search for
  - `model_name`: TEXT - Which model to use ('facenet', 'arcface', etc.)
  - `limit_count`: INTEGER - Max results (default: 10)
  - `distance_threshold`: FLOAT - Max distance (default: 1.0)

**`search_similar_faces_all_models(query_embedding, limit_count, distance_threshold)`**
- Search across ALL models and return combined results
- Returns results sorted by distance across all models

**`get_database_stats()`**
- Updated to show counts per model:
  - `faces_with_facenet`
  - `faces_with_arcface`
  - `faces_with_vggface2`
  - `faces_with_insightface`
  - `faces_with_statistical`

---

### 2. Installation Script (install.sh)

#### Updates:
- Banner: "Multi-Model Support Edition"
- New `.env` configuration template with:
  ```bash
  # Multi-Model Configuration
  EMBEDDING_MODELS=facenet,arcface
  DEFAULT_SEARCH_MODEL=facenet
  ```

---

### 3. Embedding Script (run_embedding.sh)

#### Before:
- Single model selection (1-6)
- Process one model at a time

#### After:
- **Multi-model selection (1-7):**
  1. statistical
  2. facenet
  3. arcface
  4. vggface2
  5. insightface
  6. **all** - Process with ALL models
  7. **multi** - Custom comma-separated list

- **Sequential processing:**
  - Loops through each selected model
  - Tracks success/failure for each
  - Provides final summary

- **Example usage:**
  ```bash
  ./run_embedding.sh
  # Select option 7 for custom
  # Enter: facenet,arcface,vggface2
  ```

---

### 4. System Launcher (start_system.sh)

#### Updates:
- Banner: "Multi-Model Support Edition"
- Loads and displays `.env` configuration
- Shows:
  - Current embedding models
  - Default search model
- Updated feature list highlighting multi-model capabilities

---

## Configuration

### Environment Variables (.env)

```bash
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vector_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Multi-Model Configuration
EMBEDDING_MODELS=facenet,arcface
DEFAULT_SEARCH_MODEL=facenet

# Connection Pool Settings
DB_MIN_CONNECTIONS=1
DB_MAX_CONNECTIONS=10

# Vector Search Settings
DEFAULT_SEARCH_LIMIT=10
DISTANCE_METRIC=cosine
```

---

## Usage Examples

### 1. Installing the System
```bash
cd /home/pi/hybridrag/faces10_mm
./install.sh
```

### 2. Embedding with Multiple Models
```bash
./run_embedding.sh
# Choose option 6 for ALL models
# Or option 7 for custom selection
```

### 3. Starting the Application
```bash
./start_system.sh
```

### 4. Database Queries

#### Search with specific model:
```sql
SELECT * FROM search_similar_faces(
    '[0.1, 0.2, ..., 0.5]'::vector(512),
    'facenet',
    10,
    0.8
);
```

#### Search across all models:
```sql
SELECT * FROM search_similar_faces_all_models(
    '[0.1, 0.2, ..., 0.5]'::vector(512),
    10,
    0.8
);
```

#### Get statistics:
```sql
SELECT * FROM get_database_stats();
```

#### Check which models processed a face:
```sql
SELECT face_id, file_path, models_processed
FROM faces
WHERE face_id = 'some_face_id';
```

---

## Supported Models

| Model | Library | Dimensions | Accuracy | Speed |
|-------|---------|------------|----------|-------|
| **FaceNet** | facenet-pytorch | 512 | High | Fast |
| **ArcFace** | Custom/InsightFace | 512 | Very High | Medium |
| **VGGFace2** | facenet-pytorch | 512 | High | Medium |
| **InsightFace** | insightface | 512 | Very High | Medium |
| **Statistical** | Built-in | 512* | Low | Very Fast |

*Statistical model uses 7 features padded to 512 dimensions

---

## Benefits of Multi-Model Approach

### 1. **Robustness**
- Different models excel with different face types
- Lighting, angles, age variations handled differently
- Fallback options if one model performs poorly

### 2. **Comparison**
- Identify best model for your dataset
- Track which model gives best results
- A/B test different models

### 3. **Ensemble Methods**
- Combine results from multiple models
- Vote-based or weighted averaging
- Improved overall accuracy

### 4. **Flexibility**
- Search with one model or all models
- Switch models without re-processing
- Add new models without breaking existing data

---

## Performance Considerations

### Storage Impact:
- Each model adds ~2KB per face (512 dims × 4 bytes)
- 5 models = ~10KB per face
- 10,000 faces = ~100MB of embedding data

### Processing Time:
- Multiple models take longer initially
- Can process models in parallel (future enhancement)
- Once embedded, searches are equally fast

### Index Building:
- Each HNSW index takes time to build
- Built automatically on schema creation
- Maintains fast search performance

---

## Migration from Single-Model

If migrating from the old single-model system:

1. **Backup your data:**
   ```bash
   pg_dump vector_db > backup.sql
   ```

2. **Run install.sh:**
   - Will prompt to recreate schema
   - **WARNING: This deletes all data!**

3. **Re-embed all faces:**
   ```bash
   ./run_embedding.sh
   # Choose option 6 for all models
   ```

---

## Troubleshooting

### Issue: Model not installed
```
Error: No module named 'facenet_pytorch'
```
**Solution:**
```bash
source venv/bin/activate
pip install facenet-pytorch torch torchvision
```

### Issue: Out of memory
**Solution:**
- Reduce number of parallel workers
- Process one model at a time
- Increase system RAM or swap

### Issue: Slow searches
**Solution:**
- Ensure HNSW indexes are created
- Check `ef_search` parameter
- Use specific model instead of all models

---

## Future Enhancements

1. **Parallel model processing** - Process multiple models simultaneously
2. **Model ensemble voting** - Combine results intelligently
3. **Dynamic model loading** - Load models on demand
4. **Model performance metrics** - Track accuracy per model
5. **Web API** - REST API for multi-model search

---

## File Structure

```
faces10_mm/
├── schema.sql                    # Multi-model database schema ✓
├── install.sh                    # Installation with multi-model config ✓
├── run_embedding.sh              # Multi-model embedding script ✓
├── start_system.sh               # Launcher with model display ✓
├── .env                          # Configuration file
├── faces.py                      # Main application (requires update)
├── pgvector_db.py               # Database layer (requires update)
├── embedding_manager_cli.py     # Embedding manager (requires update)
└── MULTIMODEL_SETUP_SUMMARY.md  # This document
```

---

## Next Steps for Full Implementation

### Python Code Updates Required:

1. **pgvector_db.py:**
   - Update `insert_face()` to handle multiple embedding columns
   - Update `search_similar_faces()` to accept model parameter
   - Add `search_all_models()` method

2. **core.py:**
   - Support multiple model loading
   - Update embedding generation for all configured models
   - Track models_processed array

3. **embedding_manager_cli.py:**
   - Support `--models` parameter with comma-separated list
   - Process multiple models per face

4. **faces.py (GUI):**
   - Add model selection dropdown for search
   - Display which models are available
   - Show comparison results

---

## Testing Checklist

- [ ] Install system with `./install.sh`
- [ ] Verify schema has 5 embedding columns
- [ ] Test single model embedding
- [ ] Test multi-model embedding (2-3 models)
- [ ] Test "all models" option
- [ ] Verify search with specific model
- [ ] Verify search across all models
- [ ] Check database stats show correct counts
- [ ] Test with 100+ faces
- [ ] Monitor memory usage
- [ ] Compare search accuracy across models

---

## Contact & Support

For issues or questions about this multi-model implementation, check:
- Database schema: `schema.sql`
- Configuration: `.env` file
- Logs: Check console output during embedding

---

**System Status: ✅ Multi-Model Infrastructure Complete**

**Remaining Work:** Python code updates for full integration
