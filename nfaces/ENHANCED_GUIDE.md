

# Enhanced pgvector Monitor & Query Tools - Complete Guide

## 🎯 What You Have Now

Three powerful tools for working with your pgvector database:

1. **Enhanced GUI Monitor** (`monitor_enhanced.py`) - Cleaner UI with query interface
2. **CLI Query Tool** (`vector_query_cli.py`) - Terminal-based querying
3. **Original Monitor** (`monitor.py`) - Basic monitoring (still available)

---

## 🖥️ Tool 1: Enhanced GUI Monitor

### Quick Start
```bash
python3 monitor_enhanced.py
```

### Features

#### **Tab 1: 🔍 Query & Search** (NEW!)

**Two Query Modes:**

##### A. Metadata Search
Search faces by attributes without writing SQL:

- **Gender Filter**: male, female, unknown
- **Age Range**: min/max age
- **Brightness**: 0-255 range
- **Skin Tone**: very_light, light, medium, tan, brown, dark
- **Hair Color**: black, brown, blonde, red, gray, etc.
- **Max Results**: limit number of results

**Example Query:**
```
Gender: female
Age Range: 25 to 35
Brightness: 100 to 200
Skin Tone: medium
→ Click "SEARCH"
```

Results show as image grid with details!

##### B. Vector Similarity Search
Find similar faces using embeddings:

**Two Ways to Select Query Face:**
1. **Pick Random Face** - Get a random face from database
2. **Select from Results** - Choose from previous search results

**Distance Metrics:**
- **Cosine** (default) - Best for normalized vectors
- **L2** (Euclidean) - Geometric distance
- **Inner Product** - Dot product similarity

**Example:**
```
1. Click "Pick Random Face"
2. Select distance metric: cosine
3. Top K Results: 10
4. Click "SEARCH"
```

Shows 10 most similar faces with distance scores!

#### **Tab 2: 📊 Statistics**
Clean statistics view:
- Total faces and vectors
- Database sizes
- Model breakdown
- Date ranges

#### **Tab 3: ⚡ SQL Console**
Execute custom SQL queries:

**Example Queries** (click to load):
- Vector Count
- Recent 10 Faces
- Gender Distribution
- Avg Brightness by Gender

**Custom Query Example:**
```sql
SELECT gender, AVG(brightness), COUNT(*)
FROM faces
WHERE brightness > 100
GROUP BY gender
ORDER BY AVG(brightness) DESC;
```

Click "Execute Query" → See results in table!

### Top Status Bar
Simple real-time info:
- Total Faces
- Vectors
- DB Size
- Last Update
- Auto-refresh toggle

---

## 💻 Tool 2: CLI Query Tool

### Quick Start
```bash
python3 vector_query_cli.py
```

### Main Menu

```
1. 📊 Show Statistics
2. 🔍 Metadata Search
3. 🎯 Vector Similarity Search
4. ⚡ Quick Queries
5. 💻 Custom SQL Query
6. ❌ Exit
```

### Feature Details

#### 1. Show Statistics
```
Select option [1-6]: 1

DATABASE STATISTICS
======================================================================
Total Faces: 4332
Vectors (with embeddings): 4332

Embedding Models:
  facenet: 4332

Date Range:
  Oldest: 2025-10-25 04:24:29
  Newest: 2025-10-31 11:53:36

Database Size: 37 MB
======================================================================
```

#### 2. Metadata Search
Interactive prompts:
```
Gender (male/female/unknown/Enter to skip): female
Minimum age (Enter to skip): 25
Maximum age (Enter to skip): 40
Minimum brightness 0-255 (Enter to skip): 120
Maximum brightness 0-255 (Enter to skip): 200
Maximum results [default 20]: 10

🔍 Searching...

✓ Found 10 results:

#    Face ID                             Gender     Age      Brightness Created
----------------------------------------------------------------------------------------------------
1    face_1761878433_f8ea9d27...        female     32       145.2      2025-10-31 11:40
2    face_1761878432_555687df...        female     28       158.7      2025-10-31 11:39
...

Enter row number to see details (or Enter to skip):
```

#### 3. Vector Similarity Search
```
Select option [1-6]: 3

VECTOR SIMILARITY SEARCH
======================================================================

🎲 Picking a random face for similarity search...
✓ Query face: face_1761878433_f8ea9d27

Distance metrics:
  1. Cosine (default)
  2. L2 (Euclidean)
  3. Inner Product

Select metric [1-3, default 1]: 1
Number of results [default 10]: 5

🔍 Finding similar faces using cosine distance...

✓ Found 5 similar faces:

#    Face ID                             Gender     Brightness Distance
----------------------------------------------------------------------------------
🎯 1  face_1761878433_f8ea9d27...        female     145.2      0.000000
   2  face_1761878420_abc123de...        female     148.5      0.023451
   3  face_1761878410_def456gh...        female     142.1      0.034872
   4  face_1761878405_ghi789jk...        female     150.3      0.045123
   5  face_1761878398_lmn012op...        female     143.9      0.052678
```

#### 4. Quick Queries
Pre-defined useful queries:

```
1. Vector count
2. Gender distribution
3. Avg brightness by gender
4. Recent 10 faces
5. Brightest faces
6. Darkest faces

Select query [1-6]: 3

🔍 Executing: Avg brightness by gender...

✓ Results:

gender               | avg
--------------------------------------------------
male                 | 128.45
female               | 135.72
unknown              | 131.20
```

#### 5. Custom SQL Query
Execute any SQL:

```
Enter your SQL query (end with semicolon and press Enter twice):
Example: SELECT COUNT(*) FROM faces WHERE gender = 'female';

SELECT embedding_model, COUNT(*), AVG(brightness)
FROM faces
GROUP BY embedding_model;

🔍 Executing query...

✓ Query returned 1 rows:

embedding_model      | count               | avg
------------------------------------------------------------------------
facenet              | 4332                | 132.15
```

---

## 📚 Vector Query Examples

### Essential Queries

#### 1. Count Vectors
```sql
SELECT COUNT(*) FROM faces WHERE embedding IS NOT NULL;
```

#### 2. Find by Gender
```sql
SELECT face_id, brightness, created_at
FROM faces
WHERE gender = 'female'
ORDER BY created_at DESC
LIMIT 20;
```

#### 3. Age Range Search
```sql
SELECT face_id, age_estimate, gender, brightness
FROM faces
WHERE age_estimate BETWEEN 25 AND 35
ORDER BY brightness DESC;
```

#### 4. Brightness Range
```sql
SELECT face_id, brightness, gender
FROM faces
WHERE brightness BETWEEN 100 AND 150
ORDER BY created_at DESC
LIMIT 50;
```

#### 5. Vector Similarity (Cosine)
```sql
-- First get embedding from a face
SELECT embedding FROM faces WHERE face_id = 'your_face_id' LIMIT 1;

-- Then find similar (replace [...] with actual embedding)
SELECT face_id, gender, brightness,
       embedding <=> '[...]'::vector AS distance
FROM faces
WHERE embedding IS NOT NULL
ORDER BY distance
LIMIT 10;
```

#### 6. Metadata JSON Queries
```sql
-- Find by skin tone (in metadata JSONB)
SELECT face_id, metadata->>'skin_tone', metadata->>'hair_color'
FROM faces
WHERE metadata->>'skin_tone' = 'medium'
LIMIT 20;
```

#### 7. Combine Metadata + Vector
```sql
-- Find similar faces with gender filter
SELECT face_id, gender, brightness,
       embedding <=> '[...]'::vector AS distance
FROM faces
WHERE embedding IS NOT NULL
  AND gender = 'female'
ORDER BY distance
LIMIT 10;
```

#### 8. Statistics by Model
```sql
SELECT embedding_model,
       COUNT(*) as total,
       AVG(brightness) as avg_brightness,
       MIN(created_at) as first_face,
       MAX(created_at) as last_face
FROM faces
GROUP BY embedding_model;
```

#### 9. Recent Additions
```sql
SELECT face_id, gender, brightness,
       created_at,
       created_at > NOW() - INTERVAL '1 hour' as is_recent
FROM faces
ORDER BY created_at DESC
LIMIT 20;
```

#### 10. Advanced Aggregations
```sql
SELECT
    gender,
    COUNT(*) as count,
    AVG(brightness) as avg_brightness,
    MIN(brightness) as min_brightness,
    MAX(brightness) as max_brightness,
    STDDEV(brightness) as stddev_brightness
FROM faces
WHERE gender IS NOT NULL
GROUP BY gender
ORDER BY count DESC;
```

### Vector-Specific Queries

#### 11. L2 Distance (Euclidean)
```sql
SELECT face_id, gender,
       embedding <-> '[...]'::vector AS l2_distance
FROM faces
WHERE embedding IS NOT NULL
ORDER BY l2_distance
LIMIT 10;
```

#### 12. Inner Product Similarity
```sql
SELECT face_id, gender,
       embedding <#> '[...]'::vector AS inner_product
FROM faces
WHERE embedding IS NOT NULL
ORDER BY inner_product
LIMIT 10;
```

#### 13. Check Embedding Dimensions
```sql
SELECT face_id, array_length(embedding, 1) as dimension
FROM faces
WHERE embedding IS NOT NULL
LIMIT 10;
```

#### 14. Faces Without Embeddings
```sql
SELECT face_id, file_path, created_at
FROM faces
WHERE embedding IS NULL
ORDER BY created_at DESC;
```

### DDL Operations (Database Management)

#### 15. Create Index on Gender
```sql
CREATE INDEX IF NOT EXISTS idx_faces_gender ON faces(gender);
```

#### 16. Create Index on Brightness
```sql
CREATE INDEX IF NOT EXISTS idx_faces_brightness ON faces(brightness);
```

#### 17. Create Index on Metadata JSONB
```sql
CREATE INDEX IF NOT EXISTS idx_faces_metadata ON faces USING GIN(metadata);
```

#### 18. Analyze Table for Query Optimization
```sql
ANALYZE faces;
```

#### 19. Vacuum Table
```sql
VACUUM ANALYZE faces;
```

#### 20. Check Table Size Details
```sql
SELECT
    pg_size_pretty(pg_total_relation_size('faces')) as total_size,
    pg_size_pretty(pg_relation_size('faces')) as table_size,
    pg_size_pretty(pg_total_relation_size('faces') - pg_relation_size('faces')) as index_size;
```

---

## 🎓 Use Cases

### Use Case 1: Find Similar Faces

**GUI Method:**
1. Open `monitor_enhanced.py`
2. Go to "Query & Search" tab
3. Select "Vector Similarity"
4. Click "Pick Random Face"
5. Select distance metric
6. Click "SEARCH"
7. View grid of similar faces with images

**CLI Method:**
```bash
python3 vector_query_cli.py
# Select option 3 (Vector Similarity Search)
# Follow prompts
```

### Use Case 2: Filter by Demographics

**GUI Method:**
1. Select "Metadata Search"
2. Set filters:
   - Gender: female
   - Age: 25-35
   - Brightness: 120-180
3. Click "SEARCH"
4. Browse results with images

**CLI Method:**
```bash
python3 vector_query_cli.py
# Select option 2 (Metadata Search)
# Enter: female, 25, 35, 120, 180, 20
```

### Use Case 3: Database Statistics

**GUI Method:**
1. Go to "Statistics" tab
2. Click "Refresh Statistics"

**CLI Method:**
```bash
python3 vector_query_cli.py
# Select option 1 (Show Statistics)
```

### Use Case 4: Custom Analysis

**GUI Method:**
1. Go to "SQL Console" tab
2. Enter query:
```sql
SELECT gender, AVG(brightness), COUNT(*)
FROM faces
GROUP BY gender;
```
3. Click "Execute Query"

**CLI Method:**
```bash
python3 vector_query_cli.py
# Select option 5 (Custom SQL Query)
# Enter your SQL
```

---

## 🔧 Tips & Tricks

### Performance Tips

1. **Use Indexes**
```sql
CREATE INDEX idx_faces_brightness ON faces(brightness);
CREATE INDEX idx_faces_gender ON faces(gender);
```

2. **Limit Results**
Always use LIMIT for large queries:
```sql
SELECT * FROM faces LIMIT 100;
```

3. **Use EXPLAIN**
Check query performance:
```sql
EXPLAIN ANALYZE
SELECT * FROM faces WHERE gender = 'female';
```

### Query Tips

1. **Metadata JSONB Queries**
```sql
-- Access nested JSON
SELECT metadata->>'skin_tone' FROM faces;

-- Filter by JSON field
WHERE metadata->>'hair_color' = 'brown'
```

2. **Date Filtering**
```sql
-- Last 24 hours
WHERE created_at > NOW() - INTERVAL '24 hours'

-- Specific date
WHERE created_at::date = '2025-10-31'
```

3. **Combined Filters**
```sql
SELECT * FROM faces
WHERE gender = 'female'
  AND brightness > 120
  AND age_estimate BETWEEN 25 AND 35
  AND metadata->>'skin_tone' = 'medium'
ORDER BY created_at DESC
LIMIT 20;
```

### Workflow Tips

**Morning Routine:**
```bash
# Check database stats
python3 vector_query_cli.py
# Select 1 (Statistics)

# Find new faces
# Select 4 (Quick Queries) → 4 (Recent 10 faces)
```

**Search Workflow:**
```bash
# Start GUI for visual browsing
python3 monitor_enhanced.py

# Quick stats in terminal
python3 vector_query_cli.py
```

---

## 🆚 Tool Comparison

| Feature | Enhanced GUI | CLI Tool | Original Monitor |
|---------|-------------|----------|-----------------|
| Metadata Search | ✅ Visual | ✅ Interactive | ❌ |
| Similarity Search | ✅ Visual | ✅ Interactive | ❌ |
| SQL Console | ✅ | ✅ | ❌ |
| Image Preview | ✅ | ❌ | ✅ |
| Export Results | ✅ | ❌ | ❌ |
| Real-time Monitor | ✅ | ❌ | ✅ |
| Scripting Friendly | ❌ | ✅ | ❌ |
| Quick Stats | ✅ | ✅ | ✅ |

**When to Use:**
- **Enhanced GUI**: Visual exploration, image browsing, drag-and-drop queries
- **CLI Tool**: Quick searches, scripting, SSH sessions
- **Original Monitor**: Pure monitoring, no queries needed

---

## 🚀 Quick Reference

### Launch Commands
```bash
# Enhanced GUI with queries
python3 monitor_enhanced.py

# Terminal CLI
python3 vector_query_cli.py

# Original monitor (monitoring only)
python3 monitor.py
```

### Common Queries

**Find by gender:**
```sql
SELECT * FROM faces WHERE gender = 'female' LIMIT 20;
```

**Age range:**
```sql
SELECT * FROM faces WHERE age_estimate BETWEEN 25 AND 35;
```

**Brightness:**
```sql
SELECT * FROM faces WHERE brightness > 150 ORDER BY brightness DESC;
```

**Similar faces:**
```sql
SELECT face_id, embedding <=> '[...]'::vector AS dist
FROM faces ORDER BY dist LIMIT 10;
```

---

## 📝 Summary

You now have a complete toolkit:

✅ **Enhanced GUI** - Visual querying with images
✅ **CLI Tool** - Fast terminal-based queries
✅ **SQL Console** - Full SQL power
✅ **Metadata Search** - No SQL needed
✅ **Vector Similarity** - Find similar faces
✅ **Export** - Save results
✅ **Real-time Stats** - Monitor growth

**Your database:**
- 4,300+ vectors
- Fully searchable
- Multiple query methods
- Production ready!

Enjoy your enhanced vector database tools! 🎉
