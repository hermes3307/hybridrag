# Advanced Vector SQL Query Examples for pgvector

## 📚 Complete Guide to Vector Queries in PostgreSQL

---

## 🎯 Table of Contents

1. [Basic Vector Operations](#basic-vector-operations)
2. [Distance Metrics Explained](#distance-metrics-explained)
3. [Similarity Search Queries](#similarity-search-queries)
4. [Hybrid Queries (Vector + Metadata)](#hybrid-queries)
5. [Advanced Vector Analytics](#advanced-vector-analytics)
6. [Performance Optimization](#performance-optimization)
7. [Vector Aggregations](#vector-aggregations)
8. [Batch Operations](#batch-operations)
9. [Quality Control Queries](#quality-control-queries)
10. [Real-World Use Cases](#real-world-use-cases)

---

## 1. Basic Vector Operations

### 1.1 Get Embedding for a Face
```sql
SELECT face_id, embedding
FROM faces
WHERE face_id = 'your_face_id_here'
LIMIT 1;
```

### 1.2 Check Embedding Dimension
```sql
SELECT face_id,
       array_length(embedding, 1) as dimension,
       embedding_model
FROM faces
WHERE embedding IS NOT NULL
LIMIT 10;
```

### 1.3 Count Vectors by Model
```sql
SELECT embedding_model,
       COUNT(*) as total_vectors
FROM faces
WHERE embedding IS NOT NULL
GROUP BY embedding_model
ORDER BY total_vectors DESC;
```

### 1.4 Find Faces Without Embeddings
```sql
SELECT face_id, file_path, created_at
FROM faces
WHERE embedding IS NULL
ORDER BY created_at DESC;
```

### 1.5 Get Random Face with Embedding
```sql
SELECT face_id, file_path, embedding
FROM faces
WHERE embedding IS NOT NULL
ORDER BY RANDOM()
LIMIT 1;
```

---

## 2. Distance Metrics Explained

### 2.1 Cosine Distance (`<=>`)
**Best for:** Normalized vectors, semantic similarity
**Range:** 0 (identical) to 2 (opposite)
**Use when:** Direction matters more than magnitude

```sql
-- Find similar faces using cosine distance
SELECT face_id, gender, brightness,
       embedding <=> (SELECT embedding FROM faces WHERE face_id = 'query_face_id')::vector AS cosine_distance
FROM faces
WHERE embedding IS NOT NULL
  AND face_id != 'query_face_id'
ORDER BY cosine_distance
LIMIT 10;
```

### 2.2 L2/Euclidean Distance (`<->`)
**Best for:** Geometric distance, spatial similarity
**Range:** 0 (identical) to infinity
**Use when:** Absolute distance matters

```sql
-- Find similar faces using L2 distance
SELECT face_id, gender, brightness,
       embedding <-> (SELECT embedding FROM faces WHERE face_id = 'query_face_id')::vector AS l2_distance
FROM faces
WHERE embedding IS NOT NULL
  AND face_id != 'query_face_id'
ORDER BY l2_distance
LIMIT 10;
```

### 2.3 Inner Product (`<#>`)
**Best for:** Dot product similarity, magnitude-aware
**Range:** Negative (dissimilar) to positive (similar)
**Use when:** Both direction and magnitude matter

```sql
-- Find similar faces using inner product
SELECT face_id, gender, brightness,
       embedding <#> (SELECT embedding FROM faces WHERE face_id = 'query_face_id')::vector AS inner_product
FROM faces
WHERE embedding IS NOT NULL
  AND face_id != 'query_face_id'
ORDER BY inner_product
LIMIT 10;
```

### 2.4 Compare All Three Metrics
```sql
WITH query_embedding AS (
    SELECT embedding FROM faces WHERE face_id = 'query_face_id'
)
SELECT face_id, gender,
       embedding <=> (SELECT embedding FROM query_embedding) AS cosine_dist,
       embedding <-> (SELECT embedding FROM query_embedding) AS l2_dist,
       embedding <#> (SELECT embedding FROM query_embedding) AS inner_prod
FROM faces
WHERE embedding IS NOT NULL
  AND face_id != 'query_face_id'
ORDER BY cosine_dist
LIMIT 10;
```

---

## 3. Similarity Search Queries

### 3.1 Find Top K Similar Faces
```sql
-- Using subquery for query embedding
SELECT face_id, file_path, gender, age_estimate,
       embedding <=> (
           SELECT embedding FROM faces WHERE face_id = 'query_face_id'
       )::vector AS distance
FROM faces
WHERE embedding IS NOT NULL
  AND face_id != 'query_face_id'
ORDER BY distance
LIMIT 10;
```

### 3.2 Similarity Search with Distance Threshold
```sql
-- Only return faces within certain similarity threshold
WITH query_embedding AS (
    SELECT embedding FROM faces WHERE face_id = 'query_face_id'
)
SELECT face_id, gender, brightness,
       embedding <=> (SELECT embedding FROM query_embedding) AS distance
FROM faces
WHERE embedding IS NOT NULL
  AND face_id != 'query_face_id'
  AND embedding <=> (SELECT embedding FROM query_embedding) < 0.5  -- Threshold
ORDER BY distance
LIMIT 20;
```

### 3.3 Similarity Search Excluding Self
```sql
-- Ensure query face is excluded from results
WITH query_face AS (
    SELECT face_id, embedding FROM faces WHERE face_id = 'query_face_id'
)
SELECT f.face_id, f.gender, f.brightness,
       f.embedding <=> q.embedding AS distance
FROM faces f, query_face q
WHERE f.embedding IS NOT NULL
  AND f.face_id != q.face_id
ORDER BY distance
LIMIT 10;
```

### 3.4 Batch Similarity Search (Multiple Queries)
```sql
-- Find similar faces for multiple query faces
WITH query_faces AS (
    SELECT face_id, embedding
    FROM faces
    WHERE face_id IN ('face_id_1', 'face_id_2', 'face_id_3')
)
SELECT qf.face_id as query_face,
       f.face_id as similar_face,
       f.gender,
       f.embedding <=> qf.embedding AS distance
FROM faces f
CROSS JOIN query_faces qf
WHERE f.embedding IS NOT NULL
  AND f.face_id != qf.face_id
  AND f.embedding <=> qf.embedding < 0.3
ORDER BY qf.face_id, distance;
```

---

## 4. Hybrid Queries (Vector + Metadata)

### 4.1 Similar Faces with Gender Filter
```sql
-- Find similar female faces only
WITH query_embedding AS (
    SELECT embedding FROM faces WHERE face_id = 'query_face_id'
)
SELECT face_id, gender, age_estimate, brightness,
       embedding <=> (SELECT embedding FROM query_embedding) AS distance
FROM faces
WHERE embedding IS NOT NULL
  AND face_id != 'query_face_id'
  AND gender = 'female'
ORDER BY distance
LIMIT 10;
```

### 4.2 Similar Faces with Age Range
```sql
-- Find similar faces within age range
WITH query_embedding AS (
    SELECT embedding FROM faces WHERE face_id = 'query_face_id'
)
SELECT face_id, gender, age_estimate,
       embedding <=> (SELECT embedding FROM query_embedding) AS distance
FROM faces
WHERE embedding IS NOT NULL
  AND face_id != 'query_face_id'
  AND age_estimate BETWEEN 25 AND 35
ORDER BY distance
LIMIT 10;
```

### 4.3 Similar Faces with Brightness Range
```sql
-- Find similar faces with similar brightness
WITH query_face AS (
    SELECT embedding, brightness FROM faces WHERE face_id = 'query_face_id'
)
SELECT f.face_id, f.gender, f.brightness,
       f.embedding <=> q.embedding AS distance,
       ABS(f.brightness - q.brightness) AS brightness_diff
FROM faces f, query_face q
WHERE f.embedding IS NOT NULL
  AND f.face_id != 'query_face_id'
  AND f.brightness BETWEEN (q.brightness - 30) AND (q.brightness + 30)
ORDER BY distance
LIMIT 10;
```

### 4.4 Similar Faces with Multiple Filters
```sql
-- Complex filtering: gender, age, brightness, and skin tone
WITH query_embedding AS (
    SELECT embedding FROM faces WHERE face_id = 'query_face_id'
)
SELECT face_id, gender, age_estimate, brightness,
       metadata->>'skin_tone' as skin_tone,
       embedding <=> (SELECT embedding FROM query_embedding) AS distance
FROM faces
WHERE embedding IS NOT NULL
  AND face_id != 'query_face_id'
  AND gender = 'female'
  AND age_estimate BETWEEN 25 AND 40
  AND brightness > 100
  AND metadata->>'skin_tone' IN ('medium', 'tan')
ORDER BY distance
LIMIT 15;
```

### 4.5 Similar Faces Created Recently
```sql
-- Find similar faces added in last 24 hours
WITH query_embedding AS (
    SELECT embedding FROM faces WHERE face_id = 'query_face_id'
)
SELECT face_id, created_at, gender,
       embedding <=> (SELECT embedding FROM query_embedding) AS distance
FROM faces
WHERE embedding IS NOT NULL
  AND face_id != 'query_face_id'
  AND created_at > NOW() - INTERVAL '24 hours'
ORDER BY distance
LIMIT 10;
```

---

## 5. Advanced Vector Analytics

### 5.1 Find Outliers (Least Similar to Any Face)
```sql
-- Faces that are unique/outliers
WITH similarity_scores AS (
    SELECT f1.face_id,
           MIN(f1.embedding <=> f2.embedding) as min_distance
    FROM faces f1
    CROSS JOIN faces f2
    WHERE f1.embedding IS NOT NULL
      AND f2.embedding IS NOT NULL
      AND f1.face_id != f2.face_id
    GROUP BY f1.face_id
)
SELECT face_id, min_distance
FROM similarity_scores
ORDER BY min_distance DESC
LIMIT 10;
```

### 5.2 Find Clusters (Most Similar Pairs)
```sql
-- Find pairs of very similar faces
SELECT f1.face_id as face1,
       f2.face_id as face2,
       f1.gender as gender1,
       f2.gender as gender2,
       f1.embedding <=> f2.embedding AS distance
FROM faces f1
JOIN faces f2 ON f1.face_id < f2.face_id
WHERE f1.embedding IS NOT NULL
  AND f2.embedding IS NOT NULL
  AND f1.embedding <=> f2.embedding < 0.1  -- Very similar threshold
ORDER BY distance
LIMIT 20;
```

### 5.3 Average Distance to All Faces
```sql
-- Calculate average similarity for each face
SELECT f1.face_id,
       f1.gender,
       AVG(f1.embedding <=> f2.embedding) as avg_distance,
       COUNT(*) as compared_faces
FROM faces f1
CROSS JOIN faces f2
WHERE f1.embedding IS NOT NULL
  AND f2.embedding IS NOT NULL
  AND f1.face_id != f2.face_id
GROUP BY f1.face_id, f1.gender
ORDER BY avg_distance DESC
LIMIT 20;
```

### 5.4 Distance Distribution
```sql
-- Analyze distribution of distances
WITH distances AS (
    SELECT f1.embedding <=> f2.embedding AS distance
    FROM faces f1
    CROSS JOIN faces f2
    WHERE f1.embedding IS NOT NULL
      AND f2.embedding IS NOT NULL
      AND f1.face_id < f2.face_id
    LIMIT 10000  -- Sample for performance
)
SELECT
    MIN(distance) as min_distance,
    MAX(distance) as max_distance,
    AVG(distance) as avg_distance,
    STDDEV(distance) as stddev_distance,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY distance) as q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY distance) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY distance) as q3
FROM distances;
```

### 5.5 Gender-Based Similarity Analysis
```sql
-- Compare average similarity within vs between genders
WITH same_gender AS (
    SELECT AVG(f1.embedding <=> f2.embedding) as avg_dist
    FROM faces f1
    JOIN faces f2 ON f1.gender = f2.gender AND f1.face_id < f2.face_id
    WHERE f1.embedding IS NOT NULL AND f2.embedding IS NOT NULL
    LIMIT 1000
),
diff_gender AS (
    SELECT AVG(f1.embedding <=> f2.embedding) as avg_dist
    FROM faces f1
    JOIN faces f2 ON f1.gender != f2.gender AND f1.face_id < f2.face_id
    WHERE f1.embedding IS NOT NULL AND f2.embedding IS NOT NULL
      AND f1.gender IS NOT NULL AND f2.gender IS NOT NULL
    LIMIT 1000
)
SELECT
    (SELECT avg_dist FROM same_gender) as same_gender_similarity,
    (SELECT avg_dist FROM diff_gender) as diff_gender_similarity;
```

---

## 6. Performance Optimization

### 6.1 Create Vector Index (HNSW - Recommended)
```sql
-- Create index for fast similarity search
CREATE INDEX IF NOT EXISTS faces_embedding_hnsw_idx
ON faces
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- For L2 distance
CREATE INDEX IF NOT EXISTS faces_embedding_l2_idx
ON faces
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);
```

### 6.2 Create IVFFlat Index (Alternative)
```sql
-- Create IVFFlat index (faster build, slower search)
CREATE INDEX IF NOT EXISTS faces_embedding_ivfflat_idx
ON faces
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### 6.3 Analyze Query Performance
```sql
-- Use EXPLAIN ANALYZE to check query performance
EXPLAIN ANALYZE
SELECT face_id,
       embedding <=> (SELECT embedding FROM faces WHERE face_id = 'query_face_id')::vector AS distance
FROM faces
WHERE embedding IS NOT NULL
ORDER BY distance
LIMIT 10;
```

### 6.4 Set Vector Search Parameters
```sql
-- Adjust HNSW search parameters (session-level)
SET hnsw.ef_search = 100;  -- Higher = more accurate but slower

-- Reset to default
RESET hnsw.ef_search;
```

### 6.5 Vacuum and Analyze
```sql
-- Optimize table and update statistics
VACUUM ANALYZE faces;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE tablename = 'faces';
```

---

## 7. Vector Aggregations

### 7.1 Average Vector by Gender
```sql
-- Calculate average embedding for each gender
-- Note: This creates a "centroid" vector
SELECT gender,
       COUNT(*) as count,
       AVG(embedding) as avg_embedding
FROM faces
WHERE embedding IS NOT NULL
  AND gender IS NOT NULL
GROUP BY gender;
```

### 7.2 Find Faces Closest to Gender Centroid
```sql
-- Find most "typical" faces for each gender
WITH gender_centroids AS (
    SELECT gender,
           AVG(embedding) as centroid_embedding
    FROM faces
    WHERE embedding IS NOT NULL
      AND gender IS NOT NULL
    GROUP BY gender
)
SELECT f.face_id, f.gender, f.brightness,
       f.embedding <=> gc.centroid_embedding AS distance_from_centroid
FROM faces f
JOIN gender_centroids gc ON f.gender = gc.gender
WHERE f.embedding IS NOT NULL
ORDER BY f.gender, distance_from_centroid
LIMIT 5;
```

### 7.3 Distance from Dataset Mean
```sql
-- Find faces furthest from dataset average
WITH dataset_mean AS (
    SELECT AVG(embedding) as mean_embedding
    FROM faces
    WHERE embedding IS NOT NULL
)
SELECT face_id, gender, brightness,
       embedding <=> (SELECT mean_embedding FROM dataset_mean) AS distance_from_mean
FROM faces
WHERE embedding IS NOT NULL
ORDER BY distance_from_mean DESC
LIMIT 20;
```

---

## 8. Batch Operations

### 8.1 Update Multiple Embeddings
```sql
-- Update embeddings in batch
UPDATE faces
SET embedding = new_data.embedding,
    updated_at = NOW()
FROM (VALUES
    ('face_id_1'::text, '[0.1, 0.2, ...]'::vector),
    ('face_id_2'::text, '[0.3, 0.4, ...]'::vector)
) AS new_data(face_id, embedding)
WHERE faces.face_id = new_data.face_id;
```

### 8.2 Bulk Similarity Search
```sql
-- Find similar faces for top 10 brightest faces
WITH bright_faces AS (
    SELECT face_id, embedding
    FROM faces
    WHERE embedding IS NOT NULL
    ORDER BY brightness DESC
    LIMIT 10
)
SELECT bf.face_id as query_face,
       f.face_id as similar_face,
       f.gender,
       f.embedding <=> bf.embedding AS distance
FROM faces f
CROSS JOIN bright_faces bf
WHERE f.embedding IS NOT NULL
  AND f.face_id != bf.face_id
  AND f.embedding <=> bf.embedding < 0.3
ORDER BY bf.face_id, distance;
```

### 8.3 Export Embeddings for ML
```sql
-- Export embeddings with metadata for external ML tools
COPY (
    SELECT face_id, gender, age_estimate, brightness,
           embedding::text as embedding_vector
    FROM faces
    WHERE embedding IS NOT NULL
    ORDER BY created_at
) TO '/tmp/embeddings_export.csv' WITH CSV HEADER;
```

---

## 9. Quality Control Queries

### 9.1 Find Potential Duplicates
```sql
-- Find very similar faces (potential duplicates)
SELECT f1.face_id as face1,
       f2.face_id as face2,
       f1.file_path as path1,
       f2.file_path as path2,
       f1.embedding <=> f2.embedding AS distance
FROM faces f1
JOIN faces f2 ON f1.face_id < f2.face_id
WHERE f1.embedding IS NOT NULL
  AND f2.embedding IS NOT NULL
  AND f1.embedding <=> f2.embedding < 0.01  -- Very similar
ORDER BY distance
LIMIT 50;
```

### 9.2 Validate Embedding Dimensions
```sql
-- Check for inconsistent embedding dimensions
SELECT embedding_model,
       array_length(embedding, 1) as dimension,
       COUNT(*) as count
FROM faces
WHERE embedding IS NOT NULL
GROUP BY embedding_model, array_length(embedding, 1)
ORDER BY embedding_model, dimension;
```

### 9.3 Find Null or Zero Embeddings
```sql
-- Find problematic embeddings
SELECT face_id, embedding_model, created_at
FROM faces
WHERE embedding IS NULL
   OR embedding::text = '[0,0,0,...]'  -- Check for all zeros
ORDER BY created_at DESC
LIMIT 50;
```

### 9.4 Embedding Quality Score
```sql
-- Calculate "quality" based on variance
WITH embedding_stats AS (
    SELECT face_id,
           (SELECT STDDEV(val) FROM UNNEST(embedding) val) as embedding_stddev,
           (SELECT AVG(val) FROM UNNEST(embedding) val) as embedding_mean
    FROM faces
    WHERE embedding IS NOT NULL
)
SELECT face_id, embedding_stddev, embedding_mean,
       CASE
           WHEN embedding_stddev < 0.01 THEN 'LOW'
           WHEN embedding_stddev < 0.1 THEN 'MEDIUM'
           ELSE 'HIGH'
       END as quality
FROM embedding_stats
ORDER BY embedding_stddev
LIMIT 50;
```

---

## 10. Real-World Use Cases

### 10.1 Face Recognition/Verification
```sql
-- Check if query face matches any known face (threshold-based)
WITH query_embedding AS (
    SELECT embedding FROM faces WHERE face_id = 'unknown_face_id'
)
SELECT face_id, gender, brightness,
       embedding <=> (SELECT embedding FROM query_embedding) AS distance,
       CASE
           WHEN embedding <=> (SELECT embedding FROM query_embedding) < 0.3 THEN 'MATCH'
           WHEN embedding <=> (SELECT embedding FROM query_embedding) < 0.5 THEN 'POSSIBLE'
           ELSE 'NO MATCH'
       END as match_status
FROM faces
WHERE embedding IS NOT NULL
  AND face_id != 'unknown_face_id'
ORDER BY distance
LIMIT 5;
```

### 10.2 Find Diverse Sample
```sql
-- Select diverse faces (maximize distance between selections)
WITH RECURSIVE diverse_faces AS (
    -- Start with a random face
    SELECT face_id, embedding, 1 as rank
    FROM faces
    WHERE embedding IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 1

    UNION ALL

    -- Add faces that are far from already selected ones
    SELECT f.face_id, f.embedding, df.rank + 1
    FROM faces f
    CROSS JOIN diverse_faces df
    WHERE f.embedding IS NOT NULL
      AND f.face_id NOT IN (SELECT face_id FROM diverse_faces)
      AND df.rank = (SELECT MAX(rank) FROM diverse_faces)
    ORDER BY f.embedding <=> df.embedding DESC
    LIMIT 1
)
SELECT face_id, rank
FROM diverse_faces
LIMIT 10;
```

### 10.3 Demographic-Aware Search
```sql
-- Find similar faces with demographic diversity
WITH query_embedding AS (
    SELECT embedding FROM faces WHERE face_id = 'query_face_id'
),
ranked_results AS (
    SELECT face_id, gender, age_estimate, brightness,
           metadata->>'skin_tone' as skin_tone,
           embedding <=> (SELECT embedding FROM query_embedding) AS distance,
           ROW_NUMBER() OVER (
               PARTITION BY gender, metadata->>'skin_tone'
               ORDER BY embedding <=> (SELECT embedding FROM query_embedding)
           ) as rn
    FROM faces
    WHERE embedding IS NOT NULL
      AND face_id != 'query_face_id'
)
SELECT face_id, gender, skin_tone, distance
FROM ranked_results
WHERE rn <= 2  -- Get top 2 from each demographic group
ORDER BY distance
LIMIT 20;
```

### 10.4 Time-Based Similarity Trends
```sql
-- Analyze how similar newly added faces are to existing dataset
WITH date_groups AS (
    SELECT face_id,
           embedding,
           DATE_TRUNC('day', created_at) as day
    FROM faces
    WHERE embedding IS NOT NULL
      AND created_at > NOW() - INTERVAL '7 days'
),
cross_day_similarity AS (
    SELECT dg1.day as day1,
           dg2.day as day2,
           AVG(dg1.embedding <=> dg2.embedding) as avg_distance
    FROM date_groups dg1
    CROSS JOIN date_groups dg2
    WHERE dg1.day != dg2.day
    GROUP BY dg1.day, dg2.day
)
SELECT day1, day2, avg_distance
FROM cross_day_similarity
ORDER BY day1, day2;
```

### 10.5 Recommendation System
```sql
-- Recommend faces based on user's liked faces
WITH user_likes AS (
    -- Faces user has liked (replace with your user_id)
    SELECT embedding
    FROM faces
    WHERE face_id IN ('liked_face_1', 'liked_face_2', 'liked_face_3')
      AND embedding IS NOT NULL
),
average_preference AS (
    SELECT AVG(embedding) as pref_embedding
    FROM user_likes
)
SELECT f.face_id, f.gender, f.brightness,
       f.embedding <=> (SELECT pref_embedding FROM average_preference) AS relevance_score
FROM faces f
WHERE f.embedding IS NOT NULL
  AND f.face_id NOT IN ('liked_face_1', 'liked_face_2', 'liked_face_3')
ORDER BY relevance_score
LIMIT 20;
```

---

## 📊 Quick Reference

### Distance Operators
- `<=>` Cosine distance (0-2, lower = more similar)
- `<->` L2/Euclidean distance (0-∞, lower = more similar)
- `<#>` Inner product (-∞ to ∞, higher = more similar)

### Common Thresholds (Cosine Distance)
- `< 0.1` - Very similar (potential duplicates)
- `< 0.3` - Similar (same person or very similar features)
- `< 0.5` - Somewhat similar (similar demographic)
- `> 0.5` - Different

### Index Types
- **HNSW** - Best for most cases (accurate + fast)
- **IVFFlat** - Good for very large datasets

### Performance Tips
1. Always create indexes on embedding column
2. Use LIMIT to restrict results
3. Add WHERE filters before ORDER BY
4. Use CTEs for complex queries
5. Run VACUUM ANALYZE regularly

---

## 🎓 Practice Queries

Try these on your database:

1. Find 10 most similar faces to a random face
2. Find outliers (unique faces)
3. Calculate average distance by gender
4. Find potential duplicate faces
5. Search with 3 different distance metrics

All examples in this guide are ready to use with your database! 🚀
