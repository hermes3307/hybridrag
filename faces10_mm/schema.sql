-- PostgreSQL + pgvector Schema for Face Recognition System
-- This schema defines tables and indexes for storing face embeddings

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing table if it exists (for clean setup)
DROP TABLE IF EXISTS faces CASCADE;

-- Main faces table with multiple vector embeddings
-- Multi-Model Support: Store embeddings from different models separately
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(255) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    image_hash VARCHAR(64) NOT NULL,

    -- Multiple embedding columns for different models
    -- Each model can have its own embedding stored independently
    -- 512 dimensions is standard for most deep learning models
    embedding_facenet vector(512),      -- FaceNet (InceptionResnetV1)
    embedding_arcface vector(512),      -- ArcFace
    embedding_vggface2 vector(512),     -- VGGFace2
    embedding_insightface vector(512),  -- InsightFace
    embedding_statistical vector(512),  -- Statistical model (padded to 512)

    -- Track which models have been processed
    models_processed TEXT[],  -- Array of model names that have embeddings

    -- Metadata columns (extracted from features)
    age_estimate INTEGER,
    gender VARCHAR(20),
    brightness FLOAT,
    contrast FLOAT,
    sharpness FLOAT,

    -- Additional metadata stored as JSONB for flexibility
    -- This allows storing any additional features without schema changes
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for fast retrieval
CREATE INDEX idx_face_id ON faces(face_id);
CREATE INDEX idx_image_hash ON faces(image_hash);
CREATE INDEX idx_timestamp ON faces(timestamp DESC);
CREATE INDEX idx_created_at ON faces(created_at DESC);
CREATE INDEX idx_models_processed ON faces USING GIN(models_processed);

-- JSONB index for flexible metadata queries
CREATE INDEX idx_metadata_gin ON faces USING GIN(metadata jsonb_path_ops);

-- Vector similarity indexes using HNSW for each model
-- This provides fast approximate nearest neighbor search
-- Using cosine distance (recommended for face embeddings)
-- m=16: number of connections per layer (higher = better recall, more memory)
-- ef_construction=64: size of dynamic candidate list (higher = better index quality, slower build)

-- FaceNet embedding index
CREATE INDEX idx_embedding_facenet_hnsw ON faces
USING hnsw (embedding_facenet vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ArcFace embedding index
CREATE INDEX idx_embedding_arcface_hnsw ON faces
USING hnsw (embedding_arcface vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- VGGFace2 embedding index
CREATE INDEX idx_embedding_vggface2_hnsw ON faces
USING hnsw (embedding_vggface2 vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- InsightFace embedding index
CREATE INDEX idx_embedding_insightface_hnsw ON faces
USING hnsw (embedding_insightface vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Statistical model embedding index
CREATE INDEX idx_embedding_statistical_hnsw ON faces
USING hnsw (embedding_statistical vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_faces_updated_at
BEFORE UPDATE ON faces
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Create a view for easy querying with all metadata
CREATE OR REPLACE VIEW faces_with_metadata AS
SELECT
    id,
    face_id,
    file_path,
    timestamp,
    image_hash,
    models_processed,
    age_estimate,
    gender,
    brightness,
    contrast,
    sharpness,
    metadata,
    created_at,
    updated_at,
    -- Don't include embeddings in view (too large)
    -- Check models_processed array to see which embeddings are available
    NULL::vector as embedding_facenet,
    NULL::vector as embedding_arcface,
    NULL::vector as embedding_vggface2,
    NULL::vector as embedding_insightface,
    NULL::vector as embedding_statistical
FROM faces;

-- Helpful query functions

-- Function to search for similar faces using a specific model
CREATE OR REPLACE FUNCTION search_similar_faces(
    query_embedding vector(512),
    model_name TEXT DEFAULT 'facenet',
    limit_count INTEGER DEFAULT 10,
    distance_threshold FLOAT DEFAULT 1.0
)
RETURNS TABLE (
    face_id VARCHAR(255),
    file_path TEXT,
    distance FLOAT,
    metadata JSONB,
    model_used TEXT
) AS $$
BEGIN
    RETURN QUERY EXECUTE format('
        SELECT
            f.face_id,
            f.file_path,
            f.embedding_%s <=> $1 AS distance,
            f.metadata,
            $2 AS model_used
        FROM faces f
        WHERE f.embedding_%s IS NOT NULL
            AND f.embedding_%s <=> $1 < $3
        ORDER BY f.embedding_%s <=> $1
        LIMIT $4',
        model_name, model_name, model_name, model_name)
    USING query_embedding, model_name, distance_threshold, limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to search across all models and return combined results
CREATE OR REPLACE FUNCTION search_similar_faces_all_models(
    query_embedding vector(512),
    limit_count INTEGER DEFAULT 10,
    distance_threshold FLOAT DEFAULT 1.0
)
RETURNS TABLE (
    face_id VARCHAR(255),
    file_path TEXT,
    distance FLOAT,
    metadata JSONB,
    model_used TEXT
) AS $$
BEGIN
    RETURN QUERY
    -- FaceNet results
    SELECT
        f.face_id,
        f.file_path,
        f.embedding_facenet <=> query_embedding AS distance,
        f.metadata,
        'facenet'::TEXT AS model_used
    FROM faces f
    WHERE f.embedding_facenet IS NOT NULL
        AND f.embedding_facenet <=> query_embedding < distance_threshold

    UNION ALL

    -- ArcFace results
    SELECT
        f.face_id,
        f.file_path,
        f.embedding_arcface <=> query_embedding AS distance,
        f.metadata,
        'arcface'::TEXT AS model_used
    FROM faces f
    WHERE f.embedding_arcface IS NOT NULL
        AND f.embedding_arcface <=> query_embedding < distance_threshold

    UNION ALL

    -- VGGFace2 results
    SELECT
        f.face_id,
        f.file_path,
        f.embedding_vggface2 <=> query_embedding AS distance,
        f.metadata,
        'vggface2'::TEXT AS model_used
    FROM faces f
    WHERE f.embedding_vggface2 IS NOT NULL
        AND f.embedding_vggface2 <=> query_embedding < distance_threshold

    UNION ALL

    -- InsightFace results
    SELECT
        f.face_id,
        f.file_path,
        f.embedding_insightface <=> query_embedding AS distance,
        f.metadata,
        'insightface'::TEXT AS model_used
    FROM faces f
    WHERE f.embedding_insightface IS NOT NULL
        AND f.embedding_insightface <=> query_embedding < distance_threshold

    ORDER BY distance
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get database statistics
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE (
    total_faces BIGINT,
    faces_with_facenet BIGINT,
    faces_with_arcface BIGINT,
    faces_with_vggface2 BIGINT,
    faces_with_insightface BIGINT,
    faces_with_statistical BIGINT,
    oldest_face TIMESTAMP,
    newest_face TIMESTAMP,
    database_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_faces,
        COUNT(embedding_facenet)::BIGINT as faces_with_facenet,
        COUNT(embedding_arcface)::BIGINT as faces_with_arcface,
        COUNT(embedding_vggface2)::BIGINT as faces_with_vggface2,
        COUNT(embedding_insightface)::BIGINT as faces_with_insightface,
        COUNT(embedding_statistical)::BIGINT as faces_with_statistical,
        MIN(timestamp) as oldest_face,
        MAX(timestamp) as newest_face,
        pg_size_pretty(pg_total_relation_size('faces')) as database_size
    FROM faces;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON TABLE faces TO your_user;
-- GRANT USAGE, SELECT ON SEQUENCE faces_id_seq TO your_user;

COMMENT ON TABLE faces IS 'Stores multiple face embeddings from different models for face recognition system';
COMMENT ON COLUMN faces.embedding_facenet IS 'FaceNet embedding vector (512 dimensions)';
COMMENT ON COLUMN faces.embedding_arcface IS 'ArcFace embedding vector (512 dimensions)';
COMMENT ON COLUMN faces.embedding_vggface2 IS 'VGGFace2 embedding vector (512 dimensions)';
COMMENT ON COLUMN faces.embedding_insightface IS 'InsightFace embedding vector (512 dimensions)';
COMMENT ON COLUMN faces.embedding_statistical IS 'Statistical model embedding vector (512 dimensions)';
COMMENT ON COLUMN faces.models_processed IS 'Array of model names that have generated embeddings for this face';
COMMENT ON COLUMN faces.metadata IS 'Additional flexible metadata stored as JSON';
COMMENT ON INDEX idx_embedding_facenet_hnsw IS 'HNSW index for fast FaceNet cosine similarity search';
COMMENT ON INDEX idx_embedding_arcface_hnsw IS 'HNSW index for fast ArcFace cosine similarity search';
COMMENT ON INDEX idx_embedding_vggface2_hnsw IS 'HNSW index for fast VGGFace2 cosine similarity search';
COMMENT ON INDEX idx_embedding_insightface_hnsw IS 'HNSW index for fast InsightFace cosine similarity search';
COMMENT ON INDEX idx_embedding_statistical_hnsw IS 'HNSW index for fast Statistical model cosine similarity search';
