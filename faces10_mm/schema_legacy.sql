-- PostgreSQL + pgvector Schema for Face Recognition System (Legacy/Single-Model Version)
-- This schema defines tables and indexes for storing face embeddings with SINGLE embedding column
-- This is the legacy/backward-compatible version for systems using one embedding model at a time

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing table if it exists (for clean setup)
DROP TABLE IF EXISTS faces_legacy CASCADE;

-- Legacy faces table with single embedding column
-- This maintains compatibility with older single-model implementations
CREATE TABLE faces_legacy (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(255) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    image_hash VARCHAR(64) NOT NULL,

    -- Single embedding column (model-agnostic)
    -- 512 dimensions is standard for most deep learning models
    embedding vector(512),

    -- Track which model was used for this embedding
    embedding_model VARCHAR(50),  -- e.g., 'facenet', 'arcface', 'statistical', etc.

    -- Metadata columns (extracted from features)
    age_estimate INTEGER,
    gender VARCHAR(20),
    brightness FLOAT,
    contrast FLOAT,
    sharpness FLOAT,

    -- Additional metadata stored as JSONB for flexibility
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for fast retrieval
CREATE INDEX idx_face_id_legacy ON faces_legacy(face_id);
CREATE INDEX idx_image_hash_legacy ON faces_legacy(image_hash);
CREATE INDEX idx_timestamp_legacy ON faces_legacy(timestamp DESC);
CREATE INDEX idx_created_at_legacy ON faces_legacy(created_at DESC);
CREATE INDEX idx_embedding_model_legacy ON faces_legacy(embedding_model);

-- JSONB index for flexible metadata queries
CREATE INDEX idx_metadata_gin_legacy ON faces_legacy USING GIN(metadata jsonb_path_ops);

-- Vector similarity index using HNSW
-- This provides fast approximate nearest neighbor search
-- Using cosine distance (recommended for face embeddings)
-- m=16: number of connections per layer (higher = better recall, more memory)
-- ef_construction=64: size of dynamic candidate list (higher = better index quality, slower build)
CREATE INDEX idx_embedding_hnsw_legacy ON faces_legacy
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column_legacy()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_faces_legacy_updated_at
BEFORE UPDATE ON faces_legacy
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column_legacy();

-- Create a view for easy querying with all metadata
CREATE OR REPLACE VIEW faces_legacy_with_metadata AS
SELECT
    id,
    face_id,
    file_path,
    timestamp,
    image_hash,
    embedding_model,
    age_estimate,
    gender,
    brightness,
    contrast,
    sharpness,
    metadata,
    created_at,
    updated_at,
    NULL::vector as embedding  -- Don't include embedding in view (too large)
FROM faces_legacy;

-- Function to search for similar faces (legacy single-embedding version)
CREATE OR REPLACE FUNCTION search_similar_faces_legacy(
    query_embedding vector(512),
    limit_count INTEGER DEFAULT 10,
    distance_threshold FLOAT DEFAULT 1.0,
    model_filter TEXT DEFAULT NULL
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
    SELECT
        f.face_id,
        f.file_path,
        f.embedding <=> query_embedding AS distance,
        f.metadata,
        f.embedding_model AS model_used
    FROM faces_legacy f
    WHERE f.embedding IS NOT NULL
        AND f.embedding <=> query_embedding < distance_threshold
        AND (model_filter IS NULL OR f.embedding_model = model_filter)
    ORDER BY f.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get database statistics (legacy version)
CREATE OR REPLACE FUNCTION get_database_stats_legacy()
RETURNS TABLE (
    total_faces BIGINT,
    faces_per_model JSONB,
    oldest_face TIMESTAMP,
    newest_face TIMESTAMP,
    database_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_faces,
        jsonb_object_agg(
            COALESCE(embedding_model, 'unknown'),
            COUNT(*)
        ) FILTER (WHERE embedding IS NOT NULL) as faces_per_model,
        MIN(timestamp) as oldest_face,
        MAX(timestamp) as newest_face,
        pg_size_pretty(pg_total_relation_size('faces_legacy')) as database_size
    FROM faces_legacy;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON TABLE faces_legacy TO your_user;
-- GRANT USAGE, SELECT ON SEQUENCE faces_legacy_id_seq TO your_user;

COMMENT ON TABLE faces_legacy IS 'Legacy single-embedding storage for face recognition system (backward compatibility)';
COMMENT ON COLUMN faces_legacy.embedding IS 'Face embedding vector (512 dimensions) from single model';
COMMENT ON COLUMN faces_legacy.embedding_model IS 'Name of the model used to generate this embedding';
COMMENT ON COLUMN faces_legacy.metadata IS 'Additional flexible metadata stored as JSON';
COMMENT ON INDEX idx_embedding_hnsw_legacy IS 'HNSW index for fast cosine similarity search (single embedding)';

-- Migration helper function: Copy from multi-model to legacy (single model extraction)
CREATE OR REPLACE FUNCTION migrate_multimodel_to_legacy(
    source_model TEXT DEFAULT 'facenet'
)
RETURNS INTEGER AS $$
DECLARE
    rows_migrated INTEGER;
BEGIN
    -- This function copies data from the multi-model 'faces' table to 'faces_legacy'
    -- extracting embeddings from the specified model column

    EXECUTE format('
        INSERT INTO faces_legacy (
            face_id, file_path, timestamp, image_hash,
            embedding, embedding_model,
            age_estimate, gender, brightness, contrast, sharpness,
            metadata, created_at, updated_at
        )
        SELECT
            face_id, file_path, timestamp, image_hash,
            embedding_%s, %L,
            age_estimate, gender, brightness, contrast, sharpness,
            metadata, created_at, updated_at
        FROM faces
        WHERE embedding_%s IS NOT NULL
        ON CONFLICT (face_id) DO NOTHING',
        source_model, source_model, source_model
    );

    GET DIAGNOSTICS rows_migrated = ROW_COUNT;
    RETURN rows_migrated;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION migrate_multimodel_to_legacy IS 'Migrate data from multi-model faces table to legacy single-embedding table';

