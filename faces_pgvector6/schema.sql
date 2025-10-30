-- PostgreSQL + pgvector Schema for Face Recognition System
-- This schema defines tables and indexes for storing face embeddings

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing table if it exists (for clean setup)
DROP TABLE IF EXISTS faces CASCADE;

-- Main faces table with vector embeddings
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(255) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    image_hash VARCHAR(64) NOT NULL,
    embedding_model VARCHAR(50) NOT NULL,

    -- Vector embedding (dimension depends on model)
    -- Statistical model: 7 dimensions
    -- FaceNet: 512 dimensions
    -- ArcFace: 512 dimensions
    -- We'll use 512 as default (statistical will pad with zeros)
    embedding vector(512),

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
CREATE INDEX idx_embedding_model ON faces(embedding_model);
CREATE INDEX idx_created_at ON faces(created_at DESC);

-- JSONB index for flexible metadata queries
CREATE INDEX idx_metadata_gin ON faces USING GIN(metadata jsonb_path_ops);

-- Vector similarity index using HNSW (Hierarchical Navigable Small World)
-- This provides fast approximate nearest neighbor search
-- Using cosine distance (recommended for face embeddings)
-- m=16: number of connections per layer (higher = better recall, more memory)
-- ef_construction=64: size of dynamic candidate list (higher = better index quality, slower build)
CREATE INDEX idx_embedding_hnsw_cosine ON faces
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (less memory, slightly slower queries)
-- Uncomment if you prefer IVFFlat over HNSW
-- CREATE INDEX idx_embedding_ivfflat ON faces
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- L2 distance index (alternative distance metric)
-- Uncomment if you want to use L2 distance instead of cosine
-- CREATE INDEX idx_embedding_hnsw_l2 ON faces
-- USING hnsw (embedding vector_l2_ops)
-- WITH (m = 16, ef_construction = 64);

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
    embedding_model,
    age_estimate,
    gender,
    brightness,
    contrast,
    sharpness,
    metadata,
    created_at,
    updated_at,
    -- Don't include embedding in view (too large)
    NULL::vector as embedding
FROM faces;

-- Helpful query functions

-- Function to search for similar faces with cosine distance
CREATE OR REPLACE FUNCTION search_similar_faces(
    query_embedding vector(512),
    limit_count INTEGER DEFAULT 10,
    distance_threshold FLOAT DEFAULT 1.0
)
RETURNS TABLE (
    face_id VARCHAR(255),
    file_path TEXT,
    distance FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.face_id,
        f.file_path,
        f.embedding <=> query_embedding AS distance,
        f.metadata
    FROM faces f
    WHERE f.embedding IS NOT NULL
        AND f.embedding <=> query_embedding < distance_threshold
    ORDER BY f.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get database statistics
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE (
    total_faces BIGINT,
    faces_with_embeddings BIGINT,
    embedding_models TEXT[],
    oldest_face TIMESTAMP,
    newest_face TIMESTAMP,
    database_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_faces,
        COUNT(embedding)::BIGINT as faces_with_embeddings,
        ARRAY_AGG(DISTINCT embedding_model) as embedding_models,
        MIN(timestamp) as oldest_face,
        MAX(timestamp) as newest_face,
        pg_size_pretty(pg_total_relation_size('faces')) as database_size
    FROM faces;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON TABLE faces TO your_user;
-- GRANT USAGE, SELECT ON SEQUENCE faces_id_seq TO your_user;

COMMENT ON TABLE faces IS 'Stores face embeddings and metadata for face recognition system';
COMMENT ON COLUMN faces.embedding IS 'Face embedding vector (512 dimensions)';
COMMENT ON COLUMN faces.metadata IS 'Additional flexible metadata stored as JSON';
COMMENT ON INDEX idx_embedding_hnsw_cosine IS 'HNSW index for fast cosine similarity search';
