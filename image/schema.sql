-- PostgreSQL + pgvector Schema for Image Recognition System
-- This schema defines tables and indexes for storing image embeddings

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing table if it exists (for clean setup)
DROP TABLE IF EXISTS images CASCADE;

-- Main images table with vector embeddings
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    image_hash VARCHAR(64) NOT NULL,
    embedding_model VARCHAR(50) NOT NULL,

    -- Vector embedding (dimension depends on model)
    -- CLIP: 512 dimensions
    -- YOLO: 80 dimensions
    embedding vector(512),

    -- Metadata columns (extracted from features)
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
CREATE INDEX idx_image_id ON images(image_id);
CREATE INDEX idx_image_hash ON images(image_hash);
CREATE INDEX idx_timestamp ON images(timestamp DESC);
CREATE INDEX idx_embedding_model ON images(embedding_model);
CREATE INDEX idx_created_at ON images(created_at DESC);

-- JSONB index for flexible metadata queries
CREATE INDEX idx_metadata_gin ON images USING GIN(metadata jsonb_path_ops);

-- Vector similarity index using HNSW (Hierarchical Navigable Small World)
-- This provides fast approximate nearest neighbor search
-- Using cosine distance (recommended for image embeddings)
-- m=16: number of connections per layer (higher = better recall, more memory)
-- ef_construction=64: size of dynamic candidate list (higher = better index quality, slower build)
CREATE INDEX idx_embedding_hnsw_cosine ON images
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (less memory, slightly slower queries)
-- Uncomment if you prefer IVFFlat over HNSW
-- CREATE INDEX idx_embedding_ivfflat ON images
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- L2 distance index (alternative distance metric)
-- Uncomment if you want to use L2 distance instead of cosine
-- CREATE INDEX idx_embedding_hnsw_l2 ON images
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
CREATE TRIGGER update_images_updated_at
BEFORE UPDATE ON images
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Create a view for easy querying with all metadata
CREATE OR REPLACE VIEW images_with_metadata AS
SELECT
    id,
    image_id,
    file_path,
    timestamp,
    image_hash,
    embedding_model,
    brightness,
    contrast,
    sharpness,
    metadata,
    created_at,
    updated_at,
    -- Don't include embedding in view (too large)
    NULL::vector as embedding
FROM images;

-- Helpful query functions

-- Function to search for similar images with cosine distance
CREATE OR REPLACE FUNCTION search_similar_images(
    query_embedding vector(512),
    limit_count INTEGER DEFAULT 10,
    distance_threshold FLOAT DEFAULT 1.0
)
RETURNS TABLE (
    image_id VARCHAR(255),
    file_path TEXT,
    distance FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.image_id,
        i.file_path,
        i.embedding <=> query_embedding AS distance,
        i.metadata
    FROM images i
    WHERE i.embedding IS NOT NULL
        AND i.embedding <=> query_embedding < distance_threshold
    ORDER BY i.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get database statistics
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE (
    total_images BIGINT,
    images_with_embeddings BIGINT,
    embedding_models TEXT[],
    oldest_image TIMESTAMP,
    newest_image TIMESTAMP,
    database_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_images,
        COUNT(embedding)::BIGINT as images_with_embeddings,
        ARRAY_AGG(DISTINCT embedding_model::TEXT) as embedding_models,
        MIN(timestamp) as oldest_image,
        MAX(timestamp) as newest_image,
        pg_size_pretty(pg_total_relation_size('images')) as database_size
    FROM images;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON TABLE images TO your_user;
-- GRANT USAGE, SELECT ON SEQUENCE images_id_seq TO your_user;

COMMENT ON TABLE images IS 'Stores image embeddings and metadata for image recognition system';
COMMENT ON COLUMN images.embedding IS 'Image embedding vector (512 dimensions)';
COMMENT ON COLUMN images.metadata IS 'Additional flexible metadata stored as JSON';
COMMENT ON INDEX idx_embedding_hnsw_cosine IS 'HNSW index for fast cosine similarity search';
