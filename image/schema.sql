-- PostgreSQL + pgvector Schema for Image Recognition System
-- This schema defines tables and indexes for storing image embeddings

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS image_embeddings CASCADE;
DROP TABLE IF EXISTS images CASCADE;

-- Main images table
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    image_hash VARCHAR(64) NOT NULL,

    -- Metadata columns (extracted from features)
    brightness FLOAT,
    contrast FLOAT,
    sharpness FLOAT,

    -- Additional metadata stored as JSONB for flexibility
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Separate table for embeddings (supports multiple embeddings per image)
CREATE TABLE image_embeddings (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255) NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    embedding_model VARCHAR(50) NOT NULL,
    embedding_dimension INTEGER NOT NULL,

    -- Three separate embedding columns for different dimensions
    embedding_512 vector(512),  -- For CLIP, ResNet, Statistical
    embedding_512_alt vector(512),  -- For alternative 512-dim models
    embedding_1024 vector(1024),  -- For larger models like DINOv2

    created_at TIMESTAMP DEFAULT NOW(),

    -- Ensure one embedding per model per image
    UNIQUE(image_id, embedding_model)
);

-- Indexes for fast retrieval on images table
CREATE INDEX idx_image_id ON images(image_id);
CREATE INDEX idx_image_hash ON images(image_hash);
CREATE INDEX idx_timestamp ON images(timestamp DESC);
CREATE INDEX idx_created_at ON images(created_at DESC);

-- JSONB index for flexible metadata queries
CREATE INDEX idx_metadata_gin ON images USING GIN(metadata jsonb_path_ops);

-- Indexes for embeddings table
CREATE INDEX idx_embeddings_image_id ON image_embeddings(image_id);
CREATE INDEX idx_embeddings_model ON image_embeddings(embedding_model);

-- Vector similarity indexes using HNSW for each embedding column
-- 512-dimension embeddings (CLIP, ResNet, Statistical)
CREATE INDEX idx_embedding_512_hnsw_cosine ON image_embeddings
USING hnsw (embedding_512 vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative 512-dimension embeddings
CREATE INDEX idx_embedding_512_alt_hnsw_cosine ON image_embeddings
USING hnsw (embedding_512_alt vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 1024-dimension embeddings (DINOv2, etc.)
CREATE INDEX idx_embedding_1024_hnsw_cosine ON image_embeddings
USING hnsw (embedding_1024 vector_cosine_ops)
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
CREATE TRIGGER update_images_updated_at
BEFORE UPDATE ON images
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Create a view for easy querying with all metadata and embeddings
CREATE OR REPLACE VIEW images_with_embeddings AS
SELECT
    i.id,
    i.image_id,
    i.file_path,
    i.timestamp,
    i.image_hash,
    i.brightness,
    i.contrast,
    i.sharpness,
    i.metadata,
    i.created_at,
    i.updated_at,
    ARRAY_AGG(DISTINCT e.embedding_model) as embedding_models,
    COUNT(DISTINCT e.id) as embedding_count
FROM images i
LEFT JOIN image_embeddings e ON i.image_id = e.image_id
GROUP BY i.id, i.image_id, i.file_path, i.timestamp, i.image_hash,
         i.brightness, i.contrast, i.sharpness, i.metadata, i.created_at, i.updated_at;

-- Helpful query functions

-- Function to search for similar images using a specific embedding model
CREATE OR REPLACE FUNCTION search_similar_images(
    query_embedding vector(512),
    model_name VARCHAR(50),
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
        e.embedding_512 <=> query_embedding AS distance,
        i.metadata
    FROM images i
    JOIN image_embeddings e ON i.image_id = e.image_id
    WHERE e.embedding_model = model_name
        AND e.embedding_512 IS NOT NULL
        AND e.embedding_512 <=> query_embedding < distance_threshold
    ORDER BY e.embedding_512 <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function for multi-embedding fusion search (combines CLIP, YOLO, ResNet)
CREATE OR REPLACE FUNCTION search_multi_embedding(
    clip_embedding vector(512),
    yolo_embedding vector(512),
    resnet_embedding vector(512),
    clip_weight FLOAT DEFAULT 0.5,
    yolo_weight FLOAT DEFAULT 0.25,
    resnet_weight FLOAT DEFAULT 0.25,
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    image_id VARCHAR(255),
    file_path TEXT,
    combined_distance FLOAT,
    clip_distance FLOAT,
    yolo_distance FLOAT,
    resnet_distance FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.image_id,
        i.file_path,
        (COALESCE(clip_e.embedding_512 <=> clip_embedding, 1.0) * clip_weight +
         COALESCE(yolo_e.embedding_512 <=> yolo_embedding, 1.0) * yolo_weight +
         COALESCE(resnet_e.embedding_512 <=> resnet_embedding, 1.0) * resnet_weight) AS combined_distance,
        clip_e.embedding_512 <=> clip_embedding AS clip_distance,
        yolo_e.embedding_512 <=> yolo_embedding AS yolo_distance,
        resnet_e.embedding_512 <=> resnet_embedding AS resnet_distance,
        i.metadata
    FROM images i
    LEFT JOIN image_embeddings clip_e ON i.image_id = clip_e.image_id AND clip_e.embedding_model = 'clip'
    LEFT JOIN image_embeddings yolo_e ON i.image_id = yolo_e.image_id AND yolo_e.embedding_model = 'yolo'
    LEFT JOIN image_embeddings resnet_e ON i.image_id = resnet_e.image_id AND resnet_e.embedding_model = 'resnet'
    WHERE (clip_e.embedding_512 IS NOT NULL OR yolo_e.embedding_512 IS NOT NULL OR resnet_e.embedding_512 IS NOT NULL)
    ORDER BY combined_distance
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get database statistics
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE (
    total_images BIGINT,
    total_embeddings BIGINT,
    embedding_models TEXT[],
    oldest_image TIMESTAMP,
    newest_image TIMESTAMP,
    database_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (SELECT COUNT(*)::BIGINT FROM images) as total_images,
        (SELECT COUNT(*)::BIGINT FROM image_embeddings) as total_embeddings,
        (SELECT ARRAY_AGG(DISTINCT embedding_model::TEXT) FROM image_embeddings) as embedding_models,
        (SELECT MIN(timestamp) FROM images) as oldest_image,
        (SELECT MAX(timestamp) FROM images) as newest_image,
        pg_size_pretty(pg_total_relation_size('images') + pg_total_relation_size('image_embeddings')) as database_size;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON TABLE images TO your_user;
-- GRANT USAGE, SELECT ON SEQUENCE images_id_seq TO your_user;

COMMENT ON TABLE images IS 'Stores image metadata for image recognition system';
COMMENT ON TABLE image_embeddings IS 'Stores multiple embeddings per image using different models';
COMMENT ON COLUMN images.metadata IS 'Additional flexible metadata stored as JSON';
COMMENT ON INDEX idx_embedding_512_hnsw_cosine IS 'HNSW index for fast cosine similarity search on 512-dim embeddings';
