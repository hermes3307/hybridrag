-- Migration script to add single embedding column to faces table
-- This adds the standard 'embedding' column used by the current code

-- Add the embedding column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'faces' AND column_name = 'embedding'
    ) THEN
        ALTER TABLE faces ADD COLUMN embedding vector(512);
        RAISE NOTICE 'Added embedding column to faces table';

        -- Copy existing embeddings from the model-specific columns
        -- Prioritize facenet > statistical > others
        UPDATE faces SET embedding = COALESCE(
            embedding_facenet,
            embedding_statistical,
            embedding_arcface,
            embedding_vggface2,
            embedding_insightface
        );

        RAISE NOTICE 'Copied existing embeddings to new embedding column';
    ELSE
        RAISE NOTICE 'embedding column already exists';
    END IF;
END $$;

-- Create HNSW index for the embedding column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE indexname = 'idx_embedding_hnsw_cosine'
    ) THEN
        CREATE INDEX idx_embedding_hnsw_cosine ON faces
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        RAISE NOTICE 'Created HNSW index on embedding column';
    ELSE
        RAISE NOTICE 'Index idx_embedding_hnsw_cosine already exists';
    END IF;
END $$;

-- Display stats
SELECT
    COUNT(*) as total_faces,
    COUNT(embedding) as faces_with_embedding,
    COUNT(embedding_statistical) as faces_with_statistical,
    COUNT(embedding_facenet) as faces_with_facenet
FROM faces;
