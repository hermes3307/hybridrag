-- Migration script to add embedding_model column to existing faces table
-- Run this if you get error: column "embedding_model" does not exist

-- Add the embedding_model column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'faces' AND column_name = 'embedding_model'
    ) THEN
        ALTER TABLE faces ADD COLUMN embedding_model VARCHAR(50) DEFAULT 'statistical' NOT NULL;
        RAISE NOTICE 'Added embedding_model column to faces table';
    ELSE
        RAISE NOTICE 'embedding_model column already exists';
    END IF;
END $$;

-- Create index if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE indexname = 'idx_embedding_model'
    ) THEN
        CREATE INDEX idx_embedding_model ON faces(embedding_model);
        RAISE NOTICE 'Created index on embedding_model column';
    ELSE
        RAISE NOTICE 'Index idx_embedding_model already exists';
    END IF;
END $$;

-- Display confirmation
SELECT
    COUNT(*) as total_faces,
    COUNT(DISTINCT embedding_model) as distinct_models,
    ARRAY_AGG(DISTINCT embedding_model) as models
FROM faces;
