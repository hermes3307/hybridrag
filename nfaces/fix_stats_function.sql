-- Fix the get_database_stats() function to work with the current schema
-- The current schema has separate embedding columns for each model
-- instead of a single 'embedding' column

DROP FUNCTION IF EXISTS get_database_stats();

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
        -- Count faces that have at least one embedding (any of the model-specific columns)
        COUNT(CASE
            WHEN embedding_statistical IS NOT NULL
                OR embedding_facenet IS NOT NULL
                OR embedding_arcface IS NOT NULL
                OR embedding_vggface2 IS NOT NULL
                OR embedding_insightface IS NOT NULL
            THEN 1
        END)::BIGINT as faces_with_embeddings,
        -- Get list of models from models_processed array
        ARRAY_AGG(DISTINCT unnested_model) FILTER (WHERE unnested_model IS NOT NULL) as embedding_models,
        MIN(timestamp) as oldest_face,
        MAX(timestamp) as newest_face,
        pg_size_pretty(pg_total_relation_size('faces')) as database_size
    FROM faces
    LEFT JOIN LATERAL unnest(models_processed) AS unnested_model ON true;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION get_database_stats() TO PUBLIC;
