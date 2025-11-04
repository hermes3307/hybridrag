#!/bin/bash
# Simple embedding status checker

echo "=================================="
echo "   Embedding Status"
echo "=================================="
echo ""

PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c "
SELECT
    COUNT(*) as total_faces,
    COUNT(embedding_facenet) as facenet,
    COUNT(embedding_arcface) as arcface,
    COUNT(embedding_vggface2) as vggface2,
    COUNT(embedding_insightface) as insightface,
    COUNT(embedding_statistical) as statistical
FROM faces;
"

echo ""
echo "To run embedding: ./run_embedding.sh"
