#!/bin/bash
# Test search functionality for each model

MODEL="${1:-facenet}"

echo "=================================="
echo "   Testing Search: $MODEL"
echo "=================================="
echo ""

# Map model name to column
case "$MODEL" in
    facenet)
        COLUMN="embedding_facenet"
        ;;
    arcface)
        COLUMN="embedding_arcface"
        ;;
    vggface2)
        COLUMN="embedding_vggface2"
        ;;
    insightface)
        COLUMN="embedding_insightface"
        ;;
    statistical)
        COLUMN="embedding_statistical"
        ;;
    *)
        echo "Invalid model: $MODEL"
        echo "Usage: $0 [facenet|arcface|vggface2|insightface|statistical]"
        exit 1
        ;;
esac

# Get a random face with this model's embedding
echo "1. Getting a random face with $MODEL embedding..."
RANDOM_FACE=$(PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -t -A -F'|' -c "
SELECT face_id, file_path
FROM faces
WHERE $COLUMN IS NOT NULL
  AND $COLUMN::text != array_fill(0::float, ARRAY[512])::vector::text
ORDER BY RANDOM()
LIMIT 1;
")

if [ -z "$RANDOM_FACE" ]; then
    echo "❌ No faces found with $MODEL embedding!"
    exit 1
fi

IFS='|' read -r FACE_ID FILE_PATH <<< "$RANDOM_FACE"
echo "   ✅ Face ID: $FACE_ID"
echo "   📁 File: $FILE_PATH"
echo ""

# Test similarity search
echo "2. Finding similar faces using $MODEL..."
PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -c "
SELECT
    face_id,
    file_path,
    $COLUMN <=> (SELECT $COLUMN FROM faces WHERE face_id = '$FACE_ID') AS distance
FROM faces
WHERE $COLUMN IS NOT NULL
  AND face_id != '$FACE_ID'
ORDER BY distance
LIMIT 5;
"

echo ""
echo "✅ Search test complete!"
echo ""
echo "Usage examples:"
echo "  $0 facenet"
echo "  $0 arcface"
echo "  $0 vggface2"
