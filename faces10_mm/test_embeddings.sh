#!/bin/bash
# Test if embeddings are valid (not all zeros)

echo "=================================="
echo "   Testing Embedding Quality"
echo "=================================="
echo ""

# Test FaceNet
echo "Testing FaceNet embeddings..."
FACENET_ZEROS=$(PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -t -A -c "
SELECT COUNT(*)
FROM faces
WHERE embedding_facenet IS NOT NULL
  AND embedding_facenet::text = array_fill(0::float, ARRAY[512])::vector::text;
")
FACENET_VALID=$(PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -t -A -c "
SELECT COUNT(*)
FROM faces
WHERE embedding_facenet IS NOT NULL
  AND embedding_facenet::text != array_fill(0::float, ARRAY[512])::vector::text;
")
echo "  Valid: $FACENET_VALID"
echo "  Zero embeddings (bad): $FACENET_ZEROS"

# Test ArcFace
echo ""
echo "Testing ArcFace embeddings..."
ARCFACE_ZEROS=$(PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -t -A -c "
SELECT COUNT(*)
FROM faces
WHERE embedding_arcface IS NOT NULL
  AND embedding_arcface::text = array_fill(0::float, ARRAY[512])::vector::text;
")
ARCFACE_VALID=$(PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -t -A -c "
SELECT COUNT(*)
FROM faces
WHERE embedding_arcface IS NOT NULL
  AND embedding_arcface::text != array_fill(0::float, ARRAY[512])::vector::text;
")
echo "  Valid: $ARCFACE_VALID"
echo "  Zero embeddings (bad): $ARCFACE_ZEROS"

# Test VGGFace2
echo ""
echo "Testing VGGFace2 embeddings..."
VGGFACE_ZEROS=$(PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -t -A -c "
SELECT COUNT(*)
FROM faces
WHERE embedding_vggface2 IS NOT NULL
  AND embedding_vggface2::text = array_fill(0::float, ARRAY[512])::vector::text;
")
VGGFACE_VALID=$(PGPASSWORD=postgres psql -h localhost -U postgres -d vector_db -t -A -c "
SELECT COUNT(*)
FROM faces
WHERE embedding_vggface2 IS NOT NULL
  AND embedding_vggface2::text != array_fill(0::float, ARRAY[512])::vector::text;
")
echo "  Valid: $VGGFACE_VALID"
echo "  Zero embeddings (bad): $VGGFACE_ZEROS"

echo ""
echo "=================================="
echo "Zero embeddings = face detection failed"
echo "Valid embeddings = ready for search"
echo "=================================="
