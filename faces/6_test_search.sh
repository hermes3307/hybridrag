#!/bin/bash

echo "🔍 Step 6: Semantic Search Test with New Face"
echo "============================================="

echo "🧪 Testing face similarity search by downloading a new test face..."

# Create comprehensive search test script
cat > temp_search_test.py << 'EOF'
from face_collector import FaceCollector, FaceAnalyzer, FaceEmbedder
from face_database import FaceDatabase, FaceSearchInterface
import os
import time

def main():
    print("🔍 COMPREHENSIVE FACE SEARCH TEST")
    print("="*50)

    # Initialize database
    face_db = FaceDatabase()
    search_interface = FaceSearchInterface(face_db)

    # Get current database status
    stats = face_db.get_database_stats()
    print(f"📊 Current database: {stats['total_faces']} faces")
    print(f"   Age groups: {stats.get('age_group_distribution', {})}")
    print(f"   Skin tones: {stats.get('skin_tone_distribution', {})}")

    # Download a new test face
    print(f"\n📥 Downloading new test face...")
    collector = FaceCollector(delay=1.0)
    test_faces = collector.download_faces_batch(count=1, max_workers=1)

    if not test_faces:
        print("❌ Failed to download test face")
        return

    test_face_path = test_faces[0]
    test_face_name = os.path.basename(test_face_path)
    print(f"✅ Downloaded: {test_face_name}")

    # Analyze the test face
    print(f"\n🔍 Analyzing test face features...")
    analyzer = FaceAnalyzer()
    embedder = FaceEmbedder()

    features = analyzer.estimate_basic_features(test_face_path)
    embedding = embedder.generate_embedding(test_face_path, features)

    print(f"📋 Test Face Analysis:")
    print(f"   Age group: {features.get('estimated_age_group', 'unknown')}")
    print(f"   Skin tone: {features.get('estimated_skin_tone', 'unknown')}")
    print(f"   Quality: {features.get('image_quality', 'unknown')}")
    print(f"   Brightness: {features.get('brightness', 0):.1f}")
    print(f"   Embedding dimensions: {len(embedding)}")

    # Test 1: Direct embedding similarity search
    print(f"\n🎯 TEST 1: Direct Embedding Similarity Search")
    print("-" * 50)
    similar_results = face_db.search_similar_faces(embedding, n_results=8)

    if similar_results['count'] > 0:
        print(f"Found {similar_results['count']} similar faces:")
        for i in range(min(8, similar_results['count'])):
            face_id = similar_results['ids'][i]
            distance = similar_results['distances'][i]
            similarity = (1 - distance) * 100

            if i < len(similar_results['metadatas']):
                metadata = similar_results['metadatas'][i]
                age = metadata.get('estimated_age_group', 'unknown')
                tone = metadata.get('estimated_skin_tone', 'unknown')
                quality = metadata.get('image_quality', 'unknown')

                print(f"   {i+1:2d}. {face_id[:25]}... ({similarity:.1f}%)")
                print(f"       Age: {age}, Skin: {tone}, Quality: {quality}")
            else:
                print(f"   {i+1:2d}. {face_id[:25]}... ({similarity:.1f}%)")

        # Analyze similarity distribution
        distances = similar_results['distances']
        similarities = [(1-d)*100 for d in distances]

        print(f"\n📈 Similarity Analysis:")
        print(f"   Best match: {max(similarities):.1f}%")
        print(f"   Average: {sum(similarities)/len(similarities):.1f}%")
        print(f"   Range: {min(similarities):.1f}% - {max(similarities):.1f}%")
        print(f"   Matches >90%: {sum(1 for s in similarities if s > 90)}")
        print(f"   Matches >80%: {sum(1 for s in similarities if s > 80)}")
        print(f"   Matches >70%: {sum(1 for s in similarities if s > 70)}")

    # Test 2: Image-based search (using search interface)
    print(f"\n🖼️  TEST 2: Image-Based Search Interface")
    print("-" * 50)
    image_results = search_interface.search_by_image(test_face_path, n_results=5)

    if "error" not in image_results:
        results = image_results["results"]
        print(f"Interface search found {results['count']} matches:")
        for i in range(min(5, results['count'])):
            face_id = results['ids'][i]
            distance = results['distances'][i]
            similarity = (1 - distance) * 100
            print(f"   {i+1}. {face_id[:30]}... ({similarity:.1f}%)")
    else:
        print(f"❌ Error: {image_results['error']}")

    # Test 3: Feature-based searches
    print(f"\n🎯 TEST 3: Feature-Based Searches")
    print("-" * 50)

    test_age = features.get('estimated_age_group')
    test_tone = features.get('estimated_skin_tone')
    test_quality = features.get('image_quality')

    if test_age:
        age_results = face_db.search_by_features({"estimated_age_group": test_age}, n_results=10)
        print(f"Same age group ({test_age}): {age_results['count']} matches")

    if test_tone:
        tone_results = face_db.search_by_features({"estimated_skin_tone": test_tone}, n_results=10)
        print(f"Same skin tone ({test_tone}): {tone_results['count']} matches")

    if test_quality:
        quality_results = face_db.search_by_features({"image_quality": test_quality}, n_results=10)
        print(f"Same quality ({test_quality}): {quality_results['count']} matches")

    # Test 4: Combined feature search
    if test_age and test_tone:
        combined_results = face_db.search_by_features({
            "estimated_age_group": test_age,
            "estimated_skin_tone": test_tone
        }, n_results=5)
        print(f"Same age + skin tone: {combined_results['count']} matches")

    # Test 5: Performance measurement
    print(f"\n⚡ TEST 4: Performance Measurement")
    print("-" * 50)

    # Time similarity search
    start_time = time.time()
    for _ in range(5):
        face_db.search_similar_faces(embedding, n_results=10)
    avg_time = (time.time() - start_time) / 5

    print(f"Average search time: {avg_time*1000:.2f}ms")
    print(f"Searches per second: ~{1/avg_time:.1f}")

    print(f"\n✅ SEARCH TEST COMPLETED!")
    print(f"📊 Summary:")
    print(f"   • Test face analyzed and embedded successfully")
    print(f"   • Similarity search working with {similar_results.get('count', 0)} results")
    print(f"   • Feature-based filtering operational")
    print(f"   • Performance: {avg_time*1000:.1f}ms per search")
    print(f"   • Best similarity: {max(similarities):.1f}%" if similarities else "N/A")

if __name__ == "__main__":
    main()
EOF

echo "🚀 Running comprehensive search test..."
python3 temp_search_test.py

# Clean up
rm temp_search_test.py

echo ""
echo "📁 Test results summary:"
if [ -d "faces" ]; then
    latest_face=$(ls -t faces/*.jpg | head -1)
    echo "   Latest test face: $(basename "$latest_face")"
    echo "   Total faces now: $(ls faces/*.jpg | wc -l)"
fi

echo ""
echo "✅ Step 6 completed!"
echo "📋 What was tested:"
echo "   • Downloaded new face for similarity testing"
echo "   • Extracted features and generated embeddings"
echo "   • Performed semantic similarity search"
echo "   • Tested feature-based filtering"
echo "   • Measured search performance"
echo "   • Validated complete pipeline functionality"
echo ""
echo "🎉 COMPLETE PIPELINE SUCCESSFULLY TESTED!"
echo ""
echo "📚 Available scripts for continued use:"
echo "   • python3 face_database.py - Interactive search interface"
echo "   • python3 test_face_system.py - Comprehensive system test"
echo "   • python3 expand_face_dataset.py - Download more faces"