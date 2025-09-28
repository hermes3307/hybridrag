#!/usr/bin/env python3
"""
Expand Face Dataset and Test Comparison Quality
Downloads more faces and evaluates embedding quality for better comparisons
"""

import os
import json
from face_collector import FaceCollector, process_faces, save_face_data, FaceData
from face_database import FaceDatabase
import numpy as np
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_more_faces(target_count: int = 50) -> List[str]:
    """Download more faces to reach target count"""
    faces_dir = "./faces"
    existing_files = []

    if os.path.exists(faces_dir):
        existing_files = [f for f in os.listdir(faces_dir) if f.endswith('.jpg')]

    current_count = len(existing_files)
    needed = max(0, target_count - current_count)

    print(f"📊 Current faces: {current_count}")
    print(f"📥 Downloading {needed} more faces to reach {target_count} total...")

    if needed <= 0:
        print("✅ Already have enough faces!")
        return [os.path.join(faces_dir, f) for f in existing_files]

    # Download more faces
    collector = FaceCollector(delay=1.5)  # Respectful rate limiting
    new_files = collector.download_faces_batch(count=needed * 2, max_workers=3)  # Download extra to account for duplicates

    # Get all face files
    all_files = []
    if os.path.exists(faces_dir):
        all_files = [os.path.join(faces_dir, f) for f in os.listdir(faces_dir) if f.endswith('.jpg')]

    print(f"✅ Total faces now: {len(all_files)}")
    return all_files

def analyze_face_diversity(face_data_list: List[FaceData]) -> Dict[str, Any]:
    """Analyze diversity in the face dataset"""
    analysis = {
        'total_faces': len(face_data_list),
        'age_groups': {},
        'skin_tones': {},
        'qualities': {},
        'brightness_range': {},
        'embedding_stats': {}
    }

    brightness_values = []
    embeddings = []

    for face in face_data_list:
        features = face.features

        # Count age groups
        age_group = features.get('estimated_age_group', 'unknown')
        analysis['age_groups'][age_group] = analysis['age_groups'].get(age_group, 0) + 1

        # Count skin tones
        skin_tone = features.get('estimated_skin_tone', 'unknown')
        analysis['skin_tones'][skin_tone] = analysis['skin_tones'].get(skin_tone, 0) + 1

        # Count qualities
        quality = features.get('image_quality', 'unknown')
        analysis['qualities'][quality] = analysis['qualities'].get(quality, 0) + 1

        # Collect brightness values
        brightness = features.get('brightness', 0)
        brightness_values.append(brightness)

        # Collect embeddings
        if face.embedding:
            embeddings.append(face.embedding)

    # Analyze brightness distribution
    if brightness_values:
        analysis['brightness_range'] = {
            'min': float(np.min(brightness_values)),
            'max': float(np.max(brightness_values)),
            'mean': float(np.mean(brightness_values)),
            'std': float(np.std(brightness_values))
        }

    # Analyze embedding diversity
    if embeddings:
        embeddings_array = np.array(embeddings)

        # Calculate pairwise distances to measure diversity
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                dist = np.linalg.norm(embeddings_array[i] - embeddings_array[j])
                distances.append(dist)

        analysis['embedding_stats'] = {
            'dimensions': len(embeddings[0]),
            'avg_pairwise_distance': float(np.mean(distances)) if distances else 0,
            'min_distance': float(np.min(distances)) if distances else 0,
            'max_distance': float(np.max(distances)) if distances else 0,
            'std_distance': float(np.std(distances)) if distances else 0
        }

    return analysis

def test_comparison_quality(face_db: FaceDatabase, face_data_list: List[FaceData]) -> Dict[str, Any]:
    """Test the quality of face comparisons"""
    results = {
        'total_tests': 0,
        'perfect_matches': 0,
        'high_similarity': 0,  # > 0.8
        'medium_similarity': 0,  # 0.5-0.8
        'low_similarity': 0,  # < 0.5
        'similarity_distribution': [],
        'feature_correlation': {}
    }

    print("🔍 Testing comparison quality...")

    # Test similarity for different faces
    test_count = min(10, len(face_data_list))

    for i in range(test_count):
        face = face_data_list[i]

        # Search for similar faces
        similar_results = face_db.search_similar_faces(face.embedding, n_results=5)

        if similar_results['count'] > 1:  # Should find itself + others
            distances = similar_results['distances']

            # The first result should be the face itself (distance ~0)
            self_similarity = 1 - distances[0] if distances else 0

            # Check similarities with other faces
            for j, distance in enumerate(distances[1:], 1):  # Skip self
                similarity = 1 - distance
                results['similarity_distribution'].append(similarity)

                if similarity > 0.95:
                    results['perfect_matches'] += 1
                elif similarity > 0.8:
                    results['high_similarity'] += 1
                elif similarity > 0.5:
                    results['medium_similarity'] += 1
                else:
                    results['low_similarity'] += 1

                results['total_tests'] += 1

    # Test feature-based correlations
    print("🎯 Testing feature-based searches...")

    feature_tests = [
        ('estimated_age_group', 'adult'),
        ('estimated_age_group', 'young_adult'),
        ('estimated_skin_tone', 'light'),
        ('estimated_skin_tone', 'medium'),
        ('image_quality', 'high')
    ]

    for feature_name, feature_value in feature_tests:
        search_results = face_db.search_by_features({feature_name: feature_value}, n_results=20)
        results['feature_correlation'][f"{feature_name}={feature_value}"] = search_results['count']

    return results

def evaluate_embedding_effectiveness(face_data_list: List[FaceData]) -> Dict[str, Any]:
    """Evaluate how well embeddings distinguish between different faces"""
    if len(face_data_list) < 2:
        return {'error': 'Need at least 2 faces for evaluation'}

    embeddings = np.array([face.embedding for face in face_data_list])

    # Calculate all pairwise similarities
    similarities = []
    same_group_similarities = []
    diff_group_similarities = []

    for i in range(len(face_data_list)):
        for j in range(i+1, len(face_data_list)):
            # Calculate cosine similarity
            emb_i = embeddings[i]
            emb_j = embeddings[j]

            similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
            similarities.append(similarity)

            # Check if faces are from same demographic group
            face_i_features = face_data_list[i].features
            face_j_features = face_data_list[j].features

            same_age = face_i_features.get('estimated_age_group') == face_j_features.get('estimated_age_group')
            same_tone = face_i_features.get('estimated_skin_tone') == face_j_features.get('estimated_skin_tone')

            if same_age and same_tone:
                same_group_similarities.append(similarity)
            else:
                diff_group_similarities.append(similarity)

    analysis = {
        'total_comparisons': len(similarities),
        'avg_similarity': float(np.mean(similarities)),
        'similarity_std': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
    }

    if same_group_similarities:
        analysis['same_group_avg'] = float(np.mean(same_group_similarities))
        analysis['same_group_count'] = len(same_group_similarities)

    if diff_group_similarities:
        analysis['diff_group_avg'] = float(np.mean(diff_group_similarities))
        analysis['diff_group_count'] = len(diff_group_similarities)

    # Check if embeddings can distinguish between groups
    if same_group_similarities and diff_group_similarities:
        analysis['separability'] = analysis['same_group_avg'] - analysis['diff_group_avg']
        analysis['good_separation'] = analysis['separability'] > 0.1

    return analysis

def main():
    print("🎭 Expanding Face Dataset and Testing Comparison Quality")
    print("="*70)

    # Step 1: Download more faces
    target_count = 50
    face_files = download_more_faces(target_count)

    if len(face_files) < 20:
        print("⚠️  Warning: Less than 20 faces available. Results may be limited.")

    # Step 2: Process all faces
    print(f"\n🔍 Processing {len(face_files)} faces...")

    # Check if we have existing processed data
    existing_data = []
    if os.path.exists("face_data.json"):
        with open("face_data.json", 'r') as f:
            face_data_dicts = json.load(f)
        existing_data = [FaceData(**data) for data in face_data_dicts]
        print(f"📁 Loaded {len(existing_data)} existing processed faces")

    # Process any new faces
    existing_ids = {face.face_id for face in existing_data}
    new_files = []
    for file_path in face_files:
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        if file_id not in existing_ids:
            new_files.append(file_path)

    if new_files:
        print(f"🔄 Processing {len(new_files)} new faces...")
        new_processed = process_faces(new_files)
        all_face_data = existing_data + new_processed

        # Save updated data
        save_face_data(all_face_data)
        print(f"💾 Saved {len(all_face_data)} total faces")
    else:
        all_face_data = existing_data
        print("✅ All faces already processed")

    # Step 3: Initialize database and add faces
    print(f"\n💾 Setting up database with {len(all_face_data)} faces...")
    face_db = FaceDatabase()

    # Clear existing faces to avoid duplicates
    try:
        face_db.client.delete_collection("faces")
        face_db._initialize_db()  # Recreate collection
    except:
        pass  # Collection might not exist

    added_count = face_db.add_faces(all_face_data)
    print(f"✅ Added {added_count} faces to database")

    # Step 4: Analyze dataset diversity
    print(f"\n📊 Analyzing dataset diversity...")
    diversity_analysis = analyze_face_diversity(all_face_data)

    print(f"📈 Dataset Analysis:")
    print(f"   Total faces: {diversity_analysis['total_faces']}")
    print(f"   Age groups: {diversity_analysis['age_groups']}")
    print(f"   Skin tones: {diversity_analysis['skin_tones']}")
    print(f"   Qualities: {diversity_analysis['qualities']}")
    print(f"   Embedding dimensions: {diversity_analysis['embedding_stats'].get('dimensions', 'N/A')}")
    print(f"   Avg pairwise distance: {diversity_analysis['embedding_stats'].get('avg_pairwise_distance', 'N/A'):.3f}")

    # Step 5: Test comparison quality
    print(f"\n🎯 Testing comparison quality...")
    comparison_results = test_comparison_quality(face_db, all_face_data)

    print(f"🔍 Comparison Quality Results:")
    print(f"   Total comparisons tested: {comparison_results['total_tests']}")
    print(f"   High similarity (>80%): {comparison_results['high_similarity']}")
    print(f"   Medium similarity (50-80%): {comparison_results['medium_similarity']}")
    print(f"   Low similarity (<50%): {comparison_results['low_similarity']}")

    if comparison_results['similarity_distribution']:
        avg_sim = np.mean(comparison_results['similarity_distribution'])
        print(f"   Average similarity: {avg_sim:.3f}")

    print(f"\n📋 Feature search results:")
    for feature, count in comparison_results['feature_correlation'].items():
        print(f"   {feature}: {count} matches")

    # Step 6: Evaluate embedding effectiveness
    print(f"\n🧮 Evaluating embedding effectiveness...")
    embedding_eval = evaluate_embedding_effectiveness(all_face_data)

    if 'error' not in embedding_eval:
        print(f"📊 Embedding Evaluation:")
        print(f"   Total comparisons: {embedding_eval['total_comparisons']}")
        print(f"   Average similarity: {embedding_eval['avg_similarity']:.3f}")
        print(f"   Similarity range: {embedding_eval['min_similarity']:.3f} - {embedding_eval['max_similarity']:.3f}")

        if 'same_group_avg' in embedding_eval and 'diff_group_avg' in embedding_eval:
            print(f"   Same group avg: {embedding_eval['same_group_avg']:.3f}")
            print(f"   Different group avg: {embedding_eval['diff_group_avg']:.3f}")
            print(f"   Separability: {embedding_eval.get('separability', 0):.3f}")
            print(f"   Good separation: {'Yes' if embedding_eval.get('good_separation', False) else 'No'}")

    # Step 7: Recommendations
    print(f"\n💡 Recommendations:")

    total_faces = len(all_face_data)
    if total_faces < 30:
        print("   ⚠️  Consider downloading more faces (30+ recommended for robust testing)")
    elif total_faces < 100:
        print("   ✅ Good dataset size for testing")
    else:
        print("   🎉 Excellent dataset size for comprehensive testing")

    diversity_score = len(diversity_analysis['age_groups']) + len(diversity_analysis['skin_tones'])
    if diversity_score < 4:
        print("   ⚠️  Limited diversity detected - consider collecting more varied faces")
    else:
        print("   ✅ Good diversity in age groups and skin tones")

    if 'avg_pairwise_distance' in diversity_analysis['embedding_stats']:
        avg_dist = diversity_analysis['embedding_stats']['avg_pairwise_distance']
        if avg_dist < 0.5:
            print("   ⚠️  Low embedding diversity - faces may be too similar")
        elif avg_dist > 2.0:
            print("   ⚠️  High embedding variance - may indicate noise")
        else:
            print("   ✅ Good embedding diversity for comparisons")

    print(f"\n✅ Dataset expansion and quality testing completed!")
    print(f"📁 Dataset: {total_faces} faces ready for semantic search")

if __name__ == "__main__":
    main()