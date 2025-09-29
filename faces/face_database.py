#!/usr/bin/env python3
"""
Face Database Integration with ChromaDB
Stores face embeddings and metadata for semantic search
"""

import chromadb
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import os
from face_collector import FaceData, FaceCollector, process_faces
import base64
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceDatabase:
    """ChromaDB integration for face embeddings and semantic search"""

    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "faces"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(path=self.db_path)

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Face embeddings for semantic search",
                    "created_at": datetime.now().isoformat(),
                    "embedding_type": "face_features"
                }
            )

            logger.info(f"Initialized face database: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def add_faces(self, face_data_list: List[FaceData]) -> int:
        """Add face data to the database"""
        try:
            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for face_data in face_data_list:
                # Create unique ID
                face_id = f"face_{face_data.face_id}_{face_data.image_hash[:8]}"
                ids.append(face_id)

                # Prepare embedding
                embeddings.append(face_data.embedding)

                # Prepare metadata - convert tuples to strings for ChromaDB compatibility
                metadata = {
                    "file_path": face_data.file_path,
                    "timestamp": face_data.timestamp,
                    "image_hash": face_data.image_hash,
                }

                # Add features, converting tuples to strings
                for key, value in face_data.features.items():
                    if isinstance(value, tuple):
                        metadata[key] = str(value)
                    elif isinstance(value, (int, float, str, bool)) or value is None:
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)

                metadatas.append(metadata)

                # Create document (text description for search)
                document = self._create_face_description(face_data)
                documents.append(document)

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

            logger.info(f"Added {len(face_data_list)} faces to database")
            return len(face_data_list)

        except Exception as e:
            logger.error(f"Error adding faces to database: {e}")
            return 0

    def _create_face_description(self, face_data: FaceData) -> str:
        """Create a text description of the face for document search"""
        features = face_data.features

        description_parts = [
            f"Face ID: {face_data.face_id}",
            f"Estimated age group: {features.get('estimated_age_group', 'unknown')}",
            f"Estimated skin tone: {features.get('estimated_skin_tone', 'unknown')}",
            f"Image quality: {features.get('image_quality', 'unknown')}",
            f"Brightness level: {features.get('brightness', 0):.1f}",
            f"Image captured at: {face_data.timestamp}"
        ]

        return " | ".join(description_parts)

    def search_similar_faces(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Search for similar faces using embedding similarity"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )

            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "count": len(results["ids"][0]) if results["ids"] else 0
            }

        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            return {"ids": [], "metadatas": [], "documents": [], "distances": [], "count": 0}

    def search_by_features(self, feature_filters: Dict[str, Any], n_results: int = 10) -> Dict[str, Any]:
        """Search faces by feature criteria"""
        try:
            # Build where clause for filtering
            where_clause = {}

            for key, value in feature_filters.items():
                if isinstance(value, str):
                    where_clause[key] = value
                elif isinstance(value, (int, float)):
                    # For numeric values, you might want range queries
                    where_clause[key] = value

            results = self.collection.get(
                where=where_clause,
                limit=n_results,
                include=["metadatas", "documents"]
            )

            return {
                "ids": results["ids"] if results["ids"] else [],
                "metadatas": results["metadatas"] if results["metadatas"] else [],
                "documents": results["documents"] if results["documents"] else [],
                "count": len(results["ids"]) if results["ids"] else 0
            }

        except Exception as e:
            logger.error(f"Error searching by features: {e}")
            return {"ids": [], "metadatas": [], "documents": [], "count": 0}

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            count = self.collection.count()

            # Get sample data to analyze
            sample = self.collection.peek(limit=min(count, 100))

            stats = {
                "total_faces": count,
                "collection_name": self.collection_name,
                "database_path": self.db_path
            }

            if sample["metadatas"]:
                # Analyze feature distributions
                age_groups = {}
                skin_tones = {}
                qualities = {}

                for metadata in sample["metadatas"]:
                    # Count age groups
                    age_group = metadata.get("estimated_age_group", "unknown")
                    age_groups[age_group] = age_groups.get(age_group, 0) + 1

                    # Count skin tones
                    skin_tone = metadata.get("estimated_skin_tone", "unknown")
                    skin_tones[skin_tone] = skin_tones.get(skin_tone, 0) + 1

                    # Count qualities
                    quality = metadata.get("image_quality", "unknown")
                    qualities[quality] = qualities.get(quality, 0) + 1

                stats.update({
                    "age_group_distribution": age_groups,
                    "skin_tone_distribution": skin_tones,
                    "quality_distribution": qualities
                })

            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"total_faces": 0, "error": str(e)}

    def find_duplicate_faces(self, similarity_threshold: float = 0.95) -> List[List[str]]:
        """Find potential duplicate faces based on embedding similarity"""
        try:
            # Get all faces
            all_faces = self.collection.get(include=["embeddings"])

            duplicates = []
            processed_ids = set()

            if (not all_faces["ids"] or len(all_faces["ids"]) == 0 or
                not all_faces["embeddings"] or len(all_faces["embeddings"]) == 0):
                return duplicates

            embeddings = np.array(all_faces["embeddings"])
            ids = all_faces["ids"]

            # Compare each face with others
            for i, face_id in enumerate(ids):
                if face_id in processed_ids:
                    continue

                similar_group = [face_id]
                current_embedding = embeddings[i]

                for j, other_id in enumerate(ids):
                    if i != j and other_id not in processed_ids:
                        other_embedding = embeddings[j]

                        # Calculate cosine similarity
                        similarity = np.dot(current_embedding, other_embedding) / (
                            np.linalg.norm(current_embedding) * np.linalg.norm(other_embedding)
                        )

                        if similarity >= similarity_threshold:
                            similar_group.append(other_id)
                            processed_ids.add(other_id)

                if len(similar_group) > 1:
                    duplicates.append(similar_group)
                    processed_ids.update(similar_group)
                else:
                    processed_ids.add(face_id)

            return duplicates

        except Exception as e:
            logger.error(f"Error finding duplicates: {e}")
            return []

class FaceSearchInterface:
    """Interactive interface for face search"""

    def __init__(self, face_db: FaceDatabase):
        self.face_db = face_db

    def search_by_image(self, image_path: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for similar faces using an uploaded image"""
        try:
            # Import here to avoid circular imports
            from face_collector import FaceAnalyzer, FaceEmbedder

            analyzer = FaceAnalyzer()
            embedder = FaceEmbedder()

            # Process the query image
            features = analyzer.estimate_basic_features(image_path)
            embedding = embedder.generate_embedding(image_path, features)

            if not embedding:
                return {"error": "Could not generate embedding for query image"}

            # Search for similar faces
            results = self.face_db.search_similar_faces(embedding, n_results)

            return {
                "query_features": features,
                "results": results
            }

        except Exception as e:
            logger.error(f"Error searching by image: {e}")
            return {"error": str(e)}

    def interactive_search(self):
        """Interactive search interface"""
        print("\n🔍 Face Search Interface")
        print("="*40)

        while True:
            print("\nSearch Options:")
            print("1. 🖼️  Search by image file")
            print("2. 🎯 Search by features")
            print("3. 📊 Show database stats")
            print("4. 🔍 Find duplicate faces")
            print("5. 🚪 Exit")

            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == "1":
                self._search_by_image_interface()
            elif choice == "2":
                self._search_by_features_interface()
            elif choice == "3":
                self._show_stats()
            elif choice == "4":
                self._find_duplicates()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")

    def _search_by_image_interface(self):
        """Interactive image search"""
        image_path = input("Enter path to image file: ").strip()

        if not os.path.exists(image_path):
            print("❌ Image file not found.")
            return

        try:
            n_results = int(input("Number of results (default 5): ") or "5")
        except ValueError:
            n_results = 5

        print("🔍 Searching...")
        results = self.search_by_image(image_path, n_results)

        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return

        self._display_search_results(results["results"])

    def _search_by_features_interface(self):
        """Interactive feature search"""
        print("\n🎯 Search by Features")
        print("Available features: estimated_age_group, estimated_skin_tone, image_quality")

        filters = {}

        age_group = input("Age group (young_adult/adult/mature_adult, or press Enter): ").strip()
        if age_group:
            filters["estimated_age_group"] = age_group

        skin_tone = input("Skin tone (light/medium/dark, or press Enter): ").strip()
        if skin_tone:
            filters["estimated_skin_tone"] = skin_tone

        quality = input("Image quality (high/medium/low, or press Enter): ").strip()
        if quality:
            filters["image_quality"] = quality

        if not filters:
            print("❌ No filters specified.")
            return

        try:
            n_results = int(input("Number of results (default 10): ") or "10")
        except ValueError:
            n_results = 10

        print("🔍 Searching...")
        results = self.face_db.search_by_features(filters, n_results)
        self._display_search_results(results)

    def _display_search_results(self, results: Dict[str, Any]):
        """Display search results"""
        if results["count"] == 0:
            print("❌ No results found.")
            return

        print(f"\n✅ Found {results['count']} results:")
        print("-" * 50)

        for i in range(results["count"]):
            print(f"\n{i+1}. Face ID: {results['ids'][i]}")

            if i < len(results["metadatas"]):
                metadata = results["metadatas"][i]
                print(f"   File: {os.path.basename(metadata.get('file_path', 'unknown'))}")
                print(f"   Age Group: {metadata.get('estimated_age_group', 'unknown')}")
                print(f"   Skin Tone: {metadata.get('estimated_skin_tone', 'unknown')}")
                print(f"   Quality: {metadata.get('image_quality', 'unknown')}")

            if i < len(results.get("distances", [])):
                print(f"   Similarity: {1 - results['distances'][i]:.3f}")

    def _show_stats(self):
        """Show database statistics"""
        print("\n📊 Database Statistics")
        print("="*40)

        stats = self.face_db.get_database_stats()

        print(f"Total Faces: {stats['total_faces']:,}")
        print(f"Collection: {stats['collection_name']}")

        if "age_group_distribution" in stats:
            print(f"\n🎂 Age Group Distribution:")
            for group, count in stats["age_group_distribution"].items():
                print(f"   {group}: {count}")

        if "skin_tone_distribution" in stats:
            print(f"\n🎨 Skin Tone Distribution:")
            for tone, count in stats["skin_tone_distribution"].items():
                print(f"   {tone}: {count}")

        if "quality_distribution" in stats:
            print(f"\n📸 Quality Distribution:")
            for quality, count in stats["quality_distribution"].items():
                print(f"   {quality}: {count}")

    def _find_duplicates(self):
        """Find and display duplicate faces"""
        print("\n🔍 Finding duplicate faces...")

        duplicates = self.face_db.find_duplicate_faces()

        if not duplicates:
            print("✅ No duplicate faces found.")
            return

        print(f"Found {len(duplicates)} groups of similar faces:")

        for i, group in enumerate(duplicates):
            print(f"\n{i+1}. Similar faces ({len(group)} images):")
            for face_id in group:
                print(f"   • {face_id}")

def main():
    """Main function to demonstrate face database functionality"""
    print("🎭 Face Database System")
    print("="*40)

    # Initialize database
    face_db = FaceDatabase()

    # Check if we have any faces in the database
    stats = face_db.get_database_stats()
    print(f"Current database has {stats['total_faces']} faces")

    if stats['total_faces'] == 0:
        print("\n📥 No faces in database. Let's collect some...")

        # Import and run face collector
        collector = FaceCollector(delay=2.0)
        print("Downloading faces from ThisPersonDoesNotExist.com...")

        face_files = collector.download_faces_batch(count=5, max_workers=2)

        if face_files:
            print("Processing faces...")
            processed_faces = process_faces(face_files)

            if processed_faces:
                print("Adding faces to database...")
                face_db.add_faces(processed_faces)
                print(f"✅ Added {len(processed_faces)} faces to database")

    # Launch search interface only if not in non-interactive mode
    import sys
    if '--no-interactive' not in sys.argv:
        search_interface = FaceSearchInterface(face_db)
        search_interface.interactive_search()
    else:
        print("Face database loaded. Use the API for programmatic access.")

if __name__ == "__main__":
    main()