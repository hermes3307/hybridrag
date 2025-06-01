import os
import numpy as np
import pickle
from typing import List, Tuple, Optional
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import faiss
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class ImageVectorDatabase:
    def __init__(self, db_path: str = "image_vector_db", dimension: int = 2048):
        """
        Initialize the Image Vector Database
        
        Args:
            db_path: Path to store the database files
            dimension: Dimension of the feature vectors (ResNet50 final layer is 2048)
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.dimension = dimension
        
        # Initialize FAISS index for similarity search
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Store metadata (image paths, timestamps, etc.)
        self.metadata = []
        
        # Initialize ResNet50 model for feature extraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final classification layer to get features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load existing database if available
        self.load_database()
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from an image using ResNet50
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.squeeze().cpu().numpy()
                
            # Normalize for cosine similarity
            features = features / np.linalg.norm(features)
            return features
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def add_image(self, image_path: str, metadata: dict = None) -> bool:
        """
        Add an image to the vector database
        
        Args:
            image_path: Path to the image file
            metadata: Additional metadata for the image
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return False
        
        # Extract features
        features = self.extract_features(image_path)
        if features is None:
            return False
        
        # Add to FAISS index
        self.index.add(features.reshape(1, -1).astype('float32'))
        
        # Store metadata
        image_metadata = {
            'path': image_path,
            'timestamp': datetime.now().isoformat(),
            'index_id': len(self.metadata),
            **(metadata or {})
        }
        self.metadata.append(image_metadata)
        
        print(f"Added image: {image_path}")
        return True
    
    def add_images_from_directory(self, directory_path: str, 
                                 extensions: List[str] = None) -> int:
        """
        Add all images from a directory to the database
        
        Args:
            directory_path: Path to the directory containing images
            extensions: List of file extensions to include
            
        Returns:
            Number of images successfully added
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        directory = Path(directory_path)
        if not directory.exists():
            print(f"Directory not found: {directory_path}")
            return 0
        
        added_count = 0
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in extensions:
                if self.add_image(str(file_path)):
                    added_count += 1
        
        print(f"Added {added_count} images from {directory_path}")
        return added_count
    
    def search_similar_images(self, query_image_path: str, 
                            top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for similar images in the database
        
        Args:
            query_image_path: Path to the query image
            top_k: Number of similar images to return
            
        Returns:
            List of tuples (image_path, similarity_score, metadata)
        """
        if self.index.ntotal == 0:
            print("Database is empty. Add some images first.")
            return []
        
        # Extract features from query image
        query_features = self.extract_features(query_image_path)
        if query_features is None:
            return []
        
        # Search in FAISS index
        query_vector = query_features.reshape(1, -1).astype('float32')
        similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx != -1:  # Valid result
                metadata = self.metadata[idx]
                results.append((metadata['path'], float(similarity), metadata))
        
        return results
    
    def visualize_search_results(self, query_image_path: str, 
                               results: List[Tuple[str, float, dict]],
                               max_display: int = 6):
        """
        Visualize search results using matplotlib
        
        Args:
            query_image_path: Path to the query image
            results: Search results from search_similar_images
            max_display: Maximum number of results to display
        """
        n_results = min(len(results), max_display)
        fig, axes = plt.subplots(2, (n_results + 1) // 2 + 1, figsize=(15, 8))
        axes = axes.flatten()
        
        # Display query image
        query_img = Image.open(query_image_path)
        axes[0].imshow(query_img)
        axes[0].set_title("Query Image", fontweight='bold')
        axes[0].axis('off')
        
        # Display similar images
        for i, (img_path, similarity, metadata) in enumerate(results[:n_results]):
            try:
                img = Image.open(img_path)
                axes[i + 1].imshow(img)
                axes[i + 1].set_title(f"Similarity: {similarity:.3f}\n{Path(img_path).name}", 
                                    fontsize=10)
                axes[i + 1].axis('off')
            except Exception as e:
                axes[i + 1].text(0.5, 0.5, f"Error loading\n{Path(img_path).name}", 
                                ha='center', va='center')
                axes[i + 1].axis('off')
        
        # Hide unused subplots
        for i in range(n_results + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_database(self):
        """Save the current database state to disk"""
        # Save FAISS index
        faiss.write_index(self.index, str(self.db_path / "image_index.faiss"))
        
        # Save metadata
        with open(self.db_path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Database saved to {self.db_path}")
    
    def load_database(self):
        """Load database state from disk"""
        index_path = self.db_path / "image_index.faiss"
        metadata_path = self.db_path / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                print(f"Loaded database with {len(self.metadata)} images")
            except Exception as e:
                print(f"Error loading database: {e}")
                print("Starting with empty database")
    
    def get_database_stats(self) -> dict:
        """Get statistics about the current database"""
        return {
            'total_images': len(self.metadata),
            'index_size': self.index.ntotal,
            'dimension': self.dimension,
            'database_path': str(self.db_path)
        }
    
    def remove_image(self, image_path: str) -> bool:
        """
        Remove an image from the database (Note: FAISS doesn't support efficient deletion,
        so this marks the image as deleted and rebuilds the index)
        """
        # Find the image in metadata
        for i, meta in enumerate(self.metadata):
            if meta['path'] == image_path:
                # Remove from metadata
                self.metadata.pop(i)
                
                # Rebuild FAISS index (inefficient but necessary)
                self._rebuild_index()
                print(f"Removed image: {image_path}")
                return True
        
        print(f"Image not found in database: {image_path}")
        return False
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from remaining images"""
        self.index = faiss.IndexFlatIP(self.dimension)
        
        for i, meta in enumerate(self.metadata):
            features = self.extract_features(meta['path'])
            if features is not None:
                self.index.add(features.reshape(1, -1).astype('float32'))
                meta['index_id'] = i


# Example usage and testing
def main():
    # Initialize the database
    db = ImageVectorDatabase()
    
    print("Image Vector Database initialized!")
    print(f"Database stats: {db.get_database_stats()}")
    
    # Example: Add images from a directory
    # db.add_images_from_directory("path/to/your/images")
    
    # Example: Add a single image
    # db.add_image("path/to/image.jpg", metadata={'category': 'nature'})
    
    # Example: Search for similar images
    # query_path = "path/to/query_image.jpg"
    # results = db.search_similar_images(query_path, top_k=5)
    # 
    # print(f"Found {len(results)} similar images:")
    # for img_path, similarity, metadata in results:
    #     print(f"  {img_path} - Similarity: {similarity:.3f}")
    # 
    # # Visualize results
    # db.visualize_search_results(query_path, results)
    
    # Save database
    # db.save_database()

if __name__ == "__main__":
    main()


# Additional utility functions for advanced RAG operations

class ImageRAGSystem:
    """
    Advanced RAG system for images with text descriptions and semantic search
    """
    def __init__(self, vector_db: ImageVectorDatabase):
        self.vector_db = vector_db
        self.text_embeddings = {}  # Store text descriptions and embeddings
    
    def add_image_with_description(self, image_path: str, description: str, 
                                 metadata: dict = None):
        """Add an image with text description for hybrid search"""
        # Add image to vector database
        success = self.vector_db.add_image(image_path, metadata)
        
        if success:
            # Store text description (in a real implementation, you'd use
            # a text embedding model like CLIP or sentence transformers)
            image_id = len(self.vector_db.metadata) - 1
            self.text_embeddings[image_id] = {
                'description': description,
                'path': image_path
            }
        
        return success
    
    def hybrid_search(self, query_image_path: str = None, 
                     query_text: str = None, top_k: int = 5):
        """
        Perform hybrid search using both image similarity and text matching
        """
        results = []
        
        if query_image_path:
            # Image-based search
            image_results = self.vector_db.search_similar_images(query_image_path, top_k)
            results.extend(image_results)
        
        if query_text:
            # Text-based search (simple keyword matching for demo)
            # In production, use proper text embeddings
            text_results = []
            for img_id, text_data in self.text_embeddings.items():
                if query_text.lower() in text_data['description'].lower():
                    metadata = self.vector_db.metadata[img_id]
                    text_results.append((text_data['path'], 1.0, metadata))
            results.extend(text_results)
        
        # Remove duplicates and sort by similarity
        unique_results = {}
        for path, similarity, metadata in results:
            if path not in unique_results or unique_results[path][1] < similarity:
                unique_results[path] = (path, similarity, metadata)
        
        return sorted(unique_results.values(), key=lambda x: x[1], reverse=True)[:top_k]