#!/usr/bin/env python3
"""
Core Backend for Integrated Face Processing System
Combines downloading, embedding, and search functionality
"""

import requests
import os
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from PIL import Image
import io
import json
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass, asdict
import shutil
from pathlib import Path

# Try to import cv2, use fallback if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FaceData:
    """Data class for face information"""
    face_id: str
    file_path: str
    features: Dict[str, Any]
    embedding: Optional[List[float]]
    timestamp: str
    image_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SystemConfig:
    """Configuration for the entire system"""
    faces_dir: str = "./faces"
    db_path: str = "./chroma_db"
    collection_name: str = "faces"
    download_delay: float = 1.0
    max_workers: int = 2
    batch_size: int = 50
    config_file: str = "system_config.json"

    @classmethod
    def from_file(cls, filename: str = "system_config.json") -> 'SystemConfig':
        """Load configuration from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()

    def save_to_file(self, filename: Optional[str] = None):
        """Save configuration to file"""
        if filename is None:
            filename = self.config_file
        with open(filename, 'w') as f:
            json.dump(asdict(self), f, indent=2)

class SystemStats:
    """Track system-wide statistics"""
    def __init__(self):
        self.download_attempts = 0
        self.download_success = 0
        self.download_duplicates = 0
        self.download_errors = 0
        self.embed_processed = 0
        self.embed_success = 0
        self.embed_errors = 0
        self.search_queries = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def increment_download_attempts(self):
        with self.lock:
            self.download_attempts += 1

    def increment_download_success(self):
        with self.lock:
            self.download_success += 1

    def increment_download_duplicates(self):
        with self.lock:
            self.download_duplicates += 1

    def increment_download_errors(self):
        with self.lock:
            self.download_errors += 1

    def increment_embed_processed(self):
        with self.lock:
            self.embed_processed += 1

    def increment_embed_success(self):
        with self.lock:
            self.embed_success += 1

    def increment_embed_errors(self):
        with self.lock:
            self.embed_errors += 1

    def increment_search_queries(self):
        with self.lock:
            self.search_queries += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                'download_attempts': self.download_attempts,
                'download_success': self.download_success,
                'download_duplicates': self.download_duplicates,
                'download_errors': self.download_errors,
                'embed_processed': self.embed_processed,
                'embed_success': self.embed_success,
                'embed_errors': self.embed_errors,
                'search_queries': self.search_queries,
                'elapsed_time': elapsed,
                'download_rate': self.download_success / elapsed if elapsed > 0 else 0,
                'embed_rate': self.embed_success / elapsed if elapsed > 0 else 0
            }

class FaceAnalyzer:
    """Analyze face images for features"""

    def __init__(self):
        self.cv2_available = CV2_AVAILABLE
        if not self.cv2_available:
            logger.warning("OpenCV not available. Using basic image analysis.")

    def analyze_face(self, image_path: str) -> Dict[str, Any]:
        """Analyze face image and extract features"""
        try:
            # Open image
            image = Image.open(image_path)

            # Basic features
            features = {
                'width': image.size[0],
                'height': image.size[1],
                'mode': image.mode,
                'size_bytes': os.path.getsize(image_path),
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
                'format': image.format
            }

            # Advanced features with OpenCV
            if self.cv2_available:
                cv_image = cv2.imread(image_path)
                if cv_image is not None:
                    # Color analysis
                    features.update(self._analyze_colors(cv_image))
                    # Face detection
                    features.update(self._detect_faces(cv_image))

            return features

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {'error': str(e)}

    def _analyze_colors(self, cv_image) -> Dict[str, Any]:
        """Analyze color properties of the image"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)

            # Calculate statistics
            bgr_mean = np.mean(cv_image, axis=(0, 1))
            hsv_mean = np.mean(hsv, axis=(0, 1))

            return {
                'dominant_color_bgr': bgr_mean.tolist(),
                'brightness': float(np.mean(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY))),
                'contrast': float(np.std(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY))),
                'saturation_mean': float(hsv_mean[1])
            }
        except Exception as e:
            return {'color_error': str(e)}

    def _detect_faces(self, cv_image) -> Dict[str, Any]:
        """Detect faces in the image"""
        try:
            # Use Haar cascades for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            return {
                'faces_detected': len(faces),
                'face_regions': faces.tolist() if len(faces) > 0 else []
            }
        except Exception as e:
            return {'face_detection_error': str(e)}

class FaceEmbedder:
    """Create embeddings for face images"""

    def __init__(self):
        # Simple embedding using image statistics
        self.embedding_size = 512

    def create_embedding(self, image_path: str, features: Dict[str, Any]) -> List[float]:
        """Create embedding vector for face image"""
        try:
            # Load image
            image = Image.open(image_path)

            # Convert to numpy array
            img_array = np.array(image)

            # Create embedding using statistical features
            embedding = self._create_statistical_embedding(img_array, features)

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error creating embedding for {image_path}: {e}")
            return [0.0] * self.embedding_size

    def _create_statistical_embedding(self, img_array: np.ndarray, features: Dict[str, Any]) -> np.ndarray:
        """Create embedding using statistical features"""
        # Initialize embedding vector
        embedding = np.zeros(self.embedding_size)

        try:
            # Flatten image for statistics
            if len(img_array.shape) == 3:
                # Color image
                flat_r = img_array[:, :, 0].flatten()
                flat_g = img_array[:, :, 1].flatten()
                flat_b = img_array[:, :, 2].flatten()

                # Statistical features for each channel
                stats = []
                for channel in [flat_r, flat_g, flat_b]:
                    stats.extend([
                        np.mean(channel),
                        np.std(channel),
                        np.median(channel),
                        np.min(channel),
                        np.max(channel)
                    ])

                # Histogram features
                for channel in [flat_r, flat_g, flat_b]:
                    hist, _ = np.histogram(channel, bins=32, range=(0, 256))
                    hist = hist / np.sum(hist)  # Normalize
                    stats.extend(hist[:16])  # Take first 16 bins

            else:
                # Grayscale image
                flat_img = img_array.flatten()
                stats = [
                    np.mean(flat_img),
                    np.std(flat_img),
                    np.median(flat_img),
                    np.min(flat_img),
                    np.max(flat_img)
                ]

                # Histogram features
                hist, _ = np.histogram(flat_img, bins=32, range=(0, 256))
                hist = hist / np.sum(hist)  # Normalize
                stats.extend(hist)

            # Add feature-based components
            if 'brightness' in features:
                stats.append(features['brightness'] / 255.0)
            if 'contrast' in features:
                stats.append(features['contrast'] / 255.0)
            if 'saturation_mean' in features:
                stats.append(features['saturation_mean'] / 255.0)

            # Pad or truncate to embedding size
            stats = np.array(stats)
            if len(stats) > self.embedding_size:
                embedding = stats[:self.embedding_size]
            else:
                embedding[:len(stats)] = stats

        except Exception as e:
            logger.error(f"Error in statistical embedding: {e}")
            # Return random embedding as fallback
            embedding = np.random.random(self.embedding_size)

        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

class DatabaseManager:
    """Manage ChromaDB operations"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize ChromaDB connection"""
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available. Please install: pip install chromadb")
            return False

        try:
            # Create database directory
            os.makedirs(self.config.db_path, exist_ok=True)

            # Initialize client
            self.client = chromadb.PersistentClient(
                path=self.config.db_path,
                settings=Settings(
                    allow_reset=False,
                    anonymized_telemetry=False
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"description": "Face embeddings collection"}
            )

            self.initialized = True
            logger.info(f"Database initialized: {self.config.db_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False

    def add_face(self, face_data: FaceData) -> bool:
        """Add face data to the database"""
        if not self.initialized:
            return False

        try:
            self.collection.add(
                embeddings=[face_data.embedding],
                metadatas=[{
                    'face_id': face_data.face_id,
                    'file_path': face_data.file_path,
                    'timestamp': face_data.timestamp,
                    'image_hash': face_data.image_hash,
                    **face_data.features
                }],
                ids=[face_data.face_id]
            )
            return True

        except Exception as e:
            logger.error(f"Error adding face to database: {e}")
            return False

    def search_faces(self, query_embedding: List[float], n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for similar faces"""
        if not self.initialized:
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            return [
                {
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                for i in range(len(results['ids'][0]))
            ]

        except Exception as e:
            logger.error(f"Error searching faces: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        if not self.initialized:
            return {}

        try:
            count = self.collection.count()
            return {
                'name': self.config.collection_name,
                'count': count,
                'path': self.config.db_path
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

class FaceDownloader:
    """Download faces from ThisPersonDoesNotExist.com"""

    def __init__(self, config: SystemConfig, stats: SystemStats):
        self.config = config
        self.stats = stats
        self.running = False
        self.downloaded_hashes = set()

        # Create faces directory
        os.makedirs(self.config.faces_dir, exist_ok=True)

        # Load existing hashes
        self._load_existing_hashes()

    def _load_existing_hashes(self):
        """Load hashes of existing images"""
        for file_path in Path(self.config.faces_dir).rglob("*.jpg"):
            if file_path.is_file():
                try:
                    file_hash = self._get_file_hash(str(file_path))
                    if file_hash:
                        self.downloaded_hashes.add(file_hash)
                except Exception:
                    pass

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def download_face(self) -> Optional[str]:
        """Download a single face image"""
        self.stats.increment_download_attempts()

        try:
            # Download image
            response = requests.get("https://thispersondoesnotexist.com/",
                                 headers={'User-Agent': 'Mozilla/5.0'},
                                 timeout=30)
            response.raise_for_status()

            # Calculate hash
            image_hash = hashlib.md5(response.content).hexdigest()

            # Check for duplicates
            if image_hash in self.downloaded_hashes:
                self.stats.increment_download_duplicates()
                return None

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"face_{timestamp}_{image_hash[:8]}.jpg"
            file_path = os.path.join(self.config.faces_dir, filename)

            with open(file_path, 'wb') as f:
                f.write(response.content)

            # Add to downloaded hashes
            self.downloaded_hashes.add(image_hash)

            self.stats.increment_download_success()
            logger.info(f"Downloaded: {filename}")
            return file_path

        except Exception as e:
            self.stats.increment_download_errors()
            logger.error(f"Download error: {e}")
            return None

    def start_download_loop(self, callback=None):
        """Start continuous download loop"""
        self.running = True

        def download_worker():
            while self.running:
                file_path = self.download_face()
                if callback and file_path:
                    callback(file_path)
                time.sleep(self.config.download_delay)

        thread = threading.Thread(target=download_worker, daemon=True)
        thread.start()
        return thread

    def stop_download_loop(self):
        """Stop download loop"""
        self.running = False

class FaceProcessor:
    """Process faces for embedding"""

    def __init__(self, config: SystemConfig, stats: SystemStats, db_manager: DatabaseManager):
        self.config = config
        self.stats = stats
        self.db_manager = db_manager
        self.analyzer = FaceAnalyzer()
        self.embedder = FaceEmbedder()
        self.processed_files = set()

    def process_face_file(self, file_path: str) -> bool:
        """Process a single face file"""
        if file_path in self.processed_files:
            return True

        self.stats.increment_embed_processed()

        try:
            # Analyze face
            features = self.analyzer.analyze_face(file_path)
            if 'error' in features:
                self.stats.increment_embed_errors()
                return False

            # Create embedding
            embedding = self.embedder.create_embedding(file_path, features)

            # Create face data
            file_hash = self._get_file_hash(file_path)
            face_id = f"face_{int(time.time())}_{file_hash[:8]}"

            face_data = FaceData(
                face_id=face_id,
                file_path=file_path,
                features=features,
                embedding=embedding,
                timestamp=datetime.now().isoformat(),
                image_hash=file_hash
            )

            # Add to database
            if self.db_manager.add_face(face_data):
                self.processed_files.add(file_path)
                self.stats.increment_embed_success()
                logger.info(f"Processed: {os.path.basename(file_path)}")
                return True
            else:
                self.stats.increment_embed_errors()
                return False

        except Exception as e:
            self.stats.increment_embed_errors()
            logger.error(f"Error processing {file_path}: {e}")
            return False

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def process_all_faces(self, callback=None):
        """Process all faces in the faces directory"""
        face_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            face_files.extend(Path(self.config.faces_dir).rglob(ext))

        for file_path in face_files:
            if self.process_face_file(str(file_path)):
                if callback:
                    callback(str(file_path))

class IntegratedFaceSystem:
    """Main system integrating all components"""

    def __init__(self, config_file: str = "system_config.json"):
        self.config = SystemConfig.from_file(config_file)
        self.stats = SystemStats()
        self.db_manager = DatabaseManager(self.config)
        self.downloader = FaceDownloader(self.config, self.stats)
        self.processor = None  # Initialize after database

    def initialize(self) -> bool:
        """Initialize the system"""
        if not self.db_manager.initialize():
            return False

        self.processor = FaceProcessor(self.config, self.stats, self.db_manager)
        logger.info("Integrated Face System initialized")
        return True

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        stats = self.stats.get_stats()
        db_info = self.db_manager.get_collection_info()

        return {
            'database': db_info,
            'statistics': stats,
            'config': asdict(self.config),
            'faces_directory': self.config.faces_dir,
            'faces_count': len(list(Path(self.config.faces_dir).rglob("*.jpg"))) if os.path.exists(self.config.faces_dir) else 0
        }