#!/usr/bin/env python3
"""
Face Collection and Embedding System
Downloads synthetic faces from ThisPersonDoesNotExist.com and creates embeddings for semantic search
"""

import requests
import os
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
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
# Try to import cv2, use fallback if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Using basic image processing.")

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
class DownloadConfig:
    """Configuration for background downloader"""
    faces_dir: str = "./faces"
    delay: float = 1.0
    max_workers: int = 2
    download_limit: Optional[int] = None
    unlimited_download: bool = True
    check_duplicates: bool = True
    config_file: str = "download_config.json"

    @classmethod
    def from_file(cls, filename: str = "download_config.json") -> 'DownloadConfig':
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

class DownloadStats:
    """Track download statistics"""
    def __init__(self):
        self.total_attempts = 0
        self.successful_downloads = 0
        self.duplicates_skipped = 0
        self.errors = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def increment_attempts(self):
        with self.lock:
            self.total_attempts += 1

    def increment_success(self):
        with self.lock:
            self.successful_downloads += 1

    def increment_duplicates(self):
        with self.lock:
            self.duplicates_skipped += 1

    def increment_errors(self):
        with self.lock:
            self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            elapsed_time = time.time() - self.start_time
            download_rate = self.successful_downloads / elapsed_time if elapsed_time > 0 else 0
            return {
                'total_attempts': self.total_attempts,
                'successful_downloads': self.successful_downloads,
                'duplicates_skipped': self.duplicates_skipped,
                'errors': self.errors,
                'elapsed_time': elapsed_time,
                'download_rate': download_rate
            }

class BackgroundDownloader:
    """Background face downloader with statistics tracking"""

    def __init__(self, config: DownloadConfig):
        self.config = config
        self.stats = DownloadStats()
        self.running = False
        self.collector = FaceCollector(storage_dir=config.faces_dir, delay=config.delay)

    def start_background_download(self):
        """Start downloading faces in the background"""
        self.running = True
        self.stats = DownloadStats()  # Reset stats

        logger.info("Starting background download...")

        # Determine how many faces to download
        if self.config.unlimited_download:
            # Download continuously until stopped
            face_id_counter = 0
            while self.running:
                self._download_single_face(face_id_counter)
                face_id_counter += 1
        else:
            # Download up to limit
            download_limit = self.config.download_limit or 100
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for i in range(download_limit):
                    if not self.running:
                        break
                    future = executor.submit(self._download_single_face, i)
                    futures.append(future)

                # Wait for all to complete
                for future in as_completed(futures):
                    if not self.running:
                        break
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Download error: {e}")

        logger.info("Background download stopped")
        self.running = False

    def _download_single_face(self, face_id: int) -> Optional[str]:
        """Download a single face and update statistics"""
        if not self.running:
            return None

        self.stats.increment_attempts()

        try:
            face_id_str = f"{face_id:06d}"
            result = self.collector.download_face(face_id_str)

            if result is None:
                # Duplicate detected
                self.stats.increment_duplicates()
            else:
                # Success
                self.stats.increment_success()

            return result

        except Exception as e:
            logger.error(f"Error downloading face {face_id}: {e}")
            self.stats.increment_errors()
            return None

    def stop(self):
        """Stop the background download"""
        self.running = False

class FaceCollector:
    """Downloads and processes faces from ThisPersonDoesNotExist.com"""

    def __init__(self, storage_dir: str = "./faces", delay: float = 1.0):
        self.storage_dir = storage_dir
        self.delay = delay
        self.downloaded_hashes = set()
        self.lock = threading.Lock()

        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)

        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def download_face(self, face_id: str) -> Optional[str]:
        """Download a single face image with metadata JSON"""
        try:
            # ThisPersonDoesNotExist.com generates new face each time
            url = "https://thispersondoesnotexist.com/"

            logger.info(f"Downloading face {face_id}...")
            download_time = datetime.now()
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Create image hash for deduplication
            image_hash = hashlib.md5(response.content).hexdigest()

            with self.lock:
                if image_hash in self.downloaded_hashes:
                    logger.info(f"Duplicate image detected, skipping...")
                    return None
                self.downloaded_hashes.add(image_hash)

            # Save image
            filename = f"face_{face_id}_{image_hash[:8]}.jpg"
            file_path = os.path.join(self.storage_dir, filename)

            with open(file_path, 'wb') as f:
                f.write(response.content)

            # Extract image properties and analyze face features
            try:
                with Image.open(file_path) as img:
                    image_width, image_height = img.size
                    image_format = img.format
                    image_mode = img.mode

                # Analyze face features
                analyzer = FaceAnalyzer()
                face_features = analyzer.estimate_basic_features(file_path)
                face_bbox = analyzer.detect_face(file_path)

                # Create metadata with important queryable fields
                metadata = {
                    # Basic identifiers
                    'filename': filename,
                    'file_path': file_path,
                    'face_id': face_id,
                    'md5_hash': image_hash,

                    # Download info
                    'download_timestamp': download_time.isoformat(),
                    'download_date': download_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'source_url': url,
                    'http_status_code': response.status_code,

                    # File properties
                    'file_size_bytes': len(response.content),
                    'file_size_kb': round(len(response.content) / 1024, 2),

                    # Image properties (queryable)
                    'image_properties': {
                        'width': image_width,
                        'height': image_height,
                        'format': image_format,
                        'mode': image_mode,
                        'dimensions': f"{image_width}x{image_height}"
                    },

                    # Face features (IMPORTANT FOR QUERYING)
                    'face_features': face_features,
                    'face_bbox': face_bbox if face_bbox else None,

                    # Queryable attributes extracted from features
                    'queryable_attributes': {
                        'estimated_age_group': face_features.get('estimated_age_group', 'unknown'),
                        'estimated_skin_tone': face_features.get('estimated_skin_tone', 'unknown'),
                        'image_quality': face_features.get('image_quality', 'unknown'),
                        'brightness_level': 'bright' if face_features.get('brightness', 0) > 150 else 'dark',
                        'image_size': face_features.get('image_size', 'unknown')
                    },

                    # HTTP headers for debugging
                    'http_headers': dict(response.headers),

                    # Downloader config
                    'downloader_config': {
                        'storage_dir': self.storage_dir,
                        'delay': self.delay
                    }
                }

                # Save metadata JSON alongside image
                json_filename = f"face_{face_id}_{image_hash[:8]}.json"
                json_path = os.path.join(self.storage_dir, json_filename)

                with open(json_path, 'w') as json_file:
                    json.dump(metadata, json_file, indent=2)

                logger.info(f"Saved face to: {file_path}")
                logger.info(f"Saved metadata to: {json_filename}")

            except Exception as meta_error:
                logger.warning(f"Failed to save metadata for {face_id}: {meta_error}")

            if self.delay > 0:
                time.sleep(self.delay)

            return file_path

        except Exception as e:
            logger.error(f"Error downloading face {face_id}: {e}")
            return None

    def download_faces_batch(self, count: int, max_workers: int = 3) -> List[str]:
        """Download multiple faces concurrently"""
        face_ids = [f"{i:04d}" for i in range(count)]
        downloaded_files = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(self.download_face, face_id): face_id
                           for face_id in face_ids}

            for future in as_completed(future_to_id):
                face_id = future_to_id[future]
                try:
                    file_path = future.result()
                    if file_path:
                        downloaded_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error processing face {face_id}: {e}")

        logger.info(f"Downloaded {len(downloaded_files)} unique faces out of {count} attempts")
        return downloaded_files

class FaceAnalyzer:
    """Analyzes faces to extract features like age, gender, ethnicity"""

    def __init__(self):
        self.face_cascade = None
        self._load_opencv_models()

    def _load_opencv_models(self):
        """Load OpenCV face detection models"""
        if not CV2_AVAILABLE:
            logger.info("OpenCV not available, using basic image processing")
            return

        try:
            # Try to load Haar cascade (comes with OpenCV)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("OpenCV face detection loaded")
        except Exception as e:
            logger.warning(f"Could not load OpenCV models: {e}")

    def detect_face(self, image_path: str) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in image and return bounding box"""
        try:
            if CV2_AVAILABLE:
                image = cv2.imread(image_path)
                if image is None:
                    return None

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                if self.face_cascade is not None:
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        # Return the largest face
                        largest_face = max(faces, key=lambda x: x[2] * x[3])
                        return tuple(largest_face)

                # If no face detected, assume whole image is face (common for ThisPersonDoesNotExist)
                h, w = gray.shape
                return (0, 0, w, h)
            else:
                # Fallback: assume whole image is face
                from PIL import Image
                with Image.open(image_path) as img:
                    w, h = img.size
                    return (0, 0, w, h)

        except Exception as e:
            logger.error(f"Error detecting face in {image_path}: {e}")
            return None

    def estimate_basic_features(self, image_path: str) -> Dict[str, Any]:
        """Estimate basic features from image using simple heuristics"""
        try:
            if CV2_AVAILABLE:
                # Use OpenCV for detailed analysis
                image = cv2.imread(image_path)
                if image is None:
                    return {}

                h, w, c = image.shape

                # Basic image statistics
                brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

                # Color analysis for rough skin tone estimation
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # Get center region for face analysis
                center_h, center_w = h//2, w//2
                face_region = hsv[center_h-50:center_h+50, center_w-50:center_w+50]

                if face_region.size > 0:
                    avg_hue = np.mean(face_region[:,:,0])
                    avg_saturation = np.mean(face_region[:,:,1])
                    avg_value = np.mean(face_region[:,:,2])
                else:
                    avg_hue = avg_saturation = avg_value = 0

                image_quality = self._estimate_quality_cv2(image)
            else:
                # Use PIL for basic analysis
                from PIL import Image
                with Image.open(image_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Convert to numpy array
                    img_array = np.array(img)
                    h, w, c = img_array.shape

                    # Basic brightness
                    brightness = np.mean(img_array)

                    # Simple color analysis
                    center_h, center_w = h//2, w//2
                    face_region = img_array[center_h-50:center_h+50, center_w-50:center_w+50]

                    if face_region.size > 0:
                        avg_hue = np.mean(face_region[:,:,0])  # Red channel as proxy
                        avg_saturation = np.std(face_region)    # Standard deviation as proxy
                        avg_value = np.mean(face_region)
                    else:
                        avg_hue = avg_saturation = avg_value = 0

                    image_quality = self._estimate_quality_pil(img_array)

            # Simple heuristics (these are very basic estimates)
            features = {
                'brightness': float(brightness),
                'avg_hue': float(avg_hue),
                'avg_saturation': float(avg_saturation),
                'avg_value': float(avg_value),
                'image_size': f"{w}x{h}",

                # Rough estimates based on color values
                'estimated_age_group': self._estimate_age_group(brightness, avg_value),
                'estimated_skin_tone': self._estimate_skin_tone(avg_hue, avg_saturation),
                'image_quality': image_quality
            }

            return features

        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            return {}

    def _estimate_age_group(self, brightness: float, value: float) -> str:
        """Very rough age group estimation based on image characteristics"""
        # This is a simplified heuristic - real age estimation requires trained models
        if brightness > 180:
            return "young_adult"  # Brighter images often show younger people
        elif brightness > 120:
            return "adult"
        else:
            return "mature_adult"

    def _estimate_skin_tone(self, hue: float, saturation: float) -> str:
        """Rough skin tone categorization"""
        # Very simplified categorization
        if saturation < 50:
            return "light"
        elif saturation < 100:
            return "medium"
        else:
            return "dark"

    def _estimate_quality_cv2(self, image: np.ndarray) -> str:
        """Estimate image quality using OpenCV"""
        # Calculate image sharpness using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        if variance > 500:
            return "high"
        elif variance > 100:
            return "medium"
        else:
            return "low"

    def _estimate_quality_pil(self, image: np.ndarray) -> str:
        """Estimate image quality using PIL/numpy"""
        # Simple quality estimation based on image variance
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        variance = np.var(gray)

        if variance > 1000:
            return "high"
        elif variance > 500:
            return "medium"
        else:
            return "low"

class FaceEmbedder:
    """Generate embeddings for faces using available models"""

    def __init__(self):
        self.model = None
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Load face embedding model"""
        try:
            # Try to use simple CNN-based features for now
            # In production, you'd use models like FaceNet, ArcFace, etc.
            logger.info("Using basic CNN features for embeddings")
            self.model = "basic_cnn"
        except Exception as e:
            logger.warning(f"Could not load advanced embedding model: {e}")
            self.model = "basic_features"

    def generate_embedding(self, image_path: str, features: Dict[str, Any]) -> List[float]:
        """Generate face embedding vector"""
        try:
            if CV2_AVAILABLE:
                # Load and preprocess image with OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    return []

                # Resize to standard size
                image = cv2.resize(image, (224, 224))

                if self.model == "basic_cnn":
                    return self._generate_cnn_features_cv2(image, features)
                else:
                    return self._generate_basic_features_cv2(image, features)
            else:
                # Load and preprocess image with PIL
                from PIL import Image
                with Image.open(image_path) as img:
                    # Convert to RGB and resize
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((224, 224))
                    image = np.array(img)

                    if self.model == "basic_cnn":
                        return self._generate_cnn_features_pil(image, features)
                    else:
                        return self._generate_basic_features_pil(image, features)

        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {e}")
            return []

    def _generate_cnn_features_cv2(self, image: np.ndarray, features: Dict[str, Any]) -> List[float]:
        """Generate CNN-based features using OpenCV"""
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract various features
        feature_vector = []

        # Histogram features
        hist_gray = cv2.calcHist([gray], [0], None, [32], [0, 256])
        feature_vector.extend(hist_gray.flatten().tolist())

        # Color moments
        for channel in cv2.split(hsv):
            feature_vector.extend([
                np.mean(channel),
                np.std(channel),
                np.mean(np.power(channel - np.mean(channel), 3))  # skewness
            ])

        # Texture features (simplified)
        for i in range(0, gray.shape[0], 32):
            for j in range(0, gray.shape[1], 32):
                patch = gray[i:i+32, j:j+32]
                if patch.size > 0:
                    feature_vector.extend([
                        np.mean(patch),
                        np.std(patch)
                    ])

        # Add extracted features
        feature_vector.extend([
            features.get('brightness', 0),
            features.get('avg_hue', 0),
            features.get('avg_saturation', 0),
            features.get('avg_value', 0)
        ])

        # Normalize to unit vector
        feature_vector = np.array(feature_vector)
        if np.linalg.norm(feature_vector) > 0:
            feature_vector = feature_vector / np.linalg.norm(feature_vector)

        return feature_vector.tolist()

    def _generate_cnn_features_pil(self, image: np.ndarray, features: Dict[str, Any]) -> List[float]:
        """Generate CNN-based features using PIL/numpy"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image

        feature_vector = []

        # Histogram features
        hist, _ = np.histogram(gray, bins=32, range=[0, 256])
        feature_vector.extend(hist.tolist())

        # Color moments for each channel
        for channel in range(image.shape[2]):
            ch = image[:,:,channel]
            feature_vector.extend([
                np.mean(ch),
                np.std(ch),
                np.mean(np.power(ch - np.mean(ch), 3))  # skewness
            ])

        # Texture features (simplified)
        for i in range(0, gray.shape[0], 32):
            for j in range(0, gray.shape[1], 32):
                patch = gray[i:i+32, j:j+32]
                if patch.size > 0:
                    feature_vector.extend([
                        np.mean(patch),
                        np.std(patch)
                    ])

        # Add extracted features
        feature_vector.extend([
            features.get('brightness', 0),
            features.get('avg_hue', 0),
            features.get('avg_saturation', 0),
            features.get('avg_value', 0)
        ])

        # Normalize to unit vector
        feature_vector = np.array(feature_vector)
        if np.linalg.norm(feature_vector) > 0:
            feature_vector = feature_vector / np.linalg.norm(feature_vector)

        return feature_vector.tolist()

    def _generate_basic_features_cv2(self, image: np.ndarray, features: Dict[str, Any]) -> List[float]:
        """Generate basic statistical features using OpenCV"""
        feature_vector = []

        # Basic color statistics
        for channel in cv2.split(image):
            feature_vector.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel)
            ])

        # Add extracted features
        feature_vector.extend([
            features.get('brightness', 0),
            features.get('avg_hue', 0),
            features.get('avg_saturation', 0),
            features.get('avg_value', 0)
        ])

        return feature_vector

    def _generate_basic_features_pil(self, image: np.ndarray, features: Dict[str, Any]) -> List[float]:
        """Generate basic statistical features using PIL/numpy"""
        feature_vector = []

        # Basic color statistics for each channel
        for channel in range(image.shape[2]):
            ch = image[:,:,channel]
            feature_vector.extend([
                np.mean(ch),
                np.std(ch),
                np.min(ch),
                np.max(ch)
            ])

        # Add extracted features
        feature_vector.extend([
            features.get('brightness', 0),
            features.get('avg_hue', 0),
            features.get('avg_saturation', 0),
            features.get('avg_value', 0)
        ])

        return feature_vector

def process_faces(face_files: List[str]) -> List[FaceData]:
    """Process downloaded faces: load metadata from JSON and generate embeddings"""
    embedder = FaceEmbedder()
    processed_faces = []

    for i, file_path in enumerate(face_files):
        try:
            logger.info(f"Processing face {i+1}/{len(face_files)}: {file_path}")

            # Generate JSON path from image path
            base_name = os.path.splitext(file_path)[0]
            json_path = f"{base_name}.json"

            # Load metadata from JSON if it exists
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                face_id = metadata.get('face_id', os.path.splitext(os.path.basename(file_path))[0])
                features = metadata.get('face_features', {})
                face_bbox = metadata.get('face_bbox')
                image_hash = metadata.get('md5_hash', '')

                logger.info(f"Loaded metadata from {os.path.basename(json_path)}")
            else:
                # Fallback: analyze if no JSON exists
                logger.warning(f"No JSON metadata found for {file_path}, analyzing now...")
                analyzer = FaceAnalyzer()
                face_id = os.path.splitext(os.path.basename(file_path))[0]
                face_bbox = analyzer.detect_face(file_path)
                features = analyzer.estimate_basic_features(file_path)
                features['face_bbox'] = face_bbox

                with open(file_path, 'rb') as f:
                    image_hash = hashlib.md5(f.read()).hexdigest()

            if not face_bbox:
                logger.warning(f"No face bbox in {file_path}")
                continue

            # Generate embedding
            embedding = embedder.generate_embedding(file_path, features)
            if not embedding:
                logger.warning(f"Could not generate embedding for {file_path}")
                continue

            # Create FaceData object with enhanced metadata
            face_data = FaceData(
                face_id=face_id,
                file_path=file_path,
                features=features,
                embedding=embedding,
                timestamp=datetime.now().isoformat(),
                image_hash=image_hash
            )

            processed_faces.append(face_data)
            logger.info(f"Successfully processed {face_id}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    return processed_faces

def save_face_data(face_data_list: List[FaceData], filename: str = "face_data.json"):
    """Save face data to JSON file"""
    data = [face.to_dict() for face in face_data_list]

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(face_data_list)} face records to {filename}")

def main():
    """Main function to demonstrate face collection and processing"""
    import argparse

    parser = argparse.ArgumentParser(description='Face Collection and Embedding System')
    parser.add_argument('--count', type=int, default=1000000,
                       help='Number of faces to download (default: 10)')
    args = parser.parse_args()

    print("🎭 Face Collection and Embedding System")
    print("="*50)

    # Download faces
    collector = FaceCollector(delay=2.0)  # Be respectful with requests

    print(f"📥 Downloading {args.count} synthetic faces...")
    face_files = collector.download_faces_batch(count=args.count, max_workers=2)

    if not face_files:
        print("❌ No faces downloaded. Check your internet connection.")
        return

    print(f"✅ Downloaded {len(face_files)} faces")

    # Process faces
    print("🔍 Processing faces and generating embeddings...")
    processed_faces = process_faces(face_files)

    if not processed_faces:
        print("❌ No faces processed successfully.")
        return

    print(f"✅ Processed {len(processed_faces)} faces")

    # Save data
    save_face_data(processed_faces)

    # Display summary
    print(f"\n📊 Summary:")
    print(f"Total faces processed: {len(processed_faces)}")

    embedding_dims = len(processed_faces[0].embedding) if processed_faces else 0
    print(f"Embedding dimensions: {embedding_dims}")

    # Show sample features
    if processed_faces:
        print(f"\n🔍 Sample face features:")
        for i, face in enumerate(processed_faces[:3]):
            print(f"\n{i+1}. Face ID: {face.face_id}")
            print(f"   File: {os.path.basename(face.file_path)}")
            print(f"   Features: {face.features}")
            print(f"   Embedding size: {len(face.embedding)}")

    return processed_faces

if __name__ == "__main__":
    main()
