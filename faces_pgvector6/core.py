#!/usr/bin/env python3
"""
Core Backend for Face Processing System

This module provides the core functionality for the face processing system including:
- Face image downloading from AI generation services
- Face feature analysis and demographic estimation
- Vector embedding generation using multiple models
- ChromaDB vector database management
- Face similarity search with metadata filtering

Architecture:
- FaceAnalyzer: Analyzes images for features and demographics
- FaceEmbedder: Creates vector embeddings using various models
- DatabaseManager: Manages ChromaDB operations
- FaceDownloader: Downloads AI-generated face images
- FaceProcessor: Processes faces and creates embeddings
- IntegratedFaceSystem: Main orchestrator class

Dependencies:
- ChromaDB for vector storage
- PIL/Pillow for image processing
- OpenCV (optional) for advanced analysis
- Various embedding libraries (FaceNet, ArcFace, DeepFace, etc.)
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

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

# Try to import OpenCV (optional for advanced analysis)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import ChromaDB (optional for legacy support)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Import pgvector database manager
try:
    from pgvector_db import PgVectorDatabaseManager
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    PgVectorDatabaseManager = None


def check_embedding_models():
    """Check availability of various embedding models"""
    models_status = {}

    # FaceNet (facenet-pytorch)
    try:
        from facenet_pytorch import InceptionResnetV1
        models_status['facenet'] = True
    except ImportError:
        models_status['facenet'] = False

    # ArcFace/InsightFace
    try:
        import insightface
        models_status['arcface'] = True
    except ImportError:
        models_status['arcface'] = False

    # DeepFace
    try:
        from deepface import DeepFace
        models_status['deepface'] = True
    except ImportError:
        models_status['deepface'] = False

    # VGGFace2 (via deepface or keras)
    try:
        from deepface import DeepFace
        models_status['vggface2'] = True
    except ImportError:
        models_status['vggface2'] = False

    # OpenFace (via deepface)
    try:
        from deepface import DeepFace
        models_status['openface'] = True
    except ImportError:
        models_status['openface'] = False

    # Statistical (always available)
    models_status['statistical'] = True

    return models_status

# Check models at module load
AVAILABLE_MODELS = check_embedding_models()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

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

    # Database type selection
    db_type: str = "pgvector"  # Options: chromadb, pgvector

    # ChromaDB settings (legacy)
    db_path: str = "./chroma_db"
    collection_name: str = "faces"

    # PostgreSQL + pgvector settings
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "vector_db"
    db_user: str = "postgres"
    db_password: str = ""

    # Application settings
    download_delay: float = 1.0
    max_workers: int = 2
    batch_size: int = 50
    embedding_model: str = "statistical"  # Options: statistical, facenet, arcface, deepface, vggface2, openface
    download_source: str = "thispersondoesnotexist"  # Options: thispersondoesnotexist, fakeface, randomface
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
    """
    System-wide Statistics Tracker

    Tracks comprehensive statistics for the face processing system including:
    - Download metrics (attempts, success, duplicates, errors)
    - Embedding metrics (processed, success, duplicates, errors)
    - Performance metrics (average times, rates)
    - Session statistics

    Thread-safe implementation for multi-threaded operations.
    """

    def __init__(self):
        self.download_attempts = 0
        self.download_success = 0
        self.download_duplicates = 0
        self.download_errors = 0
        self.embed_processed = 0
        self.embed_success = 0
        self.embed_duplicates = 0
        self.embed_errors = 0
        self.search_queries = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

        # Detailed timing statistics
        self.download_times = []  # Track individual download times
        self.embed_times = []     # Track individual embedding times
        self.last_download_time = None
        self.last_embed_time = None
        self.session_start_time = time.time()

        # Session statistics
        self.session_downloads = 0
        self.session_embeds = 0

    def increment_download_attempts(self):
        with self.lock:
            self.download_attempts += 1

    def increment_download_success(self, elapsed_time: float = None):
        with self.lock:
            self.download_success += 1
            self.session_downloads += 1
            if elapsed_time is not None:
                self.download_times.append(elapsed_time)
                # Keep only last 100 times for average calculation
                if len(self.download_times) > 100:
                    self.download_times.pop(0)
                self.last_download_time = elapsed_time

    def increment_download_duplicates(self):
        with self.lock:
            self.download_duplicates += 1

    def increment_download_errors(self):
        with self.lock:
            self.download_errors += 1

    def increment_embed_processed(self):
        with self.lock:
            self.embed_processed += 1

    def increment_embed_success(self, elapsed_time: float = None):
        with self.lock:
            self.embed_success += 1
            self.session_embeds += 1
            if elapsed_time is not None:
                self.embed_times.append(elapsed_time)
                # Keep only last 100 times for average calculation
                if len(self.embed_times) > 100:
                    self.embed_times.pop(0)
                self.last_embed_time = elapsed_time

    def increment_embed_duplicates(self):
        with self.lock:
            self.embed_duplicates += 1

    def increment_embed_errors(self):
        with self.lock:
            self.embed_errors += 1

    def increment_search_queries(self):
        with self.lock:
            self.search_queries += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = time.time() - self.start_time
            session_elapsed = time.time() - self.session_start_time

            # Calculate average download time
            avg_download_time = sum(self.download_times) / len(self.download_times) if self.download_times else 0
            # Calculate average embed time
            avg_embed_time = sum(self.embed_times) / len(self.embed_times) if self.embed_times else 0

            # Calculate success rates
            download_success_rate = (self.download_success / self.download_attempts * 100) if self.download_attempts > 0 else 0
            embed_success_rate = (self.embed_success / self.embed_processed * 100) if self.embed_processed > 0 else 0

            return {
                # Download statistics
                'download_attempts': self.download_attempts,
                'download_success': self.download_success,
                'download_duplicates': self.download_duplicates,
                'download_errors': self.download_errors,
                'download_rate': self.download_success / elapsed if elapsed > 0 else 0,
                'avg_download_time': avg_download_time,
                'last_download_time': self.last_download_time or 0,
                'download_success_rate': download_success_rate,
                'session_downloads': self.session_downloads,
                'session_download_rate': self.session_downloads / session_elapsed if session_elapsed > 0 else 0,

                # Embed statistics
                'embed_processed': self.embed_processed,
                'embed_success': self.embed_success,
                'embed_duplicates': self.embed_duplicates,
                'embed_errors': self.embed_errors,
                'embed_rate': self.embed_success / elapsed if elapsed > 0 else 0,
                'avg_embed_time': avg_embed_time,
                'last_embed_time': self.last_embed_time or 0,
                'embed_success_rate': embed_success_rate,
                'session_embeds': self.session_embeds,
                'session_embed_rate': self.session_embeds / session_elapsed if session_elapsed > 0 else 0,

                # General statistics
                'search_queries': self.search_queries,
                'elapsed_time': elapsed,
                'session_elapsed_time': session_elapsed,
            }

    def reset_session_stats(self):
        """Reset session statistics"""
        with self.lock:
            self.session_downloads = 0
            self.session_embeds = 0
            self.session_start_time = time.time()

# ============================================================================
# FACE ANALYSIS
# ============================================================================

class FaceAnalyzer:
    """
    Face Image Analysis Engine

    Analyzes face images to extract:
    - Basic image properties (dimensions, format, size)
    - Color analysis (brightness, contrast, saturation)
    - Face detection using Haar cascades
    - Demographic estimation (age, sex, skin tone, hair color)

    Uses OpenCV when available for advanced analysis,
    falls back to PIL/numpy for basic analysis.
    """

    def __init__(self):
        self.cv2_available = CV2_AVAILABLE
        if not self.cv2_available:
            logger.warning("OpenCV not available. Using basic image analysis.")

    def analyze_face(self, image_path: str) -> Dict[str, Any]:
        """Analyze face image and extract features including demographics"""
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

            # Advanced features with OpenCV or PIL
            if self.cv2_available:
                cv_image = cv2.imread(image_path)
                if cv_image is not None:
                    # Color analysis
                    features.update(self._analyze_colors(cv_image))
                    # Face detection
                    face_info = self._detect_faces(cv_image)
                    features.update(face_info)
                    # Demographic analysis
                    features.update(self._analyze_demographics(cv_image, face_info))
            else:
                # Fallback to PIL-based analysis
                img_array = np.array(image)
                features.update(self._analyze_colors_pil(img_array))
                features.update(self._analyze_demographics_pil(img_array))

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

    def _analyze_colors_pil(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze colors using PIL/numpy"""
        try:
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array

            return {
                'brightness': float(np.mean(gray)),
                'contrast': float(np.std(gray)),
                'saturation_mean': float(np.std(img_array))
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

    def _analyze_demographics(self, cv_image, face_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze demographic features: age, sex, skin tone, hair color"""
        demographics = {}

        try:
            h, w = cv_image.shape[:2]

            # Get face region or use center of image
            if face_info.get('faces_detected', 0) > 0 and len(face_info.get('face_regions', [])) > 0:
                x, y, fw, fh = face_info['face_regions'][0]
                face_region = cv_image[y:y+fh, x:x+fw]
            else:
                # Use center region as fallback
                cy, cx = h//2, w//2
                size = min(h, w) // 2
                y1, y2 = max(0, cy-size//2), min(h, cy+size//2)
                x1, x2 = max(0, cx-size//2), min(w, cx+size//2)
                face_region = cv_image[y1:y2, x1:x2]

            if face_region.size == 0:
                face_region = cv_image

            # Convert to different color spaces for analysis
            hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            lab_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)

            # Skin tone analysis (from face center region)
            fh, fw = face_region.shape[:2]
            skin_y1, skin_y2 = fh//3, 2*fh//3
            skin_x1, skin_x2 = fw//3, 2*fw//3
            skin_sample = face_region[skin_y1:skin_y2, skin_x1:skin_x2]

            skin_tone = self._estimate_skin_tone(skin_sample)
            demographics['skin_tone'] = skin_tone
            demographics['skin_color'] = self._categorize_skin_color(skin_tone)

            # Hair color analysis (top portion of image)
            hair_region = cv_image[0:h//4, :]
            hair_color = self._estimate_hair_color(hair_region)
            demographics['hair_color'] = hair_color

            # Age estimation (based on image characteristics)
            age_group = self._estimate_age_group(face_region)
            demographics['age_group'] = age_group
            demographics['estimated_age'] = self._age_group_to_range(age_group)

            # Sex estimation (based on facial features and colors)
            sex = self._estimate_sex(face_region, hair_color, skin_tone)
            demographics['estimated_sex'] = sex

        except Exception as e:
            logger.error(f"Error in demographic analysis: {e}")
            demographics['demographic_error'] = str(e)

        return demographics

    def _analyze_demographics_pil(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze demographics using PIL/numpy (fallback)"""
        demographics = {}

        try:
            h, w = img_array.shape[:2] if len(img_array.shape) > 1 else (img_array.shape[0], 1)

            # Use center region
            cy, cx = h//2, w//2 if w > 1 else 0
            size = min(h, w) // 2 if w > 1 else h // 2

            if len(img_array.shape) == 3:
                y1, y2 = max(0, cy-size//2), min(h, cy+size//2)
                x1, x2 = max(0, cx-size//2), min(w, cx+size//2)
                face_region = img_array[y1:y2, x1:x2]

                # Simple skin tone from RGB
                avg_color = np.mean(face_region, axis=(0, 1))
                brightness = np.mean(avg_color)

                if brightness > 200:
                    demographics['skin_tone'] = 'very_light'
                    demographics['skin_color'] = 'light'
                elif brightness > 160:
                    demographics['skin_tone'] = 'light'
                    demographics['skin_color'] = 'light'
                elif brightness > 120:
                    demographics['skin_tone'] = 'medium'
                    demographics['skin_color'] = 'medium'
                elif brightness > 80:
                    demographics['skin_tone'] = 'tan'
                    demographics['skin_color'] = 'medium'
                else:
                    demographics['skin_tone'] = 'dark'
                    demographics['skin_color'] = 'dark'

                # Hair color from top
                hair_region = img_array[0:h//4, :]
                hair_avg = np.mean(hair_region, axis=(0, 1))
                demographics['hair_color'] = self._rgb_to_hair_color(hair_avg)

                # Simple age/sex estimates
                demographics['age_group'] = 'adult'
                demographics['estimated_age'] = '25-35'
                demographics['estimated_sex'] = 'unknown'

        except Exception as e:
            logger.error(f"Error in PIL demographic analysis: {e}")

        return demographics

    def _estimate_skin_tone(self, skin_sample) -> str:
        """Estimate skin tone from skin sample"""
        try:
            lab = cv2.cvtColor(skin_sample, cv2.COLOR_BGR2LAB)
            l_mean = np.mean(lab[:, :, 0])  # Lightness channel

            # Fitzpatrick-inspired scale
            if l_mean > 200:
                return 'very_light'
            elif l_mean > 170:
                return 'light'
            elif l_mean > 140:
                return 'medium'
            elif l_mean > 110:
                return 'tan'
            elif l_mean > 80:
                return 'brown'
            else:
                return 'dark'
        except:
            return 'unknown'

    def _categorize_skin_color(self, skin_tone: str) -> str:
        """Categorize skin tone into broad categories"""
        tone_map = {
            'very_light': 'light',
            'light': 'light',
            'medium': 'medium',
            'tan': 'medium',
            'brown': 'dark',
            'dark': 'dark'
        }
        return tone_map.get(skin_tone, 'unknown')

    def _estimate_hair_color(self, hair_region) -> str:
        """Estimate hair color from top region"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            v_mean = np.mean(hsv[:, :, 2])

            # Black/dark hair
            if v_mean < 50:
                return 'black'
            # Gray/white hair
            elif s_mean < 30 and v_mean > 150:
                return 'gray'
            elif s_mean < 30 and v_mean > 100:
                return 'light_gray'
            # Brown hair
            elif h_mean < 30 and v_mean < 150:
                return 'brown'
            elif h_mean < 30 and v_mean < 100:
                return 'dark_brown'
            # Blonde hair
            elif h_mean < 40 and v_mean > 150:
                return 'blonde'
            # Red hair
            elif h_mean < 20 or h_mean > 340:
                return 'red'
            # Other colors
            else:
                return 'other'

        except:
            return 'unknown'

    def _rgb_to_hair_color(self, rgb_avg) -> str:
        """Convert RGB average to hair color (fallback)"""
        r, g, b = rgb_avg
        brightness = np.mean(rgb_avg)

        if brightness < 50:
            return 'black'
        elif brightness > 200:
            return 'blonde'
        elif r > g and r > b:
            return 'red'
        elif brightness < 100:
            return 'dark_brown'
        else:
            return 'brown'

    def _estimate_age_group(self, face_region) -> str:
        """Estimate age group from facial characteristics"""
        try:
            # Analyze texture and smoothness
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # Calculate variance (smoother = younger)
            variance = np.var(gray)

            # Calculate edge density (more edges = older)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size

            # Simple heuristic
            if variance < 300 and edge_density < 0.05:
                return 'child'
            elif variance < 600 and edge_density < 0.08:
                return 'young_adult'
            elif variance < 900 and edge_density < 0.12:
                return 'adult'
            elif variance < 1200 and edge_density < 0.15:
                return 'middle_aged'
            else:
                return 'senior'

        except:
            return 'adult'

    def _age_group_to_range(self, age_group: str) -> str:
        """Convert age group to age range"""
        age_ranges = {
            'child': '0-12',
            'young_adult': '18-25',
            'adult': '25-40',
            'middle_aged': '40-60',
            'senior': '60+'
        }
        return age_ranges.get(age_group, '25-40')

    def _estimate_sex(self, face_region, hair_color: str, skin_tone: str) -> str:
        """Estimate biological sex from facial features"""
        try:
            # Analyze facial structure and features
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # Calculate aspect ratio (males tend to have wider faces)
            h, w = gray.shape
            aspect_ratio = w / h if h > 0 else 1.0

            # Analyze jawline sharpness (males tend to have sharper jawlines)
            bottom_third = gray[2*h//3:, :]
            jaw_variance = np.var(bottom_third)

            # Analyze skin smoothness (females tend to have smoother skin)
            skin_variance = np.var(gray)

            # Simple heuristic combination
            male_score = 0
            female_score = 0

            if aspect_ratio > 0.85:
                male_score += 1
            else:
                female_score += 1

            if jaw_variance > 500:
                male_score += 1
            else:
                female_score += 1

            if skin_variance < 600:
                female_score += 1
            else:
                male_score += 1

            if male_score > female_score:
                return 'male'
            elif female_score > male_score:
                return 'female'
            else:
                return 'unknown'

        except:
            return 'unknown'

# ============================================================================
# VECTOR EMBEDDING
# ============================================================================

class FaceEmbedder:
    """
    Face Embedding Generator

    Creates vector embeddings from face images using various models:
    - Statistical: Basic statistical features (always available)
    - FaceNet: Deep learning model via facenet-pytorch
    - ArcFace: State-of-the-art via InsightFace
    - DeepFace: Multi-purpose deep learning
    - VGGFace2: Deep CNN model
    - OpenFace: Lightweight deep learning

    Automatically falls back to statistical model if specified model unavailable.
    """

    def __init__(self, model_name: str = "statistical"):
        self.model_name = model_name
        self.model = None
        self.embedding_size = 512  # Default size

        # Initialize the selected model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the selected embedding model"""
        try:
            if self.model_name == "facenet":
                self._init_facenet()
            elif self.model_name == "arcface":
                self._init_arcface()
            elif self.model_name == "deepface":
                self._init_deepface()
            elif self.model_name == "vggface2":
                self._init_vggface2()
            elif self.model_name == "openface":
                self._init_openface()
            elif self.model_name == "statistical":
                self.embedding_size = 512
                logger.info("Using statistical embedding (default)")
            else:
                logger.warning(f"Unknown model '{self.model_name}', falling back to statistical")
                self.model_name = "statistical"
                self.embedding_size = 512
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_name}, falling back to statistical: {e}")
            self.model_name = "statistical"
            self.embedding_size = 512

    def _init_facenet(self):
        """Initialize FaceNet model"""
        try:
            from facenet_pytorch import InceptionResnetV1
            import torch

            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            self.embedding_size = 512
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            logger.info("FaceNet model initialized successfully")
        except ImportError:
            raise ImportError("FaceNet requires: pip install facenet-pytorch torch torchvision")

    def _init_arcface(self):
        """Initialize ArcFace/InsightFace model"""
        try:
            import insightface
            from insightface.app import FaceAnalysis

            self.model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            self.embedding_size = 512
            logger.info("ArcFace model initialized successfully")
        except ImportError:
            raise ImportError("ArcFace requires: pip install insightface onnxruntime")

    def _init_deepface(self):
        """Initialize DeepFace model"""
        try:
            from deepface import DeepFace
            # DeepFace will auto-download models on first use
            self.model = "DeepFace"  # Use DeepFace backend
            self.embedding_size = 4096  # DeepFace default
            logger.info("DeepFace model initialized successfully")
        except ImportError:
            raise ImportError("DeepFace requires: pip install deepface")

    def _init_vggface2(self):
        """Initialize VGGFace2 model"""
        try:
            from deepface import DeepFace
            self.model = "VGG-Face"  # Use VGG-Face backend
            self.embedding_size = 2622  # VGGFace default
            logger.info("VGGFace2 model initialized successfully")
        except ImportError:
            raise ImportError("VGGFace2 requires: pip install deepface")

    def _init_openface(self):
        """Initialize OpenFace model"""
        try:
            from deepface import DeepFace
            self.model = "OpenFace"  # Use OpenFace backend
            self.embedding_size = 128  # OpenFace default
            logger.info("OpenFace model initialized successfully")
        except ImportError:
            raise ImportError("OpenFace requires: pip install deepface")

    def create_embedding(self, image_path: str, features: Dict[str, Any]) -> List[float]:
        """Create embedding vector for face image"""
        try:
            if self.model_name == "facenet":
                return self._embed_facenet(image_path)
            elif self.model_name == "arcface":
                return self._embed_arcface(image_path)
            elif self.model_name in ["deepface", "vggface2", "openface"]:
                return self._embed_deepface(image_path)
            else:
                # Statistical embedding (default)
                image = Image.open(image_path)
                img_array = np.array(image)
                embedding = self._create_statistical_embedding(img_array, features)
                return embedding.tolist()

        except Exception as e:
            logger.error(f"Error creating embedding for {image_path}: {e}")
            return [0.0] * self.embedding_size

    def _embed_facenet(self, image_path: str) -> List[float]:
        """Create embedding using FaceNet"""
        try:
            from facenet_pytorch import MTCNN
            import torch
            from PIL import Image

            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')

            # Detect face
            mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
            img_cropped = mtcnn(img)

            if img_cropped is None:
                # If no face detected, use whole image
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                img_cropped = transform(img).unsqueeze(0).to(self.device)
            else:
                img_cropped = img_cropped.unsqueeze(0).to(self.device)

            # Generate embedding
            with torch.no_grad():
                embedding = self.model(img_cropped)

            return embedding.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"FaceNet embedding error: {e}")
            return [0.0] * self.embedding_size

    def _embed_arcface(self, image_path: str) -> List[float]:
        """Create embedding using ArcFace/InsightFace"""
        try:
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to load image")

            faces = self.model.get(img)

            if len(faces) > 0:
                # Use first face
                embedding = faces[0].embedding
                return embedding.tolist()
            else:
                logger.warning(f"No face detected in {image_path}, returning zero embedding")
                return [0.0] * self.embedding_size
        except Exception as e:
            logger.error(f"ArcFace embedding error: {e}")
            return [0.0] * self.embedding_size

    def _embed_deepface(self, image_path: str) -> List[float]:
        """Create embedding using DeepFace (supports multiple backends)"""
        try:
            from deepface import DeepFace

            # DeepFace.represent returns embeddings
            result = DeepFace.represent(
                img_path=image_path,
                model_name=self.model,
                enforce_detection=False
            )

            if isinstance(result, list) and len(result) > 0:
                embedding = result[0]["embedding"]
                return embedding
            else:
                return [0.0] * self.embedding_size
        except Exception as e:
            logger.error(f"DeepFace embedding error: {e}")
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

# ============================================================================
# DATABASE MANAGEMENT
# ============================================================================

class DatabaseManager:
    """
    ChromaDB Vector Database Manager

    Manages all database operations including:
    - Database initialization and connection
    - Adding face embeddings with metadata
    - Vector similarity search
    - Metadata filtering and hybrid search
    - Duplicate detection via image hashing
    - Model mismatch detection

    Uses persistent storage with ChromaDB for vector operations.
    """

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

    def add_face(self, face_data: FaceData, embedding_model: str = "statistical") -> bool:
        """Add face data to the database"""
        if not self.initialized:
            logger.warning("Database not initialized, cannot add face")
            return False

        try:
            # Prepare metadata - ChromaDB only accepts str, int, float, bool, or None
            metadata = {
                'face_id': face_data.face_id,
                'file_path': face_data.file_path,
                'timestamp': face_data.timestamp,
                'image_hash': face_data.image_hash,
                'embedding_model': embedding_model,  # Track which model created this embedding
            }

            # Filter features to only include valid metadata types
            valid_types = (str, int, float, bool, type(None))
            for key, value in face_data.features.items():
                if isinstance(value, valid_types):
                    metadata[key] = value
                elif isinstance(value, (list, tuple)):
                    # Convert lists/tuples to JSON strings
                    metadata[f"{key}_json"] = json.dumps(value)
                elif isinstance(value, dict):
                    # Convert dicts to JSON strings
                    metadata[f"{key}_json"] = json.dumps(value)
                else:
                    # Convert other types to string
                    metadata[key] = str(value)

            # Add to collection
            self.collection.add(
                embeddings=[face_data.embedding],
                metadatas=[metadata],
                ids=[face_data.face_id]
            )

            logger.debug(f"Successfully added face {face_data.face_id} to database")
            return True

        except Exception as e:
            logger.error(f"Error adding face {face_data.face_id} to database: {e}")
            logger.debug(f"Problematic metadata keys: {list(face_data.features.keys())}")
            return False

    def search_faces(self, query_embedding: List[float], n_results: int = 10,
                    metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar faces with optional metadata filtering"""
        if not self.initialized:
            return []

        try:
            # Build query parameters
            query_params = {
                'query_embeddings': [query_embedding],
                'n_results': n_results
            }

            # Add metadata filter if provided (for hybrid search)
            if metadata_filter:
                query_params['where'] = metadata_filter

            results = self.collection.query(**query_params)

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

    def search_by_metadata(self, metadata_filter: Dict[str, Any], n_results: int = 10) -> List[Dict[str, Any]]:
        """Search faces by metadata only (no vector similarity)"""
        if not self.initialized:
            return []

        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=n_results
            )

            return [
                {
                    'id': results['ids'][i],
                    'metadata': results['metadatas'][i]
                }
                for i in range(len(results['ids']))
            ]

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []

    def hybrid_search(self, query_embedding: List[float], metadata_filter: Dict[str, Any],
                     n_results: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search combining vector similarity and metadata filtering"""
        return self.search_faces(query_embedding, n_results, metadata_filter)

    def hash_exists(self, image_hash: str) -> bool:
        """Check if an image with this hash already exists in the database"""
        if not self.initialized:
            return False

        try:
            results = self.collection.get(
                where={"image_hash": image_hash},
                limit=1
            )
            return len(results['ids']) > 0
        except Exception as e:
            logger.error(f"Error checking hash existence: {e}")
            return False

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

    def check_embedding_model_mismatch(self, current_model: str) -> Dict[str, Any]:
        """Check if database has embeddings from different models"""
        if not self.initialized:
            return {'has_mismatch': False, 'models_found': [], 'total_count': 0}

        try:
            # Get all data to check embedding models
            all_data = self.collection.get(limit=10000)  # Get up to 10k records

            if not all_data or not all_data.get('metadatas'):
                return {'has_mismatch': False, 'models_found': [], 'total_count': 0}

            # Count embedding models
            model_counts = {}
            total = len(all_data['metadatas'])

            for metadata in all_data['metadatas']:
                model = metadata.get('embedding_model', 'unknown')
                model_counts[model] = model_counts.get(model, 0) + 1

            # Check for mismatch
            has_mismatch = False
            if current_model not in model_counts:
                # Current model not in database at all
                has_mismatch = len(model_counts) > 0
            elif len(model_counts) > 1:
                # Multiple models in database
                has_mismatch = True
            elif model_counts.get(current_model, 0) != total:
                # Some entries don't have current model
                has_mismatch = True

            return {
                'has_mismatch': has_mismatch,
                'models_found': model_counts,
                'total_count': total,
                'current_model': current_model
            }
        except Exception as e:
            logger.error(f"Error checking model mismatch: {e}")
            return {'has_mismatch': False, 'models_found': {}, 'total_count': 0}

    def clear_all_data(self) -> bool:
        """Clear all data from the collection"""
        if not self.initialized:
            return False

        try:
            # Delete the collection
            self.client.delete_collection(name=self.config.collection_name)

            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"description": "Face embeddings collection"}
            )

            logger.info("All data cleared from collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

# ============================================================================
# FACE IMAGE DOWNLOADING
# ============================================================================

class FaceDownloader:
    """
    AI-Generated Face Image Downloader

    Downloads face images from AI generation services:
    - ThisPersonDoesNotExist.com (high quality 1024x1024)
    - 100K Faces API (generated.photos dataset)

    Features:
    - Automatic duplicate detection via MD5 hashing
    - Comprehensive metadata generation and storage
    - Configurable download delays
    - Background download loop support
    """

    # Available download sources
    DOWNLOAD_SOURCES = {
        'thispersondoesnotexist': {
            'name': 'ThisPersonDoesNotExist.com',
            'url': 'https://thispersondoesnotexist.com/',
            'description': 'High quality AI-generated faces (1024x1024)',
        },
        '100k-faces': {
            'name': '100K AI Faces',
            'url': 'https://100k-faces.vercel.app/api/random-image',
            'description': '100K AI-generated faces from generated.photos',
        }
    }

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
        """Download a single face image with metadata JSON"""
        self.stats.increment_download_attempts()
        download_start = time.time()

        try:
            # Download image based on selected source
            download_time = datetime.now()
            source = self.config.download_source

            if source == 'thispersondoesnotexist':
                response = self._download_from_thispersondoesnotexist()
            elif source == '100k-faces':
                response = self._download_from_100k_faces()
            else:
                logger.warning(f"Unknown download source: {source}, using default")
                response = self._download_from_thispersondoesnotexist()

            if response is None:
                self.stats.increment_download_errors()
                return None

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

            # Analyze image and create metadata
            try:
                # Analyze the downloaded image
                analyzer = FaceAnalyzer()
                features = analyzer.analyze_face(file_path)

                # Get image properties
                with Image.open(file_path) as img:
                    image_width, image_height = img.size
                    image_format = img.format
                    image_mode = img.mode

                # Create comprehensive metadata
                metadata = {
                    # Basic identifiers
                    'filename': filename,
                    'file_path': file_path,
                    'face_id': timestamp,
                    'md5_hash': image_hash,

                    # Download info
                    'download_timestamp': download_time.isoformat(),
                    'download_date': download_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'source_url': "https://thispersondoesnotexist.com/",
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

                    # Face features from analyzer (IMPORTANT FOR QUERYING)
                    'face_features': features,

                    # Queryable attributes extracted from features
                    'queryable_attributes': {
                        'brightness_level': 'bright' if features.get('brightness', 0) > 150 else 'dark',
                        'image_quality': 'high' if features.get('contrast', 0) > 50 else 'medium',
                        'has_face': features.get('faces_detected', 0) > 0,
                        'face_count': features.get('faces_detected', 0),
                        # Demographic attributes
                        'sex': features.get('estimated_sex', 'unknown'),
                        'age_group': features.get('age_group', 'unknown'),
                        'estimated_age': features.get('estimated_age', 'unknown'),
                        'skin_tone': features.get('skin_tone', 'unknown'),
                        'skin_color': features.get('skin_color', 'unknown'),
                        'hair_color': features.get('hair_color', 'unknown')
                    },

                    # HTTP headers for debugging
                    'http_headers': dict(response.headers),

                    # Downloader config
                    'downloader_config': {
                        'storage_dir': self.config.faces_dir,
                        'delay': self.config.download_delay
                    }
                }

                # Save metadata JSON alongside image
                json_filename = f"face_{timestamp}_{image_hash[:8]}.json"
                json_path = os.path.join(self.config.faces_dir, json_filename)

                with open(json_path, 'w') as json_file:
                    json.dump(metadata, json_file, indent=2, default=self._json_numpy_fallback)

                logger.info(f"Saved metadata to: {json_filename}")

            except Exception as meta_error:
                logger.warning(f"Failed to save metadata for {filename}: {meta_error}")

            # Add to downloaded hashes
            self.downloaded_hashes.add(image_hash)

            # Calculate download time
            download_elapsed = time.time() - download_start
            self.stats.increment_download_success(download_elapsed)
            logger.info(f"Downloaded: {filename} ({download_elapsed:.2f}s)")
            return file_path

        except Exception as e:
            self.stats.increment_download_errors()
            logger.error(f"Download error: {e}")
            return None

    def _download_from_thispersondoesnotexist(self) -> Optional[requests.Response]:
        """Download from thispersondoesnotexist.com"""
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/",
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=30
            )
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error downloading from thispersondoesnotexist: {e}")
            return None

    def _download_from_100k_faces(self) -> Optional[requests.Response]:
        """Download from 100k-faces.vercel.app API"""
        try:
            # This API redirects to a random face image
            response = requests.get(
                "https://100k-faces.vercel.app/api/random-image",
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=30,
                allow_redirects=True
            )
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error downloading from 100k-faces: {e}")
            return None

    def _json_numpy_fallback(self, obj):
        """Fallback for JSON serialization of numpy types"""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

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

# ============================================================================
# FACE PROCESSING
# ============================================================================

class FaceProcessor:
    """
    Face Image Processor

    Processes face images to create and store embeddings:
    - Analyzes face images for features
    - Creates vector embeddings using configured model
    - Stores embeddings in database with metadata
    - Tracks processed files to avoid duplicates
    - Supports batch processing

    Works in conjunction with FaceAnalyzer, FaceEmbedder, and DatabaseManager.
    """

    def __init__(self, config: SystemConfig, stats: SystemStats, db_manager: DatabaseManager):
        self.config = config
        self.stats = stats
        self.db_manager = db_manager
        self.analyzer = FaceAnalyzer()
        self.embedder = FaceEmbedder(model_name=config.embedding_model)
        self.processed_files = set()

    def process_face_file(self, file_path: str, callback=None) -> bool:
        """Process a single face file"""
        if file_path in self.processed_files:
            return True

        self.stats.increment_embed_processed()
        embed_start = time.time()

        try:
            # Calculate hash first to check for duplicates
            file_hash = self._get_file_hash(file_path)

            # Check if duplicate exists in database
            if self.db_manager.hash_exists(file_hash):
                self.stats.increment_embed_duplicates()
                logger.info(f"Skipping duplicate: {os.path.basename(file_path)} (hash: {file_hash[:8]})")
                return True  # Skip but don't count as error

            # Analyze face
            features = self.analyzer.analyze_face(file_path)
            if 'error' in features:
                self.stats.increment_embed_errors()
                return False

            # Create embedding
            embedding = self.embedder.create_embedding(file_path, features)

            # Create face data
            face_id = f"face_{int(time.time())}_{file_hash[:8]}"

            face_data = FaceData(
                face_id=face_id,
                file_path=file_path,
                features=features,
                embedding=embedding,
                timestamp=datetime.now().isoformat(),
                image_hash=file_hash
            )

            # Add to database with embedding model info
            if self.db_manager.add_face(face_data, embedding_model=self.config.embedding_model):
                self.processed_files.add(file_path)

                # Calculate embedding time
                embed_elapsed = time.time() - embed_start
                self.stats.increment_embed_success(embed_elapsed)
                logger.info(f"Processed: {os.path.basename(file_path)} ({embed_elapsed:.2f}s)")

                # Call callback with detailed face data
                if callback:
                    callback(face_data)
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

    def get_new_files_only(self) -> List[str]:
        """Get list of files that are NOT in the database yet"""
        # Get all face files
        all_face_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            all_face_files.extend(Path(self.config.faces_dir).rglob(ext))

        # Filter to only new files (not in database)
        new_files = []
        for file_path in all_face_files:
            file_path_str = str(file_path)

            # Skip if already processed in this session
            if file_path_str in self.processed_files:
                continue

            # Calculate hash and check if in database
            file_hash = self._get_file_hash(file_path_str)
            if not self.db_manager.hash_exists(file_hash):
                new_files.append(file_path_str)

        logger.info(f"Found {len(new_files)} new files out of {len(all_face_files)} total files")
        return new_files

    def process_new_faces_only(self, callback=None) -> Dict[str, int]:
        """Process only new faces (not in database)"""
        new_files = self.get_new_files_only()

        stats = {
            'total_files': len(new_files),
            'processed': 0,
            'skipped': 0,
            'errors': 0
        }

        for file_path in new_files:
            if self.process_face_file(file_path, callback=callback):
                stats['processed'] += 1
            else:
                stats['errors'] += 1

        logger.info(f"Processed {stats['processed']} new files, {stats['errors']} errors")
        return stats

    def process_all_faces(self, callback=None):
        """Process all faces in the faces directory"""
        face_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            face_files.extend(Path(self.config.faces_dir).rglob(ext))

        for file_path in face_files:
            # Pass callback to process_face_file - it will call it with FaceData
            self.process_face_file(str(file_path), callback=callback)

# ============================================================================
# INTEGRATED SYSTEM
# ============================================================================

class IntegratedFaceSystem:
    """
    Integrated Face Processing System

    Main orchestrator class that integrates all system components:
    - Configuration management
    - Statistics tracking
    - Database management
    - Face downloading
    - Face processing and embedding

    Provides a unified interface for the entire face processing workflow.

    Usage:
        system = IntegratedFaceSystem()
        system.initialize()
        system.downloader.download_face()
        system.processor.process_all_faces()
        results = system.db_manager.search_faces(embedding)
    """

    def __init__(self, config_file: str = "system_config.json"):
        self.config = SystemConfig.from_file(config_file)
        self.stats = SystemStats()

        # Select database manager based on configuration
        if self.config.db_type == "pgvector":
            if not PGVECTOR_AVAILABLE:
                logger.error("pgvector database selected but not available. Install: pip install psycopg2-binary")
                raise ImportError("PgVectorDatabaseManager not available")
            self.db_manager = PgVectorDatabaseManager(self.config)
            logger.info("Using PostgreSQL + pgvector database")
        else:  # Default to ChromaDB
            if not CHROMADB_AVAILABLE:
                logger.error("ChromaDB not available. Install: pip install chromadb")
                raise ImportError("ChromaDB not available")
            self.db_manager = DatabaseManager(self.config)
            logger.info("Using ChromaDB database")

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