#!/usr/bin/env python3
"""
Core Backend for Image Processing System

This module provides the core functionality for the image processing system including:
- Image downloading from AI generation services
- Image feature analysis and demographic estimation
- Vector embedding generation using multiple models
- PostgreSQL + pgvector database management
- Image similarity search with metadata filtering

Architecture:
- ImageAnalyzer: Analyzes images for features and demographics
- ImageEmbedder: Creates vector embeddings using various models
- ImageDownloader: Downloads AI-generated images
- ImageProcessor: Processes images and creates embeddings
- IntegratedImageSystem: Main orchestrator class

Dependencies:
- PostgreSQL with pgvector extension for vector storage
- PIL/Pillow for image processing
- OpenCV (optional) for advanced analysis
- Various embedding libraries (CLIP, YOLO, etc.)
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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

# Try to import OpenCV (optional for advanced analysis)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import pgvector database manager
from pgvector_db import PgVectorDatabaseManager


def check_embedding_models():
    """Check availability of various embedding models"""
    models_status = {}

    # CLIP
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        models_status['clip'] = True
    except Exception:
        models_status['clip'] = False

    # YOLO
    try:
        import torch
        models_status['yolo'] = True
    except Exception:
        models_status['yolo'] = False


    # ResNet
    try:
        import torch
        import torchvision
        models_status['resnet'] = True
    except Exception:
        models_status['resnet'] = False

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
class ImageData:
    """Data class for image information"""
    image_id: str
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
    images_dir: str = os.getenv("IMAGES_DIR", "./images")

    # PostgreSQL + pgvector settings
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "image_vector")
    db_user: str = os.getenv("DB_USER", "pi")
    db_password: str = os.getenv("DB_PASSWORD", "")

    # Application settings
    download_delay: float = float(os.getenv("DOWNLOAD_DELAY", "1.0"))
    max_workers: int = 2
    batch_size: int = 50
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "statistical")  # Options: statistical, clip, yolo
    download_source: str = os.getenv("DOWNLOAD_SOURCE", "picsum_landscape")  # Options: see ImageDownloader.DOWNLOAD_SOURCES
    config_file: str = "system_config.json"

    @classmethod
    def from_file(cls, filename: str = "system_config.json") -> 'SystemConfig':
        """Load configuration from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                # Remove legacy fields that are no longer in SystemConfig
                data.pop('db_type', None)
                data.pop('db_path', None)
                data.pop('collection_name', None)
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

    Tracks comprehensive statistics for the image processing system including:
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
# IMAGE ANALYSIS
# ============================================================================

class ImageAnalyzer:
    """
    Image Analysis Engine

    Analyzes images to extract:
    - Basic image properties (dimensions, format, size)
    - Color analysis (brightness, contrast, saturation)
    - Quality metrics (sharpness, noise, edge density)
    - Advanced features using OpenCV when available

    Uses OpenCV when available for advanced analysis,
    falls back to PIL/numpy for basic analysis.
    """

    def __init__(self):
        self.cv2_available = CV2_AVAILABLE
        if not self.cv2_available:
            logger.warning("OpenCV not available. Using basic image analysis.")

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image and extract features"""
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

            # Extract EXIF metadata
            exif_data = self._extract_exif_data(image)
            features.update(exif_data)

            # Extract file metadata
            file_metadata = self._extract_file_metadata(image_path)
            features.update(file_metadata)

            # Calculate image quality metrics
            quality_metrics = self._calculate_quality_metrics(image)
            features.update(quality_metrics)

            # Advanced features with OpenCV or PIL
            if self.cv2_available:
                cv_image = cv2.imread(image_path)
                if cv_image is not None:
                    # Color analysis
                    features.update(self._analyze_colors(cv_image))

            else:
                # Fallback to PIL-based analysis
                img_array = np.array(image)
                features.update(self._analyze_colors_pil(img_array))
                features.update(self._analyze_demographics_pil(img_array))

            # Convert all values to JSON-serializable format
            features = self._make_json_serializable(features)

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



    def _make_json_serializable(self, value):
        """Convert non-JSON-serializable types to serializable formats"""
        # Handle None
        if value is None:
            return None

        # Handle numpy types
        if hasattr(value, 'item'):
            # numpy scalar (np.int64, np.float64, np.bool_, etc.)
            return value.item()

        # Handle standard Python types (already JSON serializable)
        if isinstance(value, (int, float, str, bool)):
            return value

        # Handle bytes
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8', errors='ignore')
            except:
                return str(value)

        # Handle PIL IFDRational, TiffImagePlugin.IFDRational
        if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
            # Convert IFDRational to float
            try:
                return float(value)
            except:
                return str(value)

        # Handle dict
        if isinstance(value, dict):
            return {k: self._make_json_serializable(v) for k, v in value.items()}

        # Handle lists/tuples/iterables (but not strings)
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            try:
                # Try to convert to list
                return [self._make_json_serializable(v) for v in value]
            except:
                return str(value)

        # Convert everything else to string
        return str(value)

    def _extract_exif_data(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract EXIF metadata from image

        Returns comprehensive camera, GPS, and shooting information
        """
        exif_info = {}

        try:
            # Get EXIF data
            exif = image.getexif()

            if exif:
                # Import EXIF tags
                from PIL.ExifTags import TAGS, GPSTAGS

                # Extract common EXIF fields
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, tag_id)

                    # Convert to JSON-serializable format
                    value = self._make_json_serializable(value)

                    # Store useful metadata
                    if tag_name in ['Make', 'Model', 'Software', 'DateTime',
                                   'DateTimeOriginal', 'DateTimeDigitized',
                                   'Orientation', 'XResolution', 'YResolution',
                                   'ResolutionUnit', 'ExposureTime', 'FNumber',
                                   'ISO', 'ISOSpeedRatings', 'Flash', 'FocalLength',
                                   'WhiteBalance', 'ExposureMode', 'ColorSpace']:
                        exif_info[f'exif_{tag_name.lower()}'] = value

                # GPS data extraction
                gps_data = exif.get_ifd(0x8825)  # GPS IFD
                if gps_data:
                    gps_info = {}
                    for tag_id, value in gps_data.items():
                        tag_name = GPSTAGS.get(tag_id, tag_id)
                        gps_info[tag_name] = value

                    # Convert GPS coordinates if available
                    if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                        lat = self._convert_gps_to_decimal(
                            gps_info['GPSLatitude'],
                            gps_info.get('GPSLatitudeRef', 'N')
                        )
                        lon = self._convert_gps_to_decimal(
                            gps_info['GPSLongitude'],
                            gps_info.get('GPSLongitudeRef', 'E')
                        )
                        exif_info['gps_latitude'] = lat
                        exif_info['gps_longitude'] = lon
                        exif_info['has_gps'] = True
                    else:
                        exif_info['has_gps'] = False
                else:
                    exif_info['has_gps'] = False

                exif_info['has_exif'] = True
            else:
                exif_info['has_exif'] = False

        except Exception as e:
            logger.debug(f"EXIF extraction error: {e}")
            exif_info['has_exif'] = False
            exif_info['exif_error'] = str(e)

        return exif_info

    def _convert_gps_to_decimal(self, coord_tuple, ref):
        """Convert GPS coordinates from degrees/minutes/seconds to decimal"""
        try:
            degrees = float(coord_tuple[0])
            minutes = float(coord_tuple[1])
            seconds = float(coord_tuple[2])

            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

            if ref in ['S', 'W']:
                decimal = -decimal

            return decimal
        except:
            return None

    def _extract_file_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Extract file system metadata

        Returns file creation, modification times, and file properties
        """
        file_meta = {}

        try:
            from pathlib import Path
            import time

            path = Path(image_path)
            stat = path.stat()

            file_meta['file_name'] = path.name
            file_meta['file_extension'] = path.suffix.lower()
            file_meta['file_size_kb'] = round(stat.st_size / 1024, 2)
            file_meta['file_size_mb'] = round(stat.st_size / (1024 * 1024), 2)

            # Timestamps
            file_meta['file_created_timestamp'] = stat.st_ctime
            file_meta['file_modified_timestamp'] = stat.st_mtime
            file_meta['file_accessed_timestamp'] = stat.st_atime

            # Human-readable dates
            file_meta['file_created_date'] = time.ctime(stat.st_ctime)
            file_meta['file_modified_date'] = time.ctime(stat.st_mtime)

        except Exception as e:
            logger.debug(f"File metadata extraction error: {e}")
            file_meta['file_meta_error'] = str(e)

        return file_meta

    def _calculate_quality_metrics(self, image: Image.Image) -> Dict[str, Any]:
        """
        Calculate advanced image quality metrics

        Returns sharpness, noise level, compression artifacts, etc.
        """
        quality = {}

        try:
            # Convert to numpy array
            img_array = np.array(image.convert('RGB'))

            # Calculate aspect ratio
            width, height = image.size
            quality['aspect_ratio'] = round(width / height, 3)
            quality['is_square'] = abs(width - height) < 10
            quality['is_landscape'] = width > height
            quality['is_portrait'] = height > width

            # Calculate megapixels
            quality['megapixels'] = round((width * height) / 1_000_000, 2)

            # Sharpness estimation (Laplacian variance)
            if self.cv2_available:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                quality['sharpness_score'] = float(laplacian_var)

                # Classify sharpness
                if laplacian_var > 500:
                    quality['sharpness_level'] = 'high'
                elif laplacian_var > 100:
                    quality['sharpness_level'] = 'medium'
                else:
                    quality['sharpness_level'] = 'low'

                # Noise estimation
                noise_level = self._estimate_noise(gray)
                quality['noise_level'] = float(noise_level)

                # Edge density
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.count_nonzero(edges) / (width * height)
                quality['edge_density'] = float(edge_density)
            else:
                # Fallback using numpy
                if len(img_array.shape) == 3:
                    gray = np.mean(img_array, axis=2)
                else:
                    gray = img_array

                # Simple sharpness estimation
                dx = np.diff(gray, axis=0)
                dy = np.diff(gray, axis=1)
                sharpness = np.mean(np.abs(dx)) + np.mean(np.abs(dy))
                quality['sharpness_score'] = float(sharpness)

            # Color diversity (standard deviation across channels)
            if len(img_array.shape) == 3:
                color_std = np.std(img_array, axis=(0, 1))
                quality['color_diversity'] = float(np.mean(color_std))
                quality['red_std'] = float(color_std[0])
                quality['green_std'] = float(color_std[1])
                quality['blue_std'] = float(color_std[2])

            # Dynamic range
            img_min, img_max = np.min(img_array), np.max(img_array)
            quality['dynamic_range'] = int(img_max - img_min)
            quality['uses_full_range'] = (img_min < 10) and (img_max > 245)

        except Exception as e:
            logger.debug(f"Quality metrics calculation error: {e}")
            quality['quality_error'] = str(e)

        return quality

    def _estimate_noise(self, gray_image):
        """Estimate noise level in image using median absolute deviation"""
        try:
            # Use median absolute deviation method
            h, w = gray_image.shape
            center_h, center_w = h // 2, w // 2
            size = min(h, w) // 4

            # Sample from center
            patch = gray_image[
                center_h - size:center_h + size,
                center_w - size:center_w + size
            ]

            # Calculate noise
            sigma = np.median(np.abs(patch - np.median(patch))) * 1.4826
            return sigma
        except:
            return 0.0

# ============================================================================
# VECTOR EMBEDDING
# ============================================================================

class ImageEmbedder:
    """
    Image Embedding Generator

    Creates vector embeddings from images using various models:
    - Statistical: Basic statistical features (always available)
    - CLIP: For image and text similarity
    - YOLO: For object detection
    - ResNet: For deep visual features

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
            if self.model_name == "clip":
                self._init_clip()
            elif self.model_name == "yolo":
                self._init_yolo()
            elif self.model_name == "resnet":
                self._init_resnet()
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

    def _init_clip(self):
        """Initialize CLIP model"""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.embedding_size = 512
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            logger.info("CLIP model initialized successfully")
        except ImportError:
            raise ImportError("CLIP requires: pip install torch transformers")

    def _init_yolo(self):
        """Initialize YOLO model"""
        try:
            import torch
            from ultralytics import YOLO

            # Use ultralytics YOLO instead of torch.hub
            self.model = YOLO('yolov8n.pt')  # YOLOv8 nano model
            self.embedding_size = 80 # For bag of objects, this will be the number of classes in COCO
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info("YOLOv8 model initialized successfully")
        except ImportError:
            raise ImportError("YOLO requires: pip install torch torchvision ultralytics")
        except Exception as e:
            raise ImportError(f"YOLO initialization failed: {e}")


    def _init_resnet(self):
        """Initialize ResNet model"""
        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms

            # Use ResNet50 pretrained on ImageNet
            self.model = models.resnet50(pretrained=True)
            # Remove the final classification layer to get features
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.embedding_size = 2048  # ResNet50 produces 2048-dim features
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)

            # Define preprocessing transforms
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            logger.info("ResNet50 model initialized successfully")
        except ImportError:
            raise ImportError("ResNet requires: pip install torch torchvision")




    def create_embedding(self, image_path: str, features: Dict[str, Any]) -> List[float]:
        """Create embedding vector for image"""
        try:
            if self.model_name == "clip":
                return self._embed_clip(image_path)
            elif self.model_name == "yolo":
                return self._embed_yolo(image_path)
            elif self.model_name == "resnet":
                return self._embed_resnet(image_path)
            else:
                # Statistical embedding (default)
                image = Image.open(image_path)
                img_array = np.array(image)
                embedding = self._create_statistical_embedding(img_array, features)
                return embedding.tolist()

        except Exception as e:
            logger.error(f"Error creating embedding for {image_path}: {e}")
            return [0.0] * self.embedding_size


    def _embed_clip(self, image_path: str) -> List[float]:
        """Create embedding using CLIP"""
        try:
            from PIL import Image
            import torch

            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"CLIP embedding error: {e}")
            return [0.0] * self.embedding_size

    def _embed_yolo(self, image_path: str) -> List[float]:
        """Create embedding using YOLO"""
        try:
            # Run YOLO inference
            results = self.model(image_path, verbose=False)

            # Create a bag-of-objects embedding
            embedding = [0.0] * self.embedding_size

            # Extract detected objects
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id < self.embedding_size:
                        embedding[cls_id] += 1

            return embedding
        except Exception as e:
            logger.error(f"YOLO embedding error: {e}")
            return [0.0] * self.embedding_size

    def _embed_resnet(self, image_path: str) -> List[float]:
        """Create embedding using ResNet"""
        try:
            from PIL import Image
            import torch

            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)

            # Flatten and convert to list
            embedding = features.squeeze().cpu().numpy().flatten().tolist()

            # Normalize to 512 dimensions via pooling for consistency with other models
            if len(embedding) == 2048:
                # Average pool to 512 dimensions
                embedding = [sum(embedding[i:i+4])/4 for i in range(0, 2048, 4)]

            return embedding
        except Exception as e:
            logger.error(f"ResNet embedding error: {e}")
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
# AI IMAGE DOWNLOADING
# ============================================================================

class ImageDownloader:
    """
    AI-Generated Image Downloader

    Downloads diverse AI-generated images from multiple sources:
    - Faces: ThisPersonDoesNotExist, 100K Faces
    - Artwork: ThisArtworkDoesNotExist
    - Animals: ThisCatDoesNotExist, ThisHorseDoesNotExist
    - Mixed: UnrealPerson

    Features:
    - Automatic duplicate detection via MD5 hashing
    - Comprehensive metadata generation and storage
    - Configurable download delays
    - Background download loop support
    - Support for diverse image categories (not just faces)
    """

    # Available download sources - diverse real and AI-generated images
    DOWNLOAD_SOURCES = {
        'picsum_general': {
            'name': 'Picsum Photos - General',
            'url': 'https://picsum.photos/1024/768',
            'description': 'Random high-quality photos from Unsplash (general)',
            'category': 'general'
        },
        'picsum_landscape': {
            'name': 'Picsum Photos - Landscape',
            'url': 'https://picsum.photos/1920/1080',
            'description': 'Random high-quality landscape format photos',
            'category': 'landscape'
        },
        'picsum_square': {
            'name': 'Picsum Photos - Square',
            'url': 'https://picsum.photos/1024/1024',
            'description': 'Random high-quality square photos',
            'category': 'general'
        },
        'picsum_portrait': {
            'name': 'Picsum Photos - Portrait',
            'url': 'https://picsum.photos/768/1024',
            'description': 'Random high-quality portrait format photos',
            'category': 'general'
        },
        'picsum_hd': {
            'name': 'Picsum Photos - HD',
            'url': 'https://picsum.photos/2560/1440',
            'description': 'Random high-quality HD photos',
            'category': 'landscape'
        }
    }

    def __init__(self, config: SystemConfig, stats: SystemStats):
        self.config = config
        self.stats = stats
        self.running = False
        self.downloaded_hashes = set()
        self._hashes_loaded = False
        self._hash_loading_thread = None
        self._hash_loading_callback = None

        # Create images directory
        os.makedirs(self.config.images_dir, exist_ok=True)

        logger.info("ImageDownloader initialized (hash loading can be started in background)")

    def start_background_hash_loading(self, progress_callback=None, completion_callback=None):
        """Start loading hashes in background thread

        Args:
            progress_callback: Called with (current, total, message) for progress updates
            completion_callback: Called when loading completes with (total_loaded, elapsed_time)
        """
        if self._hashes_loaded:
            if completion_callback:
                completion_callback(len(self.downloaded_hashes), 0)
            return

        if self._hash_loading_thread and self._hash_loading_thread.is_alive():
            logger.info("Hash loading already in progress")
            return

        logger.info("Starting background hash loading...")

        def load_worker():
            try:
                self._load_existing_hashes_with_callback(progress_callback, completion_callback)
            except Exception as e:
                logger.error(f"Error loading hashes in background: {e}")

        import threading
        self._hash_loading_thread = threading.Thread(target=load_worker, daemon=True)
        self._hash_loading_thread.start()

    def _load_existing_hashes(self):
        """Load hashes of existing images (lazy-loaded on first download)"""
        if self._hashes_loaded:
            return  # Already loaded

        logger.info("Loading existing image hashes for duplicate detection...")
        start_time = time.time()
        count = 0

        for file_path in Path(self.config.images_dir).rglob("*.jpg"):
            # Skip macOS metadata files
            if file_path.name.startswith('._'):
                continue
            if file_path.is_file():
                try:
                    file_hash = self._get_file_hash(str(file_path))
                    if file_hash:
                        self.downloaded_hashes.add(file_hash)
                        count += 1
                        # Log progress every 5000 files
                        if count % 5000 == 0:
                            logger.info(f"  Processed {count} images...")
                except Exception:
                    pass

        elapsed = time.time() - start_time
        logger.info(f"✓ Loaded {count} image hashes in {elapsed:.2f}s")
        self._hashes_loaded = True

    def _load_existing_hashes_with_callback(self, progress_callback=None, completion_callback=None):
        """Load hashes with progress callbacks for GUI updates"""
        if self._hashes_loaded:
            return

        logger.info("Loading existing image hashes for duplicate detection...")
        start_time = time.time()
        count = 0

        # First, count total files for progress percentage (exclude macOS metadata files)
        all_files = [f for f in Path(self.config.images_dir).rglob("*.jpg") if not f.name.startswith('._')]
        total_files = len(all_files)

        if progress_callback:
            progress_callback(0, total_files, "Starting hash loading...")

        for file_path in all_files:
            if file_path.is_file():
                try:
                    file_hash = self._get_file_hash(str(file_path))
                    if file_hash:
                        self.downloaded_hashes.add(file_hash)
                        count += 1

                        # Progress callback every 1000 files
                        if progress_callback and count % 1000 == 0:
                            progress_callback(count, total_files, f"Processed {count}/{total_files} images...")

                except Exception:
                    pass

        elapsed = time.time() - start_time
        self._hashes_loaded = True

        logger.info(f"✓ Loaded {count} image hashes in {elapsed:.2f}s")

        if completion_callback:
            completion_callback(count, elapsed)

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def download_image(self) -> Optional[str]:
        """Download a single image with metadata JSON"""
        # Lazy-load hashes on first download
        if not self._hashes_loaded:
            self._load_existing_hashes()

        self.stats.increment_download_attempts()
        download_start = time.time()

        try:
            # Download image based on selected source
            download_time = datetime.now()
            source = self.config.download_source

            # Get source info
            if source in self.DOWNLOAD_SOURCES:
                source_url = self.DOWNLOAD_SOURCES[source]['url']
                response = self._download_from_generic_source(source_url)
            else:
                logger.warning(f"Unknown download source: {source}, using default (picsum_general)")
                response = self._download_from_generic_source(self.DOWNLOAD_SOURCES['picsum_general']['url'])

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
            filename = f"image_{timestamp}_{image_hash[:8]}.jpg"
            file_path = os.path.join(self.config.images_dir, filename)

            with open(file_path, 'wb') as f:
                f.write(response.content)

            # Analyze image and create metadata
            try:
                # Analyze the downloaded image
                analyzer = ImageAnalyzer()
                features = analyzer.analyze_image(file_path)

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
                    'image_id': timestamp,
                    'md5_hash': image_hash,

                    # Download info
                    'download_timestamp': download_time.isoformat(),
                    'download_date': download_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'source': self.config.download_source,
                    'source_url': self.DOWNLOAD_SOURCES.get(self.config.download_source, {}).get('url', 'unknown'),
                    'source_category': self.DOWNLOAD_SOURCES.get(self.config.download_source, {}).get('category', 'unknown'),
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

                    # Image features from analyzer (IMPORTANT FOR QUERYING)
                    'image_features': features,

                    # Queryable attributes extracted from features
                    'queryable_attributes': {
                        'brightness_level': 'bright' if features.get('brightness', 0) > 150 else 'dark',
                        'image_quality': 'high' if features.get('contrast', 0) > 50 else 'medium',
                        'sharpness_level': features.get('sharpness_level', 'unknown'),
                        'color_diversity': features.get('color_diversity', 0),
                        'aspect_ratio': features.get('aspect_ratio', 1.0),
                        'megapixels': features.get('megapixels', 0)
                    },

                    # HTTP headers for debugging
                    'http_headers': dict(response.headers),

                    # Downloader config
                    'downloader_config': {
                        'storage_dir': self.config.images_dir,
                        'delay': self.config.download_delay
                    }
                }

                # Save metadata JSON alongside image
                json_filename = f"image_{timestamp}_{image_hash[:8]}.json"
                json_path = os.path.join(self.config.images_dir, json_filename)

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

    def _download_from_generic_source(self, url: str) -> Optional[requests.Response]:
        """Download from a generic AI image generation source"""
        try:
            response = requests.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=30
            )
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error downloading from {url}: {e}")
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
                file_path = self.download_image()
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
# IMAGE PROCESSING
# ============================================================================

class ImageProcessor:
    """
    Image Processor

    Processes images to create and store embeddings:
    - Analyzes images for features
    - Creates vector embeddings using configured model
    - Stores embeddings in database with metadata
    - Tracks processed files to avoid duplicates
    - Supports batch processing

    Works in conjunction with ImageAnalyzer, ImageEmbedder, and PgVectorDatabaseManager.
    """

    def __init__(self, config: SystemConfig, stats: SystemStats, db_manager: PgVectorDatabaseManager):
        self.config = config
        self.stats = stats
        self.db_manager = db_manager
        self.analyzer = ImageAnalyzer()

        # Initialize multiple embedders for comprehensive feature extraction
        self.embedders = {}
        self.embedders['clip'] = ImageEmbedder(model_name='clip') if AVAILABLE_MODELS.get('clip') else None
        self.embedders['yolo'] = ImageEmbedder(model_name='yolo') if AVAILABLE_MODELS.get('yolo') else None
        self.embedders['resnet'] = ImageEmbedder(model_name='resnet') if AVAILABLE_MODELS.get('resnet') else None
        self.embedders['statistical'] = ImageEmbedder(model_name='statistical')  # Always available

        # Filter out None embedders
        self.embedders = {k: v for k, v in self.embedders.items() if v is not None}
        logger.info(f"Initialized embedders: {list(self.embedders.keys())}")

        self.processed_files = set()

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def get_new_files_only(self) -> List[str]:
        """Get list of files that are NOT in the database yet"""
        # Get all image files
        all_image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            all_image_files.extend(Path(self.config.images_dir).rglob(ext))

        # Filter out macOS metadata files (._*)
        all_image_files = [f for f in all_image_files if not f.name.startswith('._')]

        # Filter to only new files (not in database)
        new_files = []
        for file_path in all_image_files:
            file_path_str = str(file_path)

            # Skip if already processed in this session
            if file_path_str in self.processed_files:
                continue

            # Calculate hash and check if in database
            file_hash = self._get_file_hash(file_path_str)
            if not self.db_manager.hash_exists(file_hash):
                new_files.append(file_path_str)

        logger.info(f"Found {len(new_files)} new files out of {len(all_image_files)} total files")
        return new_files

    def process_new_images_only(self, callback=None, progress_callback=None) -> Dict[str, int]:
        """Process only new images (not in database)

        Args:
            callback: Called for each processed image with ImageData
            progress_callback: Called with (current, total, message) for progress updates
        """
        new_files = self.get_new_files_only()
        total_files = len(new_files)

        stats = {
            'total_files': total_files,
            'processed': 0,
            'skipped': 0,
            'errors': 0
        }

        logger.info(f"Starting to process {total_files} new files")

        for idx, file_path in enumerate(new_files, 1):
            # Update progress
            if progress_callback:
                progress_callback(idx, total_files, f"Processing {idx}/{total_files}")

            if self.process_image_file(file_path, callback=callback):
                stats['processed'] += 1
            else:
                stats['errors'] += 1

        logger.info(f"Processed {stats['processed']} new files, {stats['errors']} errors")
        return stats

    def process_image_file(self, file_path: str, callback=None) -> bool:
        """Process a single image file and store its embeddings

        Args:
            file_path: Path to image file
            callback: Optional callback function to call with ImageData

        Returns:
            True if processing was successful, False otherwise
        """
        start_time = time.time()
        self.stats.increment_embed_processed()

        try:
            # Check if already in database
            file_hash = self._get_file_hash(file_path)
            if self.db_manager.hash_exists(file_hash):
                self.stats.increment_embed_duplicates()
                logger.debug(f"Image already in database: {file_path}")
                return False

            # Analyze image
            features = self.analyzer.analyze_image(file_path)
            if 'error' in features:
                logger.error(f"Failed to analyze image {file_path}: {features['error']}")
                self.stats.increment_embed_errors()
                return False

            # Generate embeddings using all available models
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_id = f"img_{timestamp}_{file_hash[:8]}"

            # First, add the image record (without embeddings)
            image_data = ImageData(
                image_id=image_id,
                file_path=file_path,
                features=features,
                embedding=None,  # No single embedding anymore
                timestamp=datetime.now().isoformat(),
                image_hash=file_hash
            )

            # Add image to database
            if not self.db_manager.add_image(image_data):
                logger.error(f"Failed to add image to database: {file_path}")
                self.stats.increment_embed_errors()
                return False

            # Generate and store embeddings for each model
            for model_name, embedder in self.embedders.items():
                try:
                    embedding = embedder.create_embedding(file_path, features)
                    if embedding and any(v != 0.0 for v in embedding):
                        # Store embedding
                        if not self.db_manager.add_embedding(image_id, model_name, embedding):
                            logger.warning(f"Failed to store {model_name} embedding for {file_path}")
                except Exception as e:
                    logger.error(f"Error creating {model_name} embedding for {file_path}: {e}")

            # Track as processed
            self.processed_files.add(file_path)

            # Calculate time and update stats
            elapsed_time = time.time() - start_time
            self.stats.increment_embed_success(elapsed_time)
            logger.info(f"Processed {file_path} in {elapsed_time:.2f}s with {len(self.embedders)} embeddings")

            # Call callback if provided
            if callback:
                callback(image_data)

            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats.increment_embed_errors()
            return False

    def process_all_images(self, callback=None, progress_callback=None):
        """Process all images in the images directory

        Args:
            callback: Called for each processed image with ImageData
            progress_callback: Called with (current, total, message) for progress updates
        """
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(self.config.images_dir).rglob(ext))

        # Filter out macOS metadata files (._*)
        image_files = [f for f in image_files if not f.name.startswith('._')]

        total_files = len(image_files)
        logger.info(f"Starting to process {total_files} total files")

        for idx, file_path in enumerate(image_files, 1):
            # Update progress
            if progress_callback:
                progress_callback(idx, total_files, f"Processing {idx}/{total_files}")

            # Pass callback to process_image_file - it will call it with ImageData
            self.process_image_file(str(file_path), callback=callback)

# ============================================================================
# INTEGRATED SYSTEM
# ============================================================================

class IntegratedImageSystem:
    """
    Integrated Image Processing System

    Main orchestrator class that integrates all system components:
    - Configuration management
    - Statistics tracking
    - Database management
    - Image downloading
    - Image processing and embedding

    Provides a unified interface for the entire image processing workflow.

    Usage:
        system = IntegratedImageSystem()
        system.initialize()
        system.downloader.download_image()
        system.processor.process_all_images()
        results = system.db_manager.search_images(embedding)
    """

    def __init__(self, config_file: str = "system_config.json"):
        self.config = SystemConfig.from_file(config_file)
        self.stats = SystemStats()

        # Initialize PostgreSQL + pgvector database manager
        self.db_manager = PgVectorDatabaseManager(self.config)
        logger.info("Using PostgreSQL + pgvector database")

        self.downloader = ImageDownloader(self.config, self.stats)
        self.processor = None  # Initialize after database

    def initialize(self) -> bool:
        """Initialize the system"""
        if not self.db_manager.initialize():
            return False

        self.processor = ImageProcessor(self.config, self.stats, self.db_manager)
        logger.info("Integrated Image System initialized")
        return True

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        stats = self.stats.get_stats()
        db_info = self.db_manager.get_collection_info()

        return {
            'database': db_info,
            'statistics': stats,
            'config': asdict(self.config),
            'images_directory': self.config.images_dir,
            'images_count': len([f for f in Path(self.config.images_dir).rglob("*.jpg") if not f.name.startswith('._')]) if os.path.exists(self.config.images_dir) else 0
        }