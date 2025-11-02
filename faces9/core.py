#!/usr/bin/env python3
"""
Core Backend for Face Processing System

This module provides the core functionality for the face processing system including:
- Face image downloading from AI generation services
- Face feature analysis and demographic estimation
- Vector embedding generation using multiple models
- PostgreSQL + pgvector database management
- Face similarity search with metadata filtering

Architecture:
- FaceAnalyzer: Analyzes images for features and demographics
- FaceEmbedder: Creates vector embeddings using various models
- FaceDownloader: Downloads AI-generated face images
- FaceProcessor: Processes faces and creates embeddings
- IntegratedFaceSystem: Main orchestrator class

Dependencies:
- PostgreSQL with pgvector extension for vector storage
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

# Import pgvector database manager
from pgvector_db import PgVectorDatabaseManager


def check_embedding_models():
    """Check availability of various embedding models"""
    models_status = {}

    # FaceNet (facenet-pytorch)
    try:
        from facenet_pytorch import InceptionResnetV1
        models_status['facenet'] = True
    except Exception:
        models_status['facenet'] = False

    # ArcFace/InsightFace
    try:
        import insightface
        models_status['arcface'] = True
    except Exception:
        models_status['arcface'] = False

    # DeepFace
    try:
        from deepface import DeepFace
        models_status['deepface'] = True
    except Exception:
        models_status['deepface'] = False

    # VGGFace2 (via deepface or keras)
    try:
        from deepface import DeepFace
        models_status['vggface2'] = True
    except Exception:
        models_status['vggface2'] = False

    # OpenFace (via deepface)
    try:
        from deepface import DeepFace
        models_status['openface'] = True
    except Exception:
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

                    # Convert bytes to string
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = str(value)

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
        self._hashes_loaded = False
        self._hash_loading_thread = None
        self._hash_loading_callback = None

        # Create faces directory
        os.makedirs(self.config.faces_dir, exist_ok=True)

        logger.info("FaceDownloader initialized (hash loading can be started in background)")

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

        logger.info("Loading existing face image hashes for duplicate detection...")
        start_time = time.time()
        count = 0

        for file_path in Path(self.config.faces_dir).rglob("*.jpg"):
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

        logger.info("Loading existing face image hashes for duplicate detection...")
        start_time = time.time()
        count = 0

        # First, count total files for progress percentage
        all_files = list(Path(self.config.faces_dir).rglob("*.jpg"))
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

    def download_face(self) -> Optional[str]:
        """Download a single face image with metadata JSON"""
        # Lazy-load hashes on first download
        if not self._hashes_loaded:
            self._load_existing_hashes()

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

    Works in conjunction with FaceAnalyzer, FaceEmbedder, and PgVectorDatabaseManager.
    """

    def __init__(self, config: SystemConfig, stats: SystemStats, db_manager: PgVectorDatabaseManager):
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

        # Filter out macOS metadata files (._*)
        all_face_files = [f for f in all_face_files if not f.name.startswith('._')]

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

    def process_new_faces_only(self, callback=None, progress_callback=None) -> Dict[str, int]:
        """Process only new faces (not in database)

        Args:
            callback: Called for each processed face with FaceData
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

            if self.process_face_file(file_path, callback=callback):
                stats['processed'] += 1
            else:
                stats['errors'] += 1

        logger.info(f"Processed {stats['processed']} new files, {stats['errors']} errors")
        return stats

    def process_all_faces(self, callback=None, progress_callback=None):
        """Process all faces in the faces directory

        Args:
            callback: Called for each processed face with FaceData
            progress_callback: Called with (current, total, message) for progress updates
        """
        face_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            face_files.extend(Path(self.config.faces_dir).rglob(ext))

        # Filter out macOS metadata files (._*)
        face_files = [f for f in face_files if not f.name.startswith('._')]

        total_files = len(face_files)
        logger.info(f"Starting to process {total_files} total files")

        for idx, file_path in enumerate(face_files, 1):
            # Update progress
            if progress_callback:
                progress_callback(idx, total_files, f"Processing {idx}/{total_files}")

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

        # Initialize PostgreSQL + pgvector database manager
        self.db_manager = PgVectorDatabaseManager(self.config)
        logger.info("Using PostgreSQL + pgvector database")

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