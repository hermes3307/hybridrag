#!/usr/bin/env python3
"""
🖼️ Image Feature Extraction and Vector Storage System
이미지 특성 추출 및 Qdrant 벡터 데이터베이스 저장 시스템
"""

import os
import json
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
from datetime import datetime
import uuid

# Image processing
import cv2
import numpy as np
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import colorsys

# ML/AI libraries
import torch
import clip
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.cluster import KMeans
import torchvision.transforms as transforms

# Vector database
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageFeatureExtractor:
    """🎯 이미지 특성 추출기"""
    
    def __init__(self, device: str = "auto"):
        """Initialize feature extractor with models"""
        
        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model for semantic embeddings
        self.clip_model = None
        self.clip_preprocess = None
        self._load_clip_model()
        
        # Load ViT model for additional features (optional)
        self.vit_model = None
        self.vit_feature_extractor = None
        self._load_vit_model()
        
        logger.info("✅ Feature extractor initialized")
    
    def _load_clip_model(self):
        """Load CLIP model for semantic embeddings"""
        try:
            # CLIP ViT-B/32 모델 로드 (512차원 벡터)
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("✅ CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
    
    def _load_vit_model(self):
        """Load additional ViT model for enhanced features"""
        try:
            # DINOv2 또는 다른 ViT 모델 (선택적)
            model_name = "facebook/dino-vitb16"
            self.vit_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.vit_model = AutoModel.from_pretrained(model_name).to(self.device)
            logger.info("✅ ViT model loaded successfully")
        except Exception as e:
            logger.warning(f"ViT model not loaded (optional): {e}")
    
    def extract_basic_metadata(self, image_path: str) -> Dict:
        """Tier 1: 기본 메타데이터 추출"""
        try:
            image_path = Path(image_path)
            
            # 파일 기본 정보
            stat = image_path.stat()
            basic_info = {
                'filename': image_path.name,
                'file_size': stat.st_size,
                'file_extension': image_path.suffix.lower(),
                'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
            
            # 이미지 정보
            with Image.open(image_path) as img:
                basic_info.update({
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                })
                
                # EXIF 데이터 추출
                exif_data = self._extract_exif(img)
                if exif_data:
                    basic_info['exif'] = exif_data
            
            return basic_info
            
        except Exception as e:
            logger.error(f"Error extracting basic metadata from {image_path}: {e}")
            return {}
    
    def _extract_exif(self, img: Image.Image) -> Dict:
        """EXIF 데이터 추출"""
        try:
            exif_dict = {}
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except:
                            value = str(value)
                    exif_dict[tag] = value
            return exif_dict
        except Exception as e:
            logger.debug(f"EXIF extraction failed: {e}")
            return {}
    
    def extract_color_features(self, image_path: str) -> Dict:
        """Tier 1: 색상 히스토그램 및 주요 색상 추출"""
        try:
            # OpenCV로 이미지 로드
            img = cv2.imread(str(image_path))
            if img is None:
                return {}
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 1. RGB 히스토그램
            hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten()
            hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten()
            
            # 2. HSV 히스토그램
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([img_hsv], [0], None, [180], [0, 180]).flatten()
            hist_s = cv2.calcHist([img_hsv], [1], None, [256], [0, 256]).flatten()
            hist_v = cv2.calcHist([img_hsv], [2], None, [256], [0, 256]).flatten()
            
            # 3. 주요 색상 추출 (K-means clustering)
            dominant_colors = self._extract_dominant_colors(img_rgb, k=5)
            
            # 4. 색상 통계
            color_stats = self._calculate_color_statistics(img_rgb, img_hsv)
            
            return {
                'rgb_histogram': {
                    'red': hist_r.tolist(),
                    'green': hist_g.tolist(),
                    'blue': hist_b.tolist()
                },
                'hsv_histogram': {
                    'hue': hist_h.tolist(),
                    'saturation': hist_s.tolist(),
                    'value': hist_v.tolist()
                },
                'dominant_colors': dominant_colors,
                'color_statistics': color_stats
            }
            
        except Exception as e:
            logger.error(f"Error extracting color features from {image_path}: {e}")
            return {}
    
    def _extract_dominant_colors(self, img_rgb: np.ndarray, k: int = 5) -> List[Dict]:
        """K-means를 사용한 주요 색상 추출"""
        try:
            # 이미지를 1차원으로 변환
            pixels = img_rgb.reshape(-1, 3)
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # 각 클러스터의 비율 계산
            labels = kmeans.labels_
            unique, counts = np.unique(labels, return_counts=True)
            
            dominant_colors = []
            for i, color in enumerate(kmeans.cluster_centers_):
                percentage = (counts[i] / len(pixels)) * 100
                rgb = color.astype(int)
                
                # HSV 변환
                hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
                
                dominant_colors.append({
                    'rgb': rgb.tolist(),
                    'hex': f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
                    'hsv': [int(hsv[0]*360), int(hsv[1]*100), int(hsv[2]*100)],
                    'percentage': round(percentage, 2)
                })
            
            # 비율 순으로 정렬
            dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
            return dominant_colors
            
        except Exception as e:
            logger.error(f"Error extracting dominant colors: {e}")
            return []
    
    def _calculate_color_statistics(self, img_rgb: np.ndarray, img_hsv: np.ndarray) -> Dict:
        """색상 통계 계산"""
        try:
            return {
                'brightness': float(np.mean(img_rgb)),
                'contrast': float(np.std(img_rgb)),
                'saturation_mean': float(np.mean(img_hsv[:, :, 1])),
                'saturation_std': float(np.std(img_hsv[:, :, 1])),
                'hue_mean': float(np.mean(img_hsv[:, :, 0])),
                'value_mean': float(np.mean(img_hsv[:, :, 2])),
                'color_diversity': float(np.std(img_rgb.reshape(-1, 3), axis=0).mean())
            }
        except Exception as e:
            logger.error(f"Error calculating color statistics: {e}")
            return {}
    
    def extract_quality_metrics(self, image_path: str) -> Dict:
        """Tier 1: 기본 이미지 품질 지표"""
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {}
            
            # 1. 선명도 (Laplacian variance)
            sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
            
            # 2. 노이즈 레벨 (추정)
            noise_level = self._estimate_noise_level(img)
            
            # 3. 대비
            contrast = img.std()
            
            # 4. 엣지 밀도
            edges = cv2.Canny(img, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 5. 텍스처 복잡도
            texture_complexity = self._calculate_texture_complexity(img)
            
            return {
                'sharpness': float(sharpness),
                'noise_level': float(noise_level),
                'contrast': float(contrast),
                'edge_density': float(edge_density),
                'texture_complexity': float(texture_complexity),
                'resolution_score': float(img.shape[0] * img.shape[1])  # 해상도 점수
            }
            
        except Exception as e:
            logger.error(f"Error extracting quality metrics from {image_path}: {e}")
            return {}
    
    def _estimate_noise_level(self, img: np.ndarray) -> float:
        """노이즈 레벨 추정"""
        try:
            # Gaussian blur 적용 후 차이로 노이즈 추정
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            noise = cv2.absdiff(img, blurred)
            return float(noise.mean())
        except:
            return 0.0
    
    def _calculate_texture_complexity(self, img: np.ndarray) -> float:
        """텍스처 복잡도 계산"""
        try:
            # 그래디언트 기반 텍스처 복잡도
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return float(magnitude.mean())
        except:
            return 0.0
    
    def extract_clip_embeddings(self, image_path: str) -> Optional[np.ndarray]:
        """Tier 2: CLIP 임베딩 추출 (벡터 검색용)"""
        if self.clip_model is None:
            logger.warning("CLIP model not available")
            return None
        
        try:
            # 이미지 로드 및 전처리
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # CLIP 임베딩 추출
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 정규화
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error extracting CLIP embeddings from {image_path}: {e}")
            return None
    
    def extract_vit_embeddings(self, image_path: str) -> Optional[np.ndarray]:
        """Tier 2: ViT 임베딩 추출 (추가 특성)"""
        if self.vit_model is None:
            return None
        
        try:
            # 이미지 로드 및 전처리
            image = Image.open(image_path).convert('RGB')
            inputs = self.vit_feature_extractor(images=image, return_tensors="pt").to(self.device)
            
            # ViT 임베딩 추출
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
            
            return embeddings.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error extracting ViT embeddings from {image_path}: {e}")
            return None
    
    def extract_all_features(self, image_path: str) -> Dict:
        """모든 특성 추출"""
        logger.info(f"🔍 Extracting features from {Path(image_path).name}")
        
        features = {
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'basic_metadata': self.extract_basic_metadata(image_path),
            'color_features': self.extract_color_features(image_path),
            'quality_metrics': self.extract_quality_metrics(image_path),
        }
        
        # CLIP 임베딩 (벡터 검색용 메인)
        clip_embeddings = self.extract_clip_embeddings(image_path)
        if clip_embeddings is not None:
            features['clip_embeddings'] = clip_embeddings
        
        # ViT 임베딩 (추가 특성)
        vit_embeddings = self.extract_vit_embeddings(image_path)
        if vit_embeddings is not None:
            features['vit_embeddings'] = vit_embeddings
        
        return features

class ImageVectorStorage:
    """🗄️ Qdrant 벡터 데이터베이스 저장 관리"""
    
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "image_vectors"):
        """Initialize Qdrant client and collection"""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = 512  # CLIP ViT-B/32 dimension
        
        # Create collection if not exists
        self._create_collection()
        
        logger.info(f"✅ Connected to Qdrant at {host}:{port}")
    
    def _create_collection(self):
        """Create Qdrant collection for image vectors"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE  # 코사인 유사도 (이미지 검색에 적합)
                    )
                )
                logger.info(f"✅ Created collection: {self.collection_name}")
            else:
                logger.info(f"✅ Collection already exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def store_image_features(self, features: Dict) -> str:
        """이미지 특성을 벡터 DB에 저장"""
        try:
            # Generate unique ID
            image_path = features['image_path']
            image_id = self._generate_image_id(image_path)
            
            # CLIP 임베딩을 메인 벡터로 사용
            if 'clip_embeddings' not in features:
                logger.error("No CLIP embeddings found in features")
                return None
            
            vector = features['clip_embeddings'].tolist()
            
            # Prepare payload (메타데이터)
            payload = {
                'filename': features['basic_metadata'].get('filename', ''),
                'file_path': image_path,
                'file_size': features['basic_metadata'].get('file_size', 0),
                'dimensions': [
                    features['basic_metadata'].get('width', 0),
                    features['basic_metadata'].get('height', 0)
                ],
                'format': features['basic_metadata'].get('format', ''),
                'timestamp': features['timestamp'],
                
                # 색상 정보 (검색용)
                'dominant_colors': features['color_features'].get('dominant_colors', []),
                'brightness': features['color_features'].get('color_statistics', {}).get('brightness', 0),
                'saturation': features['color_features'].get('color_statistics', {}).get('saturation_mean', 0),
                'contrast': features['color_features'].get('color_statistics', {}).get('contrast', 0),
                
                # 품질 지표
                'sharpness': features['quality_metrics'].get('sharpness', 0),
                'edge_density': features['quality_metrics'].get('edge_density', 0),
                'resolution_score': features['quality_metrics'].get('resolution_score', 0),
                
                # 추가 임베딩 (검색 보조용)
                'has_vit_embeddings': 'vit_embeddings' in features,
            }
            
            # ViT 임베딩이 있으면 payload에 추가 (별도 검색용)
            if 'vit_embeddings' in features:
                payload['vit_vector'] = features['vit_embeddings'].tolist()
            
            # Store in Qdrant
            point = PointStruct(
                id=image_id,
                vector=vector,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"✅ Stored features for {Path(image_path).name} (ID: {image_id})")
            return image_id
            
        except Exception as e:
            logger.error(f"Error storing image features: {e}")
            return None
    
    def _generate_image_id(self, image_path: str) -> str:
        """이미지 경로 기반 고유 ID 생성"""
        # 파일 경로의 해시값 사용
        hash_object = hashlib.md5(str(image_path).encode())
        return hash_object.hexdigest()
    
    def search_similar_images(self, query_vector: np.ndarray, limit: int = 10, 
                             score_threshold: float = 0.7) -> List[Dict]:
        """유사한 이미지 검색"""
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for point in search_result:
                result = {
                    'id': point.id,
                    'score': point.score,
                    'metadata': point.payload
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar images: {e}")
            return []
    
    def search_by_image_path(self, image_path: str, limit: int = 10) -> List[Dict]:
        """이미지 파일로 유사 이미지 검색"""
        try:
            # 임시로 특성 추출기 생성
            extractor = ImageFeatureExtractor()
            clip_vector = extractor.extract_clip_embeddings(image_path)
            
            if clip_vector is None:
                logger.error("Failed to extract CLIP embeddings for search")
                return []
            
            return self.search_similar_images(clip_vector, limit)
            
        except Exception as e:
            logger.error(f"Error searching by image path: {e}")
            return []
    
    def filter_images(self, filters: Dict, limit: int = 100) -> List[Dict]:
        """메타데이터 기반 이미지 필터링"""
        try:
            # Build filter conditions
            must_conditions = []
            
            if 'min_width' in filters:
                must_conditions.append(
                    models.FieldCondition(
                        key="dimensions[0]",
                        match=models.Range(gte=filters['min_width'])
                    )
                )
            
            if 'min_height' in filters:
                must_conditions.append(
                    models.FieldCondition(
                        key="dimensions[1]",
                        match=models.Range(gte=filters['min_height'])
                    )
                )
            
            if 'format' in filters:
                must_conditions.append(
                    models.FieldCondition(
                        key="format",
                        match=models.MatchValue(value=filters['format'])
                    )
                )
            
            if 'min_sharpness' in filters:
                must_conditions.append(
                    models.FieldCondition(
                        key="sharpness",
                        match=models.Range(gte=filters['min_sharpness'])
                    )
                )
            
            # Execute search with filters
            if must_conditions:
                filter_condition = models.Filter(must=must_conditions)
                search_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    limit=limit
                )
                return [{'id': point.id, 'metadata': point.payload} for point in search_result[0]]
            else:
                # No filters, return all
                search_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit
                )
                return [{'id': point.id, 'metadata': point.payload} for point in search_result[0]]
                
        except Exception as e:
            logger.error(f"Error filtering images: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """컬렉션 정보 조회"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.config.params.vectors.size,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

class ImageVectorProcessor:
    """🎯 통합 이미지 처리 및 벡터 저장 시스템"""
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333, 
                 collection_name: str = "image_vectors"):
        """Initialize image processor with feature extractor and vector storage"""
        self.extractor = ImageFeatureExtractor()
        self.storage = ImageVectorStorage(qdrant_host, qdrant_port, collection_name)
        
        logger.info("🚀 Image Vector Processor initialized")
    
    async def process_image_directory(self, directory_path: str, 
                                    image_extensions: List[str] = None) -> Dict:
        """디렉토리의 모든 이미지 처리"""
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif']
        
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return {'success': False, 'error': 'Directory not found'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(f"**/*{ext}"))
            image_files.extend(directory.glob(f"**/*{ext.upper()}"))
        
        logger.info(f"📁 Found {len(image_files)} images in {directory_path}")
        
        # Process images
        results = {
            'total_files': len(image_files),
            'processed': 0,
            'failed': 0,
            'stored_ids': []
        }
        
        for image_path in image_files:
            try:
                logger.info(f"🔄 Processing {image_path.name}...")
                
                # Extract features
                features = self.extractor.extract_all_features(str(image_path))
                
                # Store in vector database
                image_id = self.storage.store_image_features(features)
                
                if image_id:
                    results['stored_ids'].append(image_id)
                    results['processed'] += 1
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results['failed'] += 1
        
        logger.info(f"✅ Processing complete: {results['processed']} success, {results['failed']} failed")
        return results
    
    def process_single_image(self, image_path: str) -> Optional[str]:
        """단일 이미지 처리"""
        try:
            logger.info(f"🔄 Processing single image: {Path(image_path).name}")
            
            # Extract features
            features = self.extractor.extract_all_features(image_path)
            
            # Store in vector database
            image_id = self.storage.store_image_features(features)
            
            if image_id:
                logger.info(f"✅ Successfully processed and stored image (ID: {image_id})")
            else:
                logger.error("Failed to store image features")
            
            return image_id
            
        except Exception as e:
            logger.error(f"Error processing single image {image_path}: {e}")
            return None
    
    def search_similar_images(self, query_image_path: str, limit: int = 10) -> List[Dict]:
        """쿼리 이미지와 유사한 이미지 검색"""
        return self.storage.search_by_image_path(query_image_path, limit)
    
    def get_stats(self) -> Dict:
        """시스템 통계 조회"""
        collection_info = self.storage.get_collection_info()
        return {
            'collection_info': collection_info,
            'extractor_device': self.extractor.device,
            'models_loaded': {
                'clip': self.extractor.clip_model is not None,
                'vit': self.extractor.vit_model is not None
            }
        }

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize processor
        processor = ImageVectorProcessor(
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="my_image_collection"
        )
        
        # Process directory of downloaded images
        results = await processor.process_image_directory("downloaded_images")
        print(f"📊 Results: {results}")
        
        # Search for similar images
        if results['stored_ids']:
            # Use first image as query
            similar_images = processor.search_similar_images("downloaded_images/sample.jpg")
            print(f"🔍 Found {len(similar_images)} similar images")
        
        # Get system stats
        stats = processor.get_stats()
        print(f"📈 System Stats: {stats}")
    
    asyncio.run(main())