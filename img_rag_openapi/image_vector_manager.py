#!/usr/bin/env python3
"""
🖼️ Image Vector Store Manager
Manages image embeddings and similarity search with Qdrant for images
"""

import os
import asyncio
import numpy as np
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
import time
import base64

try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError as e:
    print("❌ Missing required packages. Please install:")
    print("pip install sentence-transformers qdrant-client")
    raise e

# For image embeddings
try:
    import clip
    import torch
    from PIL import Image
except ImportError:
    print("❌ Missing CLIP packages. Please install:")
    print("pip install torch torchvision ftfy regex tqdm")
    print("pip install git+https://github.com/openai/CLIP.git")
    clip = None
    torch = None

from smart_image_chunker import ImageChunk

logger = logging.getLogger(__name__)

class ImageVectorStoreManager:
    """🖼️ Manages image vector embeddings and search operations"""
    
    def __init__(self, 
                 text_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 image_model_name: str = "ViT-B/32",
                 qdrant_path: str = "./qdrant_image_vector",
                 collection_name: str = "image_collection"):
        
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        
        # Initialize components
        self.text_vectorizer = None
        self.image_model = None
        self.image_preprocess = None
        self.qdrant_client = None
        self.text_vector_size = None
        self.image_vector_size = None
        self.device = None
        
        # Stats
        self.indexing_stats = {
            'total_indexed': 0,
            'last_build_time': None,
            'text_embedding_time': 0,
            'image_embedding_time': 0,
            'index_time': 0
        }
        
        self._initialize_components()

    def _initialize_components(self):
        """🚀 Initialize vectorizers and Qdrant client"""
        print("🤖 Loading image and text embedding models...")
        
        try:
            # Initialize text vectorizer
            self.text_vectorizer = SentenceTransformer(self.text_model_name)
            self.text_vector_size = self.text_vectorizer.get_sentence_embedding_dimension()
            print(f"✅ Text vectorizer loaded. Vector size: {self.text_vector_size}")
            
            # Initialize image model (CLIP)
            if clip and torch:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.image_model, self.image_preprocess = clip.load(self.image_model_name, device=self.device)
                self.image_vector_size = self.image_model.visual.output_dim
                print(f"✅ Image model (CLIP) loaded on {self.device}. Vector size: {self.image_vector_size}")
            else:
                print("⚠️ CLIP not available. Image similarity search will be limited to text descriptions.")
                self.image_vector_size = self.text_vector_size  # Fallback to text embeddings
            
            # Initialize Qdrant client
            os.makedirs(self.qdrant_path, exist_ok=True)
            self.qdrant_client = QdrantClient(path=self.qdrant_path)
            print(f"✅ Connected to Qdrant at {self.qdrant_path}")
            
            # Ensure collections exist
            self._ensure_collections()
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _ensure_collections(self):
        """🏗️ Ensure the Qdrant collections exist"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            # Text description collection
            text_collection = f"{self.collection_name}_text"
            if text_collection not in collection_names:
                print(f"🏗️ Creating text collection '{text_collection}'...")
                self.qdrant_client.create_collection(
                    collection_name=text_collection,
                    vectors_config=VectorParams(
                        size=self.text_vector_size,
                        distance=Distance.COSINE,
                        on_disk=True
                    )
                )
                print(f"✅ Text collection '{text_collection}' created")
            
            # Image visual collection (if CLIP is available)
            if self.image_model:
                image_collection = f"{self.collection_name}_image"
                if image_collection not in collection_names:
                    print(f"🏗️ Creating image collection '{image_collection}'...")
                    self.qdrant_client.create_collection(
                        collection_name=image_collection,
                        vectors_config=VectorParams(
                            size=self.image_vector_size,
                            distance=Distance.COSINE,
                            on_disk=True
                        )
                    )
                    print(f"✅ Image collection '{image_collection}' created")
                
        except Exception as e:
            logger.error(f"Error ensuring collections: {e}")
            raise

    async def build_index(self, chunks: List[ImageChunk], 
                         collection_name: Optional[str] = None,
                         batch_size: int = 32) -> bool:
        """🏗️ Build vector index from image chunks"""
        if not chunks:
            print("❌ No image chunks provided for indexing")
            return False
        
        collection = collection_name or self.collection_name
        
        print(f"🏗️ Building vector index for {len(chunks)} image chunks...")
        start_time = time.time()
        
        try:
            # Step 1: Create text embeddings from AI descriptions
            print("🧠 Creating text embeddings from AI descriptions...")
            text_embedding_start = time.time()
            
            descriptions = [chunk.ai_description for chunk in chunks]
            text_embeddings = self.text_vectorizer.encode(
                descriptions,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            text_embedding_time = time.time() - text_embedding_start
            self.indexing_stats['text_embedding_time'] = text_embedding_time
            print(f"✅ Text embeddings created in {text_embedding_time:.2f}s")
            
            # Step 2: Create image embeddings (if available)
            image_embeddings = None
            if self.image_model and any(chunk.image_data for chunk in chunks):
                print("🖼️ Creating visual embeddings from images...")
                image_embedding_start = time.time()
                
                image_embeddings = []
                for chunk in chunks:
                    if chunk.image_data:
                        try:
                            img_embedding = await self._create_image_embedding(chunk.image_data)
                            image_embeddings.append(img_embedding)
                        except Exception as e:
                            logger.warning(f"Failed to create image embedding for {chunk.source_file}: {e}")
                            # Use zero vector as fallback
                            image_embeddings.append(np.zeros(self.image_vector_size))
                    else:
                        # No image data available, use zero vector
                        image_embeddings.append(np.zeros(self.image_vector_size))
                
                image_embeddings = np.array(image_embeddings)
                
                image_embedding_time = time.time() - image_embedding_start
                self.indexing_stats['image_embedding_time'] = image_embedding_time
                print(f"✅ Image embeddings created in {image_embedding_time:.2f}s")
            
            # Step 3: Upload to Qdrant
            print("🚀 Uploading to vector database...")
            index_start = time.time()
            
            # Upload text embeddings
            await self._upload_text_embeddings(chunks, text_embeddings, f"{collection}_text")
            
            # Upload image embeddings (if available)
            if image_embeddings is not None:
                await self._upload_image_embeddings(chunks, image_embeddings, f"{collection}_image")
            
            index_time = time.time() - index_start
            self.indexing_stats['index_time'] = index_time
            
            # Update stats
            total_time = time.time() - start_time
            self.indexing_stats['total_indexed'] = len(chunks)
            self.indexing_stats['last_build_time'] = total_time
            
            print(f"🎉 Index built successfully!")
            print(f"📊 Stats: {len(chunks)} image chunks in {total_time:.2f}s")
            print(f"   • Text embedding: {text_embedding_time:.2f}s")
            if image_embeddings is not None:
                print(f"   • Image embedding: {image_embedding_time:.2f}s")
            print(f"   • Indexing: {index_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            print(f"❌ Index building failed: {e}")
            return False

    async def _create_image_embedding(self, image_data: str) -> np.ndarray:
        """Create CLIP embedding from base64 image data"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Preprocess and encode
            image_input = self.image_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.image_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error creating image embedding: {e}")
            raise

    async def _upload_text_embeddings(self, chunks: List[ImageChunk], 
                                    embeddings: np.ndarray, collection_name: str):
        """Upload text embeddings to Qdrant"""
        points = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create comprehensive payload
            payload = {
                'image_path': chunk.image_path,
                'source_file': os.path.basename(chunk.source_file),
                'chunk_id': chunk.chunk_id,
                'ai_description': chunk.ai_description,
                
                # Image metadata
                'width': chunk.metadata.get('width', 0),
                'height': chunk.metadata.get('height', 0),
                'file_size': chunk.metadata.get('file_size', 0),
                'file_format': chunk.metadata.get('file_format', 'Unknown'),
                
                # AI analysis
                'ai_analysis_type': chunk.ai_analysis.get('analysis_type', 'unknown'),
                'detected_categories': chunk.ai_analysis.get('detected_categories', []),
                'extracted_keywords': chunk.ai_analysis.get('extracted_keywords', []),
                
                # Searchable fields
                'has_text': 'text' in chunk.ai_description.lower(),
                'has_ui_elements': any(cat in chunk.ai_analysis.get('detected_categories', []) 
                                     for cat in ['ui_element', 'screenshot']),
                'has_diagram': 'diagram' in chunk.ai_analysis.get('detected_categories', []),
                'has_code': 'code' in chunk.ai_analysis.get('detected_categories', []),
                
                # Color information
                'brightness': chunk.metadata.get('brightness', 0),
                'is_grayscale': chunk.metadata.get('is_grayscale', False),
                
                # Processing info
                'processing_timestamp': time.time(),
                'embedding_type': 'text_description'
            }
            
            # Add AI analysis metadata
            for key, value in chunk.ai_analysis.items():
                if key not in payload and isinstance(value, (str, int, float, bool)):
                    payload[f"ai_{key}"] = value
            
            point = PointStruct(
                id=f"text_{chunk.chunk_id}",
                vector=embedding.tolist(),
                payload=payload
            )
            
            points.append(point)
        
        # Upload in batches
        upload_batch_size = 100
        for i in range(0, len(points), upload_batch_size):
            batch = points[i:i + upload_batch_size]
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True
            )
            
            print(f"📤 Uploaded text batch {i//upload_batch_size + 1}/{(len(points)-1)//upload_batch_size + 1}")

    async def _upload_image_embeddings(self, chunks: List[ImageChunk], 
                                     embeddings: np.ndarray, collection_name: str):
        """Upload image embeddings to Qdrant"""
        points = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create payload for image embeddings
            payload = {
                'image_path': chunk.image_path,
                'source_file': os.path.basename(chunk.source_file),
                'chunk_id': chunk.chunk_id,
                'ai_description': chunk.ai_description,
                
                # Image metadata
                'width': chunk.metadata.get('width', 0),
                'height': chunk.metadata.get('height', 0),
                'file_size': chunk.metadata.get('file_size', 0),
                'file_format': chunk.metadata.get('file_format', 'Unknown'),
                
                # Visual features
                'aspect_ratio': chunk.metadata.get('width', 1) / max(chunk.metadata.get('height', 1), 1),
                'pixel_count': chunk.metadata.get('width', 0) * chunk.metadata.get('height', 0),
                
                # Processing info
                'processing_timestamp': time.time(),
                'embedding_type': 'visual_clip'
            }
            
            point = PointStruct(
                id=f"image_{chunk.chunk_id}",
                vector=embedding.tolist(),
                payload=payload
            )
            
            points.append(point)
        
        # Upload in batches
        upload_batch_size = 100
        for i in range(0, len(points), upload_batch_size):
            batch = points[i:i + upload_batch_size]
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True
            )
            
            print(f"📤 Uploaded image batch {i//upload_batch_size + 1}/{(len(points)-1)//upload_batch_size + 1}")

    async def search(self, query: str, k: int = 5, 
                    search_type: str = 'both',  # 'text', 'image', 'both'
                    filter_conditions: Optional[Dict] = None,
                    collection_name: Optional[str] = None) -> List[Dict]:
        """🔍 Search the image vector database"""
        if not query.strip():
            return []
        
        collection = collection_name or self.collection_name
        results = []
        
        try:
            # Search text descriptions
            if search_type in ['text', 'both']:
                text_results = await self._search_text_descriptions(
                    query, k, filter_conditions, f"{collection}_text"
                )
                results.extend(text_results)
            
            # Search visual similarity (if available and requested)
            if search_type in ['image', 'both'] and self.image_model:
                # For image search, we'll encode the text query using CLIP's text encoder
                image_results = await self._search_visual_similarity(
                    query, k, filter_conditions, f"{collection}_image"
                )
                results.extend(image_results)
            
            # Deduplicate and re-rank if searching both
            if search_type == 'both':
                results = self._merge_and_rerank_results(results, k)
            else:
                results = sorted(results, key=lambda x: x['score'], reverse=True)[:k]
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    async def _search_text_descriptions(self, query: str, k: int, 
                                      filter_conditions: Optional[Dict], 
                                      collection_name: str) -> List[Dict]:
        """Search based on text descriptions"""
        try:
            # Create query embedding
            query_embedding = self.text_vectorizer.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            # Prepare filters
            search_filter = self._prepare_search_filters(filter_conditions)
            
            # Perform search
            search_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=k,
                with_payload=True
            )
            
            # Format results
            results = []
            for hit in search_results:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload,
                    'search_type': 'text_description'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in text description search: {e}")
            return []

    async def _search_visual_similarity(self, query: str, k: int, 
                                       filter_conditions: Optional[Dict], 
                                       collection_name: str) -> List[Dict]:
        """Search based on visual similarity using CLIP text encoder"""
        try:
            # Encode text query using CLIP text encoder
            text_tokens = clip.tokenize([query]).to(self.device)
            
            with torch.no_grad():
                text_features = self.image_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            query_embedding = text_features.cpu().numpy().flatten()
            
            # Prepare filters
            search_filter = self._prepare_search_filters(filter_conditions)
            
            # Perform search
            search_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=k,
                with_payload=True
            )
            
            # Format results
            results = []
            for hit in search_results:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload,
                    'search_type': 'visual_similarity'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in visual similarity search: {e}")
            return []

    def _prepare_search_filters(self, filter_conditions: Optional[Dict]) -> Optional[models.Filter]:
        """Prepare Qdrant search filters"""
        if not filter_conditions:
            return None
        
        must_conditions = []
        
        for key, value in filter_conditions.items():
            try:
                if isinstance(value, bool):
                    condition = models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                elif isinstance(value, (int, float)):
                    condition = models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                elif isinstance(value, str):
                    condition = models.FieldCondition(
                        key=key,
                        match=models.MatchText(text=value)
                    )
                elif isinstance(value, list):
                    condition = models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                else:
                    continue
                
                must_conditions.append(condition)
                
            except Exception as e:
                logger.warning(f"Failed to create filter condition for {key}={value}: {e}")
                continue
        
        if must_conditions:
            return models.Filter(must=must_conditions)
        
        return None

    def _merge_and_rerank_results(self, results: List[Dict], k: int) -> List[Dict]:
        """Merge and re-rank results from different search types"""
        # Group by chunk_id
        result_groups = {}
        
        for result in results:
            chunk_id = result['payload'].get('chunk_id')
            if not chunk_id:
                continue
                
            if chunk_id not in result_groups:
                result_groups[chunk_id] = {
                    'results': [],
                    'best_score': 0,
                    'search_types': set()
                }
            
            result_groups[chunk_id]['results'].append(result)
            result_groups[chunk_id]['best_score'] = max(
                result_groups[chunk_id]['best_score'], 
                result['score']
            )
            result_groups[chunk_id]['search_types'].add(result['search_type'])
        
        # Calculate combined scores and create final results
        final_results = []
        
        for chunk_id, group in result_groups.items():
            # Use the best result as the base
            best_result = max(group['results'], key=lambda x: x['score'])
            
            # Boost score if found by multiple search types
            boost_factor = 1.0 + (len(group['search_types']) - 1) * 0.1
            combined_score = best_result['score'] * boost_factor
            
            # Create combined result
            final_result = best_result.copy()
            final_result['score'] = combined_score
            final_result['found_by_search_types'] = list(group['search_types'])
            final_result['search_type_count'] = len(group['search_types'])
            
            final_results.append(final_result)
        
        # Sort by combined score and return top k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:k]

    async def search_by_image(self, image_path: str, k: int = 5,
                            filter_conditions: Optional[Dict] = None) -> List[Dict]:
        """🖼️ Search for similar images using an image as query"""
        if not self.image_model:
            raise Exception("Image model (CLIP) not available for image-based search")
        
        try:
            # Load and preprocess the query image
            with Image.open(image_path) as query_image:
                image_input = self.image_preprocess(query_image).unsqueeze(0).to(self.device)
            
            # Create image embedding
            with torch.no_grad():
                image_features = self.image_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            query_embedding = image_features.cpu().numpy().flatten()
            
            # Search in image collection
            search_filter = self._prepare_search_filters(filter_conditions)
            
            search_results = self.qdrant_client.search(
                collection_name=f"{self.collection_name}_image",
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=k,
                with_payload=True
            )
            
            # Format results
            results = []
            for hit in search_results:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload,
                    'search_type': 'image_similarity'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image-based search: {e}")
            return []

    def search_with_filters(self, query: str, 
                           has_text: Optional[bool] = None,
                           has_ui_elements: Optional[bool] = None,
                           has_diagram: Optional[bool] = None,
                           has_code: Optional[bool] = None,
                           file_format: Optional[str] = None,
                           min_width: Optional[int] = None,
                           min_height: Optional[int] = None,
                           k: int = 5,
                           search_type: str = 'both') -> List[Dict]:
        """🔍 Search with common filter shortcuts"""
        filters = {}
        
        if has_text is not None:
            filters['has_text'] = has_text
        if has_ui_elements is not None:
            filters['has_ui_elements'] = has_ui_elements
        if has_diagram is not None:
            filters['has_diagram'] = has_diagram
        if has_code is not None:
            filters['has_code'] = has_code
        if file_format is not None:
            filters['file_format'] = file_format
        
        # For size filters, we'd need range filtering (simplified for now)
        if min_width is not None or min_height is not None:
            logger.warning("Size range filtering not implemented in this version")
        
        return asyncio.run(self.search(query, k=k, search_type=search_type, filter_conditions=filters))

    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict:
        """📊 Get information about the collections"""
        collection = collection_name or self.collection_name
        
        try:
            info = {}
            
            # Text collection info
            try:
                text_info = self.qdrant_client.get_collection(f"{collection}_text")
                info['text_collection'] = {
                    'name': f"{collection}_text",
                    'status': str(text_info.status),
                    'points_count': text_info.points_count,
                    'vectors_count': text_info.vectors_count,
                    'vector_size': self.text_vector_size
                }
            except Exception as e:
                logger.warning(f"Could not get text collection info: {e}")
            
            # Image collection info
            if self.image_model:
                try:
                    image_info = self.qdrant_client.get_collection(f"{collection}_image")
                    info['image_collection'] = {
                        'name': f"{collection}_image",
                        'status': str(image_info.status),
                        'points_count': image_info.points_count,
                        'vectors_count': image_info.vectors_count,
                        'vector_size': self.image_vector_size
                    }
                except Exception as e:
                    logger.warning(f"Could not get image collection info: {e}")
            
            info.update({
                'text_model': self.text_model_name,
                'image_model': self.image_model_name if self.image_model else 'Not available',
                'device': self.device if self.image_model else 'CPU only',
                'storage_path': self.qdrant_path
            })
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def get_indexing_stats(self) -> Dict:
        """📈 Get indexing statistics"""
        return self.indexing_stats.copy()

    def clear_collections(self, collection_name: Optional[str] = None) -> bool:
        """🗑️ Clear all data from collections"""
        collection = collection_name or self.collection_name
        
        try:
            # Clear text collection
            try:
                self.qdrant_client.delete_collection(f"{collection}_text")
                print(f"✅ Text collection '{collection}_text' cleared")
            except Exception as e:
                logger.warning(f"Could not clear text collection: {e}")
            
            # Clear image collection
            if self.image_model:
                try:
                    self.qdrant_client.delete_collection(f"{collection}_image")
                    print(f"✅ Image collection '{collection}_image' cleared")
                except Exception as e:
                    logger.warning(f"Could not clear image collection: {e}")
            
            # Recreate collections
            self._ensure_collections()
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collections: {e}")
            return False

    def export_vectors(self, output_file: str, 
                      collection_name: Optional[str] = None,
                      export_type: str = 'text') -> bool:
        """💾 Export vectors to file"""
        collection = collection_name or self.collection_name
        full_collection_name = f"{collection}_{export_type}"
        
        try:
            # Get all points
            result = self.qdrant_client.scroll(
                collection_name=full_collection_name,
                limit=10000,  # Adjust as needed
                with_payload=True,
                with_vectors=True
            )
            
            points = result[0]
            
            # Export to JSON
            import json
            export_data = []
            
            for point in points:
                export_data.append({
                    'id': point.id,
                    'vector': point.vector,
                    'payload': point.payload
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"📁 Exported {len(export_data)} {export_type} vectors to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting vectors: {e}")
            return False