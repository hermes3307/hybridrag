#!/usr/bin/env python3
"""
🗄️ Vector Store Manager
Manages vector embeddings and similarity search with Qdrant
"""

import os
import asyncio
import numpy as np
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
import time

try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError as e:
    print("❌ Missing required packages. Please install:")
    print("pip install sentence-transformers qdrant-client")
    raise e

from smart_chunker import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """🗄️ Manages vector embeddings and search operations"""
    
    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 qdrant_path: str = "./qdrant_vector",
                 collection_name: str = "conversation_docs"):
        
        self.model_name = model_name
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        
        # Initialize components
        self.vectorizer = None
        self.qdrant_client = None
        self.vector_size = None
        
        # Stats
        self.indexing_stats = {
            'total_indexed': 0,
            'last_build_time': None,
            'embedding_time': 0,
            'index_time': 0
        }
        
        self._initialize_components()

    def _initialize_components(self):
        """🚀 Initialize vectorizer and Qdrant client"""
        print("🤖 Loading sentence transformer model...")
        
        try:
            # Initialize vectorizer
            self.vectorizer = SentenceTransformer(self.model_name)
            self.vector_size = self.vectorizer.get_sentence_embedding_dimension()
            print(f"✅ Vectorizer loaded. Vector size: {self.vector_size}")
            
            # Initialize Qdrant client
            os.makedirs(self.qdrant_path, exist_ok=True)
            self.qdrant_client = QdrantClient(path=self.qdrant_path)
            print(f"✅ Connected to Qdrant at {self.qdrant_path}")
            
            # Ensure collection exists
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _ensure_collection(self):
        """🏗️ Ensure the Qdrant collection exists"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                print(f"🏗️ Creating collection '{self.collection_name}'...")
                
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        on_disk=True
                    )
                )
                print(f"✅ Collection '{self.collection_name}' created")
            else:
                print(f"✅ Collection '{self.collection_name}' exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise

    async def build_index(self, chunks: List[DocumentChunk], 
                         collection_name: Optional[str] = None,
                         batch_size: int = 32) -> bool:
        """🏗️ Build vector index from document chunks"""
        if not chunks:
            print("❌ No chunks provided for indexing")
            return False
        
        collection = collection_name or self.collection_name
        
        print(f"🏗️ Building vector index for {len(chunks)} chunks...")
        start_time = time.time()
        
        try:
            # Step 1: Create embeddings
            print("🧠 Creating embeddings...")
            embedding_start = time.time()
            
            texts = [chunk.text for chunk in chunks]
            embeddings = self.vectorizer.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            embedding_time = time.time() - embedding_start
            self.indexing_stats['embedding_time'] = embedding_time
            print(f"✅ Embeddings created in {embedding_time:.2f}s")
            
            # Step 2: Prepare points for Qdrant
            print("📦 Preparing data points...")
            points = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create payload with rich metadata
                payload = {
                    'text': chunk.text,
                    'source_file': os.path.basename(chunk.source_file),
                    'chunk_index': chunk.chunk_index,
                    'chunk_id': chunk.chunk_id,
                    
                    # Metadata
                    'source_type': chunk.metadata.get('source_type', 'Unknown'),
                    'word_count': chunk.metadata.get('word_count', 0),
                    'chunk_size': chunk.metadata.get('chunk_size', 0),
                    
                    # Processing info
                    'processing_params': chunk.metadata.get('processing_params', {}),
                    
                    # Additional searchable fields
                    'has_code': 'code' in chunk.text.lower() or '```' in chunk.text,
                    'has_table': '|' in chunk.text or 'table' in chunk.text.lower(),
                    'is_header': chunk.text.isupper() or chunk.text.startswith('#'),
                    
                    # Text statistics
                    'char_count': len(chunk.text),
                    'sentence_count': len([s for s in chunk.text.split('.') if s.strip()]),
                    
                    # Timestamp
                    'indexed_at': time.time()
                }
                
                # Add file-specific metadata
                for key, value in chunk.metadata.items():
                    if key not in payload and isinstance(value, (str, int, float, bool)):
                        payload[f"meta_{key}"] = value
                
                point = PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
                
                points.append(point)
            
            # Step 3: Upload to Qdrant
            print("🚀 Uploading to vector database...")
            index_start = time.time()
            
            # Upload in batches
            upload_batch_size = 100
            for i in range(0, len(points), upload_batch_size):
                batch = points[i:i + upload_batch_size]
                
                self.qdrant_client.upsert(
                    collection_name=collection,
                    points=batch,
                    wait=True
                )
                
                print(f"📤 Uploaded batch {i//upload_batch_size + 1}/{(len(points)-1)//upload_batch_size + 1}")
            
            index_time = time.time() - index_start
            self.indexing_stats['index_time'] = index_time
            
            # Update stats
            total_time = time.time() - start_time
            self.indexing_stats['total_indexed'] = len(chunks)
            self.indexing_stats['last_build_time'] = total_time
            
            print(f"🎉 Index built successfully!")
            print(f"📊 Stats: {len(chunks)} chunks in {total_time:.2f}s")
            print(f"   • Embedding: {embedding_time:.2f}s")
            print(f"   • Indexing: {index_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            print(f"❌ Index building failed: {e}")
            return False

    async def search(self, query: str, k: int = 5, 
                    filter_conditions: Optional[Dict] = None,
                    collection_name: Optional[str] = None) -> List[Dict]:
        """🔍 Search the vector database"""
        if not query.strip():
            return []
        
        collection = collection_name or self.collection_name
        
        try:
            # Create query embedding
            query_embedding = self.vectorizer.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            # Prepare filters
            search_filter = None
            if filter_conditions:
                must_conditions = []
                
                for key, value in filter_conditions.items():
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
                    else:
                        continue
                    
                    must_conditions.append(condition)
                
                if must_conditions:
                    search_filter = models.Filter(must=must_conditions)
            
            # Perform search
            search_results = self.qdrant_client.search(
                collection_name=collection,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=k,
                with_payload=True
            )
            
            # Format results
            results = []
            for hit in search_results:
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict:
        """📊 Get information about the collection"""
        collection = collection_name or self.collection_name
        
        try:
            info = self.qdrant_client.get_collection(collection)
            
            return {
                'name': collection,
                'status': str(info.status),
                'points_count': info.points_count,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'segments_count': info.segments_count,
                'vector_size': self.vector_size,
                'distance_metric': 'COSINE',
                'storage_path': self.qdrant_path
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def search_with_filters(self, query: str, 
                           has_code: Optional[bool] = None,
                           has_table: Optional[bool] = None,
                           source_type: Optional[str] = None,
                           min_word_count: Optional[int] = None,
                           k: int = 5) -> List[Dict]:
        """🔍 Search with common filter shortcuts"""
        filters = {}
        
        if has_code is not None:
            filters['has_code'] = has_code
        if has_table is not None:
            filters['has_table'] = has_table
        if source_type is not None:
            filters['source_type'] = source_type
        if min_word_count is not None:
            # This would need range filtering - simplified for now
            pass
        
        return asyncio.run(self.search(query, k=k, filter_conditions=filters))

    def get_indexing_stats(self) -> Dict:
        """📈 Get indexing statistics"""
        return self.indexing_stats.copy()

    def clear_collection(self, collection_name: Optional[str] = None) -> bool:
        """🗑️ Clear all data from collection"""
        collection = collection_name or self.collection_name
        
        try:
            self.qdrant_client.delete_collection(collection)
            self._ensure_collection()
            print(f"✅ Collection '{collection}' cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    def export_vectors(self, output_file: str, 
                      collection_name: Optional[str] = None) -> bool:
        """💾 Export vectors to file"""
        collection = collection_name or self.collection_name
        
        try:
            # Get all points
            result = self.qdrant_client.scroll(
                collection_name=collection,
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
            
            print(f"📁 Exported {len(export_data)} vectors to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting vectors: {e}")
            return False