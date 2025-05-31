import os
import json
import numpy as np
from tqdm import tqdm
import re
import time
import gc
import logging
import traceback
import hashlib
import psutil
import sys
import subprocess  # For checking/installing packages
import glob  # ✅ ADD: For lock file cleanup
from typing import List, Dict, Any, Tuple, Optional

# --- Package Installation Check ---
def check_and_install_packages():
    """Checks for required packages and suggests installation if missing."""
    required_packages = {
        "sentence_transformers": "sentence-transformers",
        "qdrant_client": "qdrant-client",
        "numpy": "numpy",
        "tqdm": "tqdm",
        "psutil": "psutil"
    }
    missing_packages = []
    try:
        import sentence_transformers
    except ImportError:
        missing_packages.append(required_packages["sentence_transformers"])
    try:
        import qdrant_client
    except ImportError:
        missing_packages.append(required_packages["qdrant_client"])
    # Other packages are common, but good to list
    try:
        import numpy
    except ImportError:
        missing_packages.append(required_packages["numpy"])
    try:
        import tqdm
    except ImportError:
        missing_packages.append(required_packages["tqdm"])
    try:
        import psutil
    except ImportError:
        missing_packages.append(required_packages["psutil"])

    if missing_packages:
        print("ERROR: The following required packages are missing:")
        for pkg_install_name in missing_packages:
            print(f"  - {pkg_install_name.split()[0]} (install with: pip install {pkg_install_name})")
        
        install_prompt = input("Do you want to attempt to install them now? (y/n): ").strip().lower()
        if install_prompt == 'y':
            for pkg_install_name in missing_packages:
                print(f"Installing {pkg_install_name}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_install_name])
                    print(f"Successfully installed {pkg_install_name}.")
                except subprocess.CalledProcessError as e:
                    print(f"ERROR: Failed to install {pkg_install_name}. Please install it manually: pip install {pkg_install_name}")
                    print(f"Error details: {e}")
                    sys.exit(1)
            print("All required packages checked/installed. Please re-run the script.")
            sys.exit(0) # Exit so user can re-run with packages loaded
        else:
            print("Please install the missing packages and try again.")
            sys.exit(1)
    else:
        print("All required packages are installed.")

check_and_install_packages()

# Now import them
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Distance, VectorParams, UpdateStatus
from typing import List, Dict, Any, Tuple, Optional

# 로깅 설정 (Logging Configuration)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 메모리 사용량 모니터링 (Memory Usage Monitoring)
def get_memory_usage():
    """현재 메모리 사용량 반환 (MB) (Returns current memory usage in MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# 유틸리티 함수 (Utility Functions)
def clean_text(text: str) -> str:
    """텍스트 정리: 불필요한 공백 제거 및 특수문자 처리 (Cleans text: removes unnecessary spaces and handles special characters)"""
    if not text or not isinstance(text, str):
        return ""
    
    # 마크다운 문법 제거 (Remove Markdown syntax)
    text = re.sub(r'#{1,6}\s+', '', text)  # 헤더 제거 (Remove headers)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # 링크 제거 (Remove links)
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)  # 코드 블록 제거 (Remove code blocks)
    text = re.sub(r'`[^`]+`', '', text)  # 인라인 코드 제거 (Remove inline code)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # 볼드/이탤릭 제거 (Remove bold/italic)
    
    text = re.sub(r'\s+', ' ', text) # 연속된 공백을 하나로 치환 (Replace multiple spaces with one)
    text = re.sub(r'\n+', '\n', text) # 줄바꿈 정리 (Normalize newlines)
    return text.strip()

def create_chunks_smart(text: str, max_chunk_size: int = 1000, overlap: int = 200, min_chunk_size: int = 100) -> List[str]:
    """
    텍스트를 스마트하게 청크로 분할 (문단, 문장 경계 고려)
    (Smartly splits text into chunks, considering paragraph and sentence boundaries)
    """
    if not text or not isinstance(text, str):
        return []
    
    text = text.strip()
    if not text:
        return []
        
    if len(text) <= max_chunk_size:
        return [text] if len(text) >= min_chunk_size else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        
        if end < len(text): # Try to find a natural break if not at the end
            # Prefer paragraph, then sentence, then newline, then space
            break_points = []
            for sep in ['\n\n', '. ', '! ', '? ', '\n', ' ']:
                p = text.rfind(sep, start, end)
                if p != -1 and p > start : # Ensure break is within the current segment and not at the very start
                    break_points.append(p + len(sep)) # Include separator in break point
            
            if break_points:
                end = max(break_points) # Take the latest possible natural break
            elif end - start < min_chunk_size and end < len(text): # If chunk is too small and not end of text, extend it
                end = min(start + min_chunk_size, len(text))

        chunk_text = text[start:end].strip()
        if chunk_text and len(chunk_text) >= min_chunk_size:
            chunks.append(chunk_text)
        
        next_start = end - overlap
        if next_start <= start : # Ensure progress
             next_start = start + 1 
        start = next_start
        
        if start >= len(text): # Break if we've processed the whole text
            break
            
    return [c for c in chunks if c] # Filter out any empty chunks that might have slipped through

def generate_chunk_hash(text: str) -> str:
    """청크의 해시값 생성 (중복 제거용) (Generates a hash for a chunk for deduplication)"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def log_memory_usage(step_name: str):
    """메모리 사용량 로깅 (Logs memory usage)"""
    memory_mb = get_memory_usage()
    logger.info(f"{step_name} - Memory Usage: {memory_mb:.1f} MB")

# Remove the duplicate function completely

def enhance_chunk_metadata(chunk_data: Dict) -> Dict:
    """Enhances chunk metadata with additional information for better search results"""
    enhanced_chunk = chunk_data.copy()
    
    # Extract existing metadata
    metadata = enhanced_chunk.get('metadata', {})
    
    # Add enhanced metadata fields
    enhanced_metadata = {
        # Core document info
        'manual_name': metadata.get('title', metadata.get('manual_title', 'Unknown Manual')),
        'manual_file': metadata.get('source_pdf', metadata.get('filename', 'Unknown File')),
        'page_number': metadata.get('page', metadata.get('page_number', 0)),
        
        # Chapter and section info
        'chapter_number': metadata.get('chapter_number', ''),
        'chapter_title': metadata.get('chapter', metadata.get('chapter_title', '')),
        'section_number': metadata.get('section_number', ''),
        'section_title': metadata.get('section_title', ''),
        'subsection_number': metadata.get('subsection_number', ''),
        'subsection_title': metadata.get('subsection_title', ''),
        
        # Content type and categorization
        'content_type': metadata.get('section_type', metadata.get('content_type', 'general')),
        'topic_category': metadata.get('topic_category', ''),
        'language': metadata.get('language', 'korean'),
        
        # Technical details
        'has_code_examples': 'code' in chunk_data.get('text', '').lower() or '```' in chunk_data.get('text', ''),
        'has_tables': 'table' in chunk_data.get('text', '').lower() or '|' in chunk_data.get('text', ''),
        'has_images': metadata.get('has_images', False),
        
        # Content length and quality metrics
        'text_length': len(chunk_data.get('text', '')),
        'word_count': len(chunk_data.get('text', '').split()),
        
        # Indexing info
        'chunk_index': metadata.get('chunk_index', 0),
        'total_chunks': metadata.get('total_chunks', 1),
        'processing_timestamp': metadata.get('processing_timestamp', time.time()),
    }
    
    # Merge with existing metadata
    enhanced_metadata.update(metadata)
    enhanced_chunk['metadata'] = enhanced_metadata
    
    # Add manual-level information to root level for easy access
    enhanced_chunk['manual_title'] = enhanced_metadata['manual_name']
    enhanced_chunk['section_key'] = f"{enhanced_metadata.get('chapter_number', '')}.{enhanced_metadata.get('section_number', '')}.{enhanced_metadata.get('subsection_number', '')}"
    
    return enhanced_chunk

def handle_qdrant_build_with_retry():
    """Enhanced build function with error handling and retry logic."""
    import argparse
    
    # Simple retry wrapper for the build process
    def build_with_recreation():
        try:
            # Try normal build first
            return build_qdrant_db(
                chunk_data_dir="chunked_data",
                qdrant_collection_name="altibase_manuals",
                qdrant_path="./qdrant_vector"
            )
        except Exception as e:
            if "CollectionParams" in str(e) or "size" in str(e):
                logger.warning("Detected API compatibility issue. Recreating collection...")
                
                # Try to recreate collection
                try:
                    from sentence_transformers import SentenceTransformer
                    vectorizer_temp = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                    
                    db_temp = QdrantVectorDB(
                        vector_size=vectorizer_temp.get_sentence_embedding_dimension(),
                        collection_name="altibase_manuals",
                        qdrant_path="./qdrant_vector"
                    )
                    
                    if db_temp.recreate_collection_if_needed():
                        logger.info("Collection recreated. Retrying build...")
                        return build_qdrant_db(
                            chunk_data_dir="chunked_data",
                            qdrant_collection_name="altibase_manuals", 
                            qdrant_path="./qdrant_vector"
                        )
                except Exception as recreate_error:
                    logger.error(f"Failed to recreate collection: {recreate_error}")
            
            raise  # Re-raise the original error

    return build_with_recreation()


# 매뉴얼 로더 클래스 (Manual Loader Class - adapted from your script)
class ManualLoader:
    def __init__(self, directory: str = "chunked_data"):  # ✅ CORRECT DEFAULT
        self.directory = directory
        self.all_chunks_data = []
        self.error_log = []
        self.processed_files = set()
        self.chunk_hashes = set()  # For deduplication across all files

    def load_chunked_data(self, max_files: Optional[int] = None) -> List[Dict]:
        """Loads pre-chunked data from JSON files in the directory with enhanced metadata."""
        if not os.path.exists(self.directory):
            logger.error(f"Chunk data directory does not exist: {self.directory}")
            print(f"ERROR: Chunk data directory '{self.directory}' not found. Please run the PDF processing script first.")
            return []

        files = [f for f in os.listdir(self.directory) if f.endswith('_chunks.json')]
        if not files:
            logger.warning(f"No '_chunks.json' files found in {self.directory}.")
            print(f"WARNING: No chunk files found in '{self.directory}'. Ensure PDFs have been processed and chunked.")
            return []

        if max_files:
            files = files[:max_files]

        logger.info(f"Found {len(files)} chunk JSON files to process.")
        log_memory_usage("Chunk JSON loading started")

        for filename in tqdm(files, desc="Loading chunked JSON files"):
            if filename in self.processed_files:
                continue
            
            file_path = os.path.join(self.directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    single_pdf_chunks = json.load(f) 
                
                if not isinstance(single_pdf_chunks, list):
                    logger.warning(f"Skipping {filename}: Expected a list of chunks, got {type(single_pdf_chunks)}")
                    self.error_log.append(f"Format error - {file_path}: Expected list.")
                    continue

                for chunk_data in single_pdf_chunks:
                    if not isinstance(chunk_data, dict) or 'text' not in chunk_data or 'metadata' not in chunk_data:
                        logger.warning(f"Skipping invalid chunk in {filename}: {str(chunk_data)[:100]}")
                        continue

                    # Enhance metadata
                    enhanced_chunk = enhance_chunk_metadata(chunk_data)
                    
                    # Add a unique hash for deduplication if not present
                    if 'chunk_hash' not in enhanced_chunk:
                        enhanced_chunk['chunk_hash'] = generate_chunk_hash(enhanced_chunk['text'])
                    
                    if enhanced_chunk['chunk_hash'] not in self.chunk_hashes:
                        self.all_chunks_data.append(enhanced_chunk)
                        self.chunk_hashes.add(enhanced_chunk['chunk_hash'])
                
                self.processed_files.add(filename)

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error - {file_path}: {e}")
                self.error_log.append(f"JSON parsing error - {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error processing file - {file_path}: {e}")
                self.error_log.append(f"File processing error - {file_path}: {e}")
        
        logger.info(f"Successfully loaded {len(self.all_chunks_data)} unique chunks from {len(self.processed_files)} files.")
        log_memory_usage("Chunk JSON loading completed")
        return self.all_chunks_data

# 벡터 임베딩 (Vector Embedding Class)
class ManualVectorizer:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", 
                 device: Optional[str] = None, cache_dir: Optional[str] = None):
        logger.info(f"Loading sentence transformer model: {model_name}...")
        log_memory_usage("Model loading started")
        
        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model '{model_name}' loaded. Vector size: {self.vector_size}, Device: {device}")
            log_memory_usage("Model loading completed")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            logger.error(traceback.format_exc())
            raise

    def vectorize_chunks_batch(self, chunks_data: List[Dict], batch_size: int = 32, 
                               show_progress: bool = True) -> Tuple[Optional[np.ndarray], List[Dict]]:
        if not chunks_data:
            return None, []
        
        logger.info(f"Vectorizing {len(chunks_data)} chunks...")
        log_memory_usage("Vectorization started")
        
        texts = [chunk['text'] for chunk in chunks_data]
        
        try:
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True # Important for cosine similarity
            )
            logger.info(f"Vectorization complete. Vector shape: {vectors.shape}")
            log_memory_usage("Vectorization completed")
            return vectors.astype('float32'), chunks_data
            
        except Exception as e:
            logger.error(f"Error during vectorization: {e}")
            logger.error(traceback.format_exc())
            if "out of memory" in str(e).lower() and batch_size > 1:
                logger.warning("Out of memory. Trying with smaller batch size.")
                return self.vectorize_chunks_batch(chunks_data, batch_size // 2, show_progress)
            return None, chunks_data

class QdrantVectorDB:
    def __init__(self, vector_size: int, collection_name: str = "altibase_manuals", 
                 qdrant_host: str = "localhost", qdrant_port: int = 6333, 
                 qdrant_path: str = "./qdrant_data", api_key: Optional[str] = None,
                 prefer_grpc: bool = True):
        self.vector_size = vector_size
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        self.is_persistent = True
        self.client = None
        
        # ✅ NEW: Clean up any existing lock files before connecting
        self._cleanup_existing_locks()
        
        # Ensure qdrant_path directory exists for persistence
        if qdrant_path and not os.path.exists(qdrant_path):
            os.makedirs(qdrant_path, exist_ok=True)
            logger.info(f"Created Qdrant data directory: {qdrant_path}")
        
        logger.info(f"Initializing Qdrant client with persistent storage at: {qdrant_path}")
        
        try:
            from qdrant_client import QdrantClient
            
            if qdrant_path:
                self.client = QdrantClient(path=qdrant_path)
                logger.info(f"Using Qdrant with persistent storage at: {qdrant_path}")
            else:
                self.client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=api_key, prefer_grpc=prefer_grpc)
                logger.info(f"Connected to Qdrant server at {qdrant_host}:{qdrant_port}")
                self.is_persistent = False
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

        self.ensure_collection()

    def _cleanup_existing_locks(self):
        """✅ NEW: Clean up any existing lock files before connecting."""
        if not self.qdrant_path or not os.path.exists(self.qdrant_path):
            return
            
        try:
            import glob
            
            # Find and remove various types of lock files
            lock_patterns = [
                os.path.join(self.qdrant_path, "*.lock"),
                os.path.join(self.qdrant_path, ".lock"),
                os.path.join(self.qdrant_path, "**", "*.lock"),
                os.path.join(self.qdrant_path, "**", ".lock"),
                os.path.join(self.qdrant_path, "storage.lock"),
                os.path.join(self.qdrant_path, ".qdrant.lock")
            ]
            
            removed_files = []
            for pattern in lock_patterns:
                for lock_file in glob.glob(pattern, recursive=True):
                    try:
                        os.remove(lock_file)
                        removed_files.append(lock_file)
                        logger.info(f"Removed lock file: {lock_file}")
                    except OSError as e:
                        logger.warning(f"Could not remove lock file {lock_file}: {e}")
            
            if removed_files:
                logger.info(f"Cleaned up {len(removed_files)} lock files before connecting")
                
        except Exception as e:
            logger.warning(f"Error during lock cleanup: {e}")

    def close(self):
        """✅ NEW: Properly close the Qdrant client connection."""
        if self.client:
            try:
                # Force close any open connections
                self.client = None
                logger.info("Qdrant client connection closed")
            except Exception as e:
                logger.warning(f"Error closing Qdrant client: {e}")

    def __enter__(self):
        """✅ NEW: Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """✅ NEW: Context manager exit with cleanup."""
        self.close()

    def __del__(self):
        """✅ NEW: Cleanup when object is destroyed."""
        self.close()

    # ✅ UPDATED: All your existing methods remain the same
    def ensure_collection(self):
        """Ensures the collection exists in Qdrant, creates it if not."""
        try:
            from qdrant_client.http.models import Distance, VectorParams
            
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Collection '{self.collection_name}' not found. Creating it...")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, 
                        distance=Distance.COSINE,
                        on_disk=True
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully with on-disk storage.")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
                collection_info = self.client.get_collection(self.collection_name)
                current_points = collection_info.points_count
                logger.info(f"Existing collection has {current_points} points.")
                
        except Exception as e:
            logger.error(f"Error ensuring collection '{self.collection_name}': {e}")
            logger.error(traceback.format_exc())
            raise


    def upsert_vectors(self, vectors: np.ndarray, chunks_data: List[Dict]):
        """Upserts vectors and their enhanced payloads into Qdrant."""
        if vectors is None or len(vectors) == 0 or len(vectors) != len(chunks_data):
            logger.warning("No valid vectors or mismatched vectors and chunks_data. Skipping upsert.")
            return

        logger.info(f"Upserting {len(vectors)} vectors into Qdrant collection '{self.collection_name}'...")
        log_memory_usage("Qdrant upsert started")

        from qdrant_client.http.models import PointStruct, UpdateStatus
        
        points_to_upsert = []
        for i, (vector, chunk_meta) in enumerate(zip(vectors, chunks_data)):
            point_id = chunk_meta.get('chunk_hash', hashlib.md5(str(i).encode() + chunk_meta['text'].encode()).hexdigest())
            
            # Prepare enhanced payload for better search results
            payload = {
                'text': chunk_meta['text'],
                'manual_title': chunk_meta.get('manual_title', ''),
                'section_key': chunk_meta.get('section_key', ''),
                'metadata': chunk_meta.get('metadata', {}),
                
                # Flatten important metadata for easier filtering
                'manual_name': chunk_meta.get('metadata', {}).get('manual_name', ''),
                'manual_file': chunk_meta.get('metadata', {}).get('manual_file', ''),
                'page_number': chunk_meta.get('metadata', {}).get('page_number', 0),
                'chapter_number': chunk_meta.get('metadata', {}).get('chapter_number', ''),
                'chapter_title': chunk_meta.get('metadata', {}).get('chapter_title', ''),
                'section_number': chunk_meta.get('metadata', {}).get('section_number', ''),
                'section_title': chunk_meta.get('metadata', {}).get('section_title', ''),
                'subsection_number': chunk_meta.get('metadata', {}).get('subsection_number', ''),
                'subsection_title': chunk_meta.get('metadata', {}).get('subsection_title', ''),
                'content_type': chunk_meta.get('metadata', {}).get('content_type', 'general'),
                'has_code_examples': chunk_meta.get('metadata', {}).get('has_code_examples', False),
                'has_tables': chunk_meta.get('metadata', {}).get('has_tables', False),
                'word_count': chunk_meta.get('metadata', {}).get('word_count', 0),
            }
            
            points_to_upsert.append(PointStruct(id=point_id, vector=vector.tolist(), payload=payload))

        batch_size = 100 
        for i in tqdm(range(0, len(points_to_upsert), batch_size), desc="Upserting to Qdrant"):
            batch = points_to_upsert[i:i + batch_size]
            try:
                response = self.client.upsert(collection_name=self.collection_name, points=batch, wait=True)
                if response.status != UpdateStatus.COMPLETED:
                    logger.warning(f"Qdrant upsert batch {i//batch_size} did not complete successfully: {response.status}")
            except Exception as e:
                logger.error(f"Error upserting batch {i//batch_size} to Qdrant: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"Upsert complete for {len(points_to_upsert)} points.")
        log_memory_usage("Qdrant upsert completed")
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' now contains {collection_info.points_count} points.")
        except Exception as e:
            logger.warning(f"Could not get updated point count for collection: {e}")

    def search(self, query_text: str, vectorizer, k: int = 5, 
               filter_payload: Optional[Dict] = None) -> List[Dict]:
        """Searches Qdrant for similar vectors with enhanced result information."""
        if not query_text:
            return []
        
        logger.info(f"Searching Qdrant for query: '{query_text[:50]}...' with k={k}")
        query_vector_np, _ = vectorizer.vectorize_chunks_batch([{"text": query_text}], show_progress=False)
        
        if query_vector_np is None or query_vector_np.ndim == 0:
            logger.error("Could not vectorize query text.")
            return []
        
        query_vector = query_vector_np[0].tolist()

        try:
            from qdrant_client import models
            
            search_filter = None
            if filter_payload:
                must_conditions = []
                for key, value in filter_payload.items():
                    if isinstance(value, bool):
                        must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
                    elif isinstance(value, (int, float)):
                        must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
                    else:
                        must_conditions.append(models.FieldCondition(key=key, match=models.MatchText(text=str(value))))
                
                if must_conditions:
                    search_filter = models.Filter(must=must_conditions)

            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=k,
                with_payload=True 
            )
            
            results = []
            for hit in hits:
                results.append({
                    'id': hit.id,
                    'score': hit.score, 
                    'payload': hit.payload 
                })
            logger.info(f"Qdrant search returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during Qdrant search: {e}")
            logger.error(traceback.format_exc())
            return []

    def get_collection_info_dict(self) -> Optional[Dict]:
        """Get detailed collection information including persistence status - Fixed API compatibility."""
        try:
            info = self.client.get_collection(self.collection_name)
            
            # Try to get vector configuration information safely
            vector_info = "Unknown"
            try:
                vector_config = info.config.params.vectors
                if hasattr(vector_config, 'size'):
                    vector_info = f"Size: {vector_config.size}, Distance: {vector_config.distance}"
                elif isinstance(vector_config, dict):
                    # Handle named vectors or different structure
                    if len(vector_config) > 0:
                        first_config = list(vector_config.values())[0]
                        if hasattr(first_config, 'size'):
                            vector_info = f"Size: {first_config.size}, Distance: {first_config.distance}"
                        elif isinstance(first_config, dict):
                            size = first_config.get('size', 'Unknown')
                            distance = first_config.get('distance', 'Unknown')
                            vector_info = f"Size: {size}, Distance: {distance}"
            except Exception:
                vector_info = "Could not retrieve vector config"
            
            return {
                "name": self.collection_name,
                "status": str(info.status),
                "optimizer_status": str(info.optimizer_status),
                "vectors_count": info.vectors_count, 
                "indexed_vectors_count": info.indexed_vectors_count, 
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "vector_config": vector_info,
                "persistence_path": self.qdrant_path if self.is_persistent else "Server/In-Memory",
                "is_persistent": self.is_persistent,
                "payload_schema": {k: str(v) for k,v in info.payload_schema.items()} if info.payload_schema else {}
            }
        except Exception as e:
            logger.error(f"Could not get info for collection {self.collection_name}: {e}")
            return None

    def recreate_collection_if_needed(self):
        """Recreate collection if there are compatibility issues."""
        try:
            logger.info(f"Recreating collection '{self.collection_name}' to ensure compatibility...")
            
            from qdrant_client.http.models import Distance, VectorParams
            
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, 
                    distance=Distance.COSINE,
                    on_disk=True
                )
            )
            logger.info(f"Collection '{self.collection_name}' recreated successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to recreate collection: {e}")
            return False

def build_qdrant_db(
    chunk_data_dir: str = "chunked_data", 
    qdrant_collection_name: str = "altibase_manuals",
    qdrant_host: str = "localhost", 
    qdrant_port: int = 6333,
    qdrant_path: str = "./qdrant_vector",  # ✅ FIX: Proper default
    embedding_batch_size: int = 32,
    max_files_to_load: Optional[int] = None
) -> Tuple[Optional[QdrantVectorDB], Optional[ManualVectorizer]]:  
    # ✅ FIX: Proper return type
    
    """✅ UPDATED: Loads chunks, vectorizes them, and upserts into Qdrant with proper cleanup."""
        
    vectorizer = None
    qdrant_db = None
    
    try:
        logger.info("=== Persistent Qdrant Vector Database Build Process Started ===")
        logger.info(f"Chunk Data Directory: {chunk_data_dir}")
        logger.info(f"Qdrant Collection: {qdrant_collection_name}")
        logger.info(f"Qdrant Persistence Path: {qdrant_path}")

        # Load chunks
        loader = ManualLoader(directory=chunk_data_dir)
        all_loaded_chunks = loader.load_chunked_data(max_files=max_files_to_load)

        if not all_loaded_chunks:
            logger.error("No chunks loaded. Aborting build process.")
            return None, None
        
        logger.info(f"Total unique chunks loaded: {len(all_loaded_chunks)}")

        # Initialize vectorizer
        vectorizer = ManualVectorizer()
        
        # Create and use QdrantVectorDB normally (no context manager in build)
        qdrant_db = QdrantVectorDB(
            vector_size=vectorizer.vector_size, 
            collection_name=qdrant_collection_name,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_path=qdrant_path
        )
        
        logger.info(f"Preparing to vectorize {len(all_loaded_chunks)} chunks for Qdrant.")
        
        vectors_np, chunks_for_db = vectorizer.vectorize_chunks_batch(
            all_loaded_chunks, 
            batch_size=embedding_batch_size, 
            show_progress=True
        )

        if vectors_np is not None and len(vectors_np) > 0:
            qdrant_db.upsert_vectors(vectors_np, chunks_for_db)
            logger.info("All processed chunks have been vectorized and upserted to Qdrant.")
        else:
            logger.error("Vectorization failed or produced no vectors.")
            return None, vectorizer

        logger.info("=== Qdrant Vector Database Build Process Completed ===")
        return qdrant_db, vectorizer

    except Exception as e:
        logger.error(f"Critical error during Qdrant DB build: {e}")
        logger.error(traceback.format_exc())
        return qdrant_db, vectorizer

def display_enhanced_search_results(results: List[Dict], show_full_text: bool = False):
    """Display search results with enhanced metadata information."""
    if not results:
        print("No results found.")
        return
    
    print(f"\n{'='*80}")
    print(f"                    SEARCH RESULTS ({len(results)} found)")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        payload = result.get('payload', {})
        score = result.get('score', 0)
        
        print(f"\n┌─ Result #{i} (Similarity Score: {score:.4f}) ─{'─'*(50-len(str(i)))}")
        print(f"│")
        print(f"│ 📖 Manual: {payload.get('manual_name', 'N/A')}")
        print(f"│ 📄 File: {payload.get('manual_file', 'N/A')}")
        print(f"│ 📃 Page: {payload.get('page_number', 'N/A')}")
        print(f"│")
        
        # Chapter and section information
        chapter_info = []
        if payload.get('chapter_number'):
            chapter_info.append(f"Ch.{payload.get('chapter_number')}")
        if payload.get('chapter_title'):
            chapter_info.append(payload.get('chapter_title'))
        
        section_info = []
        if payload.get('section_number'):
            section_info.append(f"Sec.{payload.get('section_number')}")
        if payload.get('section_title'):
            section_info.append(payload.get('section_title'))
            
        subsection_info = []
        if payload.get('subsection_number'):
            subsection_info.append(f"Sub.{payload.get('subsection_number')}")
        if payload.get('subsection_title'):
            subsection_info.append(payload.get('subsection_title'))
        
        if chapter_info:
            print(f"│ 📂 Chapter: {' - '.join(chapter_info)}")
        if section_info:
            print(f"│ 📑 Section: {' - '.join(section_info)}")
        if subsection_info:
            print(f"│ 📋 Subsection: {' - '.join(subsection_info)}")
        
        # Content type and features
        content_type = payload.get('content_type', 'general')
        features = []
        if payload.get('has_code_examples'):
            features.append("💻 Code")
        if payload.get('has_tables'):
            features.append("📊 Tables")
        
        print(f"│ 🏷️  Content Type: {content_type.title()}")
        if features:
            print(f"│ ✨ Features: {' | '.join(features)}")
        
        # Word count
        word_count = payload.get('word_count', 0)
        if word_count > 0:
            print(f"│ 📝 Words: {word_count}")
        
        print(f"│")
        print(f"│ 📄 Text Preview:")
        text_content = payload.get('text', '')
        preview_length = 200 if not show_full_text else len(text_content)
        preview_text = text_content[:preview_length]
        
        # Format text preview with proper indentation
        lines = preview_text.split('\n')
        for line in lines[:5]:  # Show max 5 lines in preview
            print(f"│    {line[:76]}")
        
        if len(text_content) > preview_length:
            print(f"│    ... (truncated, {len(text_content)} total chars)")
        
        print(f"└─{'─'*78}")
        
        # Ask if user wants to see full text for this result
        if not show_full_text and len(text_content) > 200:
            user_input = input(f"    📖 View full text for Result #{i}? (y/n/all): ").strip().lower()
            if user_input == 'y':
                print(f"\n📄 Full Text for Result #{i}:")
                print(f"{'─'*80}")
                print(text_content)
                print(f"{'─'*80}")
            elif user_input == 'all':
                show_full_text = True
                print(f"\n📄 Full Text for Result #{i}:")
                print(f"{'─'*80}")
                print(text_content)
                print(f"{'─'*80}")

def interactive_search_qdrant():
    """Enhanced interactive search interface using persistent Qdrant."""
    qdrant_db_instance: Optional[QdrantVectorDB] = None
    vectorizer_instance: Optional[ManualVectorizer] = None
    
    # Default to persistent storage
    q_path = "./qdrant_vector"
    q_collection = "altibase_manuals"
    chunk_json_dir = "chunked_data"

    def load_or_init_db_components(force_reinit=False):
        nonlocal qdrant_db_instance, vectorizer_instance
        try:
            # ✅ FIX: Close existing connection first
            if qdrant_db_instance is not None:
                qdrant_db_instance.close()
                qdrant_db_instance = None
            
            if vectorizer_instance is None or force_reinit:
                print("🤖 Initializing vectorizer...")
                vectorizer_instance = ManualVectorizer()
            
            print("🗄️  Connecting to persistent Qdrant storage...")
            qdrant_db_instance = QdrantVectorDB(
                vector_size=vectorizer_instance.vector_size,
                collection_name=q_collection,
                qdrant_path=q_path
            )
            
            if qdrant_db_instance:
                col_info = qdrant_db_instance.get_collection_info_dict()
                if col_info:
                    points_count = col_info.get('points_count', 0)
                    if points_count > 0:
                        print(f"✅ Connected! Found {points_count} existing vectors in collection '{q_collection}'")
                    else:
                        print(f"✅ Connected! Collection '{q_collection}' is empty - ready for data import")
                else:
                    print(f"⚠️  Connected, but could not retrieve collection info")

        except Exception as e:
            print(f"❌ Error initializing DB components: {e}")
            logger.error(traceback.format_exc())
            qdrant_db_instance = None

    # Initialize on startup
    load_or_init_db_components()

    def display_menu():
        print(f"\n{'='*80}")
        print(f"           🔍 Altibase Manual Search System (Persistent)")
        print(f"{'='*80}")
        print("1. 🔍 Search Manuals")
        print("2. 🏗️  Build/Rebuild Vector Database")
        print("3. ℹ️  Database Information")
        print("4. ⚙️  Settings")
        print("5. 🚪 Exit")
        print(f"{'─'*80}")
        
        persistence_status = f"📁 Storage: {q_path} (PERSISTENT)" if q_path else "💾 Storage: In-Memory"
        print(f"{persistence_status}")
        print(f"📊 Collection: '{q_collection}' | 📂 Data Dir: '{chunk_json_dir}'")
        
        if qdrant_db_instance and vectorizer_instance:
            col_info = qdrant_db_instance.get_collection_info_dict()
            if col_info:
                points_count = col_info.get('points_count', 0)
                status_icon = "✅" if points_count > 0 else "📭"
                print(f"{status_icon} Status: Ready | 📊 Vectors: {points_count}")
            else:
                print("⚠️  Status: Connected but cannot read collection info")
        else:
            print("❌ Status: Not Connected - Check settings or rebuild database")
        print(f"{'='*80}")

    while True:
        display_menu()
        option = input("👉 Select option: ").strip()

        try:
            if option == '1':
                if not qdrant_db_instance or not vectorizer_instance:
                    print("❌ Database not ready. Please build database (option 2) first.")
                    continue
                
                # Check if database has data
                col_info = qdrant_db_instance.get_collection_info_dict()
                if col_info and col_info.get('points_count', 0) == 0:
                    print("📭 Database is empty. Please build database (option 2) first.")
                    continue
                
                query = input("\n🔍 Enter search query: ").strip()
                if not query:
                    continue
                
                try:
                    k_results = int(input("📊 Number of results (default 5): ").strip() or "5")
                except ValueError:
                    k_results = 5
                
                # Enhanced filtering options
                print("\n🔧 Advanced Filters (optional):")
                print("   Examples: manual_name:AltibaseSQL, content_type:example, has_code_examples:true")
                filter_str = input("🔧 Enter filter (key:value): ").strip()
                query_filter = None
                
                if filter_str:
                    try:
                        key, value = filter_str.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Handle boolean values
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        # Handle numeric values
                        elif value.isdigit():
                            value = int(value)
                        
                        query_filter = {key: value}
                        print(f"🔧 Applying filter: {key} = {value}")
                    except ValueError:
                        print("⚠️  Invalid filter format. Ignoring filter.")

                print(f"\n🔍 Searching for: '{query}'...")
                start_time = time.time()
                results = qdrant_db_instance.search(
                    query, 
                    vectorizer_instance, 
                    k=k_results, 
                    filter_payload=query_filter
                )
                search_time = time.time() - start_time
                print(f"⏱️  Search completed in {search_time:.2f} seconds")

                display_enhanced_search_results(results)
            
            elif option == '2':
                print(f"\n{'─'*60}")
                print("🏗️  Build/Rebuild Vector Database")
                print(f"{'─'*60}")
                
                if not chunk_json_dir or not os.path.exists(chunk_json_dir):
                    print(f"❌ Chunk directory '{chunk_json_dir}' not found!")
                    print("   Please set correct directory in Settings (option 4)")
                    continue

                files_count = len([f for f in os.listdir(chunk_json_dir) if f.endswith('_chunks.json')])
                if files_count == 0:
                    print(f"❌ No '_chunks.json' files found in '{chunk_json_dir}'")
                    print("   Please process your PDF files first to create chunks")
                    continue

                print(f"📁 Found {files_count} chunk files in '{chunk_json_dir}'")
                print(f"💾 Will store vectors persistently at: {q_path}")
                
                if qdrant_db_instance:
                    col_info = qdrant_db_instance.get_collection_info_dict()
                    if col_info and col_info.get('points_count', 0) > 0:
                        print(f"⚠️  Existing database has {col_info['points_count']} vectors")
                        print("   New data will be added/updated (duplicates will be handled by hash)")

                confirm = input("\n✅ Continue with build? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("❌ Build cancelled")
                    continue

                max_files_input = input(f"📊 Max files to process (Enter for all {files_count}): ").strip()
                max_files = int(max_files_input) if max_files_input.isdigit() else None

                print(f"\n🚀 Starting build process...")
                returned_db, returned_vectorizer = build_qdrant_db(
                    chunk_data_dir=chunk_json_dir,
                    qdrant_collection_name=q_collection,
                    qdrant_path=q_path,
                    max_files_to_load=max_files
                )

                if returned_db and returned_vectorizer:
                    qdrant_db_instance = returned_db
                    vectorizer_instance = returned_vectorizer
                    print("✅ Build completed successfully!")
                    
                    col_info = qdrant_db_instance.get_collection_info_dict()
                    if col_info:
                        print(f"📊 Database now contains {col_info['points_count']} vectors")
                        print(f"💾 Data persisted at: {q_path}")
                else:
                    print("❌ Build process failed. Check logs for details.")

            elif option == '3':
                if not qdrant_db_instance:
                    print("❌ Database not connected. Try Settings → Build Database")
                    continue
                
                info = qdrant_db_instance.get_collection_info_dict()
                if info:
                    print(f"\n{'─'*60}")
                    print("ℹ️  Database Information")
                    print(f"{'─'*60}")
                    print(f"📊 Collection Name: {info['name']}")
                    print(f"🟢 Status: {info['status']}")
                    print(f"📈 Total Points: {info['points_count']:,}")
                    print(f"🔍 Indexed Vectors: {info['indexed_vectors_count']:,}")
                    print(f"📁 Segments: {info['segments_count']}")
                    print(f"💾 Persistence: {'YES' if info['is_persistent'] else 'NO'}")
                    print(f"📂 Storage Path: {info['persistence_path']}")
                    
                    if info['points_count'] == 0:
                        print("\n📭 Database is empty - use option 2 to build")
                    elif info['indexed_vectors_count'] < info['points_count']:
                        print(f"\n⚠️  Indexing in progress: {info['indexed_vectors_count']}/{info['points_count']} vectors indexed")
                else:
                    print("❌ Could not retrieve database information")

            elif option == '4':
                print(f"\n{'─'*60}")
                print("⚙️  Settings")
                print(f"{'─'*60}")
                print(f"📂 Current Storage Path: {q_path}")
                print(f"📊 Current Collection: {q_collection}")
                print(f"📁 Current Chunk Dir: {chunk_json_dir}")

                new_path = input(f"\n📂 New storage path (Enter to keep current): ").strip()
                if new_path:
                    q_path = new_path

                new_collection = input(f"📊 New collection name (Enter to keep current): ").strip()
                if new_collection:
                    q_collection = new_collection

                new_chunk_dir = input(f"📁 New chunk directory (Enter to keep current): ").strip()
                if new_chunk_dir:
                    if os.path.isdir(new_chunk_dir):
                        chunk_json_dir = new_chunk_dir
                        print(f"✅ Updated chunk directory: {chunk_json_dir}")
                    else:
                        print(f"❌ Directory '{new_chunk_dir}' not found. Keeping current.")

                print("🔄 Re-initializing with new settings...")
                load_or_init_db_components(force_reinit=True)

            elif option == '5':
                print("👋 Goodbye!")
                if qdrant_db_instance and qdrant_db_instance.is_persistent:
                    print(f"💾 Your data is safely stored at: {q_path}")
                break
            else:
                print("❌ Invalid option. Please try again.")
        
        except KeyboardInterrupt:
            print("\n⏸️  Operation cancelled by user")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            logger.error(traceback.format_exc())

# Main execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Altibase Manual Search System with Persistent Storage")
    parser.add_argument('--build', action='store_true', help='Run build process immediately')
    parser.add_argument('--query', type=str, help='Run a single query and exit')
    parser.add_argument('--chunk_dir', type=str, default="chunked_data", help='Directory containing pre-chunked JSON files')
    parser.add_argument('--qdrant_path', type=str, default="./qdrant_vector", help='Path for Qdrant persistent storage')
    parser.add_argument('--collection', type=str, default="altibase_manuals", help='Qdrant collection name')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum chunk JSON files to load during build')
    parser.add_argument('--k', type=int, default=5, help='Number of results for direct query')

    args = parser.parse_args()

    if args.build:
        print("🚀 Starting build process...")
        db, vectorizer = build_qdrant_db(
            chunk_data_dir=args.chunk_dir,
            qdrant_collection_name=args.collection,
            qdrant_path=args.qdrant_path,
            max_files_to_load=args.max_files
        )
        if db:
            info = db.get_collection_info_dict()
            if info:
                print(f"✅ Build completed! Database contains {info['points_count']} vectors")
                print(f"💾 Data persisted at: {args.qdrant_path}")
        else:
            print("❌ Build failed")

    elif args.query:
        print(f"🔍 Searching for: '{args.query}'")
        try:
            vectorizer = ManualVectorizer()
            q_db = QdrantVectorDB(
                vector_size=vectorizer.vector_size,
                collection_name=args.collection,
                qdrant_path=args.qdrant_path
            )
            
            col_info = q_db.get_collection_info_dict()
            if col_info and col_info.get('points_count', 0) == 0:
                print("📭 Database is empty. Run --build first.")
                sys.exit(1)

            results = q_db.search(args.query, vectorizer, k=args.k)
            display_enhanced_search_results(results, show_full_text=True)
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            logger.error(traceback.format_exc())
    else:
        # Default to interactive mode
        interactive_search_qdrant()