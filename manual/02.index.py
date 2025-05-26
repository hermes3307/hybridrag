import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import re
import time
import gc
import logging
from typing import List, Dict, Any, Tuple, Optional, Generator, Iterator
import traceback
from multiprocessing import Pool, cpu_count
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 메모리 사용량 모니터링
def get_memory_usage():
    """현재 메모리 사용량 반환 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory_usage(step_name: str):
    """메모리 사용량 로깅"""
    memory_mb = get_memory_usage()
    logger.info(f"{step_name} - 메모리 사용량: {memory_mb:.1f} MB")

# 유틸리티 함수
def clean_text(text: str) -> str:
    """텍스트 정리: 불필요한 공백 제거 및 특수문자 처리"""
    if not text or not isinstance(text, str):
        return ""
    
    # 마크다운 문법 제거
    text = re.sub(r'#{1,6}\s+', '', text)  # 헤더 제거
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # 링크 제거
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)  # 코드 블록 제거
    text = re.sub(r'`[^`]+`', '', text)  # 인라인 코드 제거
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # 볼드/이탤릭 제거
    
    # 연속된 공백을 하나로 치환
    text = re.sub(r'\s+', ' ', text)
    # 줄바꿈 정리
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def create_chunks_smart(text: str, max_chunk_size: int = 1000, overlap: int = 200, min_chunk_size: int = 100) -> List[str]:
    """
    텍스트를 스마트하게 청크로 분할 (문단, 문장 경계 고려)
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
        # 청크 끝 위치 결정
        end = min(start + max_chunk_size, len(text))
        
        # 문단 경계에서 자르기 시도
        if end < len(text):
            # 문단 끝을 찾음 (\n\n)
            paragraph_end = text.rfind('\n\n', start, end)
            if paragraph_end > start:
                end = paragraph_end + 2
            else:
                # 문장 끝을 찾음
                sentence_ends = [
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('\n', start, end)
                ]
                sentence_end = max(sentence_ends)
                
                # 문장 끝을 찾지 못했다면 단어 경계에서 자름
                if sentence_end == -1:
                    sentence_end = text.rfind(' ', start, end)
                
                if sentence_end > start:
                    end = sentence_end + 1
        
        # 청크 추가
        chunk_text = text[start:end].strip()
        if chunk_text and len(chunk_text) >= min_chunk_size:
            chunks.append(chunk_text)
        
        # 다음 시작 위치 (오버랩 적용)
        start = max(end - overlap, start + 1)
        
        # 무한 루프 방지
        if start >= len(text) - min_chunk_size:
            break
    
    return chunks

def generate_chunk_hash(text: str) -> str:
    """청크의 해시값 생성 (중복 제거용)"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# 매뉴얼 로더 클래스 (개선된 버전)
class ManualLoader:
    def __init__(self, directory: str = "manuals"):
        self.directory = directory
        self.manuals = []
        self.error_log = []
        self.processed_files = set()
        self.chunk_hashes = set()  # 중복 청크 방지
    
    def load_manuals(self, max_files: Optional[int] = None) -> List[Dict]:
        """모든 매뉴얼 파일을 로드하고 구조화된 데이터로 변환"""
        if not os.path.exists(self.directory):
            logger.error(f"디렉토리가 존재하지 않습니다: {self.directory}")
            return []
        
        files = [f for f in os.listdir(self.directory) if f.endswith('.json')]
        
        if max_files:
            files = files[:max_files]
            
        logger.info(f"총 {len(files)}개의 매뉴얼 파일을 발견했습니다.")
        log_memory_usage("매뉴얼 로딩 시작")
        
        for filename in tqdm(files, desc="매뉴얼 로딩 중"):
            if filename in self.processed_files:
                continue
                
            file_path = os.path.join(self.directory, filename)
            
            try:
                # 파일 크기 확인
                file_size = os.path.getsize(file_path)
                if file_size > 50 * 1024 * 1024:  # 50MB 초과
                    logger.warning(f"파일 크기가 너무 큽니다 ({file_size/1024/1024:.1f}MB): {filename}")
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    manual_data = json.load(f)
                
                # 데이터 유효성 검사
                if not isinstance(manual_data, dict):
                    logger.warning(f"잘못된 형식의 매뉴얼: {filename}")
                    continue
                
                # 섹션 수 확인
                sections = manual_data.get('sections', {})
                if not sections:
                    logger.warning(f"섹션이 없는 매뉴얼: {filename}")
                    continue
                
                self.manuals.append({
                    'filename': filename,
                    'manual_data': manual_data,
                    'file_size': file_size
                })
                
                self.processed_files.add(filename)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류 - {file_path}: {str(e)}")
                self.error_log.append(f"JSON 파싱 오류 - {file_path}: {str(e)}")
            except Exception as e:
                logger.error(f"파일 처리 중 오류 발생 - {file_path}: {str(e)}")
                self.error_log.append(f"파일 처리 중 오류 - {file_path}: {str(e)}")
        
        logger.info(f"{len(self.manuals)}개의 매뉴얼을 성공적으로 로드했습니다.")
        log_memory_usage("매뉴얼 로딩 완료")
        return self.manuals
    
    def process_single_manual(self, manual_idx: int, max_chunk_size: int = 1000, 
                            overlap: int = 200, min_chunk_size: int = 100) -> List[Dict]:
        """단일 매뉴얼 파일을 처리하여 청크로 분할 (개선된 버전)"""
        if manual_idx >= len(self.manuals):
            logger.error(f"인덱스 {manual_idx}에 해당하는 매뉴얼이 없습니다.")
            return []
        
        manual = self.manuals[manual_idx]
        filename = manual['filename']
        manual_data = manual['manual_data']
        
        title = manual_data.get('title', '제목 없음')
        toc = manual_data.get('toc', [])
        sections = manual_data.get('sections', {})
        
        logger.info(f"매뉴얼 처리 중: {title} ({filename})")
        logger.info(f"총 {len(sections)}개 섹션을 처리합니다.")
        
        manual_chunks = []
        processed_sections = 0
        skipped_sections = 0
        duplicate_chunks = 0
        
        for section_key, section_content in sections.items():
            try:
                # 목차 섹션 건너뛰기
                if self._is_toc_section(section_key):
                    skipped_sections += 1
                    continue
                
                # 섹션 유효성 검사
                if not section_content or not isinstance(section_content, str):
                    skipped_sections += 1
                    continue
                
                # 너무 짧은 섹션 건너뛰기
                if len(section_content.strip()) < min_chunk_size:
                    skipped_sections += 1
                    continue
                
                # 섹션 메타데이터 추출
                section_info = self._extract_section_metadata(section_key, section_content, toc)
                
                # 섹션 내용을 청크로 분할
                section_text = clean_text(section_content)
                if not section_text:
                    skipped_sections += 1
                    continue
                
                section_chunks = create_chunks_smart(
                    section_text, max_chunk_size, overlap, min_chunk_size
                )
                
                # 각 청크를 구조화된 형태로 저장
                for i, chunk_text in enumerate(section_chunks):
                    # 중복 청크 확인
                    chunk_hash = generate_chunk_hash(chunk_text)
                    if chunk_hash in self.chunk_hashes:
                        duplicate_chunks += 1
                        continue
                    
                    self.chunk_hashes.add(chunk_hash)
                    
                    chunk_data = {
                        'manual_title': title,
                        'filename': filename,
                        'section_key': section_key,
                        'chunk_id': i,
                        'total_chunks': len(section_chunks),
                        'text': chunk_text,
                        'text_length': len(chunk_text),
                        'chunk_hash': chunk_hash,
                        'metadata': section_info
                    }
                    manual_chunks.append(chunk_data)
                
                processed_sections += 1
                
            except Exception as e:
                logger.error(f"섹션 처리 중 오류 - '{section_key}': {str(e)}")
                self.error_log.append(f"섹션 처리 중 오류 - '{section_key}': {str(e)}")
                continue
        
        logger.info(f"매뉴얼 '{title}' 처리 완료: {processed_sections}개 섹션 처리, "
                   f"{skipped_sections}개 섹션 건너뜀, {duplicate_chunks}개 중복 청크 제거, "
                   f"{len(manual_chunks)}개 청크 생성")
        
        return manual_chunks
    
    def _is_toc_section(self, section_key: str) -> bool:
        """목차 섹션인지 확인"""
        toc_keywords = ['목차', 'toc', 'table_of_contents', 'contents', 'index']
        return any(keyword in section_key.lower() for keyword in toc_keywords)
    
    def _extract_section_metadata(self, section_key: str, section_content: str, 
                                 toc: List) -> Dict:
        """섹션의 메타데이터 추출 (개선된 버전)"""
        metadata = {
            'chapter': '',
            'title': '',
            'level': 0,
            'hierarchy': [],
            'section_type': self._classify_section_type(section_content)
        }
        
        # 섹션 첫 줄에서 헤더 정보 추출
        lines = section_content.split('\n')
        for line in lines[:5]:  # 처음 5줄만 확인
            header_match = re.search(r'^#+\s+(.*?)$', line.strip())
            if header_match:
                header_text = header_match.group(1).strip()
                
                # 챕터 번호와 제목 분리
                chapter_match = re.match(r'^(\d+(?:\.\d+)*)[\.:]?\s*(.*)$', header_text)
                if chapter_match:
                    metadata['chapter'] = chapter_match.group(1)
                    metadata['title'] = chapter_match.group(2).strip()
                else:
                    metadata['title'] = header_text
                break
        
        # 목차에서 추가 메타데이터 탐색
        self._find_section_in_toc(metadata, section_key, toc)
        
        return metadata
    
    def _classify_section_type(self, content: str) -> str:
        """섹션 타입 분류"""
        content_lower = content.lower()
        
        if any(keyword in content_lower for keyword in ['예제', 'example', '샘플', 'sample']):
            return 'example'
        elif any(keyword in content_lower for keyword in ['오류', 'error', '에러', '문제']):
            return 'error'
        elif any(keyword in content_lower for keyword in ['설치', 'install', 'setup']):
            return 'installation'
        elif any(keyword in content_lower for keyword in ['설정', 'config', 'configuration']):
            return 'configuration'
        elif any(keyword in content_lower for keyword in ['api', '함수', 'function', 'method']):
            return 'api'
        else:
            return 'general'
    
    def _find_section_in_toc(self, metadata: Dict, section_key: str, toc: List, 
                           current_path: List = None) -> bool:
        """목차에서 섹션을 찾아 계층 구조 등 메타데이터 업데이트"""
        if current_path is None:
            current_path = []
        
        for item in toc:
            if not isinstance(item, dict):
                continue
            
            chapter = item.get('chapter', '')
            title = item.get('title', '')
            level = item.get('level', 0)
            
            new_path = current_path + [(chapter, title, level)]
            
            # 키 생성하여 비교
            item_key = f"{chapter}_{title}" if chapter else title
            item_key = re.sub(r'[^\w가-힣]', '_', item_key)
            
            if item_key == section_key:
                metadata['chapter'] = chapter
                metadata['title'] = title
                metadata['level'] = level
                metadata['hierarchy'] = new_path
                return True
            
            # 자식 항목 탐색
            if 'children' in item and isinstance(item['children'], list):
                if self._find_section_in_toc(metadata, section_key, item['children'], new_path):
                    return True
        
        return False

# 벡터 임베딩 및 인덱싱 클래스 (개선된 버전)
class ManualVectorizer:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", 
                 device: str = None, cache_dir: str = None):
        """매뉴얼 청크를 벡터화하는 클래스 초기화"""
        logger.info(f"모델 {model_name} 로딩 중...")
        log_memory_usage("모델 로딩 시작")
        
        # 디바이스 설정
        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        
        # 캐시 디렉토리 설정
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"모델 로드 완료 - 벡터 차원: {self.vector_size}, 디바이스: {device}")
            log_memory_usage("모델 로딩 완료")
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise
    
    def vectorize_chunks_batch(self, chunks: List[Dict], batch_size: int = 16, 
                             show_progress: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """청크를 배치 단위로 벡터화 (메모리 효율성 개선)"""
        if not chunks:
            return np.array([]), []
        
        logger.info(f"총 {len(chunks)}개 청크 벡터화 시작...")
        log_memory_usage("벡터화 시작")
        
        # 텍스트만 추출
        texts = [chunk['text'] for chunk in chunks]
        
        # 메모리 사용량에 따른 배치 크기 조정
        memory_mb = get_memory_usage()
        if memory_mb > 4000:  # 4GB 이상 사용 시 배치 크기 줄임
            batch_size = max(8, batch_size // 2)
            logger.warning(f"메모리 사용량이 높아 배치 크기를 {batch_size}로 줄입니다.")
        
        try:
            # 벡터화
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # 정규화 추가
            )
            
            logger.info(f"벡터화 완료 - shape: {vectors.shape}")
            log_memory_usage("벡터화 완료")
            
            return vectors, chunks
            
        except Exception as e:
            logger.error(f"벡터화 중 오류: {e}")
            # 메모리 부족 시 더 작은 배치로 재시도
            if "out of memory" in str(e).lower() and batch_size > 1:
                logger.warning("메모리 부족으로 배치 크기를 줄여서 재시도합니다.")
                return self.vectorize_chunks_batch(chunks, batch_size // 2, show_progress)
            raise

class ManualVectorDB:
    def __init__(self, vector_size: int):
        """벡터 데이터베이스 초기화 (개선된 버전)"""
        self.vector_size = vector_size
        self.index = None
        self.chunks = []
        self.total_vectors = 0
        self.chunk_metadata = {}  # 추가 메타데이터 저장
    
    def build_index(self, vectors: np.ndarray, chunks: List[Dict], use_gpu: bool = False):
        """벡터와 청크를 사용하여 FAISS 인덱스 구축 (개선된 버전)"""
        if len(vectors) == 0:
            logger.warning("벡터가 비어있습니다. 인덱스를 구축할 수 없습니다.")
            return
        
        logger.info(f"FAISS 인덱스 구축 중... (벡터 수: {len(vectors)})")
        log_memory_usage("인덱스 구축 시작")
        
        self.chunks = chunks
        self.total_vectors = len(vectors)
        
        # 벡터 정규화
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)
        
        try:
            # 인덱스 타입 선택 (벡터 수에 따라)
            if len(vectors) > 50000:
                # 대용량: IVF 클러스터링 인덱스
                nlist = min(int(np.sqrt(len(vectors))), 1000)
                quantizer = faiss.IndexFlatIP(self.vector_size)  # 내적 사용 (정규화된 벡터)
                self.index = faiss.IndexIVFFlat(quantizer, self.vector_size, nlist)
                
                # GPU 사용 가능 시
                if use_gpu and faiss.get_num_gpus() > 0:
                    logger.info("GPU를 사용한 인덱스 구축을 시도합니다.")
                    try:
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    except Exception as e:
                        logger.warning(f"GPU 사용 실패, CPU로 전환: {e}")
                
                logger.info("IVF 인덱스 훈련 중...")
                self.index.train(vectors)
                self.index.nprobe = min(10, nlist // 4)  # 검색 프로브 수 설정
                
            elif len(vectors) > 10000:
                # 중간 크기: HNSW 인덱스 (더 정확한 검색)
                self.index = faiss.IndexHNSWFlat(self.vector_size, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 50
                
            else:
                # 소규모: Flat 인덱스 (정확한 검색)
                self.index = faiss.IndexFlatIP(self.vector_size)
            
            # 벡터 추가
            logger.info("벡터를 인덱스에 추가 중...")
            self.index.add(vectors)
            
            # 메타데이터 구축
            self._build_metadata()
            
            logger.info(f"인덱스 구축 완료 - 총 벡터 수: {self.index.ntotal}")
            log_memory_usage("인덱스 구축 완료")
            
        except Exception as e:
            logger.error(f"인덱스 구축 중 오류: {e}")
            raise
    
    def _build_metadata(self):
        """메타데이터 인덱스 구축"""
        self.chunk_metadata = {
            'manual_titles': set(),
            'section_types': set(),
            'chapters': set()
        }
        
        for chunk in self.chunks:
            metadata = chunk.get('metadata', {})
            self.chunk_metadata['manual_titles'].add(chunk.get('manual_title', ''))
            self.chunk_metadata['section_types'].add(metadata.get('section_type', ''))
            self.chunk_metadata['chapters'].add(metadata.get('chapter', ''))
    
    def add_vectors(self, vectors: np.ndarray, chunks: List[Dict]):
        """기존 인덱스에 벡터 추가 (개선된 버전)"""
        if self.index is None:
            self.build_index(vectors, chunks)
            return
        
        if len(vectors) == 0:
            return
        
        # 벡터 정규화
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)
        
        # 벡터 추가
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.total_vectors += len(vectors)
        
        # 메타데이터 업데이트
        self._build_metadata()
        
        logger.info(f"벡터 추가 완료 - 추가된 벡터: {len(vectors)}, 총 벡터: {self.total_vectors}")
    
    def save(self, directory: str = "manual_vector_db"):
        """벡터 데이터베이스를 파일로 저장 (개선된 버전)"""
        # 디렉토리 경로 검증
        if not directory or directory.strip() == "":
            directory = "manual_vector_db"
            logger.warning(f"빈 디렉토리 경로가 제공되어 기본값으로 설정: {directory}")
        
        directory = directory.strip()
        os.makedirs(directory, exist_ok=True)
        
        logger.info(f"벡터 데이터베이스 저장 중... (디렉토리: {directory})")
        log_memory_usage("저장 시작")
        
        try:
            # 메타데이터 저장
            metadata = {
                'vector_size': self.vector_size,
                'total_vectors': self.total_vectors,
                'total_chunks': len(self.chunks),
                'index_type': type(self.index).__name__ if self.index else None,
                'chunk_metadata': {
                    'manual_titles': list(self.chunk_metadata.get('manual_titles', [])),
                    'section_types': list(self.chunk_metadata.get('section_types', [])),
                    'chapters': list(self.chunk_metadata.get('chapters', []))
                }
            }
            
            with open(os.path.join(directory, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 청크를 배치로 나누어 저장 (메모리 효율성)
            batch_size = 1000
            for i in range(0, len(self.chunks), batch_size):
                batch_chunks = self.chunks[i:i + batch_size]
                batch_file = os.path.join(directory, f'chunks_batch_{i//batch_size:03d}.pkl')
                with open(batch_file, 'wb') as f:
                    pickle.dump(batch_chunks, f)
            
            # FAISS 인덱스 저장
            if self.index is not None:
                index_path = os.path.join(directory, "manual_index.faiss")
                faiss.write_index(self.index, index_path)
            
            logger.info(f"벡터 데이터베이스 저장 완료")
            log_memory_usage("저장 완료")
            
        except Exception as e:
            logger.error(f"저장 중 오류: {e}")
            raise
    
    @classmethod
    def load(cls, directory: str = "manual_vector_db", vector_size: int = 384):
        """저장된 벡터 데이터베이스 로드 (개선된 버전)"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"디렉토리가 존재하지 않습니다: {directory}")
        
        logger.info(f"벡터 데이터베이스 로드 중... (디렉토리: {directory})")
        log_memory_usage("로드 시작")
        
        try:
            # 메타데이터 로드
            metadata_path = os.path.join(directory, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    vector_size = metadata.get('vector_size', vector_size)
            
            # 인스턴스 생성
            instance = cls(vector_size)
            
            # 청크 로드 (배치 파일들)
            instance.chunks = []
            batch_files = sorted([f for f in os.listdir(directory) if f.startswith('chunks_batch_') and f.endswith('.pkl')])
            
            for batch_file in batch_files:
                batch_path = os.path.join(directory, batch_file)
                with open(batch_path, 'rb') as f:
                    batch_chunks = pickle.load(f)
                    instance.chunks.extend(batch_chunks)
            
            # 구 버전 호환성 (단일 chunks.pkl 파일)
            if not batch_files:
                chunks_path = os.path.join(directory, 'chunks.pkl')
                if os.path.exists(chunks_path):
                    with open(chunks_path, 'rb') as f:
                        instance.chunks = pickle.load(f)
            
            # 메타데이터 구축
            instance._build_metadata()
            
            # FAISS 인덱스 로드
            index_path = os.path.join(directory, "manual_index.faiss")
            if os.path.exists(index_path):
                instance.index = faiss.read_index(index_path)
                instance.total_vectors = instance.index.ntotal
            
            logger.info(f"벡터 데이터베이스 로드 완료 - 총 벡터: {instance.total_vectors}, "
                       f"총 청크: {len(instance.chunks)}")
            log_memory_usage("로드 완료")
            
            return instance
            
        except Exception as e:
            logger.error(f"로드 중 오류: {e}")
            raise
    
    def search(self, query: str, k: int = 5, vectorizer: ManualVectorizer = None, 
              filter_manual: str = None, filter_section_type: str = None) -> List[Dict]:
        """쿼리와 유사한 매뉴얼 청크 검색 (필터링 기능 추가)"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("인덱스가 비어있습니다.")
            return []
        
        if vectorizer is None:
            raise ValueError("쿼리 벡터화를 위한 vectorizer가 필요합니다.")
        
        # 쿼리 벡터화
        query_vector = vectorizer.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        # 검색할 청크 수 조정 (필터링을 고려하여 더 많이 검색)
        search_k = min(k * 3, self.index.ntotal) if (filter_manual or filter_section_type) else min(k, self.index.ntotal)
        
        # FAISS로 검색
        distances, indices = self.index.search(query_vector, search_k)
        
        # 결과 구성
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS가 반환한 유효하지 않은 인덱스
                continue
            
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                
                # 필터링 적용
                if filter_manual and chunk.get('manual_title', '') != filter_manual:
                    continue
                
                if filter_section_type and chunk.get('metadata', {}).get('section_type', '') != filter_section_type:
                    continue
                
                # 유사도 계산 (내적 기반)
                similarity = float(1.0 - distance)  # 거리를 유사도로 변환
                
                result = {
                    'rank': len(results) + 1,
                    'distance': float(distance),
                    'similarity': similarity,
                    'chunk': chunk
                }
                results.append(result)
                
                # 원하는 개수만큼 결과를 얻으면 종료
                if len(results) >= k:
                    break
        
        return results

def build_vector_db_optimized(manuals_dir: str = "manuals", 
                             vector_db_dir: str = "manual_vector_db",
                             max_chunk_size: int = 1000, 
                             overlap: int = 200,
                             min_chunk_size: int = 100,
                             batch_size: int = 16,
                             save_interval: int = 3,
                             max_files: Optional[int] = None,
                             use_gpu: bool = False) -> Tuple[Optional[ManualVectorDB], Optional[ManualVectorizer]]:
    """매뉴얼을 최적화된 방식으로 처리하여 벡터 데이터베이스 구축"""
    try:
        # 매개변수 검증
        if not manuals_dir or manuals_dir.strip() == "":
            manuals_dir = "manuals"
            logger.warning(f"빈 매뉴얼 디렉토리가 제공되어 기본값으로 설정: {manuals_dir}")
        
        if not vector_db_dir or vector_db_dir.strip() == "":
            vector_db_dir = "manual_vector_db"
            logger.warning(f"빈 벡터 DB 디렉토리가 제공되어 기본값으로 설정: {vector_db_dir}")
        
        manuals_dir = manuals_dir.strip()
        vector_db_dir = vector_db_dir.strip()
        
        logger.info("=== 벡터 데이터베이스 구축 시작 ===")
        logger.info(f"매뉴얼 디렉토리: {manuals_dir}")
        logger.info(f"벡터 DB 디렉토리: {vector_db_dir}")
        log_memory_usage("시작")
        
        # 1. 매뉴얼 로드
        loader = ManualLoader(manuals_dir)
        manuals = loader.load_manuals(max_files)
        
        if not manuals:
            logger.error("로드된 매뉴얼이 없습니다.")
            return None, None
        
        # 2. 벡터라이저 초기화
        vectorizer = ManualVectorizer()
        
        # 3. 벡터 데이터베이스 초기화
        db = ManualVectorDB(vectorizer.vector_size)
        
        # 4. 각 매뉴얼을 순차적으로 처리
        total_processed = 0
        total_chunks = 0
        
        for i, manual in enumerate(manuals):
            logger.info(f"\n매뉴얼 {i+1}/{len(manuals)} 처리 중...")
            log_memory_usage(f"매뉴얼 {i+1} 시작")
            
            try:
                # 현재 매뉴얼 처리
                manual_chunks = loader.process_single_manual(
                    i, max_chunk_size, overlap, min_chunk_size
                )
                
                if not manual_chunks:
                    logger.warning(f"매뉴얼 {i+1}에서 청크가 생성되지 않았습니다.")
                    continue
                
                # 메모리 사용량 확인
                memory_mb = get_memory_usage()
                if memory_mb > 6000:  # 6GB 초과 시 가비지 컬렉션
                    logger.warning(f"메모리 사용량이 높습니다 ({memory_mb:.1f}MB). 정리를 수행합니다.")
                    gc.collect()
                
                # 청크를 더 작은 배치로 나누어 처리 (메모리 효율성)
                chunk_batch_size = min(500, len(manual_chunks))
                
                for batch_start in range(0, len(manual_chunks), chunk_batch_size):
                    batch_end = min(batch_start + chunk_batch_size, len(manual_chunks))
                    chunk_batch = manual_chunks[batch_start:batch_end]
                    
                    logger.info(f"배치 처리 중: {batch_start+1}-{batch_end}/{len(manual_chunks)}")
                    
                    # 청크 벡터화
                    batch_vectors, _ = vectorizer.vectorize_chunks_batch(
                        chunk_batch, 
                        batch_size=batch_size,
                        show_progress=False
                    )
                    
                    # 벡터를 데이터베이스에 추가
                    if db.index is None:
                        db.build_index(batch_vectors, chunk_batch, use_gpu)
                    else:
                        db.add_vectors(batch_vectors, chunk_batch)
                    
                    total_chunks += len(chunk_batch)
                    
                    # 배치 메모리 정리
                    del batch_vectors
                    del chunk_batch
                    gc.collect()
                
                total_processed += 1
                
                # 중간 저장
                if (i + 1) % save_interval == 0 or (i + 1) == len(manuals):
                    logger.info(f"중간 저장 중... (처리된 매뉴얼: {i+1}/{len(manuals)})")
                    try:
                        db.save(vector_db_dir)
                        log_memory_usage(f"중간 저장 완료")
                    except Exception as save_error:
                        logger.error(f"중간 저장 중 오류: {save_error}")
                        logger.error(f"벡터 DB 디렉토리: '{vector_db_dir}'")
                        # 저장 실패해도 계속 진행
                
                # 매뉴얼 메모리 정리
                del manual_chunks
                gc.collect()
                
            except Exception as e:
                logger.error(f"매뉴얼 {i+1} 처리 중 오류: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        # 최종 저장
        logger.info("최종 저장 중...")
        try:
            db.save(vector_db_dir)
            logger.info(f"최종 저장 완료: {vector_db_dir}")
        except Exception as save_error:
            logger.error(f"최종 저장 중 오류: {save_error}")
            logger.error(f"벡터 DB 디렉토리: '{vector_db_dir}'")
            # 저장 실패해도 db 객체는 반환
        
        logger.info(f"\n=== 벡터 데이터베이스 구축 완료! ===")
        logger.info(f"처리된 매뉴얼: {total_processed}개")
        logger.info(f"생성된 청크: {total_chunks}개")
        logger.info(f"벡터 개수: {db.total_vectors}개")
        log_memory_usage("최종 완료")
        
        # 오류 로그 출력
        if loader.error_log:
            logger.warning(f"\n처리 중 발생한 오류 ({len(loader.error_log)}개):")
            for error in loader.error_log[:10]:  # 최대 10개만 표시
                logger.warning(f"- {error}")
        
        return db, vectorizer
        
    except Exception as e:
        logger.error(f"벡터 데이터베이스 구축 중 치명적 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def interactive_search():
    """개선된 대화형 검색 인터페이스"""
    # 초기 설정
    source_dir = "manuals"
    vector_db_dir = "manual_vector_db"
    
    # 벡터 DB 로드 함수
    def load_vector_db():
        if not os.path.exists(vector_db_dir):
            print(f"벡터 데이터베이스를 찾을 수 없습니다. ({vector_db_dir})")
            return None, None
        
        try:
            print("벡터 데이터베이스 로딩 중...")
            vectorizer = ManualVectorizer()
            db = ManualVectorDB.load(vector_db_dir, vectorizer.vector_size)
            print(f"로드 완료: {db.total_vectors}개 벡터, {len(db.chunks)}개 청크")
            return db, vectorizer
        except Exception as e:
            print(f"벡터 데이터베이스 로드 중 오류: {str(e)}")
            return None, None
    
    # 초기 로드
    db, vectorizer = load_vector_db()
    
    # 메뉴 표시
    def display_menu():
        print("\n" + "="*70)
        print("           Altibase 매뉴얼 검색 시스템 (개선된 버전)")
        print("="*70)
        print("1. 매뉴얼 검색")
        print("2. 고급 검색 (필터링)")
        print("3. 벡터 데이터 분석")
        print("4. 벡터 데이터베이스 구축")
        print("5. 설정 변경")
        print("6. 시스템 정보")
        print("7. 종료")
        print(f"\n현재 설정:")
        print(f"- 소스 디렉토리: {source_dir}")
        print(f"- 벡터 DB 디렉토리: {vector_db_dir}")
        if db:
            print(f"- 로드된 벡터 수: {db.total_vectors:,}")
            print(f"- 로드된 청크 수: {len(db.chunks):,}")
            print(f"- 매뉴얼 수: {len(db.chunk_metadata.get('manual_titles', []))}")
        print("="*70)
    
    while True:
        display_menu()
        
        try:
            option = input("\n옵션을 선택하세요 (1-7): ").strip()
            
            if option == '1':  # 기본 검색
                if db is None:
                    print("벡터 데이터베이스가 로드되지 않았습니다.")
                    continue
                
                query = input("\n검색어를 입력하세요: ").strip()
                if not query:
                    continue
                
                k = int(input("결과 수 (기본값: 5): ") or "5")
                
                print("\n검색 중...")
                start_time = time.time()
                results = db.search(query, k=k, vectorizer=vectorizer)
                search_time = time.time() - start_time
                
                if not results:
                    print("검색 결과가 없습니다.")
                    continue
                
                print(f"\n검색 완료 ({search_time:.2f}초)")
                print("="*60)
                
                for i, result in enumerate(results):
                    chunk = result['chunk']
                    metadata = chunk.get('metadata', {})
                    
                    print(f"\n{i+1}. [{chunk['manual_title']}]")
                    print(f"   섹션: {chunk['section_key']}")
                    print(f"   챕터: {metadata.get('chapter', 'N/A')}")
                    print(f"   유사도: {result['similarity']*100:.1f}%")
                    print(f"   길이: {chunk.get('text_length', len(chunk['text']))} 문자")
                    
                    # 텍스트 미리보기
                    preview_text = chunk['text'][:200]
                    print(f"   내용: {preview_text}...")
                    
                    if input("\n   전체 내용을 보시겠습니까? (y/n): ").lower() == 'y':
                        print(f"\n--- 전체 내용 ---")
                        print(chunk['text'])
                        print("-" * 50)
            
            elif option == '2':  # 고급 검색
                if db is None:
                    print("벡터 데이터베이스가 로드되지 않았습니다.")
                    continue
                
                query = input("\n검색어를 입력하세요: ").strip()
                if not query:
                    continue
                
                # 필터 옵션
                print("\n필터 옵션:")
                manual_titles = list(db.chunk_metadata.get('manual_titles', []))
                if manual_titles:
                    print("매뉴얼 목록:")
                    for i, title in enumerate(manual_titles[:10]):
                        print(f"  {i+1}. {title}")
                    
                filter_manual = input("매뉴얼 필터 (정확한 제목 입력, 없으면 Enter): ").strip()
                filter_section = input("섹션 타입 필터 (예: api, example, error, 없으면 Enter): ").strip()
                
                k = int(input("결과 수 (기본값: 5): ") or "5")
                
                print("\n검색 중...")
                start_time = time.time()
                results = db.search(
                    query, k=k, vectorizer=vectorizer,
                    filter_manual=filter_manual if filter_manual else None,
                    filter_section_type=filter_section if filter_section else None
                )
                search_time = time.time() - start_time
                
                if not results:
                    print("검색 결과가 없습니다.")
                    continue
                
                print(f"\n검색 완료 ({search_time:.2f}초)")
                print("="*60)
                
                for i, result in enumerate(results):
                    chunk = result['chunk']
                    metadata = chunk.get('metadata', {})
                    
                    print(f"\n{i+1}. [{chunk['manual_title']}]")
                    print(f"   섹션: {chunk['section_key']}")
                    print(f"   타입: {metadata.get('section_type', 'N/A')}")
                    print(f"   챕터: {metadata.get('chapter', 'N/A')}")
                    print(f"   유사도: {result['similarity']*100:.1f}%")
                    print(f"   내용: {chunk['text'][:200]}...")
            
            elif option == '3':  # 분석
                if db is None:
                    print("벡터 데이터베이스가 로드되지 않았습니다.")
                    continue
                
                print(f"\n=== 벡터 데이터베이스 분석 ===")
                print(f"총 벡터 수: {db.total_vectors:,}")
                print(f"총 청크 수: {len(db.chunks):,}")
                print(f"벡터 차원: {db.vector_size}")
                print(f"인덱스 타입: {type(db.index).__name__ if db.index else 'None'}")
                
                # 매뉴얼별 통계
                manual_stats = {}
                section_type_stats = {}
                
                for chunk in db.chunks:
                    title = chunk.get('manual_title', '제목 없음')
                    manual_stats[title] = manual_stats.get(title, 0) + 1
                    
                    section_type = chunk.get('metadata', {}).get('section_type', 'unknown')
                    section_type_stats[section_type] = section_type_stats.get(section_type, 0) + 1
                
                print(f"\n매뉴얼별 청크 수 (상위 10개):")
                for title, count in sorted(manual_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"- {title}: {count:,}개")
                
                print(f"\n섹션 타입별 통계:")
                for section_type, count in sorted(section_type_stats.items(), key=lambda x: x[1], reverse=True):
                    print(f"- {section_type}: {count:,}개")
                
                # 메모리 사용량
                print(f"\n시스템 정보:")
                print(f"- 현재 메모리 사용량: {get_memory_usage():.1f} MB")
            
            elif option == '4':  # 구축
                print("\n벡터 데이터베이스를 구축합니다...")
                
                max_chunk_size = int(input("청크 크기 (기본값: 1000): ") or "1000")
                overlap = int(input("오버랩 크기 (기본값: 200): ") or "200")
                min_chunk_size = int(input("최소 청크 크기 (기본값: 100): ") or "100")
                batch_size = int(input("배치 크기 (기본값: 16): ") or "16")
                max_files = input("최대 파일 수 (전체는 Enter): ").strip()
                max_files = int(max_files) if max_files else None
                
                use_gpu = input("GPU 사용 여부 (y/n): ").lower() == 'y'
                
                print("\n구축을 시작합니다...")
                start_time = time.time()
                
                db, vectorizer = build_vector_db_optimized(
                    source_dir, vector_db_dir, 
                    max_chunk_size, overlap, min_chunk_size,
                    batch_size, 3, max_files, use_gpu
                )
                
                build_time = time.time() - start_time
                
                if db:
                    print(f"\n벡터 데이터베이스가 성공적으로 구축되었습니다! ({build_time:.1f}초)")
                    print(f"총 벡터: {db.total_vectors:,}개")
                    print(f"총 청크: {len(db.chunks):,}개")
                else:
                    print("\n벡터 데이터베이스 구축에 실패했습니다.")
            
            elif option == '5':  # 설정 변경
                print("\n=== 설정 변경 ===")
                print("1. 소스 디렉토리 변경")
                print("2. 벡터 DB 디렉토리 변경")
                print("3. 벡터 DB 다시 로드")
                
                sub_option = input("선택 (1-3): ").strip()
                
                if sub_option == '1':
                    new_dir = input(f"새 소스 디렉토리 (현재: {source_dir}): ").strip()
                    if os.path.exists(new_dir):
                        source_dir = new_dir
                        print(f"소스 디렉토리가 '{source_dir}'로 변경되었습니다.")
                    else:
                        print(f"디렉토리 '{new_dir}'가 존재하지 않습니다.")
                
                elif sub_option == '2':
                    new_dir = input(f"새 벡터 DB 디렉토리 (현재: {vector_db_dir}): ").strip()
                    vector_db_dir = new_dir
                    print(f"벡터 DB 디렉토리가 '{vector_db_dir}'로 변경되었습니다.")
                    
                    # 새 디렉토리에서 DB 로드 시도
                    if os.path.exists(vector_db_dir):
                        print("새 디렉토리에서 벡터 DB를 로드합니다...")
                        db, vectorizer = load_vector_db()
                
                elif sub_option == '3':
                    print("벡터 DB를 다시 로드합니다...")
                    db, vectorizer = load_vector_db()
            
            elif option == '6':  # 시스템 정보
                print(f"\n=== 시스템 정보 ===")
                print(f"메모리 사용량: {get_memory_usage():.1f} MB")
                print(f"CPU 개수: {cpu_count()}")
                
                try:
                    import torch
                    print(f"PyTorch 버전: {torch.__version__}")
                    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        print(f"GPU 개수: {torch.cuda.device_count()}")
                        print(f"현재 GPU: {torch.cuda.get_device_name()}")
                except ImportError:
                    print("PyTorch가 설치되지 않았습니다.")
                
                print(f"FAISS GPU 개수: {faiss.get_num_gpus()}")
            
            elif option == '7':  # 종료
                print("프로그램을 종료합니다.")
                break
            
            else:
                print("올바른 옵션을 선택하세요 (1-7)")
        
        except KeyboardInterrupt:
            print("\n\n작업이 중단되었습니다.")
        except Exception as e:
            print(f"\n오류 발생: {str(e)}")
            logger.error(traceback.format_exc())

# 메인 함수
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Altibase 매뉴얼 벡터 검색 시스템 (개선된 버전)')
    parser.add_argument('--build', action='store_true', help='벡터 데이터베이스 구축')
    parser.add_argument('--search', action='store_true', help='대화형 검색 모드')
    parser.add_argument('--query', type=str, help='직접 검색 쿼리')
    parser.add_argument('--input-dir', type=str, default='manuals', help='매뉴얼 디렉토리')
    parser.add_argument('--output-dir', type=str, default='manual_vector_db', help='벡터 DB 디렉토리')
    parser.add_argument('--chunk-size', type=int, default=1000, help='청크 크기')
    parser.add_argument('--overlap', type=int, default=200, help='오버랩 크기')
    parser.add_argument('--min-chunk-size', type=int, default=100, help='최소 청크 크기')
    parser.add_argument('--batch-size', type=int, default=16, help='배치 크기')
    parser.add_argument('--top-k', type=int, default=5, help='검색 결과 수')
    parser.add_argument('--max-files', type=int, help='최대 처리 파일 수')
    parser.add_argument('--use-gpu', action='store_true', help='GPU 사용')
    
    args = parser.parse_args()
    
    if args.build:
        # 벡터 데이터베이스 구축
        print("벡터 데이터베이스 구축을 시작합니다...")
        db, vectorizer = build_vector_db_optimized(
            args.input_dir, args.output_dir,
            args.chunk_size, args.overlap, args.min_chunk_size,
            args.batch_size, 3, args.max_files, args.use_gpu
        )
        
        if db:
            print(f"구축 완료: {db.total_vectors:,}개 벡터, {len(db.chunks):,}개 청크")
        else:
            print("구축 실패")
    
    elif args.query:
        # 직접 검색
        try:
            print("벡터 데이터베이스 로딩 중...")
            vectorizer = ManualVectorizer()
            db = ManualVectorDB.load(args.output_dir, vectorizer.vector_size)
            
            print(f"검색 중: '{args.query}'")
            results = db.search(args.query, k=args.top_k, vectorizer=vectorizer)
            
            print(f"\n검색 결과 ({len(results)}개):")
            print("="*80)
            
            for i, result in enumerate(results):
                chunk = result['chunk']
                metadata = chunk.get('metadata', {})
                
                print(f"\n{i+1}. [{chunk['manual_title']}] - {chunk['section_key']}")
                print(f"   챕터: {metadata.get('chapter', 'N/A')}")
                print(f"   타입: {metadata.get('section_type', 'N/A')}")
                print(f"   유사도: {result['similarity']*100:.1f}%")
                print(f"   내용: {chunk['text'][:300]}...")
                print("-" * 80)
        
        except Exception as e:
            print(f"검색 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
    
    else:
        # 대화형 모드
        interactive_search()