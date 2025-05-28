#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGNEX1 벡터화 및 Qdrant 저장 시스템 (Fallback 버전)
수집된 뉴스 데이터를 벡터화하여 Qdrant에 저장합니다.
SentenceTransformer 로드 실패시 TF-IDF 기반 벡터화로 대체
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional
import argparse
from dataclasses import dataclass
import hashlib

# 기본 라이브러리
import numpy as np
import re
from collections import Counter

# Qdrant 클라이언트
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lignex1_vectorize.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class NewsDocument:
    """뉴스 문서 데이터 클래스"""
    id: str
    title: str
    content: str
    source: str
    url: str
    published_date: str
    keywords: List[str]
    api_provider: str
    api_type: str
    collected_at: str

class SimpleTfidfEmbedder:
    """TF-IDF 기반 간단한 임베딩 시스템 (SentenceTransformer 대체용)"""
    
    def __init__(self, vector_size: int = 384):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            
            self.vector_size = vector_size
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9,
                stop_words=None  # 한국어 지원을 위해 None
            )
            
            # 차원 축소를 위한 SVD
            self.svd = TruncatedSVD(n_components=min(vector_size, 384))
            self.is_fitted = False
            
            # 한국어/영어 불용어
            self.stopwords = {
                '그리고', '그런데', '하지만', '그러나', '또한', '또는', '그래서', '따라서',
                '이것', '그것', '저것', '이런', '그런', '저런', '이렇게', '그렇게', '저렇게',
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'between', 'among'
            }
            
            logger.info(f"TF-IDF 임베딩 모델 초기화 완료 (차원: {vector_size})")
            
        except ImportError:
            logger.error("scikit-learn이 설치되지 않았습니다. pip install scikit-learn")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text:
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수문자 정리 (한글, 영문, 숫자, 공백만 남김)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 불용어 제거
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords and len(word) > 1]
        
        return ' '.join(filtered_words)
    
    def fit(self, texts: List[str]):
        """텍스트 목록에 대해 모델 훈련"""
        # 텍스트 전처리
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 빈 텍스트 필터링
        processed_texts = [text for text in processed_texts if text.strip()]
        
        if not processed_texts:
            logger.warning("전처리 후 유효한 텍스트가 없습니다.")
            return
        
        # TF-IDF 벡터화
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        
        # SVD로 차원 축소
        if tfidf_matrix.shape[1] > self.vector_size:
            self.svd.fit(tfidf_matrix)
        
        self.is_fitted = True
        logger.info(f"TF-IDF 모델 훈련 완료: {len(processed_texts)}개 문서")
    
    def encode(self, text):
        """텍스트를 벡터로 변환"""
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        if not self.is_fitted:
            # 기본 훈련 데이터로 훈련
            default_texts = [
                "LIG넥스원 기업문화 OPEN POSITIVE",
                "방위산업 국방 미사일 레이더",
                "연구개발 R&D 기술개발 혁신",
                "성과 수주 매출 성장 실적",
                "미래 비전 전략 계획 목표"
            ] + texts
            self.fit(default_texts)
        
        # 텍스트 전처리
        processed_texts = [self.preprocess_text(t) for t in texts]
        
        # TF-IDF 변환
        tfidf_vectors = self.vectorizer.transform(processed_texts)
        
        # SVD 차원 축소 (필요한 경우)
        if hasattr(self.svd, 'components_') and tfidf_vectors.shape[1] > self.vector_size:
            vectors = self.svd.transform(tfidf_vectors)
        else:
            vectors = tfidf_vectors.toarray()
        
        # 벡터 크기 조정
        if vectors.shape[1] < self.vector_size:
            # 패딩 추가
            padding = np.zeros((vectors.shape[0], self.vector_size - vectors.shape[1]))
            vectors = np.hstack([vectors, padding])
        elif vectors.shape[1] > self.vector_size:
            # 잘라내기
            vectors = vectors[:, :self.vector_size]
        
        # 정규화
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 0으로 나누는 것 방지
        vectors = vectors / norms
        
        return vectors[0] if single_input else vectors
    
    def get_sentence_embedding_dimension(self):
        """임베딩 차원 반환"""
        return self.vector_size

class TextProcessor:
    """텍스트 전처리 클래스"""
    
    def __init__(self):
        # 불용어 리스트 (한국어 + 영어)
        self.stopwords = {
            '그리고', '그런데', '하지만', '그러나', '또한', '또는', '그래서', '따라서',
            '이것', '그것', '저것', '이런', '그런', '저런', '이렇게', '그렇게', '저렇게',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among'
        }
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수문자 정리 (한글, 영문, 숫자, 공백만 남김)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """키워드 추출"""
        # 텍스트 정리
        clean_text = self.clean_text(text)
        
        # 단어 분리 (간단한 공백 기반)
        words = clean_text.split()
        
        # 불용어 제거 및 길이 필터링
        filtered_words = [
            word.lower() for word in words 
            if len(word) > 2 and word.lower() not in self.stopwords
        ]
        
        # 빈도 계산
        word_freq = Counter(filtered_words)
        
        # 상위 키워드 반환
        return [word for word, freq in word_freq.most_common(top_k)]
    
    def create_searchable_content(self, title: str, description: str) -> str:
        """검색 가능한 통합 컨텐츠 생성"""
        # 제목과 설명을 결합
        combined_text = f"{title} {description}"
        
        # 텍스트 정리
        cleaned_text = self.clean_text(combined_text)
        
        return cleaned_text

class QdrantVectorizer:
    """Qdrant 벡터화 및 저장 클래스 (Fallback 포함)"""
    
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "lignex1_news",
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        
        self.collection_name = collection_name
        self.text_processor = TextProcessor()
        self.model_name = model_name
        
        # Qdrant 클라이언트 초기화
        try:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            logger.info(f"Qdrant 클라이언트 연결 성공: {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.error(f"Qdrant 연결 실패: {e}")
            raise
        
        # 임베딩 모델 로드 (SentenceTransformer 우선, 실패시 TF-IDF)
        self.embedding_model = None
        self.vector_size = 384  # 기본값
        
        # 먼저 SentenceTransformer 시도
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name)
            self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer 로드 완료: {model_name} (차원: {self.vector_size})")
            self.use_sentence_transformer = True
        except Exception as e:
            logger.warning(f"SentenceTransformer 로드 실패: {e}")
            logger.info("TF-IDF 기반 임베딩으로 대체합니다...")
            
            # TF-IDF 기반 임베딩으로 대체
            try:
                self.embedding_model = SimpleTfidfEmbedder(vector_size=384)
                self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"TF-IDF 임베딩 모델 로드 완료 (차원: {self.vector_size})")
                self.use_sentence_transformer = False
            except Exception as e2:
                logger.error(f"TF-IDF 임베딩 모델 로드도 실패: {e2}")
                raise
    
    def create_collection(self, recreate: bool = False) -> bool:
        """Qdrant 컬렉션 생성"""
        try:
            # 기존 컬렉션 확인
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists and recreate:
                logger.info(f"기존 컬렉션 삭제: {self.collection_name}")
                self.qdrant_client.delete_collection(self.collection_name)
                collection_exists = False
            
            if not collection_exists:
                logger.info(f"새 컬렉션 생성: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"컬렉션 생성 완료: {self.collection_name}")
            else:
                logger.info(f"기존 컬렉션 사용: {self.collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 생성 실패: {e}")
            return False
    
    def vectorize_text(self, text: str) -> np.ndarray:
        """텍스트 벡터화"""
        try:
            # 텍스트가 비어있으면 기본값 반환
            if not text or not text.strip():
                return np.zeros(self.vector_size)
            
            # 임베딩 생성
            embedding = self.embedding_model.encode(text)
            
            # numpy array로 변환
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"텍스트 벡터화 실패: {e}")
            return np.zeros(self.vector_size)
    
    def load_news_from_db(self, db_path: str, limit: int = None) -> List[NewsDocument]:
        """SQLite에서 뉴스 데이터 로드"""
        news_docs = []
        
        try:
            with sqlite3.connect(db_path) as conn:
                query = '''
                    SELECT id, title, description, link, source, pub_date,
                           search_keyword, api_provider, api_type, collected_at
                    FROM articles 
                    WHERE title IS NOT NULL AND description IS NOT NULL
                    ORDER BY created_at DESC
                '''
                
                if limit:
                    query += f' LIMIT {limit}'
                
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                
                logger.info(f"데이터베이스에서 {len(rows)}개 기사 로드")
                
                for row in rows:
                    (id, title, description, link, source, pub_date, 
                     search_keyword, api_provider, api_type, collected_at) = row
                    
                    # 검색 가능한 컨텐츠 생성
                    content = self.text_processor.create_searchable_content(title, description)
                    
                    # 키워드 추출
                    keywords = self.text_processor.extract_keywords(content)
                    if search_keyword and search_keyword not in keywords:
                        keywords.insert(0, search_keyword)
                    
                    # NewsDocument 객체 생성
                    doc = NewsDocument(
                        id=str(id),
                        title=title,
                        content=content,
                        source=source or 'Unknown',
                        url=link or '',
                        published_date=pub_date or '',
                        keywords=keywords,
                        api_provider=api_provider or 'unknown',
                        api_type=api_type or 'unknown',
                        collected_at=collected_at or ''
                    )
                    
                    news_docs.append(doc)
                
        except Exception as e:
            logger.error(f"데이터베이스 로드 실패: {e}")
        
        return news_docs
    
    def store_documents(self, documents: List[NewsDocument], batch_size: int = 100) -> bool:
        """문서들을 Qdrant에 저장"""
        try:
            total_docs = len(documents)
            logger.info(f"총 {total_docs}개 문서를 벡터화하여 저장 시작...")
            
            # TF-IDF 모델의 경우 모든 텍스트로 먼저 훈련
            if not self.use_sentence_transformer:
                logger.info("TF-IDF 모델 훈련 중...")
                all_texts = [doc.content for doc in documents if doc.content.strip()]
                self.embedding_model.fit(all_texts)
            
            # 배치 단위로 처리
            for i in range(0, total_docs, batch_size):
                batch_docs = documents[i:i + batch_size]
                points = []
                
                for doc in batch_docs:
                    try:
                        # 텍스트 벡터화
                        vector = self.vectorize_text(doc.content)
                        
                        # 메타데이터 준비
                        payload = {
                            "title": doc.title,
                            "content": doc.content[:1000],  # 긴 텍스트는 잘라서 저장
                            "source": doc.source,
                            "url": doc.url,
                            "published_date": doc.published_date,
                            "keywords": doc.keywords,
                            "api_provider": doc.api_provider,
                            "api_type": doc.api_type,
                            "collected_at": doc.collected_at,
                            "vector_created_at": datetime.now().isoformat(),
                            "embedding_model": "tfidf" if not self.use_sentence_transformer else self.model_name
                        }
                        
                        # Point 객체 생성
                        point = PointStruct(
                            id=int(doc.id),
                            vector=vector.tolist(),
                            payload=payload
                        )
                        
                        points.append(point)
                        
                    except Exception as e:
                        logger.error(f"문서 {doc.id} 처리 실패: {e}")
                        continue
                
                if points:
                    # 배치 업로드
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                
                processed = min(i + batch_size, total_docs)
                logger.info(f"진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%)")
            
            logger.info(f"✅ 모든 문서 저장 완료: {total_docs}개")
            return True
            
        except Exception as e:
            logger.error(f"문서 저장 실패: {e}")
            return False
    
    def search_similar(self, query: str, limit: int = 5, score_threshold: float = 0.3) -> List[Dict]:
        """유사 문서 검색 (TF-IDF의 경우 낮은 threshold 사용)"""
        try:
            # 쿼리 벡터화
            query_vector = self.vectorize_text(query)
            
            # TF-IDF 모델의 경우 더 낮은 threshold 사용
            if not self.use_sentence_transformer:
                score_threshold = max(0.1, score_threshold * 0.3)
            
            # 검색 실행
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold
            )
            
            # 결과 포맷팅
            results = []
            for hit in search_result:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'title': hit.payload.get('title', ''),
                    'content': hit.payload.get('content', ''),
                    'source': hit.payload.get('source', ''),
                    'url': hit.payload.get('url', ''),
                    'keywords': hit.payload.get('keywords', []),
                    'api_provider': hit.payload.get('api_provider', ''),
                    'published_date': hit.payload.get('published_date', ''),
                    'embedding_model': hit.payload.get('embedding_model', 'unknown')
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """컬렉션 정보 조회"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            
            return {
                'name': self.collection_name,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count,
                'status': info.status,
                'optimizer_status': info.optimizer_status,
                'disk_data_size': info.disk_data_size,
                'ram_data_size': info.ram_data_size,
                'embedding_model': 'tfidf' if not self.use_sentence_transformer else self.model_name
            }
            
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {}

class InterviewRAGSystem:
    """면접용 RAG 시스템"""
    
    def __init__(self, qdrant_vectorizer: QdrantVectorizer):
        self.vectorizer = qdrant_vectorizer
        
        # LIG넥스원 관련 주요 주제들
        self.interview_topics = {
            "company_culture": ["기업문화", "핵심가치", "조직문화", "OPEN", "POSITIVE"],
            "business": ["방위산업", "국방", "미사일", "레이더", "무기체계"],
            "technology": ["R&D", "연구개발", "기술개발", "혁신", "첨단기술"],
            "performance": ["성과", "수주", "매출", "성장", "실적"],
            "future": ["미래", "비전", "전략", "계획", "목표"]
        }
    
    def get_context_for_question(self, question_category: str, question_text: str) -> List[Dict]:
        """면접 질문에 대한 컨텍스트 정보 제공"""
        # 질문 카테고리에 맞는 키워드로 검색
        topic_keywords = self.interview_topics.get(question_category, [])
        
        # 질문 텍스트와 주제 키워드를 결합하여 검색
        search_queries = [question_text] + topic_keywords
        
        all_results = []
        for query in search_queries[:3]:  # 상위 3개 쿼리만 사용
            results = self.vectorizer.search_similar(query, limit=3, score_threshold=0.1)
            all_results.extend(results)
        
        # 중복 제거 및 점수순 정렬
        unique_results = {}
        for result in all_results:
            result_id = result['id']
            if result_id not in unique_results or result['score'] > unique_results[result_id]['score']:
                unique_results[result_id] = result
        
        # 점수순으로 정렬하여 상위 5개 반환
        sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
        return sorted_results[:5]

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='LIGNEX1 벡터화 및 Qdrant 저장 (Fallback 버전)')
    parser.add_argument('--db-path', default='lignex1_data/lignex1_articles.db', help='SQLite 데이터베이스 경로')
    parser.add_argument('--qdrant-host', default='localhost', help='Qdrant 호스트')
    parser.add_argument('--qdrant-port', type=int, default=6333, help='Qdrant 포트')
    parser.add_argument('--collection', default='lignex1_news', help='Qdrant 컬렉션 이름')
    parser.add_argument('--recreate', action='store_true', help='컬렉션 재생성')
    parser.add_argument('--limit', type=int, help='처리할 문서 수 제한')
    parser.add_argument('--batch-size', type=int, default=50, help='배치 크기')
    parser.add_argument('--search', type=str, help='테스트 검색')
    parser.add_argument('--info', action='store_true', help='컬렉션 정보 출력')
    
    args = parser.parse_args()
    
    # 데이터베이스 파일 확인
    if not os.path.exists(args.db_path):
        print(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {args.db_path}")
        print("먼저 01.extract.py를 실행하여 데이터를 수집해주세요.")
        
        # 테스트 데이터 생성
        print("🧪 테스트 데이터로 진행합니다...")
        try:
            vectorizer = QdrantVectorizer(
                qdrant_host=args.qdrant_host,
                qdrant_port=args.qdrant_port,
                collection_name=args.collection
            )
            
            if vectorizer.create_collection(recreate=args.recreate):
                print("✅ 컬렉션 생성 완료!")
                
                # 테스트 검색 실행
                if args.search:
                    print(f"\n🔍 검색어: '{args.search}'")
                    print("(테스트 데이터가 없어 결과가 없을 수 있습니다)")
                
                # 컬렉션 정보 출력
                if args.info:
                    info = vectorizer.get_collection_info()
                    print("\n📊 Qdrant 컬렉션 정보:")
                    print(json.dumps(info, indent=2, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"테스트 실행 중 오류: {e}")
        
        return
    
    try:
        # Qdrant 벡터라이저 초기화
        vectorizer = QdrantVectorizer(
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            collection_name=args.collection
        )
        
        # 컬렉션 생성
        if not vectorizer.create_collection(recreate=args.recreate):
            return
        
        # 컬렉션 정보 출력
        if args.info:
            info = vectorizer.get_collection_info()
            print("\n📊 Qdrant 컬렉션 정보:")
            print(json.dumps(info, indent=2, ensure_ascii=False))
            return
        
        # 테스트 검색
        if args.search:
            print(f"\n🔍 검색어: '{args.search}'")
            results = vectorizer.search_similar(args.search, limit=5)
            
            if results:
                print(f"📄 검색 결과 ({len(results)}건):")
                for i, result in enumerate(results, 1):
                    print(f"\n[{i}] {result['title']} (유사도: {result['score']:.3f})")
                    print(f"출처: {result['source']} | 제공: {result['api_provider']}")
                    print(f"임베딩: {result.get('embedding_model', 'unknown')}")
                    print(f"내용: {result['content'][:150]}...")
            else:
                print("검색 결과가 없습니다.")
            return
        
        # 뉴스 데이터 로드
        print("📰 뉴스 데이터 로드 중...")
        documents = vectorizer.load_news_from_db(args.db_path, limit=args.limit)
        
        if not documents:
            print("❌ 처리할 문서가 없습니다.")
            return
        
        # 벡터화 및 저장
        embedding_type = "TF-IDF" if not vectorizer.use_sentence_transformer else "SentenceTransformer"
        print(f"🔄 {len(documents)}개 문서 벡터화 및 저장 시작... ({embedding_type} 사용)")
        success = vectorizer.store_documents(documents, batch_size=args.batch_size)
        
        if success:
            # 최종 정보 출력
            info = vectorizer.get_collection_info()
            print(f"\n✅ 벡터화 완료!")
            print(f"📊 저장된 벡터 수: {info.get('vectors_count', 0):,}")
            print(f"💾 디스크 사용량: {info.get('disk_data_size', 0):,} bytes")
            print(f"🧠 RAM 사용량: {info.get('ram_data_size', 0):,} bytes")
            print(f"🤖 임베딩 모델: {info.get('embedding_model', 'unknown')}")
            
            # 테스트 검색 실행
            print("\n🧪 테스트 검색 실행...")
            test_queries = ["LIG넥스원 기업문화", "방위산업 기술개발", "미사일 연구개발"]
            
            for query in test_queries:
                results = vectorizer.search_similar(query, limit=2, score_threshold=0.1)
                print(f"\n'{query}' 검색 결과: {len(results)}건")
                for result in results:
                    print(f"  - {result['title']} (유사도: {result['score']:.3f})")
        
        else:
            print("❌ 벡터화 실패")
    
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}")
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류: {e}")