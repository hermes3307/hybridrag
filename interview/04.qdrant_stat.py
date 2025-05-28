#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant 벡터 데이터베이스 상태 확인 및 검색 도구
벡터 데이터베이스의 정보를 조회하고 간단한 검색 기능을 제공합니다.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import sys

# 기본 라이브러리
import numpy as np
import re
from collections import Counter

# Qdrant 클라이언트
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from qdrant_client.http import models
except ImportError:
    print("❌ Qdrant 클라이언트가 설치되지 않았습니다.")
    print("설치: pip install qdrant-client")
    sys.exit(1)

# 임베딩 모델 (fallback 포함)
class SimpleTfidfEmbedder:
    """TF-IDF 기반 간단한 임베딩 시스템"""
    
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
                stop_words=None
            )
            
            self.svd = TruncatedSVD(n_components=min(vector_size, 384))
            self.is_fitted = False
            
            # 불용어
            self.stopwords = {
                '그리고', '그런데', '하지만', '그러나', '또한', '또는', '그래서', '따라서',
                '이것', '그것', '저것', '이런', '그런', '저런', '이렇게', '그렇게', '저렇게',
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'between', 'among'
            }
            
        except ImportError:
            print("❌ scikit-learn이 설치되지 않았습니다.")
            print("설치: pip install scikit-learn")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text:
            return ""
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords and len(word) > 1]
        
        return ' '.join(filtered_words)
    
    def fit(self, texts: List[str]):
        """모델 훈련"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        processed_texts = [text for text in processed_texts if text.strip()]
        
        if not processed_texts:
            return
        
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        
        if tfidf_matrix.shape[1] > self.vector_size:
            self.svd.fit(tfidf_matrix)
        
        self.is_fitted = True
    
    def encode(self, text):
        """텍스트를 벡터로 변환"""
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        if not self.is_fitted:
            default_texts = [
                "LIG넥스원 기업문화 OPEN POSITIVE",
                "방위산업 국방 미사일 레이더",
                "연구개발 R&D 기술개발 혁신"
            ] + texts
            self.fit(default_texts)
        
        processed_texts = [self.preprocess_text(t) for t in texts]
        tfidf_vectors = self.vectorizer.transform(processed_texts)
        
        if hasattr(self.svd, 'components_') and tfidf_vectors.shape[1] > self.vector_size:
            vectors = self.svd.transform(tfidf_vectors)
        else:
            vectors = tfidf_vectors.toarray()
        
        if vectors.shape[1] < self.vector_size:
            padding = np.zeros((vectors.shape[0], self.vector_size - vectors.shape[1]))
            vectors = np.hstack([vectors, padding])
        elif vectors.shape[1] > self.vector_size:
            vectors = vectors[:, :self.vector_size]
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        
        return vectors[0] if single_input else vectors

class QdrantStats:
    """Qdrant 통계 및 검색 클래스"""
    
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "lignex1_news"):
        
        self.collection_name = collection_name
        
        # Qdrant 클라이언트 초기화
        try:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            print(f"✅ Qdrant 연결 성공: {qdrant_host}:{qdrant_port}")
        except Exception as e:
            print(f"❌ Qdrant 연결 실패: {e}")
            raise
        
        # 임베딩 모델 초기화 (검색용)
        self.embedding_model = None
        self.use_sentence_transformer = False
        
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.use_sentence_transformer = True
            print("✅ SentenceTransformer 로드 완료")
        except Exception:
            print("⚠️  SentenceTransformer 로드 실패, TF-IDF 사용")
            self.embedding_model = SimpleTfidfEmbedder()
            self.use_sentence_transformer = False
    
    def list_collections(self) -> List[str]:
        """모든 컬렉션 목록 조회"""
        try:
            collections = self.qdrant_client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            print(f"❌ 컬렉션 목록 조회 실패: {e}")
            return []
    
    def get_collection_info(self, collection_name: str = None) -> Dict:
        """컬렉션 상세 정보 조회"""
        if not collection_name:
            collection_name = self.collection_name
        
        try:
            info = self.qdrant_client.get_collection(collection_name)
            
            return {
                'name': collection_name,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count,
                'status': str(info.status),
                'optimizer_status': str(info.optimizer_status),
                'disk_data_size': info.disk_data_size,
                'ram_data_size': info.ram_data_size,
                'config': {
                    'distance': str(info.config.params.vectors.distance) if hasattr(info.config.params, 'vectors') else 'unknown',
                    'vector_size': info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else 'unknown'
                }
            }
            
        except Exception as e:
            print(f"❌ 컬렉션 정보 조회 실패: {e}")
            return {}
    
    def sample_points(self, collection_name: str = None, limit: int = 5) -> List[Dict]:
        """컬렉션의 샘플 포인트 조회"""
        if not collection_name:
            collection_name = self.collection_name
        
        try:
            # 스크롤을 사용하여 샘플 데이터 가져오기
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            points = []
            for point in scroll_result[0]:  # scroll_result는 (points, next_page_offset) 튜플
                point_data = {
                    'id': point.id,
                    'payload': point.payload
                }
                points.append(point_data)
            
            return points
            
        except Exception as e:
            print(f"❌ 샘플 포인트 조회 실패: {e}")
            return []
    
    def get_payload_stats(self, collection_name: str = None, sample_size: int = 100) -> Dict:
        """페이로드 통계 정보"""
        if not collection_name:
            collection_name = self.collection_name
        
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=sample_size,
                with_payload=True,
                with_vectors=False
            )
            
            field_stats = {}
            sources = Counter()
            api_providers = Counter()
            embedding_models = Counter()
            
            for point in scroll_result[0]:
                payload = point.payload
                
                # 필드 통계
                for key, value in payload.items():
                    if key not in field_stats:
                        field_stats[key] = {'count': 0, 'types': set(), 'sample_values': []}
                    
                    field_stats[key]['count'] += 1
                    field_stats[key]['types'].add(type(value).__name__)
                    
                    if len(field_stats[key]['sample_values']) < 3:
                        field_stats[key]['sample_values'].append(str(value)[:50])
                
                # 특별 필드 통계
                if 'source' in payload:
                    sources[payload['source']] += 1
                if 'api_provider' in payload:
                    api_providers[payload['api_provider']] += 1
                if 'embedding_model' in payload:
                    embedding_models[payload['embedding_model']] += 1
            
            # 타입을 문자열로 변환
            for field in field_stats:
                field_stats[field]['types'] = list(field_stats[field]['types'])
            
            return {
                'sample_size': len(scroll_result[0]),
                'fields': field_stats,
                'top_sources': dict(sources.most_common(10)),
                'api_providers': dict(api_providers.most_common()),
                'embedding_models': dict(embedding_models.most_common())
            }
            
        except Exception as e:
            print(f"❌ 페이로드 통계 조회 실패: {e}")
            return {}
    
    def search_similar(self, query: str, collection_name: str = None, limit: int = 5, score_threshold: float = 0.1) -> List[Dict]:
        """유사도 검색"""
        if not collection_name:
            collection_name = self.collection_name
        
        try:
            # 쿼리 벡터화
            if self.use_sentence_transformer:
                query_vector = self.embedding_model.encode(query)
            else:
                query_vector = self.embedding_model.encode(query)
            
            # 검색 실행
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
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
                    'payload': hit.payload
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ 검색 실패: {e}")
            return []
    
    def search_by_filter(self, filter_conditions: Dict, collection_name: str = None, limit: int = 10) -> List[Dict]:
        """필터 기반 검색"""
        if not collection_name:
            collection_name = self.collection_name
        
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        ) for key, value in filter_conditions.items()
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for point in scroll_result[0]:
                result = {
                    'id': point.id,
                    'payload': point.payload
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ 필터 검색 실패: {e}")
            return []

def print_collection_info(stats: QdrantStats, collection_name: str):
    """컬렉션 정보 출력"""
    info = stats.get_collection_info(collection_name)
    
    if not info:
        print(f"❌ 컬렉션 '{collection_name}' 정보를 가져올 수 없습니다.")
        return
    
    print(f"\n📊 컬렉션 정보: {collection_name}")
    print("=" * 50)
    print(f"벡터 수: {info['vectors_count']:,}")
    print(f"인덱스된 벡터 수: {info['indexed_vectors_count']:,}")
    print(f"포인트 수: {info['points_count']:,}")
    print(f"상태: {info['status']}")
    print(f"최적화 상태: {info['optimizer_status']}")
    print(f"디스크 사용량: {info['disk_data_size']:,} bytes ({info['disk_data_size']/(1024*1024):.1f} MB)")
    print(f"RAM 사용량: {info['ram_data_size']:,} bytes ({info['ram_data_size']/(1024*1024):.1f} MB)")
    print(f"거리 함수: {info['config']['distance']}")
    print(f"벡터 차원: {info['config']['vector_size']}")

def print_payload_stats(stats: QdrantStats, collection_name: str):
    """페이로드 통계 출력"""
    payload_stats = stats.get_payload_stats(collection_name)
    
    if not payload_stats:
        print(f"❌ 컬렉션 '{collection_name}' 페이로드 통계를 가져올 수 없습니다.")
        return
    
    print(f"\n📋 페이로드 통계 (샘플 크기: {payload_stats['sample_size']})")
    print("=" * 50)
    
    # 필드 정보
    print("\n🔍 필드 정보:")
    for field, stats_data in payload_stats['fields'].items():
        print(f"  • {field}:")
        print(f"    - 개수: {stats_data['count']}")
        print(f"    - 타입: {', '.join(stats_data['types'])}")
        if stats_data['sample_values']:
            print(f"    - 샘플: {stats_data['sample_values'][:2]}")
    
    # 소스 통계
    if payload_stats['top_sources']:
        print(f"\n📰 주요 소스 (상위 {len(payload_stats['top_sources'])}개):")
        for source, count in payload_stats['top_sources'].items():
            print(f"  • {source}: {count}개")
    
    # API 제공자 통계
    if payload_stats['api_providers']:
        print(f"\n🔌 API 제공자:")
        for provider, count in payload_stats['api_providers'].items():
            print(f"  • {provider}: {count}개")
    
    # 임베딩 모델 통계
    if payload_stats['embedding_models']:
        print(f"\n🤖 임베딩 모델:")
        for model, count in payload_stats['embedding_models'].items():
            print(f"  • {model}: {count}개")

def print_sample_points(stats: QdrantStats, collection_name: str, limit: int = 3):
    """샘플 포인트 출력"""
    points = stats.sample_points(collection_name, limit)
    
    if not points:
        print(f"❌ 컬렉션 '{collection_name}' 샘플을 가져올 수 없습니다.")
        return
    
    print(f"\n📄 샘플 데이터 (상위 {len(points)}개):")
    print("=" * 50)
    
    for i, point in enumerate(points, 1):
        print(f"\n[{i}] ID: {point['id']}")
        payload = point['payload']
        
        # 주요 필드만 출력
        title = payload.get('title', 'No Title')
        content = payload.get('content', 'No Content')
        source = payload.get('source', 'Unknown')
        
        print(f"제목: {title}")
        print(f"내용: {content[:100]}...")
        print(f"출처: {source}")
        
        if 'api_provider' in payload:
            print(f"제공자: {payload['api_provider']}")
        if 'published_date' in payload:
            print(f"발행일: {payload['published_date']}")

def interactive_search(stats: QdrantStats, collection_name: str):
    """대화형 검색"""
    print(f"\n🔍 대화형 검색 모드 (컬렉션: {collection_name})")
    print("검색어를 입력하세요. 종료하려면 'quit' 또는 'exit'을 입력하세요.")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n검색어: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 검색을 종료합니다.")
                break
            
            # 검색 실행
            results = stats.search_similar(query, collection_name, limit=5, score_threshold=0.05)
            
            if results:
                print(f"\n📄 검색 결과 ({len(results)}건):")
                for i, result in enumerate(results, 1):
                    payload = result['payload']
                    title = payload.get('title', 'No Title')
                    content = payload.get('content', 'No Content')[:100]
                    source = payload.get('source', 'Unknown')
                    
                    print(f"\n[{i}] {title} (유사도: {result['score']:.3f})")
                    print(f"    출처: {source}")
                    print(f"    내용: {content}...")
            else:
                print("검색 결과가 없습니다.")
        
        except KeyboardInterrupt:
            print("\n👋 검색을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 검색 중 오류: {e}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Qdrant 벡터 데이터베이스 상태 확인 및 검색 도구')
    parser.add_argument('--host', default='localhost', help='Qdrant 호스트 (기본값: localhost)')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant 포트 (기본값: 6333)')
    parser.add_argument('--collection', default='lignex1_news', help='컬렉션 이름 (기본값: lignex1_news)')
    parser.add_argument('--list-collections', action='store_true', help='모든 컬렉션 목록 출력')
    parser.add_argument('--info', action='store_true', help='컬렉션 상세 정보 출력')
    parser.add_argument('--stats', action='store_true', help='페이로드 통계 출력')
    parser.add_argument('--sample', type=int, help='샘플 데이터 출력 (개수 지정)')
    parser.add_argument('--search', type=str, help='검색어로 유사도 검색')
    parser.add_argument('--interactive', action='store_true', help='대화형 검색 모드')
    parser.add_argument('--filter', type=str, help='필터 검색 (JSON 형식)')
    
    args = parser.parse_args()
    
    # 기본값: 모든 정보 출력
    if not any([args.list_collections, args.info, args.stats, args.sample, args.search, args.interactive, args.filter]):
        args.info = True
        args.stats = True
        args.sample = 3
    
    try:
        # QdrantStats 초기화
        stats = QdrantStats(
            qdrant_host=args.host,
            qdrant_port=args.port,
            collection_name=args.collection
        )
        
        # 컬렉션 목록 출력
        if args.list_collections:
            collections = stats.list_collections()
            print(f"\n📚 사용 가능한 컬렉션 ({len(collections)}개):")
            for i, name in enumerate(collections, 1):
                print(f"  {i}. {name}")
        
        # 컬렉션 정보 출력
        if args.info:
            print_collection_info(stats, args.collection)
        
        # 페이로드 통계 출력
        if args.stats:
            print_payload_stats(stats, args.collection)
        
        # 샘플 데이터 출력
        if args.sample:
            print_sample_points(stats, args.collection, args.sample)
        
        # 검색 실행
        if args.search:
            print(f"\n🔍 검색어: '{args.search}'")
            results = stats.search_similar(args.search, args.collection, limit=5)
            
            if results:
                print(f"📄 검색 결과 ({len(results)}건):")
                for i, result in enumerate(results, 1):
                    payload = result['payload']
                    title = payload.get('title', 'No Title')
                    source = payload.get('source', 'Unknown')
                    
                    print(f"\n[{i}] {title} (유사도: {result['score']:.3f})")
                    print(f"    출처: {source}")
                    print(f"    내용: {payload.get('content', '')[:150]}...")
            else:
                print("검색 결과가 없습니다.")
        
        # 필터 검색 실행
        if args.filter:
            try:
                filter_conditions = json.loads(args.filter)
                print(f"\n🔍 필터 검색: {filter_conditions}")
                results = stats.search_by_filter(filter_conditions, args.collection)
                
                if results:
                    print(f"📄 필터 결과 ({len(results)}건):")
                    for i, result in enumerate(results, 1):
                        payload = result['payload']
                        title = payload.get('title', 'No Title')
                        source = payload.get('source', 'Unknown')
                        
                        print(f"\n[{i}] {title}")
                        print(f"    출처: {source}")
                else:
                    print("필터 결과가 없습니다.")
            except json.JSONDecodeError:
                print("❌ 필터 조건은 유효한 JSON 형식이어야 합니다.")
                print("예시: --filter '{\"source\": \"네이버\", \"api_provider\": \"naver\"}'")
        
        # 대화형 검색 모드
        if args.interactive:
            interactive_search(stats, args.collection)
    
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류: {e}")

if __name__ == "__main__":
    main()