#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGNEX1 RAG 기반 조직적합도 인성면접 시스템
벡터 데이터베이스를 활용한 지능형 면접 프로그램
"""

import streamlit as st
import json
import sqlite3
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import asyncio
from collections import Counter, defaultdict
import time  # 상단에 추가

# 벡터 검색 관련
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st
import numpy as np



try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    st.warning("⚠️ Qdrant가 설치되지 않았습니다. RAG 기능이 제한됩니다.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ SentenceTransformers가 설치되지 않았습니다. TF-IDF를 사용합니다.")

# 자연어 처리
import nltk
try:
    # 새 버전 먼저 시도
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        # 구 버전으로 fallback
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

# 설정
st.set_page_config(
    page_title="LIGNEX1 RAG 면접 시스템",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lignex1_interview.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 스타일 설정
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #3B82F6, #1E40AF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-bottom: 1rem;
        padding-left: 10px;
        border-left: 4px solid #3B82F6;
    }
    
    .interview-card {
        background: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin: 1rem 0;
    }
    
    .context-box {
        background: #EEF2FF;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    
    .score-good {
        color: #059669;
        font-weight: bold;
    }
    
    .score-average {
        color: #D97706;
        font-weight: bold;
    }
    
    .score-poor {
        color: #DC2626;
        font-weight: bold;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class InterviewQuestion:
    """면접 질문 데이터 클래스"""
    id: str
    text: str
    category: str
    context_keywords: List[str]
    follow_up_questions: List[str]
    difficulty_level: str = "medium"
    expected_duration: int = 5  # 분
    evaluation_criteria: List[str] = None

@dataclass
class InterviewAnswer:
    """면접 답변 데이터 클래스"""
    question_id: str
    question_text: str
    answer: str
    context_used: List[Dict]
    timestamp: str
    evaluation_score: float = 0.0
    keyword_match_score: float = 0.0
    answer_length: int = 0
    sentiment_score: float = 0.0
    coherence_score: float = 0.0

@dataclass
class InterviewSession:
    """면접 세션 데이터 클래스"""
    session_id: str
    candidate_name: str
    position: str
    start_time: str
    end_time: Optional[str] = None
    answers: List[InterviewAnswer] = None
    overall_score: float = 0.0
    feedback: str = ""
    recommendations: List[str] = None

class TextAnalyzer:
    """텍스트 분석 클래스"""
    
    def __init__(self):
        # 감정 분석용 키워드
        self.positive_keywords = {
            '성취', '성공', '극복', '열정', '도전', '성장', '발전', '협력', '소통',
            '창의', '혁신', '책임', '리더십', '팀워크', '목표', '비전', '긍정',
            '자신감', '능력', '경험', '학습', '개선', '효율', '품질', '우수',
            'achievement', 'success', 'overcome', 'passion', 'challenge', 'growth'
        }
        
        self.negative_keywords = {
            '실패', '포기', '어려움', '문제', '한계', '부족', '걱정', '스트레스',
            '갈등', '압박', '위기', '미흡', '실수', '지연', '손실',
            'failure', 'give up', 'difficult', 'problem', 'limitation', 'worry'
        }
        
        # LIGNEX1 관련 핵심 키워드
        self.company_keywords = {
            'culture': ['OPEN', 'POSITIVE', '개방', '긍정', '기업문화', '핵심가치', '조직문화'],
            'business': ['방위산업', '국방', '미사일', '레이더', '무기체계', '안보', '국가'],
            'technology': ['R&D', '연구개발', '기술개발', '혁신', '첨단기술', '엔지니어링'],
            'performance': ['성과', '수주', '매출', '성장', '실적', '목표달성'],
            'future': ['미래', '비전', '전략', '계획', '목표', '발전방향']
        }
    
    def analyze_sentiment(self, text: str) -> float:
        """감정 분석 (긍정도 점수 반환)"""
        if not text:
            return 0.5
        
        text_lower = text.lower()
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.5
        
        # 정규화된 감정 점수 (0.0 ~ 1.0)
        sentiment_score = 0.5 + (positive_count - negative_count) / max(total_words, 1)
        return max(0.0, min(1.0, sentiment_score))
    
    def calculate_keyword_match(self, text: str, keywords: List[str]) -> float:
        """키워드 매칭 점수 계산"""
        if not text or not keywords:
            return 0.0
        
        text_lower = text.lower()
        matched_keywords = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        
        return min(1.0, matched_keywords / len(keywords))
    
    def calculate_coherence_score(self, text: str) -> float:
        """답변 일관성 점수 계산"""
        if not text or len(text.split()) < 10:
            return 0.3
        
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return 0.5
        
        # 문장 간 연결성 분석 (간단한 휴리스틱)
        transition_words = ['그래서', '따라서', '또한', '하지만', '그러나', '즉', '예를 들어', '결과적으로']
        transition_count = sum(1 for word in transition_words if word in text)
        
        # 문장 길이 일관성
        sentence_lengths = [len(s.split()) for s in sentences]
        length_variance = np.var(sentence_lengths) if sentence_lengths else 0
        
        # 점수 계산
        coherence_score = 0.5 + (transition_count / len(sentences)) * 0.3
        coherence_score -= min(0.2, length_variance / 100)  # 길이 편차 페널티
        
        return max(0.0, min(1.0, coherence_score))

class QdrantRAGService:
    """Qdrant RAG 서비스 (Fallback 포함)"""
    
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "lignex1_news",
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        
        self.collection_name = collection_name
        self.analyzer = TextAnalyzer()
        self.model_name = model_name
        
        try:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            logger.info(f"Qdrant 연결 성공: {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.error(f"Qdrant 연결 실패: {e}")
            self.qdrant_client = None
        
        # 임베딩 모델 로드 (SentenceTransformer 우선, 실패시 TF-IDF)
        self.embedding_model = None
        self.vector_size = 384  # 기본값
        
        # 먼저 SentenceTransformer 시도
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name)
            self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer 로드 완료: {model_name}")
            self.use_sentence_transformer = True
        except Exception as e:
            logger.warning(f"SentenceTransformer 로드 실패: {e}")
            logger.info("TF-IDF 기반 임베딩으로 대체합니다...")
            
            # TF-IDF 기반 임베딩으로 대체
            try:
                self.embedding_model = SimpleTfidfEmbedder(vector_size=384)
                self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"TF-IDF 임베딩 모델 로드 완료")
                self.use_sentence_transformer = False
            except Exception as e2:
                logger.error(f"TF-IDF 임베딩 모델 로드도 실패: {e2}")
                self.embedding_model = None
    
    def is_available(self) -> bool:
        """RAG 서비스 사용 가능 여부 확인"""
        return self.qdrant_client is not None and self.embedding_model is not None
    
    def search_context(self, query: str, keywords: List[str] = None, limit: int = 5) -> List[Dict]:
        """컨텍스트 검색"""
        if not self.is_available():
            return []
        
        try:
            # 쿼리 벡터화
            query_vector = self.embedding_model.encode(query)
            
            # TF-IDF 모델의 경우 더 낮은 threshold 사용
            score_threshold = 0.3 if self.use_sentence_transformer else 0.1
            
            # 검색 실행
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold
            )
            
            # 결과 포맷팅
            contexts = []
            for hit in search_result:
                context = {
                    'id': hit.id,
                    'score': hit.score,
                    'title': hit.payload.get('title', ''),
                    'content': hit.payload.get('content', ''),
                    'source': hit.payload.get('source', ''),
                    'url': hit.payload.get('url', ''),
                    'keywords': hit.payload.get('keywords', []),
                    'published_date': hit.payload.get('published_date', ''),
                    'relevance_score': hit.score,
                    'embedding_model': hit.payload.get('embedding_model', 'unknown')
                }
                contexts.append(context)
            
            return contexts
            
        except Exception as e:
            logger.error(f"컨텍스트 검색 실패: {e}")
            return []
    
    def get_enhanced_context(self, question: InterviewQuestion) -> List[Dict]:
        """질문에 대한 향상된 컨텍스트 제공"""
        all_contexts = []
        
        # 질문 텍스트로 검색
        contexts = self.search_context(question.text, limit=3)
        all_contexts.extend(contexts)
        
        # 키워드별 검색
        for keyword in question.context_keywords[:3]:
            contexts = self.search_context(keyword, limit=2)
            all_contexts.extend(contexts)
        
        # 중복 제거 및 점수순 정렬
        unique_contexts = {}
        for context in all_contexts:
            context_id = context['id']
            if context_id not in unique_contexts or context['score'] > unique_contexts[context_id]['score']:
                unique_contexts[context_id] = context
        
        return sorted(unique_contexts.values(), key=lambda x: x['score'], reverse=True)[:5]

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

class InterviewEvaluator:
    """면접 평가 클래스"""
    
    def __init__(self):
        self.analyzer = TextAnalyzer()
        
        # 평가 기준 가중치
        self.evaluation_weights = {
            'content_relevance': 0.3,      # 내용 관련성
            'keyword_match': 0.2,          # 키워드 매칭
            'sentiment': 0.15,             # 긍정도
            'coherence': 0.2,              # 일관성
            'length_appropriateness': 0.15  # 답변 길이 적절성
        }
    
    def evaluate_answer(self, question: InterviewQuestion, answer: str, contexts: List[Dict]) -> Dict:
        """답변 평가"""
        if not answer.strip():
            return {
                'overall_score': 0.0,
                'detailed_scores': {},
                'feedback': "답변이 제공되지 않았습니다.",
                'strengths': [],
                'improvements': ["답변을 제공해 주세요."]
            }
        
        # 각 평가 항목별 점수 계산
        scores = {}
        
        # 1. 키워드 매칭 점수
        scores['keyword_match'] = self.analyzer.calculate_keyword_match(answer, question.context_keywords)
        
        # 2. 감정 분석 점수
        scores['sentiment'] = self.analyzer.analyze_sentiment(answer)
        
        # 3. 일관성 점수
        scores['coherence'] = self.analyzer.calculate_coherence_score(answer)
        
        # 4. 답변 길이 적절성
        word_count = len(answer.split())
        if question.difficulty_level == "easy":
            optimal_range = (50, 150)
        elif question.difficulty_level == "hard":
            optimal_range = (150, 300)
        else:
            optimal_range = (100, 200)
        
        if optimal_range[0] <= word_count <= optimal_range[1]:
            scores['length_appropriateness'] = 1.0
        elif word_count < optimal_range[0]:
            scores['length_appropriateness'] = max(0.3, word_count / optimal_range[0])
        else:
            scores['length_appropriateness'] = max(0.5, optimal_range[1] / word_count)
        
        # 5. 컨텍스트 관련성 (RAG 기반)
        if contexts:
            context_keywords = []
            for ctx in contexts:
                context_keywords.extend(ctx.get('keywords', []))
            
            scores['content_relevance'] = self.analyzer.calculate_keyword_match(answer, context_keywords)
        else:
            scores['content_relevance'] = 0.5
        
        # 전체 점수 계산
        overall_score = sum(
            scores[key] * self.evaluation_weights[key] 
            for key in scores
        )
        
        # 피드백 생성
        feedback, strengths, improvements = self._generate_feedback(scores, word_count, optimal_range)
        
        return {
            'overall_score': round(overall_score, 2),
            'detailed_scores': {k: round(v, 2) for k, v in scores.items()},
            'feedback': feedback,
            'strengths': strengths,
            'improvements': improvements
        }
    
    def _generate_feedback(self, scores: Dict, word_count: int, optimal_range: Tuple[int, int]) -> Tuple[str, List[str], List[str]]:
        """피드백 생성"""
        strengths = []
        improvements = []
        
        # 강점 분석
        if scores['keyword_match'] >= 0.7:
            strengths.append("질문과 관련된 핵심 키워드를 잘 활용하셨습니다.")
        
        if scores['sentiment'] >= 0.7:
            strengths.append("긍정적이고 적극적인 태도가 잘 드러났습니다.")
        
        if scores['coherence'] >= 0.7:
            strengths.append("논리적이고 일관성 있는 답변 구조를 보여주셨습니다.")
        
        if scores['length_appropriateness'] >= 0.8:
            strengths.append("적절한 길이로 답변해 주셨습니다.")
        
        # 개선점 분석
        if scores['keyword_match'] < 0.5:
            improvements.append("질문의 핵심 키워드와 더 밀접한 연관성을 보여주세요.")
        
        if scores['sentiment'] < 0.5:
            improvements.append("더 긍정적이고 자신감 있는 표현을 사용해 보세요.")
        
        if scores['coherence'] < 0.5:
            improvements.append("답변의 논리적 구조를 개선하고 연결어를 활용해 보세요.")
        
        if word_count < optimal_range[0]:
            improvements.append(f"답변을 더 구체적으로 설명해 주세요. (현재 {word_count}단어, 권장 {optimal_range[0]}-{optimal_range[1]}단어)")
        elif word_count > optimal_range[1]:
            improvements.append(f"답변을 더 간결하게 정리해 주세요. (현재 {word_count}단어, 권장 {optimal_range[0]}-{optimal_range[1]}단어)")
        
        # 전체 피드백
        overall_score = sum(scores.values()) / len(scores)
        if overall_score >= 0.8:
            feedback = "우수한 답변입니다! 🎉"
        elif overall_score >= 0.6:
            feedback = "양호한 답변입니다. 몇 가지 개선하면 더 좋겠습니다. 👍"
        else:
            feedback = "답변을 더 발전시킬 여지가 있습니다. 💪"
        
        return feedback, strengths, improvements

class InterviewManager:
    """면접 관리 클래스"""
    
    def __init__(self):
        self.rag_service = QdrantRAGService()
        self.evaluator = InterviewEvaluator()
        self.db_path = "lignex1_interview_sessions.db"
        self._init_database()
        
        # 면접 질문들
        self.questions = self._load_interview_questions()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 면접 세션 테이블
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS interview_sessions (
                        session_id TEXT PRIMARY KEY,
                        candidate_name TEXT NOT NULL,
                        position TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        overall_score REAL DEFAULT 0.0,
                        feedback TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 답변 테이블
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS interview_answers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        question_id TEXT NOT NULL,
                        question_text TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        evaluation_score REAL DEFAULT 0.0,
                        keyword_match_score REAL DEFAULT 0.0,
                        sentiment_score REAL DEFAULT 0.0,
                        coherence_score REAL DEFAULT 0.0,
                        answer_length INTEGER DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        context_used TEXT,
                        FOREIGN KEY (session_id) REFERENCES interview_sessions (session_id)
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def _load_interview_questions(self) -> List[InterviewQuestion]:
        """면접 질문 로드"""
        return [
            InterviewQuestion(
                id="culture_open",
                text="LIG넥스원의 핵심가치 중 '개방(OPEN)'에 대해 어떻게 이해하고 계시며, 본인의 경험 중 개방적 사고로 문제를 해결한 사례가 있다면 구체적으로 말씀해 주세요.",
                category="핵심가치 이해도",
                context_keywords=["기업문화", "핵심가치", "OPEN", "개방", "조직문화", "소통", "협력"],
                follow_up_questions=[
                    "그 상황에서 다른 사람들의 의견을 어떻게 수용하셨나요?",
                    "개방적 사고가 어려웠던 순간은 언제였고, 어떻게 극복하셨나요?",
                    "팀 내 갈등 상황에서 개방적 태도로 해결한 경험이 있나요?"
                ],
                difficulty_level="medium",
                expected_duration=5,
                evaluation_criteria=["핵심가치 이해도", "실제 경험 구체성", "문제해결 능력", "소통 능력"]
            ),
            InterviewQuestion(
                id="culture_positive",
                text="LIG넥스원의 또 다른 핵심가치인 '긍정(POSITIVE)'에 대한 본인의 해석을 들려주시고, 어려운 상황에서도 긍정적 태도를 유지하며 성과를 창출했던 경험을 상세히 공유해 주세요.",
                category="긍정적 사고",
                context_keywords=["POSITIVE", "긍정", "도전", "극복", "열정", "성과", "동기부여"],
                follow_up_questions=[
                    "실패를 성공의 밑거름으로 활용한 구체적인 사례가 있나요?",
                    "팀원들의 사기가 낮을 때 어떻게 동기부여 하시겠나요?",
                    "스트레스가 높은 상황에서 긍정성을 유지하는 본인만의 방법은?"
                ],
                difficulty_level="medium",
                expected_duration=5,
                evaluation_criteria=["긍정적 사고", "역경 극복 능력", "리더십", "팀워크"]
            ),
            InterviewQuestion(
                id="mission_defense",
                text="방위산업은 국가 안보와 직결되는 중요한 분야입니다. 이러한 막중한 책임감이 개인의 업무 수행에 어떤 영향을 미칠 것이라고 생각하시며, 국가를 위한 일이라는 사명감을 어떻게 실무에 적용하실 계획인가요?",
                category="책임감과 사명감",
                context_keywords=["방위산업", "국방", "안보", "책임감", "사명감", "국가", "보안"],
                follow_up_questions=[
                    "보안과 기밀 유지에 대한 본인의 각오는 어떠한지요?",
                    "압박감과 책임감을 건설적으로 관리하는 방법은?",
                    "개인의 성장과 국가적 사명 사이의 균형을 어떻게 맞추시겠나요?"
                ],
                difficulty_level="hard",
                expected_duration=6,
                evaluation_criteria=["사명감", "책임감", "보안 의식", "압박감 관리"]
            ),
            InterviewQuestion(
                id="growth_rd",
                text="LIG넥스원은 전체 직원의 50%가 R&D 인력으로 구성된 기술 중심 조직입니다. 급변하는 기술 환경에서 지속적인 학습과 혁신이 필요한 조직에서 본인만의 성장 전략과 기여 방안을 구체적으로 설명해 주세요.",
                category="학습 지향성과 혁신",
                context_keywords=["R&D", "연구개발", "기술개발", "혁신", "학습", "성장", "첨단기술"],
                follow_up_questions=[
                    "새로운 기술 트렌드를 습득하고 적용하는 본인만의 방법론은?",
                    "동료들과 지식을 공유하고 시너지를 창출하는 방안은?",
                    "기술적 난제에 부딪혔을 때의 문제해결 접근법은?"
                ],
                difficulty_level="hard",
                expected_duration=6,
                evaluation_criteria=["학습 능력", "기술 이해도", "혁신 사고", "지식 공유"]
            ),
            InterviewQuestion(
                id="vision_future",
                text="LIG넥스원에서 10년 후 본인의 모습을 구체적으로 그려보시고, 개인의 성장과 회사 발전에 기여하기 위한 단계적 계획과 비전을 상세히 말씀해 주세요.",
                category="비전과 성장 의지",
                context_keywords=["미래", "비전", "성장", "목표", "계획", "발전", "커리어"],
                follow_up_questions=[
                    "회사의 미래 성장 동력에 어떤 기여를 하고 싶으신가요?",
                    "개인 성장과 조직 발전의 균형을 어떻게 맞추시겠나요?",
                    "예상되는 어려움과 이를 극복하는 전략은?"
                ],
                difficulty_level="medium",
                expected_duration=5,
                evaluation_criteria=["비전 설정", "목표 구체성", "실행 계획", "성장 의지"]
            ),
            InterviewQuestion(
                id="teamwork_collaboration",
                text="다양한 배경을 가진 팀원들과 협업하여 큰 성과를 달성했던 경험을 구체적으로 설명하시고, LIG넥스원에서도 효과적인 협업을 위해 어떤 역할을 하실 것인지 말씀해 주세요.",
                category="팀워크와 협업",
                context_keywords=["팀워크", "협업", "소통", "리더십", "갈등해결", "성과", "프로젝트"],
                follow_up_questions=[
                    "팀 내 갈등 상황을 해결한 경험이 있다면?",
                    "본인의 협업 스타일과 강점은 무엇인가요?",
                    "다양한 부서와의 협업에서 중요한 요소는?"
                ],
                difficulty_level="medium",
                expected_duration=5,
                evaluation_criteria=["협업 능력", "소통 능력", "갈등 해결", "리더십"]
            ),
            InterviewQuestion(
                id="problem_solving",
                text="복잡하고 어려운 문제 상황에서 창의적이고 체계적인 접근으로 해결책을 찾아낸 경험을 상세히 설명하시고, LIG넥스원에서 마주할 수 있는 기술적/업무적 도전에 어떻게 대응하실 것인지 말씀해 주세요.",
                category="문제해결 능력",
                context_keywords=["문제해결", "창의성", "분석", "체계적", "논리적", "혁신", "도전"],
                follow_up_questions=[
                    "문제 분석과 해결 과정에서 가장 중요하게 생각하는 요소는?",
                    "기존 방식으로 해결되지 않는 문제에 어떻게 접근하시나요?",
                    "문제해결 과정에서 다른 사람들과 어떻게 협력하시나요?"
                ],
                difficulty_level="hard",
                expected_duration=6,
                evaluation_criteria=["논리적 사고", "창의성", "체계적 접근", "실행력"]
            ),
            InterviewQuestion(
                id="communication_leadership",
                text="다른 사람들을 설득하고 이끌어야 했던 어려운 상황에서, 어떤 소통 전략과 리더십을 발휘하여 성공적인 결과를 만들어낸 경험을 구체적으로 공유해 주세요.",
                category="소통과 리더십",
                context_keywords=["소통", "리더십", "설득", "영향력", "동기부여", "변화관리", "성과"],
                follow_up_questions=[
                    "반대 의견이 많은 상황에서 어떻게 합의를 이끌어내시나요?",
                    "본인의 리더십 스타일과 철학은 무엇인가요?",
                    "팀원들의 다양한 의견을 조율하는 방법은?"
                ],
                difficulty_level="medium",
                expected_duration=5,
                evaluation_criteria=["소통 능력", "리더십", "설득력", "갈등 조정"]
            )
        ]
        
    def create_session(self, candidate_name: str, position: str) -> str:
        """면접 세션 생성"""
        try:
            session_id = hashlib.md5(f"{candidate_name}_{position}_{datetime.now().isoformat()}".encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO interview_sessions 
                    (session_id, candidate_name, position, start_time)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, candidate_name, position, datetime.now().isoformat()))
                
                conn.commit()
                logger.info(f"면접 세션 생성: {session_id}")
                return session_id
                
        except Exception as e:
            logger.error(f"세션 생성 실패: {e}")
            st.error(f"세션 생성 중 오류가 발생했습니다: {str(e)}")
            return None

        
    def save_answer(self, session_id: str, question: InterviewQuestion, answer: str, contexts: List[Dict]) -> Dict:
        """답변 저장 및 평가"""
        if not answer or not answer.strip():
            return {'error': '답변이 비어있습니다.'}
        
        # 추가 로그로 디버깅
        logger.info(f"답변 저장 시도: session_id={session_id}, answer_length={len(answer)}")
        
        # 답변 평가
        evaluation = self.evaluator.evaluate_answer(question, answer, contexts)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO interview_answers 
                    (session_id, question_id, question_text, answer, evaluation_score,
                     keyword_match_score, sentiment_score, coherence_score, answer_length,
                     timestamp, context_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    question.id,
                    question.text,
                    answer,
                    evaluation['overall_score'],
                    evaluation['detailed_scores'].get('keyword_match', 0),
                    evaluation['detailed_scores'].get('sentiment', 0),
                    evaluation['detailed_scores'].get('coherence', 0),
                    len(answer.split()),
                    datetime.now().isoformat(),
                    json.dumps(contexts, ensure_ascii=False)
                ))
                
                conn.commit()
                logger.info(f"답변 저장 완료: {session_id} - {question.id}")
                
        except Exception as e:
            logger.error(f"답변 저장 실패: {e}")
            return {'error': f'저장 실패: {str(e)}'}
        
        return evaluation
    
    def complete_session(self, session_id: str) -> Dict:
        """면접 세션 완료"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 전체 점수 계산
                cursor = conn.execute('''
                    SELECT AVG(evaluation_score), COUNT(*)
                    FROM interview_answers
                    WHERE session_id = ?
                ''', (session_id,))
                
                result = cursor.fetchone()
                overall_score = result[0] if result[0] else 0.0
                answer_count = result[1]
                
                # 세션 업데이트
                conn.execute('''
                    UPDATE interview_sessions 
                    SET end_time = ?, overall_score = ?
                    WHERE session_id = ?
                ''', (datetime.now().isoformat(), overall_score, session_id))
                
                conn.commit()
                
                # 상세 분석 데이터 조회
                cursor = conn.execute('''
                    SELECT question_id, evaluation_score, keyword_match_score, 
                           sentiment_score, coherence_score, answer_length
                    FROM interview_answers
                    WHERE session_id = ?
                    ORDER BY timestamp
                ''', (session_id,))
                
                answers_data = cursor.fetchall()
                
                return {
                    'session_id': session_id,
                    'overall_score': round(overall_score, 2),
                    'answer_count': answer_count,
                    'answers_data': answers_data,
                    'completed_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"세션 완료 처리 실패: {e}")
            return {'error': str(e)}
    
    def get_session_analytics(self, session_id: str) -> Dict:
        """세션 분석 데이터"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 기본 정보
                cursor = conn.execute('''
                    SELECT candidate_name, position, start_time, end_time, overall_score
                    FROM interview_sessions
                    WHERE session_id = ?
                ''', (session_id,))
                
                session_info = cursor.fetchone()
                if not session_info:
                    return {'error': '세션을 찾을 수 없습니다.'}
                
                # 답변별 상세 데이터
                cursor = conn.execute('''
                    SELECT question_id, question_text, answer, evaluation_score,
                           keyword_match_score, sentiment_score, coherence_score,
                           answer_length, timestamp, context_used
                    FROM interview_answers
                    WHERE session_id = ?
                    ORDER BY timestamp
                ''', (session_id,))
                
                answers = cursor.fetchall()
                
                # 통계 계산
                if answers:
                    scores = [a[3] for a in answers]  # evaluation_score
                    avg_score = np.mean(scores)
                    score_std = np.std(scores)
                    
                    sentiment_scores = [a[5] for a in answers]
                    avg_sentiment = np.mean(sentiment_scores)
                    
                    coherence_scores = [a[6] for a in answers]
                    avg_coherence = np.mean(coherence_scores)
                    
                    answer_lengths = [a[7] for a in answers]
                    avg_length = np.mean(answer_lengths)
                else:
                    avg_score = score_std = avg_sentiment = avg_coherence = avg_length = 0
                
                return {
                    'session_info': {
                        'candidate_name': session_info[0],
                        'position': session_info[1],
                        'start_time': session_info[2],
                        'end_time': session_info[3],
                        'overall_score': session_info[4]
                    },
                    'answers': [
                        {
                            'question_id': a[0],
                            'question_text': a[1],
                            'answer': a[2],
                            'evaluation_score': a[3],
                            'keyword_match_score': a[4],
                            'sentiment_score': a[5],
                            'coherence_score': a[6],
                            'answer_length': a[7],
                            'timestamp': a[8],
                            'context_used': json.loads(a[9]) if a[9] else []
                        } for a in answers
                    ],
                    'statistics': {
                        'average_score': round(avg_score, 2),
                        'score_deviation': round(score_std, 2),
                        'average_sentiment': round(avg_sentiment, 2),
                        'average_coherence': round(avg_coherence, 2),
                        'average_answer_length': round(avg_length, 1),
                        'total_answers': len(answers)
                    }
                }
                
        except Exception as e:
            logger.error(f"세션 분석 실패: {e}")
            return {'error': str(e)}
    
    def get_all_sessions(self) -> List[Dict]:
        """모든 세션 목록"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT session_id, candidate_name, position, 
                        start_time, end_time, overall_score
                    FROM interview_sessions
                    ORDER BY start_time DESC
                ''')
                
                sessions = cursor.fetchall()
                
                # 진행중인 세션의 답변 수도 조회
                result = []
                for s in sessions:
                    session_data = {
                        'session_id': s[0],
                        'candidate_name': s[1],
                        'position': s[2],
                        'start_time': s[3],
                        'end_time': s[4],
                        'overall_score': s[5] or 0.0,
                        'status': 'completed' if s[4] else 'in_progress'
                    }
                    
                    # 진행중인 세션의 답변 수 확인
                    if not s[4]:  # end_time이 없으면 진행중
                        cursor2 = conn.execute('''
                            SELECT COUNT(*) FROM interview_answers
                            WHERE session_id = ?
                        ''', (s[0],))
                        answer_count = cursor2.fetchone()[0]
                        session_data['answer_count'] = answer_count
                    
                    result.append(session_data)
                
                return result
                
        except Exception as e:
            logger.error(f"세션 목록 조회 실패: {e}")
            return []

def show_interview_records_page():
    """면접 기록 페이지"""
    st.markdown('<h2 class="sub-header">📋 면접 기록 관리</h2>', unsafe_allow_html=True)
    
    # 전체 세션 조회
    sessions = st.session_state.interview_manager.get_all_sessions()
    
    if not sessions:
        st.info("📋 아직 진행된 면접이 없습니다.")
        return
    
    # 필터링 옵션
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "상태 필터",
            ["전체", "진행중", "완료"]
        )
    
    with col2:
        # 직무별 필터
        positions = list(set([s['position'] for s in sessions]))
        position_filter = st.selectbox(
            "직무 필터",
            ["전체"] + positions
        )
    
    with col3:
        # 날짜 범위 필터
        date_filter = st.selectbox(
            "기간 필터",
            ["전체", "오늘", "최근 7일", "최근 30일"]
        )
    
    # 필터링 적용
    filtered_sessions = sessions.copy()
    
    if status_filter != "전체":
        status_map = {"진행중": "in_progress", "완료": "completed"}
        filtered_sessions = [s for s in filtered_sessions if s['status'] == status_map[status_filter]]
    
    if position_filter != "전체":
        filtered_sessions = [s for s in filtered_sessions if s['position'] == position_filter]
    
    if date_filter != "전체":
        from datetime import datetime, timedelta
        now = datetime.now()
        
        if date_filter == "오늘":
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_filter == "최근 7일":
            cutoff = now - timedelta(days=7)
        elif date_filter == "최근 30일":
            cutoff = now - timedelta(days=30)
        
        filtered_sessions = [
            s for s in filtered_sessions 
            if datetime.fromisoformat(s['start_time']) >= cutoff
        ]
    
    # 통계 요약
    st.markdown("### 📊 요약 통계")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("전체 면접", len(sessions))
    
    with col2:
        completed = len([s for s in sessions if s['status'] == 'completed'])
        st.metric("완료된 면접", completed)
    
    with col3:
        in_progress = len([s for s in sessions if s['status'] == 'in_progress'])
        st.metric("진행중인 면접", in_progress)
    
    with col4:
        if completed > 0:
            avg_score = np.mean([s['overall_score'] for s in sessions if s['overall_score'] > 0])
            st.metric("평균 점수", f"{avg_score:.2f}")
        else:
            st.metric("평균 점수", "N/A")
    
    # 세션 목록 표시
    st.markdown("### 📋 면접 목록")
    
    if not filtered_sessions:
        st.info("선택한 조건에 맞는 면접이 없습니다.")
        return
    
    # 데이터프레임으로 변환
    df_sessions = pd.DataFrame(filtered_sessions)
    df_sessions['start_date'] = pd.to_datetime(df_sessions['start_time']).dt.strftime('%Y-%m-%d %H:%M')
    df_sessions['status_kr'] = df_sessions['status'].map({'completed': '완료', 'in_progress': '진행중'})

    # 세션 상세 정보에도 한국어 상태 추가
    for session in filtered_sessions:
        session['status_kr'] = '완료' if session['status'] == 'completed' else '진행중'
    
    # 표 표시
    display_df = df_sessions[['candidate_name', 'position', 'start_date', 'status_kr', 'overall_score']].copy()
    display_df.columns = ['면접자', '지원직무', '시작시간', '상태', '점수']
    display_df['점수'] = display_df['점수'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
    
    st.dataframe(display_df, use_container_width=True)
    
    # 개별 세션 상세 보기
    st.markdown("### 🔍 상세 보기")
    
    session_options = {
        f"{s['candidate_name']} - {s['position']} ({s['start_time'][:10]})": s['session_id'] 
        for s in filtered_sessions
    }
    
    if session_options:
        selected_session_label = st.selectbox(
            "상세히 볼 면접 선택",
            ["선택하세요..."] + list(session_options.keys())
        )
        
        if selected_session_label != "선택하세요...":
            selected_session_id = session_options[selected_session_label]
            
            # 세션 상세 정보
            session_detail = next(s for s in filtered_sessions if s['session_id'] == selected_session_id)
            
            with st.expander("📋 세션 정보", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**세션 ID**: {session_detail['session_id'][:16]}...")
                    st.write(f"**면접자**: {session_detail['candidate_name']}")
                    st.write(f"**지원 직무**: {session_detail['position']}")
                
                with col2:
                    st.write(f"**시작 시간**: {session_detail['start_time'][:19]}")
                    if session_detail['end_time']:
                        st.write(f"**종료 시간**: {session_detail['end_time'][:19]}")
                    st.write(f"**상태**: {session_detail['status_kr']}")
                    if session_detail['overall_score'] > 0:
                        st.write(f"**종합 점수**: {session_detail['overall_score']:.2f}")
            
            # 답변 목록 (완료된 세션만)
            if session_detail['status'] == 'completed':
                analytics = st.session_state.interview_manager.get_session_analytics(selected_session_id)
                
                if 'error' not in analytics:
                    answers = analytics['answers']
                    
                    st.markdown("#### 📝 답변 목록")
                    
                    for i, answer in enumerate(answers, 1):
                        with st.expander(f"질문 {i} (점수: {answer['evaluation_score']:.2f})", expanded=False):
                            st.markdown(f"**질문**: {answer['question_text']}")
                            st.markdown(f"**답변**: {answer['answer'][:300]}...")
                            st.markdown(f"**답변 길이**: {answer['answer_length']}단어")
                            st.markdown(f"**작성 시간**: {answer['timestamp'][:19]}")

def show_system_status_page():
    """시스템 상태 페이지"""
    st.markdown('<h2 class="sub-header">🔧 시스템 상태</h2>', unsafe_allow_html=True)
    
    # RAG 시스템 상태
    st.markdown("### 🤖 RAG 시스템 상태")
    
    rag_service = st.session_state.interview_manager.rag_service
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔗 연결 상태")
        
        if rag_service.is_available():
            st.success("✅ RAG 시스템 정상 작동")
            
            # Qdrant 정보
            try:
                collections = rag_service.qdrant_client.get_collections().collections
                collection_exists = any(c.name == rag_service.collection_name for c in collections)
                
                if collection_exists:
                    st.success(f"✅ 컬렉션 '{rag_service.collection_name}' 존재")
                    
                    # 컬렉션 상세 정보
                    try:
                        # 컬렉션 기본 정보만 조회 (버전 호환성 문제 회피)
                        collection_info = rag_service.qdrant_client.get_collection(rag_service.collection_name)
                        
                        # 안전하게 정보 추출
                        if hasattr(collection_info, 'vectors_count'):
                            vector_count = collection_info.vectors_count
                        else:
                            vector_count = "알 수 없음"
                        
                        st.info(f"📊 컬렉션: {rag_service.collection_name}")
                        st.info(f"📊 벡터 데이터: {vector_count}")
                        st.success("✅ 컬렉션 정상 작동")
                        
                    except Exception as detail_error:
                        # 상세 정보 조회 실패시 기본 정보만 표시
                        st.success(f"✅ 컬렉션 '{rag_service.collection_name}' 존재")
                        st.info("📊 상세 정보 조회 불가 (버전 호환성 문제)")
                        logger.warning(f"Collection detail query failed: {detail_error}")


                else:
                    st.error(f"❌ 컬렉션 '{rag_service.collection_name}' 없음")
                    
            except Exception as e:
                st.error(f"❌ Qdrant 정보 조회 실패: {e}")
        else:
            st.error("❌ RAG 시스템 연결 실패")
            st.warning("Qdrant 서버가 실행 중인지 확인하세요.")
    
    with col2:
        st.markdown("#### 🧠 임베딩 모델")
        
        if rag_service.embedding_model:
            st.success("✅ 임베딩 모델 로드됨")
            st.info(f"📏 벡터 차원: {rag_service.vector_size}")
            
            # 테스트 임베딩
            if st.button("🧪 임베딩 테스트"):
                test_text = "LIG넥스원 기업문화"
                try:
                    with st.spinner("임베딩 생성 중..."):
                        embedding = rag_service.embedding_model.encode(test_text)
                    st.success(f"✅ 테스트 성공 (벡터 길이: {len(embedding)})")
                except Exception as e:
                    st.error(f"❌ 임베딩 테스트 실패: {e}")
        else:
            st.error("❌ 임베딩 모델 로드 실패")
    
    # 데이터베이스 상태
    st.markdown("---")
    st.markdown("### 🗄️ 데이터베이스 상태")
    
    try:
        with sqlite3.connect(st.session_state.interview_manager.db_path) as conn:
            # 테이블 존재 확인
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('interview_sessions', 'interview_answers')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 테이블 상태")
                if 'interview_sessions' in tables:
                    st.success("✅ interview_sessions 테이블 존재")
                else:
                    st.error("❌ interview_sessions 테이블 없음")
                
                if 'interview_answers' in tables:
                    st.success("✅ interview_answers 테이블 존재")
                else:
                    st.error("❌ interview_answers 테이블 없음")
            
            with col2:
                st.markdown("#### 📊 데이터 통계")
                
                # 세션 수
                cursor = conn.execute("SELECT COUNT(*) FROM interview_sessions")
                session_count = cursor.fetchone()[0]
                st.metric("총 면접 세션", session_count)
                
                # 답변 수
                cursor = conn.execute("SELECT COUNT(*) FROM interview_answers")
                answer_count = cursor.fetchone()[0]
                st.metric("총 답변 수", answer_count)
                
                # 완료된 세션 수
                cursor = conn.execute("SELECT COUNT(*) FROM interview_sessions WHERE end_time IS NOT NULL")
                completed_count = cursor.fetchone()[0]
                st.metric("완료된 면접", completed_count)
                
    except Exception as e:
        st.error(f"❌ 데이터베이스 연결 실패: {e}")
    
    # 시스템 설정
    st.markdown("---")
    st.markdown("### ⚙️ 시스템 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 RAG 설정")
        st.code(f"""
Qdrant Host: {rag_service.qdrant_client.host if rag_service.qdrant_client else 'N/A'}
Qdrant Port: {rag_service.qdrant_client.port if rag_service.qdrant_client else 'N/A'}
Collection: {rag_service.collection_name}
Vector Size: {rag_service.vector_size if rag_service.embedding_model else 'N/A'}
        """)
    
    with col2:
        st.markdown("#### 📁 파일 경로")
        st.code(f"""
Database: {st.session_state.interview_manager.db_path}
Log File: lignex1_interview.log
        """)
    
    # 시스템 테스트
    st.markdown("---")
    st.markdown("### 🧪 시스템 테스트")
    
    if st.button("🔍 전체 시스템 테스트", use_container_width=True):
        with st.spinner("시스템 테스트 진행 중..."):
            test_results = []
            
            # RAG 시스템 테스트
            if rag_service.is_available():
                try:
                    contexts = rag_service.search_context("LIG넥스원", limit=1)
                    if contexts:
                        test_results.append("✅ RAG 검색 테스트 성공")
                    else:
                        test_results.append("⚠️ RAG 검색 결과 없음 (데이터 부족)")
                except Exception as e:
                    test_results.append(f"❌ RAG 검색 테스트 실패: {e}")
            else:
                test_results.append("❌ RAG 시스템 연결 안됨")
            
            # 데이터베이스 테스트
            try:
                test_session_id = st.session_state.interview_manager.create_session("테스트", "테스트직무")
                if test_session_id:
                    test_results.append("✅ 데이터베이스 쓰기 테스트 성공")
                    
                    # 테스트 세션 삭제
                    with sqlite3.connect(st.session_state.interview_manager.db_path) as conn:
                        conn.execute("DELETE FROM interview_sessions WHERE session_id = ?", (test_session_id,))
                        conn.commit()
                else:
                    test_results.append("❌ 데이터베이스 쓰기 테스트 실패")
            except Exception as e:
                test_results.append(f"❌ 데이터베이스 테스트 실패: {e}")
            
            # 평가 시스템 테스트
            try:
                test_question = st.session_state.interview_manager.questions[0]
                evaluation = st.session_state.interview_manager.evaluator.evaluate_answer(
                    test_question, "테스트 답변입니다.", []
                )
                if evaluation and 'overall_score' in evaluation:
                    test_results.append("✅ 평가 시스템 테스트 성공")
                else:
                    test_results.append("❌ 평가 시스템 테스트 실패")
            except Exception as e:
                test_results.append(f"❌ 평가 시스템 테스트 실패: {e}")
        
        # 테스트 결과 표시
        st.markdown("#### 🔍 테스트 결과")
        for result in test_results:
            st.write(result)
    
    # 로그 파일 조회
    st.markdown("---")
    st.markdown("### 📋 최근 로그")
    
    log_file = "lignex1_interview.log"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
            
            # 최근 20줄만 표시
            recent_logs = log_lines[-20:] if len(log_lines) > 20 else log_lines
            
            st.text_area(
                "로그 내용",
                value=''.join(recent_logs),
                height=200,
                disabled=True
            )
            
            if st.button("📥 전체 로그 다운로드"):
                with open(log_file, 'rb') as f:
                    st.download_button(
                        label="로그 파일 다운로드",
                        data=f.read(),
                        file_name=f"lignex1_interview_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                        mime="text/plain"
                    )
        except Exception as e:
            st.error(f"로그 파일 읽기 실패: {e}")
    else:
        st.info("로그 파일이 없습니다.")


def show_new_interview_page():
    """새 면접 시작 페이지"""
    st.markdown('<h2 class="sub-header">🆕 새 면접 시작</h2>', unsafe_allow_html=True)

    # Ensure interview_manager is initialized
    if 'interview_manager' not in st.session_state:
        st.session_state.interview_manager = InterviewManager()

    # Form for new interview
    with st.form("new_interview_form"):
        st.markdown("### 👤 면접자 정보")

        candidate_name = st.text_input(
            "면접자 이름 *",
            placeholder="예: 홍길동"
        )

        position = st.selectbox(
            "지원 직무 *",
            [
                "연구개발(R&D)",
                "소프트웨어 엔지니어",
                "하드웨어 엔지니어",
                "시스템 엔지니어",
                "품질관리",
                "기술영업",
                "사업관리",
                "기타"
            ]
        )

        st.markdown("### 📋 면접 설정")

        interview_type = st.selectbox(
            "면접 유형",
            [
                "조직적합도 인성면접 (기본)",
                "핵심가치 중심 면접",
                "리더십 역량 면접",
                "커스텀 면접"
            ]
        )

        question_count = st.slider(
            "질문 수",
            min_value=3,
            max_value=8,
            value=5,
            help="면접 시간에 맞춰 3-8개 질문 선택"
        )

        submitted = st.form_submit_button(
            "🚀 면접 시작",
            use_container_width=True
        )

        if submitted:
            if not candidate_name or not position:
                st.error("⚠️ 모든 필수 항목을 입력해주세요.")
            else:
                # Create new session
                session_id = st.session_state.interview_manager.create_session(
                    candidate_name, position
                )

                if session_id:
                    st.session_state.current_session_id = session_id
                    st.session_state.current_question_index = 0
                    st.session_state.interview_answers = []
                    st.session_state.selected_questions = st.session_state.interview_manager.questions[:question_count]
                    st.session_state.interview_started = True

                    st.success(f"✅ 면접이 시작되었습니다! (세션 ID: {session_id[:8]}...)")
                    st.info("📝 아래 버튼을 클릭하여 면접을 시작하세요.")
                else:
                    st.error("❌ 면접 세션 생성에 실패했습니다.")

# 폼 밖에서 면접 진행 버튼 표시
    if st.session_state.get('interview_started', False):
        if st.button("🚀 면접 진행하기", use_container_width=True, key="goto_interview"):
            st.session_state.auto_navigate_to_interview = True
            st.session_state.interview_started = False  # 버튼 숨기기
            st.rerun()

    # Interview statistics section
    st.markdown("---")
    st.markdown("### 📊 면접 통계")
    col1, col2 = st.columns(2)  # Define columns here

    with col1:
        sessions = st.session_state.interview_manager.get_all_sessions()
        total_sessions = len(sessions)
        completed_sessions = len([s for s in sessions if s['status'] == 'completed'])

        st.metric("전체 면접", total_sessions)
        st.metric("완료된 면접", completed_sessions)

    with col2:
        if completed_sessions > 0:
            avg_score = np.mean([s['overall_score'] for s in sessions if s['overall_score'] > 0])
            st.metric("평균 점수", f"{avg_score:.1f}점")
        else:
            st.metric("평균 점수", "N/A")

        st.markdown("---")
        st.markdown("### 💡 면접 가이드")

        with st.expander("면접 진행 방법"):
            st.markdown("""
            1. **면접자 정보 입력**: 이름과 지원 직무를 정확히 입력
            2. **질문 수 선택**: 면접 시간에 맞춰 3-8개 질문 선택
            3. **면접 진행**: 각 질문에 대해 충분히 생각한 후 답변
            4. **실시간 피드백**: AI가 답변을 분석하여 즉시 피드백 제공
            5. **종합 분석**: 면접 완료 후 상세한 분석 리포트 제공
            """)

        with st.expander("평가 기준"):
            st.markdown("""
            - **내용 관련성** (30%): 질문과 답변의 연관성
            - **키워드 매칭** (20%): 핵심 키워드 활용도
            - **일관성** (20%): 논리적 답변 구조
            - **긍정도** (15%): 적극적이고 긍정적인 태도
            - **답변 길이** (15%): 적절한 답변 분량
            """)

def show_interview_progress_page():
    """면접 진행 페이지"""
    st.markdown('<h2 class="sub-header">📝 면접 진행</h2>', unsafe_allow_html=True)
    
    # 세션 상태 확인
    if not st.session_state.current_session_id:
        st.warning("⚠️ 진행 중인 면접이 없습니다.")
        st.info("💡 '새 면접 시작' 페이지에서 면접을 시작하세요.")
        return
    
    # 현재 면접 정보 표시
    current_session = st.session_state.current_session_id
    current_index = st.session_state.current_question_index
    questions = st.session_state.selected_questions
    
    if current_index >= len(questions):
        st.success("🎉 모든 질문을 완료했습니다!")
        
        if st.button("📊 면접 완료 및 결과 확인", use_container_width=True):
            completion_result = st.session_state.interview_manager.complete_session(current_session)
            
            if 'error' not in completion_result:
                st.success("✅ 면접이 성공적으로 완료되었습니다!")
                st.balloons()
                
                # 세션 상태 초기화
                st.session_state.current_session_id = None
                st.session_state.current_question_index = 0
                st.session_state.interview_answers = []
                st.session_state.selected_questions = []
                
                st.info("📋 '결과 분석' 페이지에서 상세한 분석을 확인하세요.")
            else:
                st.error(f"❌ 면접 완료 처리 실패: {completion_result['error']}")
        
        return
    
    # 현재 질문
    current_question = questions[current_index]
    
    # 진행률 표시
    progress = (current_index) / len(questions)
    st.progress(progress, text=f"진행률: {current_index}/{len(questions)} ({progress*100:.1f}%)")
    
    # 질문 표시
    st.markdown(f"### 질문 {current_index + 1}/{len(questions)}")
    st.markdown(f'<div class="interview-card">{current_question.text}</div>', unsafe_allow_html=True)
    
    # 질문 정보
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📂 카테고리: {current_question.category}")
    with col2:
        st.info(f"⏱️ 예상 시간: {current_question.expected_duration}분")
    with col3:
        st.info(f"📊 난이도: {current_question.difficulty_level}")
    
    # RAG 컨텍스트 검색
    if st.session_state.interview_manager.rag_service.is_available():
        # 컨텍스트 검색을 세션 상태에 캐시
        context_key = f"contexts_{current_session}_{current_index}"
        
        if context_key not in st.session_state:
            with st.spinner("🔍 관련 컨텍스트 검색 중..."):
                st.session_state[context_key] = st.session_state.interview_manager.rag_service.get_enhanced_context(current_question)
        
        contexts = st.session_state[context_key]
        
        if contexts:
            with st.expander("💡 참고 정보 (AI가 찾은 관련 자료)", expanded=False):
                for i, ctx in enumerate(contexts, 1):
                    st.markdown(f"**{i}. {ctx['title']}** (관련도: {ctx['score']:.3f})")
                    st.markdown(f"{ctx['content'][:200]}...")
                    if ctx['url']:
                        st.markdown(f"[원문 보기]({ctx['url']})")
                    st.markdown("---")
    else:
        contexts = []
        st.warning("⚠️ RAG 시스템이 연결되지 않아 참고 정보를 제공할 수 없습니다.")
    
    # 답변 입력
    st.markdown("### 📝 답변을 입력하세요")
    
    # 답변 입력 키를 현재 질문에 고유하게 만들기
    answer_key = f"answer_{current_session}_{current_index}"
    
    answer = st.text_area(
        "답변",
        height=200,
        placeholder="구체적인 경험과 사례를 포함하여 답변해 주세요...",
        key=answer_key
    )
    
    # 답변 길이 정보
    if answer:
        word_count = len(answer.split())
        st.caption(f"💬 현재 답변 길이: {word_count}단어")
        
        if current_question.difficulty_level == "easy":
            optimal_range = "50-150단어"
        elif current_question.difficulty_level == "hard":
            optimal_range = "150-300단어"
        else:
            optimal_range = "100-200단어"
        
        st.caption(f"📏 권장 길이: {optimal_range}")
    
    # 답변 유효성 검사
    answer_valid = answer and answer.strip() and len(answer.strip()) > 10
    
    # 답변 제출 버튼들
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("📤 답변 제출", disabled=not answer_valid, use_container_width=True, key=f"submit_{current_index}"):
            if not answer or not answer.strip():
                st.error("답변을 입력해주세요.")
                return
            
            # 로딩 상태 표시
            with st.spinner("📊 답변 분석 중..."):
                # 답변 저장 및 평가
                evaluation = st.session_state.interview_manager.save_answer(
                    current_session, current_question, answer, contexts
                )
            
            if 'error' not in evaluation:
                # 평가 결과를 세션에 저장 (다음 화면에서 표시하기 위해)
                st.session_state.last_evaluation = evaluation
                
                # 답변 기록 저장
                st.session_state.interview_answers.append({
                    'question': current_question,
                    'answer': answer,
                    'evaluation': evaluation,
                    'contexts': contexts
                })
                
                # 다음 질문을 위해 컨텍스트 캐시 삭제
                next_context_key = f"contexts_{current_session}_{current_index + 1}"
                if next_context_key in st.session_state:
                    del st.session_state[next_context_key]
                
                # 질문 인덱스 증가
                st.session_state.current_question_index += 1
                
                # 답변 성공 플래그 설정
                st.session_state.answer_submitted = True
                
                # 페이지 새로고침
                st.rerun()
            else:
                st.error(f"❌ 답변 저장 실패: {evaluation['error']}")
    
    with col2:
        if st.button("⏭️ 건너뛰기", use_container_width=True, key=f"skip_{current_index}"):
            # 컨텍스트 캐시 삭제
            next_context_key = f"contexts_{current_session}_{current_index + 1}"
            if next_context_key in st.session_state:
                del st.session_state[next_context_key]
            
            st.session_state.current_question_index += 1
            st.rerun()
    
    with col3:
        if st.button("❌ 면접 중단", use_container_width=True, key=f"abort_{current_index}"):
            if st.session_state.get('confirm_abort', False):
                # 세션 상태 초기화
                st.session_state.current_session_id = None
                st.session_state.current_question_index = 0
                st.session_state.interview_answers = []
                st.session_state.selected_questions = []
                st.session_state.confirm_abort = False
                
                st.warning("면접이 중단되었습니다.")
                st.rerun()
            else:
                st.session_state.confirm_abort = True
                st.warning("⚠️ 정말로 면접을 중단하시겠습니까? 다시 한 번 클릭하면 중단됩니다.")
    
    # 답변 제출 직후 피드백 표시
    if st.session_state.get('answer_submitted', False) and st.session_state.get('last_evaluation'):
        st.session_state.answer_submitted = False  # 플래그 리셋
        
        evaluation = st.session_state.last_evaluation
        
        # 성공 메시지
        st.success("✅ 답변이 저장되었습니다!")
        
        # 점수 표시
        score = evaluation['overall_score']
        if score >= 0.8:
            score_class = "score-good"
            score_emoji = "🎉"
        elif score >= 0.6:
            score_class = "score-average"
            score_emoji = "👍"
        else:
            score_class = "score-poor"
            score_emoji = "💪"
        
        st.markdown(f'<div class="{score_class}">종합 점수: {score:.2f}/1.00 {score_emoji}</div>', unsafe_allow_html=True)
        
        # 피드백 표시
        st.markdown("#### 📝 즉시 피드백")
        st.info(evaluation['feedback'])
        
        # 강점과 개선점
        col1, col2 = st.columns(2)
        
        with col1:
            if evaluation['strengths']:
                st.markdown("**✅ 강점**")
                for strength in evaluation['strengths']:
                    st.markdown(f"• {strength}")
        
        with col2:
            if evaluation['improvements']:
                st.markdown("**💡 개선점**")
                for improvement in evaluation['improvements']:
                    st.markdown(f"• {improvement}")
        
        # 상세 점수 차트
        with st.expander("📊 상세 점수 보기"):
            detailed_scores = evaluation['detailed_scores']
            
            score_data = {
                '평가 항목': ['키워드 매칭', '감정 분석', '일관성', '길이 적절성', '내용 관련성'],
                '점수': [
                    detailed_scores.get('keyword_match', 0),
                    detailed_scores.get('sentiment', 0),
                    detailed_scores.get('coherence', 0),
                    detailed_scores.get('length_appropriateness', 0),
                    detailed_scores.get('content_relevance', 0)
                ]
            }
            
            df_scores = pd.DataFrame(score_data)
            
            fig = px.bar(
                df_scores, 
                x='평가 항목', 
                y='점수',
                color='점수',
                color_continuous_scale='RdYlGn',
                range_color=[0, 1]
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # 다음 질문으로 이동 버튼
        if current_index + 1 < len(questions):
            if st.button("➡️ 다음 질문으로", use_container_width=True, key="next_question"):
                st.session_state.last_evaluation = None  # 평가 결과 초기화
                st.rerun()
        
        # 평가 결과 초기화 (3초 후 자동)
        if 'last_evaluation' in st.session_state:
            time.sleep(2)  # 2초 대기
            st.session_state.last_evaluation = None
            st.rerun()

def show_result_analysis_page():
    """결과 분석 페이지"""
    st.markdown('<h2 class="sub-header">📊 결과 분석</h2>', unsafe_allow_html=True)
    
    # 세션 선택
    sessions = st.session_state.interview_manager.get_all_sessions()
    
    if not sessions:
        st.info("📋 분석할 면접 기록이 없습니다.")
        return
    
    # 완료된 세션만 필터링
    completed_sessions = [s for s in sessions if s['status'] == 'completed']
    
    if not completed_sessions:
        st.info("📋 완료된 면접이 없습니다.")
        return
    
    # 세션 선택
    session_options = {
        f"{s['candidate_name']} - {s['position']} ({s['start_time'][:10]})": s['session_id'] 
        for s in completed_sessions
    }
    
    selected_session_label = st.selectbox(
        "분석할 면접 선택",
        list(session_options.keys())
    )
    
    selected_session_id = session_options[selected_session_label]
    
    # 분석 데이터 조회
    analytics = st.session_state.interview_manager.get_session_analytics(selected_session_id)
    
    if 'error' in analytics:
        st.error(f"❌ 분석 데이터 조회 실패: {analytics['error']}")
        return
    
    session_info = analytics['session_info']
    answers = analytics['answers']
    statistics = analytics['statistics']
    
    # 기본 정보 표시
    st.markdown("### 📋 면접 기본 정보")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("면접자", session_info['candidate_name'])
        st.metric("지원 직무", session_info['position'])
    
    with col2:
        st.metric("종합 점수", f"{session_info['overall_score']:.2f}")
        st.metric("답변 수", statistics['total_answers'])
    
    with col3:
        if session_info['end_time']:
            start_time = datetime.fromisoformat(session_info['start_time'])
            end_time = datetime.fromisoformat(session_info['end_time'])
            duration = end_time - start_time
            st.metric("면접 시간", f"{duration.seconds // 60}분")
        st.metric("평균 답변 길이", f"{statistics['average_answer_length']:.0f}단어")
    
    # 점수 분석 차트
    st.markdown("### 📈 점수 분석")
    
    if answers:
        # 질문별 점수 차트
        score_data = {
            'question': [f"Q{i+1}" for i in range(len(answers))],
            'overall_score': [a['evaluation_score'] for a in answers],
            'keyword_match': [a['keyword_match_score'] for a in answers],
            'sentiment': [a['sentiment_score'] for a in answers],
            'coherence': [a['coherence_score'] for a in answers]
        }
        
        df_scores = pd.DataFrame(score_data)
        
        # 종합 점수 차트
        fig1 = px.line(
            df_scores, 
            x='question', 
            y='overall_score',
            title='질문별 종합 점수',
            markers=True,
            range_y=[0, 1]
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # 상세 점수 비교 차트
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=df_scores['question'],
            y=df_scores['keyword_match'],
            mode='lines+markers',
            name='키워드 매칭',
            line=dict(color='blue')
        ))
        
        fig2.add_trace(go.Scatter(
            x=df_scores['question'],
            y=df_scores['sentiment'],
            mode='lines+markers',
            name='감정 분석',
            line=dict(color='green')
        ))
        
        fig2.add_trace(go.Scatter(
            x=df_scores['question'],
            y=df_scores['coherence'],
            mode='lines+markers',
            name='일관성',
            line=dict(color='red')
        ))
        
        fig2.update_layout(
            title='질문별 세부 점수 비교',
            xaxis_title='질문',
            yaxis_title='점수',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # 통계 요약
    st.markdown("### 📊 통계 요약")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 점수 통계")
        st.metric("평균 점수", f"{statistics['average_score']:.2f}")
        st.metric("점수 편차", f"{statistics['score_deviation']:.2f}")
        st.metric("평균 감정 점수", f"{statistics['average_sentiment']:.2f}")
    
    with col2:
        st.markdown("#### 답변 통계")
        st.metric("평균 일관성", f"{statistics['average_coherence']:.2f}")
        st.metric("평균 답변 길이", f"{statistics['average_answer_length']:.1f}단어")
        st.metric("총 답변 수", statistics['total_answers'])
    
    # 개별 답변 상세 분석
    st.markdown("### 📝 답변 상세 분석")
    
    for i, answer in enumerate(answers, 1):
        with st.expander(f"질문 {i}: {answer['question_text'][:50]}... (점수: {answer['evaluation_score']:.2f})"):
            
            # 질문과 답변
            st.markdown("**질문:**")
            st.write(answer['question_text'])
            
            st.markdown("**답변:**")
            st.write(answer['answer'])
            
            # 점수 상세
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("종합 점수", f"{answer['evaluation_score']:.2f}")
                st.metric("키워드 매칭", f"{answer['keyword_match_score']:.2f}")
            
            with col2:
                st.metric("감정 분석", f"{answer['sentiment_score']:.2f}")
                st.metric("일관성", f"{answer['coherence_score']:.2f}")
            
            with col3:
                st.metric("답변 길이", f"{answer['answer_length']}단어")
                st.metric("작성 시간", answer['timestamp'][:16])
            
            # 사용된 컨텍스트
            if answer['context_used']:
                st.markdown("**참고한 자료:**")
                for ctx in answer['context_used']:
                    st.write(f"- {ctx.get('title', 'Unknown')} (관련도: {ctx.get('score', 0):.3f})")
    
    # 종합 평가 및 권장사항
    st.markdown("### 💡 종합 평가 및 권장사항")
    
    overall_score = statistics['average_score']
    
    if overall_score >= 0.8:
        st.success("🎉 **우수한 면접 성과입니다!**")
        st.write("대부분의 질문에서 높은 점수를 기록했으며, 체계적이고 논리적인 답변을 보여주었습니다.")
    elif overall_score >= 0.6:
        st.info("👍 **양호한 면접 성과입니다.**")
        st.write("전반적으로 좋은 답변을 했지만, 몇 가지 영역에서 개선의 여지가 있습니다.")
    else:
        st.warning("💪 **개선이 필요한 면접 성과입니다.**")
        st.write("면접 기술과 답변 구성에 더 많은 연습이 필요합니다.")
    
    # 개선 권장사항
    st.markdown("#### 📈 개선 권장사항")
    
    recommendations = []
    
    if statistics['average_sentiment'] < 0.6:
        recommendations.append("더 긍정적이고 적극적인 표현을 사용해보세요.")
    
    if statistics['average_coherence'] < 0.6:
        recommendations.append("답변의 논리적 구조를 개선하고 연결어를 더 활용해보세요.")
    
    if statistics['average_answer_length'] < 80:
        recommendations.append("답변을 더 구체적이고 상세하게 작성해보세요.")
    elif statistics['average_answer_length'] > 200:
        recommendations.append("답변을 더 간결하고 핵심적으로 정리해보세요.")
    
    if statistics['score_deviation'] > 0.3:
        recommendations.append("모든 질문에 대해 일관된 수준의 답변을 유지해보세요.")
    
    if recommendations:
        for rec in recommendations:
            st.write(f"• {rec}")
    else:
        st.write("• 전반적으로 균형잡힌 좋은 면접 성과를 보여주었습니다.")

def initialize_session_state():
    """세션 상태 초기화"""
    try:
        if 'interview_manager' not in st.session_state:
            st.session_state.interview_manager = InterviewManager()
        
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
                
        if 'current_question_index' not in st.session_state:
            st.session_state.current_question_index = 0
        
        if 'interview_answers' not in st.session_state:
            st.session_state.interview_answers = []
        
        if 'selected_questions' not in st.session_state:
            st.session_state.selected_questions = []
    except Exception as e:
        st.error(f"세션 상태 초기화 실패: {e}")
        logger.error(f"Session state initialization error: {e}")

def main():
    """메인 애플리케이션"""
    # Header
    st.markdown('<h1 class="main-header">🛡️ LIGNEX1 RAG 기반 조직적합도 인성면접 시스템</h1>', unsafe_allow_html=True)

    # 세션 상태 초기화를 가장 먼저 실행
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📋 메뉴</h2>', unsafe_allow_html=True)
        
        page = st.selectbox(
            "페이지 선택",
            ["새 면접 시작", "면접 진행", "결과 분석", "면접 기록", "시스템 상태"]
        )

        # RAG system status
        st.markdown("---")
        st.markdown("### 🔧 RAG 시스템 상태")
        rag_available = st.session_state.interview_manager.rag_service.is_available()

        if rag_available:
            st.success("✅ RAG 시스템 연결됨")
            try:
                # 먼저 컬렉션 존재 여부 확인
                collections = st.session_state.interview_manager.rag_service.qdrant_client.get_collections().collections
                collection_name = st.session_state.interview_manager.rag_service.collection_name
                collection_exists = any(c.name == collection_name for c in collections)
                
                if collection_exists:
                    try:
                        # 컬렉션 기본 정보만 조회 (버전 호환성 문제 회피)
                        collection_info = st.session_state.interview_manager.rag_service.qdrant_client.get_collection(collection_name)
                        
                        # 안전하게 정보 추출
                        if hasattr(collection_info, 'vectors_count'):
                            vector_count = collection_info.vectors_count
                        else:
                            # 대체 방법: 점검 API 사용
                            vector_count = "알 수 없음"
                        
                        st.info(f"📊 컬렉션: {collection_name}")
                        st.info(f"📊 벡터 데이터: {vector_count}")
                        st.success("✅ 컬렉션 정상 작동")
                        
                    except Exception as detail_error:
                        # 상세 정보 조회 실패시 기본 정보만 표시
                        st.success(f"✅ 컬렉션 '{collection_name}' 존재")
                        st.info("📊 상세 정보 조회 불가 (버전 호환성 문제)")
                        logger.warning(f"Collection detail query failed: {detail_error}")
                else:
                    st.warning(f"⚠️ 컬렉션 '{collection_name}'이 존재하지 않습니다")
                    st.info("💡 RAG 데이터를 먼저 인덱싱해주세요")

            except Exception as e:
                st.error(f"❌ 컬렉션 정보 조회 실패: {str(e)}")
                logger.error(f"Collection info query failed: {e}")
        else:
            st.error("❌ RAG 시스템 연결 안됨")
            st.info("💡 Qdrant가 실행 중인지 확인하세요")
        

    # 자동 네비게이션 처리
    if st.session_state.get('auto_navigate_to_interview', False):
        st.session_state.auto_navigate_to_interview = False
        page = "면접 진행"

    # Page routing
    if page == "새 면접 시작":
        show_new_interview_page()
    elif page == "면접 진행":
        show_interview_progress_page()

    elif page == "결과 분석":
        show_result_analysis_page()
    elif page == "면접 기록":
        show_interview_records_page()
    elif page == "시스템 상태":
        show_system_status_page()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            🛡️ LIGNEX1 RAG 기반 면접 시스템 v1.0<br>
            Powered by Streamlit, Qdrant, and Sentence Transformers
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"애플리케이션 실행 중 오류가 발생했습니다: {e}")
        logger.error(f"Application error: {e}")
        if st.button("🔄 애플리케이션 재시작"):
            st.rerun()