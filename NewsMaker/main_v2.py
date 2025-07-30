import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import chromadb
from chromadb.config import Settings
import anthropic
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import requests
import urllib.parse
import time
from bs4 import BeautifulSoup
import re
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """네이버 뉴스 기사 구조"""
    title: str
    link: str
    description: str
    pub_date: str
    content: str = ""
    
@dataclass
class NewsMetadata:
    """뉴스 메타데이터 구조"""
    relevance_score: int
    topics: List[str]
    keywords: List[str]
    summary: str
    sentiment: str
    importance: int
    company_mentions: List[str]
    date: str
    source: str

@dataclass
class NewsChunk:
    """뉴스 청크 구조"""
    chunk_id: int
    content: str
    topics: List[str]
    keywords: List[str]
    chunk_type: str

class EnhancedNaverNewsAPI:
    """향상된 네이버 뉴스 검색 API 클라이언트"""
    
    def __init__(self, client_id: str, client_secret: str):
        logger.debug(f"Initializing EnhancedNaverNewsAPI with client_id={client_id}")
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/search/news.json"
        
        if not client_id or client_id == "YOUR_NAVER_CLIENT_ID":
            logger.warning("네이버 API 키가 설정되지 않았습니다. 테스트 모드로 실행됩니다.")
            self.test_mode = True
        else:
            self.test_mode = False
        logger.debug(f"EnhancedNaverNewsAPI initialized. test_mode={self.test_mode}")
    
    def search_news_with_keywords(self, company_name: str, additional_keywords: List[str] = None, 
                                display: int = 10, start: int = 1, sort: str = "date") -> List[NewsArticle]:
        logger.debug(f"search_news_with_keywords called with company_name={company_name}, additional_keywords={additional_keywords}, display={display}, start={start}, sort={sort}")
        all_articles = []
        
        # 기본 검색 쿼리들 생성
        search_queries = [company_name]
        
        # 추가 키워드와 조합
        if additional_keywords:
            for keyword in additional_keywords:
                search_queries.append(f"{company_name} {keyword}")
        
        # 기본 조합 쿼리 추가
        search_queries.extend([
            f"{company_name} 신제품",
            f"{company_name} 발표",
            f"{company_name} 기술",
            f"{company_name} 실적"
        ])
        
        # 중복 제거
        search_queries = list(set(search_queries))
        logger.debug(f"search_queries after deduplication: {search_queries}")
        
        for query in search_queries[:5]:  # 최대 5개 쿼리만 실행
            try:
                logger.debug(f"Searching news for query: {query}")
                articles = self.search_news(query, display=min(display, 10), start=start, sort=sort)
                all_articles.extend(articles)
                logger.debug(f"Found {len(articles)} articles for query: {query}")
                # API 호출 제한
                time.sleep(1)
            except Exception as e:
                logger.error(f"검색 쿼리 '{query}' 실행 실패: {e}")
        
        # 중복 기사 제거 (제목 기준)
        unique_articles = []
        seen_titles = set()
        
        for article in all_articles:
            if article.title not in seen_titles:
                unique_articles.append(article)
                seen_titles.add(article.title)
        
        logger.info(f"중복 제거 후 unique 기사: {len(unique_articles)}개")
        logger.debug(f"Returning {len(unique_articles[:display])} unique articles")
        return unique_articles[:display]  # 요청된 개수만큼 반환
    
    def search_news(self, query: str, display: int = 10, start: int = 1, 
                   sort: str = "date") -> List[NewsArticle]:
        logger.debug(f"search_news called with query={query}, display={display}, start={start}, sort={sort}")
        if self.test_mode:
            logger.debug("Test mode active, returning dummy news.")
            return self._get_dummy_news(query, display)
        
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"{self.base_url}?query={encoded_query}&display={display}&start={start}&sort={sort}"
            logger.debug(f"Requesting URL: {url}")
            headers = {
                "X-Naver-Client-Id": self.client_id,
                "X-Naver-Client-Secret": self.client_secret
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = []
            
            for item in data.get('items', []):
                article = NewsArticle(
                    title=self._clean_html(item['title']),
                    link=item['link'],
                    description=self._clean_html(item['description']),
                    pub_date=item['pubDate']
                )
                
                # 기사 본문 크롤링
                article.content = self._fetch_article_content(article.link)
                articles.append(article)
                logger.debug(f"Appended article: {article.title}")
                
                # API 호출 제한을 위한 딜레이
                time.sleep(0.1)
            
            logger.info(f"뉴스 검색 완료: {len(articles)}개 기사")
            return articles
            
        except Exception as e:
            logger.error(f"네이버 뉴스 검색 실패: {e}")
            return self._get_dummy_news(query, display)
    
    def _clean_html(self, text: str) -> str:
        """HTML 태그 제거"""
        return re.sub(r'<[^>]+>', '', text)
    
    def _fetch_article_content(self, url: str) -> str:
        """기사 본문 크롤링 (개선됨)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 네이버 뉴스 본문 선택자들 (확장됨)
            selectors = [
                '#dic_area',  # 네이버 뉴스 본문
                '#articleBodyContents',
                '.news_text',
                '.article_body',
                '.view_text',
                '.news_view',
                '#newsContentDiv',
                '.article-view-content'
            ]
            
            content = ""
            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    # 광고, 스크립트 등 불필요한 요소 제거
                    for unwanted in element.find_all(['script', 'style', 'ins', 'iframe']):
                        unwanted.decompose()
                    
                    content = element.get_text(separator='\n').strip()
                    break
            
            # 본문이 없으면 전체 텍스트에서 추출
            if not content or len(content) < 100:
                content = soup.get_text(separator='\n')
                # 불필요한 줄 제거
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                content = '\n'.join(lines)
            
            return content[:3000]  # 길이 제한 증가
            
        except Exception as e:
            logger.warning(f"본문 크롤링 실패 ({url}): {e}")
            return ""
    
    def _get_dummy_news(self, query: str, display: int = 10) -> List[NewsArticle]:
        """테스트용 더미 뉴스 (회사명별 동적 생성)"""
        dummy_articles = []
        
        # 쿼리에서 회사명 추출
        company_name = query.split()[0] if query else "테스트회사"
        
        base_articles = [
            {
                "title_template": "{company} 관련 최신 기술 동향 발표",
                "description_template": "{company}의 새로운 혁신이 업계에 주목받고 있습니다.",
                "content_template": "{company}가 새로운 기술 혁신을 통해 시장에서 주목받고 있습니다. 전문가들은 이번 발표가 업계 전반에 큰 변화를 가져올 것으로 예상한다고 밝혔습니다. 특히 성능 향상과 안정성 개선에 중점을 둔 이번 업데이트는 고객들로부터 긍정적인 반응을 얻고 있습니다."
            },
            {
                "title_template": "{company} 시장 점유율 확대 소식",
                "description_template": "{company}의 시장 영향력이 계속 확대되고 있습니다.",
                "content_template": "{company}의 시장 점유율이 지속적으로 확대되고 있어 업계의 관심이 집중되고 있습니다. 분석가들은 향후 성장 전망을 긍정적으로 평가하고 있으며, 신제품 출시와 함께 더욱 강력한 시장 지배력을 보일 것으로 전망됩니다."
            },
            {
                "title_template": "{company} 신규 파트너십 체결",
                "description_template": "{company}가 글로벌 기업과의 전략적 파트너십을 발표했습니다.",
                "content_template": "{company}가 글로벌 선도 기업과의 전략적 파트너십을 체결했다고 발표했습니다. 이번 협업을 통해 기술력 향상과 시장 확대에 탄력을 받을 것으로 기대되며, 양사는 상호 보완적인 기술과 경험을 바탕으로 시너지 효과를 창출할 계획입니다."
            }
        ]
        
        for i in range(min(display, len(base_articles) * 2)):
            template = base_articles[i % len(base_articles)]
            
            dummy_articles.append(NewsArticle(
                title=template["title_template"].format(company=company_name),
                link=f"http://test.com/news{i+1}",
                description=template["description_template"].format(company=company_name),
                pub_date=f"Mon, {7 + i} Jul 2025 {10 + i}:00:00 +0900",
                content=template["content_template"].format(company=company_name)
            ))
        
        return dummy_articles

class EnhancedPromptManager:
    """향상된 프롬프트 관리 클래스"""
    
    @staticmethod
    def get_news_analysis_prompt(news_content: str, company_name: str) -> str:
        return f"""당신은 뉴스 분석 전문가입니다. 다음 뉴스 기사를 분석하여 구조화된 정보를 추출해주세요.

**분석할 뉴스 기사:**
{news_content}

**대상 회사:** {company_name}

**요구사항:**
1. 이 뉴스가 대상 회사와 관련이 있는지 확인하고 관련도를 1-10점으로 평가
2. 뉴스의 주요 토픽 분류 (최대 3개)
3. 핵심 키워드 추출 (최대 10개)
4. 뉴스 요약 (2-3문장)
5. 감정 분석 (긍정/부정/중립)
6. 중요도 평가 (1-10점)

**출력 형식:**
```json
{{
  "relevance_score": 0,
  "topics": ["topic1", "topic2", "topic3"],
  "keywords": ["keyword1", "keyword2", ...],
  "summary": "뉴스 요약 내용",
  "sentiment": "긍정/부정/중립",
  "importance": 0,
  "company_mentions": ["회사명1", "회사명2"],
  "date": "YYYY-MM-DD",
  "source": "뉴스 출처"
}}
```

관련도가 5점 이상인 경우만 벡터 DB에 저장하세요."""

    @staticmethod
    def get_news_chunking_prompt(news_content: str) -> str:
        return f"""다음 뉴스 기사를 의미 있는 청크로 분할해주세요.

**원본 뉴스:**
{news_content}

**청킹 규칙:**
1. 각 청크는 200-400자 사이
2. 문장의 완전성 유지
3. 주제별로 논리적 분할
4. 각 청크는 독립적으로 이해 가능해야 함

**출력 형식:**
```json
{{
  "chunks": [
    {{
      "chunk_id": 1,
      "content": "청크 내용",
      "topics": ["관련 토픽"],
      "keywords": ["관련 키워드"],
      "chunk_type": "제목/본문/인용/통계"
    }}
  ]
}}
```"""

    @staticmethod
    def get_enhanced_news_generation_prompt(topic: str, keywords: List[str], 
                                        user_facts: str, reference_materials: str,
                                        length_specification: str = "") -> str:
        keywords_str = ", ".join(keywords)
        
        # Parse length specification properly
        length_instruction = ""
        if "줄 수" in length_specification:
            lines_match = re.search(r'(\d+)', length_specification)
            if lines_match:
                target_lines = int(lines_match.group(1))
                length_instruction = f"""
**중요: 정확히 {target_lines}줄 분량의 뉴스를 작성하세요.**
- 제목: 1줄
- 리드: 3-4줄  
- 본문: {target_lines-8}줄 (여러 단락으로 구성)
- 결론: 2-3줄
총 {target_lines}줄을 맞춰주세요."""
        elif "단어 수" in length_specification:
            words_match = re.search(r'(\d+)', length_specification)
            if words_match:
                target_words = int(words_match.group(1))
                length_instruction = f"""
**중요: 정확히 {target_words}단어 분량의 뉴스를 작성하세요.**
각 단락을 충분히 상세하게 작성하여 {target_words}단어를 맞춰주세요."""
    
        return f"""당신은 한국의 경제 전문 기자입니다. 다음 조건에 맞춰 전문적이고 상세한 뉴스 기사를 작성해주세요.

**입력 정보:**
- 토픽: {topic}
- 핵심 키워드: {keywords_str}
- 주요 사실 정보: {user_facts}

{length_instruction}

**참고 자료 (RAG):**
{reference_materials}

**전문 뉴스 작성 원칙:**
1. **사실 정보 완전 반영**: 제공된 주요 사실을 모두 포함하여 상세히 서술
2. **키워드 활용**: 제공된 키워드들을 자연스럽게 포함
3. **전문적 구성**: 경제지 수준의 체계적이고 논리적인 구조
4. **상세한 설명**: 각 사실에 대해 배경, 의미, 영향을 구체적으로 분석
5. **인용 활용**: 관계자 발언이나 전문가 의견을 자연스럽게 포함
6. **시장 분석**: 업계 영향과 향후 전망을 전문적으로 분석

**필수 구조 (상세 작성):**

**제목:** [임팩트 있는 핵심 내용, 30-40자]

**리드 (Lead):** 
핵심 내용을 3-4문장으로 요약하되, 5W1H를 모두 포함하여 상세히 작성

**본문 1단락 - 핵심 발표 내용:**
주요 사실 정보를 바탕으로 무엇이 발표되었는지 구체적으로 설명

**본문 2단락 - 기술적/사업적 세부사항:**
제품이나 서비스의 구체적인 특징, 기술적 혁신점, 성능 지표 등을 상세히 서술

**본문 3단락 - 관계자 발언:**
경영진이나 관련 담당자의 발언을 인용하여 의도와 목표를 설명

**본문 4단락 - 시장 분석:**
업계 현황, 경쟁사 대비 위치, 시장에서의 의미를 전문적으로 분석

**본문 5단락 - 고객/파트너 반응:**
고객사, 파트너사의 반응이나 업계의 평가를 포함

**결론 - 향후 전망:**
향후 계획, 시장 전망, 기대 효과를 종합적으로 정리

**작성 시 주의사항:**
- 각 단락마다 충분한 내용으로 구성 (최소 3-4문장)
- 구체적인 수치나 데이터 활용
- 전문적이고 객관적인 어조 유지
- 참고 자료의 정보를 자연스럽게 통합
- 추상적 표현보다는 구체적이고 명확한 서술
- **반드시 제공된 키워드들을 자연스럽게 포함시키세요**

**반드시 지정된 길이를 정확히 맞춰서 작성하세요.**"""
                
    @staticmethod
    def get_quality_check_prompt(news_content: str) -> str:
        return f"""작성된 뉴스 기사의 품질을 전문적으로 평가해주세요.

**뉴스 기사:**
{news_content}

**평가 기준:**
1. 사실성 (Facts): 정확한 정보 포함 여부 (1-10점)
2. 완성도 (Completeness): 5W1H 충족 및 구조 완성도 (1-10점)
3. 객관성 (Objectivity): 편향성 없는 균형잡힌 보도 (1-10점)
4. 가독성 (Readability): 이해하기 쉬운 구조와 문체 (1-10점)
5. 신뢰성 (Credibility): 출처 명확성과 전문성 (1-10점)

**평가 결과:**
```json
{{
  "overall_score": 0,
  "detailed_scores": {{
    "facts": 0,
    "completeness": 0,
    "objectivity": 0,
    "readability": 0,
    "credibility": 0
  }},
  "improvements": ["개선사항 목록"],
  "strengths": ["강점 목록"],
  "approval": true
}}
```

전체 점수 7점 이상일 때 최종 승인하세요."""


# AFTER: Smart handling of embedding function conflicts

import chromadb
from chromadb.utils import embedding_functions
import hashlib
import math

class EnhancedChromaDBManager:
    """향상된 ChromaDB 관리 클래스 (임베딩 충돌 해결)"""
    

    def __init__(self, db_path: str = "./chroma_db"):
        logger.debug(f"Initializing EnhancedChromaDBManager with db_path={db_path}")
        
        # Always create the embedding function instance first
        self.embedding_function = self._create_768_embedding_function()
            
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            
            collection_name = "enhanced_news_collection"
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if collection_name in existing_collections:
                # Collection exists - use it without changing embedding function
                logger.info(f"기존 컬렉션 사용: {collection_name}")
                self.collection = self.client.get_collection(collection_name)
                
                # Check if we need to handle dimension mismatches later
                self._check_embedding_compatibility()
                
            else:
                # New collection - create with custom embedding function
                logger.info(f"새 컬렉션 생성: {collection_name}")
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Enhanced AI News Writer 뉴스 컬렉션"},
                    embedding_function=self.embedding_function
                )
            
            logger.info(f"ChromaDB 초기화 완료: {db_path}")
            logger.debug(f"ChromaDBManager initialized. Collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패: {e}")
            # FALLBACK: Create collection without custom embedding function
            try:
                self.collection = self.client.get_or_create_collection(
                    name="enhanced_news_collection",
                    metadata={"description": "Enhanced AI News Writer 뉴스 컬렉션"}
                    # No embedding_function specified - use default
                )
                self.embedding_function = None
                logger.warning("기본 임베딩 함수로 폴백하여 초기화 완료")
            except Exception as fallback_error:
                logger.error(f"폴백 초기화도 실패: {fallback_error}")
                raise fallback_error
            
    def _check_embedding_compatibility(self):
        """기존 컬렉션의 임베딩 호환성 확인 (✅ NEW METHOD)"""
        try:
            # 컬렉션에 데이터가 있는지 확인
            count = self.collection.count()
            if count > 0:
                # 샘플 검색으로 임베딩 차원 확인
                try:
                    # 더미 검색으로 차원 확인
                    dummy_embedding = [0.1] * 768
                    self.collection.query(
                        query_embeddings=[dummy_embedding],
                        n_results=1,
                        include=['documents']
                    )
                    logger.info("기존 컬렉션: 768차원 임베딩 호환")
                    self._embedding_dimension = 768
                except Exception as e:
                    if "384" in str(e):
                        logger.warning("기존 컬렉션: 384차원 임베딩 감지")
                        self._embedding_dimension = 384
                    else:
                        logger.warning(f"임베딩 차원 확인 실패: {e}")
                        self._embedding_dimension = 768  # 기본값
            else:
                self._embedding_dimension = 768  # 빈 컬렉션은 768로 가정
                
        except Exception as e:
            logger.error(f"임베딩 호환성 확인 실패: {e}")
            self._embedding_dimension = 768
    
    def _create_768_embedding_function(self):
        """768차원 임베딩 함수 생성"""
        class Custom768EmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __call__(self, input_texts):
                """텍스트를 768차원 임베딩으로 변환"""
                embeddings = []
                for text in input_texts:
                    embedding = self._text_to_768_embedding(text)
                    embeddings.append(embedding)
                return embeddings
            
            def _text_to_768_embedding(self, text: str):
                """텍스트를 768차원 벡터로 변환"""
                # 텍스트의 해시값을 기반으로 768차원 벡터 생성
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                
                # 해시를 숫자로 변환하여 768개 값 생성
                embedding = []
                for i in range(768):
                    # 해시의 각 부분을 사용하여 -1~1 범위의 값 생성
                    hash_part = text_hash[(i * 2) % len(text_hash):((i * 2) + 2) % len(text_hash)]
                    if len(hash_part) < 2:
                        hash_part = text_hash[:2]
                    
                    # 16진수를 부동소수점으로 변환 (-1~1 범위)
                    value = (int(hash_part, 16) / 255.0) * 2 - 1
                    embedding.append(value)
                
                # 정규화
                norm = math.sqrt(sum(x * x for x in embedding))
                if norm > 0:
                    embedding = [x / norm for x in embedding]
                
                return embedding
        
        return Custom768EmbeddingFunction()
    
    def store_news_chunk(self, chunk, metadata, embedding):
        logger.debug(f"store_news_chunk called for chunk_id={chunk.chunk_id}")
        try:
            # ✅ FIXED: Adjust embedding dimension based on collection
            target_dim = getattr(self, '_embedding_dimension', 768)
            
            if len(embedding) != target_dim:
                if len(embedding) < target_dim:
                    embedding = embedding + [0.0] * (target_dim - len(embedding))
                else:
                    embedding = embedding[:target_dim]
            
            # 고유 ID 생성
            import time
            content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
            unique_id = f"chunk_{content_hash}_{chunk.chunk_id}_{int(time.time())}"
            
            # 메타데이터 확장
            chunk_metadata = {
                "topics": json.dumps(chunk.topics, ensure_ascii=False),
                "keywords": json.dumps(chunk.keywords, ensure_ascii=False),
                "chunk_type": chunk.chunk_type,
                "chunk_id": chunk.chunk_id,
                "sentiment": metadata.sentiment,
                "importance": metadata.importance,
                "relevance_score": metadata.relevance_score,
                "company_mentions": json.dumps(metadata.company_mentions, ensure_ascii=False),
                "date": metadata.date,
                "source": metadata.source,
                "summary": metadata.summary,
                "created_at": datetime.now().isoformat(),
                "embedding_dim": len(embedding)
            }
            
            self.collection.add(
                documents=[chunk.content],
                metadatas=[chunk_metadata],
                embeddings=[embedding],
                ids=[unique_id]
            )
            
            logger.info(f"청크 저장 완료: {chunk.chunk_id} (ID: {unique_id[:20]}..., 임베딩: {len(embedding)}차원)")
            
        except Exception as e:
            logger.error(f"청크 저장 실패: {e}")
            raise
    
    def search_relevant_news(self, query: str, n_results: int = 10, 
                             min_relevance: int = 5, target_companies: List[str] = None):
        """관련 뉴스 검색 (회사 필터링 강화)"""
        try:
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.warning("컬렉션이 비어있습니다.")
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            
            actual_n_results = min(n_results, collection_count)
            target_dim = getattr(self, '_embedding_dimension', 768)
            
            # ✅ METHOD 1: 임베딩 기반 검색 (수정됨)
            try:
                logger.info(f"임베딩 기반 검색 시도: '{query[:50]}...'")

                # Manually create the embedding to ensure correct dimension
                query_embedding = self.embedding_function([query])[0]

                results = self.collection.query(
                    query_embeddings=[query_embedding], # Use query_embeddings
                    n_results=actual_n_results * 2,  # 필터링을 위해 더 많이 가져옴
                    include=['documents', 'metadatas', 'distances']
                )
                
                if results and results.get('documents') and results['documents'][0]:
                    # 회사 필터링 적용
                    filtered_results = self._filter_by_companies_and_relevance(
                        results, target_companies, min_relevance, actual_n_results
                    )
                    
                    if filtered_results['documents'][0]:
                        logger.info(f"✅ 임베딩 검색 성공: {len(filtered_results['documents'][0])}개 결과")
                        return filtered_results
            
            except Exception as e:
                logger.warning(f"임베딩 검색 실패: {e}")
            
            # ✅ METHOD 2: 더미 임베딩으로 검색
            try:
                logger.info(f"더미 임베딩 검색 시도... ({target_dim}차원)")
                dummy_embedding = [0.1] * target_dim
                
                results = self.collection.query(
                    query_embeddings=[dummy_embedding],
                    n_results=actual_n_results * 2,
                    include=['documents', 'metadatas', 'distances']
                )
                
                if results and results.get('documents') and results['documents'][0]:
                    filtered_results = self._filter_by_companies_and_relevance(
                        results, target_companies, min_relevance, actual_n_results
                    )
                    
                    if filtered_results['documents'][0]:
                        logger.info(f"✅ 더미 임베딩 검색 성공: {len(filtered_results['documents'][0])}개 결과")
                        return filtered_results
            
            except Exception as e:
                logger.warning(f"더미 임베딩 검색 실패: {e}")
            
            # METHOD 3: 키워드 기반 검색 (폴백)
            try:
                logger.info("키워드 기반 검색 시도...")
                return self._keyword_based_search(query, actual_n_results, min_relevance, target_companies)
                
            except Exception as e:
                logger.warning(f"키워드 검색 실패: {e}")
            
            # METHOD 4: 최후 수단
            logger.info("전체 데이터 검색 (최후 수단)")
            return self._get_all_data(actual_n_results, min_relevance, target_companies)
                    
        except Exception as e:
            logger.error(f"검색 완전 실패: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
     
    def _filter_by_companies_and_relevance(self, results, target_companies: List[str], 
                                         min_relevance: int, max_results: int):
        """회사명과 관련도로 필터링"""
        if not results.get('metadatas') or not results['metadatas'][0]:
            return results
        
        filtered_results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        
        for i, metadata in enumerate(results['metadatas'][0]):
            # 관련도 체크
            relevance = metadata.get('relevance_score', 10)
            if relevance < min_relevance:
                continue
            
            # 회사명 필터링 (target_companies가 지정된 경우만)
            if target_companies:
                company_mentions_str = metadata.get('company_mentions', '[]')
                try:
                    company_mentions = json.loads(company_mentions_str)
                except:
                    company_mentions = []
                
                # 문서 내용에서도 회사명 검색
                doc_content = results['documents'][0][i].lower()
                
                # 대상 회사 중 하나라도 언급되어야 함
                found_company = False
                for target_company in target_companies:
                    target_lower = target_company.lower()
                    if (target_lower in [c.lower() for c in company_mentions] or 
                        target_lower in doc_content):
                        found_company = True
                        break
                
                if not found_company:
                    continue
            
            # 필터 통과한 항목 추가
            filtered_results['documents'][0].append(results['documents'][0][i])
            filtered_results['metadatas'][0].append(metadata)
            if results.get('distances') and results['distances'][0]:
                filtered_results['distances'][0].append(results['distances'][0][i])
            
            # 최대 결과 수 도달 시 중단
            if len(filtered_results['documents'][0]) >= max_results:
                break
        
        logger.info(f"회사/관련도 필터링 완료: {len(filtered_results['documents'][0])}개 결과")
        return filtered_results
    
    def _keyword_based_search(self, query: str, n_results: int, min_relevance: int, 
                            target_companies: List[str] = None):
        """키워드 기반 검색 (폴백 방법)"""
        try:
            all_data = self.collection.get(include=['documents', 'metadatas'])
            
            if not all_data.get('documents'):
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            
            query_keywords = query.lower().split()
            scored_results = []
            
            for i, doc in enumerate(all_data['documents']):
                metadata = all_data['metadatas'][i] if i < len(all_data['metadatas']) else {}
                
                # 관련도 체크
                relevance = metadata.get('relevance_score', 5)
                if relevance < min_relevance:
                    continue
                
                # 회사명 필터링
                if target_companies:
                    company_mentions_str = metadata.get('company_mentions', '[]')
                    try:
                        company_mentions = json.loads(company_mentions_str)
                    except:
                        company_mentions = []
                    
                    doc_content = doc.lower()
                    found_company = False
                    for target_company in target_companies:
                        target_lower = target_company.lower()
                        if (target_lower in [c.lower() for c in company_mentions] or 
                            target_lower in doc_content):
                            found_company = True
                            break
                    
                    if not found_company:
                        continue
                
                # 키워드 점수 계산
                doc_text = doc.lower()
                score = sum(1 for keyword in query_keywords if keyword in doc_text)
                
                if score > 0:
                    scored_results.append((score, doc, metadata, 1.0 - (score / len(query_keywords))))
            
            scored_results.sort(key=lambda x: x[0], reverse=True)
            top_results = scored_results[:n_results]
            
            return {
                'documents': [[item[1] for item in top_results]],
                'metadatas': [[item[2] for item in top_results]],
                'distances': [[item[3] for item in top_results]]
            }
            
        except Exception as e:
            logger.error(f"키워드 검색 실패: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    def _get_all_data(self, n_results: int, min_relevance: int, target_companies: List[str] = None):
        """모든 데이터 가져오기 (최후 수단)"""
        try:
            all_data = self.collection.get(include=['documents', 'metadatas'])
            
            if not all_data.get('documents'):
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            
            filtered_docs = []
            filtered_metas = []
            
            for i, doc in enumerate(all_data['documents']):
                metadata = all_data['metadatas'][i] if i < len(all_data['metadatas']) else {}
                
                # 관련도 체크
                relevance = metadata.get('relevance_score', 5)
                if relevance < min_relevance:
                    continue
                
                # 회사명 필터링
                if target_companies:
                    company_mentions_str = metadata.get('company_mentions', '[]')
                    try:
                        company_mentions = json.loads(company_mentions_str)
                    except:
                        company_mentions = []
                    
                    doc_content = doc.lower()
                    found_company = False
                    for target_company in target_companies:
                        target_lower = target_company.lower()
                        if (target_lower in [c.lower() for c in company_mentions] or 
                            target_lower in doc_content):
                            found_company = True
                            break
                    
                    if not found_company:
                        continue
                
                filtered_docs.append(doc)
                filtered_metas.append(metadata)
            
            # 제한된 결과만 반환
            filtered_docs = filtered_docs[:n_results]
            filtered_metas = filtered_metas[:n_results]
            
            logger.info(f"전체 데이터 검색: {len(filtered_docs)}개 결과")
            
            return {
                'documents': [filtered_docs],
                'metadatas': [filtered_metas], 
                'distances': [[0.5] * len(filtered_docs)]
            }
            
        except Exception as e:
            logger.error(f"전체 데이터 가져오기 실패: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    def get_collection_stats(self):
        """컬렉션 통계 조회"""
        try:
            count = self.collection.count()
            embedding_dim = getattr(self, '_embedding_dimension', 'unknown')
            return {
                "total_chunks": count,
                "collection_name": self.collection.name,
                "embedding_dimension": embedding_dim
            }
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {"total_chunks": 0, "collection_name": "unknown", "embedding_dimension": "error"}
    
    def safe_reset_collection(self):
        """안전한 컬렉션 재설정"""
        try:
            # 1. 기존 데이터 백업
            logger.info("기존 데이터 백업 중...")
            old_data = self.collection.get(include=['documents', 'metadatas'])
            
            # 2. 컬렉션 삭제
            logger.info("기존 컬렉션 삭제 중...")
            self.client.delete_collection("enhanced_news_collection")
            
            # 3. 새 컬렉션 생성 (768차원 임베딩 함수 포함)
            logger.info("새 컬렉션 생성 중...")
            self.embedding_function = self._create_768_embedding_function()
            self.collection = self.client.create_collection(
                name="enhanced_news_collection",
                metadata={"description": "Enhanced AI News Writer 뉴스 컬렉션 (768차원)"},
                embedding_function=self.embedding_function
            )
            self._embedding_dimension = 768
            
            # 4. 데이터 복원
            if old_data.get('documents'):
                logger.info(f"데이터 복원 중... ({len(old_data['documents'])}개 항목)")
                for i, doc in enumerate(old_data['documents']):
                    metadata = old_data['metadatas'][i] if i < len(old_data['metadatas']) else {}
                    
                    # 새로운 768차원 임베딩 생성
                    new_embedding = self.embedding_function([doc])[0]
                    
                    self.collection.add(
                        documents=[doc],
                        metadatas=[metadata],
                        embeddings=[new_embedding],
                        ids=[f"restored_{i}_{int(time.time())}"]
                    )
            
            logger.info("컬렉션이 768차원 임베딩으로 안전하게 재설정되었습니다.")
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 재설정 실패: {e}")
            return False

class EnhancedClaudeClient:
    """향상된 Claude API 클라이언트"""
    
    def __init__(self, api_key: str = None):
        logger.debug(f"Initializing EnhancedClaudeClient with api_key={'set' if api_key else 'not set'}")
        self.api_key = api_key
        if api_key and api_key != "YOUR_CLAUDE_API_KEY":
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None
            logger.warning("Claude API 키가 설정되지 않았습니다. 테스트 모드로 실행됩니다.")
        
        # API 호출 제한을 위한 설정
        self.last_call_time = 0
        self.min_interval = 2  # 최소 2초 간격
        self.request_count = 0
        self.max_requests_per_hour = 50  # 시간당 최대 요청 수
        logger.debug("EnhancedClaudeClient initialized.")
    
    async def generate_response(self, prompt: str, max_tokens: int = 4000) -> dict:
        """Claude API 호출 및 분석 정보 반환"""
        import time
        logger.debug(f"generate_response called with prompt length={len(prompt)} and max_tokens={max_tokens}")
        # API 호출 간격 제한
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            logger.info(f"API 호출 제한을 위해 {sleep_time:.1f}초 대기중...")
            await asyncio.sleep(sleep_time)
        self.last_call_time = time.time()
        self.request_count += 1
        model_name = "claude-3-5-sonnet-20241022"
        start_time = time.time()
        if not self.client:
            logger.debug("No Claude client, returning dummy response.")
            response_text = self._get_enhanced_dummy_response(prompt)
            elapsed = time.time() - start_time
            return {
                "elapsed": elapsed,
                "model": "dummy",
                "prompt": prompt,
                "response": response_text
            }
        try:
            # 재시도 로직 강화
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Claude API call attempt {attempt+1}")
                    response = self.client.messages.create(
                        model=model_name,
                        max_tokens=max_tokens,
                        temperature=0.1,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    logger.info(f"Claude API 호출 성공 (요청 #{self.request_count})")
                    elapsed = time.time() - start_time
                    return {
                        "elapsed": elapsed,
                        "model": model_name,
                        "prompt": prompt,
                        "response": response.content[0].text
                    }
                except Exception as e:
                    error_msg = str(e).lower()
                    if "529" in error_msg or "overloaded" in error_msg or "rate limit" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 10  # 10, 20, 30초 대기
                            logger.warning(f"Claude API 과부하/제한 (시도 {attempt + 1}/{max_retries}). {wait_time}초 후 재시도...")
                            await asyncio.sleep(wait_time)
                            continue
                    raise e
        except Exception as e:
            logger.error(f"Claude API 호출 실패: {e}")
            elapsed = time.time() - start_time
            return {
                "elapsed": elapsed,
                "model": model_name,
                "prompt": prompt,
                "response": self._get_enhanced_dummy_response(prompt)
            }

    def _get_enhanced_dummy_response(self, prompt: str) -> str:
        """향상된 테스트용 더미 응답 생성 (동적 회사명 지원)"""
        
        # 프롬프트에서 회사명 추출
        company_name = "테스트회사"
        keywords = []
        
        # 키워드 추출
        if "키워드:" in prompt or "핵심 키워드:" in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if "키워드:" in line or "핵심 키워드:" in line:
                    keyword_part = line.split(':')[1].strip()
                    keywords = [k.strip() for k in keyword_part.split(',')]
                    if keywords:
                        company_name = keywords[0]  # 첫 번째 키워드를 회사명으로
                    break
        
        # 토픽에서 회사명 추출
        if "토픽:" in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if "토픽:" in line:
                    topic_part = line.split(':')[1].strip()
                    # 토픽에서 회사명 추출 시도
                    for keyword in keywords:
                        if keyword in topic_part:
                            company_name = keyword
                            break
                    break
        
        if "뉴스 분석" in prompt or "분석할 뉴스" in prompt:
            return f"""{{
  "relevance_score": 8,
  "topics": ["기업기술", "제품출시", "시장동향"],
  "keywords": {json.dumps(keywords[:7] if keywords else [company_name, "기술혁신", "제품출시"], ensure_ascii=False)},
  "summary": "{company_name}가 새로운 기술 혁신을 통해 시장에서 주목받고 있습니다. 업계 전문가들은 긍정적인 반응을 보이고 있습니다.",
  "sentiment": "긍정",
  "importance": 8,
  "company_mentions": ["{company_name}"],
  "date": "2025-07-18",
  "source": "테스트뉴스"
}}"""
        elif "청크로 분할" in prompt:
            return f"""{{
  "chunks": [
    {{
      "chunk_id": 1,
      "content": "{company_name}가 새로운 혁신 기술을 공식 출시했다고 발표했습니다.",
      "topics": ["제품출시", "기업발표"],
      "keywords": {json.dumps([company_name] + keywords[:3], ensure_ascii=False)},
      "chunk_type": "제목"
    }},
    {{
      "chunk_id": 2,
      "content": "이번 기술은 기존 대비 향상된 성능을 보여주며, 시장에서 큰 관심을 받고 있습니다. 전문가들은 업계 전반에 긍정적인 영향을 미칠 것으로 예상한다고 밝혔습니다.",
      "topics": ["성능개선", "기술혁신"],
      "keywords": {json.dumps(keywords[:4] if keywords else ["성능향상", "기술혁신"], ensure_ascii=False)},
      "chunk_type": "본문"
    }}
  ]
}}"""
        elif "품질을 평가" in prompt:
            return """{
  "overall_score": 8,
  "detailed_scores": {
    "facts": 8,
    "completeness": 9,
    "objectivity": 8,
    "readability": 9,
    "credibility": 7
  },
  "improvements": ["출처 정보 보강", "전문가 의견 추가"],
  "strengths": ["명확한 구조", "균형잡힌 내용", "전문적 문체"],
  "approval": true
}"""
        else:
            # 뉴스 생성용 더미 응답 (동적 회사명 반영)
            keyword_str = ", ".join(keywords) if keywords else f"{company_name}, 기술혁신, 시장선도"
            
            return f"""제목: {company_name}, 차세대 혁신 기술로 시장 선도 나서

리드: {company_name}가 새로운 혁신 기술을 공식 출시하며 업계의 주목을 받고 있다. 이번 기술은 기존 대비 크게 향상된 성능을 자랑하며, 시장에서 긍정적인 반응을 얻고 있다. 전문가들은 이번 발표가 업계 전반에 큰 변화를 가져올 것으로 예상한다고 밝혔다.

본문:
{company_name}는 이날 새로운 혁신 기술을 공식 발표하며 "기존 기술의 한계를 뛰어넘는 획기적인 솔루션"이라고 강조했다.

새로운 기술은 향상된 성능과 안정성을 바탕으로 다양한 분야에서 활용 가능하며, 특히 {keywords[1] if len(keywords) > 1 else "핵심 기능"}과 {keywords[2] if len(keywords) > 2 else "고성능 처리"} 부분에서 차별화된 경쟁력을 보여준다.

{company_name} 관계자는 "이번 기술 출시를 통해 시장에서의 입지를 더욱 강화할 것"이라며 "고객들에게 더 나은 서비스를 제공할 수 있게 됐다"고 밝혔다.

업계 전문가들은 "{company_name}의 이번 혁신이 시장 전반에 긍정적인 영향을 미칠 것"이라고 평가했으며, "기술력 측면에서 한 단계 발전한 모습을 보여주고 있다"고 분석했다.

결론: {company_name}는 이번 기술 출시를 바탕으로 시장 확대와 고객 만족도 향상을 동시에 추진할 계획이며, 지속적인 기술 혁신을 통해 업계 선두주자로 자리매김할 것으로 전망된다.

키워드: {keyword_str}"""

class EnhancedNewsCollector:
    """향상된 뉴스 수집기"""
    
    def __init__(self, claude_client: EnhancedClaudeClient, db_manager: EnhancedChromaDBManager, 
                 naver_api: EnhancedNaverNewsAPI):
        logger.debug("Initializing EnhancedNewsCollector")
        self.claude_client = claude_client
        self.db_manager = db_manager
        self.naver_api = naver_api
        logger.debug("EnhancedNewsCollector initialized.")
    
    async def collect_company_news_enhanced(self, company_name: str, additional_keywords: List[str] = None,
                                          days_back: int = 365, max_articles: int = 50) -> int:
        logger.debug(f"collect_company_news_enhanced called with company_name={company_name}, additional_keywords={additional_keywords}, days_back={days_back}, max_articles={max_articles}")
        collected_count = 0
        
        try:
            # 향상된 키워드 검색 사용
            articles = self.naver_api.search_news_with_keywords(
                company_name, 
                additional_keywords, 
                display=max_articles
            )
            logger.debug(f"Found {len(articles)} articles for collection.")
            
            for article in articles[:max_articles]:
                # 날짜 필터링
                if self._is_recent_article(article.pub_date, days_back):
                    logger.debug(f"Collecting and storing article: {article.title}")
                    success = await self.collect_and_store_news(company_name, article)
                    if success:
                        collected_count += 1
                
                # API 호출 제한을 위한 딜레이
                await asyncio.sleep(2)
                
                # 최대 수집 개수 도달시 중단
                if collected_count >= max_articles:
                    logger.info("Max articles collected, breaking loop.")
                    break
                
        except Exception as e:
            logger.error(f"뉴스 수집 중 오류: {e}")
                
        logger.info(f"뉴스 수집 완료: {collected_count}개 기사 저장")
        return collected_count
        
    async def collect_and_store_news(self, company_name: str, article: NewsArticle) -> bool:
        logger.debug(f"collect_and_store_news called for company_name={company_name}, article_title={article.title}")
        try:
            # 본문이 없으면 제목+설명 사용
            news_content = article.content if article.content else f"{article.title}\n{article.description}"
            
            if len(news_content.strip()) < 50:
                logger.warning(f"뉴스 내용이 너무 짧음: {article.title}")
                return False
            
            # 1. 간단한 텍스트 기반 관련도 검사 (NO API CALL!)
            full_text = f"{article.title} {article.description} {news_content}".lower()
            company_lower = company_name.lower()
            
            # 기본 관련도 계산
            relevance_score = 0
            
            # 회사명 언급 횟수
            company_mentions = full_text.count(company_lower)
            if company_mentions == 0:
                logger.info(f"회사명 없음: {article.title}")
                return False
            
            relevance_score += min(4, company_mentions)  # 최대 4점
            
            # 제목에 회사명 있으면 보너스
            if company_lower in article.title.lower():
                relevance_score += 3
            
            # 중요 키워드 체크
            important_keywords = ["출시", "발표", "개발", "계약", "파트너십", "투자", "매출", "실적"]
            for keyword in important_keywords:
                if keyword in full_text:
                    relevance_score += 1
            
            # 최종 관련도 (1-10 범위)
            final_relevance = max(1, min(10, relevance_score))
            
            # 관련도 5점 미만 제외
            if final_relevance < 5:
                logger.info(f"관련도 부족 ({final_relevance}): {article.title}")
                return False
            
            # 2. 간단한 청킹 (NO API CALL!)
            chunks = []
            
            # 제목 청크
            chunks.append(NewsChunk(
                chunk_id=1,
                content=article.title,
                topics=["제목"],
                keywords=[company_name],
                chunk_type="제목"
            ))
            
            # 본문 청크들 (문단별 분할)
            paragraphs = [p.strip() for p in news_content.split('\n') if len(p.strip()) > 30]
            
            for i, paragraph in enumerate(paragraphs[:3]):  # 최대 3개 문단
                chunks.append(NewsChunk(
                    chunk_id=i + 2,
                    content=paragraph,
                    topics=["본문"],
                    keywords=[company_name],
                    chunk_type="본문"
                ))
            
            # 3. 간단한 메타데이터 생성 (NO API CALL!)
            metadata = NewsMetadata(
                relevance_score=final_relevance,
                topics=["기업뉴스"],
                keywords=[company_name, "뉴스"],
                summary=article.description[:100] if article.description else article.title,
                sentiment="중립",
                importance=final_relevance,
                company_mentions=[company_name],
                date=self._convert_pub_date(article.pub_date),
                source=article.link
            )
            
            # 4. 벡터 DB에 저장
            for chunk in chunks:
                embedding = [0.1] * 768  # 더미 임베딩
                logger.debug(f"Storing chunk_id={chunk.chunk_id} for article: {article.title}")
                self.db_manager.store_news_chunk(chunk, metadata, embedding)
            
            logger.info(f"간단 저장 완료: {article.title[:50]}... ({len(chunks)}개 청크, 관련도: {final_relevance})")
            return True
            
        except Exception as e:
            logger.error(f"뉴스 수집 실패: {e}")
            return False

    def _extract_json_from_response(self, response: str) -> dict:
        """응답에서 JSON 추출 (개선됨)"""
        # JSON 블록 추출 시도
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSON 블록이 없으면 전체 텍스트에서 JSON 추출
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
            else:
                raise ValueError("JSON을 찾을 수 없습니다")
        
        # JSON 정리
        json_str = json_str.strip()
        
        # 첫 번째 완전한 JSON 객체만 추출
        if json_str.count('}') > 1:
            brace_count = 0
            json_end_pos = 0
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end_pos = i + 1
                        break
            json_str = json_str[:json_end_pos]
        
        return json.loads(json_str)
    
    def _is_recent_article(self, pub_date: str, days_back: int) -> bool:
        """최근 기사인지 확인"""
        try:
            from datetime import datetime
            article_date = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
            cutoff_date = datetime.now().astimezone() - timedelta(days=days_back)
            return article_date >= cutoff_date
        except:
            return True  # 파싱 실패시 포함
    
    def _convert_pub_date(self, pub_date: str) -> str:
        """날짜 형식 변환"""
        try:
            from datetime import datetime
            dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
            return dt.strftime("%Y-%m-%d")
        except:
            return datetime.now().strftime("%Y-%m-%d")

class EnhancedNewsWriter:
    """향상된 뉴스 작성기"""
    
    def __init__(self, claude_client, db_manager):
        logger.debug("Initializing EnhancedNewsWriter")
        self.claude_client = claude_client
        self.db_manager = db_manager
        logger.debug("EnhancedNewsWriter initialized.")
    
    def get_full_generation_prompt(self, topic, keywords, user_facts, reference_materials, length_specification=""):
        """Return the exact prompt string that will be sent to Claude."""
        return EnhancedPromptManager.get_enhanced_news_generation_prompt(
            topic, keywords, user_facts, reference_materials, length_specification
        )
    
    async def generate_enhanced_news(self, topic: str, keywords: List[str], 
                                   user_facts: str, style: str = "기업 보도형",
                                   length_specification: str = "",
                                   use_rag: bool = True, rag_count: int = 10) -> dict:
        logger.debug(f"generate_enhanced_news called with topic={topic}, keywords={keywords}, user_facts length={len(user_facts)}, style={style}, length_specification={length_specification}, use_rag={use_rag}, rag_count={rag_count}")
        try:
            reference_materials = ""
            if use_rag:
                target_companies = []
                for keyword in keywords:
                    if any(company_indicator in keyword.lower() for company_indicator in 
                          ['삼성', '알티베이스', 'lg', 'sk', '현대', '기업', '회사']):
                        target_companies.append(keyword)
                logger.debug(f"대상 회사 필터: {target_companies}")
                search_query = f"{topic} {' '.join(keywords)}"
                logger.info(f"RAG 검색: '{search_query[:50]}...' (상위 {rag_count}개)")
                search_results = self.db_manager.search_relevant_news(
                    search_query, 
                    n_results=rag_count,
                    min_relevance=6,  # 높은 관련도
                    target_companies=target_companies  # 회사 필터링 추가
                )
                result_count = len(search_results.get('documents', [[]])[0])
                logger.debug(f"RAG 검색 결과 개수: {result_count}")
                if result_count < 3:
                    logger.info(f"결과 부족({result_count}개), 낮은 관련도로 재검색...")
                    search_results = self.db_manager.search_relevant_news(
                        search_query,
                        n_results=rag_count,
                        min_relevance=3,  # 낮은 관련도
                        target_companies=target_companies
                    )
                    result_count = len(search_results.get('documents', [[]])[0])
                if result_count > 0:
                    reference_materials = self._build_comprehensive_reference_materials(search_results)
                    logger.info(f"RAG 참고 자료 구성 완료: {result_count}개 문서")
                else:
                    company_guide = target_companies[0] if target_companies else "해당 기업"
                    reference_materials = f"""참고 자료가 없습니다. 다음 가이드를 따라 작성하세요:\n\n**{company_guide} 관련 {topic} 뉴스 작성 가이드:**\n1. {company_guide}의 공식 발표 내용을 중심으로 구성\n2. 기술적 혁신점과 시장에서의 의미 강조  \n3. 업계 전문가 관점에서 분석\n4. 향후 전망과 기대효과 포함\n5. 고객과 파트너사에 미치는 영향 분석\n\n키워드 활용: {', '.join(keywords)}\n\n**중요: 반드시 제공된 모든 키워드를 자연스럽게 포함시켜 주세요.**\n"""
            generation_prompt = self.get_full_generation_prompt(
                topic, keywords, user_facts, reference_materials, length_specification
            )
            # Claude API 호출 및 분석 정보 수집
            result = await self.claude_client.generate_response(generation_prompt, max_tokens=8000)
            # 길이 검증 및 재생성 (분석 정보 유지)
            news_draft = result["response"]
            if length_specification and ("줄 수" in length_specification or "단어 수" in length_specification):
                news_draft = await self._ensure_proper_length(news_draft, length_specification, 
                                                            topic, keywords, user_facts, reference_materials)
                result["response"] = news_draft
            # 품질 검증 (분석 정보 유지)
            quality_check_prompt = EnhancedPromptManager.get_quality_check_prompt(news_draft)
            _ = await self.claude_client.generate_response(quality_check_prompt)  # 품질 평가는 분석에 포함하지 않음
            return result
        except Exception as e:
            logger.error(f"뉴스 생성 실패: {e}")
            return {"elapsed": 0, "model": "error", "prompt": "", "response": ""}
    
    def _build_comprehensive_reference_materials(self, search_results: Dict) -> str:
        """포괄적인 참고 자료 구성 (전체 내용 포함)"""
        if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
            return "관련 참고 자료가 없습니다."
        
        materials = []
        documents = search_results['documents'][0]
        metadatas = search_results.get('metadatas', [[]])[0] if search_results.get('metadatas') else []
        distances = search_results.get('distances', [[]])[0] if search_results.get('distances') else []
        
        for i, doc in enumerate(documents[:10]):  # 최대 10개
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 0
            
            # 메타데이터에서 정보 추출
            try:
                topics = json.loads(metadata.get('topics', '[]'))
                keywords = json.loads(metadata.get('keywords', '[]'))
                company_mentions = json.loads(metadata.get('company_mentions', '[]'))
            except:
                topics = []
                keywords = []
                company_mentions = []
            
            source = metadata.get('source', f'참고자료 {i+1}')
            date = metadata.get('date', 'N/A')
            importance = metadata.get('importance', 'N/A')
            relevance_score = metadata.get('relevance_score', 'N/A')
            sentiment = metadata.get('sentiment', 'N/A')
            summary = metadata.get('summary', '')
            
            # 종합적인 참고 자료 정보 구성
            material = f"""=== 참고자료 {i+1} ===
출처: {source}
날짜: {date}
관련도: {relevance_score}/10 (유사도: {1-distance:.3f})
중요도: {importance}/10
감정: {sentiment}
주요 토픽: {', '.join(topics[:3])}
키워드: {', '.join(keywords[:5])}
언급 기업: {', '.join(company_mentions)}

요약: {summary}

전체 내용:
{doc}

----------------------------------------

"""
            materials.append(material)
        
        reference_text = "\n".join(materials) if materials else "관련 참고 자료가 없습니다."
        
        # 통계 정보 추가
        stats_info = f"""
[참고 자료 통계]
- 총 {len(materials)}개 문서 참조
- 평균 관련도: {sum([float(m.get('relevance_score', 0)) for m in metadatas[:len(materials)]]) / len(materials) if materials else 0:.1f}/10
- 날짜 범위: {min([m.get('date', '') for m in metadatas[:len(materials)]])} ~ {max([m.get('date', '') for m in metadatas[:len(materials)]])}

"""
        
        return stats_info + reference_text

    async def _ensure_proper_length(self, news_draft: str, length_specification: str,
                                      topic: str, keywords: List[str], user_facts: str, 
                                      reference_materials: str) -> str:
        """길이 요구사항을 정확히 맞추기 위한 재생성"""
        
        if "줄 수" in length_specification:
            target_lines = int(re.search(r'(\d+)', length_specification).group(1))
            current_lines = len([line for line in news_draft.split('\n') if line.strip()])
            
            logger.info(f"현재 줄 수: {current_lines}, 목표: {target_lines}")
            
            # 목표 대비 80% 미만이거나 120% 초과시 재생성
            if current_lines < target_lines * 0.8 or current_lines > target_lines * 1.2:
                logger.info("길이 기준 미달/초과로 재생성 시도")
                
                # 더 명확한 길이 지시와 함께 재생성
                enhanced_prompt = f"""이전 생성된 뉴스가 {current_lines}줄이었는데, 정확히 {target_lines}줄이 필요합니다.

**반드시 {target_lines}줄로 작성하세요:**
- 더 상세한 내용 추가
- 각 단락을 충분히 확장
- 구체적인 사례와 분석 포함

{EnhancedPromptManager.get_enhanced_news_generation_prompt(topic, keywords, user_facts, reference_materials, length_specification)}

**중요: 정확히 {target_lines}줄을 맞춰주세요. 각 줄에 충분한 내용을 포함하세요.**"""
                
                regenerated_result = await self.claude_client.generate_response(enhanced_prompt, max_tokens=8000)
                return regenerated_result["response"]
        
        elif "단어 수" in length_specification:
            target_words = int(re.search(r'(\d+)', length_specification).group(1))
            current_words = len(news_draft.split())
            
            logger.info(f"현재 단어 수: {current_words}, 목표: {target_words}")
            
            if current_words < target_words * 0.8 or current_words > target_words * 1.2:
                logger.info("단어 수 기준 미달/초과로 재생성 시도")
                
                enhanced_prompt = f"""이전 생성된 뉴스가 {current_words}단어였는데, 정확히 {target_words}단어가 필요합니다.

**반드시 {target_words}단어로 작성하세요:**
- 더 상세하고 구체적인 설명
- 전문적인 분석과 해석 추가
- 다양한 관점과 의견 포함

{EnhancedPromptManager.get_enhanced_news_generation_prompt(topic, keywords, user_facts, reference_materials, length_specification)}

**중요: 정확히 {target_words}단어를 맞춰주세요.**"""
                
                regenerated_result = await self.claude_client.generate_response(enhanced_prompt, max_tokens=8000)
                return regenerated_result["response"]
        
        return news_draft
       
    def _extract_json_from_response(self, response: str) -> dict:
        """응답에서 JSON 추출"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
            else:
                raise ValueError("JSON을 찾을 수 없습니다")
        
        return json.loads(json_str.strip())

class EnhancedAINewsWriterSystem:
    """향상된 AI News Writer 시스템 메인 클래스"""
    
    def __init__(self, claude_api_key: str = None, naver_client_id: str = None, 
                 naver_client_secret: str = None, db_path: str = "./chroma_db"):
        logger.debug("Initializing EnhancedAINewsWriterSystem")
        self.claude_client = EnhancedClaudeClient(claude_api_key)
        self.db_manager = EnhancedChromaDBManager(db_path)
        self.naver_api = EnhancedNaverNewsAPI(naver_client_id, naver_client_secret)
        self.news_collector = EnhancedNewsCollector(self.claude_client, self.db_manager, self.naver_api)
        self.news_writer = EnhancedNewsWriter(self.claude_client, self.db_manager)
        logger.info("Enhanced AI News Writer 시스템 초기화 완료")
        logger.debug("EnhancedAINewsWriterSystem initialized.")
    
    async def collect_news_background(self, company_name: str, additional_keywords: List[str] = None,
                                    days_back: int = 365, max_articles: int = 50) -> int:
        """향상된 백그라운드 뉴스 수집"""
        return await self.news_collector.collect_company_news_enhanced(
            company_name, additional_keywords, days_back, max_articles
        )
    
    async def collect_manual_news(self, company_name: str, news_content: str) -> bool:
        """수동 뉴스 입력"""
        from datetime import datetime
        
        article = NewsArticle(
            title="수동 입력 뉴스",
            link="manual_input",
            description="사용자가 직접 입력한 뉴스",
            pub_date=datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z"),
            content=news_content
        )
        
        return await self.news_collector.collect_and_store_news(company_name, article)
    
    async def write_news(self, topic: str, keywords: List[str], user_facts: str, 
                        style: str = "기업 보도형", length_specification: str = "",
                        use_rag: bool = True, rag_count: int = 10) -> Optional[str]:
        """향상된 뉴스 작성"""
        return await self.news_writer.generate_enhanced_news(
            topic, keywords, user_facts, style, length_specification, use_rag, rag_count
        )
    
    def get_system_stats(self) -> Dict:
        """시스템 통계 조회"""
        try:
            db_stats = self.db_manager.get_collection_stats()
            return {
                "database": db_stats,
                "api_requests": self.claude_client.request_count,
                "naver_test_mode": self.naver_api.test_mode
            }
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {}
    
    async def schedule_news_collection(self, company_name: str, additional_keywords: List[str] = None,
                                     interval_hours: int = 6, max_articles: int = 20):
        """뉴스 수집 스케줄러"""
        logger.info(f"{company_name} 뉴스 수집 스케줄러 시작 (간격: {interval_hours}시간)")
        
        while True:
            try:
                collected = await self.collect_news_background(
                    company_name, additional_keywords, days_back=1, max_articles=max_articles
                )
                logger.info(f"스케줄 수집 완료: {collected}개 기사")
                
                # 다음 수집까지 대기
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"스케줄 수집 오류: {e}")
                await asyncio.sleep(60)  # 1분 후 재시도

# 이전 버전과의 호환성을 위한 별칭
AINewsWriterSystem = EnhancedAINewsWriterSystem

# 사용 예시
async def enhanced_main():
    """향상된 시스템 데모"""
    print("=== Enhanced AI News Writer 시스템 ===\n")
    
    # .env 파일에서 API 키 로드
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    naver_client_id = os.getenv('NAVER_CLIENT_ID') 
    naver_client_secret = os.getenv('NAVER_CLIENT_SECRET')
    
    # API 키 상태 확인
    print("=== 🔑 API 키 상태 확인 ===")
    print(f"Claude API Key: {'✅ 설정됨' if claude_api_key else '❌ 없음 (테스트 모드)'}")
    print(f"네이버 Client ID: {'✅ 설정됨' if naver_client_id else '❌ 없음 (테스트 모드)'}")
    print(f"네이버 Client Secret: {'✅ 설정됨' if naver_client_secret else '❌ 없음 (테스트 모드)'}")
    print()
    
    # 시스템 초기화
    system = EnhancedAINewsWriterSystem(
        claude_api_key=claude_api_key,
        naver_client_id=naver_client_id,
        naver_client_secret=naver_client_secret
    )
    
    company_name = "삼성전자"  # 변경된 회사명
    additional_keywords = ["스마트폰", "갤럭시", "반도체", "OLED"]  # 변경된 키워드
    
    # 1. 향상된 뉴스 수집 테스트
    print("1. 향상된 키워드 조합 뉴스 수집 중...")
    collected_count = await system.collect_news_background(
        company_name, 
        additional_keywords, 
        days_back=365,  # 12개월
        max_articles=20
    )
    print(f"수집 결과: {collected_count}개 기사\n")
    
    # 2. 수동 뉴스 추가
    print("2. 수동 뉴스 추가 중...")
    manual_news = f"""
    {company_name}가 차세대 스마트폰 '갤럭시 S30'을 공식 출시했다.
    이번 신제품은 기존 버전 대비 30% 향상된 배터리 성능과 20% 개선된 카메라 화질을 자랑한다.
    AI 기능이 대폭 강화되어 사용자들의 주목을 받고 있으며,
    독자 개발한 엑시노스 프로세서와 고해상도 OLED 디스플레이가 적용되었다.
    회사 관계자는 "모바일 혁신의 새로운 기준을 제시하는 제품"이라고 밝혔다.
    """
    
    manual_success = await system.collect_manual_news(company_name, manual_news)
    print(f"수동 뉴스 추가: {'✅ 성공' if manual_success else '❌ 실패'}\n")
    
    # 3. 향상된 RAG 뉴스 작성
    print("3. 향상된 RAG AI 뉴스 작성 중...")
    topic = "스마트폰 신제품 출시"
    keywords = ["삼성전자", "갤럭시", "스마트폰", "AI기능", "엑시노스"]  # 변경된 키워드
    user_facts = f"""{company_name}가 갤럭시 S30이라는 혁신적인 스마트폰을 출시했다.
주요 특징:
- 기존 대비 30% 배터리 성능 향상
- 카메라 화질 20% 개선
- AI 기능 대폭 강화
- 엑시노스 프로세서 탑재
- 고해상도 OLED 디스플레이"""
    
    generated_news = await system.write_news(
        topic=topic,
        keywords=keywords,
        user_facts=user_facts,
        style="기업 보도형",
        length_specification="100줄 분량의 상세한 뉴스",
        use_rag=True,
        rag_count=10
    )
    
    if generated_news:
        print("=== 🗞️ 생성된 Enhanced AI 뉴스 ===")
        print(generated_news)
        print("\n" + "="*70)
    else:
        print("❌ 뉴스 생성 실패")
    
    # 4. 시스템 통계
    print("\n=== 📊 시스템 통계 ===")
    stats = system.get_system_stats()
    print(f"DB 저장된 청크 수: {stats.get('database', {}).get('total_chunks', 0)}")
    print(f"API 호출 횟수: {stats.get('api_requests', 0)}")
    print(f"네이버 API 모드: {'테스트' if stats.get('naver_test_mode', True) else '실제'}")
    
    print("\n=== ✨ 주요 수정사항 ===")
    print("✅ 회사명별 동적 더미 뉴스 생성")
    print("✅ RAG 검색 시 회사 필터링 강화")
    print("✅ 키워드 기반 뉴스 내용 생성")
    print("✅ 테스트 모드에서 입력 키워드 반영")
    print("✅ 회사별 맞춤형 참고자료 가이드")

# 동기 함수로 래핑
def run_enhanced_main():
    """향상된 메인 함수를 동기적으로 실행"""
    asyncio.run(enhanced_main())

if __name__ == "__main__":
    run_enhanced_main()