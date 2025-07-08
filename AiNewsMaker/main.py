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
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/search/news.json"
        
        if not client_id or client_id == "YOUR_NAVER_CLIENT_ID":
            logger.warning("네이버 API 키가 설정되지 않았습니다. 테스트 모드로 실행됩니다.")
            self.test_mode = True
        else:
            self.test_mode = False
    
    def search_news_with_keywords(self, company_name: str, additional_keywords: List[str] = None, 
                                display: int = 10, start: int = 1, sort: str = "date") -> List[NewsArticle]:
        """회사명과 추가 키워드를 조합한 향상된 뉴스 검색"""
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
        logger.info(f"검색 쿼리 생성: {search_queries}")
        
        for query in search_queries[:5]:  # 최대 5개 쿼리만 실행
            try:
                articles = self.search_news(query, display=min(display, 10), start=start, sort=sort)
                all_articles.extend(articles)
                
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
        return unique_articles[:display]  # 요청된 개수만큼 반환
    
    def search_news(self, query: str, display: int = 10, start: int = 1, 
                   sort: str = "date") -> List[NewsArticle]:
        """기본 네이버 뉴스 검색"""
        if self.test_mode:
            return self._get_dummy_news(query, display)
        
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"{self.base_url}?query={encoded_query}&display={display}&start={start}&sort={sort}"
            
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
        """테스트용 더미 뉴스 (확장됨)"""
        dummy_articles = []
        
        base_articles = [
            {
                "title_template": "{query} 관련 최신 기술 동향 발표",
                "description_template": "{query}의 새로운 혁신이 업계에 주목받고 있습니다.",
                "content_template": "{query}가 새로운 기술 혁신을 통해 시장에서 주목받고 있습니다. 전문가들은 이번 발표가 업계 전반에 큰 변화를 가져올 것으로 예상한다고 밝혔습니다. 특히 성능 향상과 안정성 개선에 중점을 둔 이번 업데이트는 고객들로부터 긍정적인 반응을 얻고 있습니다."
            },
            {
                "title_template": "{query} 시장 점유율 확대 소식",
                "description_template": "{query}의 시장 영향력이 계속 확대되고 있습니다.",
                "content_template": "{query}의 시장 점유율이 지속적으로 확대되고 있어 업계의 관심이 집중되고 있습니다. 분석가들은 향후 성장 전망을 긍정적으로 평가하고 있으며, 신제품 출시와 함께 더욱 강력한 시장 지배력을 보일 것으로 전망됩니다."
            },
            {
                "title_template": "{query} 신규 파트너십 체결",
                "description_template": "{query}가 글로벌 기업과의 전략적 파트너십을 발표했습니다.",
                "content_template": "{query}가 글로벌 선도 기업과의 전략적 파트너십을 체결했다고 발표했습니다. 이번 협업을 통해 기술력 향상과 시장 확대에 탄력을 받을 것으로 기대되며, 양사는 상호 보완적인 기술과 경험을 바탕으로 시너지 효과를 창출할 계획입니다."
            }
        ]
        
        for i in range(min(display, len(base_articles) * 2)):
            template = base_articles[i % len(base_articles)]
            
            dummy_articles.append(NewsArticle(
                title=template["title_template"].format(query=query),
                link=f"http://test.com/news{i+1}",
                description=template["description_template"].format(query=query),
                pub_date=f"Mon, {7 + i} Jul 2025 {10 + i}:00:00 +0900",
                content=template["content_template"].format(query=query)
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
        return f"""당신은 전문적인 뉴스 기자입니다. 다음 조건에 맞춰 고품질 뉴스 기사를 작성해주세요.

**입력 정보:**
- 토픽: {topic}
- 키워드: {keywords_str}
- 사용자 제공 사실: {user_facts}
- 길이 요구사항: {length_specification}

**참고 자료 (RAG):**
{reference_materials}

**작성 가이드라인:**
1. 객관적이고 사실적인 보도 스타일
2. 5W1H (Who, What, When, Where, Why, How) 원칙 준수
3. 제목, 리드, 본문, 결론의 명확한 구조
4. 참고 자료의 정보를 자연스럽게 통합
5. 전문적이고 신뢰성 있는 표현 사용

**뉴스 구조:**
1. **제목** (25-35자, 임팩트 있고 명확하게)
2. **리드** (핵심 내용 요약, 2-3문장, 육하원칙 포함)
3. **본문** (상세 내용, 논리적 단락 구성)
   - 배경 정보
   - 주요 내용 상세 설명
   - 관련자 발언 또는 전문가 의견
   - 시장 반응 또는 영향 분석
4. **결론** (전망, 의미, 후속 계획)

**품질 기준:**
- 정확한 정보만 사용
- 균형잡힌 시각 제시
- 참고 자료 내용을 적절히 인용
- 전문적인 뉴스 문체 유지
- {length_specification}

**출력 형식:**
```
제목: [뉴스 제목]

리드: [핵심 내용 요약]

본문:
[배경 정보 단락]

[주요 내용 단락]

[전문가 의견/관련자 발언 단락]

[시장 반응/영향 분석 단락]

결론: [전망 또는 의미, 후속 계획]

키워드: [관련 키워드 나열]
```

참고 자료의 내용을 바탕으로 사실적이고 신뢰성 있는 전문 뉴스를 작성해주세요."""

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

class EnhancedChromaDBManager:
    """향상된 ChromaDB 관리 클래스"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name="enhanced_news_collection",
                metadata={"description": "Enhanced AI News Writer 뉴스 컬렉션"}
            )
            logger.info(f"ChromaDB 초기화 완료: {db_path}")
        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패: {e}")
            raise
    
    def store_news_chunk(self, chunk: NewsChunk, metadata: NewsMetadata, 
                        embedding: List[float]) -> None:
        """뉴스 청크를 벡터 DB에 저장 (개선됨)"""
        try:
            # 임베딩 차원을 768로 통일
            if len(embedding) != 768:
                if len(embedding) < 768:
                    embedding = embedding + [0.0] * (768 - len(embedding))
                else:
                    embedding = embedding[:768]
            
            # 고유 ID 생성 (중복 방지)
            import hashlib
            content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
            unique_id = f"chunk_{content_hash}_{chunk.chunk_id}_{int(time.time())}"
            
            # 메타데이터 확장
            chunk_metadata = {
                "topics": json.dumps(chunk.topics, ensure_ascii=False),
                "keywords": json.dumps(chunk.keywords, ensure_ascii=False),
                "chunk_type": chunk.chunk_type,
                "sentiment": metadata.sentiment,
                "importance": metadata.importance,
                "relevance_score": metadata.relevance_score,
                "company_mentions": json.dumps(metadata.company_mentions, ensure_ascii=False),
                "date": metadata.date,
                "source": metadata.source,
                "summary": metadata.summary,
                "created_at": datetime.now().isoformat()
            }
            
            self.collection.add(
                documents=[chunk.content],
                metadatas=[chunk_metadata],
                embeddings=[embedding],
                ids=[unique_id]
            )
            logger.info(f"청크 저장 완료: {chunk.chunk_id} (ID: {unique_id})")
        except Exception as e:
            logger.error(f"청크 저장 실패: {e}")
    
    def search_relevant_news(self, query: str, n_results: int = 10, 
                           min_relevance: int = 5) -> Dict:
        """관련 뉴스 검색 (개선됨)"""
        try:
            # 컬렉션이 비어있는지 확인
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.warning("컬렉션이 비어있습니다.")
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            
            # 실제 사용 가능한 결과 수 제한
            actual_n_results = min(n_results, collection_count)
            
            # 텍스트 기반 검색 실행
            results = self.collection.query(
                query_texts=[query],
                n_results=actual_n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 관련도 필터링 (메타데이터가 있는 경우)
            if results.get('metadatas') and results['metadatas'][0]:
                filtered_results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
                
                for i, metadata in enumerate(results['metadatas'][0]):
                    relevance = metadata.get('relevance_score', 10)  # 기본값 10
                    if relevance >= min_relevance:
                        filtered_results['documents'][0].append(results['documents'][0][i])
                        filtered_results['metadatas'][0].append(metadata)
                        filtered_results['distances'][0].append(results['distances'][0][i])
                
                logger.info(f"검색 완료: {len(filtered_results['documents'][0])}개 결과 (관련도 {min_relevance}+ 필터링)")
                return filtered_results
            else:
                logger.info(f"검색 완료: {len(results['documents'][0])}개 결과")
                return results
                
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    def get_collection_stats(self) -> Dict:
        """컬렉션 통계 조회"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {"total_chunks": 0, "collection_name": "unknown"}

class EnhancedClaudeClient:
    """향상된 Claude API 클라이언트"""
    
    def __init__(self, api_key: str = None):
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
    
    async def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        """Claude API를 통해 응답 생성 (향상된 속도 제한 및 오류 처리)"""
        # API 호출 간격 제한
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            logger.info(f"API 호출 제한을 위해 {sleep_time:.1f}초 대기중...")
            await asyncio.sleep(sleep_time)
        
        self.last_call_time = time.time()
        self.request_count += 1
        
        if not self.client:
            # API 키가 없을 때 테스트용 더미 응답
            return self._get_enhanced_dummy_response(prompt)
        
        try:
            # 재시도 로직 강화
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=max_tokens,
                        temperature=0.1,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    logger.info(f"Claude API 호출 성공 (요청 #{self.request_count})")
                    return response.content[0].text
                    
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
            return self._get_enhanced_dummy_response(prompt)
    
    def _get_enhanced_dummy_response(self, prompt: str) -> str:
        """향상된 테스트용 더미 응답 생성"""
        if "뉴스 분석" in prompt or "분석할 뉴스" in prompt:
            return """{
  "relevance_score": 8,
  "topics": ["기업기술", "제품출시", "시장동향"],
  "keywords": ["알티베이스", "HyperDB", "인메모리DB", "성능향상", "실시간분석", "데이터베이스", "기술혁신"],
  "summary": "알티베이스가 새로운 인메모리 데이터베이스 엔진 HyperDB 3.0을 출시하여 30% 성능 향상을 달성했습니다. 실시간 분석 기능이 강화되어 금융권의 관심이 높아지고 있습니다.",
  "sentiment": "긍정",
  "importance": 8,
  "company_mentions": ["알티베이스"],
  "date": "2025-07-07",
  "source": "테스트뉴스"
}"""
        elif "청크로 분할" in prompt:
            return """{
  "chunks": [
    {
      "chunk_id": 1,
      "content": "알티베이스가 새로운 인메모리 데이터베이스 엔진 'HyperDB 3.0'을 공식 출시했다고 7일 발표했습니다.",
      "topics": ["제품출시", "기업발표"],
      "keywords": ["알티베이스", "HyperDB", "출시", "발표"],
      "chunk_type": "제목"
    },
    {
      "chunk_id": 2,
      "content": "이번 버전은 기존 대비 30% 향상된 성능을 보여주며, 실시간 분석 기능이 대폭 강화되었습니다. 메모리 최적화 알고리즘과 병렬 처리 기술이 적용되어 대용량 데이터 처리가 가능합니다.",
      "topics": ["성능개선", "기술혁신"],
      "keywords": ["성능향상", "실시간분석", "메모리최적화", "병렬처리"],
      "chunk_type": "본문"
    },
    {
      "chunk_id": 3,
      "content": "알티베이스 관계자는 'HyperDB 3.0은 디지털 전환 시대의 요구에 부합하는 혁신적인 제품'이라며 '금융, 통신, 제조업 등에서 안정적인 성능을 제공할 것'이라고 밝혔습니다.",
      "topics": ["관계자발언", "산업적용"],
      "keywords": ["디지털전환", "금융", "통신", "제조업", "안정성"],
      "chunk_type": "인용"
    }
  ]
}"""
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
            # 뉴스 생성용 더미 응답 (향상됨)
            return """제목: 알티베이스, 차세대 인메모리 DB 엔진 'HyperDB 3.0' 출시로 시장 선도

리드: 국내 데이터베이스 전문기업 알티베이스가 기존 대비 30% 향상된 성능의 차세대 인메모리 데이터베이스 엔진 'HyperDB 3.0'을 공식 출시했다고 7일 발표했다. 실시간 분석 기능 강화와 메모리 최적화로 대기업과 금융권의 주목을 받고 있다.

본문:
알티베이스(대표 이홍성)는 이날 서울 강남구 본사에서 기자간담회를 열고 "HyperDB 3.0이 기존 버전 대비 처리 속도 30% 향상, 메모리 사용량 20% 절감을 달성했다"고 밝혔다.

새로운 엔진은 독자 개발한 메모리 최적화 알고리즘과 멀티코어 병렬 처리 기술을 적용해 대용량 데이터 실시간 처리 성능을 크게 개선했다. 특히 복잡한 분석 쿼리 처리 시간을 기존 대비 절반으로 단축시켰다는 것이 회사 측 설명이다.

이홍성 대표는 "디지털 전환 가속화로 실시간 데이터 처리 수요가 급증하는 상황에서 HyperDB 3.0은 차별화된 기술력으로 시장을 선도할 것"이라며 "특히 금융, 통신, 제조업 등 미션 크리티컬한 환경에서 안정적이고 빠른 성능을 보장한다"고 강조했다.

업계에서는 이번 제품 출시가 국내 데이터베이스 시장의 경쟁 구도를 바꿀 수 있을 것으로 관측하고 있다. 한 DB 전문가는 "알티베이스의 기술력이 글로벌 수준에 근접했다"며 "오라클, IBM 등 해외 업체들과의 경쟁에서도 충분히 승부할 수 있을 것"이라고 평가했다.

결론: 알티베이스는 HyperDB 3.0을 통해 국내외 시장 확대를 본격화할 계획이며, 올 하반기 글로벌 진출을 위한 마케팅을 강화할 예정이다. 회사는 2025년 매출 300억원 달성을 목표로 하고 있다.

키워드: 알티베이스, HyperDB, 인메모리데이터베이스, 성능향상, 실시간분석, 디지털전환, 데이터베이스시장"""

class EnhancedNewsCollector:
    """향상된 뉴스 수집기"""
    
    def __init__(self, claude_client: EnhancedClaudeClient, db_manager: EnhancedChromaDBManager, 
                 naver_api: EnhancedNaverNewsAPI):
        self.claude_client = claude_client
        self.db_manager = db_manager
        self.naver_api = naver_api
        
    async def collect_company_news_enhanced(self, company_name: str, additional_keywords: List[str] = None,
                                          days_back: int = 365, max_articles: int = 50) -> int:
        """향상된 회사 관련 뉴스 자동 수집"""
        logger.info(f"{company_name} 뉴스 수집 시작 (키워드: {additional_keywords}, 최근 {days_back}일, 최대 {max_articles}개)")
        
        collected_count = 0
        
        try:
            # 향상된 키워드 검색 사용
            articles = self.naver_api.search_news_with_keywords(
                company_name, 
                additional_keywords, 
                display=max_articles
            )
            
            for article in articles[:max_articles]:
                # 날짜 필터링
                if self._is_recent_article(article.pub_date, days_back):
                    success = await self.collect_and_store_news(company_name, article)
                    if success:
                        collected_count += 1
                
                # API 호출 제한을 위한 딜레이
                await asyncio.sleep(2)
                
                # 최대 수집 개수 도달시 중단
                if collected_count >= max_articles:
                    break
                
        except Exception as e:
            logger.error(f"뉴스 수집 중 오류: {e}")
                
        logger.info(f"뉴스 수집 완료: {collected_count}개 기사 저장")
        return collected_count
    
    async def collect_and_store_news(self, company_name: str, article: NewsArticle) -> bool:
        """개별 뉴스 분석 및 저장 (개선됨)"""
        try:
            # 본문이 없으면 제목+설명 사용
            news_content = article.content if article.content else f"{article.title}\n{article.description}"
            
            if len(news_content.strip()) < 50:  # 너무 짧은 내용 제외
                logger.warning(f"뉴스 내용이 너무 짧음: {article.title}")
                return False
            
            # 1. 뉴스 분석 및 메타데이터 추출
            analysis_prompt = EnhancedPromptManager.get_news_analysis_prompt(news_content, company_name)
            analysis_result = await self.claude_client.generate_response(analysis_prompt)
            
            # JSON 파싱 개선
            try:
                metadata_dict = self._extract_json_from_response(analysis_result)
                metadata = NewsMetadata(**metadata_dict)
            except Exception as e:
                logger.error(f"메타데이터 파싱 실패: {e}")
                return False
            
            # 관련도 검사
            if metadata.relevance_score < 5:
                logger.info(f"관련도 부족 ({metadata.relevance_score}): {article.title}")
                return False
            
            # 2. 뉴스 청킹
            chunking_prompt = EnhancedPromptManager.get_news_chunking_prompt(news_content)
            chunking_result = await self.claude_client.generate_response(chunking_prompt)
            
            try:
                chunks_dict = self._extract_json_from_response(chunking_result)
                chunks = [NewsChunk(**chunk) for chunk in chunks_dict['chunks']]
            except Exception as e:
                logger.error(f"청킹 결과 파싱 실패: {e}")
                return False
            
            # 3. 메타데이터 업데이트
            metadata.source = article.link
            metadata.date = self._convert_pub_date(article.pub_date)
            
            # 4. 벡터 DB에 저장
            for chunk in chunks:
                # 768차원 더미 임베딩 생성 (실제 구현시 Sentence Transformers 사용 권장)
                embedding = [0.1] * 768
                self.db_manager.store_news_chunk(chunk, metadata, embedding)
            
            logger.info(f"뉴스 저장 완료: {article.title[:50]}... ({len(chunks)}개 청크, 관련도: {metadata.relevance_score})")
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
    
    def __init__(self, claude_client: EnhancedClaudeClient, db_manager: EnhancedChromaDBManager):
        self.claude_client = claude_client
        self.db_manager = db_manager
    
    async def generate_enhanced_news(self, topic: str, keywords: List[str], 
                                   user_facts: str, style: str = "기업 보도형",
                                   length_specification: str = "",
                                   use_rag: bool = True, rag_count: int = 10) -> Optional[str]:
        """향상된 뉴스 생성 (RAG 개선)"""
        try:
            reference_materials = ""
            
            if use_rag:
                # 1. 관련 뉴스 검색 (개선된 RAG)
                search_query = f"{topic} {' '.join(keywords)}"
                logger.info(f"RAG 검색: '{search_query}' (상위 {rag_count}개)")
                
                search_results = self.db_manager.search_relevant_news(
                    search_query, 
                    n_results=rag_count,
                    min_relevance=6  # 관련도 6점 이상만
                )
                
                # 2. 참고 자료 구성 (전체 내용 포함)
                reference_materials = self._build_comprehensive_reference_materials(search_results)
                logger.info(f"RAG 참고 자료 구성 완료: {len(search_results.get('documents', [[]])[0])}개 문서")
            
            # 3. 뉴스 작성
            generation_prompt = EnhancedPromptManager.get_enhanced_news_generation_prompt(
                topic, keywords, user_facts, reference_materials, length_specification
            )
            
            news_draft = await self.claude_client.generate_response(generation_prompt, max_tokens=6000)
            
            # 4. 품질 검증
            quality_check_prompt = EnhancedPromptManager.get_quality_check_prompt(news_draft)
            quality_result = await self.claude_client.generate_response(quality_check_prompt)
            
            try:
                quality_dict = self._extract_json_from_response(quality_result)
                logger.info(f"뉴스 품질 평가: {quality_dict.get('overall_score', 0)}점")
                
                if quality_dict.get('approval', False):
                    logger.info("뉴스 품질 검증 통과")
                    return news_draft
                else:
                    logger.warning("뉴스 품질 검증 실패하지만 결과 반환")
                    return news_draft  # 실패해도 뉴스는 반환
            except:
                logger.warning("품질 검증 결과 파싱 실패")
                return news_draft  # 파싱 실패시 원본 반환
            
        except Exception as e:
            logger.error(f"뉴스 생성 실패: {e}")
            return None
    
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
        self.claude_client = EnhancedClaudeClient(claude_api_key)
        self.db_manager = EnhancedChromaDBManager(db_path)
        self.naver_api = EnhancedNaverNewsAPI(naver_client_id, naver_client_secret)
        self.news_collector = EnhancedNewsCollector(self.claude_client, self.db_manager, self.naver_api)
        self.news_writer = EnhancedNewsWriter(self.claude_client, self.db_manager)
        
        logger.info("Enhanced AI News Writer 시스템 초기화 완료")
    
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
    
    company_name = "알티베이스"
    additional_keywords = ["데이터베이스", "DBMS", "오라클", "인메모리"]
    
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
    manual_news = """
    알티베이스가 차세대 인메모리 데이터베이스 'HyperDB 3.0'을 공식 출시했다.
    이번 신제품은 기존 버전 대비 30% 향상된 처리 성능과 20% 절약된 메모리 사용량을 자랑한다.
    실시간 분석 기능이 대폭 강화되어 금융권과 대기업의 주목을 받고 있으며,
    독자 개발한 메모리 최적화 알고리즘과 멀티코어 병렬 처리 기술이 적용되었다.
    회사 관계자는 "디지털 전환 시대의 요구에 부합하는 혁신적인 제품"이라고 밝혔다.
    """
    
    manual_success = await system.collect_manual_news(company_name, manual_news)
    print(f"수동 뉴스 추가: {'✅ 성공' if manual_success else '❌ 실패'}\n")
    
    # 3. 향상된 RAG 뉴스 작성
    print("3. 향상된 RAG AI 뉴스 작성 중...")
    topic = "데이터베이스 신기술 출시"
    keywords = ["알티베이스", "HyperDB", "인메모리", "성능향상", "실시간분석"]
    user_facts = """알티베이스가 HyperDB 3.0이라는 혁신적인 인메모리 데이터베이스를 출시했다.
주요 특징:
- 기존 대비 30% 성능 향상
- 메모리 사용량 20% 절약
- 실시간 분석 기능 강화
- 멀티코어 병렬 처리 지원
- 금융권과 대기업 타겟"""
    
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
    
    print("\n=== ✨ 향상된 기능 ===")
    print("✅ 회사명 + 추가 키워드 조합 검색")
    print("✅ 12개월(365일) 기본 수집 기간")
    print("✅ 로컬 파일 자동 저장")
    print("✅ 개선된 RAG (10개 뉴스 전체 내용 참조)")
    print("✅ 향상된 뉴스 품질 평가")
    print("✅ 뉴스 길이 조절 지원")
    print("✅ 종합적인 참고 자료 메타데이터")

# 동기 함수로 래핑
def run_enhanced_main():
    """향상된 메인 함수를 동기적으로 실행"""
    asyncio.run(enhanced_main())

if __name__ == "__main__":
    run_enhanced_main()