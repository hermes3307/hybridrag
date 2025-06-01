#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGNEX1 데이터 수집 시스템 (Naver + Kakao API)
다중 API를 이용한 백그라운드 데이터 수집 및 저장
"""

import os
import json
import hashlib
import sqlite3
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import threading
from dataclasses import dataclass, asdict
import requests
from urllib.parse import quote
import schedule
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lignex1_extract.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ArticleData:
    """수집된 기사 데이터 구조"""
    title: str
    description: str
    link: str
    pub_date: str
    source: str
    search_keyword: str
    api_type: str
    api_provider: str  # 'naver' 또는 'kakao'
    content_hash: str
    collected_at: str
    thumbnail: str = ""  # 썸네일 이미지 (Kakao API용)
    
class NaverAPIClient:
    """Naver API 클라이언트"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/search"
        self.headers = {
            'X-Naver-Client-Id': client_id,
            'X-Naver-Client-Secret': client_secret,
            'User-Agent': 'LIGNEX1-DataCollector/1.0'
        }
        
        # API 엔드포인트 정의
        self.endpoints = {
            'news': f"{self.base_url}/news.json",
            'blog': f"{self.base_url}/blog.json",
            'webkr': f"{self.base_url}/webkr.json",
            'cafearticle': f"{self.base_url}/cafearticle.json"
        }
    
    def search(self, query: str, api_type: str = 'news', 
               display: int = 100, start: int = 1, sort: str = 'date') -> Optional[Dict]:
        """
        Naver API 검색 실행
        
        Args:
            query: 검색어
            api_type: API 종류 (news, blog, webkr, cafearticle)
            display: 검색 결과 개수 (최대 100)
            start: 검색 시작 위치
            sort: 정렬 방식 (date, sim)
        
        Returns:
            검색 결과 딕셔너리 또는 None (오류시)
        """
        if api_type not in self.endpoints:
            logger.error(f"지원하지 않는 Naver API 타입: {api_type}")
            return None
        
        params = {
            'query': query,
            'display': min(display, 100),  # 최대 100개로 제한
            'start': start,
            'sort': sort
        }
        
        try:
            response = requests.get(
                self.endpoints[api_type],
                headers=self.headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                result['api_provider'] = 'naver'
                return result
            elif response.status_code == 429:
                logger.warning("Naver API 호출 한도 초과, 잠시 대기...")
                time.sleep(1)
                return None
            else:
                logger.error(f"Naver API 호출 실패: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Naver API 요청 중 오류 발생: {e}")
            return None
        except Exception as e:
            logger.error(f"Naver API 예상치 못한 오류: {e}")
            return None

class KakaoAPIClient:
    """Kakao API 클라이언트"""
    
    def __init__(self, rest_api_key: str):
        self.rest_api_key = rest_api_key
        self.base_url = "https://dapi.kakao.com/v2/search"
        self.headers = {
            'Authorization': f'KakaoAK {rest_api_key}',
            'User-Agent': 'LIGNEX1-DataCollector/1.0'
        }
        
        # API 엔드포인트 정의
        self.endpoints = {
            'web': f"{self.base_url}/web",
            'vclip': f"{self.base_url}/vclip",
            'image': f"{self.base_url}/image",
            'blog': f"{self.base_url}/blog",
            'book': f"{self.base_url}/book",
            'cafe': f"{self.base_url}/cafe"
        }
    
    def search(self, query: str, api_type: str = 'web', 
               size: int = 50, page: int = 1, sort: str = 'recency') -> Optional[Dict]:
        """
        Kakao API 검색 실행
        
        Args:
            query: 검색어
            api_type: API 종류 (web, vclip, image, blog, book, cafe)
            size: 검색 결과 개수 (최대 50)
            page: 페이지 번호
            sort: 정렬 방식 (accuracy, recency)
        
        Returns:
            검색 결과 딕셔너리 또는 None (오류시)
        """
        if api_type not in self.endpoints:
            logger.error(f"지원하지 않는 Kakao API 타입: {api_type}")
            return None
        
        params = {
            'query': query,
            'size': min(size, 50),  # 최대 50개로 제한
            'page': page,
            'sort': sort
        }
        
        try:
            response = requests.get(
                self.endpoints[api_type],
                headers=self.headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                result['api_provider'] = 'kakao'
                return result
            elif response.status_code == 429:
                logger.warning("Kakao API 호출 한도 초과, 잠시 대기...")
                time.sleep(1)
                return None
            else:
                logger.error(f"Kakao API 호출 실패: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Kakao API 요청 중 오류 발생: {e}")
            return None
        except Exception as e:
            logger.error(f"Kakao API 예상치 못한 오류: {e}")
            return None

class DataManager:
    """데이터 관리 및 저장 클래스"""
    
    def __init__(self, data_dir: str = "lignex1_data"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "lignex1_articles.db")
        self.json_dir = os.path.join(data_dir, "json_files")
        
        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        # 데이터베이스 초기화
        self._init_database()
        
        # 중복 방지용 해시 세트
        self.existing_hashes: Set[str] = self._load_existing_hashes()
        
    def _init_database(self):
        """SQLite 데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    link TEXT NOT NULL,
                    pub_date TEXT,
                    source TEXT,
                    search_keyword TEXT NOT NULL,
                    api_type TEXT NOT NULL,
                    api_provider TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    collected_at TEXT NOT NULL,
                    thumbnail TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 인덱스 생성
            conn.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON articles(content_hash)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_search_keyword ON articles(search_keyword)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_api_type ON articles(api_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_api_provider ON articles(api_provider)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_collected_at ON articles(collected_at)')
            
            conn.commit()
    
    def _load_existing_hashes(self) -> Set[str]:
        """기존 데이터의 해시값 로드"""
        hashes = set()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT content_hash FROM articles')
                hashes = {row[0] for row in cursor.fetchall()}
            logger.info(f"기존 데이터 {len(hashes)}개의 해시값 로드 완료")
        except Exception as e:
            logger.error(f"해시값 로드 중 오류: {e}")
        
        return hashes
    
    def _generate_content_hash(self, title: str, link: str, description: str = "") -> str:
        """컨텐츠 해시 생성"""
        content = f"{title}|{link}|{description}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, content_hash: str) -> bool:
        """중복 데이터 체크"""
        return content_hash in self.existing_hashes
    
    def save_article(self, article: ArticleData) -> bool:
        """기사 데이터 저장"""
        if self.is_duplicate(article.content_hash):
            return False  # 중복 데이터
        
        try:
            # 데이터베이스에 저장
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO articles 
                    (title, description, link, pub_date, source, search_keyword, 
                     api_type, api_provider, content_hash, collected_at, thumbnail)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.title, article.description, article.link, article.pub_date,
                    article.source, article.search_keyword, article.api_type,
                    article.api_provider, article.content_hash, article.collected_at,
                    article.thumbnail
                ))
                conn.commit()
            
            # JSON 파일로도 저장
            self._save_as_json(article)
            
            # 해시 세트에 추가
            self.existing_hashes.add(article.content_hash)
            
            return True
            
        except sqlite3.IntegrityError:
            # 중복 데이터 (content_hash UNIQUE 제약 조건)
            return False
        except Exception as e:
            logger.error(f"데이터 저장 중 오류: {e}")
            return False
    
    def _save_as_json(self, article: ArticleData):
        """JSON 파일로 저장"""
        try:
            # 날짜별 디렉토리 생성
            date_str = datetime.now().strftime('%Y-%m-%d')
            daily_dir = os.path.join(self.json_dir, date_str)
            os.makedirs(daily_dir, exist_ok=True)
            
            # 파일명 생성 (해시값 기반)
            filename = f"{article.content_hash}.json"
            filepath = os.path.join(daily_dir, filename)
            
            # JSON 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(article), f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"JSON 저장 중 오류: {e}")
    
    def get_statistics(self) -> Dict:
        """수집 통계 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 전체 기사 수
                cursor.execute('SELECT COUNT(*) FROM articles')
                total_count = cursor.fetchone()[0]
                
                # API 제공자별 통계
                cursor.execute('''
                    SELECT api_provider, COUNT(*) 
                    FROM articles 
                    GROUP BY api_provider
                ''')
                provider_stats = dict(cursor.fetchall())
                
                # API 타입별 통계
                cursor.execute('''
                    SELECT api_type, COUNT(*) 
                    FROM articles 
                    GROUP BY api_type
                ''')
                api_stats = dict(cursor.fetchall())
                
                # 키워드별 통계
                cursor.execute('''
                    SELECT search_keyword, COUNT(*) 
                    FROM articles 
                    GROUP BY search_keyword
                    ORDER BY COUNT(*) DESC
                    LIMIT 10
                ''')
                keyword_stats = dict(cursor.fetchall())
                
                # 최근 24시간 수집량
                yesterday = (datetime.now() - timedelta(days=1)).isoformat()
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM articles 
                    WHERE collected_at > ?
                ''', (yesterday,))
                recent_count = cursor.fetchone()[0]
                
                return {
                    'total_articles': total_count,
                    'by_provider': provider_stats,
                    'by_api_type': api_stats,
                    'by_keyword': keyword_stats,
                    'last_24h': recent_count,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"통계 조회 중 오류: {e}")
            return {}

class LIGNEX1DataCollector:
    """LIGNEX1 데이터 수집기 메인 클래스 (Naver + Kakao API)"""
    
    def __init__(self, naver_client_id: str = None, naver_client_secret: str = None, 
                 kakao_rest_api_key: str = None, data_dir: str = "lignex1_data"):
        
        # API 클라이언트 초기화
        self.naver_client = None
        self.kakao_client = None
        
        if naver_client_id and naver_client_secret:
            self.naver_client = NaverAPIClient(naver_client_id, naver_client_secret)
            logger.info("Naver API 클라이언트 초기화 완료")
        
        if kakao_rest_api_key:
            self.kakao_client = KakaoAPIClient(kakao_rest_api_key)
            logger.info("Kakao API 클라이언트 초기화 완료")
        
        if not self.naver_client and not self.kakao_client:
            logger.error("최소 하나의 API 클라이언트가 필요합니다.")
            raise ValueError("API 클라이언트가 설정되지 않았습니다.")
        
        self.data_manager = DataManager(data_dir)
        self.is_running = False
        
        # LIGNEX1 관련 검색 키워드
        self.keywords = [
            "LIGNEX1", "LIG넥스원", "엘아이지넥스원", "엘아이지넥스one",
            "LIG Nex1", "LIG-넥스원", "리그넥스원", "LIG 넥스원",
            "방위산업 LIG", "국방 LIG", "LIG그룹 넥스원"
        ]
        
        # Naver API 타입별 설정
        self.naver_api_configs = {
            'news': {'display': 100, 'sort': 'date'},
            'blog': {'display': 100, 'sort': 'date'},
            'webkr': {'display': 100, 'sort': 'date'},
            'cafearticle': {'display': 100, 'sort': 'date'}
        }
        
        # Kakao API 타입별 설정
        self.kakao_api_configs = {
            'web': {'size': 50, 'sort': 'recency'},
            'blog': {'size': 50, 'sort': 'recency'},
            'cafe': {'size': 50, 'sort': 'recency'},
            'vclip': {'size': 30, 'sort': 'recency'}  # 동영상은 적게
        }
    
    def collect_naver_data(self, keyword: str, api_type: str) -> int:
        """Naver API로 데이터 수집"""
        if not self.naver_client:
            return 0
            
        collected_count = 0
        config = self.naver_api_configs[api_type]
        
        try:
            result = self.naver_client.search(
                query=keyword,
                api_type=api_type,
                display=config['display'],
                sort=config['sort']
            )
            
            if not result or 'items' not in result:
                return 0
            
            current_time = datetime.now().isoformat()
            
            for item in result['items']:
                try:
                    # HTML 태그 제거
                    title = self._clean_html(item.get('title', ''))
                    description = self._clean_html(item.get('description', ''))
                    
                    # 컨텐츠 해시 생성
                    content_hash = self.data_manager._generate_content_hash(
                        title, item.get('link', ''), description
                    )
                    
                    # ArticleData 객체 생성
                    article = ArticleData(
                        title=title,
                        description=description,
                        link=item.get('link', ''),
                        pub_date=item.get('pubDate', ''),
                        source=item.get('bloggername', '') or item.get('cafename', '') or 'Unknown',
                        search_keyword=keyword,
                        api_type=api_type,
                        api_provider='naver',
                        content_hash=content_hash,
                        collected_at=current_time
                    )
                    
                    # 저장 시도
                    if self.data_manager.save_article(article):
                        collected_count += 1
                        logger.debug(f"[Naver] 새 데이터 저장: {title[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Naver 아이템 처리 중 오류: {e}")
                    continue
            
            # API 호출 간격 (Rate Limit 고려)
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Naver 키워드 '{keyword}' ({api_type}) 수집 중 오류: {e}")
        
        return collected_count
    
    def collect_kakao_data(self, keyword: str, api_type: str) -> int:
        """Kakao API로 데이터 수집"""
        if not self.kakao_client:
            return 0
            
        collected_count = 0
        config = self.kakao_api_configs[api_type]
        
        try:
            result = self.kakao_client.search(
                query=keyword,
                api_type=api_type,
                size=config['size'],
                sort=config['sort']
            )
            
            if not result or 'documents' not in result:
                return 0
            
            current_time = datetime.now().isoformat()
            
            for item in result['documents']:
                try:
                    # Kakao API 응답 구조에 맞게 데이터 추출
                    title = item.get('title', '')
                    description = item.get('contents', '') or item.get('content', '')
                    link = item.get('url', '')
                    pub_date = item.get('datetime', '')
                    
                    # 썸네일 정보 (이미지나 비디오의 경우)
                    thumbnail = item.get('thumbnail', '') or item.get('image_url', '')
                    
                    # 출처 정보
                    source = item.get('blogname', '') or item.get('cafename', '') or item.get('display_sitename', '') or 'Unknown'
                    
                    # HTML 태그 제거
                    title = self._clean_html(title)
                    description = self._clean_html(description)
                    
                    # 컨텐츠 해시 생성
                    content_hash = self.data_manager._generate_content_hash(
                        title, link, description
                    )
                    
                    # ArticleData 객체 생성
                    article = ArticleData(
                        title=title,
                        description=description,
                        link=link,
                        pub_date=pub_date,
                        source=source,
                        search_keyword=keyword,
                        api_type=api_type,
                        api_provider='kakao',
                        content_hash=content_hash,
                        collected_at=current_time,
                        thumbnail=thumbnail
                    )
                    
                    # 저장 시도
                    if self.data_manager.save_article(article):
                        collected_count += 1
                        logger.debug(f"[Kakao] 새 데이터 저장: {title[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Kakao 아이템 처리 중 오류: {e}")
                    continue
            
            # API 호출 간격 (Rate Limit 고려)
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Kakao 키워드 '{keyword}' ({api_type}) 수집 중 오류: {e}")
        
        return collected_count
    
    def _clean_html(self, text: str) -> str:
        """HTML 태그 제거"""
        import re
        
        if not text:
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # HTML 엔티티 디코딩
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        text = text.replace('&quot;', '"').replace('&apos;', "'")
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def collect_all_data(self) -> Dict:
        """모든 키워드와 API 타입에 대한 데이터 수집"""
        total_collected = 0
        collection_stats = {}
        
        logger.info("데이터 수집 시작...")
        start_time = time.time()
        
        # 멀티스레딩으로 병렬 수집
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            
            # Naver API 작업 추가
            if self.naver_client:
                for keyword in self.keywords:
                    for api_type in self.naver_api_configs.keys():
                        future = executor.submit(
                            self.collect_naver_data, 
                            keyword, 
                            api_type
                        )
                        futures.append((future, keyword, api_type, 'naver'))
            
            # Kakao API 작업 추가
            if self.kakao_client:
                for keyword in self.keywords:
                    for api_type in self.kakao_api_configs.keys():
                        future = executor.submit(
                            self.collect_kakao_data, 
                            keyword, 
                            api_type
                        )
                        futures.append((future, keyword, api_type, 'kakao'))
            
            # 결과 수집
            for future, keyword, api_type, provider in futures:
                try:
                    count = future.result(timeout=30)
                    key = f"{provider}_{keyword}_{api_type}"
                    collection_stats[key] = count
                    total_collected += count
                    
                    if count > 0:
                        logger.info(f"[{provider.upper()}] {keyword} ({api_type}): {count}개 수집")
                    
                except Exception as e:
                    logger.error(f"[{provider.upper()}] {keyword} ({api_type}) 수집 실패: {e}")
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"데이터 수집 완료: 총 {total_collected}개 (소요시간: {elapsed_time:.2f}초)")
        
        return {
            'total_collected': total_collected,
            'collection_stats': collection_stats,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def start_background_collection(self, interval_hours: int = 6):
        """백그라운드 데이터 수집 시작"""
        logger.info(f"백그라운드 데이터 수집 시작 (수집 간격: {interval_hours}시간)")
        
        # 스케줄 설정
        schedule.every(interval_hours).hours.do(self.collect_all_data)
        
        # 즉시 첫 수집 실행
        self.collect_all_data()
        
        self.is_running = True
        
        # 백그라운드 스레드에서 스케줄 실행
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 스케줄 체크
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        return scheduler_thread
    
    def stop_background_collection(self):
        """백그라운드 수집 중지"""
        logger.info("백그라운드 데이터 수집 중지")
        self.is_running = False
        schedule.clear()
    
    def get_collection_status(self) -> Dict:
        """수집 상태 조회"""
        stats = self.data_manager.get_statistics()
        stats['is_running'] = self.is_running
        stats['keywords'] = self.keywords
        stats['available_apis'] = []
        
        if self.naver_client:
            stats['available_apis'].append('naver')
            stats['naver_api_types'] = list(self.naver_api_configs.keys())
        
        if self.kakao_client:
            stats['available_apis'].append('kakao')
            stats['kakao_api_types'] = list(self.kakao_api_configs.keys())
        
        return stats

def load_config(config_file: str = "config.json") -> Dict:
    """설정 파일 로드"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"설정 파일 {config_file}을 찾을 수 없습니다. 기본 설정을 사용합니다.")
        # 기본 설정 생성
        default_config = {
            "naver_api": {
                "client_id": "YOUR_NAVER_CLIENT_ID",
                "client_secret": "YOUR_NAVER_CLIENT_SECRET"
            },
            "kakao_api": {
                "rest_api_key": "YOUR_KAKAO_REST_API_KEY"
            },
            "collection": {
                "data_dir": "lignex1_data",
                "interval_hours": 6,
                "enable_naver": True,
                "enable_kakao": True
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"기본 설정 파일 {config_file}을 생성했습니다. API 키를 설정해주세요.")
        return default_config


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='LIGNEX1 데이터 수집기')
    parser.add_argument('--config', default='config.json', help='설정 파일 경로')
    parser.add_argument('--once', action='store_true', help='한 번만 수집하고 종료')
    parser.add_argument('--stats', action='store_true', help='수집 통계 조회')
    parser.add_argument('--interval', type=int, default=6, help='수집 간격 (시간)')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # API 키 확인
    client_id = config.get('naver_api', {}).get('client_id')
    client_secret = config.get('naver_api', {}).get('client_secret')
    
    if not client_id or not client_secret or client_id == "YOUR_CLIENT_ID":
        logger.error("Naver API 키가 설정되지 않았습니다. config.json 파일을 확인해주세요.")
        return
    
    # 데이터 수집기 초기화
    data_dir = config.get('collection', {}).get('data_dir', 'lignex1_data')
    collector = LIGNEX1DataCollector(client_id, client_secret, data_dir)
    
    if args.stats:
        # 통계 조회
        stats = collector.get_collection_status()
        print("\n=== LIGNEX1 데이터 수집 통계 ===")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    
    if args.once:
        # 한 번만 수집
        result = collector.collect_all_data()
        print(f"수집 완료: {result['total_collected']}개")
    else:
        # 백그라운드 수집 시작
        try:
            scheduler_thread = collector.start_background_collection(args.interval)
            
            print(f"LIGNEX1 데이터 수집기가 시작되었습니다.")
            print(f"수집 간격: {args.interval}시간")
            print("종료하려면 Ctrl+C를 누르세요.")
            
            # 메인 스레드에서 대기
            while True:
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n수집기를 종료합니다...")
            collector.stop_background_collection()
            print("종료 완료.")

if __name__ == "__main__":
    main()