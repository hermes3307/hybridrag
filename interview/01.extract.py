#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGNEX1 데이터 수집 시스템 (간단 버전)
"""

import os
import json
import hashlib
import sqlite3
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import requests
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

class SimpleDataCollector:
    """간단한 LIGNEX1 데이터 수집기"""
    
    def __init__(self, naver_client_id=None, naver_client_secret=None, 
                 kakao_rest_api_key=None, data_dir="lignex1_data"):
        
        self.naver_client_id = naver_client_id
        self.naver_client_secret = naver_client_secret
        self.kakao_rest_api_key = kakao_rest_api_key
        self.data_dir = data_dir
        
        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "json_files"), exist_ok=True)
        
        # 데이터베이스 초기화
        self.db_path = os.path.join(data_dir, "lignex1_articles.db")
        self._init_database()
        
        # 검색 키워드
        self.keywords = [
            "LIGNEX1", "LIG넥스원", "엘아이지넥스원", "엘아이지넥스one",
            "LIG Nex1", "LIG-넥스원", "리그넥스원", "LIG 넥스원"
        ]
        
        logger.info(f"데이터 수집기 초기화 완료 (저장 경로: {data_dir})")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    link TEXT NOT NULL,
                    pub_date TEXT,
                    source TEXT,
                    search_keyword TEXT,
                    api_provider TEXT,
                    api_type TEXT,
                    content_hash TEXT UNIQUE,
                    collected_at TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def _generate_hash(self, title, link, description=""):
        """컨텐츠 해시 생성"""
        content = f"{title}|{link}|{description}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _clean_text(self, text):
        """HTML 태그 제거"""
        import re
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        text = text.replace('&quot;', '"').replace('&apos;', "'")
        return re.sub(r'\s+', ' ', text.strip())
    
    def search_naver(self, keyword, api_type='news'):
        """Naver API 검색"""
        if not self.naver_client_id or not self.naver_client_secret:
            return []
        
        url = f"https://openapi.naver.com/v1/search/{api_type}.json"
        headers = {
            'X-Naver-Client-Id': self.naver_client_id,
            'X-Naver-Client-Secret': self.naver_client_secret
        }
        params = {
            'query': keyword,
            'display': 100,
            'sort': 'date'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get('items', [])
            else:
                logger.error(f"Naver API 오류: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Naver API 요청 실패: {e}")
            return []
    
    def search_kakao(self, keyword, api_type='web'):
        """Kakao API 검색"""
        if not self.kakao_rest_api_key:
            return []
        
        url = f"https://dapi.kakao.com/v2/search/{api_type}"
        headers = {
            'Authorization': f'KakaoAK {self.kakao_rest_api_key}'
        }
        params = {
            'query': keyword,
            'size': 50,
            'sort': 'recency'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get('documents', [])
            else:
                logger.error(f"Kakao API 오류: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Kakao API 요청 실패: {e}")
            return []
    
    def save_article(self, article_data):
        """기사 데이터 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO articles 
                    (title, description, link, pub_date, source, search_keyword, 
                     api_provider, api_type, content_hash, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', article_data)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"데이터 저장 실패: {e}")
            return False
    
    def collect_data(self):
        """데이터 수집 실행"""
        total_collected = 0
        current_time = datetime.now().isoformat()
        
        logger.info("데이터 수집 시작...")
        
        for keyword in self.keywords:
            # Naver 뉴스 검색
            if self.naver_client_id:
                items = self.search_naver(keyword, 'news')
                for item in items:
                    title = self._clean_text(item.get('title', ''))
                    description = self._clean_text(item.get('description', ''))
                    link = item.get('link', '')
                    
                    if title and link:
                        content_hash = self._generate_hash(title, link, description)
                        article_data = (
                            title, description, link, item.get('pubDate', ''),
                            'Naver News', keyword, 'naver', 'news',
                            content_hash, current_time
                        )
                        
                        if self.save_article(article_data):
                            total_collected += 1
                
                time.sleep(0.1)  # API 호출 간격
            
            # Kakao 웹 검색
            if self.kakao_rest_api_key:
                items = self.search_kakao(keyword, 'web')
                for item in items:
                    title = self._clean_text(item.get('title', ''))
                    description = self._clean_text(item.get('contents', ''))
                    link = item.get('url', '')
                    
                    if title and link:
                        content_hash = self._generate_hash(title, link, description)
                        article_data = (
                            title, description, link, item.get('datetime', ''),
                            item.get('display_sitename', 'Kakao Web'), keyword,
                            'kakao', 'web', content_hash, current_time
                        )
                        
                        if self.save_article(article_data):
                            total_collected += 1
                
                time.sleep(0.1)  # API 호출 간격
        
        logger.info(f"데이터 수집 완료: 총 {total_collected}개")
        return total_collected
    
    def get_statistics(self):
        """수집 통계 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 전체 기사 수
                cursor.execute('SELECT COUNT(*) FROM articles')
                total = cursor.fetchone()[0]
                
                # API 제공자별 통계
                cursor.execute('''
                    SELECT api_provider, COUNT(*) 
                    FROM articles 
                    GROUP BY api_provider
                ''')
                by_provider = dict(cursor.fetchall())
                
                # 키워드별 통계
                cursor.execute('''
                    SELECT search_keyword, COUNT(*) 
                    FROM articles 
                    GROUP BY search_keyword
                    ORDER BY COUNT(*) DESC
                    LIMIT 5
                ''')
                by_keyword = dict(cursor.fetchall())
                
                return {
                    'total_articles': total,
                    'by_provider': by_provider,
                    'by_keyword': by_keyword,
                    'last_updated': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {}

def load_config(config_file="config.json"):
    """설정 파일 로드"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 기본 설정 파일 생성
        default_config = {
            "naver_api": {
                "client_id": "YOUR_NAVER_CLIENT_ID",
                "client_secret": "YOUR_NAVER_CLIENT_SECRET"
            },
            "kakao_api": {
                "rest_api_key": "YOUR_KAKAO_REST_API_KEY"
            },
            "collection": {
                "data_dir": "lignex1_data"
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        print(f"기본 설정 파일 {config_file}을 생성했습니다.")
        print("config.json 파일에 API 키를 입력해주세요.")
        return default_config

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='LIGNEX1 데이터 수집기')
    parser.add_argument('--config', default='config.json', help='설정 파일 경로')
    parser.add_argument('--once', action='store_true', help='한 번만 수집')
    parser.add_argument('--stats', action='store_true', help='통계 조회')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # API 키 확인
    naver_config = config.get('naver_api', {})
    kakao_config = config.get('kakao_api', {})
    
    naver_client_id = naver_config.get('client_id')
    naver_client_secret = naver_config.get('client_secret')
    kakao_rest_api_key = kakao_config.get('rest_api_key')
    
    # 기본값 체크
    if naver_client_id == "YOUR_NAVER_CLIENT_ID":
        naver_client_id = None
        naver_client_secret = None
    
    if kakao_rest_api_key == "YOUR_KAKAO_REST_API_KEY":
        kakao_rest_api_key = None
    
    # 최소 하나의 API 키는 필요
    if not naver_client_id and not kakao_rest_api_key:
        print("❌ API 키가 설정되지 않았습니다.")
        print("config.json 파일에 Naver 또는 Kakao API 키를 입력해주세요.")
        return
    
    # 데이터 수집기 초기화
    data_dir = config.get('collection', {}).get('data_dir', 'lignex1_data')
    collector = SimpleDataCollector(
        naver_client_id=naver_client_id,
        naver_client_secret=naver_client_secret,
        kakao_rest_api_key=kakao_rest_api_key,
        data_dir=data_dir
    )
    
    # 실행 모드에 따른 처리
    if hasattr(args, 'stats') and args.stats:
        # 통계 조회
        stats = collector.get_statistics()
        print("\n=== LIGNEX1 데이터 수집 통계 ===")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    elif hasattr(args, 'once') and args.once:
        # 한 번만 수집
        count = collector.collect_data()
        print(f"✅ 수집 완료: {count}개의 새로운 기사")
    
    else:
        # 기본 실행 (한 번 수집)
        count = collector.collect_data()
        print(f"✅ 수집 완료: {count}개의 새로운 기사")
        
        # 통계도 함께 출력
        stats = collector.get_statistics()
        print(f"📊 총 수집된 기사: {stats.get('total_articles', 0)}개")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}")
        print(f"❌ 오류 발생: {e}")