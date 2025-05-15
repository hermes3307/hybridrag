import os
import re
import json
import time
import random
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 디렉토리 설정
BASE_DIR = os.path.join(os.getcwd(), "altibase_rag")
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_html")

# 디렉토리 생성
for dir_path in [BASE_DIR, RAW_DIR, PROCESSED_DIR, VECTOR_DIR, DEBUG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

class AltibaseExplorer:
    """Altibase 케이스 스터디 탐색기 - 작동하는 케이스 찾기"""
    
    def __init__(self, base_url="https://altibase.com/kr/learn/case_list.php"):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Referer": "https://altibase.com/kr/learn/case_list.php",
        }
        self.session = requests.Session()
        
        # 결과 저장 디렉토리
        self.results_dir = os.path.join(BASE_DIR, "exploration")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def explore_categories(self):
        """카테고리 정보 탐색"""
        try:
            # 먼저 메인 목록 페이지 확인
            response = self.session.get(self.base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 카테고리 메뉴 찾기
            categories = []
            
            # 방법 1: 메뉴나 탭에서 카테고리 찾기
            for a_tag in soup.select('.tab_list a, .category a, ul.menu a, .lnb a'):
                href = a_tag.get('href', '')
                if 'cate=' in href:
                    cate_val = href.split('cate=')[1].split('&')[0] if '&' in href else href.split('cate=')[1]
                    try:
                        cate_val = int(cate_val)
                        categories.append({
                            'id': cate_val,
                            'name': a_tag.text.strip(),
                            'url': href
                        })
                    except ValueError:
                        pass
            
            # 방법 2: 현재 URL에서 카테고리 파라미터 가져오기
            if not categories:
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if 'cate=' in href:
                        try:
                            cate_val = href.split('cate=')[1].split('&')[0] if '&' in href else href.split('cate=')[1]
                            cate_val = int(cate_val)
                            if not any(cat['id'] == cate_val for cat in categories):
                                categories.append({
                                    'id': cate_val,
                                    'name': a_tag.text.strip() or f"카테고리 {cate_val}",
                                    'url': href
                                })
                        except ValueError:
                            pass
            
            # 방법 3: 없으면 범위로 추측
            if not categories:
                for i in range(1, 10):  # 1부터 9까지 가능한 카테고리
                    categories.append({
                        'id': i,
                        'name': f"카테고리 {i}",
                        'url': f"{self.base_url}?cate={i}"
                    })
            
            print(f"발견된 카테고리: {len(categories)}개")
            for cat in categories:
                print(f"카테고리 ID: {cat['id']}, 이름: {cat['name']}")
            
            return categories
            
        except Exception as e:
            print(f"카테고리 탐색 오류: {str(e)}")
            # 기본 카테고리 반환
            return [{'id': i, 'name': f"카테고리 {i}", 'url': f"{self.base_url}?cate={i}"} for i in range(1, 8)]
    

    def find_working_cases(self, cate_list=None, idx_range=(1, 500)):
        """모든 카테고리에서 작동하는 케이스 찾기"""
        if cate_list is None:
            cate_list = [cat['id'] for cat in self.explore_categories()]
        
        working_cases = []
        
        for cate in cate_list:
            print(f"\n카테고리 {cate} 탐색 중...")
            
            # 이 카테고리에서 작동하는 케이스 목록
            cat_working_cases = []
            
            # idx 자동 점프 간격 계산 (범위가 클수록 더 크게 건너뜀)
            idx_start, idx_end = idx_range
            total_range = idx_end - idx_start + 1
            
            if total_range <= 50:
                # 범위가 작으면 하나씩 탐색
                jump = 1
            elif total_range <= 100:
                jump = 5
            elif total_range <= 200:
                jump = 10
            else:
                jump = 25  # 더 작은 점프 간격 사용
            
            print(f"인덱스 범위: {idx_start}-{idx_end}, 탐색 간격: {jump}")
            
            # 처음 몇 개 URL을 표시하여 확인
            print(f"확인할 URL 예시:")
            for i, idx in enumerate(range(idx_start, min(idx_start + 3*jump, idx_end + 1), jump)):
                example_url = f"{self.base_url}?bgu=view&idx={idx}&cate={cate}"
                print(f"  URL {i+1}: {example_url}")
            
            # 진행 상황을 표시하는 tqdm 초기화
            progress_bar = tqdm(range(idx_start, idx_end + 1, jump), desc=f"빠른 탐색 (cate={cate})")
            
            for idx in progress_bar:
                url = f"{self.base_url}?bgu=view&idx={idx}&cate={cate}"
                
                # tqdm 설명 업데이트하여 현재 URL 표시
                progress_bar.set_description(f"탐색 중: idx={idx}, cate={cate}")
                
                try:
                    response = self.session.get(url, headers=self.headers, timeout=5)
                    
                    # HTTP 상태 코드 확인
                    status_code = response.status_code
                    if status_code != 200:
                        # tqdm은 너무 많은 출력을 방해하므로 200이 아닌 경우만 출력
                        print(f"URL: {url} - 상태 코드: {status_code}")
                        continue
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 제목이 있는지 확인 (여러 선택자 시도)
                    title_found = False
                    title_text = "제목 없음"
                    for selector in ['.con_tit', 'h1.title', '.subject', '.board_subject', 'h2.tit', '.board_view h1', '.view_tit']:
                        title_elem = soup.select_one(selector)
                        if title_elem and title_elem.text.strip():
                            title_found = True
                            title_text = title_elem.text.strip()
                            break
                    
                    # 본문이 있는지 확인 (여러 선택자 시도)
                    content_found = False
                    for selector in ['.view_con', '.content', '.board_content', '.board_view', 'article', '.con_txt']:
                        content_elem = soup.select_one(selector)
                        if content_elem and len(content_elem.text.strip()) > 100:
                            content_found = True
                            break
                    
                    if title_found:
                        print(f"제목 발견: {url} - {title_text[:30]}...")
                    
                    if content_found:
                        print(f"본문 발견: {url}")
                    
                    if title_found and content_found:
                        cat_working_cases.append({
                            'idx': idx,
                            'cate': cate,
                            'title': title_text,
                            'url': url
                        })
                        print(f"작동하는 케이스 발견: {url} - {title_text[:30]}...")
                        
                        # 이 idx 주변 ±10 범위 상세 검색
                        for nearby_idx in range(max(1, idx-10), idx+11):
                            if nearby_idx == idx or nearby_idx % jump == 0:  # 이미 확인했으면 건너뜀
                                continue
                                
                            nearby_url = f"{self.base_url}?bgu=view&idx={nearby_idx}&cate={cate}"
                            print(f"주변 확인: {nearby_url}")
                            
                            try:
                                nearby_response = self.session.get(nearby_url, headers=self.headers, timeout=5)
                                if nearby_response.status_code == 200:
                                    nearby_soup = BeautifulSoup(nearby_response.text, 'html.parser')
                                    
                                    # 제목 확인
                                    nearby_title_found = False
                                    nearby_title_text = "제목 없음"
                                    for selector in ['.con_tit', 'h1.title', '.subject', '.board_subject', 'h2.tit']:
                                        nearby_title_elem = nearby_soup.select_one(selector)
                                        if nearby_title_elem and nearby_title_elem.text.strip():
                                            nearby_title_found = True
                                            nearby_title_text = nearby_title_elem.text.strip()
                                            break
                                    
                                    # 본문 확인
                                    nearby_content_found = False
                                    for selector in ['.view_con', '.content', '.board_content', '.board_view']:
                                        nearby_content_elem = nearby_soup.select_one(selector)
                                        if nearby_content_elem and len(nearby_content_elem.text.strip()) > 100:
                                            nearby_content_found = True
                                            break
                                    
                                    if nearby_title_found:
                                        print(f"주변 제목 발견: {nearby_url} - {nearby_title_text[:30]}...")
                                    
                                    if nearby_content_found:
                                        print(f"주변 본문 발견: {nearby_url}")
                                    
                                    if nearby_title_found and nearby_content_found:
                                        cat_working_cases.append({
                                            'idx': nearby_idx,
                                            'cate': cate,
                                            'title': nearby_title_text,
                                            'url': nearby_url
                                        })
                                        print(f"주변 작동 케이스 발견: {nearby_url} - {nearby_title_text[:30]}...")
                                        
                            except Exception as e:
                                print(f"주변 확인 오류 {nearby_url}: {str(e)}")
                            
                            # 주변 검색 딜레이
                            time.sleep(0.2)
                
                except Exception as e:
                    print(f"오류 {url}: {str(e)}")
                
                # 서버에 부담을 주지 않기 위한 딜레이
                time.sleep(0.5)
            
            # 이 카테고리 결과 저장
            working_cases.extend(cat_working_cases)
            
            # 결과 파일 저장
            with open(os.path.join(self.results_dir, f"working_cases_cate_{cate}.json"), 'w', encoding='utf-8') as f:
                json.dump(cat_working_cases, f, ensure_ascii=False, indent=2)
            
            print(f"카테고리 {cate}에서 {len(cat_working_cases)}개의 작동하는 케이스 발견")
        
        # 전체 결과 저장
        with open(os.path.join(self.results_dir, "all_working_cases.json"), 'w', encoding='utf-8') as f:
            json.dump(working_cases, f, ensure_ascii=False, indent=2)
            
        print(f"\n총 {len(working_cases)}개의 작동하는 케이스 발견")
        return working_cases

    def get_working_ranges(self):
        """작동하는 케이스를 기반으로 효율적인 스크래핑 범위 추천"""
        working_ranges = []
        
        # 저장된 모든 작동하는 케이스 파일 읽기
        try:
            all_cases_path = os.path.join(self.results_dir, "all_working_cases.json")
            
            if os.path.exists(all_cases_path):
                with open(all_cases_path, 'r', encoding='utf-8') as f:
                    all_cases = json.load(f)
                
                # 카테고리별로 그룹화
                by_category = {}
                for case in all_cases:
                    cate = case['cate']
                    if cate not in by_category:
                        by_category[cate] = []
                    by_category[cate].append(case['idx'])
                
                # 각 카테고리별로 연속된 범위 찾기
                for cate, idxs in by_category.items():
                    idxs = sorted(idxs)
                    
                    # 연속된 범위 찾기
                    ranges = []
                    start = idxs[0]
                    prev = start
                    
                    for idx in idxs[1:]:
                        if idx > prev + 20:  # 20 이상 간격이 있으면 새 범위 시작
                            ranges.append((start, prev))
                            start = idx
                        prev = idx
                    
                    # 마지막 범위 추가
                    ranges.append((start, prev))
                    
                    # 효율적인 스크래핑을 위해 범위 약간 확장
                    for start_idx, end_idx in ranges:
                        working_ranges.append({
                            'cate': cate,
                            'start_idx': max(1, start_idx - 5),
                            'end_idx': end_idx + 5
                        })
                
                return working_ranges
            else:
                print("작동하는 케이스 파일을 찾을 수 없습니다.")
                return [{'cate': 3, 'start_idx': 300, 'end_idx': 330}]  # 예제로 작동하는 범위
        
        except Exception as e:
            print(f"작동하는 범위 분석 오류: {str(e)}")
            return [{'cate': 3, 'start_idx': 300, 'end_idx': 330}]  # 예제로 작동하는 범위

class AltibaseScraper:
    """Altibase 케이스 스터디 웹페이지 스크래핑 클래스"""
    
    def __init__(self, base_url="https://altibase.com/kr/learn/case_list.php"):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        self.session = requests.Session()
    
    def fetch_page(self, idx, cate):
        """특정 idx와 cate 값으로 페이지 가져오기"""
        url = f"{self.base_url}?bgu=view&idx={idx}&cate={cate}"
        print(f"페이지 가져오기: {url}")
        
        try:
            # 임의의 User-Agent 변경
            import random
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
            ]
            self.headers["User-Agent"] = random.choice(user_agents)
            
            response = self.session.get(url, headers=self.headers, timeout=10)
            status_code = response.status_code
            print(f"응답 상태 코드: {status_code}")
            
            response.raise_for_status()  # HTTP 오류 발생시 예외 발생
            
            # HTML 내용 저장 (디버깅용)
            debug_file = os.path.join(DEBUG_DIR, f"page_{idx}_{cate}.html")
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # 올바른 페이지인지 확인 (다양한 선택자로 시도)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 여러 가능한 제목 선택자 시도
            title = None
            for selector in ['.con_tit', 'h1.title', '.subject', '.board_subject', 'h2.tit']:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.text.strip():
                    title = title_elem
                    print(f"제목 발견 (선택자: {selector}): {title_elem.text.strip()[:30]}...")
                    break
            
            # 여러 가능한 본문 선택자 시도
            content = None
            for selector in ['.view_con', '.content', '.board_content', '.board_view', 'article']:
                content_elem = soup.select_one(selector)
                if content_elem and len(content_elem.text.strip()) > 100:  # 최소 길이 확인
                    content = content_elem
                    content_preview = content_elem.text.strip()[:50].replace('\n', ' ')
                    print(f"본문 발견 (선택자: {selector}): {content_preview}...")
                    break
            
            if title and content:
                print(f"페이지 가져오기 성공: {url}")
                return response.text
            else:
                # 제목이나 본문 중 하나라도 찾지 못한 경우
                if title:
                    print(f"제목은 발견했으나 본문을 찾지 못함: {url}")
                elif content:
                    print(f"본문은 발견했으나 제목을 찾지 못함: {url}")
                else:
                    print(f"제목과 본문 모두 찾지 못함: {url}")
                return None
                
        except requests.RequestException as e:
            print(f"요청 오류 {url}: {str(e)}")
            return None
        
    def extract_content(self, html_content):
        """HTML에서 케이스 스터디 내용 추출"""
        if not html_content:
            return None
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 제목 추출 (여러 선택자 시도)
        title = None
        for selector in ['.con_tit', 'h1.title', '.subject', '.board_subject', 'h2.tit', '.board_view h1', '.view_tit']:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.text.strip():
                title = title_elem.text.strip()
                break
        
        if not title:
            # 선택자가 실패한 경우 제목으로 보이는 첫 번째 큰 텍스트 찾기
            for tag in soup.find_all(['h1', 'h2', 'h3']):
                if tag.text.strip() and len(tag.text.strip()) > 5:
                    title = tag.text.strip()
                    break
        
        # 날짜 추출 (여러 선택자 시도)
        date = None
        for selector in ['.txt_date', '.date', '.board_date', '.info_txt span']:
            date_elem = soup.select_one(selector)
            if date_elem and date_elem.text.strip():
                date = date_elem.text.strip()
                break
        
        # 날짜 형식 정규식으로 찾기 (날짜를 못 찾은 경우)
        if not date:
            date_pattern = re.compile(r'\d{4}[년\./-]?\s*\d{1,2}[월\./-]?\s*\d{1,2}[일]?')
            for tag in soup.find_all(['span', 'div', 'p']):
                if tag.text and date_pattern.search(tag.text):
                    date = date_pattern.search(tag.text).group()
                    break
        
        # 본문 컨테이너 추출 (여러 선택자 시도)
        content_text = ""
        for selector in ['.view_con', '.content', '.board_content', '.board_view', 'article', '.con_txt']:
            content_container = soup.select_one(selector)
            if content_container and len(content_container.text.strip()) > 100:  # 최소 길이 확인
                # 이미지 대체 텍스트 추가
                for img in content_container.find_all('img'):
                    alt_text = img.get('alt', '이미지')
                    img.replace_with(f"[이미지: {alt_text}]")
                
                # 불필요한 스크립트, 스타일 제거
                for tag in content_container.find_all(['script', 'style']):
                    tag.decompose()
                
                # 본문 추출 및 정리
                content_text = content_container.get_text(separator='\n').strip()
                content_text = re.sub(r'\n{3,}', '\n\n', content_text)  # 과도한 줄바꿈 정리
                break
        
        # 본문을 찾지 못한 경우, 본문으로 보이는 큰 텍스트 블록 찾기
        if not content_text:
            main_content_blocks = []
            for tag in soup.find_all(['div', 'article', 'section']):
                if len(tag.text.strip()) > 200:  # 일정 길이 이상의 텍스트 블록
                    main_content_blocks.append((tag, len(tag.text.strip())))
            
            # 가장 긴 텍스트 블록 선택
            if main_content_blocks:
                main_content_blocks.sort(key=lambda x: x[1], reverse=True)
                main_block = main_content_blocks[0][0]
                
                # 불필요한 스크립트, 스타일 제거
                for tag in main_block.find_all(['script', 'style']):
                    tag.decompose()
                
                content_text = main_block.get_text(separator='\n').strip()
                content_text = re.sub(r'\n{3,}', '\n\n', content_text)  # 과도한 줄바꿈 정리
        
        # 최소한의 정보를 얻었는지 확인
        if not title:
            title = "제목 없음"
        
        if not date:
            date = "날짜 없음"
        
        if not content_text:
            return None  # 본문 내용이 없으면 None 반환
        
        return {
            "title": title,
            "date": date,
            "content": content_text
        }
    
    def scan_and_save(self, idx_range=(235, 324), cate_range=(3, 3), delay=0):
        """지정된 범위의 idx와 cate 값으로 스캔하고 결과 저장"""
        total_found = 0
        
        for cate in range(cate_range[0], cate_range[1] + 1):
            for idx in tqdm(range(idx_range[0], idx_range[1] + 1), desc=f"Scanning cate={cate}"):
                # 이미 저장된 파일이 있는지 확인
                file_path = os.path.join(RAW_DIR, f"altibase_case_{idx}_{cate}.json")
                if os.path.exists(file_path):
                    logger.info(f"File already exists: {file_path}, skipping...")
                    total_found += 1
                    continue
                
                # 페이지 가져오기
                html_content = self.fetch_page(idx, cate)
                
                if html_content:
                    # 내용 추출
                    data = self.extract_content(html_content)
                    
                    if data and len(data["content"]) > 50:  # 내용이 충분히 있는 경우만 저장
                        # 메타데이터 추가
                        data["idx"] = idx
                        data["cate"] = cate
                        data["url"] = f"{self.base_url}?bgu=view&idx={idx}&cate={cate}"
                        
                        # JSON 파일로 저장
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        
                        total_found += 1
                        logger.info(f"Saved content to {file_path}")
                    else:
                        logger.warning(f"Content extraction failed or content too short: idx={idx}, cate={cate}")
                
                # 서버에 부담을 주지 않기 위한 딜레이
                time.sleep(delay)
                
                # 랜덤 딜레이 추가 (크롤링 감지 회피)
                if idx % 10 == 0 and idx > 0:
                    rand_delay = random.uniform(1.0, 3.0)
                    logger.info(f"Random delay: {rand_delay:.2f}s")
                    time.sleep(rand_delay)
        
        logger.info(f"Total case studies found and saved: {total_found}")
        return total_found


class RAGProcessor:
    """RAG 시스템을 위한 텍스트 처리 및 벡터화 클래스"""
    
    def __init__(self, embedding_model_name="jhgan/ko-sroberta-multitask"):
        # 최신 langchain_huggingface 패키지 사용 시도
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            # 기존 방식으로 폴백
            logger.warning("langchain_huggingface package not found, using deprecated import path")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        
    def process_documents(self):
        """수집된 문서 처리 및 청크로 분할"""
        all_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
        
        if not all_files:
            logger.warning("No documents found in RAW_DIR. Please run the scraper first.")
            return []
            
        all_documents = []
        
        for file_name in tqdm(all_files, desc="Processing documents"):
            file_path = os.path.join(RAW_DIR, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 제목과 본문 결합
                full_text = f"제목: {data['title']}\n\n{data['content']}"
                
                # 메타데이터 준비
                metadata = {
                    "title": data["title"],
                    "date": data["date"],
                    "idx": data.get("idx", 0),
                    "cate": data.get("cate", 0),
                    "url": data.get("url", ""),
                    "source": file_name
                }
                
                # 텍스트 분할
                chunks = self.text_splitter.split_text(full_text)
                
                # 각 청크에 메타데이터 추가
                for i, chunk in enumerate(chunks):
                    doc_with_metadata = {
                        "text": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_id": i,
                            "chunk_count": len(chunks)
                        }
                    }
                    all_documents.append(doc_with_metadata)
                    
                # 프로세
                # 프로세스된 문서 저장
                processed_file_path = os.path.join(PROCESSED_DIR, file_name.replace('.json', '_processed.json'))
                with open(processed_file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "metadata": metadata,
                        "chunks": chunks
                    }, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Processed {len(all_documents)} chunks from {len(all_files)} documents")
        return all_documents
    
    def create_vector_store(self, documents):
        """문서로부터 벡터 스토어 생성"""
        logger.info("Creating vector store...")
        
        # 문서가 비어있는지 확인
        if not documents:
            logger.error("No documents to vectorize. Please scrape and process some data first.")
            print("오류: 벡터화할 문서가 없습니다. 먼저 데이터를 스크래핑하세요.")
            return None
        
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # FAISS 벡터 스토어 생성
        try:
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # 벡터 스토어 저장
            vector_store.save_local(VECTOR_DIR)
            logger.info(f"Vector store saved to {VECTOR_DIR}")
            
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            print(f"벡터 스토어 생성 중 오류 발생: {str(e)}")
            return None
    
    def load_vector_store(self):
        """저장된 벡터 스토어 로드"""
        logger.info("Loading vector store...")
        
        if not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
            logger.error("Vector store not found. Please create it first.")
            return None
            
        try:
            vector_store = FAISS.load_local(
                VECTOR_DIR,
                self.embeddings
            )
            
            logger.info("Vector store loaded successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            print(f"벡터 스토어 로드 중 오류 발생: {str(e)}")
            return None


class GemmaLLM:
    """로컬 Gemma LLM 래퍼 클래스"""
    
    def __init__(self, model_id="google/gemma-3-8b", quantize=True):
        self.model_id = model_id
        self.quantize = quantize
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.llm = None
        
        self._load_model()
    
    def _load_model(self):
        """Gemma 모델 로드"""
        logger.info(f"Loading Gemma model: {self.model_id}")
        
        try:
            # 장치 설정
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # GPU 메모리 확인 (CUDA 사용 시)
            if device == "cuda":
                try:
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    free_mem_gb = free_mem / (1024**3)
                    total_mem_gb = total_mem / (1024**3)
                    logger.info(f"GPU memory: {free_mem_gb:.2f}GB free of {total_mem_gb:.2f}GB total")
                    
                    if free_mem_gb < 5 and not self.quantize:
                        logger.warning("Less than 5GB GPU memory available. Enabling quantization.")
                        print("경고: GPU 메모리가 부족합니다. 자동으로 양자화를 활성화합니다.")
                        self.quantize = True
                except Exception as e:
                    logger.warning(f"Could not check GPU memory: {str(e)}")
            
            # 양자화 설정 (메모리 사용량 감소)
            if self.quantize and device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print("4비트 양자화 활성화: GPU 메모리 사용량을 줄입니다.")
            else:
                quantization_config = None
                
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                quantization_config=quantization_config,
                trust_remote_code=True,
                use_safetensors=True,
            )
            
            # 텍스트 생성 파이프라인 생성
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=1024,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1,
            )
            
            # LangChain LLM 래퍼 생성
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            logger.info("Gemma model loaded successfully")
            
            return self.llm
        
        except Exception as e:
            logger.error(f"Error loading Gemma model: {str(e)}")
            print(f"Gemma 모델 로드 중 오류 발생: {str(e)}")
            print("\n모델 로드 관련 문제 해결 방법:")
            print("1. Hugging Face에서 모델 접근 권한이 있는지 확인하세요.")
            print("2. 'huggingface-cli login' 명령으로 로그인이 되어 있는지 확인하세요.")
            print("3. 인터넷 연결을 확인하세요.")
            print("4. 더 작은 모델(예: gemma-2b)로 시도해보세요.")
            return None


class GemmaRAG:
    """Gemma 3 모델을 이용한 RAG 시스템"""
    
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        
        # RAG 프롬프트 템플릿
        self.template = """
        <context>
        {context}
        </context>

        당신은 Altibase 케이스 스터디 전문가입니다. 위의 컨텍스트 정보를 바탕으로 아래 질문에 답변해주세요.
        답변을 할 때는 컨텍스트에 있는 정보만 사용하고, 컨텍스트에 없는 내용은 '제공된 정보만으로는 알 수 없습니다'라고 명시해주세요.
        출처 인용 시 제목과 URL을 포함해 주세요.

        질문: {question}

        답변:
        """
        
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        
        # RAG 체인 구성
        self._initialize_rag_chain()
    
    def _initialize_rag_chain(self):
        """RAG 체인 초기화"""
        # 검색기 설정
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 검색 결과 포맷팅 함수
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # RAG 체인 구성
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def answer_question(self, question):
        """질문에 대한 답변 생성"""
        logger.info(f"Answering question: {question}")
        
        try:
            result = self.rag_chain.invoke(question)
            return result
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"오류가 발생했습니다: {str(e)}"
    
    def interactive_qa(self):
        """대화형 Q&A 세션"""
        print("\n===== Altibase 케이스 스터디 Q&A 시스템 =====")
        print("질문을 입력하세요. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
        
        while True:
            question = input("\n질문: ")
            
            if question.lower() in ['exit', 'quit', '종료']:
                print("프로그램을 종료합니다.")
                break
                
            if not question.strip():
                continue
                
            print("\n답변 생성 중...")
            answer = self.answer_question(question)
            print(f"\n답변: {answer}")


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("Gemma 3 기반 로컬 RAG 시스템 (Altibase 케이스 스터디)")
    print("=" * 50)
    
    # 단계 선택
    print("\n실행할 단계를 선택하세요:")
    print("1. 작동하는 케이스 스터디 탐색")
    print("2. 케이스 스터디 스크래핑")
    print("3. 문서 처리 및 벡터화")
    print("4. Gemma 3 모델 로드 및 RAG 실행")
    print("5. 전체 파이프라인 실행")
    
    choice = input("\n선택 (기본: 5): ") or "5"
    
    # 1. 작동하는 케이스 스터디 탐색
    working_ranges = []
    if choice in ["1", "5"]:
        print("\n작동하는 케이스 스터디를 탐색합니다...")
        explorer = AltibaseExplorer()
        
        # 이미 탐색 결과가 있는지 확인
        results_dir = os.path.join(BASE_DIR, "exploration")
        all_cases_path = os.path.join(results_dir, "all_working_cases.json")
        
        if os.path.exists(all_cases_path):
            print(f"이미 탐색 결과가 있습니다: {all_cases_path}")
            use_existing = input("기존 탐색 결과를 사용하시겠습니까? (y/n, 기본: y): ").lower() != 'n'
            
            if use_existing:
                working_ranges = explorer.get_working_ranges()
            else:
                # 탐색 범위 설정
                cate_start = int(input("탐색할 시작 카테고리 (기본: 1): ") or "1")
                cate_end = int(input("탐색할 종료 카테고리 (기본: 7): ") or "7")
                idx_start = int(input("탐색할 시작 인덱스 (기본: 1): ") or "1")
                idx_end = int(input("탐색할 종료 인덱스 (기본: 500): ") or "500")
                
                # 탐색 실행
                explorer.find_working_cases(
                    cate_list=list(range(cate_start, cate_end + 1)),
                    idx_range=(idx_start, idx_end)
                )
                working_ranges = explorer.get_working_ranges()
        else:
            # 탐색 범위 설정
            cate_start = int(input("탐색할 시작 카테고리 (기본: 1): ") or "1")
            cate_end = int(input("탐색할 종료 카테고리 (기본: 7): ") or "7")
            idx_start = int(input("탐색할 시작 인덱스 (기본: 1): ") or "1")
            idx_end = int(input("탐색할 종료 인덱스 (기본: 500): ") or "500")
            
            # 탐색 실행
            explorer.find_working_cases(
                cate_list=list(range(cate_start, cate_end + 1)),
                idx_range=(idx_start, idx_end)
            )
            working_ranges = explorer.get_working_ranges()
        
        # 작동하는 범위 요약
        print("\n작동하는 범위 요약:")
        for i, range_info in enumerate(working_ranges):
            print(f"{i+1}. 카테고리 {range_info['cate']}, 인덱스 {range_info['start_idx']}-{range_info['end_idx']}")
    
    # 작동하는 범위가 없으면 기본값 설정
    if not working_ranges:
        working_ranges = [{'cate': 3, 'start_idx': 300, 'end_idx': 330}]
        print("\n작동하는 범위를 찾지 못했습니다. 기본값을 사용합니다:")
        print(f"카테고리 3, 인덱스 300-330")
    
    # 2. 케이스 스터디 스크래핑
    if choice in ["2", "5"]:
        print("\n케이스 스터디 스크래핑을 시작합니다...")
        scraper = AltibaseScraper()
        
        # RAW_DIR에 파일이 있는지 확인
        existing_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
        if existing_files:
            print(f"이미 {len(existing_files)}개의 스크래핑된 파일이 있습니다.")
            scrape_new = input("새로운 데이터를 스크래핑 하시겠습니까? (y/n, 기본: n): ").lower() == 'y'
        else:
            scrape_new = True
        
        if scrape_new:
            total_found = 0
            for range_info in working_ranges:
                cate = range_info['cate']
                start_idx = range_info['start_idx']
                end_idx = range_info['end_idx']
                
                print(f"\n카테고리 {cate}, 인덱스 범위 {start_idx}-{end_idx} 스크래핑 중...")
                found = scraper.scan_and_save(
                    idx_range=(start_idx, end_idx),
                    cate_range=(cate, cate),
                    delay=1.0
                )
                total_found += found
                
            print(f"\n총 {total_found}개의 케이스 스터디를 스크래핑했습니다.")
    
    # 3. 문서 처리 및 벡터화
    vector_store = None
    if choice in ["3", "5"]:
        print("\n문서 처리 및 벡터화를 시작합니다...")
        processor = RAGProcessor()
        
        # RAW_DIR에 파일이 있는지 확인
        raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
        if not raw_files:
            print("오류: 처리할 문서가 없습니다. 스크래핑을 먼저 실행하세요.")
            return
        
        print(f"\n{len(raw_files)}개의 문서를 처리하고 벡터화합니다...")
        
        # 기존 벡터 스토어 확인
        if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
            create_new = input("기존 벡터 스토어가 있습니다. 새로 생성하시겠습니까? (y/n, 기본: n): ").lower() == 'y'
            
            if create_new:
                documents = processor.process_documents()
                if documents:  # 문서가 있는 경우에만 벡터 스토어 생성
                    vector_store = processor.create_vector_store(documents)
                else:
                    print("오류: 처리된 문서가 없습니다.")
                    return
            else:
                vector_store = processor.load_vector_store()
        else:
            documents = processor.process_documents()
            if documents:  # 문서가 있는 경우에만 벡터 스토어 생성
                vector_store = processor.create_vector_store(documents)
            else:
                print("오류: 처리된 문서가 없습니다.")
                return
        
        # 벡터 스토어 생성 실패 시 종료
        if vector_store is None:
            print("벡터 스토어 생성 또는 로드에 실패했습니다. 프로그램을 종료합니다.")
            return
    
    # 4. Gemma 3 모델 로드 및 RAG 실행
    if choice in ["4", "5"]:
        # 벡터 스토어 로드 (필요한 경우)
        if vector_store is None:
            print("\n벡터 스토어를 로드합니다...")
            processor = RAGProcessor()
            vector_store = processor.load_vector_store()
            
            if vector_store is None:
                print("벡터 스토어 로드에 실패했습니다. 프로그램을 종료합니다.")
                return
        
        print("\nGemma 3 모델을 로드합니다...")
        try:
            # 모델 선택 및 설정
            print("\n사용 가능한 Gemma 모델:")
            print("1. google/gemma-3-8b (8B 파라미터, 더 정확함)")
            print("2. google/gemma-3-2b (2B 파라미터, 더 빠름)")
            print("3. google/gemma-2-8b (이전 버전)")
            print("4. google/gemma-2-2b (이전 버전)")
            print("5. 사용자 지정 모델")
            
            model_choice = input("\n모델 선택 (기본: 1): ") or "1"
            
            if model_choice == "1":
                gemma_model_id = "google/gemma-3-8b"
            elif model_choice == "2":
                gemma_model_id = "google/gemma-3-2b"
            elif model_choice == "3":
                gemma_model_id = "google/gemma-2-8b"
            elif model_choice == "4":
                gemma_model_id = "google/gemma-2-2b"
            elif model_choice == "5":
                gemma_model_id = input("모델 ID를 입력하세요: ")
            else:
                gemma_model_id = "google/gemma-3-8b"
            
            # 양자화 설정
            print("\n모델 양자화 설정:")
            print("1. 4비트 양자화 사용 (메모리 사용량 감소, 추천)")
            print("2. 양자화 없음 (더 정확하지만 더 많은 메모리 필요)")
            
            quant_choice = input("\n양자화 설정 선택 (기본: 1): ") or "1"
            use_quantize = quant_choice == "1"
            
            # 모델 로드
            gemma_wrapper = GemmaLLM(model_id=gemma_model_id, quantize=use_quantize)
            
            # 모델 로드 실패 시 종료
            if gemma_wrapper.llm is None:
                print("Gemma 모델 로드에 실패했습니다. 프로그램을 종료합니다.")
                return
            
            # RAG 시스템 초기화 및 실행
            print("\nGemma 3 RAG 시스템을 초기화합니다...")
            rag_system = GemmaRAG(llm=gemma_wrapper.llm, vector_store=vector_store)
            rag_system.interactive_qa()
            
        except Exception as e:
            logger.error(f"Error in RAG system: {str(e)}")
            print(f"RAG 시스템 실행 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    main()