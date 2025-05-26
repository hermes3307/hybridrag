import requests
import os
import time
import sys
import argparse
import re
from urllib.parse import urlparse, unquote
import json
from bs4 import BeautifulSoup


def extract_from_github_url(url, output_file=None):
    """
    GitHub URL에서 매뉴얼 내용을 추출하는 함수
    
    Args:
        url (str): 가져올 GitHub URL
        output_file (str): 출력 파일 경로 (선택 사항)
        
    Returns:
        str: 추출된 내용 또는 실패 시 None
    """
    try:
        # GitHub API를 사용하기 위한 URL 변환
        # 예: https://github.com/ALTIBASE/Documents/blob/master/Manuals/Altibase_7.3/kor/API User's Manual.md
        # -> https://raw.githubusercontent.com/ALTIBASE/Documents/master/Manuals/Altibase_7.3/kor/API User's Manual.md
        
        # URL 파싱
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        
        # 경로에서 'blob/' 제거하고 원시 콘텐츠 URL 생성
        if 'blob' in path_parts:
            blob_index = path_parts.index('blob')
            path_parts.pop(blob_index)
            raw_path = '/'.join(path_parts)
            raw_url = f"https://raw.githubusercontent.com{raw_path}"
        else:
            raw_url = url
            
        print(f"원시 콘텐츠 URL: {raw_url}")
        
        # UTF-8 지원을 표시하는 헤더가 있는 요청 전송
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Charset': 'UTF-8',
        }
        response = requests.get(raw_url, headers=headers)
        response.encoding = 'utf-8'  # UTF-8 인코딩 강제 적용
        
        if response.status_code != 200:
            print(f"오류: HTTP 상태 코드 {response.status_code}")
            return None
            
        # 마크다운 콘텐츠 추출
        markdown_content = response.text
        
        # 구조화된 콘텐츠로 변환
        structured_content = extract_from_markdown(markdown_content, url)
        
        # output_file이 지정된 경우 추출된 내용을 파일에 저장
        if output_file and structured_content:
            # 디렉토리가 존재하는지 확인
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            
            # UTF-8 인코딩으로 쓰기
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(structured_content)
            print(f"내용이 추출되어 {output_file}에 저장되었습니다.")
        
        return structured_content
        
    except Exception as e:
        print(f"URL 가져오기 오류: {e}")
        return None


def extract_from_markdown(markdown_content, source_url=""):
    """
    마크다운 콘텐츠에서 매뉴얼 내용과 목차를 추출하는 함수
    
    Args:
        markdown_content (str): 마크다운 콘텐츠
        source_url (str): 소스 URL
        
    Returns:
        str: 추출된 내용 또는 실패 시 None
    """
    try:
        # 마크다운 콘텐츠가 비어 있는지 확인
        if not markdown_content or markdown_content.strip() == "":
            print("마크다운 콘텐츠가 비어 있습니다.")
            return None
            
        # 제목 추출 (첫 번째 # 헤더)
        title_match = re.search(r'^#\s+(.+)$', markdown_content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "제목 없음"
        
        # 파일 이름에서 제목 추출 (제목을 찾지 못한 경우)
        if title == "제목 없음" and source_url:
            url_parts = unquote(urlparse(source_url).path).split('/')
            for part in reversed(url_parts):
                if part.endswith('.md'):
                    title = part[:-3].strip()  # .md 확장자 제거
                    break
        
        # 목차 섹션 추출
        toc_match = re.search(r'(?:서문\s*)?[\*\#]?\s*목차[^\n]*\s*((?:[^\n]+\n)+?)(?:[\*\#]|\n\n|$)', markdown_content, re.MULTILINE)
        toc_text = ""
        toc_structure = {}
        
        if toc_match:
            toc_text = toc_match.group(1).strip()
            
            # 목차 구조 파싱
            toc_structure = parse_toc(toc_text)
        
        # 목차가 없거나 파싱에 실패한 경우, 헤더를 사용하여 목차 생성
        if not toc_structure:
            print("목차를 추출할 수 없어 헤더를 사용하여 구조를 파싱합니다.")
            toc_structure = parse_headers(markdown_content)
        
        # 섹션별 내용 추출
        sections_content = extract_sections(markdown_content, toc_structure)
        
        # 구조화된 내용 조합
        structured_content = {
            "title": title,
            "toc": toc_structure,
            "sections": sections_content
        }
        
        # JSON 형식으로 변환
        return json.dumps(structured_content, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"마크다운 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_toc(toc_text):
    """
    목차 텍스트를 파싱하여 구조화된 목차를 반환
    
    Args:
        toc_text (str): 목차 텍스트
        
    Returns:
        dict: 구조화된 목차
    """
    toc_structure = []
    current_level = None
    parent_stack = [toc_structure]
    
    # 목차 행 순회
    for line in toc_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # 들여쓰기 수준 또는 * 수준 감지
        indent_match = re.match(r'^(\s*)[\*\-]\s*(.+)$', line)
        if indent_match:
            indent = indent_match.group(1)
            content = indent_match.group(2).strip()
            level = len(indent) // 3  # 들여쓰기 3칸 = 1 레벨
            
            # 레벨 계산 (공백이 없는 경우는 0 레벨)
            if not indent:
                level = 0
                
            # 챕터 번호와 제목 분리
            chapter_match = re.match(r'^(\d+(?:\.\d+)*)[\.:]?\s*(.+)$', content)
            if chapter_match:
                chapter_num = chapter_match.group(1)
                title = chapter_match.group(2).strip()
            else:
                chapter_num = ""
                title = content
                
            # 현재 항목 생성
            item = {
                "level": level,
                "chapter": chapter_num,
                "title": title,
                "children": []
            }
            
            # 적절한 부모 스택 업데이트
            if current_level is None or level == current_level:
                # 같은 레벨: 현재 부모의 자식으로 추가
                parent_stack[-1].append(item)
            elif level > current_level:
                # 더 깊은 레벨: 마지막 항목이 새 부모가 됨
                parent = parent_stack[-1][-1] if parent_stack[-1] else {"children": []}
                parent_stack.append(parent["children"])
                parent_stack[-1].append(item)
            else:
                # 더 얕은 레벨: 스택에서 적절한 수준으로 돌아감
                while len(parent_stack) > level + 1:
                    parent_stack.pop()
                parent_stack[-1].append(item)
            
            current_level = level
    
    return toc_structure


def parse_headers(markdown_content):
    """
    마크다운 헤더를 파싱하여 목차 구조를 생성
    
    Args:
        markdown_content (str): 마크다운 콘텐츠
        
    Returns:
        dict: 구조화된 목차
    """
    toc_structure = []
    header_pattern = re.compile(r'^(#{1,6})\s+(.+?)(?:\s+#{1,6})?$', re.MULTILINE)
    
    for match in header_pattern.finditer(markdown_content):
        level = len(match.group(1)) - 1  # # = 0, ## = 1, ### = 2, ...
        title = match.group(2).strip()
        
        # 챕터 번호와 제목 분리
        chapter_match = re.match(r'^(\d+(?:\.\d+)*)[\.:]?\s*(.+)$', title)
        if chapter_match:
            chapter_num = chapter_match.group(1)
            title_text = chapter_match.group(2).strip()
        else:
            chapter_num = ""
            title_text = title
            
        # 현재 항목 생성
        item = {
            "level": level,
            "chapter": chapter_num,
            "title": title_text,
            "children": []
        }
        
        # 목차에 추가 (간단한 플랫 구조로)
        toc_structure.append(item)
    
    return toc_structure


def extract_sections(markdown_content, toc_structure):
    """
    목차 구조에 따라 각 섹션의 내용을 추출
    
    Args:
        markdown_content (str): 마크다운 콘텐츠
        toc_structure (dict): 목차 구조
        
    Returns:
        dict: 섹션별 내용
    """
    sections_content = {}
    
    # 목차 항목을 플랫하게 만들기
    flat_toc = flatten_toc(toc_structure)
    
    # 각 목차 항목에 대한 섹션 추출
    for i, item in enumerate(flat_toc):
        title = item["title"]
        chapter = item["chapter"]
        level = item["level"]
        
        # 헤더 패턴 생성 (챕터 번호 포함 또는 제외)
        header_patterns = []
        
        # 챕터 번호가 있는 경우
        if chapter:
            header_patterns.append(f"{'#' * (level + 1)} {chapter}[\.:]? {re.escape(title)}")
            header_patterns.append(f"{'#' * (level + 1)} {chapter}[\.:]?{re.escape(title)}")
        
        # 챕터 번호가 없는 경우 또는 추가 패턴
        header_patterns.append(f"{'#' * (level + 1)} {re.escape(title)}")
        
        # 현재 섹션의 시작 위치 찾기
        section_start = None
        for pattern in header_patterns:
            matches = list(re.finditer(pattern, markdown_content, re.MULTILINE | re.IGNORECASE))
            if matches:
                section_start = matches[0].start()
                break
        
        if section_start is None:
            continue
        
        # 다음 섹션의 시작 위치 찾기 (다음 항목이 있는 경우)
        section_end = len(markdown_content)
        if i < len(flat_toc) - 1:
            next_item = flat_toc[i + 1]
            next_title = next_item["title"]
            next_chapter = next_item["chapter"]
            next_level = next_item["level"]
            
            # 다음 헤더 패턴 생성
            next_header_patterns = []
            
            # 챕터 번호가 있는 경우
            if next_chapter:
                next_header_patterns.append(f"{'#' * (next_level + 1)} {next_chapter}[\.:]? {re.escape(next_title)}")
                next_header_patterns.append(f"{'#' * (next_level + 1)} {next_chapter}[\.:]?{re.escape(next_title)}")
            
            # 챕터 번호가 없는 경우 또는 추가 패턴
            next_header_patterns.append(f"{'#' * (next_level + 1)} {re.escape(next_title)}")
            
            # 다음 섹션의 시작 위치 찾기
            for pattern in next_header_patterns:
                matches = list(re.finditer(pattern, markdown_content, re.MULTILINE | re.IGNORECASE))
                if matches:
                    section_end = matches[0].start()
                    break
        
        # 섹션 내용 추출
        section_content = markdown_content[section_start:section_end].strip()
        
        # 키 생성 (챕터 번호가 있으면 사용, 없으면 제목 사용)
        key = f"{chapter}_{title}" if chapter else title
        key = re.sub(r'[^\w가-힣]', '_', key)  # 특수 문자 제거
        
        sections_content[key] = section_content
    
    return sections_content


def flatten_toc(toc_structure, level=0):
    """
    중첩된 목차 구조를 평탄화하여 리스트로 반환
    
    Args:
        toc_structure (list): 목차 구조
        level (int): 현재 레벨
        
    Returns:
        list: 평탄화된 목차 항목 리스트
    """
    flat_list = []
    
    for item in toc_structure:
        # 현재 항목 추가
        item_copy = item.copy()
        item_copy["level"] = level
        flat_list.append(item_copy)
        
        # 자식 항목 추가
        if "children" in item and item["children"]:
            flat_list.extend(flatten_toc(item["children"], level + 1))
    
    return flat_list

def get_github_token():
    """
    GitHub API 토큰을 환경 변수에서 가져오는 함수
    
    Returns:
        str: GitHub API 토큰 또는 None
    """
    # 환경 변수에서 토큰 가져오기 (여러 가능한 환경 변수 이름 시도)
    token = os.environ.get('GITHUB_API_TOKEN') or os.environ.get('GITHUB_TOKEN')
    
    if token:
        print("환경 변수에서 GitHub API 토큰을 찾았습니다.")
    else:
        print("환경 변수에서 GitHub API 토큰을 찾지 못했습니다.")
    
    return token

def extract_github_links_via_api(repo_url, pattern=None, github_token=None):
    """
    GitHub API를 통해 특정 패턴의 링크를 추출하는 함수
    
    Args:
        repo_url (str): GitHub 저장소 URL
        pattern (str): 링크 필터링 패턴 (정규식)
        github_token (str): GitHub API 토큰 (선택 사항)
        
    Returns:
        list: 추출된 링크 리스트
    """
    try:
        # 저장소 API URL 구성
        parsed_url = urlparse(repo_url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        # 기본 정보 추출
        owner = path_parts[0]
        repo = path_parts[1]
        
        # 경로 추출
        path = ""
        if len(path_parts) > 2:
            if 'tree' in path_parts:
                tree_index = path_parts.index('tree')
                path = '/'.join(path_parts[tree_index+2:])
            elif 'blob' in path_parts:
                blob_index = path_parts.index('blob')
                path = '/'.join(path_parts[blob_index+2:])
        
        # GitHub API를 통해 내용 가져오기
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        print(f"API URL: {api_url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        # GitHub 토큰이 제공된 경우 헤더에 추가
        if github_token:
            headers['Authorization'] = f"token {github_token}"
        
        response = requests.get(api_url, headers=headers)
        
        if response.status_code != 200:
            print(f"GitHub API 오류: HTTP 상태 코드 {response.status_code}")
            if response.status_code == 403:
                print("API 속도 제한에 도달했을 수 있습니다. 인증 토큰을 사용하거나 나중에 다시 시도하세요.")
                print("인증 토큰 없이 계속하려면 대체 방법을 사용합니다.")
                return None
            return []
            
        items = response.json()
        
        # 결과가 리스트가 아니면 하나의 파일을 가리키는 경우
        if not isinstance(items, list):
            if items.get('type') == 'file' and (not pattern or re.search(pattern, items.get('name'))):
                return [items.get('html_url')]
            return []
        
        # 링크 추출
        links = []
        for item in items:
            item_type = item.get('type')
            item_name = item.get('name')
            item_url = item.get('html_url')
            
            # 파일인 경우 패턴 확인 후 추가
            if item_type == 'file':
                if not pattern or re.search(pattern, item_name):
                    links.append(item_url)
            
            # 디렉토리인 경우 재귀적으로 탐색
            elif item_type == 'dir':
                dir_links = extract_github_links_via_api(item_url, pattern, github_token)
                if dir_links:
                    links.extend(dir_links)
        
        return links
        
    except Exception as e:
        print(f"GitHub API 링크 추출 오류: {e}")
        return None


def extract_github_links_via_web(repo_url, pattern=None):
    """
    GitHub 웹 페이지를 직접 스크랩하여 특정 패턴의 링크를 추출하는 함수
    
    Args:
        repo_url (str): GitHub 저장소 URL
        pattern (str): 링크 필터링 패턴 (정규식)
        
    Returns:
        list: 추출된 링크 리스트
    """
    try:
        print(f"웹 스크래핑 방식으로 {repo_url}에서 링크를 추출합니다...")
        
        # 웹 페이지 가져오기
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        response = requests.get(repo_url, headers=headers)
        
        if response.status_code != 200:
            print(f"웹 페이지 가져오기 오류: HTTP 상태 코드 {response.status_code}")
            return []
        
        # HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 링크 추출 (파일 및 디렉토리)
        links = []
        
        # 파일 목록 찾기
        file_rows = soup.select("div.Box-row")
        
        for row in file_rows:
            # 링크 찾기
            link_tag = row.select_one("a[href*='/blob/'] span.css-truncate, a[href*='/tree/'] span.css-truncate")
            if not link_tag:
                continue
                
            # 파일/디렉토리 이름
            item_name = link_tag.text.strip()
            
            # 링크 URL
            link_url = row.select_one("a[href*='/blob/'], a[href*='/tree/']")
            if not link_url:
                continue
                
            full_url = f"https://github.com{link_url['href']}"
            
            # 디렉토리인 경우 재귀적으로 탐색
            if "/tree/" in full_url:
                dir_links = extract_github_links_via_web(full_url, pattern)
                links.extend(dir_links)
            # 파일인 경우 패턴 확인 후 추가
            elif "/blob/" in full_url and (not pattern or re.search(pattern, item_name)):
                links.append(full_url)
        
        return links
        
    except Exception as e:
        print(f"웹 스크래핑 링크 추출 오류: {e}")
        return []
    


def extract_github_links(repo_url, pattern=None, github_token=None):
    """
    GitHub 저장소에서 특정 패턴의 링크를 추출하는 함수
    API가 실패하면 웹 스크래핑 방식으로 대체
    
    Args:
        repo_url (str): GitHub 저장소 URL
        pattern (str): 링크 필터링 패턴 (정규식)
        github_token (str): GitHub API 토큰 (선택 사항)
        
    Returns:
        list: 추출된 링크 리스트
    """
    # 먼저 API를 통해 시도
    links = extract_github_links_via_api(repo_url, pattern, github_token)
    
    # API 실패 시 웹 스크래핑으로 대체
    if links is None:
        print("API 접근 실패. 웹 스크래핑 방식으로 전환합니다...")
        links = extract_github_links_via_web(repo_url, pattern)
    
    # 링크 중복 제거
    if links:
        links = list(dict.fromkeys(links))
    
    return links

def download_batch(repo_url, pattern=".+\.md$", output_dir="manuals", github_token=None):
    """
    GitHub에서 일괄 다운로드 함수
    
    Args:
        repo_url (str): GitHub 저장소 URL
        pattern (str): 링크 필터링 패턴 (정규식)
        output_dir (str): 출력 디렉토리
        github_token (str): GitHub API 토큰 (선택 사항)
        
    Returns:
        int: 성공적으로 다운로드된 파일 수
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리 생성됨: {output_dir}")
    
    # GitHub에서 링크 추출
    print(f"GitHub 저장소 URL에서 링크 추출 중: {repo_url}")
    links = extract_github_links(repo_url, pattern, github_token)
    
    if not links:
        print("추출된 링크가 없습니다.")
        return 0
    
    print(f"총 {len(links)}개의 링크를 찾았습니다.")
    
    # 각 링크에서 콘텐츠 다운로드
    successful_downloads = 0
    
    for i, link in enumerate(links):
        print(f"\n다운로드 중 ({i+1}/{len(links)}): {link}")
        
        # 출력 파일 이름 결정
        parsed_url = urlparse(link)
        path_parts = unquote(parsed_url.path).split('/')
        file_name = None
        
        for part in reversed(path_parts):
            if part.endswith('.md'):
                file_name = part[:-3].replace(" ", "_") + ".json"  # .md -> .json
                break
        
        if not file_name:
            file_name = f"manual_{i+1}.json"
        
        output_file = os.path.join(output_dir, file_name)
        
        # 다운로드 및 추출
        content = extract_from_github_url(link, output_file)
        
        if content:
            print("성공!")
            print("--- 내용 미리보기 ---")
            print(content[:200] + "...")
            successful_downloads += 1
        else:
            print("실패. 다음으로 넘어갑니다.")
        
        # 서버에 부담을 주지 않기 위해 잠시 대기
        time.sleep(1)
    
    print(f"\n다운로드 완료! 총 {len(links)}개 중 {successful_downloads}개 성공")
    return successful_downloads


def interactive_mode():
    """대화형 모드 함수"""
    print("=" * 50)
    print("   Altibase 매뉴얼 다운로더 - 대화형 모드")
    print("=" * 50)
    
    # 환경 변수에서 GitHub 토큰 가져오기
    github_token = get_github_token()
    
    # 사용자에게 토큰 입력 요청 (환경 변수에 없는 경우)
    if not github_token:
        print("\n선택 사항: GitHub API 속도 제한을 피하기 위한 토큰 입력")
        github_token = input("GitHub 개인 액세스 토큰 (없으면 Enter): ")
    
    # 1단계: GitHub URL 테스트
    print("\n1. GitHub 저장소 URL 확인하기")
    repo_url = input("GitHub 저장소 URL을 입력하세요 [기본값: https://github.com/ALTIBASE/Documents/tree/master/Manuals/Altibase_7.3/kor]: ")
    
    if not repo_url:
        repo_url = "https://github.com/ALTIBASE/Documents/tree/master/Manuals/Altibase_7.3/kor"
    
    # 2단계: 링크 추출 테스트
    print(f"\nGitHub URL에서 링크 추출 테스트 중: {repo_url}")
    pattern = input("파일 패턴을 입력하세요 (정규식) [기본값: .+\\.md$]: ") or ".+\\.md$"
    
    links = extract_github_links(repo_url, pattern, github_token)
    
    if links:
        print(f"\n총 {len(links)}개의 링크를 찾았습니다.")
        print("처음 5개 링크 샘플:")
        for i, link in enumerate(links[:5]):
            print(f"{i+1}. {link}")
        
        # 3단계: 단일 URL 테스트
        if len(links) > 0:
            print("\n2. 특정 URL에서 내용 확인하기")
            test_url = links[0]  # 첫 번째 링크 사용
            print(f"테스트 URL: {test_url}")
            
            output_dir = "test_output"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "test_manual_output.json")
            
            content = extract_from_github_url(test_url, output_file)
            
            if content:
                print("\n테스트 성공!")
                print(f"파일 저장됨: {output_file}")
                print("\n--- 내용 미리보기 ---")
                print(content[:500] + "...")
                
                # 계속 진행 확인
                proceed = input("\n일괄 다운로드를 진행하시겠습니까? (y/n) [기본값: y]: ").lower()
                if proceed != 'n':
                    # 4단계: 배치 다운로드 설정
                    print("\n3. 일괄 다운로드 설정")
                    
                    output_dir = input("저장할 폴더명 [기본값: manuals]: ") or "manuals"
                    
                    # 배치 다운로드 실행
                    print("\n4. 일괄 다운로드 실행")
                    download_batch(repo_url, pattern, output_dir, github_token)
            else:
                print("\n테스트 실패! 프로그램을 종료합니다.")
        else:
            print("테스트를 위한 링크가 없습니다.")
    else:
        print("URL에서 링크를 추출하지 못했습니다. 프로그램을 종료합니다.")


def main():
    parser = argparse.ArgumentParser(description='Altibase 매뉴얼 다운로더')
    parser.add_argument('--url', type=str, help='단일 GitHub URL에서 다운로드')
    parser.add_argument('--file', type=str, help='로컬 마크다운 파일에서 추출')
    parser.add_argument('--output', type=str, help='출력 파일 경로')
    parser.add_argument('--repo', type=str, help='GitHub 저장소 URL', default="https://github.com/ALTIBASE/Documents/tree/master/Manuals/Altibase_7.3/kor")
    parser.add_argument('--pattern', type=str, help='파일 패턴 (정규식)', default=".+\\.md$")
    parser.add_argument('--output-dir', type=str, default='manuals', help='출력 디렉토리 (기본값: manuals)')
    parser.add_argument('--token', type=str, help='GitHub API 토큰')
    parser.add_argument('--batch', action='store_true', help='일괄 다운로드 모드')
    parser.add_argument('--interactive', action='store_true', help='대화형 모드')
    parser.add_argument('--scrape', action='store_true', help='API 대신 웹 스크래핑 방식 사용')
    
    args = parser.parse_args()
    
    # 스크래핑 방식 강제 적용
    if args.scrape:
        global extract_github_links
        extract_github_links = lambda url, pattern, token: extract_github_links_via_web(url, pattern)
        print("웹 스크래핑 방식을 강제로 사용합니다.")
    
    if args.interactive:
        interactive_mode()
    elif args.url:
        # 단일 URL 처리
        output_file = args.output or "url_output.json"
        content = extract_from_github_url(args.url, output_file)
        if content:
            print("\n추출된 내용 미리보기:")
            print(content[:500] + "...")
    elif args.file:
        # 로컬 파일 처리
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            output_file = args.output or f"{os.path.splitext(args.file)[0]}_extracted.json"
            content = extract_from_markdown(markdown_content)
            
            if content:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"내용이 추출되어 {output_file}에 저장되었습니다.")
                print("\n추출된 내용 미리보기:")
                print(content[:500] + "...")
        except Exception as e:
            print(f"파일 처리 오류: {e}")
    elif args.batch:
        # 일괄 다운로드 모드
        download_batch(args.repo, args.pattern, args.output_dir, args.token)
    else:
        # 기본적으로 대화형 모드 사용
        interactive_mode()

if __name__ == "__main__":
    main()