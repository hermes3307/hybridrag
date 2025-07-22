import requests
import os
import time
import sys
import argparse
import re
from urllib.parse import urlparse, unquote
import json
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional, Tuple
import traceback

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            
        logger.info(f"원시 콘텐츠 URL: {raw_url}")
        
        # UTF-8 지원을 표시하는 헤더가 있는 요청 전송
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Charset': 'UTF-8',
        }
        response = requests.get(raw_url, headers=headers, timeout=30)
        response.encoding = 'utf-8'  # UTF-8 인코딩 강제 적용
        
        if response.status_code != 200:
            logger.error(f"오류: HTTP 상태 코드 {response.status_code}")
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
            logger.info(f"내용이 추출되어 {output_file}에 저장되었습니다.")
        
        return structured_content
        
    except Exception as e:
        logger.error(f"URL 가져오기 오류: {e}")
        return None

def extract_from_markdown(markdown_content, source_url=""):
    """
    마크다운 콘텐츠에서 매뉴얼 내용과 목차를 추출하는 함수 (개선된 버전)
    
    Args:
        markdown_content (str): 마크다운 콘텐츠
        source_url (str): 소스 URL
        
    Returns:
        str: 추출된 내용 또는 실패 시 None
    """
    try:
        # 마크다운 콘텐츠가 비어 있는지 확인
        if not markdown_content or markdown_content.strip() == "":
            logger.warning("마크다운 콘텐츠가 비어 있습니다.")
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
        
        # 목차 구조 생성 (헤더 기반)
        toc_structure = parse_headers_improved(markdown_content)
        
        # 섹션별 내용 추출 (개선된 방법)
        sections_content = extract_sections_improved(markdown_content, toc_structure)
        
        # 구조화된 내용 조합
        structured_content = {
            "title": title,
            "source_url": source_url,
            "total_sections": len(sections_content),
            "content_length": len(markdown_content),
            "toc": toc_structure,
            "sections": sections_content
        }
        
        # JSON 형식으로 변환
        return json.dumps(structured_content, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"마크다운 처리 오류: {e}")
        logger.error(traceback.format_exc())
        return None

def parse_headers_improved(markdown_content: str) -> List[Dict]:
    """
    마크다운 헤더를 파싱하여 개선된 목차 구조를 생성
    """
    toc_structure = []
    header_pattern = re.compile(r'^(#{1,6})\s+(.+?)(?:\s+#{1,6})?$', re.MULTILINE)
    
    for match in header_pattern.finditer(markdown_content):
        level = len(match.group(1)) - 1  # # = 0, ## = 1, ### = 2, ...
        title = match.group(2).strip()
        start_pos = match.start()
        
        # 앵커 링크 제거
        title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', title)
        title = re.sub(r'<[^>]+>', '', title)  # HTML 태그 제거
        
        # 챕터 번호와 제목 분리
        chapter_match = re.match(r'^(\d+(?:\.\d+)*)\s*[\.:]?\s*(.*)$', title)
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
            "full_title": title,
            "start_position": start_pos,
            "children": []
        }
        
        toc_structure.append(item)
    
    return toc_structure

def extract_sections_improved(markdown_content: str, toc_structure: List[Dict]) -> Dict[str, str]:
    """
    개선된 섹션 내용 추출
    """
    sections_content = {}
    
    if not toc_structure:
        # 헤더가 없는 경우 전체 내용을 하나의 섹션으로 처리
        sections_content["main_content"] = markdown_content.strip()
        return sections_content
    
    # 섹션 경계 정의
    section_boundaries = []
    for item in toc_structure:
        section_boundaries.append({
            'start': item['start_position'],
            'level': item['level'],
            'key': generate_section_key(item),
            'title': item['title'],
            'chapter': item['chapter']
        })
    
    # 섹션별 내용 추출
    for i, boundary in enumerate(section_boundaries):
        start_pos = boundary['start']
        
        # 다음 섹션의 시작 위치 찾기
        end_pos = len(markdown_content)
        
        # 같은 레벨 이상의 다음 헤더 찾기
        for j in range(i + 1, len(section_boundaries)):
            next_boundary = section_boundaries[j]
            if next_boundary['level'] <= boundary['level']:
                end_pos = next_boundary['start']
                break
        
        # 섹션 내용 추출
        section_content = markdown_content[start_pos:end_pos].strip()
        
        # 빈 섹션 제외
        if section_content and len(section_content.strip()) > 10:
            sections_content[boundary['key']] = section_content
    
    return sections_content

def generate_section_key(item: Dict) -> str:
    """섹션 키 생성"""
    chapter = item.get('chapter', '')
    title = item.get('title', '')
    
    # 키 생성
    if chapter:
        key = f"{chapter}_{title}"
    else:
        key = title
    
    # 특수 문자를 안전한 문자로 변환
    key = re.sub(r'[^\w\s가-힣-]', '_', key)
    key = re.sub(r'\s+', '_', key)
    key = re.sub(r'_+', '_', key)
    key = key.strip('_')
    
    return key or "untitled_section"

def get_github_token():
    """
    GitHub API 토큰을 환경 변수에서 가져오는 함수
    """
    token = os.environ.get('GITHUB_API_TOKEN') or os.environ.get('GITHUB_TOKEN')
    
    if token:
        logger.info("환경 변수에서 GitHub API 토큰을 찾았습니다.")
    else:
        logger.info("환경 변수에서 GitHub API 토큰을 찾지 못했습니다.")
    
    return token

def extract_github_links_via_api(repo_url, pattern=None, github_token=None, max_depth=5, current_depth=0):
    """
    GitHub API를 통해 특정 패턴의 링크를 추출하는 함수 (깊이 제한 추가)
    """
    if current_depth >= max_depth:
        logger.warning(f"최대 깊이({max_depth})에 도달했습니다.")
        return []
    
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
        logger.info(f"API URL: {api_url}")
        
        headers = {
            'User-Agent': 'Manual-Extractor/1.0',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # GitHub 토큰이 제공된 경우 헤더에 추가
        if github_token:
            headers['Authorization'] = f"token {github_token}"
        
        response = requests.get(api_url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"GitHub API 오류: HTTP 상태 코드 {response.status_code}")
            if response.status_code == 403:
                logger.error("API 속도 제한에 도달했을 수 있습니다.")
            return []
            
        items = response.json()
        
        # 결과가 리스트가 아니면 하나의 파일을 가리키는 경우
        if not isinstance(items, list):
            if items.get('type') == 'file' and (not pattern or re.search(pattern, items.get('name', ''))):
                return [items.get('html_url')]
            return []
        
        # 링크 추출
        links = []
        for item in items:
            item_type = item.get('type')
            item_name = item.get('name', '')
            item_url = item.get('html_url')
            
            # 파일인 경우 패턴 확인 후 추가
            if item_type == 'file':
                if not pattern or re.search(pattern, item_name):
                    links.append(item_url)
            
            # 디렉토리인 경우 재귀적으로 탐색 (깊이 제한)
            elif item_type == 'dir' and current_depth < max_depth - 1:
                try:
                    dir_links = extract_github_links_via_api(
                        item_url, pattern, github_token, 
                        max_depth, current_depth + 1
                    )
                    if dir_links:
                        links.extend(dir_links)
                    
                    # API 호출 간 잠시 대기 (속도 제한 방지)
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"디렉토리 {item_name} 처리 중 오류: {e}")
                    continue
        
        return links
        
    except Exception as e:
        logger.error(f"GitHub API 링크 추출 오류: {e}")
        return []

def download_batch(repo_url, pattern=".+\.md$", output_dir="manuals", github_token=None, max_files=100):
    """
    GitHub에서 일괄 다운로드 함수 (개선된 버전)
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"출력 디렉토리 생성됨: {output_dir}")
    
    # GitHub에서 링크 추출
    logger.info(f"GitHub 저장소 URL에서 링크 추출 중: {repo_url}")
    links = extract_github_links_via_api(repo_url, pattern, github_token)
    
    if not links:
        logger.warning("추출된 링크가 없습니다.")
        return 0
    
    # 링크 수 제한
    if len(links) > max_files:
        logger.warning(f"링크 수가 {max_files}개를 초과합니다. 처음 {max_files}개만 처리합니다.")
        links = links[:max_files]
    
    logger.info(f"총 {len(links)}개의 링크를 처리합니다.")
    
    # 각 링크에서 콘텐츠 다운로드
    successful_downloads = 0
    failed_downloads = 0
    
    for i, link in enumerate(links):
        logger.info(f"다운로드 중 ({i+1}/{len(links)}): {link}")
        
        try:
            # 출력 파일 이름 결정
            parsed_url = urlparse(link)
            path_parts = unquote(parsed_url.path).split('/')
            file_name = None
            
            for part in reversed(path_parts):
                if part.endswith('.md'):
                    # 파일명 정리
                    clean_name = re.sub(r'[^\w\s.-]', '_', part[:-3])
                    clean_name = re.sub(r'\s+', '_', clean_name)
                    file_name = f"{clean_name}.json"
                    break
            
            if not file_name:
                file_name = f"manual_{i+1:03d}.json"
            
            output_file = os.path.join(output_dir, file_name)
            
            # 이미 존재하는 파일 건너뛰기
            if os.path.exists(output_file):
                logger.info(f"파일이 이미 존재합니다. 건너뜁니다: {output_file}")
                successful_downloads += 1
                continue
            
            # 다운로드 및 추출
            content = extract_from_github_url(link, output_file)
            
            if content:
                logger.info("성공!")
                successful_downloads += 1
                
                # 내용 검증
                try:
                    data = json.loads(content)
                    section_count = len(data.get('sections', {}))
                    content_length = data.get('content_length', 0)
                    logger.info(f"섹션 수: {section_count}, 콘텐츠 길이: {content_length}")
                except:
                    logger.warning("JSON 파싱 검증 실패")
            else:
                logger.error("실패")
                failed_downloads += 1
            
        except Exception as e:
            logger.error(f"파일 처리 중 오류: {e}")
            failed_downloads += 1
        
        # 서버에 부담을 주지 않기 위해 잠시 대기
        time.sleep(1)
    
    logger.info(f"다운로드 완료! 성공: {successful_downloads}, 실패: {failed_downloads}")
    return successful_downloads

def interactive_mode():
    """개선된 대화형 모드 함수"""
    print("=" * 60)
    print("   Altibase 매뉴얼 다운로더 - 대화형 모드 (개선된 버전)")
    print("=" * 60)
    
    # 환경 변수에서 GitHub 토큰 가져오기
    github_token = get_github_token()
    
    # 사용자에게 토큰 입력 요청 (환경 변수에 없는 경우)
    if not github_token:
        print("\n선택 사항: GitHub API 속도 제한을 피하기 위한 토큰 입력")
        github_token = input("GitHub 개인 액세스 토큰 (없으면 Enter): ").strip()
    
    # 1단계: GitHub URL 테스트
    print("\n1. GitHub 저장소 URL 확인하기")
    repo_url = input("GitHub 저장소 URL을 입력하세요 [기본값: https://github.com/ALTIBASE/Documents/tree/master/Manuals/Altibase_7.3/kor]: ").strip()
    
    if not repo_url:
        repo_url = "https://github.com/ALTIBASE/Documents/tree/master/Manuals/Altibase_7.3/kor"
    
    # 2단계: 링크 추출 테스트
    print(f"\nGitHub URL에서 링크 추출 테스트 중: {repo_url}")
    pattern = input("파일 패턴을 입력하세요 (정규식) [기본값: .+\\.md$]: ").strip() or ".+\\.md$"
    
    links = extract_github_links_via_api(repo_url, pattern, github_token)
    
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
                
                # 내용 분석
                try:
                    data = json.loads(content)
                    print(f"제목: {data.get('title', 'N/A')}")
                    print(f"섹션 수: {len(data.get('sections', {}))}")
                    print(f"콘텐츠 길이: {data.get('content_length', 0)}")
                    
                    print("\n--- 내용 미리보기 ---")
                    preview = content[:500]
                    print(preview + "..." if len(content) > 500 else preview)
                except:
                    print("JSON 파싱 오류")
                
                # 계속 진행 확인
                proceed = input("\n일괄 다운로드를 진행하시겠습니까? (y/n) [기본값: y]: ").lower()
                if proceed != 'n':
                    # 4단계: 배치 다운로드 설정
                    print("\n3. 일괄 다운로드 설정")
                    
                    output_dir = input("저장할 폴더명 [기본값: manuals]: ").strip() or "manuals"
                    max_files = int(input("최대 파일 수 [기본값: 100]: ") or "100")
                    
                    # 배치 다운로드 실행
                    print("\n4. 일괄 다운로드 실행")
                    download_batch(repo_url, pattern, output_dir, github_token, max_files)
            else:
                print("\n테스트 실패! 프로그램을 종료합니다.")
        else:
            print("테스트를 위한 링크가 없습니다.")
    else:
        print("URL에서 링크를 추출하지 못했습니다. 프로그램을 종료합니다.")

def main():
    parser = argparse.ArgumentParser(description='Altibase 매뉴얼 다운로더 (개선된 버전)')
    parser.add_argument('--url', type=str, help='단일 GitHub URL에서 다운로드')
    parser.add_argument('--file', type=str, help='로컬 마크다운 파일에서 추출')
    parser.add_argument('--output', type=str, help='출력 파일 경로')
    parser.add_argument('--repo', type=str, help='GitHub 저장소 URL', 
                       default="https://github.com/ALTIBASE/Documents/tree/master/Manuals/Altibase_7.3/kor")
    parser.add_argument('--pattern', type=str, help='파일 패턴 (정규식)', default=".+\\.md$")
    parser.add_argument('--output-dir', type=str, default='manuals', help='출력 디렉토리')
    parser.add_argument('--token', type=str, help='GitHub API 토큰')
    parser.add_argument('--batch', action='store_true', help='일괄 다운로드 모드')
    parser.add_argument('--interactive', action='store_true', help='대화형 모드')
    parser.add_argument('--max-files', type=int, default=100, help='최대 처리 파일 수')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.url:
        # 단일 URL 처리
        output_file = args.output or "url_output.json"
        content = extract_from_github_url(args.url, output_file)
        if content:
            print("\n추출 성공!")
            try:
                data = json.loads(content)
                print(f"제목: {data.get('title', 'N/A')}")
                print(f"섹션 수: {len(data.get('sections', {}))}")
                print(f"콘텐츠 길이: {data.get('content_length', 0)}")
            except:
                print("JSON 파싱 오류")
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
                
                try:
                    data = json.loads(content)
                    print(f"제목: {data.get('title', 'N/A')}")
                    print(f"섹션 수: {len(data.get('sections', {}))}")
                    print(f"콘텐츠 길이: {data.get('content_length', 0)}")
                except:
                    print("JSON 파싱 오류")
        except Exception as e:
            print(f"파일 처리 오류: {e}")
    elif args.batch:
        # 일괄 다운로드 모드
        download_batch(args.repo, args.pattern, args.output_dir, args.token, args.max_files)
    else:
        # 기본적으로 대화형 모드 사용
        interactive_mode()

if __name__ == "__main__":
    main()