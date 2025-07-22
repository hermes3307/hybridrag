import requests
from bs4 import BeautifulSoup
import os
import time
import sys
import argparse

def extract_from_url(url, output_file=None):
    """
    URL에서 케이스 스터디 내용을 추출하는 함수 (Hangul 인코딩 처리)
    
    Args:
        url (str): 가져올 URL
        output_file (str): 출력 파일 경로 (선택 사항)
        
    Returns:
        str: 추출된 내용 또는 실패 시 None
    """
    try:
        # UTF-8 지원을 표시하는 헤더가 있는 요청 전송
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Charset': 'UTF-8',
        }
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'  # UTF-8 인코딩 강제 적용
        
        # CAPTCHA 확인
        if "자동등록방지" in response.text or "security" in response.text.lower():
            print("CAPTCHA가 감지되었습니다. 이 URL은 수동 확인이 필요합니다.")
            return None
        
        # HTML에서 내용 추출
        return extract_from_html(response.text, output_file)
        
    except Exception as e:
        print(f"URL 가져오기 오류: {e}")
        return None

def extract_from_html(html_content, output_file=None):
    """
    HTML 콘텐츠에서 케이스 스터디 내용을 추출하는 함수 (Hangul 인코딩 처리)
    
    Args:
        html_content (str): HTML 콘텐츠
        output_file (str): 출력 파일 경로 (선택 사항)
        
    Returns:
        str: 추출된 내용 또는 실패 시 None
    """
    # HTML 파싱
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 케이스 스터디 콘텐츠 찾기
    case_view = soup.find('article', class_='case-view-content')
    
    if not case_view:
        print("HTML에서 케이스 스터디 콘텐츠를 찾을 수 없습니다.")
        return None
    
    # 제목 추출
    title_element = case_view.find('h3', class_='view-top-tit')
    title = title_element.text.strip() if title_element else "제목 없음"
    
    # 제목으로 추출된 내용 초기화
    extracted_text = f"제목: {title}\n\n"
    
    # 모든 섹션 div 찾기
    sections = case_view.find_all('div', class_='view-con-sec')
    
    # 추출할 섹션 정의
    target_sections = ["who", "problem", "solution", "results"]
    
    # 각 섹션 처리
    for section in sections:
        # 섹션 제목 가져오기
        section_title = section.find('h4', class_='view-sec-tit')
        if not section_title:
            continue
        
        section_name = section_title.text.strip().lower()
        
        # 이것이 우리가 원하는 섹션인지 확인
        if section_name in target_sections:
            # 섹션 내용 초기화
            section_content = ""
            
            # gray-check-box에서 내용 추출 (글머리 기호)
            check_box = section.find('div', class_='gray-check-box')
            if check_box:
                for p in check_box.find_all('p'):
                    if p.text.strip():
                        section_content += f"- {p.text.strip()}\n"
            
            # 일반 텍스트에서 내용 추출
            text_section = section.find('div', class_='view-sec-txt')
            if text_section:
                span = text_section.find('span')
                if span:
                    # <br> 태그를 줄바꿈으로 바꾸기
                    for br in span.find_all('br'):
                        br.replace_with('\n')
                    
                    text = span.text.strip()
                    if text:
                        section_content += text
            
            # 추출된 내용에 섹션 추가
            section_title_capitalized = section_name.capitalize()
            extracted_text += f"{section_title_capitalized}: {section_content.strip()}\n\n"
    
    # output_file이 지정된 경우 추출된 내용을 파일에 저장
    if output_file:
        # 디렉토리가 존재하는지 확인
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # UTF-8 인코딩으로 쓰기
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"내용이 추출되어 {output_file}에 저장되었습니다.")
    
    return extracted_text

def download_batch(cate_start, cate_end, idx_start, idx_end, output_dir="case_studies"):
    """일괄 다운로드 함수"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리 생성됨: {output_dir}")
    
    total_tasks = (cate_end - cate_start + 1) * (idx_end - idx_start + 1)
    completed_tasks = 0
    successful_downloads = 0
    
    print(f"다운로드 시작: 카테고리 {cate_start}-{cate_end}, 인덱스 {idx_start}-{idx_end}")
    print(f"총 {total_tasks}개의 케이스 스터디를 다운로드합니다.")
    
    for cate in range(cate_start, cate_end + 1):
        for idx in range(idx_start, idx_end + 1):
            url = f"https://altibase.com/kr/learn/case_list.php?bgu=view&idx={idx}&cate={cate}"
            output_file = f"{output_dir}/cate_{cate}_idx_{idx}.txt"
            
            print(f"\n다운로드 중 ({completed_tasks+1}/{total_tasks}): cate={cate}, idx={idx}")
            print(f"URL: {url}")
            
            content = extract_from_url(url, output_file)
            
            if content:
                print("성공!")
                print("--- 내용 미리보기 ---")
                print(content[:200] + "...")
                successful_downloads += 1
            else:
                print("실패. 다음으로 넘어갑니다.")
            
            completed_tasks += 1
            progress = (completed_tasks / total_tasks) * 100
            print(f"진행률: {progress:.1f}%")
            
            # 서버에 부담을 주지 않기 위해 잠시 대기
            time.sleep(2)
    
    print(f"\n다운로드 완료! 총 {total_tasks}개 중 {successful_downloads}개 성공")
    return successful_downloads

def interactive_mode():
    """대화형 모드 함수"""
    print("=" * 50)
    print("   Altibase 케이스 스터디 다운로더 - 대화형 모드")
    print("=" * 50)
    
    # 1단계: URL 예제 테스트
    print("\n1. 특정 URL에서 내용 확인하기")
    test_url = input("테스트할 URL을 입력하세요 [기본값: https://altibase.com/kr/learn/case_list.php?bgu=view&idx=324&cate=3]: ")
    
    if not test_url:
        test_url = "https://altibase.com/kr/learn/case_list.php?bgu=view&idx=324&cate=3"
    
    print(f"\nURL 테스트 중: {test_url}")
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/test_url_output.txt"
    
    content = extract_from_url(test_url, output_file)
    
    if content:
        print("\n테스트 성공!")
        print(f"파일 저장됨: {output_file}")
        print("\n--- 내용 미리보기 ---")
        print(content[:500] + "...")
        
        # 계속 진행 확인
        proceed = input("\n일괄 다운로드를 진행하시겠습니까? (y/n) [기본값: y]: ").lower()
        if proceed != 'n':
            # 2단계: 배치 다운로드 설정
            print("\n2. 일괄 다운로드 설정")
            
            try:
                cate_start = int(input("카테고리 시작값 (cate) [기본값: 1]: ") or "1")
                cate_end = int(input("카테고리 종료값 (cate) [기본값: 7]: ") or "7")
                
                if cate_start > cate_end:
                    print("오류: 시작값이 종료값보다 큽니다. 기본값을 사용합니다.")
                    cate_start, cate_end = 1, 7
                
                idx_start = int(input("인덱스 시작값 (idx) [기본값: 200]: ") or "200")
                idx_end = int(input("인덱스 종료값 (idx) [기본값: 400]: ") or "400")
                
                if idx_start > idx_end:
                    print("오류: 시작값이 종료값보다 큽니다. 기본값을 사용합니다.")
                    idx_start, idx_end = 200, 400
                
                output_dir = input("저장할 폴더명 [기본값: case_studies]: ") or "case_studies"
                
                # 배치 다운로드 실행
                print("\n3. 일괄 다운로드 실행")
                download_batch(cate_start, cate_end, idx_start, idx_end, output_dir)
                
            except ValueError:
                print("오류: 유효한 숫자를 입력해주세요.")
                return
    else:
        print("\n테스트 실패! 프로그램을 종료합니다.")

def main():
    parser = argparse.ArgumentParser(description='Altibase 케이스 스터디 다운로더')
    parser.add_argument('--url', type=str, help='단일 URL에서 다운로드 (예: https://altibase.com/kr/learn/case_list.php?bgu=view&idx=324&cate=3)')
    parser.add_argument('--file', type=str, help='로컬 HTML 파일에서 추출')
    parser.add_argument('--output', type=str, help='출력 파일 경로')
    parser.add_argument('--cate-start', type=int, default=1, help='카테고리 시작값 (기본값: 1)')
    parser.add_argument('--cate-end', type=int, default=7, help='카테고리 종료값 (기본값: 7)')
    parser.add_argument('--idx-start', type=int, default=200, help='인덱스 시작값 (기본값: 200)')
    parser.add_argument('--idx-end', type=int, default=400, help='인덱스 종료값 (기본값: 400)')
    parser.add_argument('--output-dir', type=str, default='case_studies', help='출력 디렉토리 (기본값: case_studies)')
    parser.add_argument('--batch', action='store_true', help='일괄 다운로드 모드')
    parser.add_argument('--interactive', action='store_true', help='대화형 모드')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.url:
        # 단일 URL 처리
        output_file = args.output or "url_output.txt"
        content = extract_from_url(args.url, output_file)
        if content:
            print("\n추출된 내용 미리보기:")
            print(content[:500] + "...")
    elif args.file:
        # 로컬 파일 처리
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            output_file = args.output or f"{os.path.splitext(args.file)[0]}_extracted.txt"
            content = extract_from_html(html_content, output_file)
            
            if content:
                print("\n추출된 내용 미리보기:")
                print(content[:500] + "...")
        except Exception as e:
            print(f"파일 처리 오류: {e}")
    elif args.batch:
        # 일괄 다운로드 모드
        download_batch(args.cate_start, args.cate_end, args.idx_start, args.idx_end, args.output_dir)
    else:
        # 기본적으로 대화형 모드 사용
        interactive_mode()

if __name__ == "__main__":
    main()