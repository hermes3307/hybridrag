import os
import re
import argparse
import hashlib
from difflib import SequenceMatcher
import pandas as pd
from tqdm import tqdm
import shutil
import sys

class DuplicateCaseStudyDetector:
    def __init__(self, directory="case_studies"):
        """
        중복 케이스 스터디 검출기 초기화
        
        Args:
            directory (str): 케이스 스터디 파일이 있는 디렉토리
        """
        self.directory = directory
        self.case_studies = []
        self.duplicates = []
    
    def load_case_studies(self):
        """케이스 스터디 파일 로드"""
        print(f"디렉토리 '{self.directory}'에서 케이스 스터디 파일 로드 중...")
        
        if not os.path.exists(self.directory):
            print(f"오류: 디렉토리 '{self.directory}'가 존재하지 않습니다.")
            return False
        
        # 텍스트 파일만 필터링
        files = [f for f in os.listdir(self.directory) if f.endswith('.txt')]
        
        if not files:
            print(f"디렉토리 '{self.directory}'에 텍스트 파일이 없습니다.")
            return False
        
        print(f"총 {len(files)}개의 파일을 발견했습니다.")
        
        # 각 파일 로드
        for filename in tqdm(files, desc="파일 로드 중"):
            file_path = os.path.join(self.directory, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 파일명에서 cate와 idx 추출
                cate_match = re.search(r'cate_(\d+)', filename)
                idx_match = re.search(r'idx_(\d+)', filename)
                
                cate = int(cate_match.group(1)) if cate_match else None
                idx = int(idx_match.group(1)) if idx_match else None
                
                # 내용 파싱
                title = self._extract_section(content, "제목:")
                who = self._extract_section(content, "Who:")
                problem = self._extract_section(content, "Problem:")
                solution = self._extract_section(content, "Solution:")
                results = self._extract_section(content, "Results:")
                
                # 콘텐츠 핑거프린트 생성 (중복 감지를 위해)
                # 내용의 핵심만 해시로 변환 (공백, 서식 무시)
                content_normalized = self._normalize_content(title + who + problem + solution + results)
                content_hash = hashlib.md5(content_normalized.encode('utf-8')).hexdigest()
                
                # 케이스 스터디 정보 저장
                self.case_studies.append({
                    'filename': filename,
                    'file_path': file_path,
                    'cate': cate,
                    'idx': idx,
                    'title': title,
                    'content': content,
                    'content_hash': content_hash,
                    'content_length': len(content)
                })
                
            except Exception as e:
                print(f"파일 '{filename}' 처리 중 오류 발생: {str(e)}")
        
        print(f"{len(self.case_studies)}개의 케이스 스터디를 로드했습니다.")
        return len(self.case_studies) > 0
    
    def _extract_section(self, content, section_marker):
        """내용에서 특정 섹션 추출"""
        if section_marker not in content:
            return ""
        
        parts = content.split(section_marker, 1)
        if len(parts) < 2:
            return ""
        
        section_content = parts[1].strip()
        
        # 다음 섹션 마커 찾기
        next_section_markers = ["제목:", "Who:", "Problem:", "Solution:", "Results:"]
        for marker in next_section_markers:
            if marker in section_content and marker != section_marker:
                section_content = section_content.split(marker, 1)[0]
        
        return section_content.strip()
    
    def _normalize_content(self, text):
        """텍스트 정규화 (중복 감지를 위해)"""
        # 공백, 줄바꿈, 특수문자 등 제거
        normalized = re.sub(r'\s+', '', text)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized.lower()  # 소문자로 변환
    
    def find_duplicates(self, similarity_threshold=0.9):
        """중복 케이스 스터디 찾기"""
        print(f"중복 검사 중... (유사도 임계값: {similarity_threshold})")
        
        if not self.case_studies:
            print("케이스 스터디가 로드되지 않았습니다.")
            return []
        
        # 해시 기반 중복 감지 (정확한 중복)
        hash_groups = {}
        for case in self.case_studies:
            content_hash = case['content_hash']
            if content_hash in hash_groups:
                hash_groups[content_hash].append(case)
            else:
                hash_groups[content_hash] = [case]
        
        # 중복 그룹 찾기
        duplicate_groups = []
        for hash_val, cases in hash_groups.items():
            if len(cases) > 1:
                duplicate_groups.append(cases)
        
        # 유사도 기반 중복 감지 (유사한 중복)
        # 해시 검사에서 중복으로 감지되지 않은 케이스만 확인
        unique_cases = [case for case in self.case_studies 
                       if case['content_hash'] not in [g[0]['content_hash'] for g in duplicate_groups]]
        
        # 유사도 검사
        processed_hashes = set()
        for i, case1 in enumerate(tqdm(unique_cases, desc="유사도 검사 중")):
            if case1['content_hash'] in processed_hashes:
                continue
                
            similar_cases = [case1]
            
            for case2 in unique_cases[i+1:]:
                if case2['content_hash'] in processed_hashes:
                    continue
                    
                # 제목 유사도 먼저 확인 (빠른 필터링)
                title_similarity = self._calculate_similarity(case1['title'], case2['title'])
                
                # 제목이 유사하면 전체 내용 유사도 확인
                if title_similarity > 0.8:
                    content_similarity = self._calculate_similarity(case1['content'], case2['content'])
                    
                    if content_similarity >= similarity_threshold:
                        similar_cases.append(case2)
                        processed_hashes.add(case2['content_hash'])
            
            if len(similar_cases) > 1:
                duplicate_groups.append(similar_cases)
            
            processed_hashes.add(case1['content_hash'])
        
        # 결과 저장
        self.duplicates = duplicate_groups
        
        print(f"{len(duplicate_groups)}개의 중복 그룹을 발견했습니다.")
        return duplicate_groups
    
    def _calculate_similarity(self, text1, text2):
        """두 텍스트 간의 유사도 계산"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def get_duplicate_summary(self):
        """중복 요약 데이터 반환"""
        if not self.duplicates:
            print("중복이 발견되지 않았습니다.")
            return pd.DataFrame()
        
        # 요약 데이터 구성
        summary_data = []
        
        for group_idx, group in enumerate(self.duplicates, 1):
            # 그룹 내 케이스 정렬 (파일명 기준)
            sorted_group = sorted(group, key=lambda x: x['filename'])
            
            for case_idx, case in enumerate(sorted_group, 1):
                summary_data.append({
                    'group': group_idx,
                    'case': case_idx,
                    'filename': case['filename'],
                    'cate': case['cate'],
                    'idx': case['idx'],
                    'title': case['title'],
                    'content_length': case['content_length']
                })
        
        return pd.DataFrame(summary_data)
    
    def export_duplicate_summary(self, output_file="duplicate_summary.csv"):
        """중복 요약을 CSV 파일로 내보내기"""
        summary_df = self.get_duplicate_summary()
        
        if summary_df.empty:
            return False
        
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"중복 요약이 '{output_file}'에 저장되었습니다.")
        return True
    
    def resolve_duplicates(self, interactive=True, backup_dir=None):
        """중복 해결 (대화형 또는 자동)"""
        if not self.duplicates:
            print("해결할 중복이 없습니다.")
            return
        
        # 백업 디렉토리 생성 (요청된 경우)
        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)
            print(f"백업 디렉토리 '{backup_dir}'가 생성되었습니다.")
        
        print(f"{len(self.duplicates)}개의 중복 그룹을 해결합니다...")
        
        for group_idx, group in enumerate(self.duplicates, 1):
            print("\n" + "=" * 80)
            print(f"중복 그룹 {group_idx}/{len(self.duplicates)}")
            print("=" * 80)
            
            # 그룹 내 케이스 정렬 (파일명 기준)
            sorted_group = sorted(group, key=lambda x: x['filename'])
            
            for case_idx, case in enumerate(sorted_group, 1):
                print(f"{case_idx}. {case['filename']}")
                print(f"   제목: {case['title']}")
                print(f"   길이: {case['content_length']} 문자")
                
                # 미리보기 (처음 100자)
                preview = case['content'][:100] + "..." if len(case['content']) > 100 else case['content']
                print(f"   미리보기: {preview}")
            
            if interactive:
                # 사용자에게 어떤 파일을 유지할지 질문
                keep_idx = None
                while keep_idx is None:
                    try:
                        user_input = input("\n유지할 파일 번호를 입력하세요 (0=그룹 건너뛰기): ")
                        keep_idx = int(user_input)
                        if keep_idx < 0 or keep_idx > len(sorted_group):
                            print(f"유효한 번호를 입력하세요 (0-{len(sorted_group)}).")
                            keep_idx = None
                    except ValueError:
                        print("숫자를 입력하세요.")
                
                if keep_idx == 0:
                    print("이 중복 그룹을 건너뜁니다.")
                    continue
                
                # 선택한 파일 외의 파일 삭제 또는 백업
                for case_idx, case in enumerate(sorted_group, 1):
                    if case_idx != keep_idx:
                        if backup_dir:
                            # 파일 백업
                            backup_path = os.path.join(backup_dir, case['filename'])
                            shutil.copy2(case['file_path'], backup_path)
                            print(f"파일 '{case['filename']}'을(를) '{backup_path}'에 백업했습니다.")
                        
                        # 파일 삭제
                        os.remove(case['file_path'])
                        print(f"파일 '{case['filename']}'을(를) 삭제했습니다.")
            else:
                # 자동 모드: 항상 첫 번째 파일 유지
                for case_idx, case in enumerate(sorted_group, 1):
                    if case_idx != 1:  # 첫 번째가 아닌 모든 파일 삭제
                        if backup_dir:
                            backup_path = os.path.join(backup_dir, case['filename'])
                            shutil.copy2(case['file_path'], backup_path)
                        
                        os.remove(case['file_path'])
                        print(f"파일 '{case['filename']}'을(를) 삭제했습니다.")
        
        print("\n중복 해결이 완료되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='케이스 스터디 중복 검출 및 해결')
    parser.add_argument('--dir', type=str, default='case_studies', help='케이스 스터디 디렉토리 (기본값: case_studies)')
    parser.add_argument('--backup', type=str, help='삭제된 파일의 백업 디렉토리 (선택 사항)')
    parser.add_argument('--threshold', type=float, default=0.9, help='유사도 임계값 (0.0-1.0, 기본값: 0.9)')
    parser.add_argument('--summary', type=str, help='중복 요약을 저장할 CSV 파일 (선택 사항)')
    parser.add_argument('--auto', action='store_true', help='대화형 모드 비활성화 (항상 첫 번째 파일 유지)')
    
    args = parser.parse_args()
    
    # 중복 검출기 초기화
    detector = DuplicateCaseStudyDetector(args.dir)
    
    # 케이스 스터디 로드
    if not detector.load_case_studies():
        return
    
    # 중복 찾기
    detector.find_duplicates(args.threshold)
    
    # 중복 요약 내보내기 (요청된 경우)
    if args.summary:
        detector.export_duplicate_summary(args.summary)
    
    # 중복 해결
    detector.resolve_duplicates(not args.auto, args.backup)

def interactive_mode():
    """대화형 모드 실행"""
    print("\n" + "=" * 80)
    print("케이스 스터디 중복 검출 및 해결 - 대화형 모드")
    print("=" * 80)
    
    # 디렉토리 입력
    directory = input("\n케이스 스터디 디렉토리를 입력하세요 (기본값: case_studies): ") or "case_studies"
    
    # 유사도 임계값 입력
    threshold = 0.9
    threshold_input = input(f"유사도 임계값을 입력하세요 (0.0-1.0, 기본값: {threshold}): ")
    if threshold_input:
        try:
            threshold = float(threshold_input)
            if threshold < 0 or threshold > 1:
                print("유효한 범위를 벗어났습니다. 기본값을 사용합니다.")
                threshold = 0.9
        except ValueError:
            print("올바른 숫자 형식이 아닙니다. 기본값을 사용합니다.")
    
    # 백업 설정
    backup_dir = None
    backup_choice = input("삭제된 파일을 백업하시겠습니까? (y/n, 기본값: y): ").lower() or "y"
    if backup_choice == "y":
        backup_dir = input("백업 디렉토리를 입력하세요 (기본값: duplicates_backup): ") or "duplicates_backup"
    
    # 중복 요약 내보내기 설정
    summary_file = None
    summary_choice = input("중복 요약을 CSV 파일로 내보내시겠습니까? (y/n, 기본값: y): ").lower() or "y"
    if summary_choice == "y":
        summary_file = input("CSV 파일 경로를 입력하세요 (기본값: duplicate_summary.csv): ") or "duplicate_summary.csv"
    
    # 중복 검출기 초기화
    detector = DuplicateCaseStudyDetector(directory)
    
    # 케이스 스터디 로드
    if not detector.load_case_studies():
        return
    
    # 중복 찾기
    detector.find_duplicates(threshold)
    
    # 중복 요약 내보내기 (요청된 경우)
    if summary_file:
        detector.export_duplicate_summary(summary_file)
    
    # 모드 선택
    mode_choice = input("\n자동으로 중복을 해결하시겠습니까? (y=자동, n=대화형, 기본값: n): ").lower() or "n"
    
    # 중복 해결
    detector.resolve_duplicates(mode_choice != "y", backup_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 명령줄 인수가 제공된 경우 main() 실행
        main()
    else:
        # 명령줄 인수가 없는 경우 대화형 모드 실행
        interactive_mode()