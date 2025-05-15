import os
import shutil
import sys

def find_and_copy_doc_files(source_dir, target_dir):
    """지정된 디렉토리와 그 하위 디렉토리에서 모든 .doc 파일을 찾아 대상 디렉토리로 복사합니다."""
    
    # 소스와 타겟 디렉토리 확인 메시지
    print(f"소스 디렉토리: {source_dir}")
    print(f"타겟 디렉토리: {target_dir}")
    
    # 대상 디렉토리가 없으면 생성
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"생성된 대상 디렉토리: {target_dir}")
    
    # 소스 디렉토리가 존재하지 않으면 오류 메시지 출력 후 종료
    if not os.path.exists(source_dir):
        print(f"오류: 소스 디렉토리 '{source_dir}'가 존재하지 않습니다.")
        return
    
    # 찾은 파일 개수를 추적
    found_files = 0
    
    # 모든 하위 디렉토리 검색
    for root, dirs, files in os.walk(source_dir):
        # 각 디렉토리의 파일들 중 .doc 파일 찾기
        for file in files:
            if file.lower().endswith('.doc'):
                # 원본 파일의 전체 경로
                source_path = os.path.join(root, file)
                # 대상 파일의 전체 경로
                target_path = os.path.join(target_dir, file)
                
                # 파일 복사
                try:
                    shutil.copy2(source_path, target_path)
                    print(f"복사됨: {source_path} -> {target_path}")
                    found_files += 1
                except Exception as e:
                    print(f"오류: {source_path} 복사 실패 - {str(e)}")
    
    if found_files == 0:
        print("찾은 .doc 파일이 없습니다.")
    else:
        print(f"총 {found_files}개의 .doc 파일을 복사했습니다.")

def main():
    # 기본값 설정
    default_source = "/Users/j/OneDrive"
    default_target = "/Users/j/Documents/mydocs"
    
    # 사용자 입력 받기
    source_input = input(f"소스 디렉토리를 입력하세요 (엔터키: {default_source}): ").strip()
    
    # 소스 디렉토리 결정 (입력이 없으면 기본값 사용)
    source_dir = source_input if source_input else default_source
    
    # 타겟 디렉토리는 항상 기본값 사용
    target_dir = default_target
    
    # 파일 찾기 및 복사 실행
    find_and_copy_doc_files(source_dir, target_dir)

if __name__ == "__main__":
    main()