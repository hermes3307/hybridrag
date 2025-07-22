import os
import shutil
import argparse
from collections import defaultdict
import math
from concurrent.futures import ThreadPoolExecutor

class SimpleFileOrganizer:
    def __init__(self, root_dir=r"E:\Google231213"):
        self.root_dir = root_dir
        self.files_info = []
        self.target_folders = []
        self.file_vectors = {}
        self.folder_vectors = {}
        
    def scan_directory(self):
        """디렉토리를 스캔하여 파일 정보를 수집"""
        print(f"Scanning directory: {self.root_dir}")
        file_count = 0
        
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(dirpath, self.root_dir)
                parent_dir = os.path.basename(dirpath)
                file_name, ext = os.path.splitext(filename)
                ext = ext.lstrip('.')
                
                # 파일 정보 저장
                file_info = {
                    'full_path': full_path,
                    'dir_path': dirpath,
                    'relative_path': relative_path,
                    'parent_dir': parent_dir,
                    'filename': filename,
                    'name': file_name,
                    'extension': ext
                }
                
                self.files_info.append(file_info)
                file_count += 1
                if file_count % 1000 == 0:
                    print(f"Scanned {file_count} files...")
        
        print(f"Found {len(self.files_info)} files")
        
    def create_simple_vectors(self):
        """간단한 메타데이터 기반 벡터 생성"""
        print("Creating metadata vectors for files...")
        
        # 모든 고유 단어 수집
        all_words = set()
        for file_info in self.files_info:
            # 경로, 폴더명, 파일명, 확장자로부터 단어 추출
            words = (
                file_info['relative_path'].replace('\\', ' ').replace('/', ' ').split() +
                [file_info['parent_dir']] +
                file_info['name'].split() +
                [file_info['extension']]
            )
            for word in words:
                if word:  # 빈 문자열이 아닌 경우만 추가
                    all_words.add(word.lower())
        
        # 단어에 인덱스 할당
        self.word_to_index = {word: idx for idx, word in enumerate(all_words)}
        vocabulary_size = len(self.word_to_index)
        print(f"Vocabulary size: {vocabulary_size}")
        
        # 파일별 단어 벡터 생성 (TF)
        for i, file_info in enumerate(self.files_info):
            # 벡터 초기화
            vector = [0] * vocabulary_size
            
            # 단어 추출
            words = (
                file_info['relative_path'].replace('\\', ' ').replace('/', ' ').split() +
                [file_info['parent_dir']] +
                file_info['name'].split() +
                [file_info['extension']]
            )
            
            # 단어 빈도 카운트
            word_count = defaultdict(int)
            for word in words:
                if word:
                    word_count[word.lower()] += 1
            
            # 벡터에 단어 빈도 저장
            for word, count in word_count.items():
                if word in self.word_to_index:
                    vector[self.word_to_index[word]] = count
            
            # 벡터 정규화 (L2 norm)
            norm = math.sqrt(sum(v*v for v in vector))
            if norm > 0:
                vector = [v/norm for v in vector]
            
            self.file_vectors[i] = vector
            
            if (i+1) % 1000 == 0:
                print(f"Created vectors for {i+1} files...")
        
        print(f"Created vectors for {len(self.file_vectors)} files")
        
    def get_target_folders(self):
        """사용자로부터 대상 폴더 목록 입력받기"""
        print("Enter up to 10 target folder names (enter '.' to finish):")
        while True:
            folder = input("Target folder name: ")
            if folder == '.':
                break
            self.target_folders.append(folder)
            if len(self.target_folders) >= 10:
                print("Maximum 10 folders reached.")
                break
        
        # 타겟 폴더 벡터 생성
        vocabulary_size = len(self.word_to_index)
        for i, folder in enumerate(self.target_folders):
            # 벡터 초기화
            vector = [0] * vocabulary_size
            
            # 폴더명 단어 추출
            words = folder.split()
            
            # 단어 빈도 카운트
            word_count = defaultdict(int)
            for word in words:
                if word:
                    word_count[word.lower()] += 1
            
            # 벡터에 단어 빈도 저장
            for word, count in word_count.items():
                if word in self.word_to_index:
                    vector[self.word_to_index[word]] = count
                    
            # 벡터 정규화 (L2 norm)
            norm = math.sqrt(sum(v*v for v in vector))
            if norm > 0:
                vector = [v/norm for v in vector]
            
            self.folder_vectors[i] = vector
            
        print(f"Selected target folders: {self.target_folders}")
        
    def cosine_similarity(self, vec1, vec2):
        """두 벡터 간의 코사인 유사도 계산"""
        dot_product = sum(a*b for a, b in zip(vec1, vec2))
        return dot_product  # 벡터가 이미 정규화되어 있으므로 내적만 계산
        
    def find_best_folder(self, file_idx):
        """파일에 가장 적합한 폴더 찾기"""
        file_vector = self.file_vectors[file_idx]
        best_similarity = -1
        best_folder_idx = 0
        
        for folder_idx, folder_vector in self.folder_vectors.items():
            similarity = self.cosine_similarity(file_vector, folder_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_folder_idx = folder_idx
                
        return self.target_folders[best_folder_idx]
    
    def organize_files(self):
        """파일을 대상 폴더로 이동, 최대 3단계 깊이로 제한"""
        if not self.target_folders:
            print("No target folders defined!")
            return
            
        print("Organizing files into target folders with max 3-level depth...")
        
        # 결과 저장용 딕셔너리
        results = {folder: [] for folder in self.target_folders}
        
        # 각 파일에 대해 가장 적합한 폴더 찾기
        for i in range(len(self.files_info)):
            best_folder = self.find_best_folder(i)
            results[best_folder].append(self.files_info[i])
            
            if (i+1) % 1000 == 0:
                print(f"Classified {i+1} files...")
        
        # 파일 이동
        for folder, files in results.items():
            target_dir = os.path.join(self.root_dir, folder)
            
            # 대상 폴더가 없으면 생성
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                
            print(f"Moving {len(files)} files to {folder}")
            
            # 각 파일 이동
            for file_info in files:
                source_path = file_info['full_path']
                
                # 상대 경로를 최대 3단계로 제한
                rel_path_parts = file_info['relative_path'].split(os.sep)
                if rel_path_parts[0] == '.':
                    rel_path_parts = rel_path_parts[1:]
                    
                # 최대 3단계 깊이로 제한 (대상 폴더 + 2단계)
                if len(rel_path_parts) > 2:
                    rel_path_parts = rel_path_parts[-2:]  # 마지막 2단계만 유지
                    
                rel_path = os.path.join(*rel_path_parts) if rel_path_parts else '.'
                
                if rel_path == '.':
                    target_path = os.path.join(target_dir, file_info['filename'])
                else:
                    # 제한된 하위 폴더 구조 유지
                    full_target_dir = os.path.join(target_dir, rel_path)
                    if not os.path.exists(full_target_dir):
                        os.makedirs(full_target_dir)
                    target_path = os.path.join(full_target_dir, file_info['filename'])
                
                # 파일 이동
                try:
                    shutil.move(source_path, target_path)
                except Exception as e:
                    print(f"Error moving {source_path}: {e}")
                    
            print(f"Completed moving files to {folder}")
        
        print("File organization complete!")

def main():
    # 기본 디렉토리 설정
    default_dir = r"E:\Google231213"
    
    # 사용자에게 디렉토리 입력 받기
    print(f"Enter source directory [default: {default_dir}]:")
    input_dir = input()
    
    # 입력이 없으면 기본 디렉토리 사용
    root_dir = input_dir if input_dir.strip() else default_dir
    
    print(f"Using directory: {root_dir}")
    
    organizer = SimpleFileOrganizer(root_dir)
    organizer.scan_directory()
    organizer.create_simple_vectors()
    organizer.get_target_folders()
    organizer.organize_files()

if __name__ == "__main__":
    main()