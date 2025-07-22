import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import re
import emoji

# 1. 데이터 로딩 및 전처리
def load_case_studies(directory="case_studies"):
    """
    지정된 디렉토리에서 모든 케이스 스터디 텍스트 파일을 읽어 구조화된 데이터로 변환합니다.
    
    Args:
        directory (str): 케이스 스터디 파일이 있는 디렉토리 경로
        
    Returns:
        list: 구조화된 케이스 스터디 데이터 리스트
    """
    case_studies = []
    
    if not os.path.exists(directory):
        print(f"디렉토리가 존재하지 않습니다: {directory}")
        return case_studies
    
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    print(f"총 {len(files)}개의 케이스 스터디 파일을 발견했습니다.")
    
    for filename in tqdm(files, desc="파일 로딩 중"):
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 파일명에서 cate와 idx 추출
            cate_match = re.search(r'cate_(\d+)', filename)
            idx_match = re.search(r'idx_(\d+)', filename)
            
            cate = int(cate_match.group(1)) if cate_match else None
            idx = int(idx_match.group(1)) if idx_match else None
            
            # 내용 파싱
            sections = {}
            
            # 제목 추출
            title_match = re.search(r'제목:\s*(.*?)(?:\n\n|$)', content)
            title = title_match.group(1).strip() if title_match else "제목 없음"
            sections['title'] = title
            
            # 각 섹션 추출
            for section in ['Who', 'Problem', 'Solution', 'Results']:
                pattern = rf'{section}:\s*(.*?)(?:\n\n(?:[A-Za-z]+:|$)|$)'
                section_match = re.search(pattern, content, re.DOTALL)
                sections[section.lower()] = section_match.group(1).strip() if section_match else ""
            
            # 산업 분야 추정 (카테고리 기반)
            industry_map = {
                3: "통신",
                4: "금융",
                5: "제조",
                6: "공공",
                7: "서비스",
                8: "기타"
            }
            industry = industry_map.get(cate, "기타")
            
            # 케이스 스터디 구조화
            case_study = {
                'id': f"cate_{cate}_idx_{idx}",
                'filename': filename,
                'title': title,
                'who': sections['who'],
                'problem': sections['problem'],
                'solution': sections['solution'],
                'results': sections['results'],
                'full_text': content,
                'cate': cate,
                'idx': idx,
                'industry': industry,
                # 추가 메타데이터
                'emoji': get_industry_emoji(industry)
            }
            
            case_studies.append(case_study)
            
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {file_path}, 오류: {str(e)}")
    
    print(f"{len(case_studies)}개의 케이스 스터디를 로드했습니다.")
    return case_studies

def get_industry_emoji(industry):
    """산업 분야에 맞는 이모지 반환"""
    emoji_map = {
        "통신": "📱",
        "금융": "💰",
        "제조": "🏭",
        "공공": "🏛️",
        "서비스": "🛎️",
        "기타": "📊"
    }
    return emoji_map.get(industry, "📄")

# 2. 벡터 임베딩 생성
class CaseStudyVectorizer:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        케이스 스터디를 벡터화하는 클래스 초기화
        
        Args:
            model_name (str): 사용할 Sentence Transformer 모델 이름
        """
        print(f"모델 {model_name} 로딩 중...")
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        print(f"벡터 차원: {self.vector_size}")
    
    def vectorize_case_studies(self, case_studies):
        """
        모든 케이스 스터디의 섹션별 벡터 임베딩 생성
        
        Args:
            case_studies (list): 케이스 스터디 리스트
            
        Returns:
            dict: 섹션별 벡터 임베딩과 전체 벡터 임베딩
        """
        print("케이스 스터디 벡터화 중...")
        
        # 각 섹션별 텍스트 추출
        sections = ['title', 'who', 'problem', 'solution', 'results']
        section_texts = {section: [] for section in sections}
        
        # 전체 텍스트
        full_texts = []
        
        for case in case_studies:
            for section in sections:
                section_texts[section].append(case[section])
            
            # 전체 텍스트는 모든 섹션을 결합
            combined_text = " ".join([case[section] for section in sections])
            full_texts.append(combined_text)
        
        # 각 섹션별 벡터화
        vectors = {}
        for section in sections:
            print(f"{section} 섹션 벡터화 중...")
            vectors[section] = self.model.encode(section_texts[section], show_progress_bar=True)
        
        # 전체 텍스트 벡터화
        print("전체 텍스트 벡터화 중...")
        vectors['full_text'] = self.model.encode(full_texts, show_progress_bar=True)
        
        return vectors

# 3. FAISS 인덱스 생성 및 저장
class CaseStudyVectorDB:
    def __init__(self, vector_size):
        """
        벡터 데이터베이스 초기화
        
        Args:
            vector_size (int): 벡터의 차원 크기
        """
        self.vector_size = vector_size
        self.indices = {}
        self.case_studies = []
    
    def build_indices(self, case_studies, vectors):
        """
        케이스 스터디와 벡터를 사용하여 FAISS 인덱스 구축
        
        Args:
            case_studies (list): 케이스 스터디 리스트
            vectors (dict): 섹션별 벡터 임베딩
        """
        print("FAISS 인덱스 구축 중...")
        self.case_studies = case_studies
        
        # 각 섹션별 인덱스 생성
        for section, section_vectors in vectors.items():
            print(f"{section} 인덱스 생성 중...")
            index = faiss.IndexFlatL2(self.vector_size)  # L2 거리(유클리드 거리) 사용
            if len(section_vectors) > 0:
                index.add(np.array(section_vectors).astype('float32'))
                self.indices[section] = index
    
    def save(self, directory="vector_db"):
        """
        벡터 데이터베이스를 파일로 저장
        
        Args:
            directory (str): 저장할 디렉토리 경로
        """
        os.makedirs(directory, exist_ok=True)
        
        # 케이스 스터디 메타데이터 저장
        with open(os.path.join(directory, 'case_studies.json'), 'w', encoding='utf-8') as f:
            json.dump(self.case_studies, f, ensure_ascii=False, indent=2)
        
        # FAISS 인덱스 저장
        for section, index in self.indices.items():
            index_path = os.path.join(directory, f"{section}_index.faiss")
            faiss.write_index(index, index_path)
        
        print(f"벡터 데이터베이스가 {directory}에 저장되었습니다.")
    
    @classmethod
    def load(cls, directory="vector_db", vector_size=384):
        """
        저장된 벡터 데이터베이스 로드
        
        Args:
            directory (str): 로드할 디렉토리 경로
            vector_size (int): 벡터 차원 크기
            
        Returns:
            CaseStudyVectorDB: 로드된 벡터 데이터베이스 객체
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"디렉토리가 존재하지 않습니다: {directory}")
        
        # 인스턴스 생성
        instance = cls(vector_size)
        
        # 케이스 스터디 메타데이터 로드
        with open(os.path.join(directory, 'case_studies.json'), 'r', encoding='utf-8') as f:
            instance.case_studies = json.load(f)
        
        # 가능한 모든 섹션 인덱스 찾기
        for filename in os.listdir(directory):
            if filename.endswith('_index.faiss'):
                section = filename.replace('_index.faiss', '')
                index_path = os.path.join(directory, filename)
                instance.indices[section] = faiss.read_index(index_path)
        
        return instance
    
    def search(self, query, section='full_text', k=5, vectorizer=None):
        """
        쿼리와 유사한 케이스 스터디 검색
        
        Args:
            query (str): 검색 쿼리 텍스트
            section (str): 검색할 섹션 ('title', 'who', 'problem', 'solution', 'results', 'full_text')
            k (int): 반환할 결과 수
            vectorizer (CaseStudyVectorizer): 쿼리 벡터화에 사용할 벡터라이저
            
        Returns:
            list: 검색 결과 리스트
        """
        if section not in self.indices:
            raise ValueError(f"섹션 '{section}'에 대한 인덱스가 없습니다.")
        
        if vectorizer is None:
            raise ValueError("쿼리 벡터화를 위한 vectorizer가 필요합니다.")
        
        # 쿼리 벡터화
        query_vector = vectorizer.model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # FAISS로 검색
        distances, indices = self.indices[section].search(query_vector, k)
        
        # 결과 구성
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.case_studies):
                case = self.case_studies[idx]
                result = {
                    'rank': i + 1,
                    'distance': float(distance),
                    'similarity': 1.0 / (1.0 + float(distance)),  # 거리를 유사도로 변환
                    'case_study': case
                }
                results.append(result)
        
        return results

# 4. 검색 및 표시 기능
def format_search_results(results, highlight_query=None):
    """
    검색 결과를 보기 좋게 형식화
    
    Args:
        results (list): 검색 결과 리스트
        highlight_query (str): 하이라이트할 쿼리 (선택 사항)
        
    Returns:
        str: 형식화된 검색 결과 문자열
    """
    if not results:
        return "검색 결과가 없습니다."
    
    output = "# 🔍 검색 결과\n\n"
    
    for result in results:
        case = result['case_study']
        similarity = result['similarity'] * 100
        
        industry_emoji = case.get('emoji', '📄')
        
        output += f"## {industry_emoji} {case['title']} (유사도: {similarity:.1f}%)\n\n"
        output += f"**산업:** {case['industry']}\n"
        output += f"**ID:** {case['id']}\n\n"
        
        # 섹션별 내용 표시
        for section in ['who', 'problem', 'solution', 'results']:
            content = case[section]
            
            # 쿼리 하이라이트 (선택 사항)
            if highlight_query and highlight_query.strip():
                content = content.replace(highlight_query, f"**{highlight_query}**")
            
            section_emoji = {
                'who': '👥',
                'problem': '❓',
                'solution': '💡', 
                'results': '✅'
            }
            
            emoji_icon = section_emoji.get(section, '')
            output += f"### {emoji_icon} {section.capitalize()}\n{content}\n\n"
        
        output += "---\n\n"
    
    return output

def interactive_search():
    """
    대화형 검색 인터페이스 실행
    """
    # 초기 디렉토리 설정
    source_dir = "case_studies"
    vector_db_dir = "vector_db"
    
    # 필요한 컴포넌트 로드
    print("벡터 데이터베이스 로드 중...")
    
    # 벡터 DB 로드 도우미 함수
    def load_vector_db():
        if not os.path.exists(vector_db_dir) or not os.listdir(vector_db_dir):
            print(f"벡터 데이터베이스를 찾을 수 없습니다. ({vector_db_dir})")
            print("벡터 데이터베이스 구축 옵션(7)을 선택하여 먼저 데이터베이스를 구축하세요.")
            return None, None
            
        try:
            # 사용할 모델 지정
            model_name = "paraphrase-multilingual-MiniLM-L12-v2"
            
            # 벡터라이저 초기화
            vectorizer = CaseStudyVectorizer(model_name)
            
            # 벡터 데이터베이스 로드
            db = CaseStudyVectorDB.load(vector_db_dir, vectorizer.vector_size)
            print("벡터 데이터베이스 로드 완료!")
            return db, vectorizer
        except Exception as e:
            print(f"벡터 데이터베이스 로드 중 오류 발생: {str(e)}")
            return None, None
    
    # 초기 로드 시도
    db, vectorizer = load_vector_db()
    
    # 검색 옵션 표시 함수
    def display_menu():
        print("\n========== 케이스 스터디 검색 시스템 ==========")
        print("다음 중 하나를 선택하여 검색할 수 있습니다:")
        print("1. '전체 텍스트' - 모든 섹션에서 검색")
        print("2. '제목' - 제목에서만 검색")
        print("3. '문제' - Problem 섹션에서만 검색")
        print("4. '솔루션' - Solution 섹션에서만 검색")
        print("5. '결과' - Results 섹션에서만 검색")
        print("6. '벡터 데이터 분석' - 벡터 데이터베이스 통계 및 시각화")
        print("7. '벡터데이터 재구축' - 벡터 데이터베이스 재구축")
        print("8. '소스 디렉토리 설정' - 케이스 스터디 소스 디렉토리 설정")
        print("9. '타겟 벡터스토어 설정' - 벡터 데이터베이스 저장 위치 설정")
        print("10. '나가기' - 프로그램 종료")
        print("현재 설정:")
        print(f"- 소스 디렉토리: {source_dir}")
        print(f"- 벡터스토어 위치: {vector_db_dir}")
        print("=" * 50)
        

    def analyze_vector_data():
        if db is None or vectorizer is None:
            print("벡터 데이터베이스가 로드되지 않았습니다.")
            print("옵션 7을 사용하여 먼저 데이터베이스를 구축하세요.")
            return
        
        print("\n========== 벡터 데이터베이스 분석 ==========")
        
        # 기본 통계 정보 표시
        total_cases = len(db.case_studies)
        print(f"총 케이스 스터디 수: {total_cases}")
        print(f"벡터 차원: {vectorizer.vector_size}")
        
        # 섹션별 인덱스 정보
        print("\n섹션별 벡터 데이터:")
        for section, index in db.indices.items():
            vector_count = index.ntotal
            print(f"- {section}: {vector_count}개 벡터")
        
        # 카테고리별 통계
        industry_counts = {}
        category_counts = {}
        for case in db.case_studies:
            industry = case.get('industry', '미분류')
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            
            cate = case.get('cate', '미분류')
            category_counts[cate] = category_counts.get(cate, 0) + 1
        
        print("\n산업별 케이스 수:")
        for industry, count in industry_counts.items():
            print(f"- {industry}: {count}개 ({count/total_cases*100:.1f}%)")
        
        print("\n카테고리별 케이스 수:")
        for cate, count in category_counts.items():
            print(f"- 카테고리 {cate}: {count}개 ({count/total_cases*100:.1f}%)")
        
        # 상세 분석 옵션 표시
        print("\n데이터 시각화 옵션:")
        print("1. 벡터 분포 시각화 (2D)")
        print("2. 산업별 케이스 분포")
        print("3. 카테고리별 케이스 분포")
        print("4. 섹션별 데이터 길이 분석")
        print("5. 돌아가기")
        
        viz_option = input("\n시각화 옵션을 선택하세요 (1-5): ")
        
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 한글 폰트 설정 시도
            try:
                # 윈도우의 경우
                if os.name == 'nt':
                    font_options = [
                        'Malgun Gothic',  # 맑은 고딕
                        'Gulim',          # 굴림
                        'Batang',         # 바탕
                        'Arial Unicode MS'
                    ]
                    
                    for font in font_options:
                        try:
                            plt.rcParams['font.family'] = font
                            test_text = '테스트'
                            fig, ax = plt.subplots()
                            ax.text(0.5, 0.5, test_text)
                            fig.savefig('test_font.png')
                            plt.close(fig)
                            print(f"한글 폰트 '{font}' 설정 성공")
                            break
                        except Exception:
                            continue
                
                # macOS의 경우
                elif sys.platform == 'darwin':
                    plt.rcParams['font.family'] = 'AppleGothic'
                    print("macOS 한글 폰트 설정")
                
                # 리눅스의 경우
                else:
                    plt.rcParams['font.family'] = 'NanumGothic'
                    print("Linux 한글 폰트 설정")
                    
            except Exception as e:
                print(f"한글 폰트 설정 실패: {str(e)}")
                print("영문으로 시각화를 진행합니다.")
            
            # 시각화 옵션 처리
            if viz_option == '1':  # 벡터 분포 시각화
                print("\n벡터 분포를 시각화합니다.")
                
                # 섹션 선택
                print("시각화할 섹션을 선택하세요:")
                sections = list(db.indices.keys())
                for i, section in enumerate(sections):
                    print(f"{i+1}. {section}")
                
                section_idx = int(input("선택 (번호): ")) - 1
                if 0 <= section_idx < len(sections):
                    selected_section = sections[section_idx]
                    print(f"\n{selected_section} 섹션의 벡터를 시각화합니다.")
                    
                    # 원본 텍스트에서 벡터 생성 (FAISS 인덱스 대신)
                    print("원본 텍스트에서 벡터를 재생성합니다.")
                    
                    # 섹션별 텍스트 수집
                    section_texts = []
                    for case in db.case_studies:
                        if selected_section == 'full_text':
                            # 전체 텍스트는 모든 섹션 결합
                            text = " ".join([case.get(s, "") for s in ['title', 'who', 'problem', 'solution', 'results']])
                        else:
                            text = case.get(selected_section, "")
                        section_texts.append(text)
                    
                    # 벡터 생성
                    print(f"{len(section_texts)}개 텍스트의 벡터를 생성 중...")
                    vectors = vectorizer.model.encode(section_texts, show_progress_bar=True)
                    
                    # PCA로 차원 축소
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    vectors_2d = pca.fit_transform(vectors)
                    
                    explained_variance = pca.explained_variance_ratio_
                    print(f"PCA 설명된 분산: {explained_variance[0]*100:.1f}%, {explained_variance[1]*100:.1f}%")
                    
                    # 그래프 생성
                    plt.figure(figsize=(12, 10))
                    
                    # 산업별 색상 및 마커 정의
                    industry_colors = {
                        "통신": ("blue", "o"),      # 원형
                        "금융": ("green", "s"),     # 사각형
                        "제조": ("red", "^"),       # 삼각형
                        "공공": ("purple", "d"),    # 다이아몬드
                        "서비스": ("orange", "v"),  # 역삼각형
                        "기타": ("gray", "p")       # 오각형
                    }
                    
                    # 산업별 벡터 그룹화 및 그리기
                    industry_groups = {}
                    for i, (x, y) in enumerate(vectors_2d):
                        if i < len(db.case_studies):
                            industry = db.case_studies[i].get('industry', '기타')
                            if industry not in industry_groups:
                                industry_groups[industry] = []
                            industry_groups[industry].append((x, y))
                    
                    # 산업별 그룹 그리기
                    for industry, points in industry_groups.items():
                        if points:
                            points = np.array(points)
                            color, marker = industry_colors.get(industry, ("black", "o"))
                            plt.scatter(
                                points[:, 0], 
                                points[:, 1], 
                                label=f"{industry} ({len(points)}개)",
                                color=color,
                                marker=marker,
                                alpha=0.7,
                                s=100  # 점 크기
                            )
                    
                    # 영문 제목 사용 (한글 폰트 문제 방지)
                    section_names_en = {
                        'title': 'Title',
                        'who': 'Who',
                        'problem': 'Problem',
                        'solution': 'Solution',
                        'results': 'Results',
                        'full_text': 'Full Text'
                    }
                    
                    section_en = section_names_en.get(selected_section, selected_section)
                    plt.title(f"Vector Distribution of {section_en} Section (PCA 2D)")
                    plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.1f}%)")
                    plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.1f}%)")
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # 저장 및 표시
                    viz_path = f"vector_visualization_{selected_section}.png"
                    plt.savefig(viz_path, dpi=300)
                    plt.close()
                    
                    print(f"벡터 시각화 이미지가 '{viz_path}'에 저장되었습니다.")
                
            elif viz_option == '2':  # 산업별 케이스 분포
                print("\n산업별 케이스 분포를 시각화합니다.")
                
                # 파이 차트 생성
                plt.figure(figsize=(10, 8))
                
                # 산업별 색상 정의
                industry_colors = {
                    "통신": "blue",
                    "금융": "green",
                    "제조": "red",
                    "공공": "purple",
                    "서비스": "orange",
                    "기타": "gray"
                }
                
                # 데이터 준비
                industries = list(industry_counts.keys())
                counts = [industry_counts[industry] for industry in industries]
                colors = [industry_colors.get(industry, "black") for industry in industries]
                
                # 파이 차트 그리기
                plt.pie(
                    counts, 
                    labels=[f"{ind} ({cnt})" for ind, cnt in zip(industries, counts)],
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    shadow=True,
                    explode=[0.05] * len(industries)  # 약간 분리
                )
                
                plt.title("Distribution of Case Studies by Industry")
                plt.axis('equal')  # 원형 유지
                
                # 저장
                viz_path = "industry_distribution.png"
                plt.savefig(viz_path, dpi=300)
                plt.close()
                
                print(f"산업별 분포 시각화 이미지가 '{viz_path}'에 저장되었습니다.")
                
            elif viz_option == '3':  # 카테고리별 케이스 분포
                print("\n카테고리별 케이스 분포를 시각화합니다.")
                
                # 막대 그래프 생성
                plt.figure(figsize=(12, 8))
                
                # 데이터 준비
                categories = sorted(category_counts.keys())
                counts = [category_counts[cat] for cat in categories]
                
                # 막대 그래프 그리기
                bars = plt.bar(
                    categories, 
                    counts,
                    color='skyblue',
                    edgecolor='navy'
                )
                
                # 수치 표시
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.5,
                        f'{int(height)}',
                        ha='center', 
                        va='bottom'
                    )
                
                plt.title("Distribution of Case Studies by Category")
                plt.xlabel("Category")
                plt.ylabel("Number of Cases")
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # 저장
                viz_path = "category_distribution.png"
                plt.savefig(viz_path, dpi=300)
                plt.close()
                
                print(f"카테고리별 분포 시각화 이미지가 '{viz_path}'에 저장되었습니다.")
                
            elif viz_option == '4':  # 섹션별 데이터 길이 분석
                print("\n섹션별 데이터 길이를 분석합니다.")
                
                # 섹션별 텍스트 길이 계산
                sections = ['title', 'who', 'problem', 'solution', 'results']
                section_lengths = {section: [] for section in sections}
                
                for case in db.case_studies:
                    for section in sections:
                        text = case.get(section, "")
                        section_lengths[section].append(len(text))
                
                # 박스 플롯 생성
                plt.figure(figsize=(14, 8))
                
                # 데이터 준비
                data = [section_lengths[section] for section in sections]
                
                # 영문 섹션 이름 (한글 폰트 문제 방지)
                section_names_en = ['Title', 'Who', 'Problem', 'Solution', 'Results']
                
                # 박스 플롯 그리기
                box = plt.boxplot(
                    data,
                    patch_artist=True,
                    labels=section_names_en,
                    notch=True,
                    whis=1.5
                )
                
                # 박스 색상 설정
                colors = ['lightblue', 'lightgreen', 'salmon', 'violet', 'wheat']
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                
                # 통계 정보 표시
                for i, section in enumerate(sections):
                    lengths = section_lengths[section]
                    avg_len = sum(lengths) / len(lengths)
                    max_len = max(lengths)
                    min_len = min(lengths)
                    
                    plt.text(
                        i + 1,
                        max(lengths) + 5,
                        f'Avg: {avg_len:.1f}\nMax: {max_len}\nMin: {min_len}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
                    )
                
                plt.title("Length Distribution of Text by Section")
                plt.xlabel("Section")
                plt.ylabel("Character Count")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # 저장
                viz_path = "section_length_analysis.png"
                plt.savefig(viz_path, dpi=300)
                plt.close()
                
                print(f"섹션별 길이 분석 시각화 이미지가 '{viz_path}'에 저장되었습니다.")
                
            elif viz_option == '5':  # 돌아가기
                return
            else:
                print("올바른 옵션을 선택하세요 (1-5)")
                
        except ImportError as e:
            print(f"시각화에 필요한 패키지가 설치되지 않았습니다: {str(e)}")
            print("필요한 패키지를 설치하세요: pip install matplotlib scikit-learn numpy")
        except Exception as e:
            print(f"시각화 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            

    while True:
        display_menu()
        search_option = input("검색 옵션을 선택하세요 (1-10): ")
        
        if search_option == '10':
            print("검색을 종료합니다.")
            break
        
        elif search_option == '6':
            analyze_vector_data()
            continue
        
        elif search_option == '7':
            print("\n벡터 데이터베이스를 재구축합니다...")
            try:
                build_vector_db(source_dir, vector_db_dir)
                print("벡터 데이터베이스 재구축 완료!")
                # 새로 구축된 DB 로드
                db, vectorizer = load_vector_db()
            except Exception as e:
                print(f"벡터 데이터베이스 구축 중 오류 발생: {str(e)}")
            continue
            
        elif search_option == '8':
            new_source_dir = input(f"새 소스 디렉토리 경로를 입력하세요 (현재: {source_dir}): ")
            if os.path.exists(new_source_dir):
                source_dir = new_source_dir
                print(f"소스 디렉토리가 '{source_dir}'로 설정되었습니다.")
            else:
                print(f"오류: 디렉토리 '{new_source_dir}'가 존재하지 않습니다.")
            continue
            
        elif search_option == '9':
            new_vector_db_dir = input(f"새 벡터스토어 위치를 입력하세요 (현재: {vector_db_dir}): ")
            vector_db_dir = new_vector_db_dir
            print(f"벡터스토어 위치가 '{vector_db_dir}'로 설정되었습니다.")
            
            # 새 경로가 존재하는지 확인하고, 존재하면 DB 다시 로드
            if os.path.exists(vector_db_dir) and os.listdir(vector_db_dir):
                print("새 위치에서 벡터 데이터베이스를 로드합니다...")
                db, vectorizer = load_vector_db()
            else:
                print("새 위치에는 벡터 데이터베이스가 없습니다. 옵션 7을 사용하여 구축하세요.")
                db, vectorizer = None, None
            continue
        
        # 검색 관련 옵션 (1-5)
        section_map = {
            '1': 'full_text',
            '2': 'title',
            '3': 'problem',
            '4': 'solution',
            '5': 'results'
        }
        
        if search_option not in section_map:
            print("올바른 옵션을 선택하세요 (1-10)")
            continue
        
        # 벡터 DB가 로드되었는지 확인
        if db is None or vectorizer is None:
            print("벡터 데이터베이스가 로드되지 않았습니다.")
            print("옵션 7을 사용하여 먼저 데이터베이스를 구축하세요.")
            continue
        
        section = section_map[search_option]
        
        # 쿼리 입력
        query = input("검색할 내용을 입력하세요: ")
        if not query.strip():
            print("검색어를 입력하세요.")
            continue
        
        # 결과 수 입력
        try:
            k = int(input("반환할 결과 수를 입력하세요 (기본값: 3): ") or "3")
        except ValueError:
            k = 3
        
        # 검색 실행
        print(f"\n'{section}' 섹션에서 '{query}'에 대한 검색 결과:")
        try:
            results = db.search(query, section=section, k=k, vectorizer=vectorizer)
            
            if not results:
                print("검색 결과가 없습니다.")
                continue
            
            # 결과 표시
            for i, result in enumerate(results):
                case = result['case_study']
                similarity = result['similarity'] * 100
                
                industry_emoji = case.get('emoji', '📄')
                print(f"\n{i+1}. {industry_emoji} {case['title']} (유사도: {similarity:.1f}%)")
                print(f"   산업: {case['industry']}")
                print(f"   ID: {case['id']}")
                
                # 자세한 정보 표시 여부
                show_details = input("   자세한 정보를 보시겠습니까? (y/n): ").lower() == 'y'
                if show_details:
                    for section_name in ['who', 'problem', 'solution', 'results']:
                        section_emoji = {
                            'who': '👥',
                            'problem': '❓',
                            'solution': '💡', 
                            'results': '✅'
                        }
                        emoji_icon = section_emoji.get(section_name, '')
                        print(f"\n   {emoji_icon} {section_name.capitalize()}:")
                        print(f"   {case[section_name]}")
        except Exception as e:
            print(f"검색 중 오류 발생: {str(e)}")


# 메인 함수: 벡터 데이터베이스 구축
def build_vector_db(case_studies_dir="case_studies", vector_db_dir="vector_db"):
    """
    케이스 스터디를 로드하고 벡터 데이터베이스를 구축하는 메인 함수
    
    Args:
        case_studies_dir (str): 케이스 스터디 파일이 있는 디렉토리
        vector_db_dir (str): 벡터 데이터베이스를 저장할 디렉토리
    """
    # 1. 케이스 스터디 로드
    case_studies = load_case_studies(case_studies_dir)
    
    if not case_studies:
        print("로드된 케이스 스터디가 없습니다. 종료합니다.")
        return
    
    # 2. 벡터라이저 초기화 및 벡터 생성
    # 다국어 지원 모델 사용 (한국어 처리를 위해)
    vectorizer = CaseStudyVectorizer("paraphrase-multilingual-MiniLM-L12-v2")
    vectors = vectorizer.vectorize_case_studies(case_studies)
    
    # 3. 벡터 데이터베이스 구축 및 저장
    db = CaseStudyVectorDB(vectorizer.vector_size)
    db.build_indices(case_studies, vectors)
    db.save(vector_db_dir)
    
    print(f"벡터 데이터베이스가 {vector_db_dir}에 성공적으로 구축되었습니다.")
    print(f"총 {len(case_studies)}개의 케이스 스터디가 처리되었습니다.")
    
    return db, vectorizer




# 메인 함수: 코드 실행 부분
if __name__ == "__main__":
    # 명령줄 인자 파싱
    import argparse
    
    parser = argparse.ArgumentParser(description='Altibase 케이스 스터디 벡터 검색 시스템')
    parser.add_argument('--build', action='store_true', help='벡터 데이터베이스 구축')
    parser.add_argument('--search', action='store_true', help='대화형 검색 모드 실행')
    parser.add_argument('--input-dir', type=str, default='case_studies', help='케이스 스터디 디렉토리')
    parser.add_argument('--output-dir', type=str, default='vector_db', help='벡터 데이터베이스 디렉토리')
    
    args = parser.parse_args()
    
    if args.build:
        build_vector_db(args.input_dir, args.output_dir)
    
    if args.search or not (args.build or args.search):
        interactive_search()

        

        