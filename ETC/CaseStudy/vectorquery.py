import os
import json
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

class VectorQuery:
    def __init__(self, vector_db_dir="vector_db", model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        벡터 쿼리 클래스 초기화
        
        Args:
            vector_db_dir (str): 벡터 데이터베이스 디렉토리
            model_name (str): 사용할 Sentence Transformer 모델
        """
        self.vector_db_dir = vector_db_dir
        self.model_name = model_name
        self.vectorizer = None
        self.indices = {}
        self.case_studies = []
        
        # 섹션 이모지 정의
        self.section_emoji = {
            'title': '📝',
            'who': '👥',
            'problem': '❓',
            'solution': '💡', 
            'results': '✅',
            'full_text': '📑'
        }
        
        # 산업 이모지 정의
        self.industry_emoji = {
            "통신": "📱",
            "금융": "💰",
            "제조": "🏭",
            "공공": "🏛️",
            "서비스": "🛎️",
            "기타": "📊"
        }
    
    def load(self):
        """벡터 데이터베이스와 모델 로드"""
        # 벡터 데이터베이스 디렉토리 확인
        if not os.path.exists(self.vector_db_dir):
            print(f"오류: 벡터 데이터베이스 디렉토리 '{self.vector_db_dir}'가 존재하지 않습니다.")
            return False
        
        try:
            # 모델 로드
            print(f"모델 '{self.model_name}' 로딩 중...")
            self.vectorizer = SentenceTransformer(self.model_name)
            
            # 케이스 스터디 메타데이터 로드
            print("케이스 스터디 메타데이터 로드 중...")
            metadata_path = os.path.join(self.vector_db_dir, 'case_studies.json')
            if not os.path.exists(metadata_path):
                print(f"오류: 메타데이터 파일 '{metadata_path}'를 찾을 수 없습니다.")
                return False
                
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.case_studies = json.load(f)
            
            # FAISS 인덱스 로드
            print("FAISS 인덱스 로드 중...")
            for filename in os.listdir(self.vector_db_dir):
                if filename.endswith('_index.faiss'):
                    section = filename.replace('_index.faiss', '')
                    index_path = os.path.join(self.vector_db_dir, filename)
                    self.indices[section] = faiss.read_index(index_path)
                    print(f"  - {section} 인덱스 로드 완료")
            
            print(f"로드 완료: {len(self.case_studies)}개의 케이스 스터디, {len(self.indices)}개의 인덱스")
            return True
            
        except Exception as e:
            print(f"로드 중 오류 발생: {str(e)}")
            return False
    
    def query(self, query_text, section="full_text", k=5, print_results=True, export_csv=None):
        """
        벡터 쿼리 실행
        
        Args:
            query_text (str): 검색할 쿼리 텍스트
            section (str): 검색할 섹션 (title, who, problem, solution, results, full_text)
            k (int): 반환할 결과 수
            print_results (bool): 결과를 콘솔에 출력할지 여부
            export_csv (str): 결과를 저장할 CSV 파일 경로 (선택 사항)
            
        Returns:
            list: 검색 결과 리스트
        """
        # 인덱스 확인
        if section not in self.indices:
            available_sections = ", ".join(self.indices.keys())
            print(f"오류: '{section}' 섹션에 대한 인덱스가 없습니다. 사용 가능한 섹션: {available_sections}")
            return []
        
        # 쿼리 벡터화
        query_vector = self.vectorizer.encode([query_text])[0].reshape(1, -1).astype('float32')
        
        # FAISS로 검색
        distances, indices = self.indices[section].search(query_vector, k)
        
        # 결과 구성
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.case_studies) and idx >= 0:
                case = self.case_studies[idx]
                result = {
                    'rank': i + 1,
                    'distance': float(distance),
                    'similarity': 1.0 / (1.0 + float(distance)),  # 거리를 유사도로 변환 (0-1 사이 값)
                    'case_study': case
                }
                results.append(result)
        
        # 결과 출력
        if print_results:
            self._print_results(results, query_text, section)
        
        # CSV로 내보내기
        if export_csv:
            self._export_to_csv(results, export_csv)
        
        return results
    
    def _print_results(self, results, query_text, section):
        """검색 결과를 콘솔에 출력"""
        if not results:
            print("검색 결과가 없습니다.")
            return
        
        section_emoji = self.section_emoji.get(section, '🔍')
        
        print("\n" + "=" * 80)
        print(f"{section_emoji} '{section}' 섹션에서 '{query_text}'에 대한 검색 결과")
        print("=" * 80)
        
        for result in results:
            case = result['case_study']
            similarity = result['similarity'] * 100  # 백분율로 변환
            
            industry = case.get('industry', '기타')
            industry_emoji = self.industry_emoji.get(industry, '📄')
            
            print(f"\n{result['rank']}. {industry_emoji} {case['title']}")
            print(f"   유사도: {similarity:.1f}%")
            print(f"   ID: {case['id']}")
            print(f"   산업: {industry}")
            print(f"\n   {self.section_emoji.get('who', '')} Who: {case['who'][:150]}..." if len(case['who']) > 150 else f"\n   {self.section_emoji.get('who', '')} Who: {case['who']}")
            print(f"   {self.section_emoji.get('problem', '')} Problem: {case['problem'][:150]}..." if len(case['problem']) > 150 else f"   {self.section_emoji.get('problem', '')} Problem: {case['problem']}")
            print(f"   {self.section_emoji.get('solution', '')} Solution: {case['solution'][:150]}..." if len(case['solution']) > 150 else f"   {self.section_emoji.get('solution', '')} Solution: {case['solution']}")
            print(f"   {self.section_emoji.get('results', '')} Results: {case['results'][:150]}..." if len(case['results']) > 150 else f"   {self.section_emoji.get('results', '')} Results: {case['results']}")
            
            print("\n" + "-" * 80)
    
    def _export_to_csv(self, results, csv_path):
        """검색 결과를 CSV 파일로 내보내기"""
        if not results:
            print(f"내보낼 결과가 없습니다.")
            return
        
        # 결과 데이터 구성
        data = []
        for result in results:
            case = result['case_study']
            row = {
                'rank': result['rank'],
                'similarity': result['similarity'] * 100,  # 백분율로 변환
                'id': case['id'],
                'title': case['title'],
                'industry': case.get('industry', '기타'),
                'who': case['who'],
                'problem': case['problem'],
                'solution': case['solution'],
                'results': case['results']
            }
            data.append(row)
        
        # DataFrame 생성 및 CSV 저장
        df = pd.DataFrame(data)
        
        # 디렉토리 확인
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
        
        # CSV로 저장
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig는 Excel에서 한글이 깨지지 않도록 BOM 포함
        print(f"검색 결과를 '{csv_path}'에 저장했습니다.")

    def export_case_study_original(self, case_id, output_file=None):
        """케이스 스터디 원본 텍스트 내보내기"""
        # 케이스 스터디 찾기
        case_study = None
        for case in self.case_studies:
            if case['id'] == case_id:
                case_study = case
                break
        
        if not case_study:
            print(f"오류: ID '{case_id}'에 해당하는 케이스 스터디를 찾을 수 없습니다.")
            return None
        
        # 원본 텍스트 구성
        original_text = f"제목: {case_study['title']}\n\n"
        original_text += f"Who: {case_study['who']}\n\n"
        original_text += f"Problem: {case_study['problem']}\n\n"
        original_text += f"Solution: {case_study['solution']}\n\n"
        original_text += f"Results: {case_study['results']}\n"
        
        # 파일로 저장 (선택 사항)
        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(original_text)
            print(f"케이스 스터디 '{case_study['title']}'를 '{output_file}'에 저장했습니다.")
        
        return original_text

def main():
    parser = argparse.ArgumentParser(description='벡터 쿼리 프로그램')
    
    # 기본 옵션
    parser.add_argument('--db-dir', type=str, default='vector_db', help='벡터 데이터베이스 디렉토리 (기본값: vector_db)')
    parser.add_argument('--model', type=str, default='paraphrase-multilingual-MiniLM-L12-v2', help='사용할 모델 (기본값: paraphrase-multilingual-MiniLM-L12-v2)')
    
    # 서브커맨드 설정
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # 쿼리 커맨드
    query_parser = subparsers.add_parser('query', help='벡터 쿼리 실행')
    query_parser.add_argument('--text', type=str, required=True, help='검색할 쿼리 텍스트')
    query_parser.add_argument('--section', type=str, default='full_text', 
                             choices=['title', 'who', 'problem', 'solution', 'results', 'full_text'],
                             help='검색할 섹션 (기본값: full_text)')
    query_parser.add_argument('--top', type=int, default=5, help='반환할 결과 수 (기본값: 5)')
    query_parser.add_argument('--csv', type=str, help='결과를 저장할 CSV 파일 경로 (선택 사항)')
    
    # 원본 텍스트 내보내기 커맨드
    export_parser = subparsers.add_parser('export', help='케이스 스터디 원본 텍스트 내보내기')
    export_parser.add_argument('--id', type=str, required=True, help='케이스 스터디 ID (예: cate_3_idx_324)')
    export_parser.add_argument('--output', type=str, help='저장할 파일 경로 (선택 사항)')
    
    # 대화형 모드 커맨드
    subparsers.add_parser('interactive', help='대화형 모드 실행')
    
    args = parser.parse_args()
    
    # 벡터 쿼리 객체 생성 및 로드
    query_engine = VectorQuery(args.db_dir, args.model)
    if not query_engine.load():
        return
    
    # 명령어에 따라 실행
    if args.command == 'query':
        query_engine.query(args.text, args.section, args.top, True, args.csv)
    
    elif args.command == 'export':
        text = query_engine.export_case_study_original(args.id, args.output)
        if text and not args.output:
            print("\n원본 텍스트:")
            print("=" * 80)
            print(text)
    
    elif args.command == 'interactive' or args.command is None:
        # 대화형 모드 실행
        run_interactive_mode(query_engine)

def run_interactive_mode(query_engine):
    """대화형 모드 실행"""
    print("\n" + "=" * 80)
    print("🔍 케이스 스터디 벡터 검색 시스템 - 대화형 모드")
    print("=" * 80)
    
    while True:
        print("\n사용 가능한 명령어:")
        print("1. 검색 - 벡터 쿼리 실행")
        print("2. 내보내기 - 케이스 스터디 원본 텍스트 내보내기")
        print("3. 종료 - 프로그램 종료")
        
        choice = input("\n명령어를 선택하세요 (1-3): ")
        
        if choice == '1':
            # 검색 실행
            print("\n== 벡터 쿼리 실행 ==")
            
            # 섹션 선택
            sections = ['title', 'who', 'problem', 'solution', 'results', 'full_text']
            section_names = {
                'title': '📝 제목',
                'who': '👥 회사/기관',
                'problem': '❓ 문제',
                'solution': '💡 솔루션',
                'results': '✅ 결과',
                'full_text': '📑 전체 텍스트'
            }
            
            print("\n검색할 섹션을 선택하세요:")
            for i, section in enumerate(sections):
                print(f"{i+1}. {section_names.get(section, section)}")
            
            section_choice = input("섹션 번호를 입력하세요 (기본값: 6): ") or "6"
            try:
                section_idx = int(section_choice) - 1
                if 0 <= section_idx < len(sections):
                    section = sections[section_idx]
                else:
                    section = 'full_text'
            except ValueError:
                section = 'full_text'
            
            # 쿼리 텍스트 입력
            query_text = input("\n검색할 텍스트를 입력하세요: ")
            if not query_text.strip():
                print("검색어를 입력해야 합니다.")
                continue
            
            # 결과 수 입력
            try:
                k = int(input("반환할 결과 수를 입력하세요 (기본값: 5): ") or "5")
            except ValueError:
                k = 5
            
            # CSV 내보내기 여부
            export_csv = input("결과를 CSV로 저장하시겠습니까? (y/n, 기본값: n): ").lower() == 'y'
            csv_path = None
            if export_csv:
                csv_path = input("CSV 파일 경로를 입력하세요 (기본값: search_results.csv): ") or "search_results.csv"
            
            # 쿼리 실행
            results = query_engine.query(query_text, section, k, True, csv_path)
            
            # 케이스 스터디 원본 내보내기 여부
            if results:
                export_original = input("\n케이스 스터디 원본을 내보내시겠습니까? (y/n, 기본값: n): ").lower() == 'y'
                if export_original:
                    # 케이스 스터디 선택
                    try:
                        case_idx = int(input("내보낼 케이스 스터디 번호를 입력하세요 (1부터 시작): "))
                        if 1 <= case_idx <= len(results):
                            case_id = results[case_idx-1]['case_study']['id']
                            output_path = input("저장할 파일 경로를 입력하세요 (기본값: case_study_original.txt): ") or "case_study_original.txt"
                            query_engine.export_case_study_original(case_id, output_path)
                    except ValueError:
                        print("올바른 번호를 입력하세요.")
        
        elif choice == '2':
            # 원본 텍스트 내보내기
            print("\n== 케이스 스터디 원본 텍스트 내보내기 ==")
            case_id = input("케이스 스터디 ID를 입력하세요 (예: cate_3_idx_324): ")
            if not case_id:
                print("ID를 입력해야 합니다.")
                continue
            
            output_path = input("저장할 파일 경로를 입력하세요 (기본값: case_study_original.txt): ") or "case_study_original.txt"
            query_engine.export_case_study_original(case_id, output_path)
        
        elif choice == '3':
            print("\n프로그램을 종료합니다.")
            break
        
        else:
            print("\n올바른 명령어를 선택하세요.")

if __name__ == "__main__":
    main()