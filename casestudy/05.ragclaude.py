import os
import argparse
import json
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
import anthropic
import requests

# VectorQuery 클래스 임포트
from vectorquery import VectorQuery

class CaseStudyLLM:
    def __init__(self, model_type: str = "claude"):
        """
        Claude API를 사용하여 케이스 스터디 생성을 위한 클래스 초기화
        
        Args:
            model_type (str): 사용할 모델 유형 (기본값: "claude")
        """
        self.model_type = model_type.lower()
        self.client = None
        self.vector_query = VectorQuery()
        
        # 클라우드 서비스 설정
        if self.model_type == "claude":
            print(f"LLM 모델 유형: {self.model_type}")
            
            # API 키 가져오기
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("환경 변수 'ANTHROPIC_API_KEY'가 설정되어 있지 않습니다.")
        else:
            raise ValueError(f"지원되지 않는 모델 유형: {self.model_type}")
    
    def load_model(self):
        """모델 로드 (API 클라이언트 설정)"""
        try:
            if self.model_type == "claude":
                print("Claude API 클라이언트 설정 중...")
                self.client = anthropic.Anthropic(api_key=self.api_key)
                print("Claude API 클라이언트 설정 완료!")
                
                # 사용 가능한 모델 목록 가져오기
                try:
                    print("사용 가능한 Claude 모델:")
                    # 최신 Claude 모델 목록
                    models = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
                    for i, model in enumerate(models, 1):
                        print(f"{i}. {model}")
                except Exception as e:
                    print(f"모델 목록 가져오기 실패: {str(e)}")
                
                return True
            else:
                raise ValueError(f"지원되지 않는 모델 유형: {self.model_type}")
            
        except Exception as e:
            print(f"API 클라이언트 설정 중 오류 발생: {str(e)}")
            return False
    
    def load_vector_db(self, vector_db_dir: str = "vector_db"):
        """벡터 데이터베이스 로드"""
        print(f"벡터 데이터베이스 로드 중: {vector_db_dir}")
        return self.vector_query.load()
    
    def query_similar_cases(self, 
                           title: str, 
                           who: str, 
                           problem: str, 
                           solution: str, 
                           results: str,
                           k: int = 5) -> List[Dict[str, Any]]:
        """
        유사한 케이스 스터디 검색
        
        Args:
            title, who, problem, solution, results: 케이스 스터디 섹션
            k: 검색할 결과 수
            
        Returns:
            List[Dict]: 유사한 케이스 스터디 목록
        """
        # 섹션별 가중치 (중요도에 따라 조정 가능)
        weights = {
            'title': 0.1,
            'who': 0.1,
            'problem': 0.4,
            'solution': 0.3,
            'results': 0.1
        }
        
        # 각 섹션에 대해 쿼리 실행
        results_by_section = {}
        
        # 전체 텍스트 결합
        full_text = f"{title}\n{who}\n{problem}\n{solution}\n{results}"
        
        # 전체 텍스트 쿼리 (가장 중요)
        print("전체 텍스트 기반으로 유사한 케이스 스터디 검색 중...")
        full_text_results = self.vector_query.query(
            full_text, 
            section='full_text', 
            k=k*2,  # 더 많은 결과 가져오기
            print_results=False
        )
        
        # 섹션별 쿼리
        sections = {
            'problem': problem,
            'solution': solution
        }
        
        for section_name, section_text in sections.items():
            if section_text.strip():
                print(f"{section_name} 섹션 기반으로 유사한 케이스 스터디 검색 중...")
                results_by_section[section_name] = self.vector_query.query(
                    section_text, 
                    section=section_name, 
                    k=k, 
                    print_results=False
                )
        
        # 결과 병합 및 점수 계산
        merged_results = {}
        
        # 전체 텍스트 결과 추가
        for result in full_text_results:
            case_id = result['case_study']['id']
            merged_results[case_id] = {
                'case_study': result['case_study'],
                'score': result['similarity'] * 0.6,  # 전체 텍스트 가중치
                'matched_sections': ['full_text']
            }
        
        # 섹션별 결과 추가
        for section_name, section_results in results_by_section.items():
            for result in section_results:
                case_id = result['case_study']['id']
                if case_id in merged_results:
                    merged_results[case_id]['score'] += result['similarity'] * weights.get(section_name, 0.1)
                    merged_results[case_id]['matched_sections'].append(section_name)
                else:
                    merged_results[case_id] = {
                        'case_study': result['case_study'],
                        'score': result['similarity'] * weights.get(section_name, 0.1),
                        'matched_sections': [section_name]
                    }
        
        # 점수순으로 정렬
        sorted_results = sorted(
            merged_results.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # 상위 k개 결과 반환
        return sorted_results[:k]
    
    def generate_prompt(self, 
                        user_case: Dict[str, str], 
                        similar_cases: List[Dict[str, Any]], 
                        max_similar_cases: int = 3,
                        model_id: str = "claude-3-5-sonnet-20240620") -> str:
        """
        Claude API에 전달할 프롬프트 생성
        
        Args:
            user_case: 사용자가 입력한 케이스 스터디
            similar_cases: 유사한 케이스 스터디 목록
            max_similar_cases: 포함할 최대 유사 케이스 수
            model_id: 사용할 Claude 모델 ID
            
        Returns:
            str: 생성된 프롬프트
        """
        # 시스템 프롬프트
        system_prompt = """당신은 비즈니스 케이스 스터디 작성 전문가입니다. 고객이 제공한 정보와 유사한 케이스 스터디를 참고하여 완성도 높은 케이스 스터디를 작성해주세요.
제공된 정보를 기반으로 비즈니스 문제와 솔루션을 명확하게 설명하고, 결과를 구체적으로 서술해야 합니다.
제공된 유사 케이스 스터디의 형식과 깊이를 참고하되, 내용을 그대로 복사하지 말고 새로운 케이스 스터디를 작성하세요."""
        
        # 사용자 프롬프트
        user_prompt = "다음은 내가 작성 중인 케이스 스터디입니다:\n\n"
        user_prompt += f"제목: {user_case.get('title', '')}\n\n"
        user_prompt += f"Who: {user_case.get('who', '')}\n\n"
        user_prompt += f"Problem: {user_case.get('problem', '')}\n\n"
        user_prompt += f"Solution: {user_case.get('solution', '')}\n\n"
        user_prompt += f"Results: {user_case.get('results', '')}\n\n"
        
        # 제한된 수의 유사 케이스만 포함
        limited_cases = similar_cases[:max_similar_cases]
        
        # 유사 케이스 스터디
        if limited_cases:
            user_prompt += f"다음은 참고할 수 있는 유사한 케이스 스터디들입니다 (상위 {len(limited_cases)}개):\n\n"
            
            for i, case in enumerate(limited_cases, 1):
                case_study = case['case_study']
                score = case['score'] * 100  # 백분율로 변환
                
                user_prompt += f"유사 케이스 {i} (유사도: {score:.1f}%):\n"
                user_prompt += f"제목: {case_study.get('title', '')}\n"
                user_prompt += f"Who: {case_study.get('who', '')}\n"
                user_prompt += f"Problem: {case_study.get('problem', '')}\n"
                user_prompt += f"Solution: {case_study.get('solution', '')}\n"
                user_prompt += f"Results: {case_study.get('results', '')}\n\n"
        
        user_prompt += "위의 내용을 참고하여 내 케이스 스터디를 개선하고 완성해주세요. 제목, Who, Problem, Solution, Results 섹션으로 구분하여 작성해주세요."
        
        # 최종 프롬프트 정보 (디버깅 및 표시용)
        prompt_info = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model": model_id
        }
        
        return prompt_info
    
    def generate_response(self, prompt_info: Dict[str, str], max_tokens: int = 4000) -> str:
        """
        Claude API로 응답 생성
        
        Args:
            prompt_info: 프롬프트 정보 (시스템 프롬프트, 사용자 프롬프트, 모델 ID)
            max_tokens: 생성할 최대 토큰 수
            
        Returns:
            str: 생성된 응답
        """
        try:
            system_prompt = prompt_info["system_prompt"]
            user_prompt = prompt_info["user_prompt"]
            model = prompt_info.get("model", "claude-3-5-sonnet-20240620")
            
            print(f"Claude API ({model})로 응답 생성 중...")
            
            # Claude API 호출
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # 응답 추출
            result = message.content[0].text
            
            return result
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {str(e)}")
            return f"오류: 응답을 생성할 수 없습니다. {str(e)}"

def parse_case_study_text(text: str) -> Dict[str, str]:
    """
    텍스트에서 케이스 스터디 정보 파싱
    
    Args:
        text: 케이스 스터디 텍스트
        
    Returns:
        Dict: 파싱된 케이스 스터디 정보
    """
    case_study = {
        'title': '',
        'who': '',
        'problem': '',
        'solution': '',
        'results': ''
    }
    
    # 제목 추출
    title_match = None
    if "제목:" in text:
        parts = text.split("제목:", 1)
        if len(parts) > 1:
            title_text = parts[1].strip()
            if "\n" in title_text:
                title_text = title_text.split("\n", 1)[0]
            case_study['title'] = title_text
    
    # 섹션 추출
    sections = {
        'Who': 'who',
        'Problem': 'problem',
        'Solution': 'solution',
        'Results': 'results'
    }
    
    for section, key in sections.items():
        marker = f"{section}:"
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                section_text = parts[1].strip()
                for next_section in sections.keys():
                    next_marker = f"{next_section}:"
                    if next_marker in section_text:
                        section_text = section_text.split(next_marker, 1)[0]
                
                case_study[key] = section_text.strip()
    
    return case_study

def get_user_input() -> Dict[str, str]:
    """
    사용자로부터 케이스 스터디 정보 입력 받기
    
    Returns:
        Dict: 사용자 입력 케이스 스터디 정보
    """
    print("\n" + "=" * 80)
    print("케이스 스터디 정보를 입력해주세요. 각 섹션 입력을 마치면 '완료'를 입력하세요.")
    print("=" * 80)
    
    case_study = {}
    
    # 제목 입력
    case_study['title'] = input("\n제목: ")
    
    # 섹션별 입력
    sections = [
        ('who', 'Who (회사/기관 정보)'),
        ('problem', 'Problem (문제 상황)'),
        ('solution', 'Solution (해결 방안)'),
        ('results', 'Results (결과)')
    ]
    
    for key, prompt in sections:
        print(f"\n{prompt} (여러 줄 입력 가능, 입력을 마치려면 '완료' 입력)")
        lines = []
        while True:
            line = input("> ")
            if line.strip().lower() == '완료':
                break
            lines.append(line)
        case_study[key] = "\n".join(lines)
    
    return case_study

def get_text_input() -> Dict[str, str]:
    """
    텍스트로 케이스 스터디 정보 입력 받기
    
    Returns:
        Dict: 파싱된 케이스 스터디 정보
    """
    print("\n" + "=" * 80)
    print("케이스 스터디 텍스트를 입력해주세요. 입력을 마치면 빈 줄에서 Ctrl+D (Unix) 또는 Ctrl+Z (Windows)를 누르세요.")
    print("=" * 80)
    print("\n예시 형식:")
    print("제목: AAA 시스템 – 회사명")
    print("Who: 회사/기관 정보...")
    print("Problem: 문제 상황...")
    print("Solution: 해결 방안...")
    print("Results: 결과...")
    print("\n텍스트 입력:")
    
    # 여러 줄 입력 받기
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        # 입력 종료
        pass
    
    text = "\n".join(lines)
    return parse_case_study_text(text)

def check_api_key():
    """API 키 확인"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n오류: 환경 변수 'ANTHROPIC_API_KEY'가 설정되어 있지 않습니다.")
        print("다음과 같이 설정할 수 있습니다:")
        if sys.platform.startswith('win'):
            print("  명령 프롬프트에서: set ANTHROPIC_API_KEY=your_api_key")
            print("  PowerShell에서: $env:ANTHROPIC_API_KEY = 'your_api_key'")
        else:
            print("  Bash에서: export ANTHROPIC_API_KEY=your_api_key")
        
        # API 키 직접 입력 옵션
        use_input = input("\nAPI 키를 직접 입력하시겠습니까? (y/n, 기본값: y): ").lower() or "y"
        if use_input == "y":
            api_key = input("Anthropic API 키를 입력하세요: ").strip()
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
                print("API 키가 설정되었습니다. (이 세션에서만 유효)")
                return True
        
        return False
    return True

def interactive_mode():
    """대화형 모드 실행"""
    print("\n" + "=" * 80)
    print("🤖 케이스 스터디 생성 시스템 - 대화형 모드 (Claude API)")
    print("=" * 80)
    
    # API 키 확인
    if not check_api_key():
        print("API 키가 설정되지 않았습니다. 프로그램을 종료합니다.")
        return
    
    # LLM 객체 생성
    llm = CaseStudyLLM(model_type="claude")
    
    # 모델 로드
    if not llm.load_model():
        print("API 클라이언트 설정 실패. 프로그램을 종료합니다.")
        return
    
    # Claude 모델 선택
    print("\nClaude 모델을 선택하세요:")
    models = [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    model_choice = input("\n모델 번호를 입력하세요 (기본값: 1): ") or "1"
    try:
        model_idx = int(model_choice) - 1
        if 0 <= model_idx < len(models):
            selected_model = models[model_idx]
        else:
            print("유효하지 않은 선택입니다. 첫 번째 모델을 사용합니다.")
            selected_model = models[0]
    except ValueError:
        print("유효하지 않은 입력입니다. 첫 번째 모델을 사용합니다.")
        selected_model = models[0]
    
    print(f"\n선택된 모델: {selected_model}")
    
    # 벡터 데이터베이스 로드
    vector_db_dir = input("\n벡터 데이터베이스 경로를 입력하세요 (기본값: vector_db): ") or "vector_db"
    if not llm.load_vector_db(vector_db_dir):
        print("벡터 데이터베이스 로드 실패. 프로그램을 종료합니다.")
        return
    
    # 입력 방식 선택
    print("\n케이스 스터디 정보 입력 방식을 선택하세요:")
    print("1. 대화형 입력 (섹션별로 차례대로 입력)")
    print("2. 텍스트 입력 (전체 텍스트를 한 번에 입력)")
    
    input_choice = input("\n입력 방식 번호를 입력하세요 (기본값: 1): ") or "1"
    
    # 케이스 스터디 정보 입력
    if input_choice == "1":
        user_case = get_user_input()
    else:
        user_case = get_text_input()
    
    # 입력 확인
    print("\n" + "=" * 80)
    print("입력한 케이스 스터디 정보:")
    print("=" * 80)
    print(f"제목: {user_case.get('title', '')}")
    print(f"Who: {user_case.get('who', '')}")
    print(f"Problem: {user_case.get('problem', '')}")
    print(f"Solution: {user_case.get('solution', '')}")
    print(f"Results: {user_case.get('results', '')}")
    
    # 검색할 유사 케이스 수 입력
    try:
        k = int(input("\n검색할 유사 케이스 수를 입력하세요 (기본값: 5): ") or "5")
    except ValueError:
        k = 5
    
    # 유사 케이스 검색
    similar_cases = llm.query_similar_cases(
        user_case.get('title', ''),
        user_case.get('who', ''),
        user_case.get('problem', ''),
        user_case.get('solution', ''),
        user_case.get('results', ''),
        k=k
    )
    
    # 유사 케이스 정보 출력
    print("\n" + "=" * 80)
    print(f"유사한 케이스 스터디 {len(similar_cases)}개 찾음:")
    print("=" * 80)
    
    for i, case in enumerate(similar_cases, 1):
        case_study = case['case_study']
        score = case['score'] * 100  # 백분율로 변환
        matched_sections = case.get('matched_sections', [])
        
        print(f"\n{i}. {case_study.get('title', '')} (유사도: {score:.1f}%)")
        print(f"   ID: {case_study.get('id', '')}")
        print(f"   산업: {case_study.get('industry', '기타')}")
        print(f"   매칭 섹션: {', '.join(matched_sections)}")
    
    # 프롬프트에 포함할 유사 케이스 수 설정
    try:
        max_similar_cases = int(input(f"\n프롬프트에 포함할 유사 케이스 수를 입력하세요 (기본값: 3, 최대: {len(similar_cases)}): ") or "3")
        max_similar_cases = min(max_similar_cases, len(similar_cases))
    except ValueError:
        max_similar_cases = min(3, len(similar_cases))
    
    # 프롬프트 생성
    prompt_info = llm.generate_prompt(
        user_case, 
        similar_cases, 
        max_similar_cases=max_similar_cases,
        model_id=selected_model
    )
    
    # 프롬프트 출력 및 편집
    while True:
        print("\n" + "=" * 80)
        print("생성된 프롬프트 (Claude API에 전달됨):")
        print("=" * 80)
        print("시스템 프롬프트:")
        print(prompt_info["system_prompt"])
        print("\n사용자 프롬프트:")
        print(prompt_info["user_prompt"])
        
        edit_choice = input("\n프롬프트를 수정하시겠습니까? (y/n, 기본값: n): ").lower() or "n"
        
        if edit_choice == "y":
            print("\n어떤 부분을 수정하시겠습니까?")
            print("1. 시스템 프롬프트")
            print("2. 사용자 프롬프트")
            edit_part = input("번호를 입력하세요 (기본값: 2): ") or "2"
            
            if edit_part == "1":
                print("\n시스템 프롬프트를 수정하세요. 입력을 마치면 빈 줄에서 Ctrl+D (Unix) 또는 Ctrl+Z (Windows)를 누르세요.")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    pass
                prompt_info["system_prompt"] = "\n".join(lines)
            else:
                print("\n사용자 프롬프트를 수정하세요. 입력을 마치면 빈 줄에서 Ctrl+D (Unix) 또는 Ctrl+Z (Windows)를 누르세요.")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    pass
                prompt_info["user_prompt"] = "\n".join(lines)
        else:
            break
    
    # 최종 프롬프트 확인
    print("\n" + "=" * 80)
    print("최종 프롬프트:")
    print("=" * 80)
    print("시스템 프롬프트:")
    print(prompt_info["system_prompt"])
    print("\n사용자 프롬프트:")
    print(prompt_info["user_prompt"])
    
    # 실행 확인
    run_choice = input("\n이 프롬프트로 Claude API를 호출하시겠습니까? (y/n, 기본값: y): ").lower() or "y"
    if run_choice != "y":
        print("프로그램을 종료합니다.")
        return
    
    # 응답 생성 토큰 설정
    try:
        max_tokens = int(input("\n생성할 최대 토큰 수를 입력하세요 (기본값: 4000): ") or "4000")
    except ValueError:
        max_tokens = 4000
    
    # Claude API로 응답 생성
    print("\nClaude API로 응답 생성 중... (시간이 다소 걸릴 수 있습니다)")
    
    start_time = time.time()
    response = llm.generate_response(prompt_info, max_tokens=max_tokens)
    end_time = time.time()
    
    # 응답 출력
    print("\n" + "=" * 80)
    print(f"생성된 케이스 스터디 (처리 시간: {end_time - start_time:.2f}초):")
    print("=" * 80)
    print(response)
    
    # 저장 여부 확인
    save_choice = input("\n생성된 케이스 스터디를 파일로 저장하시겠습니까? (y/n, 기본값: n): ").lower() or "n"
    
    if save_choice == "y":
        file_path = input("저장할 파일 경로를 입력하세요 (기본값: case_study_output.txt): ") or "case_study_output.txt"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response)
        
        print(f"케이스 스터디가 '{file_path}'에 저장되었습니다.")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Claude API를 사용한 케이스 스터디 생성')
    parser.add_argument('--check-key', action='store_true', help='Anthropic API 키 확인')
    parser.add_argument('--set-key', type=str, help='Anthropic API 키 설정')
    
    args = parser.parse_args()
    
    # API 키 설정
    if args.set_key:
        os.environ["ANTHROPIC_API_KEY"] = args.set_key
        print(f"Anthropic API 키가 설정되었습니다.")
        return
    
    # API 키 확인 모드
    if args.check_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
            print(f"Anthropic API 키가 설정되어 있습니다: {masked_key}")
        else:
            print("Anthropic API 키가 설정되어 있지 않습니다.")
        return
    
    # 대화형 모드 실행
    interactive_mode()

if __name__ == "__main__":
    main()