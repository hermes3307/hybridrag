import os
import argparse
import json
import sys
import time
from typing import List, Dict, Any, Optional, Tuple

# LLM 라이브러리 임포트
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# VectorQuery 클래스 임포트
from vectorquery import VectorQuery

class CaseStudyLLM:
    def __init__(self, model_type: str = "llama", model_path: Optional[str] = None):
        """
        로컬 LLM을 사용하여 케이스 스터디 생성을 위한 클래스 초기화
        
        Args:
            model_type (str): 사용할 모델 유형 ("llama" 또는 "gemma")
            model_path (str, optional): 모델 파일 경로. 지정하지 않으면 기본 경로 사용
        """
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.vector_query = VectorQuery()
        
        # 모델 경로가 지정되지 않은 경우 기본 경로 사용
        if self.model_path is None:
            if self.model_type == "llama":
                self.model_path = "models/llama-3-8b-instruct.gguf"  # 기본 Llama 3 경로
            elif self.model_type == "gemma":
                self.model_path = "models/gemma-7b-it"  # 기본 Gemma 경로
        
        print(f"LLM 모델 유형: {self.model_type}")
        print(f"모델 경로: {self.model_path}")
    
    def load_model(self):
        """모델 로드"""
        try:
            if self.model_type == "llama":
                print("Llama 모델 로드 중...")
                
                # 모델 경로가 디렉토리인 경우, 디렉토리 내 .gguf 파일 찾기
                if os.path.isdir(self.model_path):
                    gguf_files = [f for f in os.listdir(self.model_path) if f.endswith('.gguf')]
                    if not gguf_files:
                        raise ValueError(f"지정된 디렉토리 '{self.model_path}'에서 .gguf 파일을 찾을 수 없습니다.")
                    
                    # 첫 번째 .gguf 파일 사용
                    model_file = os.path.join(self.model_path, gguf_files[0])
                    print(f"발견된 모델 파일: {model_file}")
                else:
                    model_file = self.model_path
                
                self.model = Llama(
                    model_path=model_file,
                    n_ctx=4096,  # 컨텍스트 크기
                    n_gpu_layers=-1,  # GPU 사용 최대화
                    verbose=False
                )
                print("Llama 모델 로드 완료!")
                
            elif self.model_type == "gemma":
                print("Gemma 모델 로드 중...")
                
                # 모델 경로가 디렉토리인지 확인
                if os.path.isdir(self.model_path):
                    model_dir = self.model_path
                else:
                    # 상위 디렉토리가 존재하는지 확인
                    model_dir = self.model_path
                
                # Gemma 모델 및 토크나이저 로드
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    print("Gemma 모델 로드 완료!")
                except Exception as e:
                    raise ValueError(f"Gemma 모델 로드 실패: {str(e)}")
                
            else:
                raise ValueError(f"지원되지 않는 모델 유형: {self.model_type}")
            
            return True
            
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
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
                        similar_cases: List[Dict[str, Any]]) -> str:
        """
        LLM에 전달할 프롬프트 생성
        
        Args:
            user_case: 사용자가 입력한 케이스 스터디
            similar_cases: 유사한 케이스 스터디 목록
            
        Returns:
            str: 생성된 프롬프트
        """
        # 프롬프트 템플릿
        if self.model_type == "llama":
            prompt = "<|system|>\n"
            prompt += "당신은 비즈니스 케이스 스터디 작성 전문가입니다. 고객이 제공한 정보와 유사한 케이스 스터디를 참고하여 완성도 높은 케이스 스터디를 작성해주세요.\n"
            prompt += "제공된 정보를 기반으로 비즈니스 문제와 솔루션을 명확하게 설명하고, 결과를 구체적으로 서술해야 합니다.\n"
            prompt += "제공된 유사 케이스 스터디의 형식과 깊이를 참고하되, 내용을 그대로 복사하지 말고 새로운 케이스 스터디를 작성하세요.\n"
            prompt += "</|system|>\n\n"
            
            prompt += "<|user|>\n"
            
        elif self.model_type == "gemma":
            prompt = "<start_of_turn>system\n"
            prompt += "당신은 비즈니스 케이스 스터디 작성 전문가입니다. 고객이 제공한 정보와 유사한 케이스 스터디를 참고하여 완성도 높은 케이스 스터디를 작성해주세요.\n"
            prompt += "제공된 정보를 기반으로 비즈니스 문제와 솔루션을 명확하게 설명하고, 결과를 구체적으로 서술해야 합니다.\n"
            prompt += "제공된 유사 케이스 스터디의 형식과 깊이를 참고하되, 내용을 그대로 복사하지 말고 새로운 케이스 스터디를 작성하세요.\n"
            prompt += "<end_of_turn>\n\n"
            
            prompt += "<start_of_turn>user\n"
        
        # 사용자 입력 케이스 스터디
        prompt += "다음은 내가 작성 중인 케이스 스터디입니다:\n\n"
        prompt += f"제목: {user_case.get('title', '')}\n\n"
        prompt += f"Who: {user_case.get('who', '')}\n\n"
        prompt += f"Problem: {user_case.get('problem', '')}\n\n"
        prompt += f"Solution: {user_case.get('solution', '')}\n\n"
        prompt += f"Results: {user_case.get('results', '')}\n\n"
        
        # 유사 케이스 스터디
        prompt += "다음은 참고할 수 있는 유사한 케이스 스터디들입니다:\n\n"
        
        for i, case in enumerate(similar_cases, 1):
            case_study = case['case_study']
            score = case['score'] * 100  # 백분율로 변환
            
            prompt += f"유사 케이스 {i} (유사도: {score:.1f}%):\n"
            prompt += f"제목: {case_study.get('title', '')}\n"
            prompt += f"Who: {case_study.get('who', '')}\n"
            prompt += f"Problem: {case_study.get('problem', '')}\n"
            prompt += f"Solution: {case_study.get('solution', '')}\n"
            prompt += f"Results: {case_study.get('results', '')}\n\n"
        
        prompt += "위의 내용을 참고하여 내 케이스 스터디를 개선하고 완성해주세요. 제목, Who, Problem, Solution, Results 섹션으로 구분하여 작성해주세요.\n"
        
        if self.model_type == "llama":
            prompt += "</|user|>\n\n<|assistant|>\n"
        elif self.model_type == "gemma":
            prompt += "<end_of_turn>\n\n<start_of_turn>model\n"
        
        return prompt
    
    def generate_response(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        LLM으로 응답 생성
        
        Args:
            prompt: 입력 프롬프트
            max_tokens: 생성할 최대 토큰 수
            
        Returns:
            str: 생성된 응답
        """
        try:
            if self.model_type == "llama":
                print("Llama 모델로 응답 생성 중...")
                response = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    repeat_penalty=1.1
                )
                
                # 응답 텍스트 추출
                result = response["choices"][0]["text"]
                
            elif self.model_type == "gemma":
                print("Gemma 모델로 응답 생성 중...")
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # 모델 생성
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        do_sample=True
                    )
                
                # 응답 텍스트 추출
                result = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Gemma 응답에서 필요한 부분만 추출
                end_marker = "<end_of_turn>"
                if end_marker in result:
                    result = result.split(end_marker)[0].strip()
            
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

def list_model_files(model_dir):
    """
    지정된 디렉토리에서 모델 파일 목록 표시
    
    Args:
        model_dir: 모델 디렉토리 경로
    """
    if not os.path.isdir(model_dir):
        print(f"경고: '{model_dir}'는 디렉토리가 아닙니다.")
        return
    
    print(f"\n'{model_dir}' 디렉토리의 모델 파일:")
    print("-" * 60)
    
    # Llama 모델 파일 (.gguf)
    llama_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
    if llama_files:
        print("Llama 모델 파일 (.gguf):")
        for i, file in enumerate(llama_files, 1):
            file_path = os.path.join(model_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # GB 단위
            print(f"  {i}. {file} ({file_size:.2f} GB)")
    
    # Gemma 모델 디렉토리
    gemma_dirs = [d for d in os.listdir(model_dir) 
                 if os.path.isdir(os.path.join(model_dir, d)) and 
                 (d.startswith('gemma') or 
                  any(os.path.exists(os.path.join(model_dir, d, f)) for f in ['config.json', 'tokenizer.json']))]
    
    if gemma_dirs:
        print("\nGemma 모델 디렉토리:")
        for i, dir_name in enumerate(gemma_dirs, 1):
            print(f"  {i}. {dir_name}")
    
    if not llama_files and not gemma_dirs:
        print("  발견된 모델 파일 없음")
    
    print("-" * 60)

def interactive_mode():
    """대화형 모드 실행"""
    print("\n" + "=" * 80)
    print("🤖 케이스 스터디 생성 시스템 - 대화형 모드")
    print("=" * 80)
    
    # 모델 선택
    print("\n사용할 LLM 모델을 선택하세요:")
    print("1. Llama 3")
    print("2. Gemma")
    
    model_choice = input("\n모델 번호를 입력하세요 (기본값: 1): ") or "1"
    model_type = "llama" if model_choice == "1" else "gemma"
    
    # 모델 경로 입력
    model_path = input(f"\n{model_type.capitalize()} 모델 경로를 입력하세요 (기본값 사용: 엔터): ")
    if not model_path:
        model_path = None
    elif os.path.isdir(model_path):
        # 디렉토리 내 모델 파일 목록 표시
        list_model_files(model_path)
        
        if model_type == "llama":
            # Llama 모델 파일 선택
            llama_files = [f for f in os.listdir(model_path) if f.endswith('.gguf')]
            if llama_files:
                print("\nLlama 모델 파일을 선택하세요:")
                for i, file in enumerate(llama_files, 1):
                    print(f"{i}. {file}")
                
                file_choice = input("\n파일 번호를 입력하세요 (기본값: 1): ") or "1"
                try:
                    file_idx = int(file_choice) - 1
                    if 0 <= file_idx < len(llama_files):
                        model_path = os.path.join(model_path, llama_files[file_idx])
                        print(f"선택된 모델 파일: {model_path}")
                    else:
                        print("유효하지 않은 선택입니다. 첫 번째 파일을 사용합니다.")
                        model_path = os.path.join(model_path, llama_files[0])
                except ValueError:
                    print("유효하지 않은 입력입니다. 첫 번째 파일을 사용합니다.")
                    model_path = os.path.join(model_path, llama_files[0])
        elif model_type == "gemma":
            # Gemma 모델 디렉토리 선택
            gemma_dirs = [d for d in os.listdir(model_path) 
                         if os.path.isdir(os.path.join(model_path, d)) and 
                         (d.startswith('gemma') or 
                          any(os.path.exists(os.path.join(model_path, d, f)) for f in ['config.json', 'tokenizer.json']))]
            
            if gemma_dirs:
                print("\nGemma 모델 디렉토리를 선택하세요:")
                for i, dir_name in enumerate(gemma_dirs, 1):
                    print(f"{i}. {dir_name}")
                
                dir_choice = input("\n디렉토리 번호를 입력하세요 (기본값: 1): ") or "1"
                try:
                    dir_idx = int(dir_choice) - 1
                    if 0 <= dir_idx < len(gemma_dirs):
                        model_path = os.path.join(model_path, gemma_dirs[dir_idx])
                        print(f"선택된 모델 디렉토리: {model_path}")
                    else:
                        print("유효하지 않은 선택입니다. 첫 번째 디렉토리를 사용합니다.")
                        model_path = os.path.join(model_path, gemma_dirs[0])
                except ValueError:
                    print("유효하지 않은 입력입니다. 첫 번째 디렉토리를 사용합니다.")
                    model_path = os.path.join(model_path, gemma_dirs[0])
    
    # LLM 객체 생성
    llm = CaseStudyLLM(model_type, model_path)
    
    # 모델 로드
    if not llm.load_model():
        print("모델 로드 실패. 프로그램을 종료합니다.")
        return
    
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
    
    # 프롬프트 생성
    prompt = llm.generate_prompt(user_case, similar_cases)
    
    # 프롬프트 출력 및 편집
    while True:
        print("\n" + "=" * 80)
        print("생성된 프롬프트 (LLM에 전달됨):")
        print("=" * 80)
        print(prompt)
        
        edit_choice = input("\n프롬프트를 수정하시겠습니까? (y/n, 기본값: n): ").lower() or "n"
        
        if edit_choice == "y":
            print("\n프롬프트를 수정하세요. 입력을 마치면 빈 줄에서 Ctrl+D (Unix) 또는 Ctrl+Z (Windows)를 누르세요.")
            
            # 여러 줄 입력 받기
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                # 입력 종료
                pass
            
            prompt = "\n".join(lines)
        else:
            break
    
    # LLM으로 응답 생성
    print("\nLLM으로 응답 생성 중... (시간이 다소 걸릴 수 있습니다)")
    
    start_time = time.time()
    response = llm.generate_response(prompt)
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
    parser = argparse.ArgumentParser(description='로컬 LLM을 사용한 케이스 스터디 생성')
    parser.add_argument('--model', type=str, choices=['llama', 'gemma'], default='llama', help='사용할 모델 유형 (llama 또는 gemma)')
    parser.add_argument('--model-path', type=str, help='모델 파일 경로')
    parser.add_argument('--list-models', '-l', action='store_true', help='지정된 디렉토리에서 사용 가능한 모델 목록 표시')
    
    args = parser.parse_args()
    
    # 모델 목록 표시 모드
    if args.list_models and args.model_path:
        list_model_files(args.model_path)
        return
    
    # 대화형 모드 실행
    interactive_mode()

if __name__ == "__main__":
    main()