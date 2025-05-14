import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from llama_cpp import Llama

# API 클라이언트 임포트
import openai
from anthropic import Anthropic
try:
    import google.generativeai as genai
except ImportError:
    genai = None
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class EmailAnalyzer:
    """
    다양한 LLM 모델(로컬 모델 및 API)을 사용하여 이메일 분석을 수행하는 클래스
    """
    
    def __init__(self, models_dir: str = "e:/models"):
        """
        EmailAnalyzer 초기화
        
        Args:
            models_dir: 로컬 모델이 저장된 디렉토리 경로
        """
        self.models_dir = Path(models_dir)
        self.available_local_models = self._scan_local_models()
        self.api_keys = self._load_api_keys()
        self.anthropic_models = self._get_anthropic_models()
        
    def _get_anthropic_models(self) -> Dict[str, str]:
        """최신 Anthropic 모델 정보 조회"""
        try:
            client = Anthropic(api_key=self.api_keys.get("anthropic"))
            models = client.models.list()
            
            model_info = {}
            for model in models:
                model_info[model.name] = {
                    "max_tokens": model.max_tokens,
                    "created": model.created
                }
            return model_info
        except Exception:
            # API 조회 실패 시 기본값 반환
            return {
                "claude-3-opus": {"max_tokens": 4096},
                "claude-3-sonnet": {"max_tokens": 4096},
                "claude-3-haiku": {"max_tokens": 4096}
            }
    
    def list_available_models(self) -> Dict[str, List[str]]:
        result = {
            "local": self.available_local_models,
            "api": []
        }
        
        for api_name, api_key in self.api_keys.items():
            if api_key:
                if api_name == "anthropic":
                    # 사용 가능한 모델 목록 표시
                    for model_name in self.anthropic_models.keys():
                        result["api"].append(f"anthropic:{model_name}")
                # ... 다른 API 모델들
        
        return result


    def _scan_local_models(self) -> List[str]:
        """로컬 모델 디렉토리를 스캔하여 사용 가능한 모델 이름을 반환"""
        if not self.models_dir.exists():
            print(f"경고: 모델 디렉토리 {self.models_dir}가 존재하지 않습니다.")
            return []
        
        # .gguf 파일 찾기
        valid_models = []
        for file in self.models_dir.glob("*.gguf"):
            model_name = file.stem  # 파일 확장자를 제외한 이름
            valid_models.append(model_name)
            print(f"발견된 모델: {model_name} ({file})")
        
        return valid_models

    def _load_api_keys(self) -> Dict[str, str]:
        """API 키를 환경 변수에서 로드"""
        api_keys = {
            "openai": os.environ.get("OPENAI_API_KEY", ""),
            "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
            "google": os.environ.get("GOOGLE_API_KEY", "")
        }
        return api_keys
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """
        사용 가능한 모든 모델 목록을 반환
        """
        result = {
            "local": self.available_local_models,
            "api": []
        }
        
        # API 키가 설정된 서비스 확인
        for api_name, api_key in self.api_keys.items():
            if api_key:
                if api_name == "openai":
                    result["api"].append("openai:gpt-4")
                    result["api"].append("openai:gpt-3.5-turbo")
                elif api_name == "anthropic":
                    result["api"].append("anthropic:claude-3-5-sonnet")
                    result["api"].append("anthropic:claude-3-opus")
                elif api_name == "google":
                    result["api"].append("google:gemini-pro")
                    
        return result
    
    

    def analyze_with_local_model(self, 
                                model_name: str, 
                                email_text: str, 
                                prompt_template: str) -> str:
        """
        로컬 GGUF 모델을 사용하여 이메일 분석
        """
        model_path = self.models_dir / f"{model_name}.gguf"
        
        try:
            print(f"모델 파일 경로: {model_path}")
            
            # Llama 모델 초기화
            llm = Llama(
                model_path=str(model_path),
                n_ctx=4096,  # 컨텍스트 길이
                n_batch=512,  # 배치 크기
                n_threads=4   # 사용할 스레드 수
            )
            
            # 완성된 프롬프트 생성
            full_prompt = prompt_template.format(email_text=email_text)
            
            # 응답 생성
            response = llm(
                full_prompt,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                echo=False  # 입력 프롬프트를 출력에 포함하지 않음
            )
            
            # 응답 텍스트 반환
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            error_msg = f"""
    로컬 모델 분석 오류:
    - 오류 유형: {type(e).__name__}
    - 오류 메시지: {str(e)}
    - 모델 경로: {model_path}
    - 모델 존재 여부: {'예' if model_path.exists() else '아니오'}

    가능한 해결 방법:
    1. llama-cpp-python 라이브러리가 설치되어 있는지 확인
    2. GGUF 파일이 지정된 경로에 존재하는지 확인
    3. 파일 권한이 올바른지 확인
    """
            return error_msg

    def analyze_with_openai(self, 
                          model_name: str, 
                          email_text: str, 
                          prompt_template: str) -> str:
        """
        OpenAI API를 사용하여 이메일 분석
        
        Args:
            model_name: 사용할 OpenAI 모델 이름 (gpt-4, gpt-3.5-turbo 등)
            email_text: 분석할 이메일 텍스트
            prompt_template: 프롬프트 템플릿
            
        Returns:
            분석 결과 문자열
        """
        api_key = self.api_keys.get("openai")
        if not api_key:
            return "OpenAI API 키가 설정되지 않았습니다."
        
        try:
            client = openai.OpenAI(api_key=api_key)
            
            # 완성된 프롬프트 생성
            full_prompt = prompt_template.format(email_text=email_text)
            
            # API 호출
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "당신은 이메일 커뮤니케이션을 분석하는 전문가입니다."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"OpenAI API 분석 오류: {str(e)}"
            
    def analyze_with_anthropic(self, 
                            model_name: str, 
                            email_text: str, 
                            prompt_template: str) -> str:
        """
        Anthropic API를 사용하여 이메일 분석
        """
        api_key = self.api_keys.get("anthropic")
        if not api_key:
            return "Anthropic API 키가 설정되지 않았습니다."
        
        try:
            client = Anthropic(api_key=api_key)
            
            # 모델 이름 매핑 (최신 Claude 3 모델 식별자로 업데이트)
            model_mapping = {
                "claude-3-opus": "claude-3-opus",  # 최신 버전
                "claude-3-5-sonnet": "claude-3-sonnet",  # 최신 버전
                "claude-3-sonnet-20240229": "claude-3-sonnet",  # 이전 버전 매핑
                "claude-3-opus-20240229": "claude-3-opus"  # 이전 버전 매핑
            }
            
            actual_model = model_mapping.get(model_name, model_name)
            
            # API 호출 전 모델 이름 로깅
            print(f"사용하는 모델 식별자: {actual_model}")
            
            response = client.messages.create(
                model=actual_model,
                max_tokens=2048,
                temperature=0.3,
                system="당신은 이메일 커뮤니케이션을 전문적으로 분석하는 비즈니스 컨설턴트입니다. 분석은 항상 구조화되고 일관된 형식으로 제공해야 합니다.",
                messages=[
                    {"role": "user", "content": prompt_template.format(email_text=email_text)}
                ]
            )
            
            return response.content[0].text
                
        except Exception as e:
            error_msg = f"""
    Anthropic API 호출 중 오류 발생:
    - 오류 유형: {type(e).__name__}
    - 오류 메시지: {str(e)}
    - 시도한 모델: {model_name}
    - 실제 사용된 모델: {actual_model}
    - API 키 설정 여부: {'예' if api_key else '아니오'}

    가능한 해결 방법:
    1. API 키가 올바르게 설정되어 있는지 확인
    2. 모델 이름이 정확한지 확인 (현재 지원되는 모델: claude-3-opus, claude-3-sonnet)
    3. 인터넷 연결 상태 확인
    """
            return error_msg
        

    def analyze_with_google(self, 
                          model_name: str, 
                          email_text: str, 
                          prompt_template: str) -> str:
        """
        Google Gemini API를 사용하여 이메일 분석
        
        Args:
            model_name: 사용할 Google 모델 이름
            email_text: 분석할 이메일 텍스트
            prompt_template: 프롬프트 템플릿
            
        Returns:
            분석 결과 문자열
        """
        if genai is None:
            return "Google Generative AI 라이브러리가 설치되지 않았습니다."
        
        api_key = self.api_keys.get("google")
        if not api_key:
            return "Google API 키가 설정되지 않았습니다."
        
        try:
            # API 초기화
            genai.configure(api_key=api_key)
            
            # 모델 이름에서 모델 ID 추출 (예: gemini-pro)
            model = genai.GenerativeModel(model_name)
            
            # 완성된 프롬프트 생성
            full_prompt = prompt_template.format(email_text=email_text)
            
            # API 호출
            response = model.generate_content(full_prompt)
            
            return response.text
            
        except Exception as e:
            return f"Google API 분석 오류: {str(e)}"
    

    def analyze_email(self, 
                    email_text: str, 
                    model_spec: str, 
                    prompt_template: str) -> Dict[str, Union[str, float]]:
        """
        지정된 모델을 사용하여 이메일 분석 실행
        
        Args:
            email_text: 분석할 이메일 텍스트
            model_spec: 모델 사양 (예: local:llama-3, openai:gpt-4, anthropic:claude-3-5-sonnet)
                    또는 단순 모델 이름 (예: claude-3-haiku-20240307)
            prompt_template: 프롬프트 템플릿
            
        Returns:
            분석 결과 및 메타데이터를 포함하는 사전
        """
        start_time = time.time()
        result = {"model": model_spec, "success": False}
        
        try:
            # 모델 유형과 이름 파싱
            if ":" in model_spec:
                model_type, model_name = model_spec.split(":", 1)
            else:
                # 콜론이 없으면 기본적으로 anthropic 모델로 간주
                model_type = "anthropic"
                model_name = model_spec
            
            # 적절한 분석 함수 호출
            if model_type == "local":
                if model_name not in self.available_local_models:
                    result["error"] = f"지정된 로컬 모델을 찾을 수 없습니다: {model_name}"
                    return result
                analysis = self.analyze_with_local_model(model_name, email_text, prompt_template)
            
            elif model_type == "openai":
                analysis = self.analyze_with_openai(model_name, email_text, prompt_template)
            
            elif model_type == "anthropic":
                analysis = self.analyze_with_anthropic(model_name, email_text, prompt_template)
            
            elif model_type == "google":
                analysis = self.analyze_with_google(model_name, email_text, prompt_template)
            
            else:
                result["error"] = f"지원되지 않는 모델 유형: {model_type}"
                return result
            
            # 분석 결과 저장
            result["analysis"] = analysis
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        # 실행 시간 계산
        result["execution_time"] = time.time() - start_time
        
        return result
    
def select_model_interactively(analyzer):
    """사용자가 대화형으로 모델을 선택할 수 있는 함수"""
    models = analyzer.list_available_models()
    
    print("\n=== 사용 가능한 모델 목록 ===")
    
    all_models = []
    
    # 로컬 모델 출력
    if models["local"]:
        print("\n[로컬 모델]")
        for idx, model in enumerate(models["local"]):
            print(f"{len(all_models) + 1}. local:{model}")
            all_models.append(f"local:{model}")
    
    # API 모델 출력
    if models["api"]:
        print("\n[API 모델]")
        for idx, model in enumerate(models["api"]):
            print(f"{len(all_models) + 1}. {model}")
            all_models.append(model)
    
    while True:
        try:
            choice = input("\n사용할 모델 번호를 선택하세요 (q: 종료): ")
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(all_models):
                return all_models[idx]
            else:
                print("올바른 번호를 선택해주세요.")
        except ValueError:
            print("숫자를 입력해주세요.")


def main():
    """수정된 메인 함수"""
    parser = argparse.ArgumentParser(description="다양한 LLM을 사용하여 이메일 커뮤니케이션 분석")
    parser.add_argument("--email_file", type=str, default="./email.txt", help="분석할 이메일 파일 경로")
    parser.add_argument("--models_dir", type=str, default="e:/models", help="로컬 모델 디렉토리 경로")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드로 실행")
    parser.add_argument("--output", type=str, help="결과를 저장할 JSON 파일 경로")
    
    args = parser.parse_args()
    
    # 프롬프트 템플릿 정의
    prompt_template = """
당신은 회사의 오랜 업무 경험을 가진 인사 및 조직 컨설턴트로서, 이메일 커뮤니케이션을 다각적으로 분석해주세요. 제공된 이메일 체인을 읽고 다음 항목에 따라 분석 리포트를 작성해주세요:

1. **인물 분석**
   - 각 인물의 역할과 직책
   - 성격 및 행동 특성 (이모지로 특성 표현)
   - 업무 역량 (⭐ 1-5점 척도로 평가):
     * 전문성
     * 리더십
     * 의사소통 능력
     * 업무 추진력
     * 문제 해결 능력

2. **조직 역학 분석**
   - 공식적/비공식적 권력 구조
   - 의사결정 패턴
   - 부서간 협업 방식
   - 갈등 해결 방식

3. **커뮤니케이션 스타일 분석**
   - 커뮤니케이션 명확성과 효율성
   - 언어적 특성 (공손함, 단호함, 전문성 등)
   - 암묵적 메시지 및 맥락
   - 핵심 정보 전달 방식

이메일 체인:
{email_text}
"""
    
    # EmailAnalyzer 인스턴스 생성
    analyzer = EmailAnalyzer(models_dir=args.models_dir)
    
    if args.interactive:
        # 이메일 파일 경로 확인
        email_file = args.email_file
        if not os.path.exists(email_file):
            email_file = input(f"기본 경로 {args.email_file}를 찾을 수 없습니다. 분석할 이메일 파일 경로를 입력하세요: ")
            
        # 모델 선택
        selected_model = select_model_interactively(analyzer)
        if not selected_model:
            print("프로그램을 종료합니다.")
            return
            
        print(f"\n선택된 모델: {selected_model}")
    else:
        email_file = args.email_file
        selected_model = "anthropic:claude-3-5-sonnet"  # 최신 모델로 수정

    
    # 이메일 텍스트 로드
    try:
        with open(email_file, "r", encoding="utf-8") as f:
            email_text = f.read()
    except FileNotFoundError:
        print(f"오류: 이메일 파일을 찾을 수 없습니다: {email_file}")
        print("현재 작업 디렉토리:", os.getcwd())
        return
    except Exception as e:
        print(f"이메일 파일 로드 오류:")
        print(f"- 오류 유형: {type(e).__name__}")
        print(f"- 오류 메시지: {str(e)}")
        print(f"- 시도한 파일 경로: {email_file}")
        return

    # 분석 실행
    print(f"\n{selected_model}로 분석 중...")
    result = analyzer.analyze_email(email_text, selected_model, prompt_template)
    
    # 결과 출력
    if result["success"]:
        print("\n=== 분석 결과 ===")
        print(result["analysis"])
        print("=" * 50)
        print(f"실행 시간: {result['execution_time']:.2f}초")
    else:
        print("\n=== 오류 발생 ===")
        print(result.get("error", "알 수 없는 오류"))
        print("=" * 50)
    
    # 결과 저장
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump([result], f, ensure_ascii=False, indent=2)
            print(f"\n결과가 {args.output}에 저장되었습니다.")
        except Exception as e:
            print(f"\n결과 저장 중 오류 발생:")
            print(f"- 오류 유형: {type(e).__name__}")
            print(f"- 오류 메시지: {str(e)}")


if __name__ == "__main__":
    main()