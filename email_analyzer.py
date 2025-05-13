import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

# API 클라이언트 임포트
import openai
from anthropic import Anthropic
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class EmailAnalyzer:
    """
    다양한 LLM 모델(로컬 모델 및 API)을 사용하여 이메일 분석을 수행하는 클래스
    """
    
    def __init__(self, models_dir: str = "c:/models"):
        """
        EmailAnalyzer 초기화
        
        Args:
            models_dir: 로컬 모델이 저장된 디렉토리 경로
        """
        self.models_dir = Path(models_dir)
        self.available_local_models = self._scan_local_models()
        self.api_keys = self._load_api_keys()
        
    def _scan_local_models(self) -> List[str]:
        """로컬 모델 디렉토리를 스캔하여 사용 가능한 모델 이름을 반환"""
        if not self.models_dir.exists():
            print(f"경고: 모델 디렉토리 {self.models_dir}가 존재하지 않습니다.")
            return []
        
        # 모델 디렉토리에서 모든 하위 디렉토리를 찾음 (각 디렉토리는 모델로 간주)
        model_dirs = [d.name for d in self.models_dir.iterdir() if d.is_dir()]
        
        # 모델이 실제로 사용 가능한지 확인 (config.json 파일이 있는지)
        valid_models = []
        for model_name in model_dirs:
            model_path = self.models_dir / model_name
            if (model_path / "config.json").exists():
                valid_models.append(model_name)
        
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
        로컬 모델을 사용하여 이메일 분석
        
        Args:
            model_name: 사용할 로컬 모델 이름
            email_text: 분석할 이메일 텍스트
            prompt_template: 프롬프트 템플릿
            
        Returns:
            분석 결과 문자열
        """
        model_path = self.models_dir / model_name
        
        try:
            # 토크나이저와 모델 로드
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )
            
            # 완성된 프롬프트 생성
            full_prompt = prompt_template.format(email_text=email_text)
            
            # 텍스트 생성 파이프라인 설정
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=4096,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            # 응답 생성
            response = generator(full_prompt, max_new_tokens=2048)[0]['generated_text']
            
            # 프롬프트 이후의 응답만 추출
            if full_prompt in response:
                result = response[len(full_prompt):]
            else:
                result = response
                
            return result.strip()
            
        except Exception as e:
            return f"로컬 모델 분석 오류: {str(e)}"
    
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
        
        Args:
            model_name: 사용할 Anthropic 모델 이름
            email_text: 분석할 이메일 텍스트
            prompt_template: 프롬프트 템플릿
            
        Returns:
            분석 결과 문자열
        """
        api_key = self.api_keys.get("anthropic")
        if not api_key:
            return "Anthropic API 키가 설정되지 않았습니다."
        
        try:
            client = Anthropic(api_key=api_key)
            
            # 완성된 프롬프트 생성
            full_prompt = prompt_template.format(email_text=email_text)
            
            # API 호출
            response = client.messages.create(
                model=model_name,
                max_tokens=2048,
                temperature=0.7,
                system="당신은 이메일 커뮤니케이션을 분석하는 전문가입니다.",
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Anthropic API 분석 오류: {str(e)}"
    
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
            prompt_template: 프롬프트 템플릿
            
        Returns:
            분석 결과 및 메타데이터를 포함하는 사전
        """
        start_time = time.time()
        result = {"model": model_spec, "success": False}
        
        try:
            # 모델 유형과 이름 파싱
            model_type, model_name = model_spec.split(":", 1)
            
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


def main():
    """메인 함수: 커맨드 라인 인터페이스 제공"""
    parser = argparse.ArgumentParser(description="다양한 LLM을 사용하여 이메일 커뮤니케이션 분석")
    parser.add_argument("--email_file", type=str, help="분석할 이메일 파일 경로")
    parser.add_argument("--models_dir", type=str, default="c:/models", help="로컬 모델 디렉토리 경로")
    parser.add_argument("--model", type=str, help="사용할 모델 (예: local:llama-3, openai:gpt-4)")
    parser.add_argument("--list_models", action="store_true", help="사용 가능한 모델 목록 표시")
    parser.add_argument("--output", type=str, help="결과를 저장할 JSON 파일 경로")
    parser.add_argument("--compare_all", action="store_true", help="사용 가능한 모든 모델로 분석 실행")
    
    args = parser.parse_args()
    
    # EmailAnalyzer 인스턴스 생성
    analyzer = EmailAnalyzer(models_dir=args.models_dir)
    
    # 모델 목록 표시 요청 처리
    if args.list_models:
        models = analyzer.list_available_models()
        print("사용 가능한 모델:")
        print("로컬 모델:")
        for model in models["local"]:
            print(f" - local:{model}")
        print("API 모델:")
        for model in models["api"]:
            print(f" - {model}")
        return
    
    # 이메일 파일 확인
    if not args.email_file:
        print("오류: 이메일 파일을 지정해야 합니다 (--email_file)")
        return
    
    # 이메일 텍스트 로드
    try:
        with open(args.email_file, "r", encoding="utf-8") as f:
            email_text = f.read()
    except Exception as e:
        print(f"이메일 파일 로드 오류: {str(e)}")
        return
    
    # 분석 프롬프트 템플릿
    prompt_template = """
    신은 회사의 오랜 업무 경험을 가진 인사 및 조직 컨설턴트로서, 이메일 커뮤니케이션을 다각적으로 분석해주세요. 제공된 이메일 체인을 읽고 다음 항목에 따라 분석 리포트를 작성해주세요:

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
    
    results = []
    
    # 단일 모델 분석 또는 모든 모델로 비교 분석
    if args.compare_all:
        models = analyzer.list_available_models()
        all_models = [f"local:{model}" for model in models["local"]] + models["api"]
        
        for model_spec in all_models:
            print(f"{model_spec}로 분석 중...")
            result = analyzer.analyze_email(email_text, model_spec, prompt_template)
            results.append(result)
            print(f" 완료 (실행 시간: {result['execution_time']:.2f}초)")
    
    elif args.model:
        print(f"{args.model}로 분석 중...")
        result = analyzer.analyze_email(email_text, args.model, prompt_template)
        results.append(result)
        print(f" 완료 (실행 시간: {result['execution_time']:.2f}초)")
        
        # 분석 결과 즉시 출력
        if result["success"]:
            print("\n--- 분석 결과 ---")
            print(result["analysis"])
            print("----------------")
        else:
            print(f"오류: {result.get('error', '알 수 없는 오류')}")
    
    else:
        print("오류: --model 옵션을 사용하여 모델을 지정하거나 --compare_all 옵션을 사용하세요")
        return
    
    # 결과 저장
    if args.output and results:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"결과가 {args.output}에 저장되었습니다.")
        except Exception as e:
            print(f"결과 저장 오류: {str(e)}")


if __name__ == "__main__":
    main()