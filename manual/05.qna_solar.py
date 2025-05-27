#!/usr/bin/env python3
"""
Simple Employee Q&A Chatbot for Altibase
A command-line chatbot that uses your vector database to answer questions about Altibase manuals.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleChatbot:
    """Simple Q&A Chatbot for Employee Onboarding"""
    
    def __init__(self, vector_db_path: str = "manual_vector_db", model_type: str = "claude"):
        """
        Initialize the chatbot
        
        Args:
            vector_db_path: Path to your vector database
            model_type: AI model to use ('claude', 'solar', or 'qwen')
        """
        self.vector_db_path = vector_db_path
        self.model_type = model_type.lower()
        self.vector_db = None
        self.vectorizer = None
        self.llm_client = None
        self.conversation_history = []
        
        # Model information
        self.model_info = {
            'claude': {
                'name': 'Claude 3 Sonnet',
                'provider': 'Anthropic API',
                'description': '최고 품질, API 키 필요, 유료',
                'best_for': '전반적 최고 품질'
            },
            'solar': {
                'name': 'Solar 10.7B',
                'provider': 'Ollama (Local)',
                'description': '한국어 최적화, 무료, 로컬 실행',
                'best_for': '한국어 기술 문서'
            },
            'qwen': {
                'name': 'Qwen2.5 7B',
                'provider': 'Ollama (Local)',
                'description': '기술 문서 특화, 무료, 경량',
                'best_for': '기술 문서 일반'
            }
        }
        
        print("🤖 Altibase Employee Q&A Chatbot 초기화 중...")
        self._show_model_info()
        self._load_vector_database()
        self._initialize_llm()
        print("✅ 초기화 완료!")
    
    def _show_model_info(self):
        """Show information about the selected model"""
        if self.model_type in self.model_info:
            info = self.model_info[self.model_type]
            print(f"🧠 선택된 모델: {info['name']} ({info['provider']})")
            print(f"   특징: {info['description']}")
            print(f"   최적 용도: {info['best_for']}")
        
    def _load_vector_database(self):
        """Load your existing vector database"""
        try:
            # Import your existing vector system
            from paste import ManualVectorDB, ManualVectorizer
            
            print(f"📚 벡터 데이터베이스 로딩: {self.vector_db_path}")
            self.vectorizer = ManualVectorizer()
            self.vector_db = ManualVectorDB.load(self.vector_db_path, self.vectorizer.vector_size)
            
            print(f"✅ 로드 완료: {self.vector_db.total_vectors:,}개 벡터, {len(self.vector_db.chunks):,}개 청크")
            
        except ImportError:
            print("❌ 벡터 데이터베이스 모듈을 찾을 수 없습니다.")
            print("   paste.py 파일이 같은 디렉토리에 있는지 확인하세요.")
            exit(1)
        except Exception as e:
            print(f"❌ 벡터 데이터베이스 로드 실패: {e}")
            exit(1)
    
    def _initialize_llm(self):
        """Initialize the selected LLM"""
        print(f"🚀 {self.model_type.upper()} 모델 초기화 중...")
        
        if self.model_type == "claude":
            self._init_claude()
        elif self.model_type == "solar":
            self._init_local_llm("solar:10.7b")
        elif self.model_type == "qwen":
            self._init_local_llm("qwen2.5:7b")
        else:
            print(f"❌ 지원하지 않는 모델: {self.model_type}")
            print("   지원 모델: claude, solar, qwen")
            self._show_model_comparison()
            exit(1)
    
    def _init_claude(self):
        """Initialize Claude AI"""
        try:
            import anthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                print("❌ ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
                print("   export ANTHROPIC_API_KEY='your_api_key'")
                exit(1)
            
            self.llm_client = anthropic.Anthropic(api_key=api_key)
            print("✅ Claude AI 초기화 완료")
            
        except ImportError:
            print("❌ anthropic 패키지가 설치되지 않았습니다.")
            print("   pip install anthropic")
            exit(1)
    
    def _init_local_llm(self, model_name: str):
        """Initialize local LLM via Ollama"""
        try:
            import requests
            
            # Test Ollama connection
            print(f"🔌 Ollama 서버 연결 확인 중...")
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code != 200:
                    raise Exception("Ollama 서버에 연결할 수 없습니다")
            except requests.exceptions.RequestException:
                print("❌ Ollama 서버가 실행되지 않았습니다.")
                self._show_ollama_setup(model_name)
                exit(1)
            
            # Check if model is available
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            
            # Check for exact model name or partial match
            model_found = False
            for available_model in available_models:
                if model_name in available_model or available_model.startswith(model_name.split(':')[0]):
                    model_found = True
                    # Use the exact model name found
                    self.llm_client = available_model
                    print(f"✅ 발견된 모델 사용: {available_model}")
                    break
            
            if not model_found:
                print(f"❌ {model_name} 모델이 설치되지 않았습니다.")
                print(f"   사용 가능한 모델: {[m['name'] for m in models]}")
                print(f"   다음 명령어로 설치하세요: ollama pull {model_name}")
                self._show_ollama_setup(model_name)
                exit(1)
            
            # Test model generation
            test_response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.llm_client, "prompt": "Hello", "stream": False},
                timeout=30
            )
            
            if test_response.status_code != 200:
                raise Exception(f"모델 테스트 실패: {test_response.status_code}")
                
            print(f"✅ {self.llm_client} 모델 초기화 완료")
                
        except Exception as e:
            print(f"❌ {model_name} 초기화 실패: {e}")
            self._show_ollama_setup(model_name)
            exit(1)
    
    def _show_ollama_setup(self, model_name: str):
        """Show Ollama setup instructions"""
        print(f"\n📋 {model_name} 설정 방법:")
        print("1. Ollama 설치: https://ollama.ai")
        print("2. 모델 다운로드:")
        print(f"   ollama pull {model_name}")
        print("3. Ollama 서버 시작:")
        print("   ollama serve")
        print("4. 서버가 실행 중인지 확인:")
        print("   curl http://localhost:11434/api/tags")
        print("5. 다시 실행:")
        print(f"   python {__file__} --model {self.model_type}")
    
    def _show_model_comparison(self):
        """Show model comparison"""
        print("\n🔍 사용 가능한 모델 비교:")
        print("="*60)
        for model_type, info in self.model_info.items():
            print(f"🤖 {info['name']} ({model_type})")
            print(f"   제공자: {info['provider']}")
            print(f"   특징: {info['description']}")
            print(f"   최적 용도: {info['best_for']}")
            print()
        
        print("💡 추천:")
        print("• 최고 품질을 원한다면: claude")
        print("• 한국어 기술 문서에 특화: solar") 
        print("• 가벼운 로컬 모델: qwen")
    
    def get_relevant_context(self, question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """Get relevant context from vector database"""
        try:
            # Search for relevant chunks
            results = self.vector_db.search(
                question, 
                k=top_k, 
                vectorizer=self.vectorizer
            )
            
            if not results:
                return "", []
            
            # Build context
            context_parts = []
            citations = []
            
            for i, result in enumerate(results):
                chunk = result['chunk']
                metadata = chunk.get('metadata', {})
                
                context_part = f"""
[참고자료 {i+1}] {chunk['manual_title']} - {chunk['section_key']}
챕터: {metadata.get('chapter', 'N/A')}
유사도: {result['similarity']*100:.1f}%

{chunk['text'][:600]}...
"""
                context_parts.append(context_part)
                
                citations.append({
                    'manual_title': chunk['manual_title'],
                    'section_key': chunk['section_key'],
                    'chapter': metadata.get('chapter', 'N/A'),
                    'similarity': result['similarity']
                })
            
            return "\n".join(context_parts), citations
            
        except Exception as e:
            print(f"⚠️ 컨텍스트 검색 중 오류: {e}")
            return "", []
    
    def generate_response(self, question: str, context: str) -> str:
        """Generate response using the selected LLM"""
        try:
            if self.model_type == "claude":
                return self._generate_claude_response(question, context)
            elif self.model_type in ["solar", "qwen"]:
                return self._generate_local_response(question, context, self.llm_client)
        except Exception as e:
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _generate_claude_response(self, question: str, context: str) -> str:
        """Generate response using Claude"""
        system_prompt = """당신은 Altibase 제품 전문가이며, 신입 직원들의 온보딩을 돕는 AI 어시스턴트입니다.

다음 지침을 따라주세요:
1. 제공된 매뉴얼 정보를 바탕으로 정확하고 유용한 답변을 제공하세요
2. 답변은 친근하고 이해하기 쉽게 설명하세요
3. 가능한 경우 구체적인 예시나 단계별 설명을 포함하세요
4. 확실하지 않은 정보는 추측하지 말고, 추가 확인이 필요하다고 안내하세요"""

        user_prompt = f"""관련 매뉴얼 정보:
{context}

질문: {question}

위의 매뉴얼 정보를 참고하여 질문에 대한 상세하고 도움이 되는 답변을 제공해주세요."""

        # Add conversation history
        messages = []
        for turn in self.conversation_history[-5:]:  # Last 5 turns
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        
        messages.append({"role": "user", "content": user_prompt})
        
        response = self.llm_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.3,
            system=system_prompt,
            messages=messages
        )
        
        return response.content[0].text
    
    def _generate_local_response(self, question: str, context: str, model_name: str) -> str:
        """Generate response using local LLM via Ollama"""
        import requests
        
        # Optimize prompt based on model
        if "solar" in model_name.lower():
            # Solar is optimized for Korean
            system_prompt = """당신은 Altibase 제품 전문가입니다. 신입 직원들에게 친근하고 정확한 답변을 제공하세요.

지침:
1. 제공된 매뉴얼 정보를 정확히 활용하세요
2. 단계별로 명확하게 설명하세요  
3. 실무에 도움이 되는 구체적인 예시를 포함하세요
4. 불확실한 정보는 추측하지 마세요"""
            
            prompt = f"""{system_prompt}

관련 매뉴얼 정보:
{context}

질문: {question}

답변:"""

        else:  # qwen and others
            # Qwen is good with structured prompts
            system_prompt = """당신은 Altibase 제품 전문가입니다. 신입 직원을 도와주는 AI 어시스턴트입니다.

지침:
- 제공된 매뉴얼 정보를 정확히 활용하세요
- 단계별로 명확하게 설명하세요
- 실무에 도움이 되는 예시를 포함하세요
- 불확실한 정보는 추측하지 마세요"""
            
            prompt = f"""{system_prompt}

매뉴얼 정보:
{context}

질문: {question}

한국어로 상세한 답변을 제공해주세요:"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": 2000,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=120  # Local models might be slower
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise Exception(f"Ollama API 오류: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            return "죄송합니다. 응답 생성 시간이 초과되었습니다. 다시 시도해주세요."
        except Exception as e:
            raise Exception(f"로컬 모델 오류: {str(e)}")
    
    def chat(self, question: str) -> Dict[str, Any]:
        """Main chat function"""
        start_time = time.time()
        
        # Get relevant context
        print("🔍 관련 정보 검색 중...")
        context, citations = self.get_relevant_context(question)
        
        # Generate response
        print("🤔 답변 생성 중...")
        response = self.generate_response(question, context)
        
        # Save to history
        self.conversation_history.append({
            "user": question,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 conversations
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return {
            "question": question,
            "answer": response,
            "citations": citations,
            "response_time": time.time() - start_time,
            "model": self.model_type
        }
    
    def run_cli(self):
        """Run the command-line interface"""
        print("\n" + "="*80)
        print("          🤖 Altibase Employee Q&A Chatbot")
        print("="*80)
        
        model_info = self.model_info[self.model_type]
        print(f"사용 중인 모델: {model_info['name']} ({model_info['provider']})")
        print(f"벡터 데이터베이스: {self.vector_db.total_vectors:,}개 벡터")
        print(f"최적 용도: {model_info['best_for']}")
        
        print("\n명령어:")
        print("  질문 입력      - 일반 질문")
        print("  /help         - 도움말")
        print("  /models       - 모델 비교")
        print("  /switch       - 모델 변경")
        print("  /clear        - 대화 기록 삭제")
        print("  /quit         - 종료")
        print("="*80)
        
        while True:
            try:
                question = input("\n❓ 질문을 입력하세요: ").strip()
                
                if not question:
                    continue
                
                # Handle commands
                if question == "/quit":
                    print("👋 시스템을 종료합니다.")
                    break
                
                elif question == "/help":
                    self._show_help()
                    continue
                
                elif question == "/models":
                    self._show_model_comparison()
                    continue
                
                elif question == "/switch":
                    self._switch_model()
                    continue
                
                elif question == "/clear":
                    self.conversation_history = []
                    print("🗑️ 대화 기록이 삭제되었습니다.")
                    continue
                
                # Generate response
                result = self.chat(question)
                
                # Display response
                print(f"\n🤖 **답변:**")
                print(result['answer'])
                
                # Display citations
                if result['citations']:
                    print(f"\n📚 **참고 자료:**")
                    for i, citation in enumerate(result['citations'][:3]):
                        print(f"{i+1}. {citation['manual_title']} - {citation['section_key']}")
                        print(f"   📊 유사도: {citation['similarity']*100:.1f}%")
                
                print(f"\n⏱️ 응답 시간: {result['response_time']:.2f}초 | 모델: {result['model'].upper()}")
                
            except KeyboardInterrupt:
                print("\n\n👋 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
    
    def _switch_model(self):
        """Switch between models"""
        print("\n🔄 모델 변경:")
        print("1. claude  - Claude 3 Sonnet (최고 품질, API 키 필요)")
        print("2. solar   - Solar 10.7B (한국어 특화, 로컬)")
        print("3. qwen    - Qwen2.5 7B (기술 문서 특화, 로컬)")
        
        choice = input("\n선택 (1-3): ").strip()
        
        model_map = {'1': 'claude', '2': 'solar', '3': 'qwen'}
        
        if choice in model_map:
            new_model = model_map[choice]
            if new_model != self.model_type:
                print(f"\n🔄 {new_model} 모델로 변경 중...")
                try:
                    old_model = self.model_type
                    self.model_type = new_model
                    self._initialize_llm()
                    print(f"✅ {self.model_info[new_model]['name']} 모델로 변경 완료!")
                except Exception as e:
                    print(f"❌ 모델 변경 실패: {e}")
                    print("기존 모델을 유지합니다.")
                    self.model_type = old_model
            else:
                print("ℹ️  이미 선택된 모델입니다.")
        else:
            print("❌ 잘못된 선택입니다.")
    
    def _show_help(self):
        """Show help information"""
        print(f"""
📋 사용법:
• 자연어로 질문하세요 (예: "Altibase 설치 방법을 알려주세요")
• 구체적인 질문일수록 더 정확한 답변을 받을 수 있습니다

💡 질문 예시:
• "Altibase가 무엇인가요?"
• "데이터베이스 연결은 어떻게 하나요?"
• "SQL 쿼리 최적화 방법은?"
• "백업과 복구는 어떻게 하나요?"
• "성능 튜닝 가이드를 알려주세요"
• "에러 코드별 해결 방법은?"
• "클러스터링 설정은 어떻게 하나요?"
• "메모리 관리 방법을 알려주세요"

🔧 명령어:
/help    - 이 도움말 표시
/models  - 사용 가능한 모델 비교
/switch  - 다른 모델로 변경
/clear   - 대화 기록 삭제
/quit    - 프로그램 종료

🤖 현재 사용 중인 모델: {self.model_info[self.model_type]['name']}
   특징: {self.model_info[self.model_type]['description']}
""")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Altibase Employee Q&A Chatbot with Claude API and Local LLM')
    parser.add_argument('--vector-db', type=str, default='manual_vector_db',
                       help='Vector database directory path')
    parser.add_argument('--model', type=str, default='claude',
                       choices=['claude', 'solar', 'qwen'],
                       help='AI model to use')
    
    args = parser.parse_args()
    
    # Show initial model information
    print("🚀 Altibase Employee Q&A Chatbot")
    print("="*50)
    print("사용 가능한 모델:")
    print("• claude - Claude 3 Sonnet (최고 품질, API 키 필요)")
    print("• solar  - Solar 10.7B (한국어 특화, 무료 로컬)")
    print("• qwen   - Qwen2.5 7B (기술 문서 특화, 무료 로컬)")
    print(f"\n선택된 모델: {args.model}")
    
    if args.model == 'claude':
        if not os.getenv('ANTHROPIC_API_KEY'):
            print("\n⚠️  Claude 사용을 위해 API 키가 필요합니다:")
            print("   export ANTHROPIC_API_KEY='your_api_key'")
            print("   또는 로컬 모델을 사용하세요: --model solar")
    elif args.model in ['solar', 'qwen']:
        print(f"\n💡 {args.model} 모델 설정 방법:")
        model_name = "solar:10.7b" if args.model == 'solar' else "qwen2.5:7b"
        print(f"1. Ollama 설치: https://ollama.ai")
        print(f"2. 모델 다운로드: ollama pull {model_name}")
        print(f"3. 서버 시작: ollama serve")
        print(f"4. 확인: curl http://localhost:11434/api/tags")
    
    print("="*50)
    
    try:
        # Initialize and run chatbot
        chatbot = SimpleChatbot(args.vector_db, args.model)
        chatbot.run_cli()
        
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()