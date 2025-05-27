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
            model_type: AI model to use ('claude', 'openai', or 'llama3')
        """
        self.vector_db_path = vector_db_path
        self.model_type = model_type.lower()
        self.vector_db = None
        self.vectorizer = None
        self.llm_client = None
        self.conversation_history = []
        
        print("🤖 Altibase Employee Q&A Chatbot 초기화 중...")
        self._load_vector_database()
        self._initialize_llm()
        print("✅ 초기화 완료!")
    
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
        print(f"🧠 {self.model_type.upper()} 모델 초기화 중...")
        
        if self.model_type == "claude":
            self._init_claude()
        elif self.model_type == "openai":
            self._init_openai()
        elif self.model_type == "llama3":
            self._init_llama3()
        else:
            print(f"❌ 지원하지 않는 모델: {self.model_type}")
            print("   지원 모델: claude, openai, llama3")
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
    
    def _init_openai(self):
        """Initialize OpenAI"""
        try:
            import openai
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
                print("   export OPENAI_API_KEY='your_api_key'")
                exit(1)
            
            self.llm_client = openai.OpenAI(api_key=api_key)
            print("✅ OpenAI 초기화 완료")
            
        except ImportError:
            print("❌ openai 패키지가 설치되지 않았습니다.")
            print("   pip install openai")
            exit(1)
    
    def _init_llama3(self):
        """Initialize Llama3 via Ollama"""
        try:
            import requests
            
            # Test Ollama connection
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": "test", "stream": False},
                timeout=5
            )
            
            if response.status_code == 200:
                self.llm_client = "ollama"
                print("✅ Llama3 (Ollama) 초기화 완료")
            else:
                raise Exception("Ollama 연결 실패")
                
        except Exception as e:
            print("❌ Llama3 초기화 실패")
            print("   1. Ollama를 설치하세요: https://ollama.ai")
            print("   2. Llama3 모델을 다운로드하세요: ollama pull llama3")
            print("   3. Ollama 서버를 시작하세요: ollama serve")
            exit(1)
    
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
            elif self.model_type == "openai":
                return self._generate_openai_response(question, context)
            elif self.model_type == "llama3":
                return self._generate_llama3_response(question, context)
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
    
    def _generate_openai_response(self, question: str, context: str) -> str:
        """Generate response using OpenAI"""
        system_prompt = "당신은 Altibase 제품 전문가이며 신입직원 온보딩을 돕는 AI 어시스턴트입니다. 매뉴얼 정보를 바탕으로 친근하고 정확한 답변을 제공하세요."
        
        user_prompt = f"""관련 매뉴얼 정보:
{context}

질문: {question}

위 정보를 참고하여 답변해주세요."""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for turn in self.conversation_history[-5:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        
        messages.append({"role": "user", "content": user_prompt})
        
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _generate_llama3_response(self, question: str, context: str) -> str:
        """Generate response using Llama3 via Ollama"""
        import requests
        
        prompt = f"""당신은 Altibase 제품 전문가입니다. 다음 매뉴얼 정보를 참고하여 질문에 답하세요.

매뉴얼 정보:
{context}

질문: {question}

답변:"""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_predict": 1500
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama API 오류: {response.status_code}")
    
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
        print(f"사용 중인 모델: {self.model_type.upper()}")
        print(f"벡터 데이터베이스: {self.vector_db.total_vectors:,}개 벡터")
        print("\n명령어:")
        print("  질문 입력     - 일반 질문")
        print("  /help        - 도움말")
        print("  /clear       - 대화 기록 삭제")
        print("  /quit        - 종료")
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
                
                print(f"\n⏱️ 응답 시간: {result['response_time']:.2f}초")
                
            except KeyboardInterrupt:
                print("\n\n👋 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("""
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

🔧 명령어:
/help  - 이 도움말 표시
/clear - 대화 기록 삭제
/quit  - 프로그램 종료
""")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Altibase Employee Q&A Chatbot')
    parser.add_argument('--vector-db', type=str, default='manual_vector_db',
                       help='Vector database directory path')
    parser.add_argument('--model', type=str, default='claude',
                       choices=['claude', 'openai', 'llama3'],
                       help='AI model to use')
    
    args = parser.parse_args()
    
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