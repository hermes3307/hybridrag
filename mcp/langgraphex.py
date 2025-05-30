"""
개선된 컨텍스트 유지 대화 시스템
- GPT-2 모델 사용 제거
- 더 나은 규칙 기반 응답
- OpenAI API 또는 로컬 모델 선택 가능
- 한국어/영어 대화 지원
"""

import json
import re
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# OpenAI API 선택적 import
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Ollama 선택적 import (로컬 LLM)
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

@dataclass
class UserProfile:
    """사용자 프로필 정보"""
    name: Optional[str] = None
    age: Optional[int] = None
    occupation: Optional[str] = None
    location: Optional[str] = None
    hobbies: List[str] = None
    preferences: Dict[str, Any] = None
    language_preference: str = "auto"  # auto, korean, english
    
    def __post_init__(self):
        if self.hobbies is None:
            self.hobbies = []
        if self.preferences is None:
            self.preferences = {}
    
    def update_from_text(self, text: str):
        """텍스트에서 사용자 정보 추출"""
        # 이름 추출 (한국어/영어)
        name_patterns = [
            r"제?\s?이름은\s+([가-힣A-Za-z]+)",
            r"저는\s+([가-힣A-Za-z]+)입니다",
            r"나는\s+([가-힣A-Za-z]+)이야",
            r"([가-힣A-Za-z]+)라고\s+해",
            r"my name is\s+([A-Za-z]+)",
            r"I am\s+([A-Za-z]+)",
            r"call me\s+([A-Za-z]+)",
            r"^([가-힣A-Za-z]+)$"  # 단일 이름
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                potential_name = match.group(1)
                # 일반적인 단어 제외
                if potential_name.lower() not in ['hello', 'hi', 'yes', 'no', 'okay', 'good', 'great', 'wow', 'man', 'woman']:
                    self.name = potential_name
                    break
        
        # 나이 추출
        age_patterns = [
            r"나이는?\s*(\d+)살",
            r"(\d+)살입니다",
            r"age is\s*(\d+)",
            r"I am\s*(\d+)\s*years old",
            r"(\d+)\s*years old"
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 1 <= age <= 120:  # 유효한 나이 범위
                    self.age = age
                    break
        
        # 직업 추출
        job_patterns = [
            r"직업은\s+([가-힣A-Za-z]+)",
            r"([가-힣A-Za-z]+)로\s+일하고",
            r"work as\s+a?\s*([A-Za-z]+)",
            r"I am\s+a\s+([A-Za-z]+)",
            r"job is\s+([A-Za-z]+)"
        ]
        
        for pattern in job_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self.occupation = match.group(1)
                break
        
        # 거주지 추출
        location_patterns = [
            r"([가-힣]+시|[가-힣]+구)에\s+살고",
            r"거주지는\s+([가-힣A-Za-z\s]+)",
            r"live in\s+([A-Za-z\s]+)",
            r"from\s+([A-Za-z\s]+)"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self.location = match.group(1)
                break
        
        # 언어 선호도 감지
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if korean_chars > english_chars:
            self.language_preference = "korean"
        elif english_chars > korean_chars:
            self.language_preference = "english"

class ConversationState(TypedDict):
    """대화 상태"""
    messages: List[Dict[str, str]]
    user_profile: Dict[str, Any]
    context_summary: str
    current_input: str
    response: str
    language: str

class ImprovedChatBot:
    """개선된 컨텍스트 유지 챗봇"""
    
    def __init__(self, llm_type: str = "rule_based", api_key: str = None):
        """
        llm_type: "rule_based", "openai", "ollama"
        api_key: OpenAI API 키 (필요시)
        """
        self.llm_type = llm_type
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        
        # OpenAI 설정
        if llm_type == "openai" and api_key and OPENAI_AVAILABLE:
            openai.api_key = api_key
            self.llm_available = True
            print("✅ OpenAI API 연결됨")
        elif llm_type == "ollama" and OLLAMA_AVAILABLE:
            self.llm_available = self._check_ollama()
        else:
            self.llm_available = False
            self.llm_type = "rule_based"
            print("📝 규칙 기반 응답 사용")
    
    def _check_ollama(self) -> bool:
        """Ollama 서버 연결 확인"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama 서버 연결됨")
                return True
        except:
            pass
        print("⚠️ Ollama 서버 연결 실패")
        return False
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(ConversationState)
        
        workflow.add_node("extract_info", self._extract_info)
        workflow.add_node("detect_language", self._detect_language)
        workflow.add_node("generate_response", self._generate_response)
        
        workflow.set_entry_point("extract_info")
        workflow.add_edge("extract_info", "detect_language")
        workflow.add_edge("detect_language", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _extract_info(self, state: ConversationState) -> ConversationState:
        """사용자 정보 추출"""
        current_input = state["current_input"]
        
        if state["user_profile"]:
            profile = UserProfile(**state["user_profile"])
        else:
            profile = UserProfile()
        
        profile.update_from_text(current_input)
        state["user_profile"] = asdict(profile)
        return state
    
    def _detect_language(self, state: ConversationState) -> ConversationState:
        """언어 감지"""
        current_input = state["current_input"]
        korean_chars = len(re.findall(r'[가-힣]', current_input))
        english_chars = len(re.findall(r'[a-zA-Z]', current_input))
        
        if korean_chars > english_chars:
            state["language"] = "korean"
        else:
            state["language"] = "english"
        
        return state
    
    def _generate_response(self, state: ConversationState) -> ConversationState:
        """응답 생성"""
        user_input = state["current_input"]
        user_profile = state["user_profile"]
        language = state["language"]
        
        if self.llm_available:
            if self.llm_type == "openai":
                response = self._openai_response(user_input, user_profile, language)
            elif self.llm_type == "ollama":
                response = self._ollama_response(user_input, user_profile, language)
            else:
                response = self._rule_based_response(user_input, user_profile, language)
        else:
            response = self._rule_based_response(user_input, user_profile, language)
        
        state["response"] = response
        
        # 메시지 히스토리에 추가
        state["messages"].extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response}
        ])
        
        return state
    
    def _rule_based_response(self, user_input: str, profile: Dict[str, Any], language: str) -> str:
        """개선된 규칙 기반 응답"""
        user_input_lower = user_input.lower()
        name = profile.get("name", "")
        age = profile.get("age", "")
        
        # 한국어 응답
        if language == "korean":
            # 인사말
            if any(word in user_input_lower for word in ["안녕", "처음", "반가"]):
                if name:
                    return f"안녕하세요, {name}님! 오늘은 어떤 이야기를 나누고 싶으신가요?"
                return "안녕하세요! 만나서 반갑습니다. 자기소개를 해주시면 좋겠어요."
            
            # 정보 확인
            if any(word in user_input_lower for word in ["내 정보", "내가 누구", "기억", "프로필"]):
                if name or age:
                    info_parts = []
                    if name:
                        info_parts.append(f"이름: {name}")
                    if age:
                        info_parts.append(f"나이: {age}살")
                    return f"제가 알고 있는 정보입니다:\n" + " | ".join(info_parts)
                return "아직 당신에 대한 정보를 알지 못해요. 더 많은 이야기를 들려주세요!"
            
            # 정보 입력 감지
            if any(word in user_input for word in ["이름은", "나이는", "살입니다", "직업은"]):
                responses = [
                    "알려주셔서 감사합니다! 잘 기억해두겠습니다.",
                    "좋은 정보를 공유해주셔서 고마워요.",
                    "네, 기억했습니다! 다른 이야기도 들려주세요."
                ]
                return random.choice(responses)
            
            # 감정 표현
            if any(word in user_input_lower for word in ["기뻐", "행복", "좋아", "즐거워"]):
                return f"좋은 소식이네요! {name + '님' if name else ''} 더 자세히 들려주세요!"
            
            if any(word in user_input_lower for word in ["슬퍼", "우울", "힘들어", "속상"]):
                return f"{name + '님' if name else ''} 힘든 일이 있으셨나요? 이야기하고 싶으시면 언제든 들어드릴게요."
            
            # 일반 응답
            general_responses = [
                f"그렇군요{', ' + name + '님' if name else ''}! 더 자세히 말씀해주시겠어요?",
                "흥미로운 이야기네요. 어떻게 생각하시나요?",
                "이해했습니다. 다른 관점도 있을까요?",
                "좋은 점을 말씀해주셨네요. 더 궁금한 것이 있으시면 물어보세요."
            ]
        
        # 영어 응답
        else:
            # 인사말
            if any(word in user_input_lower for word in ["hello", "hi", "hey", "nice"]):
                if name:
                    return f"Hello, {name}! What would you like to talk about today?"
                return "Hello! Nice to meet you. Please tell me about yourself!"
            
            # 정보 확인
            if any(word in user_input_lower for word in ["my info", "who am i", "remember", "profile"]):
                if name or age:
                    info_parts = []
                    if name:
                        info_parts.append(f"Name: {name}")
                    if age:
                        info_parts.append(f"Age: {age}")
                    return f"Here's what I know about you:\n" + " | ".join(info_parts)
                return "I don't have much information about you yet. Please tell me more!"
            
            # 이름 질문
            if any(phrase in user_input_lower for phrase in ["what is my name", "my name", "who am i"]):
                if name:
                    return f"Your name is {name}!"
                return "I don't know your name yet. Could you tell me?"
            
            # 정보 입력 감지
            if any(word in user_input_lower for word in ["my name is", "i am", "call me", "years old"]):
                responses = [
                    "Thank you for sharing that! I'll remember it.",
                    "Great! Thanks for the information.",
                    "Got it! Tell me more about yourself."
                ]
                return random.choice(responses)
            
            # 감정 표현
            if any(word in user_input_lower for word in ["happy", "good", "great", "wonderful", "awesome"]):
                return f"That's wonderful{', ' + name if name else ''}! Tell me more about it!"
            
            if any(word in user_input_lower for word in ["sad", "bad", "terrible", "awful", "down"]):
                return f"I'm sorry to hear that{', ' + name if name else ''}. Would you like to talk about it?"
            
            # 일반 응답
            general_responses = [
                f"That's interesting{', ' + name if name else ''}! Could you tell me more?",
                "I see. What do you think about that?",
                "Understood. Are there other perspectives?",
                "Good point! Do you have any questions for me?"
            ]
        
        return random.choice(general_responses if language == "english" else general_responses[:4])
    
    def _openai_response(self, user_input: str, profile: Dict[str, Any], language: str) -> str:
        """OpenAI API 응답"""
        try:
            name = profile.get("name", "")
            age = profile.get("age", "")
            
            if language == "korean":
                system_prompt = f"""당신은 친근하고 도움이 되는 AI 어시스턴트입니다.
사용자 정보: 이름={name}, 나이={age}
한국어로 자연스럽고 친근하게 대화하세요. 간단하고 명확한 응답을 해주세요."""
            else:
                system_prompt = f"""You are a friendly and helpful AI assistant.
User info: name={name}, age={age}
Respond naturally and warmly in English. Keep responses concise and clear."""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"OpenAI 오류: {str(e)}")
            return self._rule_based_response(user_input, profile, language)
    
    def _ollama_response(self, user_input: str, profile: Dict[str, Any], language: str) -> str:
        """Ollama 로컬 LLM 응답"""
        try:
            name = profile.get("name", "")
            age = profile.get("age", "")
            
            if language == "korean":
                prompt = f"""사용자 정보: 이름={name}, 나이={age}
사용자: {user_input}
AI: (한국어로 친근하게 응답)"""
            else:
                prompt = f"""User info: name={name}, age={age}
User: {user_input}
AI: (respond warmly in English)"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",  # 또는 사용 가능한 모델
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
        
        except Exception as e:
            print(f"Ollama 오류: {str(e)}")
        
        return self._rule_based_response(user_input, profile, language)
    
    def chat(self, message: str, thread_id: str = "default") -> str:
        """대화 함수"""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            current_state = self.graph.get_state(config)
            if current_state.values:
                state = current_state.values
            else:
                state = {
                    "messages": [],
                    "user_profile": {},
                    "context_summary": "",
                    "current_input": "",
                    "response": "",
                    "language": "auto"
                }
        except:
            state = {
                "messages": [],
                "user_profile": {},
                "context_summary": "",
                "current_input": "",
                "response": "",
                "language": "auto"
            }
        
        state["current_input"] = message
        result = self.graph.invoke(state, config)
        return result["response"]
    
    def get_profile(self, thread_id: str = "default") -> Dict[str, Any]:
        """프로필 조회"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            current_state = self.graph.get_state(config)
            if current_state.values:
                return current_state.values.get("user_profile", {})
        except:
            pass
        return {}

def main():
    """메인 실행 함수"""
    print("=== 개선된 컨텍스트 유지 대화 시스템 ===")
    print("LLM 선택:")
    print("1. 규칙 기반 (추천)")
    print("2. OpenAI API")
    print("3. Ollama (로컬)")
    
    choice = input("선택 (1-3): ").strip()
    
    if choice == "2":
        api_key = input("OpenAI API 키: ").strip()
        chatbot = ImprovedChatBot("openai", api_key)
    elif choice == "3":
        chatbot = ImprovedChatBot("ollama")
    else:
        chatbot = ImprovedChatBot("rule_based")
    
    print("\n명령어: 'quit' (종료), 'profile' (프로필 확인)")
    print("-" * 50)
    
    thread_id = "user_main"
    
    while True:
        user_input = input("\n사용자: ").strip()
        
        if user_input.lower() == 'quit':
            print("대화를 종료합니다.")
            break
        elif user_input.lower() == 'profile':
            profile = chatbot.get_profile(thread_id)
            print("저장된 프로필:")
            for key, value in profile.items():
                if value:
                    print(f"  {key}: {value}")
            continue
        
        if user_input:
            try:
                response = chatbot.chat(user_input, thread_id)
                print(f"AI: {response}")
            except Exception as e:
                print(f"오류: {str(e)}")

if __name__ == "__main__":
    main()