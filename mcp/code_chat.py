import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

# OpenAI 클라이언트 임포트 (새 버전)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI not available, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai>=1.0.0"])
    from openai import OpenAI
    OPENAI_AVAILABLE = True

# MCP 클라이언트 임포트 (선택사항)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not available - some features may be limited")

class CodingAssistantBot:
    """코딩을 도와주는 AI 챗봇"""
    
    def __init__(self, openai_api_key: str = None):
        # OpenAI API 키 설정
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        # OpenAI 클라이언트 초기화
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # 대화 히스토리 저장 (컨텍스트 유지용)
        self.conversation_history = []
        
        # MCP 서버 연결 정보
        self.mcp_session = None
        
        # 지원하는 프로그래밍 언어들
        self.supported_languages = {
            'python': {'extension': '.py', 'comment': '#'},
            'javascript': {'extension': '.js', 'comment': '//'},
            'java': {'extension': '.java', 'comment': '//'},
            'cpp': {'extension': '.cpp', 'comment': '//'},
            'c': {'extension': '.c', 'comment': '//'},
            'html': {'extension': '.html', 'comment': '<!--'},
            'css': {'extension': '.css', 'comment': '/*'},
            'sql': {'extension': '.sql', 'comment': '--'},
            'bash': {'extension': '.sh', 'comment': '#'},
            'go': {'extension': '.go', 'comment': '//'},
            'rust': {'extension': '.rs', 'comment': '//'},
            'php': {'extension': '.php', 'comment': '//'},
            'ruby': {'extension': '.rb', 'comment': '#'},
            'swift': {'extension': '.swift', 'comment': '//'},
            'kotlin': {'extension': '.kt', 'comment': '//'}
        }
        
        print(f"🤖 Coding Assistant Bot initialized")
        print(f"✅ OpenAI API: Connected")
        print(f"🔧 MCP: {'Available' if MCP_AVAILABLE else 'Not available'}")
        print(f"💬 Context: Ready to maintain conversation history")
    
    async def initialize_mcp_server(self, server_command: str = None):
        """MCP 서버 초기화 (선택사항)"""
        if not MCP_AVAILABLE:
            print("⚠️  MCP is not available")
            return False
        
        try:
            if server_command:
                # 사용자 정의 MCP 서버 명령어 (문자열로 전달)
                server_params = StdioServerParameters(command=server_command)
            else:
                # 기본 MCP 서버들을 시도해보기
                possible_commands = [
                    # 파일 시스템 서버
                    f"python -m mcp.server.filesystem {str(Path.cwd())}",
                    # SQLite 서버 (있다면)
                    "python -m mcp.server.sqlite",
                    # 기본 서버
                    "python -m mcp.server"
                ]
                
                # 각 명령어를 시도해보기
                for cmd in possible_commands:
                    try:
                        server_params = StdioServerParameters(command=cmd)
                        self.mcp_session = await stdio_client(server_params).__aenter__()
                        print(f"✅ MCP Server connected with command: {cmd}")
                        return True
                    except Exception as e:
                        print(f"⚠️  MCP command '{cmd}' failed: {e}")
                        continue
                
                print("⚠️  모든 기본 MCP 서버 연결 시도가 실패했습니다")
                return False
            
            # 사용자 정의 명령어로 연결 시도
            self.mcp_session = await stdio_client(server_params).__aenter__()
            print("✅ MCP Server connected")
            return True
            
        except Exception as e:
            print(f"⚠️  MCP Server connection failed: {e}")
            print("💡 MCP 서버 없이도 챗봇은 정상적으로 작동합니다")
            return False
    
    def add_to_conversation(self, role: str, content: str):
        """대화 히스토리에 메시지 추가"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 히스토리가 너무 길어지면 오래된 것들 제거 (최근 20개 메시지만 유지)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """OpenAI API용 대화 컨텍스트 생성"""
        context = [
            {
                "role": "system",
                "content": """You are an expert coding assistant. Your role is to:

1. Help users write, debug, and improve code in various programming languages
2. Explain code concepts and best practices
3. Debug errors and provide solutions
4. Review code and suggest improvements
5. Answer coding-related questions
6. Provide code examples and tutorials

Supported languages: Python, JavaScript, Java, C++, C, HTML, CSS, SQL, Bash, Go, Rust, PHP, Ruby, Swift, Kotlin

Guidelines:
- Always provide working, tested code when possible
- Explain your solutions clearly
- Consider edge cases and error handling
- Follow best practices for the specific language
- Be patient and helpful for beginners
- Provide multiple solutions when appropriate
- Include comments in code for clarity

Format your responses with:
- Clear explanations
- Properly formatted code blocks with language specification
- Step-by-step instructions when needed
- Error analysis and debugging tips
"""
            }
        ]
        
        # 최근 대화 히스토리 추가 (OpenAI 형식으로 변환)
        for msg in self.conversation_history[-10:]:  # 최근 10개만
            if msg["role"] in ["user", "assistant"]:
                context.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return context
    
    def detect_programming_language(self, code: str) -> str:
        """코드에서 프로그래밍 언어 감지"""
        code_lower = code.lower().strip()
        
        # 파일 확장자나 명시적 언어 표시가 있는지 확인
        if '```' in code:
            match = re.search(r'```(\w+)', code)
            if match:
                lang = match.group(1).lower()
                if lang in self.supported_languages:
                    return lang
        
        # 코드 패턴으로 언어 감지
        patterns = {
            'python': [r'def\s+\w+\s*\(', r'import\s+\w+', r'from\s+\w+\s+import', r'print\s*\(', r'if\s+__name__\s*==\s*["\']__main__["\']'],
            'javascript': [r'function\s+\w+\s*\(', r'const\s+\w+\s*=', r'let\s+\w+\s*=', r'var\s+\w+\s*=', r'console\.log\s*\('],
            'java': [r'public\s+class\s+\w+', r'public\s+static\s+void\s+main', r'System\.out\.print'],
            'cpp': [r'#include\s*<\w+>', r'int\s+main\s*\(', r'std::cout', r'using\s+namespace\s+std'],
            'c': [r'#include\s*<\w+\.h>', r'int\s+main\s*\(', r'printf\s*\('],
            'html': [r'<html', r'<head>', r'<body>', r'<div', r'<!DOCTYPE'],
            'css': [r'\{\s*[\w-]+\s*:', r'@media', r'\.[\w-]+\s*\{'],
            'sql': [r'SELECT\s+', r'FROM\s+\w+', r'WHERE\s+', r'INSERT\s+INTO', r'CREATE\s+TABLE'],
            'bash': [r'#!/bin/bash', r'echo\s+', r'\$\w+', r'if\s*\[\s*'],
            'go': [r'package\s+main', r'func\s+main\s*\(', r'import\s*\(', r'fmt\.Print'],
            'rust': [r'fn\s+main\s*\(', r'let\s+mut\s+', r'println!\s*\(', r'use\s+std::'],
            'php': [r'<\?php', r'\$\w+\s*=', r'echo\s+', r'function\s+\w+\s*\('],
            'ruby': [r'def\s+\w+', r'puts\s+', r'class\s+\w+', r'require\s+'],
            'swift': [r'func\s+\w+\s*\(', r'var\s+\w+:', r'let\s+\w+:', r'print\s*\('],
            'kotlin': [r'fun\s+main\s*\(', r'val\s+\w+\s*=', r'var\s+\w+\s*=', r'println\s*\(']
        }
        
        for lang, lang_patterns in patterns.items():
            for pattern in lang_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return lang
        
        return 'unknown'
    
    def extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """텍스트에서 코드 블록 추출"""
        code_blocks = []
        
        # ```로 감싸진 코드 블록 찾기
        pattern = r'```(\w*)\n?(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for lang, code in matches:
            if not lang:
                lang = self.detect_programming_language(code)
            code_blocks.append((lang.lower() if lang else 'unknown', code.strip()))
        
        # 백틱 없는 코드도 감지 (간단한 패턴)
        if not code_blocks:
            # 들여쓰기된 코드 블록 찾기
            lines = text.split('\n')
            code_lines = []
            for line in lines:
                if line.startswith('    ') or line.startswith('\t'):
                    code_lines.append(line)
                elif code_lines and line.strip() == '':
                    code_lines.append(line)
                else:
                    if code_lines:
                        code = '\n'.join(code_lines).strip()
                        if code:
                            lang = self.detect_programming_language(code)
                            code_blocks.append((lang, code))
                        code_lines = []
            
            # 마지막 코드 블록 처리
            if code_lines:
                code = '\n'.join(code_lines).strip()
                if code:
                    lang = self.detect_programming_language(code)
                    code_blocks.append((lang, code))
        
        return code_blocks
    
    def save_code_to_file(self, code: str, language: str, filename: str = None) -> str:
        """코드를 파일로 저장"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extension = self.supported_languages.get(language, {}).get('extension', '.txt')
                filename = f"code_{timestamp}{extension}"
            
            # 코드 저장 디렉토리 생성
            code_dir = Path.home() / "Documents" / "coding_assistant"
            code_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = code_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            return str(file_path)
            
        except Exception as e:
            return f"Error saving file: {e}"
    
    async def get_coding_help(self, user_input: str) -> str:
        """OpenAI를 통해 코딩 도움말 생성"""
        try:
            # 사용자 입력을 대화 히스토리에 추가
            self.add_to_conversation("user", user_input)
            
            # 대화 컨텍스트 생성
            messages = self.get_conversation_context()
            
            # OpenAI API 호출
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # 더 나은 코딩 지원을 위해 GPT-4 사용
                messages=messages,
                max_tokens=2000,
                temperature=0.1,  # 코딩은 일관성이 중요하므로 낮은 temperature
                stream=False
            )
            
            assistant_response = response.choices[0].message.content.strip()
            
            # 어시스턴트 응답을 대화 히스토리에 추가
            self.add_to_conversation("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"Error getting coding help: {str(e)}"
            self.add_to_conversation("assistant", error_msg)
            return error_msg
    
    def analyze_error_message(self, error: str, code: str = None) -> Dict[str, Any]:
        """에러 메시지 분석"""
        analysis = {
            "error_type": "unknown",
            "line_number": None,
            "description": error,
            "suggestions": []
        }
        
        # 일반적인 에러 패턴 분석
        error_patterns = {
            "SyntaxError": ["syntax error", "invalid syntax", "unexpected token"],
            "NameError": ["name error", "not defined", "undefined variable"],
            "TypeError": ["type error", "unsupported operand", "not callable"],
            "IndexError": ["index error", "list index out of range"],
            "KeyError": ["key error", "key not found"],
            "ImportError": ["import error", "no module named", "cannot import"],
            "IndentationError": ["indentation error", "unexpected indent"],
            "ValueError": ["value error", "invalid literal", "cannot convert"],
            "AttributeError": ["attribute error", "has no attribute"],
            "FileNotFoundError": ["file not found", "no such file"]
        }
        
        error_lower = error.lower()
        for error_type, patterns in error_patterns.items():
            if any(pattern in error_lower for pattern in patterns):
                analysis["error_type"] = error_type
                break
        
        # 라인 번호 추출
        line_match = re.search(r'line (\d+)', error, re.IGNORECASE)
        if line_match:
            analysis["line_number"] = int(line_match.group(1))
        
        # 에러별 제안사항
        suggestions_map = {
            "SyntaxError": ["문법을 확인하세요", "괄호나 따옴표가 제대로 닫혔는지 확인하세요", "들여쓰기를 확인하세요"],
            "NameError": ["변수나 함수 이름의 철자를 확인하세요", "변수가 정의되었는지 확인하세요", "import 문을 확인하세요"],
            "TypeError": ["데이터 타입을 확인하세요", "함수 호출 방법을 확인하세요", "연산자 사용법을 확인하세요"],
            "IndexError": ["리스트나 배열의 인덱스 범위를 확인하세요", "반복문의 범위를 확인하세요"],
            "ImportError": ["모듈이 설치되었는지 확인하세요", "import 경로를 확인하세요", "pip install로 패키지를 설치하세요"]
        }
        
        if analysis["error_type"] in suggestions_map:
            analysis["suggestions"] = suggestions_map[analysis["error_type"]]
        
        return analysis
    
    def format_response(self, response: str, code_blocks: List[Tuple[str, str]] = None) -> str:
        """응답 포맷팅"""
        formatted = response
        
        # 코드 블록이 있다면 파일 저장 옵션 제공
        if code_blocks:
            formatted += "\n\n📁 **코드 파일 저장 옵션:**\n"
            for i, (lang, code) in enumerate(code_blocks, 1):
                formatted += f"   {i}. {lang.upper()} 코드 ({len(code.split())} words)\n"
        
        return formatted
    
    async def chat(self, user_input: str) -> str:
        """메인 채팅 함수"""
        if not user_input.strip():
            return "❓ 코딩과 관련된 질문을 해주세요!"
        
        # 특수 명령어 처리
        if user_input.lower() in ['help', '도움말']:
            return self._get_help_message()
        
        if user_input.lower() in ['clear', '클리어', '히스토리 삭제']:
            self.conversation_history = []
            return "🗑️ 대화 히스토리가 삭제되었습니다."
        
        if user_input.lower() in ['history', '히스토리']:
            return self._show_conversation_history()
        
        if user_input.lower().startswith('save:'):
            # 코드 저장 명령
            return await self._handle_save_command(user_input[5:].strip())
        
        if user_input.lower().startswith('debug:'):
            # 디버그 명령
            return await self._handle_debug_command(user_input[6:].strip())
        
        # 일반적인 코딩 질문 처리
        print(f"🤔 Analyzing: {user_input[:50]}...")
        
        # OpenAI를 통해 응답 생성
        response = await self.get_coding_help(user_input)
        
        # 응답에서 코드 블록 추출
        code_blocks = self.extract_code_blocks(response)
        
        # 응답 포맷팅
        formatted_response = self.format_response(response, code_blocks)
        
        return formatted_response
    
    async def _handle_save_command(self, command: str) -> str:
        """코드 저장 명령 처리"""
        try:
            # 최근 대화에서 코드 블록 찾기
            recent_messages = self.conversation_history[-5:]  # 최근 5개 메시지
            
            all_code_blocks = []
            for msg in recent_messages:
                if msg["role"] == "assistant":
                    code_blocks = self.extract_code_blocks(msg["content"])
                    all_code_blocks.extend(code_blocks)
            
            if not all_code_blocks:
                return "❌ 저장할 코드 블록을 찾을 수 없습니다."
            
            saved_files = []
            for i, (lang, code) in enumerate(all_code_blocks, 1):
                filename = f"code_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                if lang in self.supported_languages:
                    filename += self.supported_languages[lang]['extension']
                else:
                    filename += '.txt'
                
                file_path = self.save_code_to_file(code, lang, filename)
                saved_files.append(f"   📄 {filename} ({lang})")
            
            return f"✅ 코드가 저장되었습니다:\n" + "\n".join(saved_files)
            
        except Exception as e:
            return f"❌ 코드 저장 실패: {e}"
    
    async def _handle_debug_command(self, error_info: str) -> str:
        """디버그 명령 처리"""
        debug_prompt = f"""다음 에러를 분석하고 해결 방법을 제시해주세요:

에러 정보:
{error_info}

이전 대화 컨텍스트도 고려해서 답변해주세요.
"""
        
        response = await self.get_coding_help(debug_prompt)
        
        # 에러 분석
        analysis = self.analyze_error_message(error_info)
        
        debug_response = f"🐛 **에러 분석 결과:**\n"
        debug_response += f"   에러 타입: {analysis['error_type']}\n"
        if analysis['line_number']:
            debug_response += f"   라인 번호: {analysis['line_number']}\n"
        debug_response += f"   설명: {analysis['description']}\n\n"
        
        if analysis['suggestions']:
            debug_response += f"💡 **제안사항:**\n"
            for suggestion in analysis['suggestions']:
                debug_response += f"   • {suggestion}\n"
            debug_response += "\n"
        
        debug_response += f"🤖 **AI 분석:**\n{response}"
        
        return debug_response
    
    def _show_conversation_history(self) -> str:
        """대화 히스토리 표시"""
        if not self.conversation_history:
            return "📝 대화 히스토리가 없습니다."
        
        history = "📝 **대화 히스토리:**\n\n"
        for i, msg in enumerate(self.conversation_history, 1):
            role_emoji = "🗣️" if msg["role"] == "user" else "🤖"
            content_preview = msg["content"][:100]
            if len(msg["content"]) > 100:
                content_preview += "..."
            
            history += f"{i}. {role_emoji} {msg['role']}: {content_preview}\n"
            history += f"   시간: {msg['timestamp']}\n\n"
        
        return history
    
    def _get_help_message(self) -> str:
        """도움말 메시지"""
        return """🤖 **코딩 도우미 챗봇 도움말**

**기본 기능:**
• 코드 작성 도움
• 버그 디버깅
• 코드 리뷰 및 개선 제안
• 프로그래밍 개념 설명
• 알고리즘 및 자료구조 도움
• 코드 최적화 제안

**지원 언어:**
Python, JavaScript, Java, C++, C, HTML, CSS, SQL, Bash, Go, Rust, PHP, Ruby, Swift, Kotlin

**특수 명령어:**
• `help` 또는 `도움말` - 이 도움말 표시
• `clear` 또는 `히스토리 삭제` - 대화 히스토리 삭제
• `history` 또는 `히스토리` - 대화 히스토리 보기
• `save: [설명]` - 최근 코드 블록들을 파일로 저장
• `debug: [에러메시지]` - 에러 분석 및 해결방법 제시

**사용 예시:**
• "Python으로 리스트를 정렬하는 방법을 알려줘"
• "이 JavaScript 코드에서 버그를 찾아줘"
• "SQL 조인에 대해 설명해줘"
• "React 컴포넌트 예제를 만들어줘"
• "알고리즘 시간복잡도를 계산해줘"

**팁:**
• 구체적인 질문일수록 더 정확한 답변을 받을 수 있습니다
• 에러 메시지와 함께 코드를 제공하면 더 나은 디버깅이 가능합니다
• 대화 컨텍스트가 유지되므로 연속된 질문이 가능합니다
• 코드 블록은 자동으로 파일로 저장할 수 있습니다
"""

def test_openai_connection(api_key: str) -> bool:
    """OpenAI API 연결 테스트"""
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print("✅ OpenAI API 연결 테스트 성공")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API 연결 테스트 실패: {e}")
        return False

async def main():
    """메인 함수"""
    print("🤖 코딩 도우미 챗봇")
    print("=" * 50)
    
    # OpenAI API 키 확인
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("💡 사용법:")
        print("   Windows: set OPENAI_API_KEY=your_api_key_here")
        print("   Linux/Mac: export OPENAI_API_KEY=your_api_key_here")
        
        openai_key = input("\nOpenAI API 키를 직접 입력하세요: ").strip()
        if not openai_key:
            print("❌ API 키가 필요합니다. 프로그램을 종료합니다.")
            return
    
    # OpenAI API 연결 테스트
    print("🔗 OpenAI API 연결 테스트 중...")
    if not test_openai_connection(openai_key):
        print("❌ OpenAI API 연결에 실패했습니다. API 키를 확인해주세요.")
        return
    
    try:
        # 챗봇 초기화
        chatbot = CodingAssistantBot(openai_key)
        
        # MCP 서버 초기화 시도 (선택사항)
        if MCP_AVAILABLE:
            print("🔧 MCP 서버 연결 시도 중...")
            await chatbot.initialize_mcp_server()
        
        print("\n🚀 코딩 도우미 챗봇이 준비되었습니다!")
        print("💬 코딩 관련 질문을 해주세요")
        print("📚 'help'를 입력하면 자세한 사용법을 볼 수 있습니다")
        print("🚪 종료하려면 'quit' 또는 'exit'를 입력하세요")
        print("-" * 50)
        
        # 샘플 질문 제안
        print("\n💡 시작하기 좋은 질문들:")
        sample_questions = [
            "Python으로 파일 읽기 예제를 보여줘",
            "JavaScript 비동기 처리 방법을 알려줘",
            "SQL 조인에 대해 설명해줘",
            "React 함수형 컴포넌트 예제를 만들어줘",
            "Python 리스트 컴프리헨션 사용법",
            "HTML CSS 레이아웃 예제",
            "버그가 있는 코드를 디버깅해줘"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            print(f"   {i}. {question}")
        
        # 대화 루프
        while True:
            try:
                user_input = input("\n🗣️  사용자: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료', '나가기']:
                    print("👋 코딩 도우미 챗봇을 종료합니다. 즐거운 코딩 되세요!")
                    break
                
                if not user_input:
                    continue
                
                # 빠른 선택 (숫자 입력)
                if user_input.isdigit():
                    num = int(user_input)
                    if 1 <= num <= len(sample_questions):
                        user_input = sample_questions[num - 1]
                        print(f"선택된 질문: {user_input}")
                    else:
                        print("❓ 잘못된 번호입니다. 다시 입력해주세요.")
                        continue
                
                # 챗봇 응답 생성
                print("\n🤖 AI가 분석 중...")
                response = await chatbot.chat(user_input)
                print(f"\n🤖 챗봇:\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 챗봇을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                print("💡 다시 시도해보세요.")
    
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
    
    finally:
        # MCP 세션 정리
        if hasattr(chatbot, 'mcp_session') and chatbot.mcp_session:
            try:
                await chatbot.mcp_session.__aexit__(None, None, None)
                print("🔧 MCP 세션이 정리되었습니다.")
            except:
                pass

# 환경변수 로드를 위한 선택적 python-dotenv 지원
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if __name__ == "__main__":
    asyncio.run(main())