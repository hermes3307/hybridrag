#!/usr/bin/env python3
"""
Claude Desktop Config MCP 서버 실행기
현재 디렉토리의 claude_desktop_config.json을 읽어서 모든 MCP 서버를 실행합니다.
"""

import os
import sys
import json
import subprocess
import asyncio
import signal
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# 필요한 패키지 설치
try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv 설치 중...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

# OpenAI 클라이언트 임포트
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI 패키지 설치 중...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai>=1.0.0"])
    from openai import OpenAI
    OPENAI_AVAILABLE = True

# MCP 클라이언트 임포트 (선택사항)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("⚠️  MCP 패키지가 없습니다. 기본 기능만 사용됩니다.")
    MCP_AVAILABLE = False

@dataclass
class MCPServerConfig:
    """MCP 서버 설정"""
    name: str
    command: str
    args: List[str]
    env: Dict[str, str] = None
    cwd: str = None
    enabled: bool = True
    process: subprocess.Popen = None
    
class MCPServerManager:
    """MCP 서버 관리자"""
    
    def __init__(self, config_path: str = "claude_desktop_config.json"):
        self.config_path = Path(config_path)
        self.servers: Dict[str, MCPServerConfig] = {}
        self.running_servers: Dict[str, subprocess.Popen] = {}
        self.shutdown_event = threading.Event()
        
        # 환경변수 로드
        load_dotenv()
        
        print(f"🔧 MCP 서버 관리자 초기화")
        print(f"📄 설정 파일: {self.config_path}")
        
    def load_config(self) -> bool:
        """Claude Desktop Config 파일을 로드합니다."""
        try:
            if not self.config_path.exists():
                print(f"❌ 설정 파일을 찾을 수 없습니다: {self.config_path}")
                return False
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # mcpServers 섹션 파싱
            mcp_servers = config.get('mcpServers', {})
            
            if not mcp_servers:
                print("⚠️  mcpServers 설정이 없습니다.")
                return False
            
            # 서버 설정 파싱
            for server_name, server_config in mcp_servers.items():
                command = server_config.get('command', '')
                args = server_config.get('args', [])
                env = server_config.get('env', {})
                cwd = server_config.get('cwd')
                
                # 환경변수 치환
                resolved_env = {}
                for key, value in env.items():
                    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                        env_var = value[2:-1]
                        resolved_env[key] = os.getenv(env_var, '')
                    else:
                        resolved_env[key] = value
                
                self.servers[server_name] = MCPServerConfig(
                    name=server_name,
                    command=command,
                    args=args,
                    env=resolved_env,
                    cwd=cwd,
                    enabled=True
                )
            
            print(f"✅ {len(self.servers)}개의 MCP 서버 설정을 로드했습니다.")
            return True
            
        except Exception as e:
            print(f"❌ 설정 파일 로드 실패: {e}")
            return False
    
    def validate_server_config(self, server: MCPServerConfig) -> bool:
        """서버 설정 유효성 검사"""
        try:
            # 명령어 존재 확인
            if server.command in ['npx', 'uvx', 'node', 'python', 'python3']:
                # 일반적인 명령어들은 통과
                pass
            elif Path(server.command).exists():
                # 절대 경로 확인
                pass
            else:
                # which 명령으로 확인
                result = subprocess.run(['which', server.command], 
                                     capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"⚠️  명령어를 찾을 수 없습니다: {server.command}")
                    return False
            
            # 작업 디렉토리 확인
            if server.cwd and not Path(server.cwd).exists():
                print(f"⚠️  작업 디렉토리가 존재하지 않습니다: {server.cwd}")
                return False
            
            # 필수 환경변수 확인
            for key, value in server.env.items():
                if not value:
                    print(f"⚠️  환경변수가 설정되지 않았습니다: {key}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"⚠️  서버 설정 검증 실패 ({server.name}): {e}")
            return False
    
    def start_server(self, server_name: str) -> bool:
        """특정 서버를 시작합니다."""
        if server_name not in self.servers:
            print(f"❌ 서버를 찾을 수 없습니다: {server_name}")
            return False
        
        server = self.servers[server_name]
        
        if not self.validate_server_config(server):
            print(f"❌ 서버 설정이 유효하지 않습니다: {server_name}")
            return False
        
        try:
            print(f"🚀 서버 시작 중: {server_name}")
            print(f"   명령어: {server.command} {' '.join(server.args)}")
            
            # 환경변수 설정
            env = os.environ.copy()
            env.update(server.env)
            
            # 서버 프로세스 시작
            process = subprocess.Popen(
                [server.command] + server.args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                env=env,
                cwd=server.cwd,
                text=True,
                bufsize=0
            )
            
            # 프로세스가 정상적으로 시작되었는지 확인
            time.sleep(1)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"❌ 서버 시작 실패: {server_name}")
                print(f"   stdout: {stdout}")
                print(f"   stderr: {stderr}")
                return False
            
            self.running_servers[server_name] = process
            server.process = process
            
            print(f"✅ 서버 시작 완료: {server_name} (PID: {process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ 서버 시작 실패 ({server_name}): {e}")
            return False
    
    def stop_server(self, server_name: str) -> bool:
        """특정 서버를 중지합니다."""
        if server_name not in self.running_servers:
            print(f"⚠️  실행 중인 서버를 찾을 수 없습니다: {server_name}")
            return False
        
        try:
            process = self.running_servers[server_name]
            print(f"🛑 서버 중지 중: {server_name} (PID: {process.pid})")
            
            # 정상 종료 시도
            process.terminate()
            
            # 5초 대기
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 강제 종료
                print(f"⚠️  강제 종료: {server_name}")
                process.kill()
                process.wait()
            
            del self.running_servers[server_name]
            if server_name in self.servers:
                self.servers[server_name].process = None
            
            print(f"✅ 서버 중지 완료: {server_name}")
            return True
            
        except Exception as e:
            print(f"❌ 서버 중지 실패 ({server_name}): {e}")
            return False
    
    def start_all_servers(self) -> int:
        """모든 서버를 시작합니다."""
        print(f"\n🚀 모든 MCP 서버 시작 중... ({len(self.servers)}개)")
        
        started_count = 0
        for server_name in self.servers:
            if self.start_server(server_name):
                started_count += 1
                time.sleep(0.5)  # 서버 간 시작 간격
        
        print(f"\n✅ {started_count}/{len(self.servers)}개 서버가 시작되었습니다.")
        return started_count
    
    def stop_all_servers(self):
        """모든 서버를 중지합니다."""
        print(f"\n🛑 모든 MCP 서버 중지 중...")
        
        for server_name in list(self.running_servers.keys()):
            self.stop_server(server_name)
        
        print("✅ 모든 서버가 중지되었습니다.")
    
    def show_server_status(self):
        """서버 상태를 표시합니다."""
        print(f"\n📊 MCP 서버 상태 ({len(self.servers)}개 서버)")
        print("=" * 60)
        
        for server_name, server in self.servers.items():
            status = "🟢 실행중" if server_name in self.running_servers else "🔴 중지됨"
            pid = f"(PID: {self.running_servers[server_name].pid})" if server_name in self.running_servers else ""
            
            print(f"{status} {server_name} {pid}")
            print(f"   명령어: {server.command} {' '.join(server.args[:3])}{'...' if len(server.args) > 3 else ''}")
            
            if server.env:
                env_keys = list(server.env.keys())[:2]
                env_str = ", ".join(env_keys)
                if len(server.env) > 2:
                    env_str += f", ... (+{len(server.env)-2}개)"
                print(f"   환경변수: {env_str}")
            
            print()
    
    def monitor_servers(self):
        """서버 모니터링"""
        print("🔍 서버 모니터링 시작...")
        
        while not self.shutdown_event.is_set():
            try:
                # 죽은 프로세스 확인
                dead_servers = []
                for server_name, process in self.running_servers.items():
                    if process.poll() is not None:
                        dead_servers.append(server_name)
                
                # 죽은 서버 정리
                for server_name in dead_servers:
                    print(f"💀 서버가 예상치 못하게 종료되었습니다: {server_name}")
                    del self.running_servers[server_name]
                    if server_name in self.servers:
                        self.servers[server_name].process = None
                
                time.sleep(5)  # 5초마다 확인
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️  모니터링 오류: {e}")
                time.sleep(5)
    
    def setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            print(f"\n📢 종료 신호 수신 ({signum})")
            self.shutdown_event.set()
            self.stop_all_servers()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

class CodingAssistantWithMCP:
    """MCP 서버와 연동된 코딩 어시스턴트"""
    
    def __init__(self, openai_api_key: str, mcp_manager: MCPServerManager):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.mcp_manager = mcp_manager
        self.conversation_history = []
    
    def add_to_conversation(self, role: str, content: str):
        """대화 히스토리에 메시지 추가"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 히스토리가 너무 길어지면 오래된 것들 제거
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    async def get_coding_help(self, user_input: str) -> str:
        """OpenAI와 MCP 서버를 활용한 코딩 도움말"""
        try:
            # 사용자 입력을 대화 히스토리에 추가
            self.add_to_conversation("user", user_input)
            
            # 시스템 프롬프트에 MCP 서버 정보 추가
            available_servers = list(self.mcp_manager.running_servers.keys())
            
            system_prompt = f"""You are an expert coding assistant with access to various MCP (Model Context Protocol) servers.

Available MCP servers: {', '.join(available_servers)}

You can help with:
- Code writing and debugging
- File system operations (if filesystem server is available)
- Web searches (if brave-search server is available)
- Database operations (if sqlite/postgres servers are available)
- GitHub operations (if github server is available)
- Memory/knowledge management (if memory server is available)
- And more based on available servers

Always provide practical, working code examples and clear explanations.
"""
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # 최근 대화 히스토리 추가
            for msg in self.conversation_history[-10:]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # OpenAI API 호출
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
            
            assistant_response = response.choices[0].message.content.strip()
            self.add_to_conversation("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"코딩 도움말 생성 실패: {str(e)}"
            self.add_to_conversation("assistant", error_msg)
            return error_msg
    
    async def chat(self, user_input: str) -> str:
        """메인 채팅 함수"""
        if not user_input.strip():
            return "❓ 코딩과 관련된 질문을 해주세요!"
        
        # 특수 명령어 처리
        if user_input.lower() in ['servers', '서버상태']:
            self.mcp_manager.show_server_status()
            return "📊 서버 상태가 출력되었습니다."
        
        if user_input.lower() == 'restart_servers':
            self.mcp_manager.stop_all_servers()
            time.sleep(2)
            started = self.mcp_manager.start_all_servers()
            return f"🔄 서버 재시작 완료: {started}개 서버"
        
        # 일반적인 코딩 질문 처리
        response = await self.get_coding_help(user_input)
        return response

def interactive_mode(manager: MCPServerManager):
    """대화형 모드"""
    print("\n🤖 MCP 코딩 어시스턴트 모드")
    print("대화형 모드를 사용하려면 OpenAI API 키가 필요합니다.")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        openai_key = input("OpenAI API 키를 입력하세요: ").strip()
        if not openai_key:
            print("❌ API 키가 필요합니다.")
            return
    
    try:
        assistant = CodingAssistantWithMCP(openai_key, manager)
        print("\n💬 코딩 질문을 해주세요! ('quit'로 종료)")
        print("특수 명령어: 'servers' (서버상태), 'restart_servers' (서버재시작)")
        
        while True:
            try:
                user_input = input("\n🗣️  사용자: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료']:
                    break
                
                if not user_input:
                    continue
                
                response = asyncio.run(assistant.chat(user_input))
                print(f"\n🤖 어시스턴트:\n{response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 오류: {e}")
    
    except Exception as e:
        print(f"❌ 어시스턴트 초기화 실패: {e}")

def main():
    """메인 함수"""
    print("🚀 Claude Desktop MCP 서버 관리자")
    print("=" * 50)
    
    # 설정 파일 경로 확인
    config_files = [
        "claude_desktop_config.json",
        "config/claude_desktop_config.json",
        str(Path.home() / ".config" / "claude" / "claude_desktop_config.json"),
        str(Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json")
    ]
    
    config_path = None
    for path in config_files:
        if Path(path).exists():
            config_path = path
            break
    
    if not config_path:
        print("❌ claude_desktop_config.json 파일을 찾을 수 없습니다.")
        print("다음 위치 중 하나에 파일이 있어야 합니다:")
        for path in config_files:
            print(f"  - {path}")
        return
    
    # MCP 서버 관리자 초기화
    manager = MCPServerManager(config_path)
    
    if not manager.load_config():
        print("❌ 설정 로드 실패")
        return
    
    # 시그널 핸들러 설정
    manager.setup_signal_handlers()
    
    # 사용자 선택
    print("\n선택하세요:")
    print("1. 모든 MCP 서버 시작")
    print("2. 서버 상태 확인")
    print("3. 대화형 코딩 어시스턴트 모드")
    print("4. 종료")
    
    try:
        choice = input("\n선택 (1-4): ").strip()
        
        if choice == "1":
            started = manager.start_all_servers()
            if started > 0:
                print(f"\n🎉 {started}개 서버가 시작되었습니다!")
                print("서버들이 백그라운드에서 실행 중입니다.")
                print("종료하려면 Ctrl+C를 누르세요.")
                
                # 모니터링 스레드 시작
                monitor_thread = threading.Thread(target=manager.monitor_servers, daemon=True)
                monitor_thread.start()
                
                # 메인 스레드에서 대기
                try:
                    while not manager.shutdown_event.is_set():
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
                
        elif choice == "2":
            manager.show_server_status()
            
        elif choice == "3":
            started = manager.start_all_servers()
            if started > 0:
                # 모니터링 스레드 시작
                monitor_thread = threading.Thread(target=manager.monitor_servers, daemon=True)
                monitor_thread.start()
                
                # 대화형 모드 시작
                interactive_mode(manager)
            else:
                print("❌ 서버 시작이 필요합니다.")
                
        elif choice == "4":
            print("👋 종료합니다.")
            return
        else:
            print("❌ 잘못된 선택입니다.")
    
    except KeyboardInterrupt:
        print("\n📢 사용자에 의해 중단되었습니다.")
    
    finally:
        manager.stop_all_servers()

if __name__ == "__main__":
    main()