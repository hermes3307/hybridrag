#!/usr/bin/env python3
"""
Enhanced MCP Claude Manager
다양한 MCP 서버를 관리하고 Claude API와 통합하는 고급 챗봇
"""

import asyncio
import json
import subprocess
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import anthropic
from datetime import datetime
import signal
import time
import re
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
def setup_logging():
    """로깅을 설정합니다."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"mcp_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

@dataclass
class MCPTool:
    """MCP 도구 정보"""
    name: str
    description: str = ""
    input_schema: Dict = field(default_factory=dict)
    server_name: str = ""

@dataclass
class MCPServer:
    """MCP 서버 정보"""
    name: str
    display_name: str
    command: str
    args: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    process: Optional[subprocess.Popen] = None
    tools: List[MCPTool] = field(default_factory=list)
    capabilities: Dict = field(default_factory=dict)
    last_heartbeat: Optional[datetime] = None
    restart_count: int = 0

class MCPServerManager:
    """향상된 MCP 서버 매니저"""
    
    def __init__(self, claude_api_key: str, config_file: str = "config/mcp_config.json"):
        self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
        self.servers: Dict[str, MCPServer] = {}
        self.config_file = config_file
        self.running = False
        self.request_id = 0
        
        # 신호 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """신호 핸들러"""
        logger.info(f"신호 {signum} 수신됨. 정리 중...")
        self.running = False

    def _expand_env_vars(self, text: str) -> str:
        """환경변수를 확장합니다."""
        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        return re.sub(r'\$\{([^}]+)\}', replace_var, text)

    def load_config(self) -> bool:
        """설정 파일을 로드합니다."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            for name, server_config in config.get("mcpServers", {}).items():
                # 환경변수 확장
                env_vars = {}
                for key, value in server_config.get("env", {}).items():
                    env_vars[key] = self._expand_env_vars(value)
                
                # 필수 환경변수가 설정되어 있는지 확인
                if env_vars and not all(val and val != f"${{{key}}}" for key, val in env_vars.items()):
                    logger.warning(f"서버 {name}: 일부 환경변수가 설정되지 않음. 비활성화됨.")
                    server_config["enabled"] = False
                
                self.servers[name] = MCPServer(
                    name=name,
                    display_name=server_config.get("name", name),
                    command=server_config["command"],
                    args=server_config["args"],
                    env=env_vars,
                    enabled=server_config.get("enabled", True)
                )
                
                logger.info(f"서버 로드됨: {name} ({'활성' if self.servers[name].enabled else '비활성'})")
            
            return True
            
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return False

    def _get_next_request_id(self) -> int:
        """다음 요청 ID를 반환합니다."""
        self.request_id += 1
        return self.request_id

    async def _send_jsonrpc_request(self, server: MCPServer, method: str, params: Dict = None) -> Dict:
        """JSON-RPC 요청을 전송합니다."""
        if not server.process:
            raise Exception(f"서버 {server.name}이 실행중이지 않습니다")
        
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": method,
            "params": params or {}
        }
        
        try:
            # 요청 전송
            request_json = json.dumps(request) + "\n"
            server.process.stdin.write(request_json)
            server.process.stdin.flush()
            
            # 응답 대기 (타임아웃 설정)
            start_time = time.time()
            response_line = None
            
            while time.time() - start_time < 10:  # 10초 타임아웃
                if server.process.stdout.readable():
                    response_line = server.process.stdout.readline()
                    if response_line:
                        break
                await asyncio.sleep(0.1)
            
            if not response_line:
                raise Exception("서버로부터 응답 타임아웃")
            
            response = json.loads(response_line.strip())
            
            # 에러 확인
            if "error" in response:
                raise Exception(f"서버 에러: {response['error']}")
            
            return response
            
        except Exception as e:
            logger.error(f"JSON-RPC 요청 실패 {server.name}/{method}: {e}")
            raise

    async def start_server(self, server_name: str) -> bool:
        """서버를 시작합니다."""
        if server_name not in self.servers:
            logger.error(f"서버를 찾을 수 없습니다: {server_name}")
            return False
        
        server = self.servers[server_name]
        
        if not server.enabled:
            logger.info(f"서버 {server_name}이 비활성화되어 있습니다")
            return False
        
        try:
            # 환경변수 설정
            env = os.environ.copy()
            if server.env:
                env.update(server.env)
            
            logger.info(f"서버 시작 중: {server_name}")
            
            # 서버 프로세스 시작
            process = subprocess.Popen(
                [server.command] + server.args,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            server.process = process
            
            # 서버 초기화 대기
            await asyncio.sleep(3)
            
            # 프로세스가 살아있는지 확인
            if process.poll() is not None:
                stderr_output = process.stderr.read()
                logger.error(f"서버 {server_name} 시작 실패: {stderr_output}")
                return False
            
            # 서버 초기화
            try:
                await self._initialize_server(server)
                await self._load_server_tools(server)
                server.last_heartbeat = datetime.now()
                logger.info(f"서버 {server_name} 시작 완료 ({len(server.tools)}개 도구)")
                return True
                
            except Exception as e:
                logger.error(f"서버 {server_name} 초기화 실패: {e}")
                self._stop_server_process(server)
                return False
            
        except Exception as e:
            logger.error(f"서버 {server_name} 시작 중 오류: {e}")
            return False

    async def _initialize_server(self, server: MCPServer):
        """서버를 초기화합니다."""
        init_request = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "enhanced-mcp-manager",
                "version": "1.0.0"
            }
        }
        
        response = await self._send_jsonrpc_request(server, "initialize", init_request)
        server.capabilities = response.get("result", {})
        
        # initialized 알림 전송
        await self._send_jsonrpc_request(server, "notifications/initialized")

    async def _load_server_tools(self, server: MCPServer):
        """서버의 도구 목록을 로드합니다."""
        try:
            response = await self._send_jsonrpc_request(server, "tools/list")
            tools_data = response.get("result", {}).get("tools", [])
            
            server.tools = []
            for tool_data in tools_data:
                tool = MCPTool(
                    name=tool_data.get("name", ""),
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    server_name=server.name
                )
                server.tools.append(tool)
            
        except Exception as e:
            logger.error(f"도구 목록 로드 실패 {server.name}: {e}")

    def _stop_server_process(self, server: MCPServer):
        """서버 프로세스를 중지합니다."""
        if server.process:
            try:
                server.process.terminate()
                server.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.process.kill()
                server.process.wait()
            except Exception as e:
                logger.error(f"서버 {server.name} 종료 중 오류: {e}")
            finally:
                server.process = None

    def stop_server(self, server_name: str):
        """서버를 중지합니다."""
        if server_name in self.servers:
            server = self.servers[server_name]
            self._stop_server_process(server)
            logger.info(f"서버 중지됨: {server_name}")

    def stop_all_servers(self):
        """모든 서버를 중지합니다."""
        for server_name in list(self.servers.keys()):
            self.stop_server(server_name)

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict = None) -> Dict:
        """도구를 호출합니다."""
        if server_name not in self.servers:
            raise Exception(f"서버를 찾을 수 없습니다: {server_name}")
        
        server = self.servers[server_name]
        
        try:
            response = await self._send_jsonrpc_request(server, "tools/call", {
                "name": tool_name,
                "arguments": arguments or {}
            })
            
            result = response.get("result", {})
            server.last_heartbeat = datetime.now()
            return result
            
        except Exception as e:
            logger.error(f"도구 호출 실패 {server_name}/{tool_name}: {e}")
            raise

    async def query_claude_with_context(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        """컨텍스트가 있는 Claude 쿼리를 실행합니다."""
        
        # 사용 가능한 도구들 수집
        available_tools = []
        server_info = []
        
        for server in self.servers.values():
            if server.process and server.tools:
                server_info.append({
                    "name": server.name,
                    "display_name": server.display_name,
                    "tool_count": len(server.tools)
                })
                
                for tool in server.tools:
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description,
                        "server": server.name,
                        "server_display": server.display_name
                    }
                    available_tools.append(tool_info)

        # 시스템 메시지 구성
        system_message = f"""
당신은 다양한 MCP (Model Context Protocol) 도구들을 활용할 수 있는 고급 AI 어시스턴트입니다.

🛠️ 현재 실행 중인 서버들 ({len(server_info)}개):
{json.dumps(server_info, indent=2, ensure_ascii=False)}

🔧 사용 가능한 도구들 ({len(available_tools)}개):
{json.dumps(available_tools, indent=2, ensure_ascii=False)}

사용자의 요청을 분석하여 가장 적절한 도구들을 선택하고 활용해주세요.
도구를 사용할 때는 다음 원칙을 따르세요:

1. 사용자의 의도를 정확히 파악하세요
2. 가장 적합한 도구를 선택하세요
3. 도구 사용 과정을 사용자에게 설명하세요
4. 결과를 명확하고 유용한 형태로 제시하세요
5. 필요시 여러 도구를 조합하여 사용하세요

한국어로 자연스럽고 도움이 되는 답변을 제공해주세요.
"""

        # 대화 기록 포함
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({
            "role": "user",
            "content": user_message
        })

        try:
            response = self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                system=system_message,
                messages=messages
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API 호출 실패: {e}")
            return f"죄송합니다. Claude API 호출 중 오류가 발생했습니다: {e}"

    async def health_check(self):
        """서버들의 상태를 확인합니다."""
        for server in self.servers.values():
            if server.process:
                # 프로세스가 살아있는지 확인
                if server.process.poll() is not None:
                    logger.warning(f"서버 {server.name} 프로세스가 종료됨")
                    server.process = None
                    continue
                
                # 마지막 하트비트 확인
                if server.last_heartbeat:
                    time_since_heartbeat = datetime.now() - server.last_heartbeat
                    if time_since_heartbeat.total_seconds() > 300:  # 5분
                        logger.warning(f"서버 {server.name} 응답 없음 (마지막: {server.last_heartbeat})")

    async def restart_failed_servers(self):
        """실패한 서버들을 재시작합니다."""
        for server in self.servers.values():
            if server.enabled and not server.process and server.restart_count < 3:
                logger.info(f"서버 {server.name} 재시작 시도 ({server.restart_count + 1}/3)")
                success = await self.start_server(server.name)
                if success:
                    server.restart_count = 0
                else:
                    server.restart_count += 1

    def get_status_summary(self) -> Dict:
        """상태 요약을 반환합니다."""
        summary = {
            "total_servers": len(self.servers),
            "running_servers": 0,
            "enabled_servers": 0,
            "total_tools": 0,
            "servers": {}
        }
        
        for name, server in self.servers.items():
            is_running = server.process and server.process.poll() is None
            
            if server.enabled:
                summary["enabled_servers"] += 1
            if is_running:
                summary["running_servers"] += 1
                summary["total_tools"] += len(server.tools)
            
            summary["servers"][name] = {
                "display_name": server.display_name,
                "enabled": server.enabled,
                "running": is_running,
                "tools_count": len(server.tools),
                "restart_count": server.restart_count
            }
        
        return summary

    def print_detailed_status(self):
        """상세 상태를 출력합니다."""
        summary = self.get_status_summary()
        
        print("\n" + "="*80)
        print("🤖 MCP Claude Manager 상태")
        print("="*80)
        print(f"📊 전체 서버: {summary['total_servers']}개 | "
              f"활성화: {summary['enabled_servers']}개 | "
              f"실행중: {summary['running_servers']}개 | "
              f"도구: {summary['total_tools']}개")
        print("="*80)
        
        for name, info in summary["servers"].items():
            status_icon = "🟢" if info["running"] else "🔴" if info["enabled"] else "⚪"
            status_text = "실행중" if info["running"] else "중지됨" if info["enabled"] else "비활성"
            
            print(f"{status_icon} {info['display_name']} ({name})")
            print(f"   상태: {status_text} | 도구: {info['tools_count']}개", end="")
            if info["restart_count"] > 0:
                print(f" | 재시작: {info['restart_count']}회", end="")
            print()
            
            if info["running"] and name in self.servers:
                server = self.servers[name]
                for tool in server.tools[:3]:  # 처음 3개만 표시
                    print(f"     🔧 {tool.name}: {tool.description[:50]}...")
                if len(server.tools) > 3:
                    print(f"     ... 및 {len(server.tools) - 3}개 더")
        
        print("="*80)

    async def initialize_all_servers(self):
        """모든 활성화된 서버들을 초기화합니다."""
        logger.info("MCP 서버들 초기화 시작...")
        
        tasks = []
        for server_name, server in self.servers.items():
            if server.enabled:
                tasks.append(self.start_server(server_name))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result is True)
            logger.info(f"서버 초기화 완료: {success_count}/{len(tasks)}개 성공")
        else:
            logger.warning("활성화된 서버가 없습니다")

    async def run_interactive_mode(self):
        """대화형 모드를 실행합니다."""
        self.running = True
        conversation_history = []
        
        print("\n🤖 Enhanced MCP Claude Manager가 준비되었습니다!")
        print("📝 명령어:")
        print("  - 일반 질문: 자유롭게 입력하세요")
        print("  - /status: 서버 상태 확인")
        print("  - /restart [서버명]: 특정 서버 재시작")
        print("  - /tools [서버명]: 서버의 도구 목록 보기")
        print("  - /clear: 대화 기록 초기화")
        print("  - /quit: 종료")
        print("\n질문을 입력하세요:")
        
        # 백그라운드 모니터링 태스크 시작
        monitor_task = asyncio.create_task(self._background_monitor())
        
        try:
            while self.running:
                try:
                    user_input = input("\n사용자: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # 명령어 처리
                    if user_input.startswith('/'):
                        await self._handle_command(user_input)
                        continue
                    
                    if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                        break
                    
                    # Claude 쿼리 실행
                    print("\n🤔 Claude: 답변을 생성중입니다...")
                    
                    response = await self.query_claude_with_context(user_input, conversation_history)
                    print(f"\n🤖 Claude: {response}")
                    
                    # 대화 기록 업데이트 (최근 10개만 유지)
                    conversation_history.append({"role": "user", "content": user_input})
                    conversation_history.append({"role": "assistant", "content": response})
                    if len(conversation_history) > 20:
                        conversation_history = conversation_history[-20:]
                    
                except KeyboardInterrupt:
                    print("\n\n종료 신호를 받았습니다.")
                    break
                except EOFError:
                    print("\n\n입력 스트림이 종료되었습니다.")
                    break
                except Exception as e:
                    logger.error(f"대화 처리 중 오류: {e}")
                    print(f"오류가 발생했습니다: {e}")
        
        finally:
            self.running = False
            monitor_task.cancel()

    async def _handle_command(self, command: str):
        """명령어를 처리합니다."""
        parts = command[1:].split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "status":
            self.print_detailed_status()
        
        elif cmd == "restart":
            if args:
                server_name = args[0]
                if server_name in self.servers:
                    print(f"🔄 서버 {server_name} 재시작 중...")
                    self.stop_server(server_name)
                    await asyncio.sleep(2)
                    success = await self.start_server(server_name)
                    if success:
                        print(f"✅ 서버 {server_name} 재시작 완료")
                    else:
                        print(f"❌ 서버 {server_name} 재시작 실패")
                else:
                    print(f"❌ 서버를 찾을 수 없습니다: {server_name}")
            else:
                print("사용법: /restart <서버명>")
        
        elif cmd == "tools":
            if args:
                server_name = args[0]
                if server_name in self.servers:
                    server = self.servers[server_name]
                    if server.tools:
                        print(f"\n🔧 {server.display_name} 도구 목록:")
                        for i, tool in enumerate(server.tools, 1):
                            print(f"  {i}. {tool.name}: {tool.description}")
                    else:
                        print(f"서버 {server_name}에 사용 가능한 도구가 없습니다.")
                else:
                    print(f"❌ 서버를 찾을 수 없습니다: {server_name}")
            else:
                print("📋 전체 도구 목록:")
                for server_name, server in self.servers.items():
                    if server.tools and server.process:
                        print(f"\n🔧 {server.display_name} ({len(server.tools)}개):")
                        for tool in server.tools:
                            print(f"  - {tool.name}: {tool.description}")
        
        elif cmd == "clear":
            print("🗑️ 대화 기록이 초기화되었습니다.")
        
        elif cmd == "help":
            print("""
📖 사용 가능한 명령어:
  /status              - 서버 상태 확인
  /restart <서버명>    - 특정 서버 재시작
  /tools [서버명]      - 도구 목록 보기
  /clear               - 대화 기록 초기화
  /help                - 이 도움말 보기
  /quit                - 프로그램 종료
""")
        
        elif cmd == "quit":
            self.running = False
        
        else:
            print(f"❌ 알 수 없는 명령어: {cmd}. /help로 도움말을 확인하세요.")

    async def _background_monitor(self):
        """백그라운드 모니터링 태스크"""
        while self.running:
            try:
                await asyncio.sleep(30)  # 30초마다 체크
                await self.health_check()
                await self.restart_failed_servers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"백그라운드 모니터링 오류: {e}")


async def main():
    """메인 실행 함수"""
    print("🚀 Enhanced MCP Claude Manager 시작...")
    
    # Claude API 키 확인
    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not claude_api_key:
        print("❌ ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        print("💡 .env 파일에 API 키를 설정하거나 환경변수로 설정해주세요.")
        claude_api_key = input("Claude API 키를 직접 입력하세요: ").strip()
        if not claude_api_key:
            print("❌ API 키가 필요합니다. 프로그램을 종료합니다.")
            return
    
    # 설정 파일 확인
    config_file = "config/mcp_config.json"
    if not Path(config_file).exists():
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_file}")
        print("💡 setup.py를 먼저 실행하여 초기 설정을 완료해주세요.")
        return
    
    # MCP 매니저 초기화
    manager = MCPServerManager(claude_api_key, config_file)
    
    # 설정 로드
    if not manager.load_config():
        print("❌ 설정 파일 로드에 실패했습니다.")
        return
    
    try:
        # 모든 서버 초기화
        await manager.initialize_all_servers()
        
        # 초기 상태 표시
        manager.print_detailed_status()
        
        # 대화형 모드 시작
        await manager.run_interactive_mode()
    
    except KeyboardInterrupt:
        print("\n🛑 프로그램이 중단되었습니다.")
    
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
    
    finally:
        # 정리 작업
        print("\n🧹 리소스 정리 중...")
        manager.stop_all_servers()
        print("✅ 모든 MCP 서버가 종료되었습니다.")
        print("👋 Enhanced MCP Claude Manager를 종료합니다.")


def check_dependencies():
    """의존성을 확인합니다."""
    try:
        import anthropic
        import dotenv
        return True
    except ImportError as e:
        print(f"❌ 필요한 패키지가 설치되지 않았습니다: {e}")
        print("💡 다음 명령어로 설치하세요:")
        print("pip install anthropic python-dotenv")
        return False


if __name__ == "__main__":
    # 의존성 확인
    if not check_dependencies():
        sys.exit(1)
    
    # 메인 프로그램 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류 발생: {e}")
        sys.exit(1)