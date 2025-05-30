#!/usr/bin/env python3
"""
MCP Claude Manager 설정 및 실행 스크립트
"""

import os
import sys
import subprocess
import json
from pathlib import Path



def check_requirements():
    """필요한 요구사항들을 확인합니다."""
    print("🔍 시스템 요구사항 확인 중...")
    
    requirements = {
        'python': {'cmd': [sys.executable, '--version'], 'min_version': '3.8'},
        'node': {'cmd': ['node', '--version'], 'min_version': '18.0'},
        'npx': {'cmd': ['npx', '--version'], 'required': True},
        'uv': {'cmd': ['uv', '--version'], 'required': False}
    }
    
    for name, req in requirements.items():
        try:
            result = subprocess.run(
                req['cmd'], 
                capture_output=True, 
                text=True, 
                shell=True  # Windows에서 PATH 인식 개선
            )
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()}")
            else:
                if req.get('required', True):
                    print(f"❌ {name}: 설치되지 않음 (에러 코드: {result.returncode}, 출력: {result.stderr})")
                    return False
                else:
                    print(f"⚠️  {name}: 설치되지 않음 (선택사항)")
        except FileNotFoundError as e:
            if req.get('required', True):
                print(f"❌ {name}: 설치되지 않음 (FileNotFoundError: {e})")
                return False
            else:
                print(f"⚠️  {name}: 설치되지 않음 (선택사항)")
    
    return True
    
def install_python_dependencies():
    """Python 의존성을 설치합니다."""
    print("\n📦 Python 의존성 설치 중...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'anthropic>=0.21.0'], check=True)
        print("✅ Python 의존성 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Python 의존성 설치 실패: {e}")
        return False

def create_env_file():
    """환경변수 파일을 생성합니다."""
    env_file = Path('.env')
    
    if env_file.exists():
        print("✅ .env 파일이 이미 존재합니다.")
        return True
    
    print("\n🔧 환경변수 설정...")
    
    # Claude API 키 입력
    claude_api_key = input("Claude API 키를 입력하세요 (Anthropic에서 발급): ").strip()
    if not claude_api_key:
        print("❌ Claude API 키가 필요합니다.")
        return False
    
    # 선택적 API 키들
    google_maps_key = input("Google Maps API 키 (선택사항, Enter로 건너뛰기): ").strip()
    brave_api_key = input("Brave Search API 키 (선택사항, Enter로 건너뛰기): ").strip()
    slack_bot_token = input("Slack Bot Token (선택사항, Enter로 건너뛰기): ").strip()
    slack_team_id = input("Slack Team ID (선택사항, Enter로 건너뛰기): ").strip()
    naver_client_id = input("Naver Client ID (선택사항, Enter로 건너뛰기): ").strip()
    naver_client_secret = input("Naver Client Secret (선택사항, Enter로 건너뛰기): ").strip()
    
    # .env 파일 생성
    env_content = f"""# Claude API 설정
ANTHROPIC_API_KEY={claude_api_key}

# Google Maps API (선택사항)
GOOGLE_MAPS_API_KEY={google_maps_key}

# Brave Search API (선택사항)
BRAVE_API_KEY={brave_api_key}

# Slack 설정 (선택사항)
SLACK_BOT_TOKEN={slack_bot_token}
SLACK_TEAM_ID={slack_team_id}

# Naver Open API (선택사항)
NAVER_CLIENT_ID={naver_client_id}
NAVER_CLIENT_SECRET={naver_client_secret}
"""
    
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("✅ .env 파일이 생성되었습니다.")
    return True

def create_directories():
    """필요한 디렉토리들을 생성합니다."""
    print("\n📁 디렉토리 구조 생성 중...")
    
    directories = [
        'data',
        'data/sqlite',
        'data/chroma',
        'data/files',
        'logs',
        'config'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ 디렉토리 생성: {dir_path}")
    
    return True

def test_mcp_servers():
    """MCP 서버들이 설치되어 있는지 테스트합니다."""
    print("\n🧪 MCP 서버 설치 상태 확인 중...")
    
    test_servers = [
        {
            'name': 'filesystem',
            'cmd': ['npx', '-y', '@modelcontextprotocol/server-filesystem', '--help']
        },
        {
            'name': 'memory',
            'cmd': ['npx', '-y', '@modelcontextprotocol/server-memory', '--help']
        },
        {
            'name': 'brave-search',
            'cmd': ['npx', '-y', '@modelcontextprotocol/server-brave-search', '--help']
        }
    ]
    
    available_servers = []
    
    for server in test_servers:
        try:
            print(f"  테스트 중: {server['name']}...", end='')
            result = subprocess.run(
                server['cmd'], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            if result.returncode == 0 or 'help' in result.stdout.lower() or 'usage' in result.stdout.lower():
                print(" ✅")
                available_servers.append(server['name'])
            else:
                print(" ❌")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            print(" ❌")
    
    print(f"\n사용 가능한 MCP 서버: {len(available_servers)}개")
    return available_servers

def create_sample_config():
    """샘플 설정 파일을 생성합니다."""
    print("\n⚙️  설정 파일 생성 중...")
    
    # 현재 사용자의 Documents 경로 감지
    documents_path = str(Path.home() / "Documents")
    
    config = {
        "mcpServers": {
            "filesystem": {
                "name": "파일시스템 관리",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", documents_path],
                "enabled": True
            },
            "memory": {
                "name": "메모리/지식 그래프",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-memory"],
                "enabled": True
            },
            "brave-search": {
                "name": "브레이브 웹 검색",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
                "enabled": False
            },
            "google-maps": {
                "name": "구글 맵스 API",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-google-maps"],
                "env": {"GOOGLE_MAPS_API_KEY": "${GOOGLE_MAPS_API_KEY}"},
                "enabled": False
            },
            "slack": {
                "name": "슬랙 통합",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-slack"],
                "env": {
                    "SLACK_BOT_TOKEN": "${SLACK_BOT_TOKEN}",
                    "SLACK_TEAM_ID": "${SLACK_TEAM_ID}"
                },
                "enabled": False
            }
        }
    }
    
    with open('config/mcp_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ 설정 파일 생성 완료: config/mcp_config.json")
    return True

def main():
    """메인 설정 함수"""
    print("🚀 MCP Claude Manager 설정을 시작합니다!\n")
    
    # 1. 시스템 요구사항 확인
    if not check_requirements():
        print("\n❌ 시스템 요구사항을 만족하지 않습니다. 필요한 소프트웨어를 설치해주세요.")
        sys.exit(1)
    
    # 2. Python 의존성 설치
    if not install_python_dependencies():
        print("\n❌ Python 의존성 설치에 실패했습니다.")
        sys.exit(1)
    
    # 3. 디렉토리 생성
    create_directories()
    
    # 4. 환경변수 설정
    if not create_env_file():
        print("\n❌ 환경변수 설정에 실패했습니다.")
        sys.exit(1)
    
    # 5. 설정 파일 생성
    create_sample_config()
    
    # 6. MCP 서버 테스트 (선택사항)
    print("\nMCP 서버 설치 상태를 확인하시겠습니까? (y/n): ", end='')
    if input().lower().startswith('y'):
        available_servers = test_mcp_servers()
        print(f"✅ {len(available_servers)}개의 MCP 서버가 사용 가능합니다.")
    
    print("\n🎉 설정이 완료되었습니다!")
    print("\n다음 명령어로 MCP Claude Manager를 실행하세요:")
    print("python mcp_claude_manager.py")
    
    print("\n📖 추가 정보:")
    print("- 설정 파일: config/mcp_config.json")
    print("- 환경변수: .env")
    print("- 로그: logs/ 디렉토리")
    print("- 데이터: data/ 디렉토리")

if __name__ == "__main__":
    main()