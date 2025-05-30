import asyncio
import json
import base64
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import sys
import os

# MCP 클라이언트 임포트
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("MCP not available, will use alternative methods")
    MCP_AVAILABLE = False

# HTTP 클라이언트 임포트
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    print("httpx not available, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
    import httpx
    HTTPX_AVAILABLE = True

async def select_mcp_server():
    """Display menu of available MCP servers and return selected server config."""
    mcp_servers = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "C://Users//A//Documents"]
        },
        "mcp-server-fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"]
        },
        "brave-search": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": "BSAjXdc9huO9V6coQeA29cT9SbRLLqN"}
        },
        "direct-download": {
            "description": "Direct HTTP download (no MCP)",
            "method": "direct"
        }
    }
    
    print("\nAvailable Download Methods:")
    print("="*50)
    for i, (server_name, config) in enumerate(mcp_servers.items(), 1):
        desc = config.get("description", server_name)
        print(f"{i}. {desc}")
    print("="*50)
    
    while True:
        try:
            choice = int(input("Select a download method (4 for direct download recommended): ")) - 1
            if 0 <= choice < len(mcp_servers):
                return list(mcp_servers.keys())[choice], mcp_servers[list(mcp_servers.keys())[choice]]
            print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Please enter a valid number.")

async def check_tool_availability(command: str) -> bool:
    """Check if a command line tool is available."""
    try:
        result = subprocess.run([command, "--version"], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        return result.returncode == 0
    except:
        return False

async def install_uv_if_needed():
    """Install uv if not available."""
    if not await check_tool_availability("uv"):
        print("Installing uv...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
            return True
        except:
            print("Failed to install uv, will use alternative methods")
            return False
    return True

async def direct_download(url: str, filepath: Path) -> Dict[str, Any]:
    """Direct HTTP download using httpx."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/pdf,application/octet-stream,*/*',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            # Write file
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return {
                "status": response.status_code,
                "size": len(response.content),
                "success": True
            }
    except Exception as e:
        return {
            "status": 0,
            "error": str(e),
            "success": False
        }

async def fetch_with_mcp_server(url: str, server_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch content using MCP server."""
    try:
        # Start MCP server process
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"],
            env=server_config.get("env", {})
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # List available tools
                tools = await session.list_tools()
                print(f"Available tools: {[tool.name for tool in tools.tools]}")
                
                # Try to call a fetch tool
                for tool in tools.tools:
                    if "fetch" in tool.name.lower() or "get" in tool.name.lower():
                        result = await session.call_tool(
                            tool.name,
                            {
                                "url": url,
                                "method": "GET"
                            }
                        )
                        
                        return {
                            "status": 200,
                            "content": result.content,
                            "success": True
                        }
                
                return {
                    "status": 0,
                    "error": "No suitable fetch tool found",
                    "success": False
                }
                
    except Exception as e:
        print(f"MCP fetch error: {e}")
        return {
            "status": 0,
            "error": str(e),
            "success": False
        }

async def save_with_filesystem_mcp(content: bytes, filepath: Path) -> bool:
    """Save file using filesystem MCP server."""
    try:
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", str(filepath.parent)]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                tools = await session.list_tools()
                
                for tool in tools.tools:
                    if "write" in tool.name.lower():
                        # Convert content to base64 for MCP
                        content_b64 = base64.b64encode(content).decode('utf-8')
                        
                        await session.call_tool(
                            tool.name,
                            {
                                "path": str(filepath),
                                "content": content_b64,
                                "encoding": "base64"
                            }
                        )
                        return True
                
                return False
                
    except Exception as e:
        print(f"MCP filesystem error: {e}")
        return False

async def simple_download_test():
    """Simple script to test PDF downloads using various methods."""
    # Working PDF URLs for testing
    test_urls = [
        {
            "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "filename": "dummy.pdf",
            "description": "Small test PDF (13 KB)"
        },
        {
            "url": "https://www.africau.edu/images/default/sample.pdf",
            "filename": "sample_africau.pdf",
            "description": "Sample PDF document"
        },
        {
            "url": "https://scholar.harvard.edu/files/torman_personal/files/samplepdffile.pdf",
            "filename": "harvard_sample.pdf",
            "description": "Harvard sample PDF"
        },
        {
            "url": "https://www.learningcontainer.com/wp-content/uploads/2019/09/sample-pdf-file.pdf",
            "filename": "learning_container.pdf",
            "description": "Learning Container sample PDF"
        },
        {
            "url": "https://www.clickdimensions.com/links/TestPDFfile.pdf",
            "filename": "clickdimensions_test.pdf",
            "description": "ClickDimensions test PDF"
        }
    ]
    
    # Download directory
    download_dir = Path.home() / "Documents" / "test_downloads"
    download_dir.mkdir(exist_ok=True)
    
    print(f"Download directory: {download_dir}")
    print("="*70)
    
    # Select download method
    server_name, server_config = await select_mcp_server()
    print(f"Selected method: {server_name}")
    
    successful_downloads = []
    
    # Check if we need to install tools
    if server_config.get("command") == "uvx":
        await install_uv_if_needed()
    
    for i, pdf_info in enumerate(test_urls, 1):
        url = pdf_info["url"]
        filename = pdf_info["filename"] 
        description = pdf_info["description"]
        output_path = download_dir / filename
        
        print(f"\n{i}. Testing: {description}")
        print(f"   URL: {url}")
        print(f"   File: {filename}")
        
        try:
            if server_config.get("method") == "direct":
                # Direct download
                result = await direct_download(url, output_path)
                
                if result["success"]:
                    file_size = result["size"]
                    print(f"   ✅ SUCCESS - Downloaded {file_size:,} bytes")
                    successful_downloads.append({
                        'filename': filename,
                        'size': file_size,
                        'path': str(output_path),
                        'method': 'direct'
                    })
                else:
                    print(f"   ❌ FAILED - {result.get('error', 'Unknown error')}")
            
            else:
                # MCP-based download
                if not MCP_AVAILABLE:
                    print("   ⚠️  MCP not available, falling back to direct download")
                    result = await direct_download(url, output_path)
                    
                    if result["success"]:
                        file_size = result["size"]
                        print(f"   ✅ SUCCESS (fallback) - Downloaded {file_size:,} bytes")
                        successful_downloads.append({
                            'filename': filename,
                            'size': file_size,
                            'path': str(output_path),
                            'method': 'direct_fallback'
                        })
                    else:
                        print(f"   ❌ FAILED - {result.get('error', 'Unknown error')}")
                else:
                    # Try MCP fetch
                    fetch_result = await fetch_with_mcp_server(url, server_config)
                    
                    if fetch_result["success"]:
                        content = fetch_result["content"]
                        if isinstance(content, str):
                            content = content.encode('utf-8')
                        
                        # Try to save with MCP filesystem, fallback to direct
                        saved = await save_with_filesystem_mcp(content, output_path)
                        
                        if not saved:
                            # Fallback to direct file write
                            with open(output_path, 'wb') as f:
                                f.write(content)
                        
                        file_size = len(content)
                        print(f"   ✅ SUCCESS - Downloaded {file_size:,} bytes")
                        successful_downloads.append({
                            'filename': filename,
                            'size': file_size,
                            'path': str(output_path),
                            'method': 'mcp'
                        })
                    else:
                        print(f"   ❌ FAILED - {fetch_result.get('error', 'Unknown error')}")
                        
                        # Fallback to direct download
                        print("   🔄 Trying direct download fallback...")
                        result = await direct_download(url, output_path)
                        
                        if result["success"]:
                            file_size = result["size"]
                            print(f"   ✅ SUCCESS (fallback) - Downloaded {file_size:,} bytes")
                            successful_downloads.append({
                                'filename': filename,
                                'size': file_size,
                                'path': str(output_path),
                                'method': 'direct_fallback'
                            })
                        else:
                            print(f"   ❌ FAILED (fallback) - {result.get('error', 'Unknown error')}")
                    
        except Exception as e:
            print(f"   ❌ ERROR - {str(e)}")
            
            # Last resort: try direct download
            try:
                print("   🔄 Trying emergency direct download...")
                result = await direct_download(url, output_path)
                
                if result["success"]:
                    file_size = result["size"]
                    print(f"   ✅ SUCCESS (emergency) - Downloaded {file_size:,} bytes")
                    successful_downloads.append({
                        'filename': filename,
                        'size': file_size,
                        'path': str(output_path),
                        'method': 'emergency'
                    })
                else:
                    print(f"   ❌ FAILED (emergency) - {result.get('error', 'Unknown error')}")
            except:
                pass
    
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY:")
    print("="*70)
    
    if successful_downloads:
        print(f"✅ {len(successful_downloads)} files downloaded successfully:")
        for download in successful_downloads:
            method = download['method']
            print(f"   • {download['filename']} ({download['size']:,} bytes) [{method}]")
            print(f"     Path: {download['path']}")
        
        # Verify files exist and are not empty
        print(f"\n📁 File verification:")
        for download in successful_downloads:
            filepath = Path(download['path'])
            if filepath.exists() and filepath.stat().st_size > 0:
                print(f"   ✅ {download['filename']} - OK")
            else:
                print(f"   ❌ {download['filename']} - Missing or empty")
        
        print(f"\n🎯 All files saved to: {download_dir}")
        
    else:
        print("❌ No files downloaded successfully")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check your internet connection")
        print("   2. Try running with direct download method")
        print("   3. Make sure you have write permissions to the download directory")
    
    return successful_downloads

async def main():
    """Main function with error handling."""
    try:
        print("🚀 Starting PDF Download Test")
        print("="*70)
        
        # Check prerequisites
        print("Checking prerequisites...")
        
        if not HTTPX_AVAILABLE:
            print("❌ httpx not available - installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
        else:
            print("✅ httpx available")
        
        print("✅ Prerequisites check complete")
        print()
        
        # Run the download test
        results = await simple_download_test()
        
        print(f"\n🎉 Test completed! {len(results)} files downloaded.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Download test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Try running with direct download method for better reliability")

if __name__ == "__main__":
    asyncio.run(main())