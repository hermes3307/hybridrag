#!/usr/bin/env python3
"""
🤖 Conversational Document Processing Assistant
Main interface for natural language document processing
"""

import os
import sys
import time
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re

# Import our modules
from web_downloader import WebDocumentDownloader
from smart_chunker import SmartDocumentChunker  
from vector_manager import VectorStoreManager
from query_engine import ConversationalQueryEngine
from status import StatusManager  # 상단에 import 추가


@dataclass
class ConversationState:
    """Tracks the current conversation state"""
    current_url: Optional[str] = None
    downloaded_files: List[str] = None
    discovered_documents: List = None
    processed_chunks: int = 0
    vector_store_ready: bool = False
    user_preferences: Dict = None
    status_manager: StatusManager = None
    
    def __post_init__(self):
        if self.downloaded_files is None:
            self.downloaded_files = []
        if self.discovered_documents is None:
            self.discovered_documents = []
        if self.user_preferences is None:
            self.user_preferences = {
                'chunk_size': 1000,
                'overlap': 200,
                'preferred_formats': ['pdf', 'doc', 'docx', 'txt'],
                'language': 'auto'
            }
        # StatusManager를 여기서 초기화하지 않고, ConversationalAssistant에서 처리
class ConversationalAssistant:
    """🎭 The main conversational assistant that orchestrates everything"""
    
    def __init__(self):
        print("🚀 Initializing Conversational Document Assistant...")
        
        # Intent patterns을 가장 먼저 설정
        self.intent_patterns = {
            'download': [
                r'download.*from\s+(.+)',
                r'get.*documents.*from\s+(.+)',
                r'fetch.*files.*from\s+(.+)',
                r'process.*url\s+(.+)',
                r'scan\s+(.+)',
            ],
            'download_all': [
                r'download.*all',
                r'get.*all.*files',
                r'download.*everything',
                r'fetch.*all',
            ],
            'download_pdfs': [
                r'download.*pdf',
                r'get.*pdf.*files',
                r'fetch.*pdf',
                r'only.*pdf',
                r'download.*all.*pdf',
            ],
            'download_english': [
                r'download.*english',
                r'get.*english.*documents',
                r'only.*english',
            ],
            'chunk': [
                r'chunk.*documents?',
                r'process.*files?',
                r'break.*into.*pieces',
                r'split.*documents?',
                r'prepare.*for.*indexing',
            ],
            'index': [
                r'index.*into.*vector',
                r'add.*to.*database',
                r'create.*embeddings?',
                r'build.*vector.*store',
                r'make.*searchable',
            ],
            'search': [
                r'search.*for\s+(.+)',
                r'find.*about\s+(.+)',
                r'look.*for\s+(.+)',
                r'query.*(.+)',
                r'what.*about\s+(.+)',
            ],
            'status': [
                r'status.*',
                r'show.*status',
                r'check.*status',
                r'current.*status',
                r'what.*status',
                r'how.*many.*files',
                r'show.*progress',
                r'processing.*status',
                r'system.*status',
            ],
            'quick_status': [
                r'quick.*status',
                r'brief.*status',
                r'summary.*status',
                r'status.*summary',
            ],
            'help': [
                r'help.*',
                r'what.*can.*do',
                r'how.*to.*',
                r'commands?',
                r'options?',
            ]
        }
        
        # Initialize components
        self.downloader = WebDocumentDownloader()

        # Chunker 초기화 (빠뜨린 부분)
        try:
            from smart_chunker import SmartDocumentChunker
            self.chunker = SmartDocumentChunker()
        except ImportError:
            print("⚠️ SmartDocumentChunker not available")
            self.chunker = None
        
        
        # StatusManager를 초기화 (자동으로 기존 상태 로드)
        self.status_manager = StatusManager("processing_status.json")
        print("📊 Status manager initialized - loading previous session data...")
        
        # Conversation state
        self.state = ConversationState()
        self.state.status_manager = self.status_manager  # StatusManager 연결
        
        # Conversation history
        self.conversation_history = []
        
        # 저장된 상태에서 기본 정보 복원
        self._restore_session_state()
        
        print("✅ Assistant ready! Let's chat! 💬")
    

    def _restore_session_state(self):
        """💾 이전 세션 상태 복원"""
        try:
            current_status = self.status_manager.current_status
            
            # 다운로드된 파일 목록 복원
            download_status = current_status.get('download_status')
            if download_status and download_status.get('file_list'):
                # 파일이 실제로 존재하는지 확인
                existing_files = []
                for file_path in download_status['file_list']:
                    if os.path.exists(file_path):
                        existing_files.append(file_path)
                
                self.state.downloaded_files = existing_files
                print(f"📥 Restored {len(existing_files)} downloaded files from previous session")
            
            # 최근 URL 복원
            url_history = current_status.get('url_history', [])
            if url_history:
                self.state.current_url = url_history[-1]['url']
                print(f"🌐 Restored last URL: {self.state.current_url}")
            
            # 청크 상태 복원 (실제 청크 데이터는 없지만 통계는 복원)
            chunk_status = current_status.get('chunk_status')
            if chunk_status:
                self.state.processed_chunks = chunk_status.get('total_chunks', 0)
                print(f"🧩 Previous session had {self.state.processed_chunks} chunks")
            
            # 벡터 상태 복원
            vector_status = current_status.get('vector_status')
            if vector_status:
                self.state.vector_store_ready = vector_status.get('is_ready', False)
                if self.state.vector_store_ready:
                    print(f"🗄️ Vector database was ready in previous session")
            
        except Exception as e:
            print(f"⚠️ Could not fully restore previous session: {e}")

    def parse_intent(self, user_input: str) -> Dict[str, Any]:
        """🧠 Parse user intent from natural language"""
        original_input = user_input.strip()  # Keep original case
        user_input_lower = user_input.lower().strip()  # Lowercase only for pattern matching
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, user_input_lower, re.IGNORECASE)
                if match:
                    # Extract data from original input to preserve case
                    if match.groups():
                        # Find the matched portion in original input
                        original_match = re.search(pattern, original_input, re.IGNORECASE)
                        extracted_data = original_match.groups()[0] if original_match and original_match.groups() else None
                    else:
                        extracted_data = None
                    return {
                        'intent': intent,
                        'data': extracted_data or original_input,  # Use original case
                        'raw_input': original_input,  # Original case preserved
                        'confidence': 0.9
                    }
        
        # If no pattern matches, try keyword detection
        keywords = {
            'download': ['download', 'get', 'fetch', 'scan', 'grab'],
            'chunk': ['chunk', 'split', 'break', 'process', 'divide'],
            'index': ['index', 'embed', 'vector', 'database', 'store'],
            'search': ['search', 'find', 'look', 'query', 'what', 'how'],
            'status': ['status', 'progress', 'state', 'files', 'count'],
        }
        
        for intent, words in keywords.items():
            if any(word in user_input_lower for word in words):
                return {
                    'intent': intent,
                    'data': original_input,  # Return original case
                    'raw_input': original_input,
                    'confidence': 0.6
                }
        
        return {
            'intent': 'unknown',
            'data': original_input,  # Return original case
            'raw_input': original_input,
            'confidence': 0.3
        }

    async def handle_download_intent(self, data: str) -> str:
        """📥 Handle document download requests"""
        if not data:
            return ("🤖 I'd love to help you download documents! Please provide a URL. "
                "For example: 'Download documents from https://example.com/docs'")
        
        # Extract URL if it's in the text
        url_match = re.search(r'https?://[^\s]+', data)
        if url_match:
            url = url_match.group()
        else:
            url = data.strip()
        
        self.state.current_url = url
        
        try:
            print(f"🔍 Analyzing URL: {url}")
            
            # Scan for documents
            documents = await self.downloader.scan_documents(url)
            
            # Store discovered documents in state
            self.state.discovered_documents = documents
            
            if not documents:
                # 실패한 경우에도 상태 저장
                self.status_manager.update_url_status(
                    url=url,
                    documents_found=0,
                    total_size_mb=0,
                    status='failed'
                )
                return f"🤖 I couldn't find any documents at {url}. Could you check the URL?"
            
            # 성공한 경우 상태 업데이트
            total_size_mb = sum(doc.size or 0 for doc in documents) / (1024 * 1024)
            self.status_manager.update_url_status(
                url=url,
                documents_found=len(documents),
                total_size_mb=total_size_mb,
                status='scanned'
            )
            print(f"💾 Saved scan status: {len(documents)} documents found")
            
            # Show what we found
            doc_summary = self.downloader.summarize_findings(documents)
            
            response = f"🎉 Great! I found documents at {url}:\n\n{doc_summary}\n\n"
            response += "💡 What would you like to do?\n"
            response += "• 'Download all PDFs'\n"
            response += "• 'Download everything'\n" 
            response += "• 'Download only English documents'\n"
            response += "• 'Show me more details first'"
            
            return response
            
        except Exception as e:
            # 오류 발생시에도 상태 저장
            self.status_manager.update_url_status(
                url=url,
                documents_found=0,
                total_size_mb=0,
                status='failed'
            )
            return f"❌ Oops! I had trouble accessing {url}. Error: {str(e)}"

    async def handle_download_all_intent(self, data: str) -> str:
        """📥 Handle download all requests"""
        if not self.state.discovered_documents:
            return ("🤖 I don't see any discovered documents to download! "
                "Please scan a URL first.")
        
        try:
            downloaded_files = await self.downloader.download_documents()
            self.state.downloaded_files.extend(downloaded_files)
            
            stats = self.downloader.get_download_stats()
            
            # 상태 업데이트 (절대 경로로 저장)
            absolute_file_paths = [os.path.abspath(f) for f in downloaded_files]
            self.status_manager.update_download_status(
                total_files=stats['total_found'],
                downloaded_files=stats['downloaded'],
                failed_files=stats['failed'],
                total_size_mb=stats['bytes_downloaded'] / (1024 * 1024),
                download_directory=str(os.path.abspath(self.downloader.download_dir)),
                file_list=absolute_file_paths
            )
            print(f"💾 Saved download status: {len(downloaded_files)} files downloaded")
            
            response = f"✅ Download complete!\n\n"
            response += f"📥 Downloaded: {stats['downloaded']} files\n"
            response += f"❌ Failed: {stats['failed']} files\n"
            if stats['bytes_downloaded'] > 0:
                mb_downloaded = stats['bytes_downloaded'] / (1024 * 1024)
                response += f"💾 Total size: {mb_downloaded:.1f} MB\n"
            
            response += f"📁 Files saved to: {self.downloader.download_dir}\n\n"
            response += "💡 You can now ask me to 'list files' or check 'status'!"
            
            return response
            
        except Exception as e:
            return f"❌ Download failed. Error: {str(e)}"

    async def handle_chunk_intent(self, data: str) -> str:
        """🧩 Handle document chunking requests"""

        """🧩 Handle document chunking requests"""
        if not self.chunker:
            return ("❌ Document chunker is not available. Please check if smart_chunker.py is accessible.")
        
        if not self.state.downloaded_files:
            return ("🤖 I don't see any downloaded files to chunk yet! "
                "Would you like me to download some documents first?")
        
        if not self.state.downloaded_files:
            return ("🤖 I don't see any downloaded files to chunk yet! "
                "Would you like me to download some documents first?")
        
        try:
            print(f"🧩 Chunking {len(self.state.downloaded_files)} files...")
            
            # Extract preferences from user input
            chunk_prefs = self._extract_chunk_preferences(data)
            
            # Process the files - use the correct parameter names
            chunks = await self.chunker.process_files(
                self.state.downloaded_files,
                chunk_size=chunk_prefs['chunk_size'],
                overlap=chunk_prefs['overlap'],
                use_semantic_splitting=chunk_prefs.get('use_semantic_splitting', True),
                preserve_structure=True
            )
            
            self.state.processed_chunks = len(chunks)
            
            # 상태 업데이트
            processing_stats = self.chunker.get_processing_stats()
            self.status_manager.update_chunk_status(
                total_chunks=len(chunks),
                total_characters=processing_stats.get('total_characters', 0),
                files_processed=processing_stats.get('files_processed', 0),
                processing_errors=processing_stats.get('errors', []),
                chunk_size=chunk_prefs['chunk_size'],
                overlap=chunk_prefs['overlap'],
                semantic_chunking=chunk_prefs.get('use_semantic_splitting', True)
            )
            print(f"💾 Saved chunking status: {len(chunks)} chunks created")
            
            response = f"✅ Perfect! I've processed your documents:\n\n"
            response += f"📄 Files processed: {len(self.state.downloaded_files)}\n"
            response += f"🧩 Chunks created: {self.state.processed_chunks:,}\n"
            response += f"⚙️ Chunk size: ~{chunk_prefs['chunk_size']} characters\n"
            response += f"🔗 Overlap: {chunk_prefs['overlap']} characters\n\n"
            response += "🎯 Ready to index into vector database?"
            
            return response
            
        except Exception as e:
            return f"❌ I had trouble chunking the documents. Error: {str(e)}"
        

    async def handle_download_pdfs_intent(self, data: str) -> str:
        """📄 Handle PDF-only download requests"""
        if not self.state.discovered_documents:  # Check state instead of downloader
            return ("🤖 I don't see any discovered documents to download! "
                "Please scan a URL first.")
        
        try:
            downloaded_files = await self.downloader.download_documents({'extensions': ['.pdf']})
            self.state.downloaded_files.extend(downloaded_files)
            
            stats = self.downloader.get_download_stats()
            
            response = f"✅ PDF download complete!\n\n"
            response += f"📄 Downloaded: {stats['downloaded']} PDF files\n"
            response += f"❌ Failed: {stats['failed']} files\n"
            if stats['bytes_downloaded'] > 0:
                mb_downloaded = stats['bytes_downloaded'] / (1024 * 1024)
                response += f"💾 Total size: {mb_downloaded:.1f} MB\n"
            
            response += f"📁 Files saved to: {self.downloader.download_dir}\n\n"
            response += "💡 You can now ask me to 'list files' or check 'status'!"
            
            return response
            
        except Exception as e:
            return f"❌ PDF download failed. Error: {str(e)}"
       
    async def handle_download_english_intent(self, data: str) -> str:
        """🌍 Handle English-only download requests"""
        if not hasattr(self.downloader, 'discovered_documents') or not self.downloader.discovered_documents:
            return ("🤖 I don't see any discovered documents to download! "
                "Please scan a URL first.")
        
        try:
            downloaded_files = await self.downloader.download_documents({'language': 'English'})
            self.state.downloaded_files.extend(downloaded_files)
            
            stats = self.downloader.get_download_stats()
            
            response = f"✅ English documents download complete!\n\n"
            response += f"📄 Downloaded: {stats['downloaded']} English files\n"
            response += f"❌ Failed: {stats['failed']} files\n"
            if stats['bytes_downloaded'] > 0:
                mb_downloaded = stats['bytes_downloaded'] / (1024 * 1024)
                response += f"💾 Total size: {mb_downloaded:.1f} MB\n"
            
            response += f"📁 Files saved to: {self.downloader.download_dir}\n\n"
            response += "💡 You can now ask me to 'list files' or check 'status'!"
            
            return response
            
        except Exception as e:
            return f"❌ English download failed. Error: {str(e)}"

    async def handle_index_intent(self, data: str) -> str:
        """🗄️ Handle vector indexing requests"""
        if self.state.processed_chunks == 0:
            return ("🤖 I need some chunked documents to index! "
                "Would you like me to chunk your downloaded files first?")
        
        # 벡터 매니저가 아직 구현되지 않았으므로 시뮬레이션
        try:
            print(f"🗄️ Indexing {self.state.processed_chunks} chunks...")
            
            # 시뮬레이션된 벡터 인덱싱 (실제로는 아직 구현되지 않음)
            await asyncio.sleep(2)  # 처리 시간 시뮬레이션
            
            # 상태를 "준비됨"으로 업데이트
            self.state.vector_store_ready = True
            
            # Status 업데이트
            self.status_manager.update_vector_status(
                is_ready=True,
                collection_name="conversation_docs",
                vector_count=self.state.processed_chunks,
                vector_dimensions=384,  # 일반적인 임베딩 차원
                index_size_mb=self.state.processed_chunks * 0.001,  # 추정 크기
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                search_capabilities=["semantic_search", "similarity_search"]
            )
            print(f"💾 Saved vector index status: {self.state.processed_chunks} vectors indexed")
            
            response = f"🎉 Indexing simulation completed!\n\n"
            response += f"🗄️ Vector database: Ready\n"
            response += f"📊 Indexed chunks: {self.state.processed_chunks:,}\n"
            response += f"🔢 Vector dimensions: 384\n"
            response += f"🔍 Collection: 'conversation_docs'\n\n"
            response += "💡 **Note**: This is a simulation. Actual vector search is not yet implemented.\n"
            response += "🎯 You can now try asking about document content (will show placeholder results)!"
            
            return response
            
        except Exception as e:
            return f"❌ Indexing simulation failed. Error: {str(e)}" 



    async def handle_search_intent(self, data: str) -> str:
        """🔍 Handle search requests"""
        if not self.state.vector_store_ready:
            return ("🤖 I'd love to search for you, but the vector database isn't ready yet! "
                   "Would you like me to index your documents first?")
        
        if not data or len(data.strip()) < 3:
            return ("🤖 What would you like me to search for? "
                   "Try something like: 'Find information about SQL performance'")
        
        try:
            print(f"🔍 Searching for: {data}")
            
            # Perform the search
            results = await self.query_engine.search(data, k=3)
            
            if not results:
                return f"🤖 I couldn't find anything specific about '{data}'. Try rephrasing your question!"
            
            # Format response
            response = f"🎯 Found {len(results)} relevant results for '{data}':\n\n"
            
            for i, result in enumerate(results, 1):
                payload = result.get('payload', {})
                score = result.get('score', 0)
                
                response += f"**📋 Result {i}** (Relevance: {score:.1%})\n"
                response += f"📖 Source: {payload.get('source', 'Unknown')}\n"
                
                # Add snippet
                text = payload.get('text', '')[:300] + "..." if len(payload.get('text', '')) > 300 else payload.get('text', '')
                response += f"📄 Content: {text}\n"
                response += "─" * 50 + "\n\n"
            
            response += "💡 Want me to search for something else or show more details?"
            return response
            
        except Exception as e:
            return f"❌ Search failed. Error: {str(e)}"

    def handle_status_intent(self, data: str) -> str:
        """📊 Handle comprehensive status requests"""
        # 간단한 상태 요청인지 확인
        if any(word in data.lower() for word in ['quick', 'brief', 'summary']):
            return self.state.status_manager.get_quick_status()
        
        # 종합 상태 보고서 반환
        return self.state.status_manager.get_comprehensive_status()

    def handle_quick_status_intent(self, data: str) -> str:
        """⚡ Handle quick status requests"""
        return self.state.status_manager.get_quick_status()

    def handle_help_intent(self, data: str) -> str:
        """❓ Handle help requests"""
        return """🤖 **I'm your conversational document assistant!** Here's what I can do:

🌐 **Download Documents**
   • "Download PDFs from https://example.com/docs"
   • "Get all documents from that GitHub repo"
   • "Scan this URL for manuals"

🧩 **Process & Chunk**
   • "Chunk the downloaded files"
   • "Break documents into pieces" 
   • "Process files for indexing"

🗄️ **Index & Store**
   • "Index into vector database"
   • "Make documents searchable"
   • "Build the vector store"

🔍 **Search & Query**
   • "Search for database optimization"
   • "Find installation procedures"
   • "What about performance tuning?"

📊 **Check Status**
   • "What's the current status?"
   • "How many files downloaded?"
   • "Show progress"

💡 **Just talk naturally!** I understand conversational language, so feel free to ask however feels natural to you!"""
    def _extract_chunk_preferences(self, data: str) -> Dict:
        """Extract chunking preferences from user input"""
        prefs = self.state.user_preferences.copy()
        
        # Look for size mentions
        size_match = re.search(r'(\d+)\s*(?:char|character|word|token)', data, re.IGNORECASE)
        if size_match:
            prefs['chunk_size'] = int(size_match.group(1))
        
        # Look for overlap mentions  
        overlap_match = re.search(r'overlap.*?(\d+)', data, re.IGNORECASE)
        if overlap_match:
            prefs['overlap'] = int(overlap_match.group(1))  # Changed from 'chunk_overlap' to 'overlap'
        
        # Look for semantic preferences
        if any(word in data.lower() for word in ['semantic', 'smart', 'intelligent', 'paragraph']):
            prefs['use_semantic_splitting'] = True
        
        return prefs

    async def chat(self, user_input: str) -> str:
        """💬 Main chat interface"""
        if not user_input.strip():
            return "🤖 I'm here to help! What would you like me to do with documents today?"
        
        # Parse intent
        intent_result = self.parse_intent(user_input)
        intent = intent_result['intent']
        data = intent_result['data']
        
        # Store conversation
        self.conversation_history.append({
            'user': user_input,
            'intent': intent,
            'timestamp': time.time()
        })

        # Route to appropriate handler
        if intent == 'download':
            response = await self.handle_download_intent(data)
        elif intent == 'download_all':
            response = await self.handle_download_all_intent(data)
        elif intent == 'download_pdfs':
            response = await self.handle_download_pdfs_intent(data)
        elif intent == 'download_english':
            response = await self.handle_download_english_intent(data)
        elif intent == 'chunk':
            response = await self.handle_chunk_intent(data)
        elif intent == 'index':
            response = await self.handle_index_intent(data)
        elif intent == 'search':
            response = await self.handle_search_intent(data)
        elif intent == 'status':
            response = self.handle_status_intent(data)
        elif intent == 'quick_status':
            response = self.handle_quick_status_intent(data)
        elif intent == 'help':
            response = self.handle_help_intent(data)
        
        else:
            response = ("🤖 I'm not quite sure what you want me to do. "
                    "Try asking me to download, chunk, index, or search documents. "
                    "Or just ask for 'help' to see what I can do!")
        
        # Store response
        self.conversation_history[-1]['assistant'] = response
        
        return response

async def main():
    """🎭 Main interactive chat loop"""
    print("🌟" + "=" * 60 + "🌟")
    print("      🤖 Conversational Document Processing Assistant")
    print("🌟" + "=" * 60 + "🌟")
    print()
    print("💬 Hi! I'm your document processing assistant!")
    print("   I can download, chunk, index, and search documents")
    print("   Just talk to me naturally - no need for special commands!")
    print()
    print("🎯 Try saying something like:")
    print("   • 'Download PDFs from https://example.com/docs'")
    print("   • 'Process the downloaded files'") 
    print("   • 'Search for database optimization'")
    print()
    print("   Type 'quit' to exit")
    print("=" * 68)
    
    assistant = ConversationalAssistant()
    conversation_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n😊 You: ").strip()
            
            # Check for exit
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print(f"\n🤖 Goodbye! We had {conversation_count} great conversations today! ✨")
                break
            
            if not user_input:
                continue
            
            # Get response
            print(f"\n🤖 Assistant: ", end="", flush=True)
            response = await assistant.chat(user_input)
            print(response)
            
            conversation_count += 1
            
        except KeyboardInterrupt:
            print(f"\n\n🤖 Goodbye! We had {conversation_count} great conversations today! ✨")
            break
        except Exception as e:
            print(f"\n❌ Oops! Something went wrong: {e}")
            print("💡 Please try again or ask for help!")

if __name__ == "__main__":
    asyncio.run(main())