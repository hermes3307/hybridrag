#!/usr/bin/env python3
"""
🤖 Conversational Document Processing Assistant with RAG
Main interface for natural language document processing and intelligent Q&A
"""
import openai
import os
from dotenv import load_dotenv
import asyncio

# 🚀 환경변수 로드 추가
load_dotenv()

import os
import sys
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from collections import Counter

# Import our modules
from web_downloader import WebDocumentDownloader
from smart_chunker import SmartDocumentChunker  
from vector_manager import VectorStoreManager
from query_engine import ConversationalQueryEngine
from status import StatusManager

# 🔍 RAG 디버거 추가
try:
    from debug_rag import RAGProcessDebugger
    RAG_DEBUG_AVAILABLE = True
    print("✅ RAG Debug module loaded successfully")
except ImportError as e:
    print(f"⚠️ RAG Debug module not available: {e}")
    RAG_DEBUG_AVAILABLE = False

# Download required NLTK data with better compatibility
import nltk

def download_nltk_data():
    """Download required NLTK data with fallback for different versions"""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),  # New format
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    
    for data_path, download_name in required_data:
        try:
            nltk.data.find(data_path)
            print(f"✅ {download_name} already available")
        except LookupError:
            try:
                print(f"📥 Downloading {download_name}...")
                nltk.download(download_name, quiet=True)
                print(f"✅ {download_name} downloaded successfully")
            except Exception as e:
                print(f"⚠️ Could not download {download_name}: {e}")
                if download_name == 'punkt_tab':
                    print("   Falling back to punkt tokenizer")

# Download NLTK data at startup
download_nltk_data()

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    print("✅ NLTK components loaded successfully")
except ImportError as e:
    print(f"⚠️ NLTK import error: {e}")
    # Provide fallback implementations
    def word_tokenize(text):
        return text.lower().split()
    
    def sent_tokenize(text):
        import re
        return re.split(r'[.!?]+', text)
    
    def pos_tag(tokens):
        return [(token, 'NN') for token in tokens]  # Default to noun
    
    class MockStopwords:
        def words(self, lang):
            return {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that', 'these', 'those'}
    
    stopwords = MockStopwords()
    print("⚠️ Using fallback NLTK implementations")


class TopicKeywordExtractor:
    """🧠 Intelligent topic and keyword extraction from user questions"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            # Fallback stopwords
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that', 'these', 'those'}
        
        # Technical domain keywords for better extraction
        self.domain_keywords = {
            'database': ['database', 'db', 'sql', 'table', 'index', 'query', 'schema', 'constraint',
                        'transaction', 'backup', 'restore', 'replication', 'partition', 'view', 'trigger'],
            'performance': ['performance', 'optimization', 'tuning', 'slow', 'fast', 'memory',
                           'cpu', 'disk', 'cache', 'bottleneck', 'latency', 'throughput', 'speed'],
            'configuration': ['config', 'setting', 'parameter', 'option', 'configure', 'setup',
                             'installation', 'deployment', 'environment', 'property'],
            'security': ['security', 'authentication', 'authorization', 'permission', 'user',
                        'role', 'privilege', 'encryption', 'ssl', 'certificate', 'password'],
            'network': ['network', 'connection', 'port', 'protocol', 'tcp', 'ip', 'socket',
                       'communication', 'client', 'server', 'connection'],
            'storage': ['storage', 'disk', 'file', 'directory', 'backup', 'archive', 'volume',
                       'tablespace', 'datafile', 'memory', 'buffer'],
            'altibase': ['altibase', 'ipcda', 'apre', 'isql', 'iloader', 'altiadmin', 'altimon',
                        'hybrid', 'inmemory', 'diskdb', 'memorydb', 'sharding']
        }
        
        # Question patterns that indicate need for detailed explanation
        self.question_patterns = {
            'what_is': [r'what\s+is\s+(.+)', r'what\s+are\s+(.+)', r'define\s+(.+)', r'explain\s+(.+)'],
            'how_to': [r'how\s+to\s+(.+)', r'how\s+can\s+i\s+(.+)', r'how\s+do\s+i\s+(.+)', r'steps\s+to\s+(.+)'],
            'why': [r'why\s+(.+)', r'what\s+causes\s+(.+)', r'reason\s+for\s+(.+)'],
            'when': [r'when\s+(.+)', r'what\s+time\s+(.+)', r'at\s+what\s+point\s+(.+)'],
            'where': [r'where\s+(.+)', r'in\s+which\s+(.+)', r'location\s+of\s+(.+)'],
            'troubleshoot': [r'error\s+(.+)', r'problem\s+(.+)', r'issue\s+(.+)', r'fix\s+(.+)', r'solve\s+(.+)'],
            'best_practices': [r'best\s+practices?\s+(.+)', r'recommended\s+(.+)', r'optimal\s+(.+)'],
            'comparison': [r'difference\s+between\s+(.+)', r'compare\s+(.+)', r'(.+)\s+vs\s+(.+)']
        }

    def extract_topics_and_keywords(self, text: str) -> Dict[str, Any]:
        """🔍 Extract topics, keywords, and question type from user input"""
        text_lower = text.lower().strip()
        
        # 1. Detect question type
        question_type = self._detect_question_type(text_lower)
        
        # 2. Extract keywords using NLP
        keywords = self._extract_keywords_nlp(text)
        
        # 3. Identify domain topics
        domains = self._identify_domains(text_lower, keywords)
        
        # 4. Extract technical terms
        technical_terms = self._extract_technical_terms(text_lower)
        
        # 5. Build search queries
        search_queries = self._build_search_queries(keywords, technical_terms, question_type)
        
        return {
            'original_text': text,
            'question_type': question_type,
            'keywords': keywords,
            'technical_terms': technical_terms,
            'domains': domains,
            'search_queries': search_queries,
            'is_complex_question': self._is_complex_question(text_lower, question_type)
        }

    def _detect_question_type(self, text: str) -> str:
        """Detect the type of question being asked"""
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return q_type
        return 'general'

    def _extract_keywords_nlp(self, text: str) -> List[str]:
        """Extract keywords using NLP techniques with fallback"""
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract meaningful words (nouns, adjectives, verbs)
            meaningful_words = []
            for word, pos in pos_tags:
                if (pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB')) and \
                   word not in self.stop_words and len(word) > 2 and word.isalpha():
                    meaningful_words.append(word)
            
            # Count frequency and return top keywords
            word_freq = Counter(meaningful_words)
            return [word for word, freq in word_freq.most_common(10)]
            
        except Exception as e:
            print(f"⚠️ NLP extraction failed, using simple fallback: {e}")
            # Simple fallback: just split and filter
            words = text.lower().split()
            return [word for word in words if word not in self.stop_words and len(word) > 2 and word.isalpha()][:10]

    def _identify_domains(self, text: str, keywords: List[str]) -> List[str]:
        """Identify technical domains mentioned in the text"""
        domains = []
        all_words = text.split() + keywords
        
        for domain, domain_words in self.domain_keywords.items():
            if any(word in all_words for word in domain_words):
                domains.append(domain)
        
        return domains

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms and acronyms"""
        # Find acronyms (2+ uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text.upper())
        
        # Find technical terms from domain keywords
        technical_terms = []
        words = text.split()
        
        for domain_words in self.domain_keywords.values():
            for word in words:
                if word.lower() in domain_words:
                    technical_terms.append(word.lower())
        
        return list(set(acronyms + technical_terms))

    def _build_search_queries(self, keywords: List[str], technical_terms: List[str], 
                             question_type: str) -> List[str]:
        """Build multiple search queries for comprehensive retrieval"""
        queries = []
        
        # Main query with all keywords
        if keywords:
            main_query = ' '.join(keywords[:5])  # Top 5 keywords
            queries.append(main_query)
        
        # Technical terms query
        if technical_terms:
            tech_query = ' '.join(technical_terms)
            queries.append(tech_query)
        
        # Question-type specific queries
        if question_type == 'how_to':
            if keywords:
                queries.append(f"tutorial {' '.join(keywords[:3])}")
                queries.append(f"guide {' '.join(keywords[:3])}")
        elif question_type == 'troubleshoot':
            if keywords:
                queries.append(f"error {' '.join(keywords[:3])}")
                queries.append(f"problem {' '.join(keywords[:3])}")
        elif question_type == 'what_is':
            if keywords:
                queries.append(f"definition {' '.join(keywords[:3])}")
        
        return list(set(queries))  # Remove duplicates

    def _is_complex_question(self, text: str, question_type: str) -> bool:
        """Determine if this is a complex question needing RAG"""
        # Questions with multiple parts
        if len(text.split('?')) > 2:
            return True
        
        # Long questions
        if len(text.split()) > 8:
            return True
        
        # Specific question types that need detailed answers
        complex_types = ['how_to', 'why', 'troubleshoot', 'best_practices', 'comparison']
        if question_type in complex_types:
            return True
        
        # Contains technical terms
        if any(domain_words for domain_words in self.domain_keywords.values() 
               if any(word in text for word in domain_words)):
            return True
        
        return False


class RAGEngine:
    """🧠 Retrieval Augmented Generation Engine"""
    
    def __init__(self, vector_manager: VectorStoreManager, query_engine: ConversationalQueryEngine):
        self.vector_manager = vector_manager
        self.query_engine = query_engine
        self.topic_extractor = TopicKeywordExtractor()
        
        # RAG prompts for different question types
        self.rag_prompts = {
            'what_is': """Based on the following context from technical documentation, provide a comprehensive explanation of what {topic} is:

Context:
{context}

Question: {question}

Please provide a detailed explanation that includes:
1. Definition and overview
2. Key characteristics or features
3. Use cases or applications
4. Any important technical details mentioned in the context

Answer:""",
            
            'how_to': """Based on the following context from technical documentation, provide step-by-step guidance on how to {topic}:

Context:
{context}

Question: {question}

Please provide:
1. Clear step-by-step instructions
2. Prerequisites or requirements
3. Important notes or warnings
4. Expected outcomes

Answer:""",
            
            'troubleshoot': """Based on the following context from technical documentation, help troubleshoot the issue with {topic}:

Context:
{context}

Question: {question}

Please provide:
1. Possible causes of the problem
2. Diagnostic steps
3. Solutions or workarounds
4. Prevention measures

Answer:""",
            
            'general': """Based on the following context from technical documentation, please answer the question about {topic}:

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the documentation context above.

Answer:"""
        }

        # 🚀 OpenAI 클라이언트 초기화 (환경변수 확인 포함)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
            print("✅ OpenAI client initialized")
        else:
            print("⚠️ OPENAI_API_KEY not found in environment variables")
            self.openai_client = None
            

    async def _generate_llm_response(self, question: str, context: str, question_type: str, main_topic: str) -> str:
        """🤖 OpenAI GPT로 실제 응답 생성"""
        
        # OpenAI 클라이언트 확인
        if not self.openai_client:
            print("⚠️ OpenAI client not available")
            return self._create_structured_response_fallback(question, context, question_type, main_topic)
        
        # 질문 타입별 프롬프트 선택 🎯
        prompt_template = self.rag_prompts.get(question_type, self.rag_prompts['general'])
        
        # 최종 프롬프트 구성 📝
        final_prompt = prompt_template.format(
            topic=main_topic,
            context=context,
            question=question
        )
        
        try:
            # 🚀 OpenAI API 호출
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": """당신은 기술 문서 전문가입니다. 
                        제공된 컨텍스트를 바탕으로 정확하고 상세한 답변을 제공하세요.
                        답변은 한국어로 작성하고, 이모지를 적절히 사용해 친근하게 설명해주세요."""
                    },
                    {
                        "role": "user", 
                        "content": final_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500,
                top_p=1.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ OpenAI API 오류: {e}")
            return self._create_structured_response_fallback(question, context, question_type, main_topic)

    def _create_structured_response_fallback(self, question: str, context: str, question_type: str, main_topic: str) -> str:
        """🛡️ OpenAI 실패시 폴백 응답"""
        return f"""🤖 OpenAI 연결에 문제가 있어 간단한 응답을 드려요:

📄 **{main_topic}에 대한 정보:**
{context[:500]}...

💡 더 자세한 답변을 위해 OpenAI API 키를 확인해주세요!"""        

    async def generate_rag_response(self, question: str, max_context_length: int = 4000) -> str:
        """🎯 Generate RAG response for complex questions"""
        
        try:
            # Extract topics and keywords
            extraction_result = self.topic_extractor.extract_topics_and_keywords(question)
            
            if not extraction_result['is_complex_question']:
                return None  # Let normal search handle simple questions
            
            print(f"🧠 RAG Analysis: {extraction_result['question_type']} question about {extraction_result['domains']}")
            print(f"🔍 Keywords: {extraction_result['keywords']}")
            print(f"⚙️ Technical terms: {extraction_result['technical_terms']}")
            
            # Perform multiple searches for comprehensive context
            all_results = []
            
            for query in extraction_result['search_queries']:
                if query.strip():
                    try:
                        results = await self.vector_manager.search(query, k=5)
                        all_results.extend(results)
                        print(f"📊 Found {len(results)} results for query: '{query}'")
                    except Exception as e:
                        print(f"⚠️ Search failed for query '{query}': {e}")
                        continue
            
            if not all_results:
                return f"🤖 I understand you're asking about {', '.join(extraction_result['keywords'])}, but I couldn't find relevant information in the indexed documents. Could you try rephrasing your question or check if the documents have been properly indexed?"
            
            # Deduplicate and rank results
            unique_results = self._deduplicate_results(all_results)
            top_results = sorted(unique_results, key=lambda x: x['score'], reverse=True)[:8]
            
            # Build context from top results
            context = self._build_context(top_results, max_context_length)
            
            # Generate RAG response
            rag_response = await self._generate_response_with_context(
                question, 
                context, 
                extraction_result,
                top_results
            )
            
            return rag_response
            
        except Exception as e:
            print(f"⚠️ RAG processing failed: {e}")
            return f"🤖 I encountered an issue processing your question. Let me try a simpler search approach. Error: {str(e)}"

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content similarity"""
        unique_results = []
        seen_content = set()
        
        for result in results:
            content = result.get('payload', {}).get('text', '')[:200]  # First 200 chars
            content_hash = hash(content)
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results

    def _build_context(self, results: List[Dict], max_length: int) -> str:
        """Build context string from search results"""
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            payload = result.get('payload', {})
            text = payload.get('text', '').strip()
            source = payload.get('source_file', 'Unknown')
            
            if text and current_length < max_length:
                # Add source attribution
                context_part = f"[Source {i}: {source}]\n{text}\n"
                
                if current_length + len(context_part) <= max_length:
                    context_parts.append(context_part)
                    current_length += len(context_part)
                else:
                    # Add truncated version
                    remaining_space = max_length - current_length - 100  # Leave some buffer
                    if remaining_space > 100:
                        truncated_text = text[:remaining_space] + "..."
                        context_parts.append(f"[Source {i}: {source}]\n{truncated_text}\n")
                    break
        
        return "\n".join(context_parts)

    async def _generate_response_with_context(self, question: str, context: str, 
                                            extraction_result: Dict, results: List[Dict]) -> str:
        """Generate the final RAG response"""
        
        question_type = extraction_result['question_type']
        keywords = extraction_result['keywords']
        main_topic = ', '.join(keywords[:3]) if keywords else '주제'
        
        # 🚀 OpenAI로 응답 생성 시도
        llm_response = await self._generate_llm_response(question, context, question_type, main_topic)
        
        # 📚 소스 정보 추가
        sources_info = self._format_sources(results)
        
        final_response = f"""{llm_response}

{sources_info}

💡 **더 궁금한 점이 있으시면 언제든 물어보세요!** 😊"""
        
        return final_response

    def _format_sources(self, results: List[Dict]) -> str:
        """📚 소스 정보 포맷팅"""
        sources = set()
        for result in results[:5]:
            source = result.get('payload', {}).get('source_file', 'Unknown')
            if source != 'Unknown':
                sources.add(source)
        
        if not sources:
            return ""
        
        sources_text = "\n📚 **참고 문서:**\n"
        for i, source in enumerate(sources, 1):
            sources_text += f"   {i}. 📄 {source}\n"
        
        return sources_text


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


class ConversationalAssistant:
    """🎭 The main conversational assistant with RAG capabilities"""
    
    def __init__(self):
        print("🚀 Initializing Conversational Document Assistant with RAG...")
        
        # Intent patterns for document processing commands
        self.command_patterns = {
            'download', 'scan', 'fetch', 'get documents',
            'chunk', 'process files', 'split documents',
            'index', 'build vector', 'create embeddings', 'make searchable',
            'status', 'progress', 'show files',
            'help', 'commands'
        }
        
        # Initialize components
        self.downloader = WebDocumentDownloader()

        # Chunker 초기화
        try:
            from smart_chunker import SmartDocumentChunker
            self.chunker = SmartDocumentChunker()
        except ImportError:
            print("⚠️ SmartDocumentChunker not available")
            self.chunker = None
        
        # Initialize vector manager and query engine
        try:
            self.vector_manager = VectorStoreManager()
            self.query_engine = ConversationalQueryEngine(self.vector_manager)
            self.rag_engine = RAGEngine(self.vector_manager, self.query_engine)
            print("✅ Vector search and RAG components initialized")
        except Exception as e:
            print(f"⚠️ Vector search and RAG not available: {e}")
            self.vector_manager = None
            self.query_engine = None
            self.rag_engine = None
        
        # 🔍 RAG 디버거 초기화
        if RAG_DEBUG_AVAILABLE and self.vector_manager and self.query_engine:
            try:
                self.rag_debugger = RAGProcessDebugger(self.vector_manager, self.query_engine)
                print("✅ RAG debugger initialized")
            except Exception as e:
                print(f"⚠️ RAG debugger initialization failed: {e}")
                self.rag_debugger = None
        else:
            self.rag_debugger = None
        
        # StatusManager 초기화
        self.status_manager = StatusManager("processing_status.json")
        print("📊 Status manager initialized - loading previous session data...")
        
        # Conversation state
        self.state = ConversationState()
        self.state.status_manager = self.status_manager
        
        # Conversation history
        self.conversation_history = []
        
        # 저장된 상태에서 기본 정보 복원
        self._restore_session_state()
        
        print("✅ Assistant ready with RAG capabilities! Let's chat! 💬")

    def _restore_session_state(self):
        """💾 이전 세션 상태 복원"""
        try:
            current_status = self.status_manager.current_status
            
            # 다운로드된 파일 목록 복원
            download_status = current_status.get('download_status')
            if download_status and download_status.get('file_list'):
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
            
            # 청크 상태 복원
            chunk_status = current_status.get('chunk_status')
            if chunk_status:
                self.state.processed_chunks = chunk_status.get('total_chunks', 0)
                print(f"🧩 Previous session had {self.state.processed_chunks} chunks")
            
            # 벡터 상태 복원
            vector_status = current_status.get('vector_status')
            if vector_status:
                self.state.vector_store_ready = vector_status.get('is_ready', False)
                if self.state.vector_store_ready:
                    print(f"🗄️ Vector database was ready - RAG is available!")
            
        except Exception as e:
            print(f"⚠️ Could not fully restore previous session: {e}")

    def _is_document_command(self, user_input: str) -> bool:
        """🔍 Check if user input is a document processing command"""
        user_input_lower = user_input.lower()
        
        # Check for explicit command keywords
        command_keywords = [
            'download', 'scan', 'fetch', 'get documents', 'get files',
            'chunk', 'process files', 'split documents', 'break into pieces',
            'index', 'build vector', 'create embeddings', 'make searchable',
            'status', 'progress', 'show files', 'check status',
            'help', 'commands', 'what can you do'
        ]
        
        return any(keyword in user_input_lower for keyword in command_keywords)

    def _is_debug_command(self, user_input: str) -> bool:
        """🔍 Check if user input is a RAG debug command"""
        debug_keywords = [
            'debug rag', 'show rag process', 'rag debug', 'debug process',
            'show process', 'explain rag', 'how rag works', 'rag analysis',
            'debug question', 'analyze question', 'rag step'
        ]
        
        user_input_lower = user_input.lower()
        return any(keyword in user_input_lower for keyword in debug_keywords)

    async def chat(self, user_input: str) -> str:
        """💬 Main chat interface with RAG capabilities"""
        if not user_input.strip():
            return "🤖 I'm here to help! Ask me questions about your documents or give me commands to process new ones."
        
        user_input = user_input.strip()
        
        # Store conversation
        self.conversation_history.append({
            'user': user_input,
            'timestamp': time.time()
        })

        # 🔍 Check if this is a RAG debug command
        if self._is_debug_command(user_input):
            print("🔍 Processing as RAG debug command...")
            response = await self._handle_debug_command(user_input)
        
        # Check if this is a document processing command
        elif self._is_document_command(user_input):
            print("🔧 Processing as document command...")
            response = await self._handle_document_command(user_input)
        
        # Check if vector database is ready for RAG
        elif self.state.vector_store_ready and self.rag_engine:
            print("🧠 Processing with RAG engine...")
            rag_response = await self.rag_engine.generate_rag_response(user_input)
            
            if rag_response:
                response = rag_response
            else:
                # Fall back to simple vector search
                print("🔍 Falling back to simple vector search...")
                response = await self._handle_simple_search(user_input)
        
        # Vector database not ready
        else:
            response = ("🤖 I'd love to answer your question, but I need to have documents indexed first! "
                       "Try asking me to 'download documents' or 'check status' to see what's available.")
        
        # Store response
        self.conversation_history[-1]['assistant'] = response
        
        return response

    async def _handle_debug_command(self, user_input: str) -> str:
        """🔍 Handle RAG debugging commands"""
        
        if not RAG_DEBUG_AVAILABLE:
            return ("🔍 RAG debugging feature is not available! \n\n"
                   "💡 To enable RAG debugging:\n"
                   "   1. Make sure debug_rag.py is in the same directory\n"
                   "   2. Install required dependencies\n"
                   "   3. Restart the application")
        
        if not self.rag_debugger:
            return ("🔍 RAG debugger is not initialized! \n\n"
                   "💡 Please make sure:\n"
                   "   • Vector database is available\n"
                   "   • Query engine is initialized\n"
                   "   • Try restarting the application")
        
        user_lower = user_input.lower()
        
        # Extract question from debug command
        question_to_debug = None
        
        # Pattern: "debug rag: question"
        if ':' in user_input:
            question_to_debug = user_input.split(':', 1)[1].strip()
        
        # Pattern: "show rag process for question"
        elif 'for' in user_lower:
            parts = user_input.split('for', 1)
            if len(parts) > 1:
                question_to_debug = parts[1].strip()
        
        # Default demo questions
        if not question_to_debug:
            demo_questions = [
                "IPCDA가 뭐야?",
                "How to configure database performance?",
                "Altibase troubleshooting guide"
            ]
            
            return f"""🔍 **RAG 디버깅 데모를 실행할게요!**

사용법:
• "debug rag: IPCDA가 뭐야?" - 특정 질문 디버깅
• "show rag process for database configuration" - 특정 주제 분석
• "rag analysis" - 데모 실행

🎯 데모 질문으로 분석을 시작합니다...

{await self._run_rag_debug_demo(demo_questions)}"""
        
        # Debug specific question
        try:
            print(f"🔍 Starting RAG debug for question: {question_to_debug}")
            debug_result = await self.rag_debugger.debug_rag_process(question_to_debug, show_details=True)
            
            if debug_result.success:
                return f"""🎉 **RAG 디버깅 완료!**

❓ **분석된 질문**: {question_to_debug}
⏱️ **총 처리 시간**: {debug_result.total_processing_time:.3f}초
✅ **처리 상태**: 성공
📊 **처리 단계**: {len(debug_result.steps)}단계

🔍 **상세 분석 결과는 콘솔에서 확인하세요!**

💡 **팁**: 
• 더 자세한 분석을 위해 "debug rag: 다른 질문" 시도해보세요
• 여러 질문 비교는 console에서 debug_multiple_questions() 사용"""
            else:
                return f"""❌ **RAG 디버깅 실패**

❓ **질문**: {question_to_debug}
🚨 **오류**: {debug_result.error_message}
⏱️ **처리 시간**: {debug_result.total_processing_time:.3f}초

💡 **해결 방법**:
• 벡터 데이터베이스가 준비되었는지 확인
• 문서가 인덱싱되었는지 확인
• OpenAI API 키가 설정되었는지 확인"""
        
        except Exception as e:
            return f"""❌ **RAG 디버깅 중 오류 발생**

🚨 **오류**: {str(e)}

💡 **해결 방법**:
• 시스템 상태 확인: 'status' 명령 실행
• 필요한 구성 요소 확인
• 문제가 지속되면 재시작 시도"""

    async def _run_rag_debug_demo(self, questions: List[str]) -> str:
        """🎯 RAG 디버깅 데모 실행"""
        
        try:
            # 첫 번째 질문으로 간단한 데모
            demo_question = questions[0]
            
            print(f"\n🎯 RAG 디버깅 데모 - 질문: {demo_question}")
            debug_result = await self.rag_debugger.debug_rag_process(demo_question, show_details=False)
            
            demo_summary = f"""📊 **데모 결과 요약**:

❓ 질문: {demo_question}
⏱️ 처리 시간: {debug_result.total_processing_time:.3f}초
📋 처리 단계: {len(debug_result.steps)}단계
✅ 성공 여부: {'성공' if debug_result.success else '실패'}

🔍 **주요 처리 단계**:"""
            
            for step in debug_result.steps:
                demo_summary += f"\n   {step.step_number}. {step.step_name} ({step.processing_time:.3f}초)"
            
            demo_summary += f"""

💡 **콘솔에서 상세 분석 결과를 확인하세요!**

🎯 **더 많은 디버깅 옵션**:
• "debug rag: 당신의 질문" - 특정 질문 분석
• "show rag process for 주제" - 주제별 분석
• console에서 debug_multiple_questions() - 여러 질문 비교"""
            
            return demo_summary
        
        except Exception as e:
            return f"❌ 데모 실행 중 오류: {str(e)}"

    async def _handle_document_command(self, user_input: str) -> str:
        """Handle document processing commands (existing functionality)"""
        # Use simplified intent detection for commands
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['download', 'scan', 'fetch', 'get']):
            if any(word in user_lower for word in ['all', 'everything']):
                return await self._handle_download_all()
            elif 'pdf' in user_lower:
                return await self._handle_download_pdfs()
            else:
                return await self._handle_download_from_url(user_input)
        
        elif any(word in user_lower for word in ['chunk', 'process', 'split']):
            return await self._handle_chunk_files()
        
        elif any(word in user_lower for word in ['index', 'vector', 'embed', 'searchable']):
            return await self._handle_index_files()
        
        elif any(word in user_lower for word in ['status', 'progress', 'files']):
            if 'quick' in user_lower or 'brief' in user_lower:
                return self.status_manager.get_quick_status()
            else:
                return self.status_manager.get_comprehensive_status()
        
        elif any(word in user_lower for word in ['help', 'commands', 'what can']):
            return self._get_help_message()
        
        else:
            return ("🤖 I understand you want to do something with documents, but I'm not sure what. "
                   "Try: 'download from URL', 'process files', 'index documents', or 'show status'")

    async def _handle_simple_search(self, user_input: str) -> str:
        """Handle simple vector search when RAG doesn't apply"""
        if not self.query_engine:
            return "❌ Search functionality is not available."
        
        try:
            results = await self.query_engine.search(user_input, k=3)
            
            if not results:
                return f"🤖 I couldn't find specific information about '{user_input}'. Try rephrasing your question!"
            
            # Simple formatted response
            response = f"🔍 **Found information about '{user_input}':**\n\n"
            
            for i, result in enumerate(results, 1):
                payload = result.get('payload', {})
                score = result.get('score', 0)
                
                response += f"**📋 Result {i}** (Relevance: {score:.1%})\n"
                response += f"📖 Source: {payload.get('source_file', 'Unknown')}\n"
                
                content = payload.get('text', '')[:300]
                if len(payload.get('text', '')) > 300:
                    content += "..."
                response += f"📄 Content: {content}\n"
                response += "─" * 40 + "\n\n"
            
            response += "💡 Need more detailed analysis? Ask a more specific question!"
            return response
            
        except Exception as e:
            return f"❌ Search failed: {str(e)}"

    # [여기에 기존의 다른 메서드들이 계속됩니다...]
    # _handle_download_from_url, _handle_download_all, 등등...
    
    def _get_help_message(self) -> str:
        """Get help message"""
        debug_help = ""
        if RAG_DEBUG_AVAILABLE:
            debug_help = """
🔍 **RAG 디버깅 & 분석**
   • "debug rag: IPCDA가 뭐야?" - 특정 질문의 RAG 프로세스 분석
   • "show rag process for database" - 주제별 RAG 분석
   • "rag analysis" - RAG 프로세스 데모"""
        
        return f"""🤖 **I'm your intelligent document assistant!** Here's what I can do:

📥 **Document Management**
   • "Download PDFs from https://example.com/docs"
   • "Get all documents from that GitHub repo"
   • "Process the downloaded files"
   • "Index documents for search"

🧠 **Intelligent Q&A (when documents are indexed)**
   • "What is IPCDA and how does it work?"
   • "How to configure database performance?"
   • "Troubleshoot connection timeout errors"
   • "Best practices for backup procedures"
   • "Explain the difference between memory and disk databases"
{debug_help}
📊 **Status & Information**
   • "What's the current status?"
   • "Show quick status"
   • "How many files are indexed?"

💡 **Just talk naturally!** I can understand:
   - Complex technical questions
   - Step-by-step guidance requests  
   - Troubleshooting help
   - Definitions and explanations
   - Comparison questions

🎯 **Current capabilities**: """ + (
    "✅ RAG-powered intelligent answers available!" if self.state.vector_store_ready 
    else "⏳ Ready for document processing - index some documents to unlock intelligent Q&A!"
) + (
    " 🔍 RAG debugging enabled!" if RAG_DEBUG_AVAILABLE else ""
)

    # [기존의 다른 메서드들을 여기에 추가...]


async def main():
    """🎭 Main interactive chat loop with RAG"""
    print("🌟" + "=" * 70 + "🌟")
    print("      🤖 Intelligent Document Assistant with RAG")
    if RAG_DEBUG_AVAILABLE:
        print("                🔍 RAG Debugging Enabled")
    print("🌟" + "=" * 70 + "🌟")
    print()
    print("💬 Hi! I'm your intelligent document processing assistant!")
    print("   I can download, process, index documents AND answer complex questions!")
    print()
    print("🎯 Try asking me:")
    print("   • Complex questions: 'What is IPCDA and how does it work?'")
    print("   • How-to questions: 'How to configure database performance?'")
    print("   • Troubleshooting: 'Fix connection timeout errors'")
    print("   • Document commands: 'Download PDFs from [URL]'")
    if RAG_DEBUG_AVAILABLE:
        print("   • RAG debugging: 'debug rag: your question here'")
    print()
    print("   Type 'quit' to exit")
    print("=" * 78)
    
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