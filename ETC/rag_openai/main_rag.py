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

    def _create_structured_response(self, question: str, context: str, question_type: str, 
                                  main_topic: str, results: List[Dict]) -> str:
        """Create a structured response based on the context (simulated LLM response)"""
        
        response = f"🎯 **Based on the documentation about {main_topic}:**\n\n"
        
        # Extract key information from context
        if question_type == 'what_is':
            response += f"📖 **Definition and Overview:**\n"
            response += self._extract_definitions(context)
            response += f"\n\n🔧 **Key Features:**\n"
            response += self._extract_features(context)
            
        elif question_type == 'how_to':
            response += f"📋 **Step-by-Step Guide:**\n"
            response += self._extract_procedures(context)
            response += f"\n\n⚠️ **Important Notes:**\n"
            response += self._extract_warnings(context)
            
        elif question_type == 'troubleshoot':
            response += f"🔍 **Problem Analysis:**\n"
            response += self._extract_problem_info(context)
            response += f"\n\n💡 **Solutions:**\n"
            response += self._extract_solutions(context)
            
        else:  # general
            response += f"📄 **Information from Documentation:**\n"
            response += self._extract_relevant_info(context)
        
        # Add sources
        response += f"\n\n📚 **Sources Referenced:**\n"
        sources = set()
        for result in results[:5]:  # Top 5 sources
            source = result.get('payload', {}).get('source_file', 'Unknown')
            if source != 'Unknown':
                sources.add(source)
        
        for i, source in enumerate(sources, 1):
            response += f"   {i}. {source}\n"
        
        # Add follow-up suggestions
        response += f"\n💡 **Need more details?** Try asking:\n"
        response += self._generate_follow_up_questions(main_topic, question_type)
        
        return response

    def _extract_definitions(self, context: str) -> str:
        """Extract definition-like content"""
        try:
            sentences = sent_tokenize(context)
            definitions = []
            
            for sentence in sentences[:5]:  # First 5 sentences
                if any(word in sentence.lower() for word in ['is', 'are', 'means', 'refers', 'defines']):
                    definitions.append(f"• {sentence.strip()}")
            
            return '\n'.join(definitions) if definitions else "• Based on the documentation context provided above."
        except Exception as e:
            print(f"⚠️ Definition extraction failed: {e}")
            return "• Based on the documentation context provided above."

    def _extract_features(self, context: str) -> str:
        """Extract feature-like content"""
        try:
            sentences = sent_tokenize(context)
            features = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['feature', 'capability', 'function', 'support', 'provide']):
                    features.append(f"• {sentence.strip()}")
                    if len(features) >= 3:
                        break
            
            return '\n'.join(features) if features else "• Detailed features are available in the source documentation."
        except Exception as e:
            print(f"⚠️ Feature extraction failed: {e}")
            return "• Detailed features are available in the source documentation."

    def _extract_procedures(self, context: str) -> str:
        """Extract procedural content"""
        try:
            sentences = sent_tokenize(context)
            procedures = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['step', 'first', 'then', 'next', 'configure', 'set', 'execute']):
                    procedures.append(f"• {sentence.strip()}")
                    if len(procedures) >= 5:
                        break
            
            return '\n'.join(procedures) if procedures else "• Please refer to the detailed procedures in the source documentation."
        except Exception as e:
            print(f"⚠️ Procedure extraction failed: {e}")
            return "• Please refer to the detailed procedures in the source documentation."

    def _extract_warnings(self, context: str) -> str:
        """Extract warning or important note content"""
        try:
            sentences = sent_tokenize(context)
            warnings = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['warning', 'caution', 'important', 'note', 'careful', 'ensure']):
                    warnings.append(f"• {sentence.strip()}")
                    if len(warnings) >= 3:
                        break
            
            return '\n'.join(warnings) if warnings else "• Follow standard best practices as outlined in the documentation."
        except Exception as e:
            print(f"⚠️ Warning extraction failed: {e}")
            return "• Follow standard best practices as outlined in the documentation."

    def _extract_problem_info(self, context: str) -> str:
        """Extract problem-related information"""
        try:
            sentences = sent_tokenize(context)
            problems = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['error', 'problem', 'issue', 'fail', 'cannot', 'unable']):
                    problems.append(f"• {sentence.strip()}")
                    if len(problems) >= 3:
                        break
            
            return '\n'.join(problems) if problems else "• Problem analysis based on documentation context."
        except Exception as e:
            print(f"⚠️ Problem extraction failed: {e}")
            return "• Problem analysis based on documentation context."

    def _extract_solutions(self, context: str) -> str:
        """Extract solution-related information"""
        try:
            sentences = sent_tokenize(context)
            solutions = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['solution', 'resolve', 'fix', 'correct', 'adjust', 'modify']):
                    solutions.append(f"• {sentence.strip()}")
                    if len(solutions) >= 3:
                        break
            
            return '\n'.join(solutions) if solutions else "• Solutions can be found in the detailed documentation."
        except Exception as e:
            print(f"⚠️ Solution extraction failed: {e}")
            return "• Solutions can be found in the detailed documentation."

    def _extract_relevant_info(self, context: str) -> str:
        """Extract generally relevant information"""
        try:
            sentences = sent_tokenize(context)
            info = []
            
            # Take first few meaningful sentences
            for sentence in sentences[:4]:
                if len(sentence.strip()) > 50:  # Meaningful sentences
                    info.append(f"• {sentence.strip()}")
            
            return '\n'.join(info) if info else "• Information extracted from the documentation context."
        except Exception as e:
            print(f"⚠️ Info extraction failed: {e}")
            return "• Information extracted from the documentation context."

    def _generate_follow_up_questions(self, topic: str, question_type: str) -> str:
        """Generate relevant follow-up questions"""
        
        if question_type == 'what_is':
            return f"   • How to configure {topic}?\n   • What are the best practices for {topic}?\n   • Common issues with {topic}?"
        elif question_type == 'how_to':
            return f"   • What are common errors when working with {topic}?\n   • Best practices for {topic}?\n   • Advanced configuration of {topic}?"
        elif question_type == 'troubleshoot':
            return f"   • How to prevent {topic} issues?\n   • What is {topic}?\n   • Configuration guide for {topic}?"
        else:
            return f"   • More details about {topic}?\n   • How to implement {topic}?\n   • Troubleshooting {topic}?"


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

        # Check if this is a document processing command
        if self._is_document_command(user_input):
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

    async def _handle_download_from_url(self, user_input: str) -> str:
        """Handle URL download requests"""
        # Extract URL from input
        url_match = re.search(r'https?://[^\s]+', user_input)
        if url_match:
            url = url_match.group()
        else:
            return ("🤖 I'd love to help you download documents! Please provide a URL. "
                   "For example: 'Download documents from https://example.com/docs'")
        
        self.state.current_url = url
        
        try:
            print(f"🔍 Analyzing URL: {url}")
            documents = await self.downloader.scan_documents(url)
            self.state.discovered_documents = documents
            
            if not documents:
                self.status_manager.update_url_status(url=url, documents_found=0, total_size_mb=0, status='failed')
                return f"🤖 I couldn't find any documents at {url}. Could you check the URL?"
            
            # Update status
            total_size_mb = sum(doc.size or 0 for doc in documents) / (1024 * 1024)
            self.status_manager.update_url_status(
                url=url, documents_found=len(documents), total_size_mb=total_size_mb, status='scanned'
            )
            
            doc_summary = self.downloader.summarize_findings(documents)
            
            response = f"🎉 Great! I found documents at {url}:\n\n{doc_summary}\n\n"
            response += "💡 What would you like to do?\n"
            response += "• 'Download all PDFs'\n"
            response += "• 'Download everything'\n"
            response += "• 'Show me more details first'"
            
            return response
            
        except Exception as e:
            self.status_manager.update_url_status(url=url, documents_found=0, total_size_mb=0, status='failed')
            return f"❌ Oops! I had trouble accessing {url}. Error: {str(e)}"

    async def _handle_download_all(self) -> str:
        """Handle download all requests"""
        if not self.state.discovered_documents:
            return ("🤖 I don't see any discovered documents to download! "
                   "Please scan a URL first with something like 'download from https://example.com'")
        
        try:
            downloaded_files = await self.downloader.download_documents()
            self.state.downloaded_files.extend(downloaded_files)
            
            stats = self.downloader.get_download_stats()
            
            # Update status
            absolute_file_paths = [os.path.abspath(f) for f in downloaded_files]
            self.status_manager.update_download_status(
                total_files=stats['total_found'],
                downloaded_files=stats['downloaded'],
                failed_files=stats['failed'],
                total_size_mb=stats['bytes_downloaded'] / (1024 * 1024),
                download_directory=str(os.path.abspath(self.downloader.download_dir)),
                file_list=absolute_file_paths
            )
            
            response = f"✅ Download complete!\n\n"
            response += f"📥 Downloaded: {stats['downloaded']} files\n"
            response += f"❌ Failed: {stats['failed']} files\n"
            if stats['bytes_downloaded'] > 0:
                mb_downloaded = stats['bytes_downloaded'] / (1024 * 1024)
                response += f"💾 Total size: {mb_downloaded:.1f} MB\n"
            
            response += f"📁 Files saved to: {self.downloader.download_dir}\n\n"
            response += "💡 Next step: 'Process the files' to chunk them for searching!"
            
            return response
            
        except Exception as e:
            return f"❌ Download failed. Error: {str(e)}"

    async def _handle_download_pdfs(self) -> str:
        """Handle PDF-only download requests"""
        if not self.state.discovered_documents:
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
            response += "💡 Ready to process these files for intelligent search!"
            
            return response
            
        except Exception as e:
            return f"❌ PDF download failed. Error: {str(e)}"

    async def _handle_chunk_files(self) -> str:
        """Handle file chunking requests"""
        if not self.chunker:
            return "❌ Document chunker is not available. Please check if smart_chunker.py is accessible."
        
        if not self.state.downloaded_files:
            return ("🤖 I don't see any downloaded files to process yet! "
                   "Would you like me to download some documents first?")
        
        try:
            print(f"🧩 Processing {len(self.state.downloaded_files)} files...")
            
            chunks = await self.chunker.process_files(
                self.state.downloaded_files,
                chunk_size=1000,
                overlap=200,
                use_semantic_splitting=True,
                preserve_structure=True
            )
            
            self.state.processed_chunks = len(chunks)
            
            # Update status
            processing_stats = self.chunker.get_processing_stats()
            self.status_manager.update_chunk_status(
                total_chunks=len(chunks),
                total_characters=processing_stats.get('total_characters', 0),
                files_processed=processing_stats.get('files_processed', 0),
                processing_errors=processing_stats.get('errors', []),
                chunk_size=1000,
                overlap=200,
                semantic_chunking=True
            )
            
            response = f"✅ Perfect! I've processed your documents:\n\n"
            response += f"📄 Files processed: {len(self.state.downloaded_files)}\n"
            response += f"🧩 Chunks created: {self.state.processed_chunks:,}\n"
            response += f"⚙️ Chunk size: ~1000 characters with 200 character overlap\n"
            response += f"🧠 Semantic chunking: Enabled\n\n"
            response += "🎯 Ready to index for intelligent search!"
            
            return response
            
        except Exception as e:
            return f"❌ I had trouble processing the documents. Error: {str(e)}"

    async def _handle_index_files(self) -> str:
        """Handle vector indexing requests"""
        if self.state.processed_chunks == 0:
            return ("🤖 I need some processed documents to index! "
                   "Would you like me to process your downloaded files first?")
        
        if not self.vector_manager:
            return ("❌ Vector indexing is not available. Please install required packages:\n"
                   "pip install sentence-transformers qdrant-client torch")
        
        try:
            print(f"🗄️ Indexing {self.state.processed_chunks} chunks...")
            
            chunks = self.chunker.get_chunks()
            if not chunks:
                return "🤖 I couldn't find the processed chunks. Please run processing again."
            
            success = await self.vector_manager.build_index(chunks)
            
            if success:
                self.state.vector_store_ready = True
                
                collection_info = self.vector_manager.get_collection_info()
                self.status_manager.update_vector_status(
                    is_ready=True,
                    collection_name=collection_info.get('name', 'conversation_docs'),
                    vector_count=collection_info.get('points_count', len(chunks)),
                    vector_dimensions=self.vector_manager.vector_size,
                    index_size_mb=len(chunks) * 0.001,
                    embedding_model=self.vector_manager.model_name,
                    search_capabilities=["semantic_search", "similarity_search", "RAG_enabled"]
                )
                
                response = f"🎉 Vector indexing completed successfully!\n\n"
                response += f"🗄️ Vector database: Ready with RAG capabilities\n"
                response += f"📊 Indexed chunks: {len(chunks):,}\n"
                response += f"🔢 Vector dimensions: {self.vector_manager.vector_size}\n"
                response += f"🤖 Model: {self.vector_manager.model_name}\n\n"
                response += "🎯 **You can now ask me intelligent questions about your documents!**\n"
                response += "💡 Try: 'What is IPCDA?', 'How to configure database?', 'Troubleshoot connection issues?'"
                
                return response
            else:
                return "❌ Vector indexing failed. Please check the logs for details."
            
        except Exception as e:
            return f"❌ Indexing failed. Error: {str(e)}"

    def _get_help_message(self) -> str:
        """Get help message"""
        return """🤖 **I'm your intelligent document assistant!** Here's what I can do:

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
)


async def main():
    """🎭 Main interactive chat loop with RAG"""
    print("🌟" + "=" * 70 + "🌟")
    print("      🤖 Intelligent Document Assistant with RAG")
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