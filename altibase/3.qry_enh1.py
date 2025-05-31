#!/usr/bin/env python3
"""
Standalone Altibase Manual Chatbot
Connects to existing Qdrant vector store and provides enhanced search with chatbot-style responses
"""

import os
import json
import numpy as np
import re
import time
import logging
import traceback
import hashlib
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import PointStruct, Distance, VectorParams, UpdateStatus
except ImportError as e:
    print(f"❌ Missing required packages. Please install them:")
    print("pip install sentence-transformers qdrant-client")
    exit(1)

class QueryProcessor:
    """Processes user queries to extract topics, keywords, and intent."""
    
    def __init__(self):
        # Technical terms and database concepts
        self.db_terms = {
            'altibase', 'database', 'dbms', 'sql', 'table', 'index', 'query', 
            'transaction', 'schema', 'column', 'row', 'primary key', 'foreign key',
            'constraint', 'trigger', 'procedure', 'function', 'view', 'sequence',
            'backup', 'restore', 'replication', 'partition', 'tablespace',
            'memory', 'disk', 'performance', 'optimization', 'tuning', 'create',
            'insert', 'update', 'delete', 'select', 'alter', 'drop', 'grant',
            'revoke', 'commit', 'rollback', 'connect', 'disconnect'
        }
        
        # Question patterns and their intent
        self.question_patterns = {
            'definition': [r'what is', r'무엇인가', r'정의', r'define', r'explain', r'meaning'],
            'how_to': [r'how to', r'어떻게', r'방법', r'how can', r'how do', r'how should'],
            'troubleshooting': [r'error', r'문제', r'오류', r'trouble', r'fix', r'solve', r'problem'],
            'configuration': [r'config', r'설정', r'setting', r'configure', r'setup'],
            'performance': [r'performance', r'성능', r'optimize', r'tuning', r'slow', r'fast', r'speed'],
            'installation': [r'install', r'설치', r'setup', r'deployment', r'deploy'],
            'comparison': [r'difference', r'compare', r'vs', r'versus', r'차이', r'비교'],
            'examples': [r'example', r'예제', r'sample', r'demo', r'show me']
        }
        
        # Stop words (Korean + English)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might',
            '을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로', '와', '과',
            '입니다', '있습니다', '합니다', '했습니다', '하는', '하기', '된다', '되는'
        }

    def extract_query_components(self, query: str) -> Dict[str, any]:
        """Extract topics, keywords, and intent from user query."""
        query_lower = query.lower()
        
        # 1. Detect question intent
        intent = self._detect_intent(query_lower)
        
        # 2. Extract technical keywords
        technical_keywords = self._extract_technical_keywords(query_lower)
        
        # 3. Extract general keywords
        general_keywords = self._extract_general_keywords(query)
        
        # 4. Generate topic variations
        topic_variations = self._generate_topic_variations(query, technical_keywords, intent)
        
        # 5. Create search strategies
        search_strategies = self._create_search_strategies(
            query, intent, technical_keywords, general_keywords, topic_variations
        )
        
        return {
            'original_query': query,
            'intent': intent,
            'technical_keywords': technical_keywords,
            'general_keywords': general_keywords,
            'topic_variations': topic_variations,
            'search_strategies': search_strategies
        }

    def _detect_intent(self, query: str) -> str:
        """Detect the intent/type of the question."""
        for intent, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent
        return 'general'

    def _extract_technical_keywords(self, query: str) -> List[str]:
        """Extract database and technical terms from query."""
        found_terms = []
        words = re.findall(r'\b\w+\b', query)
        
        for word in words:
            if word in self.db_terms:
                found_terms.append(word)
                
        # Check for multi-word terms
        for term in self.db_terms:
            if ' ' in term and term in query:
                found_terms.append(term)
                
        return list(set(found_terms))

    def _extract_general_keywords(self, query: str) -> List[str]:
        """Extract general keywords excluding stop words."""
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        return list(set(keywords))

    def _generate_topic_variations(self, query: str, technical_keywords: List[str], intent: str) -> List[str]:
        """Generate different ways to express the same topic."""
        variations = [query]  # Original query
        
        # Altibase-specific variations
        if 'altibase' in query.lower():
            variations.extend([
                "Altibase database management system",
                "Altibase DBMS features",
                "Altibase database overview",
                "Altibase introduction",
                "Altibase 데이터베이스"
            ])
        
        # Intent-based variations
        if intent == 'definition' and technical_keywords:
            for keyword in technical_keywords[:2]:
                variations.extend([
                    f"{keyword} definition explanation",
                    f"{keyword} concept overview",
                    f"{keyword} 개념 설명"
                ])
        
        elif intent == 'how_to' and technical_keywords:
            for keyword in technical_keywords[:2]:
                variations.extend([
                    f"{keyword} tutorial guide",
                    f"{keyword} step by step",
                    f"{keyword} 사용법 방법"
                ])
        
        elif intent == 'examples' and technical_keywords:
            for keyword in technical_keywords[:2]:
                variations.extend([
                    f"{keyword} example sample",
                    f"{keyword} code example",
                    f"{keyword} 예제"
                ])
        
        return variations[:8]  # Limit variations

    def _create_search_strategies(self, query: str, intent: str, 
                                technical_keywords: List[str], 
                                general_keywords: List[str],
                                topic_variations: List[str]) -> List[Dict]:
        """Create multiple search strategies for comprehensive retrieval."""
        strategies = []
        
        # Strategy 1: Original semantic search
        strategies.append({
            'type': 'semantic',
            'query': query,
            'weight': 1.0,
            'description': 'Original semantic similarity'
        })
        
        # Strategy 2: Technical keyword searches
        if technical_keywords:
            for i, keyword in enumerate(technical_keywords[:2]):
                strategies.append({
                    'type': 'technical_keyword',
                    'query': keyword,
                    'weight': 0.8 - (i * 0.1),
                    'description': f'Technical keyword: {keyword}'
                })
        
        # Strategy 3: Topic variation searches
        for i, variation in enumerate(topic_variations[1:4]):  # Skip original
            strategies.append({
                'type': 'topic_variation',
                'query': variation,
                'weight': 0.6 - (i * 0.1),
                'description': f'Topic variation'
            })
        
        # Strategy 4: Combined keyword search
        if len(general_keywords) > 1:
            combined_query = ' '.join(general_keywords[:3])
            strategies.append({
                'type': 'combined_keywords',
                'query': combined_query,
                'weight': 0.7,
                'description': 'Combined keywords'
            })
        
        return strategies

class AltibaseChatbot:
    """Main chatbot class that connects to existing Qdrant vector store."""
    
    def __init__(self, qdrant_path: str = "./qdrant_vector", 
                 collection_name: str = "altibase_manuals",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """Initialize the chatbot with vector store connection."""
        
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.query_processor = QueryProcessor()
        
        print("🤖 Initializing Altibase Manual Chatbot...")
        
        # Initialize components
        self._init_vectorizer(model_name)
        self._init_vector_db()
        self._check_database_status()
        
        print("✅ Chatbot ready!")

    def _init_vectorizer(self, model_name: str):
        """Initialize the sentence transformer model."""
        print(f"📖 Loading vectorizer model: {model_name}")
        try:
            self.vectorizer = SentenceTransformer(model_name)
            self.vector_size = self.vectorizer.get_sentence_embedding_dimension()
            print(f"✅ Vectorizer loaded. Vector size: {self.vector_size}")
        except Exception as e:
            print(f"❌ Failed to load vectorizer: {e}")
            raise

    def _init_vector_db(self):
        """Initialize connection to existing Qdrant vector database."""
        print(f"🗄️  Connecting to vector database: {self.qdrant_path}")
        try:
            self.client = QdrantClient(path=self.qdrant_path)
            print(f"✅ Connected to Qdrant database")
        except Exception as e:
            print(f"❌ Failed to connect to vector database: {e}")
            print(f"   Make sure the database exists at: {self.qdrant_path}")
            raise

    def _check_database_status(self):
        """Check if the database has data."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                print(f"❌ Collection '{self.collection_name}' not found!")
                print(f"   Available collections: {collection_names}")
                raise ValueError(f"Collection not found")
            
            collection_info = self.client.get_collection(self.collection_name)
            self.total_documents = collection_info.points_count
            
            if self.total_documents == 0:
                print(f"⚠️  Collection '{self.collection_name}' is empty!")
                print(f"   Please build the vector database first.")
            else:
                print(f"📊 Found {self.total_documents:,} documents in the database")
                
        except Exception as e:
            print(f"❌ Error checking database status: {e}")
            raise

    def vectorize_query(self, query: str) -> np.ndarray:
        """Convert query text to vector."""
        try:
            vector = self.vectorizer.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            return vector[0].astype('float32')
        except Exception as e:
            logger.error(f"Error vectorizing query: {e}")
            return None

    def search_vector_db(self, query: str, k: int = 5, 
                        filter_payload: Optional[Dict] = None) -> List[Dict]:
        """Search the vector database."""
        query_vector = self.vectorize_query(query)
        if query_vector is None:
            return []
        
        try:
            search_filter = None
            if filter_payload:
                must_conditions = []
                for key, value in filter_payload.items():
                    if isinstance(value, bool):
                        must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
                    elif isinstance(value, (int, float)):
                        must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
                    else:
                        must_conditions.append(models.FieldCondition(key=key, match=models.MatchText(text=str(value))))
                
                if must_conditions:
                    search_filter = models.Filter(must=must_conditions)

            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=search_filter,
                limit=k,
                with_payload=True 
            )
            
            results = []
            for hit in hits:
                results.append({
                    'id': hit.id,
                    'score': hit.score, 
                    'payload': hit.payload 
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []

    def enhanced_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform enhanced search using multiple strategies."""
        if not query.strip():
            return []
        
        # Extract query components
        query_components = self.query_processor.extract_query_components(query)
        
        # Execute multiple search strategies
        all_results = []
        
        for strategy in query_components['search_strategies']:
            try:
                search_query = strategy['query']
                weight = strategy['weight']
                
                # Execute search
                results = self.search_vector_db(search_query, k=k)
                
                # Apply weight to scores
                for result in results:
                    result['score'] = result['score'] * weight
                    result['strategy'] = strategy['type']
                
                all_results.extend(results)
                
            except Exception as e:
                logger.warning(f"Strategy '{strategy.get('type')}' failed: {e}")
                continue
        
        # Deduplicate and re-rank
        final_results = self._deduplicate_and_rerank(all_results, k)
        
        # Add query analysis
        if final_results:
            final_results[0]['query_analysis'] = query_components
        
        return final_results

    def _deduplicate_and_rerank(self, all_results: List[Dict], k: int) -> List[Dict]:
        """Remove duplicates and re-rank results."""
        if not all_results:
            return []
        
        # Group by document ID
        result_groups = {}
        for result in all_results:
            doc_id = result['id']
            if doc_id not in result_groups:
                result_groups[doc_id] = {
                    'result': result,
                    'combined_score': 0,
                    'strategy_count': 0,
                    'strategies': []
                }
            
            group = result_groups[doc_id]
            group['combined_score'] += result['score']
            group['strategy_count'] += 1
            group['strategies'].append(result.get('strategy', 'unknown'))
        
        # Calculate final scores
        final_results = []
        for doc_id, group in result_groups.items():
            result = group['result'].copy()
            # Boost for multiple strategies
            boost_factor = 1 + (group['strategy_count'] - 1) * 0.2
            result['score'] = (group['combined_score'] / group['strategy_count']) * boost_factor
            result['found_by_strategies'] = list(set(group['strategies']))
            result['strategy_count'] = group['strategy_count']
            final_results.append(result)
        
        # Sort and return top k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:k]

    def format_response(self, query: str, results: List[Dict]) -> str:
        """Format search results into a chatbot-style response."""
        if not results:
            return ("🤖 I couldn't find specific information about your question in the Altibase manual. "
                   "Try rephrasing your question or asking about a different topic.")
        
        # Get query analysis if available
        analysis = results[0].get('query_analysis', {})
        intent = analysis.get('intent', 'general')
        technical_keywords = analysis.get('technical_keywords', [])
        
        # Build response based on intent
        response_parts = []
        
        # Opening based on intent
        if intent == 'definition':
            response_parts.append(f"🤖 Here's what I found about {' and '.join(technical_keywords) if technical_keywords else 'your question'}:")
        elif intent == 'how_to':
            response_parts.append(f"🤖 Here's how to {' '.join(technical_keywords) if technical_keywords else 'do what you asked'}:")
        elif intent == 'troubleshooting':
            response_parts.append(f"🤖 Here's information that might help solve your issue:")
        elif intent == 'examples':
            response_parts.append(f"🤖 Here are some examples I found:")
        else:
            response_parts.append(f"🤖 I found relevant information about your question:")
        
        response_parts.append("")
        
        # Add top results
        for i, result in enumerate(results[:3], 1):  # Show top 3 results
            payload = result.get('payload', {})
            score = result.get('score', 0)
            
            # Source information
            manual_name = payload.get('manual_name', 'Unknown Manual')
            page_num = payload.get('page_number', 'N/A')
            chapter = payload.get('chapter_title', '')
            section = payload.get('section_title', '')
            
            source_info = f"📖 {manual_name}"
            if page_num != 'N/A':
                source_info += f" (Page {page_num})"
            if chapter:
                source_info += f" - {chapter}"
            if section:
                source_info += f" > {section}"
            
            response_parts.append(f"**Result {i}** (Relevance: {score:.1%})")
            response_parts.append(source_info)
            response_parts.append("")
            
            # Content
            text_content = payload.get('text', '').strip()
            if text_content:
                # Limit content length for chat response
                if len(text_content) > 300:
                    text_content = text_content[:300] + "..."
                response_parts.append(text_content)
            else:
                response_parts.append("(No content available)")
            
            response_parts.append("")
            response_parts.append("─" * 50)
            response_parts.append("")
        
        # Footer
        response_parts.append("💡 **Tip**: Ask follow-up questions for more specific information!")
        
        return "\n".join(response_parts)

    def chat(self, query: str, k: int = 3) -> str:
        """Main chat interface - returns formatted response."""
        if not query.strip():
            return "🤖 Please ask me a question about Altibase!"
        
        print(f"🔍 Searching for: {query}")
        
        # Perform enhanced search
        start_time = time.time()
        results = self.enhanced_search(query, k=k)
        search_time = time.time() - start_time
        
        print(f"⏱️  Search completed in {search_time:.2f} seconds ({len(results)} results)")
        
        # Format and return response
        return self.format_response(query, results)

def main():
    """Main interactive chatbot interface."""
    print("🚀 Starting Altibase Manual Chatbot...")
    print("=" * 60)
    
    try:
        # Initialize chatbot
        chatbot = AltibaseChatbot()
        
        print("\n" + "=" * 60)
        print("💬 Altibase Manual Chat Assistant")
        print("=" * 60)
        print("Ask me anything about Altibase! Type 'quit' or 'exit' to stop.")
        print("Examples:")
        print("  • What is Altibase?")
        print("  • How to create a table?")
        print("  • Altibase performance tuning")
        print("  • SQL syntax examples")
        print("=" * 60)
        
        conversation_count = 0
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n💬 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print(f"\n🤖 Goodbye! I answered {conversation_count} questions for you today.")
                    break
                
                if not user_input:
                    continue
                
                # Get chatbot response
                print(f"\n🤖 Assistant:")
                response = chatbot.chat(user_input)
                print(response)
                
                conversation_count += 1
                
            except KeyboardInterrupt:
                print(f"\n\n🤖 Goodbye! I answered {conversation_count} questions for you today.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try asking your question again.")
    
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("\nPlease make sure:")
        print("1. The vector database exists at './qdrant_vector'")
        print("2. The collection 'altibase_manuals' has data")
        print("3. Required packages are installed: pip install sentence-transformers qdrant-client")

if __name__ == "__main__":
    main()