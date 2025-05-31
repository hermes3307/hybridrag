#!/usr/bin/env python3
"""
🔍 Conversational Query Engine
Advanced query processing and response generation
"""

import re
import time
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """🧠 Analyzes user queries to understand intent and extract entities"""
    
    def __init__(self):
        # Intent patterns
        self.intent_patterns = {
            'definition': [
                r'what is\s+(.+)',
                r'define\s+(.+)',
                r'meaning of\s+(.+)',
                r'explain\s+(.+)',
                r'(.+)\s+정의',
                r'(.+)이란\s*무엇',
            ],
            'how_to': [
                r'how to\s+(.+)',
                r'how can I\s+(.+)',
                r'how do I\s+(.+)',
                r'(.+)\s+방법',
                r'어떻게\s+(.+)',
                r'steps to\s+(.+)',
            ],
            'troubleshooting': [
                r'error\s+(.+)',
                r'problem with\s+(.+)',
                r'(.+)\s+not working',
                r'fix\s+(.+)',
                r'solve\s+(.+)',
                r'(.+)\s+오류',
                r'(.+)\s+문제',
            ],
            'examples': [
                r'example of\s+(.+)',
                r'show me\s+(.+)',
                r'sample\s+(.+)',
                r'(.+)\s+예제',
                r'(.+)\s+예시',
            ],
            'comparison': [
                r'difference between\s+(.+)\s+and\s+(.+)',
                r'(.+)\s+vs\s+(.+)',
                r'compare\s+(.+)',
                r'(.+)\s+차이',
                r'(.+)\s+비교',
            ],
            'best_practices': [
                r'best practices for\s+(.+)',
                r'recommended\s+(.+)',
                r'(.+)\s+best practices',
                r'(.+)\s+권장사항',
            ]
        }
        
        # Technical keywords for different domains
        self.domain_keywords = {
            'database': [
                'sql', 'table', 'index', 'query', 'database', 'schema', 'constraint',
                'transaction', 'backup', 'restore', 'replication', 'partition'
            ],
            'performance': [
                'performance', 'optimization', 'tuning', 'slow', 'fast', 'memory',
                'cpu', 'disk', 'cache', 'bottleneck', 'latency', 'throughput'
            ],
            'configuration': [
                'config', 'setting', 'parameter', 'option', 'configure', 'setup',
                'installation', 'deployment', 'environment'
            ],
            'security': [
                'security', 'authentication', 'authorization', 'permission', 'user',
                'role', 'privilege', 'encryption', 'ssl', 'certificate'
            ]
        }
        
        # Query expansion synonyms
        self.synonyms = {
            'database': ['db', 'dbms', 'data store', 'repository'],
            'performance': ['speed', 'efficiency', 'optimization', 'tuning'],
            'error': ['problem', 'issue', 'bug', 'failure', 'exception'],
            'configuration': ['config', 'setup', 'settings', 'parameters'],
            'installation': ['install', 'setup', 'deployment', 'configuration']
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """🔍 Analyze user query and extract structured information"""
        analysis = {
            'original_query': query,
            'intent': self._detect_intent(query),
            'entities': self._extract_entities(query),
            'domains': self._identify_domains(query),
            'keywords': self._extract_keywords(query),
            'expanded_query': self._expand_query(query),
            'complexity': self._assess_complexity(query),
            'language': self._detect_language(query)
        }
        
        return analysis

    def _detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query"""
        query_lower = query.lower().strip()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return 'general'

    def _extract_entities(self, query: str) -> List[str]:
        """Extract important entities from the query"""
        entities = []
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        # Extract capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized)
        
        # Extract technical terms
        words = re.findall(r'\b\w+\b', query.lower())
        for word in words:
            if len(word) > 3 and word in self._get_all_technical_terms():
                entities.append(word)
        
        return list(set(entities))

    def _identify_domains(self, query: str) -> List[str]:
        """Identify relevant technical domains"""
        query_lower = query.lower()
        domains = []
        
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)
        
        return domains

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords"""
        # Remove stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might',
            'what', 'how', 'when', 'where', 'why', 'who', 'which',
            '을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        expanded = query
        
        for term, synonyms in self.synonyms.items():
            if term in query.lower():
                # Add synonyms to help with search
                expanded += f" {' '.join(synonyms)}"
        
        return expanded

    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        has_operators = any(op in query.lower() for op in ['and', 'or', 'not', 'vs'])
        has_multiple_entities = len(self._extract_entities(query)) > 2
        
        if word_count > 15 or has_operators or has_multiple_entities:
            return 'complex'
        elif word_count > 8:
            return 'medium'
        else:
            return 'simple'

    def _detect_language(self, query: str) -> str:
        """Detect query language"""
        korean_chars = len(re.findall(r'[가-힣]', query))
        total_chars = len(re.findall(r'[a-zA-Z가-힣]', query))
        
        if total_chars == 0:
            return 'unknown'
        
        korean_ratio = korean_chars / total_chars
        
        if korean_ratio > 0.3:
            return 'korean'
        else:
            return 'english'

    def _get_all_technical_terms(self) -> set:
        """Get all technical terms across domains"""
        all_terms = set()
        for keywords in self.domain_keywords.values():
            all_terms.update(keywords)
        return all_terms


class ConversationalQueryEngine:
    """🗣️ Main conversational query engine"""
    
    def __init__(self, vector_manager):
        self.vector_manager = vector_manager
        self.analyzer = QueryAnalyzer()
        self.conversation_context = []
        self.user_preferences = {
            'response_style': 'detailed',  # brief, detailed, technical
            'show_sources': True,
            'max_results': 3,
            'prefer_examples': False
        }

    async def search(self, query: str, k: int = 3, 
                    use_context: bool = True) -> List[Dict]:
        """🔍 Perform conversational search with context awareness"""
        
        # Analyze the query
        analysis = self.analyzer.analyze_query(query)
        
        # Build search strategy
        search_strategies = self._build_search_strategies(analysis, k)
        
        # Execute multiple search strategies
        all_results = []
        
        for strategy in search_strategies:
            try:
                results = await self.vector_manager.search(
                    query=strategy['query'],
                    k=strategy.get('k', k),
                    filter_conditions=strategy.get('filters')
                )
                
                # Apply strategy weight to scores
                for result in results:
                    result['score'] *= strategy['weight']
                    result['strategy'] = strategy['type']
                
                all_results.extend(results)
                
            except Exception as e:
                logger.warning(f"Search strategy '{strategy['type']}' failed: {e}")
                continue
        
        # Deduplicate and re-rank
        final_results = self._deduplicate_and_rerank(all_results, k)
        
        # Add context information
        for result in final_results:
            result['query_analysis'] = analysis
        
        # Update conversation context
        if use_context:
            self.conversation_context.append({
                'query': query,
                'analysis': analysis,
                'results_count': len(final_results),
                'timestamp': time.time()
            })
            
            # Keep only recent context
            if len(self.conversation_context) > 10:
                self.conversation_context = self.conversation_context[-10:]
        
        return final_results

    def _build_search_strategies(self, analysis: Dict, k: int) -> List[Dict]:
        """🎯 Build multiple search strategies based on query analysis"""
        strategies = []
        
        # Strategy 1: Original semantic search
        strategies.append({
            'type': 'semantic',
            'query': analysis['original_query'],
            'weight': 1.0,
            'k': k
        })
        
        # Strategy 2: Expanded query search
        if analysis['expanded_query'] != analysis['original_query']:
            strategies.append({
                'type': 'expanded',
                'query': analysis['expanded_query'],
                'weight': 0.8,
                'k': k
            })
        
        # Strategy 3: Entity-focused search
        if analysis['entities']:
            entity_query = ' '.join(analysis['entities'][:3])
            strategies.append({
                'type': 'entity_focused',
                'query': entity_query,
                'weight': 0.7,
                'k': k
            })
        
        # Strategy 4: Domain-specific search with filters
        if analysis['domains']:
            primary_domain = analysis['domains'][0]
            
            # Add domain-specific filters
            if primary_domain == 'database':
                strategies.append({
                    'type': 'domain_database',
                    'query': analysis['original_query'],
                    'weight': 0.9,
                    'k': k,
                    'filters': {'source_type': 'PDF Document'}  # Assuming technical docs are PDFs
                })
            
            elif primary_domain == 'performance':
                strategies.append({
                    'type': 'domain_performance',
                    'query': analysis['original_query'],
                    'weight': 0.9,
                    'k': k,
                    'filters': {'has_code': True}  # Performance topics often have code
                })
        
        # Strategy 5: Intent-specific search
        intent = analysis['intent']
        if intent == 'examples':
            strategies.append({
                'type': 'examples',
                'query': f"example {analysis['original_query']}",
                'weight': 1.1,  # Boost for examples
                'k': k,
                'filters': {'has_code': True}
            })
        
        elif intent == 'how_to':
            strategies.append({
                'type': 'tutorial',
                'query': f"tutorial steps {analysis['original_query']}",
                'weight': 1.0,
                'k': k
            })
        
        return strategies

    def _deduplicate_and_rerank(self, all_results: List[Dict], k: int) -> List[Dict]:
        """🔄 Remove duplicates and re-rank results"""
        if not all_results:
            return []
        
        # Group by document ID
        result_groups = {}
        for result in all_results:
            doc_id = result['id']
            if doc_id not in result_groups:
                result_groups[doc_id] = {
                    'result': result,
                    'scores': [],
                    'strategies': []
                }
            
            result_groups[doc_id]['scores'].append(result['score'])
            result_groups[doc_id]['strategies'].append(result.get('strategy', 'unknown'))
        
        # Calculate combined scores
        final_results = []
        for doc_id, group in result_groups.items():
            result = group['result'].copy()
            
            # Use max score with bonus for multiple strategies
            max_score = max(group['scores'])
            strategy_bonus = len(set(group['strategies'])) * 0.1
            result['score'] = max_score + strategy_bonus
            
            # Add metadata about strategies
            result['found_by_strategies'] = list(set(group['strategies']))
            result['strategy_count'] = len(group['strategies'])
            
            final_results.append(result)
        
        # Sort by score and return top k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:k]

    def format_conversational_response(self, query: str, results: List[Dict]) -> str:
        """💬 Format results into a conversational response"""
        if not results:
            return self._generate_no_results_response(query)
        
        analysis = results[0].get('query_analysis', {})
        intent = analysis.get('intent', 'general')
        
        # Build response based on intent and user preferences
        response_parts = []
        
        # Opening based on intent
        opening = self._generate_opening(intent, analysis)
        response_parts.append(opening)
        response_parts.append("")
        
        # Add results
        for i, result in enumerate(results, 1):
            result_text = self._format_single_result(result, i, intent)
            response_parts.append(result_text)
            response_parts.append("")
        
        # Add closing suggestions
        closing = self._generate_closing(intent, analysis)
        response_parts.append(closing)
        
        return "\n".join(response_parts)

    def _generate_opening(self, intent: str, analysis: Dict) -> str:
        """Generate conversational opening"""
        entities = analysis.get('entities', [])
        domains = analysis.get('domains', [])
        
        if intent == 'definition':
            if entities:
                return f"🤖 I found information about {', '.join(entities[:2])}:"
            else:
                return "🤖 Here's what I found about your question:"
        
        elif intent == 'how_to':
            return "🤖 I found some guidance on how to do this:"
        
        elif intent == 'troubleshooting':
            return "🤖 Here's information that might help solve your issue:"
        
        elif intent == 'examples':
            return "🤖 I found some examples for you:"
        
        elif domains:
            domain_name = domains[0].replace('_', ' ').title()
            return f"🤖 I found relevant {domain_name} information:"
        
        else:
            return "🤖 Here's what I found:"

    def _format_single_result(self, result: Dict, index: int, intent: str) -> str:
        """Format a single search result"""
        payload = result.get('payload', {})
        score = result.get('score', 0)
        
        # Header
        result_text = f"**📋 Result {index}** (Relevance: {score:.1%})\n"
        
        # Source information
        source_file = payload.get('source_file', 'Unknown')
        source_type = payload.get('source_type', 'Document')
        result_text += f"📖 Source: {source_file} ({source_type})\n"
        
        # Content preview
        content = payload.get('text', '').strip()
        
        # Adjust content length based on intent
        if intent == 'examples' and payload.get('has_code'):
            # Show more content for code examples
            preview_length = 500
        elif intent == 'definition':
            # Show more detailed content for definitions
            preview_length = 400
        else:
            preview_length = 300
        
        if len(content) > preview_length:
            content = content[:preview_length] + "..."
        
        result_text += f"📄 Content:\n{content}\n"
        
        # Add metadata if relevant
        if payload.get('has_code'):
            result_text += "💻 Contains code examples\n"
        if payload.get('has_table'):
            result_text += "📊 Contains tables/data\n"
        
        result_text += "─" * 50
        
        return result_text

    def _generate_closing(self, intent: str, analysis: Dict) -> str:
        """Generate conversational closing"""
        suggestions = []
        
        if intent == 'definition':
            suggestions = [
                "Ask for examples or use cases",
                "Learn about implementation details",
                "Explore related concepts"
            ]
        elif intent == 'how_to':
            suggestions = [
                "Ask for specific examples",
                "Request troubleshooting tips",
                "Learn about best practices"
            ]
        elif intent == 'examples':
            suggestions = [
                "Ask for more detailed explanations",
                "Request step-by-step guides",
                "Explore advanced scenarios"
            ]
        else:
            suggestions = [
                "Ask for more specific information",
                "Request examples or tutorials",
                "Explore related topics"
            ]
        
        closing = "💡 **What's next?** You can:\n"
        for suggestion in suggestions[:3]:
            closing += f"   • {suggestion}\n"
        
        return closing

    def _generate_no_results_response(self, query: str) -> str:
        """Generate response when no results found"""
        return f"""🤖 I couldn't find specific information about '{query}' in the available documents.

💡 **Try these alternatives:**
   • Rephrase your question with different keywords
   • Ask about a more general topic first
   • Check if the documents have been properly indexed
   • Try searching for related concepts

🔍 **Example searches that work well:**
   • "How to create a database table"
   • "Database performance optimization"
   • "Installation procedures"
   • "Configuration examples"
"""

    def update_preferences(self, **kwargs):
        """Update user preferences"""
        self.user_preferences.update(kwargs)

    def get_conversation_context(self) -> List[Dict]:
        """Get recent conversation context"""
        return self.conversation_context.copy()