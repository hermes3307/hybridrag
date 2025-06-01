#!/usr/bin/env python3
"""
🖼️ Image Query Engine
Advanced query processing and response generation for image search
"""

import re
import time
import asyncio
import os
import base64
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class ImageQueryAnalyzer:
    """🧠 Analyzes user queries to understand intent for image search"""
    
    def __init__(self):
        # Intent patterns for image search
        self.intent_patterns = {
            'find_similar': [
                r'find similar\s+(.+)',
                r'show me similar\s+(.+)',
                r'images like\s+(.+)',
                r'similar to\s+(.+)',
                r'비슷한\s+(.+)',
                r'유사한\s+(.+)',
            ],
            'find_by_content': [
                r'find images\s+(.+)',
                r'show me\s+(.+)',
                r'images with\s+(.+)',
                r'pictures of\s+(.+)',
                r'(.+)\s+이미지',
                r'(.+)\s+사진',
            ],
            'find_by_type': [
                r'screenshots?\s+(.+)',
                r'diagrams?\s+(.+)',
                r'charts?\s+(.+)',
                r'ui\s+(.+)',
                r'interface\s+(.+)',
                r'스크린샷\s+(.+)',
                r'다이어그램\s+(.+)',
            ],
            'find_by_feature': [
                r'images containing\s+(.+)',
                r'with text\s+(.+)',
                r'showing\s+(.+)',
                r'displaying\s+(.+)',
                r'(.+)\s+포함된',
                r'(.+)\s+있는',
            ],
            'technical_search': [
                r'code\s+(.+)',
                r'programming\s+(.+)',
                r'technical\s+(.+)',
                r'documentation\s+(.+)',
                r'코드\s+(.+)',
                r'프로그래밍\s+(.+)',
            ]
        }
        
        # Image-specific keywords for different domains
        self.domain_keywords = {
            'ui_screenshot': [
                'interface', 'ui', 'screen', 'window', 'dialog', 'button', 'menu',
                'form', 'login', 'dashboard', 'application', 'software'
            ],
            'diagram': [
                'diagram', 'flowchart', 'chart', 'graph', 'visualization', 'schema',
                'architecture', 'flow', 'process', 'structure'
            ],
            'documentation': [
                'manual', 'guide', 'documentation', 'tutorial', 'instructions',
                'help', 'reference', 'specification'
            ],
            'code': [
                'code', 'programming', 'script', 'syntax', 'function', 'class',
                'method', 'algorithm', 'source', 'snippet'
            ],
            'data_visualization': [
                'chart', 'graph', 'plot', 'visualization', 'data', 'statistics',
                'metrics', 'analytics', 'dashboard', 'report'
            ],
            'photo': [
                'photo', 'picture', 'image', 'photograph', 'snapshot',
                'portrait', 'landscape', 'scene'
            ]
        }
        
        # Visual characteristics keywords
        self.visual_keywords = {
            'color': [
                'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray',
                'colorful', 'bright', 'dark', 'light'
            ],
            'size': [
                'large', 'small', 'big', 'tiny', 'wide', 'narrow', 'tall', 'short'
            ],
            'quality': [
                'high quality', 'clear', 'blurry', 'sharp', 'crisp', 'detailed'
            ],
            'style': [
                'modern', 'classic', 'simple', 'complex', 'minimalist', 'detailed'
            ]
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """🔍 Analyze user query and extract structured information for image search"""
        analysis = {
            'original_query': query,
            'intent': self._detect_intent(query),
            'visual_features': self._extract_visual_features(query),
            'domains': self._identify_domains(query),
            'keywords': self._extract_keywords(query),
            'search_filters': self._extract_search_filters(query),
            'search_type': self._determine_search_type(query),
            'complexity': self._assess_complexity(query),
            'language': self._detect_language(query)
        }
        
        return analysis

    def _detect_intent(self, query: str) -> str:
        """Detect the primary intent of the image search query"""
        query_lower = query.lower().strip()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return 'general_search'

    def _extract_visual_features(self, query: str) -> Dict[str, List[str]]:
        """Extract visual features mentioned in the query"""
        query_lower = query.lower()
        features = {}
        
        for feature_type, keywords in self.visual_keywords.items():
            found_features = [kw for kw in keywords if kw in query_lower]
            if found_features:
                features[feature_type] = found_features
        
        return features

    def _identify_domains(self, query: str) -> List[str]:
        """Identify relevant image domains"""
        query_lower = query.lower()
        domains = []
        
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)
        
        return domains

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords for image search"""
        # Remove stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'show', 'find', 'get', 'image', 'images', 'picture', 'pictures',
            'photo', 'photos', 'me', 'some', 'any', 'all',
            '을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로',
            '보여', '찾아', '이미지', '사진', '그림'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords

    def _extract_search_filters(self, query: str) -> Dict[str, Any]:
        """Extract search filters from query"""
        filters = {}
        query_lower = query.lower()
        
        # Content type filters
        if any(word in query_lower for word in ['screenshot', 'screen', 'ui']):
            filters['has_ui_elements'] = True
        
        if any(word in query_lower for word in ['diagram', 'chart', 'graph']):
            filters['has_diagram'] = True
        
        if any(word in query_lower for word in ['code', 'programming', 'script']):
            filters['has_code'] = True
        
        if any(word in query_lower for word in ['text', 'label', 'caption']):
            filters['has_text'] = True
        
        # File format filters
        format_keywords = {
            'jpeg': ['jpeg', 'jpg'],
            'png': ['png'],
            'gif': ['gif', 'animated'],
            'svg': ['svg', 'vector']
        }
        
        for format_name, keywords in format_keywords.items():
            if any(kw in query_lower for kw in keywords):
                filters['file_format'] = format_name.upper()
                break
        
        return filters

    def _determine_search_type(self, query: str) -> str:
        """Determine the type of search to perform"""
        query_lower = query.lower()
        
        # Visual similarity search indicators
        visual_indicators = ['similar', 'like', 'looks like', 'resembles', '비슷한', '유사한']
        if any(indicator in query_lower for indicator in visual_indicators):
            return 'visual'
        
        # Text-based search indicators
        text_indicators = ['containing', 'with text', 'showing', '포함', '텍스트']
        if any(indicator in query_lower for indicator in text_indicators):
            return 'text'
        
        # Default to both
        return 'both'

    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity for image search"""
        word_count = len(query.split())
        has_filters = bool(self._extract_search_filters(query))
        has_visual_features = bool(self._extract_visual_features(query))
        
        if word_count > 10 or (has_filters and has_visual_features):
            return 'complex'
        elif word_count > 5 or has_filters or has_visual_features:
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


class ImageQueryEngine:
    """🖼️ Main image query engine for conversational image search"""
    
    def __init__(self, vector_manager):
        self.vector_manager = vector_manager
        self.analyzer = ImageQueryAnalyzer()
        self.conversation_context = []
        self.user_preferences = {
            'response_style': 'detailed',  # brief, detailed, visual
            'show_thumbnails': True,
            'max_results': 5,
            'prefer_high_quality': False,
            'search_type': 'both'  # text, visual, both
        }

    async def search(self, query: str, k: int = 5, 
                    use_context: bool = True,
                    search_type: Optional[str] = None) -> List[Dict]:
        """🔍 Perform conversational image search with context awareness"""
        
        # Analyze the query
        analysis = self.analyzer.analyze_query(query)
        
        # Determine search type
        final_search_type = search_type or analysis['search_type'] or self.user_preferences['search_type']
        
        # Build search filters
        search_filters = analysis['search_filters']
        
        # Execute search
        try:
            results = await self.vector_manager.search(
                query=analysis['original_query'],
                k=k,
                search_type=final_search_type,
                filter_conditions=search_filters
            )
            
            # Enhance results with analysis information
            for result in results:
                result['query_analysis'] = analysis
                result['enhanced_payload'] = self._enhance_result_payload(result['payload'])
            
            # Update conversation context
            if use_context:
                self.conversation_context.append({
                    'query': query,
                    'analysis': analysis,
                    'results_count': len(results),
                    'search_type': final_search_type,
                    'timestamp': time.time()
                })
                
                # Keep only recent context
                if len(self.conversation_context) > 10:
                    self.conversation_context = self.conversation_context[-10:]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image search: {e}")
            return []

    def _enhance_result_payload(self, payload: Dict) -> Dict:
        """Enhance result payload with additional computed information"""
        enhanced = payload.copy()
        
        # Add computed fields
        enhanced['image_url'] = f"file://{payload.get('image_path', '')}"
        enhanced['display_name'] = os.path.basename(payload.get('image_path', ''))
        enhanced['file_size_mb'] = payload.get('file_size', 0) / (1024 * 1024) if payload.get('file_size') else 0
        
        # Image dimensions info
        width = payload.get('width', 0)
        height = payload.get('height', 0)
        if width and height:
            enhanced['aspect_ratio'] = width / height
            enhanced['resolution'] = f"{width}x{height}"
            enhanced['megapixels'] = (width * height) / 1000000
        
        # Content categorization
        categories = payload.get('detected_categories', [])
        enhanced['primary_category'] = categories[0] if categories else 'unknown'
        enhanced['is_technical'] = any(cat in categories for cat in ['code', 'diagram', 'ui_element'])
        enhanced['is_visual_content'] = any(cat in categories for cat in ['photo', 'screenshot'])
        
        return enhanced

    async def search_by_image_file(self, image_path: str, k: int = 5) -> List[Dict]:
        """🖼️ Search for similar images using an image file as query"""
        try:
            results = await self.vector_manager.search_by_image(image_path, k)
            
            # Enhance results
            for result in results:
                result['enhanced_payload'] = self._enhance_result_payload(result['payload'])
                result['search_method'] = 'image_similarity'
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image-based search: {e}")
            return []

    def format_conversational_response(self, query: str, results: List[Dict], 
                                     include_thumbnails: bool = None) -> str:
        """💬 Format results into a conversational response for images"""
        if not results:
            return self._generate_no_results_response(query)
        
        analysis = results[0].get('query_analysis', {})
        intent = analysis.get('intent', 'general_search')
        
        # Determine if thumbnails should be included
        show_thumbnails = include_thumbnails if include_thumbnails is not None else self.user_preferences['show_thumbnails']
        
        # Build response based on intent and user preferences
        response_parts = []
        
        # Opening based on intent
        opening = self._generate_opening(intent, analysis, len(results))
        response_parts.append(opening)
        response_parts.append("")
        
        # Add results
        for i, result in enumerate(results, 1):
            result_text = self._format_single_image_result(result, i, intent, show_thumbnails)
            response_parts.append(result_text)
            response_parts.append("")
        
        # Add filter information if filters were applied
        filters_applied = analysis.get('search_filters', {})
        if filters_applied:
            response_parts.append("🔍 **Applied Filters:**")
            for filter_name, filter_value in filters_applied.items():
                response_parts.append(f"   • {filter_name}: {filter_value}")
            response_parts.append("")
        
        # Add closing suggestions
        closing = self._generate_closing(intent, analysis)
        response_parts.append(closing)
        
        return "\n".join(response_parts)

    def _generate_opening(self, intent: str, analysis: Dict, result_count: int) -> str:
        """Generate conversational opening for image search"""
        domains = analysis.get('domains', [])
        visual_features = analysis.get('visual_features', {})
        
        if intent == 'find_similar':
            return f"🖼️ I found {result_count} similar images for you:"
        
        elif intent == 'find_by_content':
            if domains:
                domain_name = domains[0].replace('_', ' ').title()
                return f"🖼️ I found {result_count} {domain_name} images matching your search:"
            else:
                return f"🖼️ I found {result_count} images with the content you're looking for:"
        
        elif intent == 'find_by_type':
            return f"🖼️ Here are {result_count} images of the type you requested:"
        
        elif intent == 'technical_search':
            return f"🖼️ I found {result_count} technical images related to your search:"
        
        else:
            if visual_features:
                feature_desc = ", ".join([f"{k}: {', '.join(v)}" for k, v in visual_features.items()])
                return f"🖼️ I found {result_count} images with these visual features ({feature_desc}):"
            else:
                return f"🖼️ Here are {result_count} images I found:"

    def _format_single_image_result(self, result: Dict, index: int, intent: str, 
                                   show_thumbnails: bool = True) -> str:
        """Format a single image search result"""
        payload = result.get('enhanced_payload', result.get('payload', {}))
        score = result.get('score', 0)
        search_types = result.get('found_by_search_types', [result.get('search_type', 'unknown')])
        
        # Header
        result_text = f"**🖼️ Image {index}** (Relevance: {score:.1%})\n"
        
        # Basic image information
        result_text += f"📁 File: {payload.get('display_name', 'Unknown')}\n"
        result_text += f"📏 Size: {payload.get('resolution', 'Unknown')} "
        
        if payload.get('file_size_mb', 0) > 0:
            result_text += f"({payload['file_size_mb']:.1f} MB)"
        result_text += "\n"
        
        result_text += f"🎨 Format: {payload.get('file_format', 'Unknown')}\n"
        
        # AI analysis summary
        ai_description = payload.get('ai_description', '')
        if ai_description:
            # Truncate long descriptions
            description = ai_description[:200] + "..." if len(ai_description) > 200 else ai_description
            result_text += f"🤖 AI Analysis: {description}\n"
        
        # Categories and features
        categories = payload.get('detected_categories', [])
        if categories:
            result_text += f"📋 Categories: {', '.join(categories[:3])}\n"
        
        # Technical features
        features = []
        if payload.get('has_text'):
            features.append("Contains text")
        if payload.get('has_ui_elements'):
            features.append("UI elements")
        if payload.get('has_diagram'):
            features.append("Diagram/Chart")
        if payload.get('has_code'):
            features.append("Code content")
        
        if features:
            result_text += f"⚙️ Features: {', '.join(features)}\n"
        
        # Search method information
        if len(search_types) > 1:
            result_text += f"🔍 Found by: {', '.join(search_types)}\n"
        
        # Visual characteristics
        if payload.get('aspect_ratio'):
            aspect_ratio = payload['aspect_ratio']
            orientation = "Landscape" if aspect_ratio > 1.2 else "Portrait" if aspect_ratio < 0.8 else "Square"
            result_text += f"📐 Orientation: {orientation}\n"
        
        # Thumbnail placeholder (in a real implementation, you'd include actual image data)
        if show_thumbnails:
            image_path = payload.get('image_path', '')
            if image_path and os.path.exists(image_path):
                result_text += f"🖼️ Path: {image_path}\n"
        
        result_text += "─" * 50
        
        return result_text

    def _generate_closing(self, intent: str, analysis: Dict) -> str:
        """Generate conversational closing for image search"""
        suggestions = []
        
        if intent == 'find_similar':
            suggestions = [
                "Search by uploading your own image",
                "Refine search with specific visual features",
                "Filter by image type or format"
            ]
        elif intent == 'find_by_content':
            suggestions = [
                "Search for images with specific UI elements",
                "Look for technical diagrams or screenshots",
                "Find images containing specific text"
            ]
        elif intent == 'technical_search':
            suggestions = [
                "Search for specific programming languages",
                "Find architecture diagrams",
                "Look for user interface screenshots"
            ]
        else:
            suggestions = [
                "Try more specific search terms",
                "Use visual feature descriptions",
                "Search by image similarity"
            ]
        
        closing = "💡 **What's next?** You can:\n"
        for suggestion in suggestions[:3]:
            closing += f"   • {suggestion}\n"
        
        # Add search tips
        closing += f"\n🔍 **Search Tips:**\n"
        closing += f"   • Use descriptive terms: 'blue interface with buttons'\n"
        closing += f"   • Specify image types: 'screenshot of login screen'\n"
        closing += f"   • Try visual similarity: 'images similar to dashboard'\n"
        
        return closing

    def _generate_no_results_response(self, query: str) -> str:
        """Generate response when no images found"""
        return f"""🖼️ I couldn't find any images matching '{query}' in the available collection.

💡 **Try these alternatives:**
   • Use more general terms (e.g., 'interface' instead of 'specific UI component')
   • Search for image types (e.g., 'screenshot', 'diagram', 'chart')
   • Try visual descriptions (e.g., 'blue interface', 'dark theme')
   • Check if images have been properly indexed

🔍 **Example searches that work well:**
   • "screenshots of login interfaces"
   • "database architecture diagrams"
   • "code examples with syntax highlighting"
   • "dashboard with charts and graphs"
"""

    async def get_image_recommendations(self, based_on_recent: int = 3) -> List[Dict]:
        """🎯 Get image recommendations based on recent searches"""
        if len(self.conversation_context) < based_on_recent:
            return []
        
        # Analyze recent search patterns
        recent_contexts = self.conversation_context[-based_on_recent:]
        
        # Extract common keywords and domains
        all_keywords = []
        all_domains = []
        
        for context in recent_contexts:
            analysis = context.get('analysis', {})
            all_keywords.extend(analysis.get('keywords', []))
            all_domains.extend(analysis.get('domains', []))
        
        # Find most common patterns
        common_keywords = [kw for kw, count in Counter(all_keywords).most_common(3)]
        common_domains = [domain for domain, count in Counter(all_domains).most_common(2)]
        
        # Generate recommendation queries
        recommendations = []
        
        if common_keywords:
            rec_query = " ".join(common_keywords[:2])
            recommendations.append({
                'query': rec_query,
                'reason': f"Based on your interest in: {', '.join(common_keywords[:2])}"
            })
        
        if common_domains:
            domain_query = common_domains[0].replace('_', ' ')
            recommendations.append({
                'query': domain_query,
                'reason': f"More {domain_query} content"
            })
        
        return recommendations

    def update_preferences(self, **kwargs):
        """Update user preferences for image search"""
        self.user_preferences.update(kwargs)

    def get_conversation_context(self) -> List[Dict]:
        """Get recent conversation context"""
        return self.conversation_context.copy()

    def get_search_statistics(self) -> Dict:
        """Get search statistics and insights"""
        if not self.conversation_context:
            return {'total_searches': 0}
        
        total_searches = len(self.conversation_context)
        
        # Analyze search patterns
        intents = [ctx['analysis']['intent'] for ctx in self.conversation_context]
        domains = []
        for ctx in self.conversation_context:
            domains.extend(ctx['analysis'].get('domains', []))
        
        search_types = [ctx.get('search_type', 'unknown') for ctx in self.conversation_context]
        
        return {
            'total_searches': total_searches,
            'most_common_intent': Counter(intents).most_common(1)[0] if intents else None,
            'most_searched_domains': Counter(domains).most_common(3),
            'preferred_search_type': Counter(search_types).most_common(1)[0] if search_types else None,
            'average_results_per_search': sum(ctx['results_count'] for ctx in self.conversation_context) / total_searches
        }

    async def find_related_images(self, image_path: str, k: int = 5) -> List[Dict]:
        """🔗 Find images related to a specific image"""
        try:
            # Use image similarity search
            similar_results = await self.search_by_image_file(image_path, k)
            
            # Also try to extract text description and search by content
            # This would require reading the image's AI description from the database
            # For now, we'll just return the visual similarity results
            
            return similar_results
            
        except Exception as e:
            logger.error(f"Error finding related images: {e}")
            return []

    def create_image_gallery_response(self, results: List[Dict], 
                                    title: str = "Image Gallery") -> Dict:
        """🖼️ Create a structured image gallery response"""
        gallery = {
            'title': title,
            'total_images': len(results),
            'images': []
        }
        
        for i, result in enumerate(results):
            payload = result.get('enhanced_payload', result.get('payload', {}))
            
            image_info = {
                'index': i + 1,
                'filename': payload.get('display_name', 'Unknown'),
                'path': payload.get('image_path', ''),
                'score': result.get('score', 0),
                'description': payload.get('ai_description', '')[:100] + "..." if payload.get('ai_description', '') else '',
                'dimensions': payload.get('resolution', 'Unknown'),
                'format': payload.get('file_format', 'Unknown'),
                'size_mb': payload.get('file_size_mb', 0),
                'categories': payload.get('detected_categories', []),
                'features': {
                    'has_text': payload.get('has_text', False),
                    'has_ui_elements': payload.get('has_ui_elements', False),
                    'has_diagram': payload.get('has_diagram', False),
                    'has_code': payload.get('has_code', False)
                }
            }
            
            gallery['images'].append(image_info)
        
        return gallery