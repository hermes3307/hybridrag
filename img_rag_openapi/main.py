#!/usr/bin/env python3
"""
🖼️ Conversational Image Processing Assistant with RAG
Main interface for natural language image processing and intelligent Q&A
"""

import os
import sys
import time
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from collections import Counter

# Import our image-specific modules
from image_web_downloader import ImageWebDownloader
from smart_image_chunker import SmartImageChunker  
from image_vector_manager import ImageVectorStoreManager
from image_query_engine import ImageQueryEngine
from status import StatusManager

@dataclass
class ImageConversationState:
    """Tracks the current image conversation state"""
    current_url: Optional[str] = None
    downloaded_images: List[str] = None
    discovered_images: List = None
    processed_chunks: int = 0
    vector_store_ready: bool = False
    user_preferences: Dict = None
    status_manager: StatusManager = None
    
    def __post_init__(self):
        if self.downloaded_images is None:
            self.downloaded_images = []
        if self.discovered_images is None:
            self.discovered_images = []
        if self.user_preferences is None:
            self.user_preferences = {
                'max_image_size': 1024,
                'preferred_formats': ['jpg', 'jpeg', 'png', 'gif', 'webp'],
                'use_ai_analysis': True,
                'analysis_type': 'detailed',
                'show_thumbnails': True,
                'search_type': 'both'  # text, visual, both
            }


class ImageRAGEngine:
    """🖼️ Retrieval Augmented Generation Engine for Images"""
    
    def __init__(self, vector_manager: ImageVectorStoreManager, query_engine: ImageQueryEngine):
        self.vector_manager = vector_manager
        self.query_engine = query_engine
        
        # Image-specific RAG prompts
        self.rag_prompts = {
            'find_images': """Based on the following image search results, provide information about the images found:

Search Results:
{results}

Query: {query}

Please provide:
1. Summary of images found
2. Key visual characteristics
3. Relevant categories and content types
4. Suggestions for related searches

Response:""",
            
            'compare_images': """Based on the following image comparison results:

Images:
{results}

Query: {query}

Please provide:
1. Visual similarities and differences
2. Content analysis comparison
3. Technical characteristics
4. Usage recommendations

Response:""",
            
            'image_analysis': """Based on the following image analysis:

Image Analysis:
{results}

Query: {query}

Please provide a comprehensive analysis including:
1. Detailed content description
2. Technical specifications
3. Potential use cases
4. Related image recommendations

Response:"""
        }

    async def generate_image_rag_response(self, question: str, max_results: int = 5) -> str:
        """🎯 Generate RAG response for image-related questions"""
        
        try:
            print(f"🧠 Processing image query: {question}")
            
            # Perform image search
            results = await self.query_engine.search(question, k=max_results)
            
            if not results:
                return f"🖼️ I couldn't find any images matching '{question}'. Try using more general terms or check if images have been properly indexed."
            
            print(f"📊 Found {len(results)} relevant images")
            
            # Generate structured response
            response = self._generate_image_response(question, results)
            
            return response
            
        except Exception as e:
            print(f"⚠️ Image RAG processing failed: {e}")
            return f"🖼️ I encountered an issue processing your image query. Error: {str(e)}"

    def _generate_image_response(self, question: str, results: List[Dict]) -> str:
        """Generate the final image RAG response"""
        
        # Analyze query type
        question_lower = question.lower()
        
        if 'similar' in question_lower or 'like' in question_lower:
            response_type = 'similarity'
        elif 'compare' in question_lower or 'difference' in question_lower:
            response_type = 'comparison'
        elif 'analyze' in question_lower or 'describe' in question_lower:
            response_type = 'analysis'
        else:
            response_type = 'general'
        
        # Build response
        response = f"🖼️ **Image Search Results for: {question}**\n\n"
        
        # Summary
        response += f"📊 **Found {len(results)} relevant images**\n\n"
        
        # Categorize results
        categories = {}
        formats = {}
        
        for result in results:
            payload = result.get('enhanced_payload', result.get('payload', {}))
            
            # Group by category
            cats = payload.get('detected_categories', ['unknown'])
            primary_cat = cats[0] if cats else 'unknown'
            categories[primary_cat] = categories.get(primary_cat, 0) + 1
            
            # Group by format
            fmt = payload.get('file_format', 'unknown')
            formats[fmt] = formats.get(fmt, 0) + 1
        
        # Add category breakdown
        if categories:
            response += f"📋 **Content Categories:**\n"
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                cat_name = cat.replace('_', ' ').title()
                response += f"   • {cat_name}: {count} images\n"
            response += "\n"
        
        # Add format breakdown
        if formats:
            response += f"🎨 **Image Formats:**\n"
            for fmt, count in sorted(formats.items(), key=lambda x: x[1], reverse=True):
                response += f"   • {fmt}: {count} images\n"
            response += "\n"
        
        # Show top results
        response += f"🔍 **Top Results:**\n\n"
        
        for i, result in enumerate(results[:3], 1):
            payload = result.get('enhanced_payload', result.get('payload', {}))
            score = result.get('score', 0)
            
            response += f"**{i}. {payload.get('display_name', 'Unknown')}** (Relevance: {score:.1%})\n"
            
            # Image details
            response += f"   📏 Size: {payload.get('resolution', 'Unknown')}\n"
            response += f"   🎨 Format: {payload.get('file_format', 'Unknown')}\n"
            
            # AI description (truncated)
            ai_desc = payload.get('ai_description', '')
            if ai_desc:
                desc = ai_desc[:150] + "..." if len(ai_desc) > 150 else ai_desc
                response += f"   🤖 Content: {desc}\n"
            
            # Categories
            cats = payload.get('detected_categories', [])
            if cats:
                response += f"   📋 Type: {', '.join(cats[:2])}\n"
            
            # Features
            features = []
            if payload.get('has_text'):
                features.append("Text")
            if payload.get('has_ui_elements'):
                features.append("UI")
            if payload.get('has_diagram'):
                features.append("Diagram")
            if payload.get('has_code'):
                features.append("Code")
            
            if features:
                response += f"   ⚙️ Features: {', '.join(features)}\n"
            
            response += "\n"
        
        # Add response based on query type
        if response_type == 'similarity':
            response += self._add_similarity_analysis(results)
        elif response_type == 'comparison':
            response += self._add_comparison_analysis(results)
        elif response_type == 'analysis':
            response += self._add_detailed_analysis(results)
        
        # Add suggestions
        response += f"💡 **Suggestions:**\n"
        response += self._generate_search_suggestions(question, results)
        
        return response

    def _add_similarity_analysis(self, results: List[Dict]) -> str:
        """Add similarity analysis to response"""
        analysis = f"🔗 **Similarity Analysis:**\n"
        
        if len(results) < 2:
            analysis += "   • Need more images for meaningful similarity comparison\n"
            return analysis + "\n"
        
        # Analyze common features
        common_categories = set()
        common_features = set()
        
        for result in results:
            payload = result.get('enhanced_payload', result.get('payload', {}))
            common_categories.update(payload.get('detected_categories', []))
            
            if payload.get('has_text'):
                common_features.add('text')
            if payload.get('has_ui_elements'):
                common_features.add('ui_elements')
            if payload.get('has_diagram'):
                common_features.add('diagrams')
        
        if common_categories:
            analysis += f"   • Common content types: {', '.join(list(common_categories)[:3])}\n"
        if common_features:
            analysis += f"   • Shared features: {', '.join(common_features)}\n"
        
        return analysis + "\n"

    def _add_comparison_analysis(self, results: List[Dict]) -> str:
        """Add comparison analysis to response"""
        analysis = f"⚖️ **Comparison Analysis:**\n"
        
        if len(results) < 2:
            analysis += "   • Need at least 2 images for comparison\n"
            return analysis + "\n"
        
        # Compare formats
        formats = [r.get('enhanced_payload', r.get('payload', {})).get('file_format', 'Unknown') for r in results]
        format_variety = len(set(formats))
        analysis += f"   • Format diversity: {format_variety} different formats\n"
        
        # Compare sizes
        sizes = []
        for result in results:
            payload = result.get('enhanced_payload', result.get('payload', {}))
            if payload.get('width') and payload.get('height'):
                sizes.append(payload['width'] * payload['height'])
        
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            analysis += f"   • Average resolution: {avg_size/1000000:.1f} megapixels\n"
        
        # Compare content types
        all_categories = []
        for result in results:
            payload = result.get('enhanced_payload', result.get('payload', {}))
            all_categories.extend(payload.get('detected_categories', []))
        
        category_counts = Counter(all_categories)
        if category_counts:
            top_category = category_counts.most_common(1)[0]
            analysis += f"   • Most common content: {top_category[0]} ({top_category[1]} images)\n"
        
        return analysis + "\n"

    def _add_detailed_analysis(self, results: List[Dict]) -> str:
        """Add detailed analysis to response"""
        analysis = f"🔬 **Detailed Analysis:**\n"
        
        if not results:
            return analysis + "   • No images to analyze\n\n"
        
        # Technical statistics
        total_size = sum(r.get('enhanced_payload', r.get('payload', {})).get('file_size_mb', 0) for r in results)
        analysis += f"   • Total collection size: {total_size:.1f} MB\n"
        
        # Resolution analysis
        resolutions = []
        for result in results:
            payload = result.get('enhanced_payload', result.get('payload', {}))
            if payload.get('resolution'):
                resolutions.append(payload['resolution'])
        
        if resolutions:
            unique_resolutions = len(set(resolutions))
            analysis += f"   • Resolution variety: {unique_resolutions} different sizes\n"
        
        # Content richness
        feature_counts = {
            'text': 0, 'ui': 0, 'diagrams': 0, 'code': 0
        }
        
        for result in results:
            payload = result.get('enhanced_payload', result.get('payload', {}))
            if payload.get('has_text'):
                feature_counts['text'] += 1
            if payload.get('has_ui_elements'):
                feature_counts['ui'] += 1
            if payload.get('has_diagram'):
                feature_counts['diagrams'] += 1
            if payload.get('has_code'):
                feature_counts['code'] += 1
        
        analysis += f"   • Content richness: "
        rich_features = [f"{k}: {v}" for k, v in feature_counts.items() if v > 0]
        analysis += ", ".join(rich_features) + "\n"
        
        return analysis + "\n"

    def _generate_search_suggestions(self, original_query: str, results: List[Dict]) -> str:
        """Generate search suggestions based on results"""
        suggestions = []
        
        # Extract common themes from results
        all_keywords = []
        all_categories = []
        
        for result in results:
            payload = result.get('enhanced_payload', result.get('payload', {}))
            keywords = payload.get('extracted_keywords', [])
            categories = payload.get('detected_categories', [])
            
            all_keywords.extend(keywords)
            all_categories.extend(categories)
        
        # Generate suggestions based on common themes
        common_keywords = [kw for kw, count in Counter(all_keywords).most_common(3)]
        common_categories = [cat for cat, count in Counter(all_categories).most_common(2)]
        
        if common_keywords:
            suggestions.append(f"More images with: {', '.join(common_keywords[:2])}")
        
        if common_categories:
            cat_name = common_categories[0].replace('_', ' ')
            suggestions.append(f"Explore more {cat_name} images")
        
        # Add general suggestions
        suggestions.extend([
            "Try visual similarity search with specific images",
            "Filter by image format or size",
            "Search for specific visual features"
        ])
        
        result = ""
        for i, suggestion in enumerate(suggestions[:4], 1):
            result += f"   {i}. {suggestion}\n"
        
        return result


class ConversationalImageAssistant:
    """🖼️ The main conversational image assistant with RAG capabilities"""
    
    def __init__(self):
        print("🚀 Initializing Conversational Image Assistant with RAG...")
        
        # Intent patterns for image processing commands
        self.command_patterns = {
            'download', 'scan', 'fetch', 'get images', 'find images',
            'chunk', 'process images', 'analyze images',
            'index', 'build vector', 'create embeddings', 'make searchable',
            'status', 'progress', 'show images',
            'help', 'commands'
        }
        
        # Initialize components
        self.downloader = ImageWebDownloader()
        
        # Image chunker initialization
        try:
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            self.chunker = SmartImageChunker(openai_api_key)
        except Exception as e:
            print(f"⚠️ SmartImageChunker not available: {e}")
            self.chunker = None
        
        # Initialize vector manager and query engine
        try:
            self.vector_manager = ImageVectorStoreManager()
            self.query_engine = ImageQueryEngine(self.vector_manager)
            self.rag_engine = ImageRAGEngine(self.vector_manager, self.query_engine)
            print("✅ Image vector search and RAG components initialized")
        except Exception as e:
            print(f"⚠️ Image vector search and RAG not available: {e}")
            self.vector_manager = None
            self.query_engine = None
            self.rag_engine = None
        
        # StatusManager initialization
        self.status_manager = StatusManager("image_processing_status.json")
        print("📊 Image status manager initialized - loading previous session data...")
        
        # Conversation state
        self.state = ImageConversationState()
        self.state.status_manager = self.status_manager
        
        # Conversation history
        self.conversation_history = []
        
        # Restore previous session state
        self._restore_session_state()
        
        print("✅ Image Assistant ready with RAG capabilities! Let's chat! 🖼️")

    def _restore_session_state(self):
        """💾 Restore previous session state"""
        try:
            current_status = self.status_manager.current_status
            
            # Restore downloaded images list
            download_status = current_status.get('download_status')
            if download_status and download_status.get('file_list'):
                existing_images = []
                for image_path in download_status['file_list']:
                    if os.path.exists(image_path):
                        existing_images.append(image_path)
                
                self.state.downloaded_images = existing_images
                print(f"📥 Restored {len(existing_images)} downloaded images from previous session")
            
            # Restore last URL
            url_history = current_status.get('url_history', [])
            if url_history:
                self.state.current_url = url_history[-1]['url']
                print(f"🌐 Restored last URL: {self.state.current_url}")
            
            # Restore chunk state
            chunk_status = current_status.get('chunk_status')
            if chunk_status:
                self.state.processed_chunks = chunk_status.get('total_chunks', 0)
                print(f"🧩 Previous session had {self.state.processed_chunks} image chunks")
            
            # Restore vector state
            vector_status = current_status.get('vector_status')
            if vector_status:
                self.state.vector_store_ready = vector_status.get('is_ready', False)
                if self.state.vector_store_ready:
                    print(f"🗄️ Image vector database was ready - RAG is available!")
            
        except Exception as e:
            print(f"⚠️ Could not fully restore previous session: {e}")

    def _is_image_command(self, user_input: str) -> bool:
        """🔍 Check if user input is an image processing command"""
        user_input_lower = user_input.lower()
        
        # Check for explicit command keywords
        command_keywords = [
            'download', 'scan', 'fetch', 'get images', 'find images',
            'chunk', 'process images', 'analyze images', 'break into pieces',
            'index', 'build vector', 'create embeddings', 'make searchable',
            'status', 'progress', 'show images', 'check status',
            'help', 'commands', 'what can you do'
        ]
        
        return any(keyword in user_input_lower for keyword in command_keywords)

    async def chat(self, user_input: str) -> str:
        """💬 Main chat interface with image RAG capabilities"""
        if not user_input.strip():
            return "🖼️ I'm here to help with image search and analysis! Ask me questions about images or give me commands to process new ones."
        
        user_input = user_input.strip()
        
        # Store conversation
        self.conversation_history.append({
            'user': user_input,
            'timestamp': time.time()
        })

        # Check if this is an image processing command
        if self._is_image_command(user_input):
            print("🔧 Processing as image command...")
            response = await self._handle_image_command(user_input)
        
        # Check if vector database is ready for RAG
        elif self.state.vector_store_ready and self.rag_engine:
            print("🧠 Processing with Image RAG engine...")
            rag_response = await self.rag_engine.generate_image_rag_response(user_input)
            response = rag_response
        
        # Vector database not ready
        else:
            response = ("🖼️ I'd love to help you search and analyze images, but I need to have images indexed first! "
                       "Try asking me to 'download images' or 'check status' to see what's available.")
        
        # Store response
        self.conversation_history[-1]['assistant'] = response
        
        return response

    async def _handle_image_command(self, user_input: str) -> str:
        """Handle image processing commands"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['download', 'scan', 'fetch', 'get']):
            if any(word in user_lower for word in ['all', 'everything']):
                return await self._handle_download_all_images()
            elif any(word in user_lower for word in ['jpg', 'jpeg', 'png', 'gif']):
                format_match = re.search(r'(jpg|jpeg|png|gif|webp|svg)', user_lower)
                if format_match:
                    return await self._handle_download_by_format(format_match.group(1))
            else:
                return await self._handle_download_from_url(user_input)
        
        elif any(word in user_lower for word in ['chunk', 'process', 'analyze']):
            return await self._handle_analyze_images()
        
        elif any(word in user_lower for word in ['index', 'vector', 'embed', 'searchable']):
            return await self._handle_index_images()
        
        elif any(word in user_lower for word in ['status', 'progress', 'images']):
            if 'quick' in user_lower or 'brief' in user_lower:
                return self.status_manager.get_quick_status()
            else:
                return self.status_manager.get_comprehensive_status()
        
        elif any(word in user_lower for word in ['help', 'commands', 'what can']):
            return self._get_help_message()
        
        else:
            return ("🖼️ I understand you want to do something with images, but I'm not sure what. "
                   "Try: 'download images from URL', 'analyze images', 'index images', or 'show status'")

    async def _handle_download_from_url(self, user_input: str) -> str:
        """Handle URL download requests for images"""
        # Extract URL from input
        url_match = re.search(r'https?://[^\s]+', user_input)
        if url_match:
            url = url_match.group()
        else:
            return ("🖼️ I'd love to help you download images! Please provide a URL. "
                   "For example: 'Download images from https://example.com/gallery'")
        
        self.state.current_url = url
        
        try:
            print(f"🔍 Analyzing URL for images: {url}")
            images = await self.downloader.scan_images(url)
            self.state.discovered_images = images
            
            if not images:
                self.status_manager.update_url_status(url=url, documents_found=0, total_size_mb=0, status='failed')
                return f"🖼️ I couldn't find any images at {url}. Could you check the URL?"
            
            # Update status
            total_size_mb = sum(img.size or 0 for img in images) / (1024 * 1024)
            self.status_manager.update_url_status(
                url=url, documents_found=len(images), total_size_mb=total_size_mb, status='scanned'
            )
            
            img_summary = self.downloader.summarize_findings(images)
            
            response = f"🎉 Great! I found images at {url}:\n\n{img_summary}\n\n"
            response += "💡 What would you like to do?\n"
            response += "• 'Download all images'\n"
            response += "• 'Download only JPEGs'\n"
            response += "• 'Download only PNGs'\n"
            response += "• 'Show me more details first'"
            
            return response
            
        except Exception as e:
            self.status_manager.update_url_status(url=url, documents_found=0, total_size_mb=0, status='failed')
            return f"❌ Oops! I had trouble accessing {url}. Error: {str(e)}"

    async def _handle_download_all_images(self) -> str:
        """Handle download all images requests"""
        if not self.state.discovered_images:
            return ("🖼️ I don't see any discovered images to download! "
                   "Please scan a URL first with something like 'download images from https://example.com'")
        
        try:
            downloaded_files = await self.downloader.download_all_images()
            self.state.downloaded_images.extend(downloaded_files)
            
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
            
            response = f"✅ Image download complete!\n\n"
            response += f"📥 Downloaded: {stats['downloaded']} images\n"
            response += f"❌ Failed: {stats['failed']} downloads\n"
            if stats['bytes_downloaded'] > 0:
                mb_downloaded = stats['bytes_downloaded'] / (1024 * 1024)
                response += f"💾 Total size: {mb_downloaded:.1f} MB\n"
            
            response += f"📁 Images saved to: {self.downloader.download_dir}\n\n"
            response += "💡 Next step: 'Analyze the images' to process them with AI for searching!"
            
            return response
            
        except Exception as e:
            return f"❌ Image download failed. Error: {str(e)}"

    async def _handle_download_by_format(self, format_type: str) -> str:
        """Handle format-specific download requests"""
        if not self.state.discovered_images:
            return ("🖼️ I don't see any discovered images to download! "
                   "Please scan a URL first.")
        
        try:
            format_extensions = {
                'jpg': ['.jpg', '.jpeg'],
                'jpeg': ['.jpg', '.jpeg'],
                'png': ['.png'],
                'gif': ['.gif'],
                'webp': ['.webp'],
                'svg': ['.svg']
            }
            
            extensions = format_extensions.get(format_type.lower(), [f'.{format_type.lower()}'])
            downloaded_files = await self.downloader.download_images({'extensions': extensions})
            self.state.downloaded_images.extend(downloaded_files)
            
            stats = self.downloader.get_download_stats()
            
            response = f"✅ {format_type.upper()} download complete!\n\n"
            response += f"📄 Downloaded: {stats['downloaded']} {format_type.upper()} images\n"
            response += f"❌ Failed: {stats['failed']} downloads\n"
            if stats['bytes_downloaded'] > 0:
                mb_downloaded = stats['bytes_downloaded'] / (1024 * 1024)
                response += f"💾 Total size: {mb_downloaded:.1f} MB\n"
            
            response += f"📁 Images saved to: {self.downloader.download_dir}\n\n"
            response += "💡 Ready to analyze these images for intelligent search!"
            
            return response
            
        except Exception as e:
            return f"❌ {format_type.upper()} download failed. Error: {str(e)}"

    async def _handle_analyze_images(self) -> str:
        """Handle image analysis requests"""
        if not self.chunker:
            return "❌ Image analyzer is not available. Please check if OpenAI API key is configured."
        
        if not self.state.downloaded_images:
            return ("🖼️ I don't see any downloaded images to analyze yet! "
                   "Would you like me to download some images first?")
        
        try:
            print(f"🧠 Analyzing {len(self.state.downloaded_images)} images with AI...")
            
            chunks = await self.chunker.process_images(
                self.state.downloaded_images,
                use_ai_analysis=self.state.user_preferences['use_ai_analysis'],
                analysis_type=self.state.user_preferences['analysis_type'],
                max_image_size=self.state.user_preferences['max_image_size']
            )
            
            self.state.processed_chunks = len(chunks)
            
            # Update status
            processing_stats = self.chunker.get_processing_stats()
            self.status_manager.update_chunk_status(
                total_chunks=len(chunks),
                total_characters=sum(len(chunk.ai_description) for chunk in chunks),
                files_processed=processing_stats.get('files_processed', 0),
                processing_errors=processing_stats.get('errors', []),
                chunk_size=0,  # Not applicable for images
                overlap=0,     # Not applicable for images
                semantic_chunking=True  # AI analysis is semantic
            )
            
            response = f"✅ Perfect! I've analyzed your images with AI:\n\n"
            response += f"📸 Images processed: {len(self.state.downloaded_images)}\n"
            response += f"🧩 Analysis chunks created: {self.state.processed_chunks:,}\n"
            response += f"🤖 AI analyzed: {processing_stats.get('ai_analyzed', 0)} images\n"
            response += f"🧠 Analysis type: {self.state.user_preferences['analysis_type']}\n\n"
            response += "🎯 Ready to index for intelligent image search!"
            
            return response
            
        except Exception as e:
            return f"❌ I had trouble analyzing the images. Error: {str(e)}"

    async def _handle_index_images(self) -> str:
        """Handle image vector indexing requests"""
        if self.state.processed_chunks == 0:
            return ("🖼️ I need some analyzed images to index! "
                   "Would you like me to analyze your downloaded images first?")
        
        if not self.vector_manager:
            return ("❌ Vector indexing is not available. Please install required packages:\n"
                   "pip install sentence-transformers qdrant-client torch torchvision\n"
                   "pip install git+https://github.com/openai/CLIP.git")
        
        try:
            print(f"🗄️ Indexing {self.state.processed_chunks} image chunks...")
            
            chunks = self.chunker.get_chunks()
            if not chunks:
                return "🖼️ I couldn't find the processed image chunks. Please run analysis again."
            
            success = await self.vector_manager.build_index(chunks)
            
            if success:
                self.state.vector_store_ready = True
                
                collection_info = self.vector_manager.get_collection_info()
                text_info = collection_info.get('text_collection', {})
                image_info = collection_info.get('image_collection', {})
                
                self.status_manager.update_vector_status(
                    is_ready=True,
                    collection_name=f"{self.vector_manager.collection_name}",
                    vector_count=text_info.get('points_count', len(chunks)),
                    vector_dimensions=self.vector_manager.text_vector_size,
                    index_size_mb=len(chunks) * 0.002,  # Estimate
                    embedding_model=self.vector_manager.text_model_name,
                    search_capabilities=["text_search", "visual_search", "similarity_search", "RAG_enabled"]
                )
                
                response = f"🎉 Image vector indexing completed successfully!\n\n"
                response += f"🗄️ Vector database: Ready with Image RAG capabilities\n"
                response += f"📊 Indexed chunks: {len(chunks):,}\n"
                response += f"🔤 Text vectors: {text_info.get('points_count', 0):,}\n"
                if image_info:
                    response += f"🖼️ Visual vectors: {image_info.get('points_count', 0):,}\n"
                response += f"📐 Vector dimensions: {self.vector_manager.text_vector_size}\n"
                response += f"🤖 Text model: {self.vector_manager.text_model_name}\n"
                if self.vector_manager.image_model:
                    response += f"👁️ Vision model: {self.vector_manager.image_model_name}\n"
                response += f"💻 Device: {self.vector_manager.device if hasattr(self.vector_manager, 'device') else 'CPU'}\n\n"
                response += "🎯 **You can now ask me intelligent questions about your images!**\n"
                response += "💡 Try: 'Show me screenshots', 'Find diagrams', 'Images with code', 'Similar to dashboard'"
                
                return response
            else:
                return "❌ Image vector indexing failed. Please check the logs for details."
            
        except Exception as e:
            return f"❌ Indexing failed. Error: {str(e)}"

    def _get_help_message(self) -> str:
        """Get help message for image assistant"""
        return """🖼️ **I'm your intelligent image assistant!** Here's what I can do:

📥 **Image Management**
   • "Download images from https://example.com/gallery"
   • "Get all images from that GitHub repo"
   • "Download only PNG images"
   • "Analyze the downloaded images"
   • "Index images for search"

🧠 **Intelligent Image Q&A (when images are indexed)**
   • "Show me screenshots of user interfaces"
   • "Find diagrams and flowcharts"
   • "Images containing code examples"
   • "Dashboard images with charts"
   • "Similar images to login screens"
   • "Photos with people in them"

🔍 **Advanced Image Search**
   • "Find images with specific colors"
   • "Screenshots of mobile apps"
   • "Technical documentation images"
   • "Images with text content"
   • "High resolution photos"

📊 **Status & Information**
   • "What's the current status?"
   • "Show quick status"
   • "How many images are indexed?"

💡 **Just talk naturally!** I can understand:
   - Visual similarity searches
   - Content-based image queries
   - Technical image searches
   - Format and quality filters
   - Multi-modal search (text + visual)

🎯 **Current capabilities**: """ + (
    "✅ Image RAG-powered intelligent search available!" if self.state.vector_store_ready 
    else "⏳ Ready for image processing - index some images to unlock intelligent Q&A!"
)


async def main():
    """🖼️ Main interactive chat loop for image RAG"""
    print("🌟" + "=" * 70 + "🌟")
    print("      🖼️ Intelligent Image Assistant with RAG")
    print("🌟" + "=" * 70 + "🌟")
    print()
    print("💬 Hi! I'm your intelligent image processing assistant!")
    print("   I can download, analyze, index images AND answer complex visual questions!")
    print()
    print("🎯 Try asking me:")
    print("   • Visual questions: 'Show me screenshots of dashboards'")
    print("   • Content search: 'Find images with code examples'")
    print("   • Similarity search: 'Images similar to login interfaces'")
    print("   • Image commands: 'Download images from [URL]'")
    print()
    print("   Type 'quit' to exit")
    print("=" * 78)
    
    assistant = ConversationalImageAssistant()
    conversation_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n😊 You: ").strip()
            
            # Check for exit
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print(f"\n🖼️ Goodbye! We had {conversation_count} great image conversations today! ✨")
                break
            
            if not user_input:
                continue
            
            # Get response
            print(f"\n🖼️ Assistant: ", end="", flush=True)
            response = await assistant.chat(user_input)
            print(response)
            
            conversation_count += 1
            
        except KeyboardInterrupt:
            print(f"\n\n🖼️ Goodbye! We had {conversation_count} great image conversations today! ✨")
            break
        except Exception as e:
            print(f"\n❌ Oops! Something went wrong: {e}")
            print("💡 Please try again or ask for help!")

if __name__ == "__main__":
    asyncio.run(main())