#!/usr/bin/env python3
"""
🖼️ Smart Image Chunker
Intelligent image processing with AI-powered content analysis using OpenAI Vision
"""

import os
import asyncio
import base64
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import hashlib
import logging
import json
from dataclasses import dataclass
from PIL import Image, ImageStat
import io

# OpenAI client
try:
    from openai import AsyncOpenAI
except ImportError:
    print("❌ OpenAI library not installed. Please install with: pip install openai")
    AsyncOpenAI = None

logger = logging.getLogger(__name__)

@dataclass
class ImageChunk:
    """Represents a processed image with AI analysis"""
    image_path: str
    image_data: Optional[bytes]  # Base64 encoded image data
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str
    ai_description: str
    ai_analysis: Dict[str, Any]  # Detailed AI analysis
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for this image chunk"""
        content = f"{self.source_file}_{self.ai_description[:100]}"
        return hashlib.md5(content.encode()).hexdigest()

class SmartImageChunker:
    """🖼️ Intelligent image processing and AI analysis"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.processed_chunks = []
        self.processing_stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'ai_analyzed': 0,
            'errors': []
        }
        
        # OpenAI client setup
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. AI analysis will be disabled.")
            self.openai_client = None
        else:
            if AsyncOpenAI:
                self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            else:
                logger.error("OpenAI library not available. Install with: pip install openai")
                self.openai_client = None
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        
        # AI analysis prompts
        self.analysis_prompts = {
            'detailed': """Analyze this image in detail and provide a comprehensive description. Include:

1. **Main Subject**: What is the primary focus of the image?
2. **Visual Elements**: Colors, composition, style, lighting
3. **Text Content**: Any text, labels, or written information visible
4. **Technical Details**: Charts, diagrams, data visualization, UI elements
5. **Context**: Setting, background, environment
6. **Objects and People**: Identify and describe key elements
7. **Quality and Style**: Image quality, artistic style, photography type

Provide a detailed, searchable description that would help someone find this image based on its content.""",
            
            'technical': """Analyze this image from a technical perspective. Focus on:

1. **Document Type**: Is this a screenshot, diagram, chart, photo, or other type?
2. **Technical Content**: Any code, configurations, system interfaces, documentation
3. **Data Visualization**: Charts, graphs, tables, metrics
4. **User Interface**: Buttons, menus, forms, navigation elements
5. **Text Extraction**: All visible text, labels, captions, titles
6. **Technical Keywords**: Programming languages, software tools, technical terms

Provide technical metadata and keywords for efficient search and categorization.""",
            
            'searchable': """Create a searchable summary of this image. Include:

1. **Keywords**: List relevant search terms and tags
2. **Categories**: What type of content is this? (e.g., documentation, interface, diagram, photo)
3. **Entities**: People, places, objects, brands, technologies mentioned or shown
4. **Actions**: What activities or processes are depicted?
5. **Concepts**: Abstract ideas or topics represented

Format as a comprehensive but concise description optimized for search and retrieval."""
        }

    async def process_images(self, image_paths: List[str], 
                           use_ai_analysis: bool = True,
                           analysis_type: str = 'detailed',
                           max_image_size: int = 1024) -> List[ImageChunk]:
        """🔄 Process multiple images and create chunks with AI analysis"""
        
        print(f"🖼️ Processing {len(image_paths)} images...")
        self.processed_chunks = []
        
        for image_path in image_paths:
            try:
                await self._process_single_image(
                    image_path, use_ai_analysis, analysis_type, max_image_size
                )
                self.processing_stats['files_processed'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {image_path}: {str(e)}"
                self.processing_stats['errors'].append(error_msg)
                logger.error(error_msg)
        
        self.processing_stats['total_chunks'] = len(self.processed_chunks)
        
        print(f"✅ Processing complete! Created {len(self.processed_chunks)} image chunks")
        return self.processed_chunks

    async def _process_single_image(self, image_path: str, use_ai_analysis: bool, 
                                  analysis_type: str, max_image_size: int):
        """Process a single image"""
        print(f"📸 Processing {os.path.basename(image_path)}...")
        
        try:
            # Load and analyze image
            image_metadata = await self._extract_image_metadata(image_path)
            
            if not image_metadata:
                print(f"⚠️ Could not process {image_path}")
                return
            
            # Prepare image for AI analysis
            image_data = None
            ai_description = "Image analysis not available"
            ai_analysis = {}
            
            if use_ai_analysis and self.openai_client:
                try:
                    image_data = await self._prepare_image_for_ai(image_path, max_image_size)
                    ai_description, ai_analysis = await self._analyze_image_with_ai(
                        image_data, analysis_type
                    )
                    self.processing_stats['ai_analyzed'] += 1
                    print(f"🧠 AI analysis completed for {os.path.basename(image_path)}")
                except Exception as e:
                    logger.error(f"AI analysis failed for {image_path}: {e}")
                    ai_description = f"AI analysis failed: {str(e)}"
            
            # Create chunk metadata
            chunk_metadata = image_metadata.copy()
            chunk_metadata.update({
                'ai_analysis_type': analysis_type if use_ai_analysis else 'none',
                'ai_analysis_timestamp': asyncio.get_event_loop().time(),
                'processing_params': {
                    'use_ai_analysis': use_ai_analysis,
                    'analysis_type': analysis_type,
                    'max_image_size': max_image_size
                }
            })
            
            # Create ImageChunk object
            chunk = ImageChunk(
                image_path=image_path,
                image_data=image_data,
                metadata=chunk_metadata,
                chunk_id="",  # Will be auto-generated
                source_file=image_path,
                ai_description=ai_description,
                ai_analysis=ai_analysis
            )
            
            self.processed_chunks.append(chunk)
            
        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            self.processing_stats['errors'].append(error_msg)
            logger.error(error_msg)

    async def _extract_image_metadata(self, image_path: str) -> Optional[Dict]:
        """🔍 Extract metadata from image file"""
        try:
            file_path = Path(image_path)
            
            # Basic file information
            metadata = {
                'source_type': 'Image',
                'source_file': os.path.basename(image_path),
                'file_size': file_path.stat().st_size,
                'file_extension': file_path.suffix.lower(),
                'file_format': file_path.suffix.lower().replace('.', '').upper()
            }
            
            # Try to open with PIL for detailed image info
            try:
                with Image.open(image_path) as img:
                    metadata.update({
                        'width': img.width,
                        'height': img.height,
                        'mode': img.mode,
                        'format': img.format,
                        'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                    })
                    
                    # Color analysis
                    if img.mode in ('RGB', 'RGBA'):
                        # Convert to RGB if needed
                        rgb_img = img.convert('RGB')
                        stat = ImageStat.Stat(rgb_img)
                        
                        metadata.update({
                            'dominant_colors': {
                                'red_mean': stat.mean[0],
                                'green_mean': stat.mean[1],
                                'blue_mean': stat.mean[2]
                            },
                            'brightness': sum(stat.mean) / 3,
                            'is_grayscale': len(set(stat.mean)) == 1
                        })
                    
                    # EXIF data (if available)
                    exif_dict = {}
                    if hasattr(img, '_getexif') and img._getexif():
                        exif = img._getexif()
                        if exif:
                            for tag_id, value in exif.items():
                                tag = Image.ExifTags.TAGS.get(tag_id, tag_id)
                                exif_dict[tag] = str(value)
                    
                    if exif_dict:
                        metadata['exif'] = exif_dict
                        
            except Exception as e:
                logger.debug(f"Could not extract detailed image info from {image_path}: {e}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {image_path}: {e}")
            return None

    async def _prepare_image_for_ai(self, image_path: str, max_size: int) -> str:
        """📐 Prepare image for AI analysis (resize and encode)"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too large
                if max(img.width, img.height) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Save to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr = img_byte_arr.getvalue()
                
                # Encode to base64
                return base64.b64encode(img_byte_arr).decode('utf-8')
                
        except Exception as e:
            raise Exception(f"Failed to prepare image for AI: {e}")

    async def _analyze_image_with_ai(self, image_data: str, analysis_type: str) -> Tuple[str, Dict]:
        """🧠 Analyze image using OpenAI Vision API"""
        if not self.openai_client:
            raise Exception("OpenAI client not available")
        
        try:
            prompt = self.analysis_prompts.get(analysis_type, self.analysis_prompts['detailed'])
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.2
            )
            
            ai_description = response.choices[0].message.content
            
            # Extract structured analysis
            ai_analysis = {
                'model_used': 'gpt-4-vision-preview',
                'analysis_type': analysis_type,
                'confidence': 'high',  # GPT-4V typically provides high confidence
                'tokens_used': response.usage.total_tokens if response.usage else 0,
                'raw_response': ai_description
            }
            
            # Try to extract keywords and categories from the description
            keywords = await self._extract_keywords_from_description(ai_description)
            categories = await self._extract_categories_from_description(ai_description)
            
            ai_analysis.update({
                'extracted_keywords': keywords,
                'detected_categories': categories
            })
            
            return ai_description, ai_analysis
            
        except Exception as e:
            raise Exception(f"OpenAI Vision API error: {e}")

    async def _extract_keywords_from_description(self, description: str) -> List[str]:
        """Extract keywords from AI description"""
        # Simple keyword extraction - could be enhanced with NLP
        import re
        
        # Remove common words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'this', 'that', 'these', 'those', 'image', 'shows', 'contains', 'depicts'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', description.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Remove duplicates and limit
        return list(set(keywords))[:20]

    async def _extract_categories_from_description(self, description: str) -> List[str]:
        """Extract categories from AI description"""
        description_lower = description.lower()
        
        category_keywords = {
            'screenshot': ['screenshot', 'screen capture', 'interface', 'application'],
            'diagram': ['diagram', 'chart', 'graph', 'flowchart', 'schematic'],
            'documentation': ['documentation', 'manual', 'guide', 'instructions'],
            'code': ['code', 'programming', 'script', 'syntax'],
            'photo': ['photograph', 'photo', 'picture', 'image'],
            'ui_element': ['button', 'menu', 'form', 'window', 'dialog'],
            'data_viz': ['chart', 'graph', 'visualization', 'plot', 'dashboard'],
            'text_document': ['document', 'text', 'page', 'article']
        }
        
        detected_categories = []
        for category, keywords in category_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_categories.append(category)
        
        return detected_categories

    def get_chunks(self) -> List[ImageChunk]:
        """Get all processed image chunks"""
        return self.processed_chunks.copy()

    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return self.processing_stats.copy()

    def export_chunks_to_json(self, output_file: str) -> None:
        """💾 Export chunks to JSON format"""
        chunks_data = []
        for chunk in self.processed_chunks:
            chunk_data = {
                'image_path': chunk.image_path,
                'metadata': chunk.metadata,
                'chunk_id': chunk.chunk_id,
                'source_file': chunk.source_file,
                'ai_description': chunk.ai_description,
                'ai_analysis': chunk.ai_analysis
            }
            # Don't include image_data in JSON export due to size
            chunks_data.append(chunk_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        print(f"📁 Exported {len(chunks_data)} image chunks to {output_file}")

    def clear_chunks(self) -> None:
        """🗑️ Clear all processed chunks"""
        self.processed_chunks = []
        self.processing_stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'ai_analyzed': 0,
            'errors': []
        }

    async def analyze_image_batch(self, image_paths: List[str], 
                                batch_size: int = 5) -> List[Dict]:
        """🔄 Analyze a batch of images efficiently"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
            
            batch_tasks = [
                self._process_single_image(path, True, 'searchable', 1024)
                for path in batch
            ]
            
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Add small delay between batches to respect rate limits
            await asyncio.sleep(1)
        
        return self.get_processing_stats()

    def get_chunk_by_description(self, search_term: str) -> List[ImageChunk]:
        """🔍 Find chunks by description content"""
        matching_chunks = []
        search_term_lower = search_term.lower()
        
        for chunk in self.processed_chunks:
            if search_term_lower in chunk.ai_description.lower():
                matching_chunks.append(chunk)
        
        return matching_chunks

    def get_chunks_by_metadata(self, **criteria) -> List[ImageChunk]:
        """🔍 Find chunks by metadata criteria"""
        matching_chunks = []
        
        for chunk in self.processed_chunks:
            match = True
            for key, value in criteria.items():
                if key not in chunk.metadata or chunk.metadata[key] != value:
                    match = False
                    break
            
            if match:
                matching_chunks.append(chunk)
        
        return matching_chunks