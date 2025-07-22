#!/usr/bin/env python3
"""
🧩 Smart Document Chunker
Intelligent text extraction and chunking with semantic awareness
"""

import os
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import hashlib
import logging
from dataclasses import dataclass

# Import text extraction libraries
try:
    import fitz  # PyMuPDF for PDFs
except ImportError:
    fitz = None

try:
    from docx import Document  # python-docx for Word docs
except ImportError:
    Document = None

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str
    chunk_index: int
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for this chunk"""
        content = f"{self.source_file}_{self.chunk_index}_{self.text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()

class SmartDocumentChunker:
    """🧩 Intelligent document processing and chunking"""
    
    def __init__(self):
        self.processed_chunks = []
        self.processing_stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'total_characters': 0,
            'errors': []
        }
        
        # Text extraction mapping
        self.extractors = {
            '.pdf': self._extract_pdf_text,
            '.txt': self._extract_text_file,
            '.md': self._extract_text_file,
            '.docx': self._extract_docx_text,
            '.doc': self._extract_doc_text,
            '.html': self._extract_html_text,
            '.htm': self._extract_html_text,
        }

    async def process_files(self, file_paths: List[str], 
                          chunk_size: int = 1000,
                          overlap: int = 200,
                          use_semantic_splitting: bool = True,
                          preserve_structure: bool = True) -> List[DocumentChunk]:
        """🔄 Process multiple files and create chunks"""
        
        print(f"🧩 Processing {len(file_paths)} files...")
        self.processed_chunks = []
        
        for file_path in file_paths:
            try:
                await self._process_single_file(
                    file_path, chunk_size, overlap, 
                    use_semantic_splitting, preserve_structure
                )
                self.processing_stats['files_processed'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                self.processing_stats['errors'].append(error_msg)
                logger.error(error_msg)
        
        self.processing_stats['total_chunks'] = len(self.processed_chunks)
        self.processing_stats['total_characters'] = sum(
            len(chunk.text) for chunk in self.processed_chunks
        )
        
        print(f"✅ Processing complete! Created {len(self.processed_chunks)} chunks")
        return self.processed_chunks


    async def _process_single_file(self, file_path: str, chunk_size: int, 
                                overlap: int, use_semantic: bool, preserve_structure: bool):
        """Process a single file"""
        print(f"📄 Processing {os.path.basename(file_path)}...")
        
        try:
            # Extract text
            text_content, metadata = await self._extract_text(file_path)
            
            if not text_content or not text_content.strip():
                print(f"⚠️  No text extracted from {file_path}")
                return
            
            # For placeholder text (failed extractions), create a minimal chunk
            if text_content.startswith('[PDF Document:') and 'extraction failed' in text_content:
                # Create a single informational chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': 0,
                    'chunk_size': len(text_content),
                    'word_count': len(text_content.split()),
                    'extraction_status': 'failed'
                })
                
                chunk = DocumentChunk(
                    text=text_content,
                    metadata=chunk_metadata,
                    chunk_id="",
                    source_file=file_path,
                    chunk_index=0
                )
                
                self.processed_chunks.append(chunk)
                return
            
            # Clean and preprocess
            cleaned_text = self._clean_text(text_content)
            
            # Create chunks
            if use_semantic:
                chunks = self._create_semantic_chunks(
                    cleaned_text, chunk_size, overlap, preserve_structure
                )
            else:
                chunks = self._create_simple_chunks(cleaned_text, chunk_size, overlap)
            
            # Convert to DocumentChunk objects
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 50:  # Skip very small chunks
                    continue
                    
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'chunk_size': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'processing_params': {
                        'chunk_size': chunk_size,
                        'overlap': overlap,
                        'semantic': use_semantic,
                        'preserve_structure': preserve_structure
                    }
                })
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id="",  # Will be auto-generated
                    source_file=file_path,
                    chunk_index=i
                )
                
                self.processed_chunks.append(chunk)
                
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.processing_stats['errors'].append(error_msg)
            logger.error(error_msg)
            

    async def _extract_text(self, file_path: str) -> Tuple[str, Dict]:
        """🔍 Extract text from various file formats"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            try:
                return await self._extract_pdf_text(file_path)
            except Exception as e:
                logger.warning(f"Primary PDF extraction failed for {file_path}, trying fallback: {e}")
                return await self._extract_pdf_text_fallback(file_path)
        
        elif file_ext in self.extractors:
            extractor = self.extractors[file_ext]
            return await extractor(file_path)
        else:
            # Fallback to text file reading
            return await self._extract_text_file(file_path)

    async def _extract_pdf_text_fallback(self, file_path: str) -> Tuple[str, Dict]:
        """Fallback PDF extraction for problematic files"""
        try:
            # Try alternative approach with different settings
            doc = fitz.open(file_path)
            text_content = ""
            
            for page_num in range(min(len(doc), 10)):  # Limit to first 10 pages for testing
                try:
                    page = doc[page_num]
                    text = page.get_text("text")  # Explicit text mode
                    if text.strip():
                        text_content += f"\n\nPage {page_num + 1}:\n{text}"
                except:
                    continue
            
            doc.close()
            
            metadata = {
                'source_type': 'PDF (Fallback)',
                'source_file': os.path.basename(file_path),
                'extraction_method': 'fallback'
            }
            
            return text_content, metadata
            
        except Exception as e:
            # Last resort - create a placeholder
            metadata = {
                'source_type': 'PDF (Failed)',
                'source_file': os.path.basename(file_path),
                'extraction_error': str(e)
            }
            
            placeholder_text = f"[PDF Document: {os.path.basename(file_path)} - Could not extract text]"
            return placeholder_text, metadata
        
    async def _extract_pdf_text(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from PDF files"""
        if not fitz:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
        
        text_content = ""
        metadata = {
            'source_type': 'PDF',
            'source_file': os.path.basename(file_path),
            'pages': []
        }
        
        doc = None
        try:
            # Open the document
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        text_content += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                        metadata['pages'].append({
                            'page_number': page_num + 1,
                            'char_count': len(page_text)
                        })
                    
                    # Clean up the page object
                    page = None
                    
                except Exception as page_error:
                    logger.warning(f"Error processing page {page_num + 1} in {file_path}: {page_error}")
                    continue
            
            metadata['total_pages'] = total_pages
            
            # If no text was extracted, the PDF might be image-based
            if not text_content.strip():
                text_content = f"[PDF Document: {os.path.basename(file_path)} - {total_pages} pages - Text extraction returned empty, possibly image-based PDF]"
                
        except Exception as e:
            # If PDF opening fails, create a placeholder
            text_content = f"[PDF Document: {os.path.basename(file_path)} - PDF extraction failed: {str(e)}]"
            metadata['extraction_error'] = str(e)
            
        finally:
            # Ensure the document is properly closed
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass  # Ignore closing errors
        
        return text_content, metadata

    async def _extract_docx_text(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from DOCX files"""
        if not Document:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        
        try:
            doc = Document(file_path)
            
            text_content = ""
            paragraph_count = 0
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content += paragraph.text + "\n\n"
                    paragraph_count += 1
            
            metadata = {
                'source_type': 'Word Document',
                'source_file': os.path.basename(file_path),
                'paragraph_count': paragraph_count
            }
            
            return text_content, metadata
            
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {e}")

    async def _extract_doc_text(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from legacy DOC files"""
        # For .doc files, we'd need python-docx2txt or similar
        # For now, fall back to text extraction
        try:
            import docx2txt
            text_content = docx2txt.process(file_path)
            
            metadata = {
                'source_type': 'Word Document (Legacy)',
                'source_file': os.path.basename(file_path)
            }
            
            return text_content, metadata
            
        except ImportError:
            # Fallback to text file reading
            return await self._extract_text_file(file_path)

    async def _extract_text_file(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from plain text files"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text_content = f.read()
                    
                    metadata = {
                        'source_type': 'Text File',
                        'source_file': os.path.basename(file_path),
                        'encoding': encoding,
                        'file_size': os.path.getsize(file_path)
                    }
                    
                    return text_content, metadata
                    
                except UnicodeDecodeError:
                    continue
            
            raise Exception("Could not decode file with any common encoding")
            
        except Exception as e:
            raise Exception(f"Text file extraction failed: {e}")

    async def _extract_html_text(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from HTML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Simple HTML tag removal (could be enhanced with BeautifulSoup)
            text_content = re.sub(r'<[^>]+>', '', html_content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            metadata = {
                'source_type': 'HTML Document',
                'source_file': os.path.basename(file_path)
            }
            
            return text_content, metadata
            
        except Exception as e:
            raise Exception(f"HTML extraction failed: {e}")

    def _clean_text(self, text: str) -> str:
        """🧹 Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines -> double newline
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
        
        # Remove page markers and artifacts
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Fix common OCR/extraction artifacts
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
        text = re.sub(r'(\w)(\.|\!|\?)([A-Z])', r'\1\2 \3', text)  # Missing spaces after punctuation
        
        # Normalize quotes and special characters
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()

    def _create_semantic_chunks(self, text: str, chunk_size: int, 
                              overlap: int, preserve_structure: bool) -> List[str]:
        """🧠 Create semantically aware chunks"""
        if not text:
            return []
        
        # Define splitting hierarchy (from largest to smallest semantic units)
        separators = [
            '\n\n\n',      # Chapter/section breaks
            '\n\n',        # Paragraph breaks
            '. \n',        # Sentence + newline
            '.\n',         # Sentence + newline (no space)
            '. ',          # Sentence break
            '! ',          # Exclamation
            '? ',          # Question
            '\n',          # Line break
            '. ',          # Period (backup)
            ' '            # Word break (last resort)
        ]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # If we're not at the end, try to find a good break point
            if end < len(text):
                best_break = end
                
                # Look for the best semantic break within the chunk
                for separator in separators:
                    # Search backwards from the end for this separator
                    sep_pos = text.rfind(separator, start, end)
                    if sep_pos != -1 and sep_pos > start + chunk_size // 2:
                        best_break = sep_pos + len(separator)
                        break
                
                end = best_break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text and len(chunk_text) > 50:  # Only include substantial chunks
                chunks.append(chunk_text)
            
            # Move start position with overlap
            start = max(end - overlap, start + 1)  # Ensure progress
            
            if start >= len(text):
                break
        
        return chunks

    def _create_simple_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """📏 Create simple fixed-size chunks"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(chunk_text)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks

    def get_chunks(self) -> List[DocumentChunk]:
        """Get all processed chunks"""
        return self.processed_chunks.copy()

    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return self.processing_stats.copy()

    def export_chunks_to_json(self, output_file: str) -> None:
        """💾 Export chunks to JSON format"""
        import json
        
        chunks_data = []
        for chunk in self.processed_chunks:
            chunks_data.append({
                'text': chunk.text,
                'metadata': chunk.metadata,
                'chunk_id': chunk.chunk_id,
                'source_file': chunk.source_file,
                'chunk_index': chunk.chunk_index
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        print(f"📁 Exported {len(chunks_data)} chunks to {output_file}")

    def clear_chunks(self) -> None:
        """🗑️ Clear all processed chunks"""
        self.processed_chunks = []
        self.processing_stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'total_characters': 0,
            'errors': []
        }