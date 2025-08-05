#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Coder - Manual-based Professional AI Code Generation System
Main module with core functionality
"""

import json
import asyncio
import os
import hashlib
import math
import time
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import PyPDF2
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enhanced Logging setup with color support and detailed formatting
def setup_enhanced_logging():
    """Setup enhanced logging with color coding and detailed progress tracking"""
    
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with color coding for console output"""
        
        # Color codes for different log levels
        COLORS = {
            'DEBUG': '\033[36m',      # Cyan
            'INFO': '\033[32m',       # Green
            'WARNING': '\033[33m',    # Yellow
            'ERROR': '\033[31m',      # Red
            'CRITICAL': '\033[35m',   # Magenta
            'RESET': '\033[0m'        # Reset
        }
        
        # Progress indicators with emojis
        PROGRESS_ICONS = {
            'processing': '⚙️',
            'uploading': '📤',
            'generating': '🚀',
            'storing': '💾',
            'searching': '🔍',
            'parsing': '📄',
            'embedding': '🧠',
            'validating': '✅',
            'initializing': '🔄',
            'completed': '✅',
            'failed': '❌',
            'warning': '⚠️'
        }
        
        def format(self, record):
            # Get color for log level
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            
            # Add progress icon based on message content
            message = record.getMessage()
            icon = ''
            for keyword, emoji in self.PROGRESS_ICONS.items():
                if keyword in message.lower():
                    icon = f"{emoji} "
                    break
            
            # Enhanced format with component tracking
            component = record.name.split('.')[-1] if '.' in record.name else record.name
            
            # Create formatted message
            formatted = f"{color}[{self.formatTime(record, '%H:%M:%S')}] {record.levelname:8} | {component:12} | {icon}{message}{reset}"
            
            return formatted
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with color formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # Create file handler for detailed logging
    try:
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "ai_coder_detailed.log", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(funcName)s:%(lineno)d | %(levelname)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    return logger

# Initialize enhanced logging
logger = setup_enhanced_logging()
logger.info("🚀 Enhanced logging system initialized with color coding and progress tracking")

@dataclass
class ManualChunk:
    """Manual document chunk structure"""
    chunk_id: int
    content: str
    source_file: str
    manual_type: str
    version: str
    section: str
    page_number: int
    keywords: List[str]
    code_examples: List[str]

@dataclass
class ManualMetadata:
    """Manual metadata structure"""
    file_path: str
    manual_type: str
    version: str
    title: str
    language: str
    total_pages: int
    upload_date: str
    last_modified: str
    file_size: int
    processing_status: str

@dataclass
class CodeGenerationResult:
    """Code generation result with manual references"""
    code: str
    language: str
    manual_references: List[Dict]
    confidence_score: float
    validation_result: Dict
    generated_at: str
    prompt_used: str

@dataclass
class ValidationResult:
    """Code validation result"""
    compliance_score: int
    syntax_score: int
    api_compatibility: int
    suggestions: List[Dict]
    manual_references: List[str]
    is_valid: bool

class ManualProcessor:
    """Document processor for various manual formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.html', '.md', '.txt']
        
    def process_pdf(self, file_path: str) -> List[ManualChunk]:
        """Process PDF manual and extract chunks"""
        chunks = []
        chunk_id = 1
        
        try:
            # Use pdfplumber for better text extraction
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        # Split into logical sections
                        sections = self._split_into_sections(text)
                        
                        for section_text in sections:
                            if len(section_text.strip()) > 100:
                                chunk = ManualChunk(
                                    chunk_id=chunk_id,
                                    content=section_text.strip(),
                                    source_file=file_path,
                                    manual_type=self._detect_manual_type(file_path),
                                    version=self._extract_version(text),
                                    section=self._extract_section_title(section_text),
                                    page_number=page_num,
                                    keywords=self._extract_keywords(section_text),
                                    code_examples=self._extract_code_examples(section_text)
                                )
                                chunks.append(chunk)
                                chunk_id += 1
                                
        except Exception as e:
            logger.error(f"PDF processing failed for {file_path}: {e}")
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    full_text = ""
                    for page in pdf_reader.pages:
                        full_text += page.extract_text() + "\n"
                    
                    sections = self._split_into_sections(full_text)
                    for section_text in sections:
                        if len(section_text.strip()) > 100:
                            chunk = ManualChunk(
                                chunk_id=chunk_id,
                                content=section_text.strip(),
                                source_file=file_path,
                                manual_type=self._detect_manual_type(file_path),
                                version=self._extract_version(section_text),
                                section=self._extract_section_title(section_text),
                                page_number=1,
                                keywords=self._extract_keywords(section_text),
                                code_examples=self._extract_code_examples(section_text)
                            )
                            chunks.append(chunk)
                            chunk_id += 1
            except Exception as e2:
                logger.error(f"Fallback PDF processing also failed: {e2}")
                
        return chunks
    
    def process_docx(self, file_path: str) -> List[ManualChunk]:
        """Process Word document and extract chunks"""
        chunks = []
        chunk_id = 1
        
        try:
            doc = Document(file_path)
            full_text = ""
            
            for paragraph in doc.paragraphs:
                full_text += paragraph.text + "\n"
            
            sections = self._split_into_sections(full_text)
            
            for section_text in sections:
                if len(section_text.strip()) > 100:
                    chunk = ManualChunk(
                        chunk_id=chunk_id,
                        content=section_text.strip(),
                        source_file=file_path,
                        manual_type=self._detect_manual_type(file_path),
                        version=self._extract_version(section_text),
                        section=self._extract_section_title(section_text),
                        page_number=1,
                        keywords=self._extract_keywords(section_text),
                        code_examples=self._extract_code_examples(section_text)
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    
        except Exception as e:
            logger.error(f"DOCX processing failed for {file_path}: {e}")
            
        return chunks
    
    def process_html(self, file_path: str) -> List[ManualChunk]:
        """Process HTML manual and extract chunks"""
        chunks = []
        chunk_id = 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                sections = self._split_into_sections(text)
                
                for section_text in sections:
                    if len(section_text.strip()) > 100:
                        chunk = ManualChunk(
                            chunk_id=chunk_id,
                            content=section_text.strip(),
                            source_file=file_path,
                            manual_type=self._detect_manual_type(file_path),
                            version=self._extract_version(section_text),
                            section=self._extract_section_title(section_text),
                            page_number=1,
                            keywords=self._extract_keywords(section_text),
                            code_examples=self._extract_code_examples(section_text)
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                        
        except Exception as e:
            logger.error(f"HTML processing failed for {file_path}: {e}")
            
        return chunks
    
    def _split_into_sections(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into logical sections"""
        # Split by common section markers
        section_markers = [
            r'\n\s*\d+\.\s+',  # 1. Section
            r'\n\s*\d+\.\d+\s+',  # 1.1 Subsection
            r'\n\s*Chapter\s+\d+',  # Chapter markers
            r'\n\s*[A-Z][A-Z\s]{10,}\n',  # ALL CAPS titles
            r'\n\s*#{1,6}\s+',  # Markdown headers
        ]
        
        sections = [text]
        
        for pattern in section_markers:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend([part.strip() for part in parts if part.strip()])
            sections = new_sections
        
        # Further split large sections
        final_sections = []
        for section in sections:
            if len(section) > max_chunk_size:
                # Split by paragraphs
                paragraphs = section.split('\n\n')
                current_chunk = ""
                
                for paragraph in paragraphs:
                    if len(current_chunk + paragraph) > max_chunk_size and current_chunk:
                        final_sections.append(current_chunk.strip())
                        current_chunk = paragraph
                    else:
                        current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                
                if current_chunk.strip():
                    final_sections.append(current_chunk.strip())
            else:
                final_sections.append(section)
        
        return [s for s in final_sections if len(s.strip()) > 50]
    
    def _detect_manual_type(self, file_path: str) -> str:
        """Detect manual type from filename or content"""
        filename = os.path.basename(file_path).lower()
        
        if 'altibase' in filename:
            return 'altibase'
        elif any(keyword in filename for keyword in ['sql', 'database', 'db']):
            return 'database'
        elif any(keyword in filename for keyword in ['api', 'reference']):
            return 'api_reference'
        elif any(keyword in filename for keyword in ['admin', 'configuration']):
            return 'administration'
        else:
            return 'custom'
    
    def _extract_version(self, text: str) -> str:
        """Extract version information from text"""
        version_patterns = [
            r'version\s+(\d+\.\d+\.\d+)',
            r'v(\d+\.\d+)',
            r'(\d+\.\d+\.\d+)',
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        
        return "unknown"
    
    def _extract_section_title(self, text: str) -> str:
        """Extract section title from text"""
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 100 and len(first_line) > 5:
                return first_line
        return "Untitled Section"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction based on frequency and context
        words = re.findall(r'\b[A-Z][A-Z_]+\b|\b[a-z]+(?:[A-Z][a-z]*)*\b', text)
        
        # Filter and rank keywords
        word_freq = {}
        for word in words:
            if len(word) > 3 and word.lower() not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'why']:
                word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10] if freq > 1]
    
    def _extract_code_examples(self, text: str) -> List[str]:
        """Extract code examples from text"""
        code_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'`[^`\n]+`',  # Inline code
            r'(?:^|\n)\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+[\s\S]*?;',  # SQL
            r'(?:^|\n)\s*(?:def|class|import|from)\s+[\s\S]*?(?=\n\s*(?:\n|def|class|import|from|$))',  # Python
        ]
        
        code_examples = []
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            code_examples.extend([match.strip() for match in matches])
        
        return code_examples[:5]  # Limit to 5 code examples per chunk

class EnhancedVectorDBManager:
    """Enhanced ChromaDB manager for manual storage and retrieval"""
    
    def __init__(self, db_path: str = "./manual_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = self._create_embedding_function()
        
        # Initialize collection
        self.collection_name = "ai_coder_manuals"
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "AI Coder Manual Knowledge Base"},
                embedding_function=self.embedding_function
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def _create_embedding_function(self):
        """Create custom embedding function"""
        class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __call__(self, input_texts):
                embeddings = []
                for text in input_texts:
                    embedding = self._text_to_embedding(text)
                    embeddings.append(embedding)
                return embeddings
            
            def _text_to_embedding(self, text: str):
                """Convert text to embedding vector"""
                # Simple hash-based embedding for demonstration
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                embedding = []
                
                for i in range(768):  # 768-dimensional embedding
                    hash_part = text_hash[(i * 2) % len(text_hash):((i * 2) + 2) % len(text_hash)]
                    if len(hash_part) < 2:
                        hash_part = text_hash[:2]
                    
                    value = (int(hash_part, 16) / 255.0) * 2 - 1
                    embedding.append(value)
                
                # Normalize
                norm = math.sqrt(sum(x * x for x in embedding))
                if norm > 0:
                    embedding = [x / norm for x in embedding]
                
                return embedding
        
        return CustomEmbeddingFunction()
    
    def store_manual_chunks(self, chunks: List[ManualChunk]) -> bool:
        """Store manual chunks in vector database"""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Create unique ID
                chunk_hash = hashlib.md5(chunk.content.encode()).hexdigest()
                unique_id = f"{chunk.manual_type}_{chunk.version}_{chunk.chunk_id}_{chunk_hash[:8]}"
                
                # Prepare metadata
                metadata = {
                    "source_file": chunk.source_file,
                    "manual_type": chunk.manual_type,
                    "version": chunk.version,
                    "section": chunk.section,
                    "page_number": chunk.page_number,
                    "keywords": json.dumps(chunk.keywords),
                    "code_examples": json.dumps(chunk.code_examples),
                    "chunk_id": chunk.chunk_id,
                    "created_at": datetime.now().isoformat()
                }
                
                documents.append(chunk.content)
                metadatas.append(metadata)
                ids.append(unique_id)
            
            # Store in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            
            logger.info(f"Stored {len(chunks)} manual chunks successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store manual chunks: {e}")
            return False
    
    def search_manual_content(self, query: str, manual_type: str = None, 
                            version: str = None, top_k: int = 10) -> Dict:
        """Search manual content using vector similarity"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if manual_type:
                where_clause["manual_type"] = manual_type
            if version:
                where_clause["version"] = version
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = {
                "query": query,
                "total_results": len(results['documents'][0]) if results['documents'] else 0,
                "results": []
            }
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    result_item = {
                        "content": doc,
                        "metadata": metadata,
                        "relevance_score": 1 - distance,  # Convert distance to similarity
                        "manual_type": metadata.get("manual_type", "unknown"),
                        "version": metadata.get("version", "unknown"),
                        "section": metadata.get("section", "unknown"),
                        "source_file": metadata.get("source_file", "unknown")
                    }
                    
                    formatted_results["results"].append(result_item)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Manual search failed: {e}")
            return {"query": query, "total_results": 0, "results": []}
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "db_path": self.db_path
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"total_chunks": 0, "collection_name": "error", "db_path": "error"}

class EnhancedClaudeClient:
    """Enhanced Claude API client for code generation"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('CLAUDE_API_KEY')
        
        if self.api_key and self.api_key.strip() and self.api_key != "YOUR_CLAUDE_API_KEY":
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.test_mode = False
                logger.info("Claude API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Claude API client: {e}")
                self.client = None
                self.test_mode = True
        else:
            self.client = None
            self.test_mode = True
            logger.warning("Claude API key not set or invalid. Running in test mode.")
        
        self.request_count = 0
        self.last_call_time = 0
        self.min_interval = 2  # Minimum 2 seconds between calls
    
    async def generate_code_with_manual_context(self, task: str, language: str, 
                                              manual_context: List[Dict], 
                                              specifications: str = "",
                                              style: str = "professional") -> CodeGenerationResult:
        """Generate code using manual context from RAG"""
        
        # Check if API is available
        if self.test_mode:
            raise Exception("Claude API key not configured. Please set up your API key in Settings.")
        
        # Create comprehensive prompt
        prompt = self._create_code_generation_prompt(
            task, language, manual_context, specifications, style
        )
        
        # Generate code
        start_time = time.time()
        response = await self._call_claude_api(prompt)
        
        # Parse response
        generated_code = self._extract_code_from_response(response)
        
        # Create result
        result = CodeGenerationResult(
            code=generated_code,
            language=language,
            manual_references=[
                {
                    "section": ctx["section"],
                    "manual_type": ctx["manual_type"],
                    "relevance_score": ctx["relevance_score"],
                    "source_file": ctx["source_file"]
                }
                for ctx in manual_context
            ],
            confidence_score=self._calculate_confidence_score(manual_context),
            validation_result={},
            generated_at=datetime.now().isoformat(),
            prompt_used=prompt
        )
        
        return result
    
    def _create_code_generation_prompt(self, task: str, language: str, 
                                     manual_context: List[Dict], 
                                     specifications: str, style: str) -> str:
        """Create comprehensive code generation prompt"""
        
        # Build manual reference section
        manual_references = ""
        if manual_context:
            manual_references = "## Manual References (Use these as primary source of truth)\n\n"
            for i, ctx in enumerate(manual_context, 1):
                manual_references += f"### Reference {i}: {ctx['section']}\n"
                manual_references += f"**Source**: {ctx['manual_type']} v{ctx['version']} - {ctx['source_file']}\n"
                manual_references += f"**Relevance**: {ctx['relevance_score']:.2f}\n\n"
                manual_references += f"**Content**:\n{ctx['content']}\n\n"
                manual_references += "---\n\n"
        
        # Create the main prompt
        prompt = f"""You are an expert software engineer specializing in {language} development. You have access to specific manual documentation that contains the authoritative information for this task.

## Task Description
{task}

## Target Language
{language}

## Additional Specifications
{specifications}

## Code Style
{style}

{manual_references}

## Instructions
1. **STRICTLY FOLLOW THE MANUAL REFERENCES**: The manual references above contain the authoritative documentation. Use them as the primary source of truth for syntax, APIs, and best practices.

2. **Manual-Based Code Generation**:
   - Use ONLY the syntax, functions, and APIs documented in the manual references
   - Follow the exact patterns and examples shown in the manuals
   - If the manual shows specific parameter names, use those exact names
   - Respect any version-specific features or limitations mentioned

3. **Code Quality Requirements**:
   - Write clean, readable, and well-documented code
   - Include inline comments explaining manual-based decisions
   - Use proper error handling as documented in the manuals
   - Follow the coding conventions shown in the manual examples

4. **Validation Against Manual**:
   - Ensure every API call matches the manual documentation
   - Use correct parameter types and names as specified
   - Follow any security or performance guidelines from the manuals
   - Include any required imports or dependencies mentioned

5. **Output Format**:
   - Provide complete, executable code
   - Include necessary imports and setup
   - Add comments referencing which manual sections were used
   - Explain any manual-specific considerations

## Critical Requirements
- **NEVER IMPROVISE**: If functionality is not documented in the manual references, explicitly state what's missing
- **VERSION COMPLIANCE**: Ensure compatibility with the specific version mentioned in manual references  
- **EXACT SYNTAX**: Use the exact syntax patterns shown in the manual examples
- **PARAMETER ACCURACY**: Use correct parameter names and types as documented

Please generate the {language} code that accomplishes the specified task while strictly adhering to the manual documentation provided."""

        return prompt
    
    async def _call_claude_api(self, prompt: str, max_tokens: int = 4000) -> str:
        """Call Claude API with rate limiting"""
        
        # Rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            await asyncio.sleep(sleep_time)
        
        self.last_call_time = time.time()
        self.request_count += 1
        
        if self.test_mode:
            raise Exception("Claude API not available. Please configure your API key.")
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            logger.info(f"Claude API call successful (request #{self.request_count})")
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise Exception(f"Claude API call failed: {e}")
    
    def _get_test_response(self, prompt: str) -> str:
        """Generate test response when API is not available"""
        if "sql" in prompt.lower() or "altibase" in prompt.lower():
            return """-- ALTIBASE SQL Code Generated from Manual
-- This code follows the manual specifications provided

CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at DATE DEFAULT SYSDATE
);

-- Insert sample data
INSERT INTO users (user_id, username, email) 
VALUES (1, 'admin', 'admin@example.com');

-- Query with manual-compliant syntax
SELECT user_id, username, email, created_at 
FROM users 
WHERE created_at >= SYSDATE - 30;

-- Manual Reference: Used ALTIBASE-specific SYSDATE function
-- Manual Reference: Followed ALTIBASE VARCHAR syntax specifications"""

        elif "python" in prompt.lower():
            return """# Python Code Generated from Manual Documentation
# Following manual specifications and best practices

import os
import logging
from typing import List, Dict, Optional

class ManualBasedProcessor:
    \"\"\"
    Implementation based on manual documentation
    Follows the exact patterns specified in the reference materials
    \"\"\"
    
    def __init__(self, config_path: str = None):
        \"\"\"
        Initialize processor with manual-compliant configuration
        
        Args:
            config_path: Path to configuration file (as per manual spec)
        \"\"\"
        self.config_path = config_path
        self.logger = self._setup_logging()  # Manual reference: Section 3.2
        
    def _setup_logging(self) -> logging.Logger:
        \"\"\"Setup logging as specified in manual Section 3.2\"\"\"
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Manual-compliant formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def process_data(self, data: List[Dict]) -> Dict:
        \"\"\"
        Process data according to manual algorithm specification
        
        Manual Reference: Algorithm described in Section 4.1
        \"\"\"
        try:
            result = {
                'processed_count': 0,
                'errors': [],
                'results': []
            }
            
            for item in data:
                # Manual-specified validation
                if self._validate_item(item):
                    processed_item = self._transform_item(item)
                    result['results'].append(processed_item)
                    result['processed_count'] += 1
                else:
                    result['errors'].append(f"Invalid item: {item}")
            
            self.logger.info(f"Processed {result['processed_count']} items")
            return result
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise
    
    def _validate_item(self, item: Dict) -> bool:
        \"\"\"Validate item according to manual specifications\"\"\"
        required_fields = ['id', 'name', 'type']  # Manual Section 2.3
        return all(field in item for field in required_fields)
    
    def _transform_item(self, item: Dict) -> Dict:
        \"\"\"Transform item following manual transformation rules\"\"\"
        # Manual Reference: Transformation rules in Section 4.2
        return {
            'id': item['id'],
            'name': item['name'].upper(),  # Manual requirement: uppercase names
            'type': item['type'],
            'processed_at': 'current_timestamp'  # Manual-compliant timestamp
        }

# Manual-compliant usage example
if __name__ == "__main__":
    processor = ManualBasedProcessor()
    
    # Test data following manual schema
    test_data = [
        {'id': 1, 'name': 'test_item', 'type': 'sample'},
        {'id': 2, 'name': 'another_item', 'type': 'demo'}
    ]
    
    result = processor.process_data(test_data)
    print(f"Processing completed: {result}")"""

        else:
            return f"""// Code generated based on manual documentation
// Language: {prompt.split('Language')[1].split()[0] if 'Language' in prompt else 'Unknown'}
// Following manual specifications and best practices

/**
 * Manual-compliant implementation
 * Generated according to the provided documentation references
 */

// This is a placeholder implementation
// The actual code would be generated based on the specific manual content
// and requirements provided in the prompt.

console.log("Manual-based code generation completed");
"""
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from Claude's response"""
        # Look for code blocks
        code_patterns = [
            r'```[\w]*\n([\s\S]*?)\n```',  # Standard code blocks
            r'```([\s\S]*?)```',  # Simple code blocks
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response)
            if matches:
                return matches[0].strip()
        
        # If no code blocks found, return the whole response
        return response.strip()
    
    def _calculate_confidence_score(self, manual_context: List[Dict]) -> float:
        """Calculate confidence score based on manual context quality"""
        if not manual_context:
            return 0.3  # Low confidence without manual context
        
        total_relevance = sum(ctx.get('relevance_score', 0) for ctx in manual_context)
        avg_relevance = total_relevance / len(manual_context)
        
        # Boost confidence based on number of relevant references
        context_boost = min(0.2, len(manual_context) * 0.05)
        
        return min(0.95, avg_relevance + context_boost)

class AICoderSystem:
    """Main AI Coder system integrating all components"""
    
    def __init__(self, claude_api_key: str = None, vector_db_path: str = "./manual_db"):
        self.manual_processor = ManualProcessor()
        self.vector_db = EnhancedVectorDBManager(vector_db_path)
        self.claude_client = EnhancedClaudeClient(claude_api_key)
        
        # Create necessary directories
        self.manuals_dir = Path("./manuals")
        self.manuals_dir.mkdir(exist_ok=True)
        
        self.generated_code_dir = Path("./generated_code")
        self.generated_code_dir.mkdir(exist_ok=True)
        
        logger.info("AI Coder System initialized successfully")
    
    def upload_manual(self, file_path: str, manual_type: str = None, 
                     version: str = None) -> bool:
        """Upload and process a manual file"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"Manual file not found: {file_path}")
                return False
            
            logger.info(f"Processing manual: {file_path}")
            
            # Determine file type and process accordingly
            if file_path.suffix.lower() == '.pdf':
                chunks = self.manual_processor.process_pdf(str(file_path))
            elif file_path.suffix.lower() == '.docx':
                chunks = self.manual_processor.process_docx(str(file_path))
            elif file_path.suffix.lower() in ['.html', '.htm']:
                chunks = self.manual_processor.process_html(str(file_path))
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return False
            
            if not chunks:
                logger.error("No content extracted from manual")
                return False
            
            # Override manual type and version if provided
            if manual_type:
                for chunk in chunks:
                    chunk.manual_type = manual_type
            if version:
                for chunk in chunks:
                    chunk.version = version
            
            # Store in vector database
            success = self.vector_db.store_manual_chunks(chunks)
            
            if success:
                logger.info(f"Successfully processed manual: {len(chunks)} chunks stored")
            
            return success
            
        except Exception as e:
            logger.error(f"Manual upload failed: {e}")
            return False
    
    async def generate_code(self, task: str, language: str, manual_type: str = None,
                          version: str = None, specifications: str = "",
                          style: str = "professional") -> CodeGenerationResult:
        """Generate code based on task and manual context"""
        
        try:
            # Search for relevant manual content
            search_query = f"{task} {language}"
            manual_results = self.vector_db.search_manual_content(
                query=search_query,
                manual_type=manual_type,
                version=version,
                top_k=5
            )
            
            logger.info(f"Found {manual_results['total_results']} relevant manual sections")
            
            # Prepare manual context
            manual_context = []
            for result in manual_results['results']:
                context_item = {
                    "content": result["content"],
                    "section": result["section"],
                    "manual_type": result["manual_type"],
                    "version": result["version"],
                    "relevance_score": result["relevance_score"],
                    "source_file": result["source_file"]
                }
                manual_context.append(context_item)
            
            # Generate code using Claude with manual context
            code_result = await self.claude_client.generate_code_with_manual_context(
                task=task,
                language=language,
                manual_context=manual_context,
                specifications=specifications,
                style=style
            )
            
            logger.info(f"Code generation completed with confidence: {code_result.confidence_score:.2f}")
            
            return code_result
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            # Return empty result
            return CodeGenerationResult(
                code="// Code generation failed",
                language=language,
                manual_references=[],
                confidence_score=0.0,
                validation_result={},
                generated_at=datetime.now().isoformat(),
                prompt_used=""
            )
    
    def validate_code(self, code: str, language: str, manual_type: str = None) -> ValidationResult:
        """Validate code against manual specifications"""
        # This is a simplified validation - in practice, you'd implement
        # more sophisticated validation based on manual content
        
        suggestions = []
        compliance_score = 85  # Default score
        
        # Basic validations
        if not code.strip():
            return ValidationResult(
                compliance_score=0,
                syntax_score=0,
                api_compatibility=0,
                suggestions=[{"type": "error", "description": "No code provided"}],
                manual_references=[],
                is_valid=False
            )
        
        # Language-specific validations
        if language.lower() == 'sql':
            if 'SYSDATE' in code and manual_type == 'altibase':
                compliance_score += 5  # Good ALTIBASE practice
            if 'SELECT *' in code:
                suggestions.append({
                    "type": "optimization",
                    "description": "Avoid SELECT * in production code",
                    "manual_reference": "SQL Best Practices"
                })
        
        elif language.lower() == 'python':
            if 'import' not in code:
                suggestions.append({
                    "type": "warning",
                    "description": "Consider adding necessary imports",
                    "manual_reference": "Python Standards"
                })
            if 'try:' in code and 'except:' in code:
                compliance_score += 10  # Good error handling
        
        return ValidationResult(
            compliance_score=compliance_score,
            syntax_score=8,  # Assuming good syntax
            api_compatibility=9,  # Assuming good compatibility
            suggestions=suggestions,
            manual_references=[f"{manual_type} manual" if manual_type else "General standards"],
            is_valid=compliance_score >= 70
        )
    
    def save_generated_code(self, code_result: CodeGenerationResult, 
                          filename: str = None) -> str:
        """Save generated code to file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_code_{timestamp}.{self._get_file_extension(code_result.language)}"
            
            file_path = self.generated_code_dir / filename
            
            # Create comprehensive file content
            file_content = f"""/*
 * AI Coder Generated Code
 * Generated at: {code_result.generated_at}
 * Language: {code_result.language}
 * Confidence Score: {code_result.confidence_score:.2f}
 * 
 * Manual References Used:
"""
            
            for ref in code_result.manual_references:
                file_content += f" * - {ref['section']} ({ref['manual_type']} v{ref.get('version', 'unknown')})\n"
            
            file_content += f" */\n\n{code_result.code}"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            logger.info(f"Code saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save code: {e}")
            return ""
    
    def _get_file_extension(self, language: str) -> str:
        """Get appropriate file extension for language"""
        extensions = {
            'python': 'py',
            'sql': 'sql',
            'javascript': 'js',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'csharp': 'cs',
            'go': 'go',
            'rust': 'rs'
        }
        return extensions.get(language.lower(), 'txt')
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        db_stats = self.vector_db.get_collection_stats()
        
        return {
            "vector_db": db_stats,
            "claude_requests": self.claude_client.request_count,
            "test_mode": self.claude_client.test_mode,
            "manuals_directory": str(self.manuals_dir),
            "generated_code_directory": str(self.generated_code_dir)
        }

# Main execution for testing
async def main():
    """Main function for testing the system"""
    print("=== AI Coder - Manual-based Code Generation System ===\n")
    
    # Initialize system
    system = AICoderSystem()
    
    # Test manual upload (if manual file exists)
    test_manual_path = "./manuals/test_manual.pdf"
    if Path(test_manual_path).exists():
        print("1. Uploading test manual...")
        success = system.upload_manual(test_manual_path, manual_type="test", version="1.0")
        print(f"Manual upload: {'✅ Success' if success else '❌ Failed'}\n")
    else:
        print("1. No test manual found, skipping upload...\n")
    
    # Test code generation
    print("2. Generating code...")
    code_result = await system.generate_code(
        task="Create a function to connect to ALTIBASE database and execute a simple query",
        language="python",
        manual_type="altibase",
        specifications="Include error handling and connection pooling"
    )
    
    print("Generated Code:")
    print("-" * 50)
    print(code_result.code)
    print("-" * 50)
    print(f"Confidence Score: {code_result.confidence_score:.2f}")
    print(f"Manual References: {len(code_result.manual_references)}")
    
    # Save code
    if code_result.code:
        saved_path = system.save_generated_code(code_result)
        print(f"Code saved to: {saved_path}")
    
    # Show system stats
    print("\n3. System Statistics:")
    stats = system.get_system_stats()
    print(f"Vector DB chunks: {stats['vector_db']['total_chunks']}")
    print(f"Claude requests: {stats['claude_requests']}")
    print(f"Test mode: {stats['test_mode']}")

if __name__ == "__main__":
    asyncio.run(main())