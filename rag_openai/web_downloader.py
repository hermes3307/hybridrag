#!/usr/bin/env python3
"""
🌐 Web Document Downloader
Smart document discovery and downloading from web URLs
"""

import os
import re
import asyncio
import aiohttp
import aiofiles
from urllib.parse import urlparse, urljoin, unquote
from typing import List, Dict, Optional, Set
import mimetypes
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

class DocumentInfo:
    """Information about a discovered document"""
    def __init__(self, url: str, filename: str, size: Optional[int] = None, 
                 doc_type: Optional[str] = None, language: Optional[str] = None):
        self.url = url
        self.filename = filename
        self.size = size
        self.doc_type = doc_type or self._detect_type()
        self.language = language or self._detect_language()
    
    def _detect_type(self) -> str:
        """Detect document type from filename"""
        ext = Path(self.filename).suffix.lower()
        type_mapping = {
            '.pdf': 'PDF Document',
            '.doc': 'Word Document',
            '.docx': 'Word Document', 
            '.txt': 'Text File',
            '.md': 'Markdown',
            '.html': 'HTML Document',
            '.htm': 'HTML Document',
            '.rtf': 'Rich Text Format',
            '.odt': 'OpenDocument Text'
        }
        return type_mapping.get(ext, 'Unknown')
    
    def _detect_language(self) -> str:
        """Detect language from filename patterns"""
        filename_lower = self.filename.lower()
        if any(lang in filename_lower for lang in ['eng', 'english', 'en_']):
            return 'English'
        elif any(lang in filename_lower for lang in ['kor', 'korean', 'ko_', 'kr_']):
            return 'Korean'
        elif any(lang in filename_lower for lang in ['jpn', 'japanese', 'ja_']):
            return 'Japanese'
        elif any(lang in filename_lower for lang in ['chn', 'chinese', 'zh_']):
            return 'Chinese'
        return 'Unknown'

class WebDocumentDownloader:
    """🌐 Smart web document downloader with conversational feedback"""
    
    def __init__(self, download_dir: str = "downloaded_docs"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Supported document extensions
        self.doc_extensions = {
            '.pdf', '.doc', '.docx', '.txt', '.md', '.html', '.htm', 
            '.rtf', '.odt', '.tex', '.rst'
        }
        
        # GitHub API patterns
        self.github_patterns = {
            'repo': re.compile(r'github\.com/([^/]+)/([^/]+)'),
            'tree': re.compile(r'github\.com/([^/]+)/([^/]+)/tree/([^/]+)(?:/(.+))?'),
            'blob': re.compile(r'github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)')
        }
        
        self.discovered_documents = []
        self.download_stats = {
            'total_found': 0,
            'downloaded': 0,
            'failed': 0,
            'bytes_downloaded': 0
        }

    async def scan_documents(self, url: str) -> List[DocumentInfo]:
        """🔍 Scan URL for downloadable documents"""
        print(f"🔍 Scanning {url} for documents...")
        
        self.discovered_documents = []
        
        if 'github.com' in url:
            await self._scan_github(url)
        else:
            await self._scan_generic_website(url)
        
        self.download_stats['total_found'] = len(self.discovered_documents)
        
        print(f"✅ Found {len(self.discovered_documents)} documents!")
        return self.discovered_documents

    async def _scan_github(self, url: str) -> None:
        """Scan GitHub repository for documents"""
        print("📁 Detected GitHub repository, using API...")
        
        # Parse GitHub URL
        if '/tree/' in url:
            match = self.github_patterns['tree'].match(url.replace('https://', ''))
            if match:
                owner, repo, branch, path = match.groups()
                await self._scan_github_api(owner, repo, path or '', branch)
        else:
            match = self.github_patterns['repo'].match(url.replace('https://', ''))
            if match:
                owner, repo = match.groups()
                await self._scan_github_api(owner, repo)

    async def _scan_github_api(self, owner: str, repo: str, path: str = '', branch: str = 'master'):
        """Scan GitHub using API"""
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        if branch != 'master':
            api_url += f"?ref={branch}"
        
        async with aiohttp.ClientSession() as session:
            await self._scan_github_recursive(session, api_url, owner, repo, branch)

    async def _scan_github_recursive(self, session: aiohttp.ClientSession, api_url: str, 
                                   owner: str, repo: str, branch: str, depth: int = 0):
        """Recursively scan GitHub directories"""
        if depth > 5:  # Prevent infinite recursion
            return
        
        try:
            async with session.get(api_url) as response:
                if response.status == 200:
                    items = await response.json()
                    
                    for item in items:
                        if item['type'] == 'file':
                            filename = item['name']
                            if any(filename.lower().endswith(ext) for ext in self.doc_extensions):
                                doc_info = DocumentInfo(
                                    url=item['download_url'],
                                    filename=filename,
                                    size=item.get('size')
                                )
                                self.discovered_documents.append(doc_info)
                                
                        elif item['type'] == 'dir':
                            # Recursively scan subdirectories
                            await self._scan_github_recursive(
                                session, item['url'], owner, repo, branch, depth + 1
                            )
                            
        except Exception as e:
            logger.warning(f"Error scanning GitHub directory: {e}")

    async def _scan_generic_website(self, url: str) -> None:
        """Scan generic website for document links"""
        print("🌐 Scanning website for document links...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        await self._extract_document_links(html, url)
                        
        except Exception as e:
            logger.error(f"Error scanning website: {e}")

    async def _extract_document_links(self, html: str, base_url: str) -> None:
        """Extract document links from HTML"""
        # Simple regex-based extraction (could be enhanced with BeautifulSoup)
        link_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
        links = link_pattern.findall(html)
        
        for link in links:
            # Resolve relative URLs
            absolute_url = urljoin(base_url, link)
            
            # Check if it's a document
            parsed = urlparse(absolute_url)
            path = parsed.path.lower()
            
            if any(path.endswith(ext) for ext in self.doc_extensions):
                filename = os.path.basename(unquote(parsed.path))
                if filename:
                    doc_info = DocumentInfo(url=absolute_url, filename=filename)
                    self.discovered_documents.append(doc_info)

    def summarize_findings(self, documents: List[DocumentInfo]) -> str:
        """📊 Create a human-friendly summary of found documents"""
        if not documents:
            return "❌ No documents found"
        
        # Group by type
        by_type = {}
        by_language = {}
        total_size = 0
        
        for doc in documents:
            # By type
            doc_type = doc.doc_type
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(doc)
            
            # By language
            lang = doc.language
            if lang not in by_language:
                by_language[lang] = 0
            by_language[lang] += 1
            
            # Size
            if doc.size:
                total_size += doc.size
        
        summary = f"📊 **Document Discovery Summary**\n\n"
        summary += f"📄 Total documents: {len(documents)}\n"
        
        if total_size > 0:
            size_mb = total_size / (1024 * 1024)
            summary += f"💾 Total size: {size_mb:.1f} MB\n"
        
        summary += "\n📋 **By Document Type:**\n"
        for doc_type, docs in by_type.items():
            summary += f"   • {doc_type}: {len(docs)} files\n"
        
        summary += "\n🌍 **By Language:**\n"
        for lang, count in by_language.items():
            summary += f"   • {lang}: {count} files\n"
        
        # Show some examples
        summary += "\n📝 **Sample Files:**\n"
        for doc in documents[:5]:
            summary += f"   • {doc.filename} ({doc.doc_type})\n"
        
        if len(documents) > 5:
            summary += f"   ... and {len(documents) - 5} more files\n"
        
        return summary

    async def download_documents(self, filter_criteria: Optional[Dict] = None) -> List[str]:
        """📥 Download documents with optional filtering"""
        if not self.discovered_documents:
            print("❌ No documents discovered yet. Run scan_documents() first.")
            return []
        
        # Apply filters
        docs_to_download = self._apply_filters(self.discovered_documents, filter_criteria or {})
        
        if not docs_to_download:
            print("❌ No documents match your criteria.")
            return []
        
        print(f"📥 Downloading {len(docs_to_download)} documents...")
        
        downloaded_files = []
        semaphore = asyncio.Semaphore(5)  # Limit concurrent downloads
        
        tasks = [
            self._download_single_document(doc, semaphore) 
            for doc in docs_to_download
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, str):  # Success - file path returned
                downloaded_files.append(result)
                self.download_stats['downloaded'] += 1
            else:  # Exception occurred
                self.download_stats['failed'] += 1
                if not isinstance(result, Exception):
                    logger.error(f"Download failed: {result}")
        
        print(f"✅ Downloaded {len(downloaded_files)} files successfully!")
        if self.download_stats['failed'] > 0:
            print(f"⚠️  {self.download_stats['failed']} downloads failed")
        
        return downloaded_files

    def _apply_filters(self, documents: List[DocumentInfo], criteria: Dict) -> List[DocumentInfo]:
        """Apply filtering criteria to document list"""
        filtered = documents
        
        # Filter by type
        if 'types' in criteria:
            allowed_types = criteria['types']
            filtered = [doc for doc in filtered if doc.doc_type in allowed_types]
        
        # Filter by extension
        if 'extensions' in criteria:
            allowed_exts = criteria['extensions']
            filtered = [doc for doc in filtered 
                       if any(doc.filename.lower().endswith(ext) for ext in allowed_exts)]
        
        # Filter by language
        if 'language' in criteria:
            target_lang = criteria['language']
            filtered = [doc for doc in filtered if doc.language == target_lang]
        
        # Filter by size
        if 'max_size' in criteria:
            max_size = criteria['max_size']
            filtered = [doc for doc in filtered if not doc.size or doc.size <= max_size]
        
        # Filter by filename pattern
        if 'filename_pattern' in criteria:
            pattern = re.compile(criteria['filename_pattern'], re.IGNORECASE)
            filtered = [doc for doc in filtered if pattern.search(doc.filename)]
        
        return filtered

    async def _download_single_document(self, doc: DocumentInfo, semaphore: asyncio.Semaphore) -> str:
        """Download a single document"""
        async with semaphore:
            try:
                # Create safe filename
                safe_filename = self._create_safe_filename(doc.filename)
                file_path = self.download_dir / safe_filename
                
                # Skip if already exists
                if file_path.exists():
                    print(f"⏭️  Skipping {safe_filename} (already exists)")
                    return str(file_path)
                
                print(f"📥 Downloading {safe_filename}...")
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(doc.url) as response:
                        if response.status == 200:
                            async with aiofiles.open(file_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    await f.write(chunk)
                            
                            # Update stats
                            if doc.size:
                                self.download_stats['bytes_downloaded'] += doc.size
                            
                            print(f"✅ Downloaded {safe_filename}")
                            return str(file_path)
                        else:
                            raise aiohttp.ClientError(f"HTTP {response.status}")
                            
            except Exception as e:
                print(f"❌ Failed to download {doc.filename}: {e}")
                raise e

    def _create_safe_filename(self, filename: str) -> str:
        """Create a safe filename for the filesystem"""
        # Remove/replace unsafe characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Handle duplicates
        base_path = self.download_dir / safe_name
        counter = 1
        
        while base_path.exists():
            name, ext = os.path.splitext(safe_name)
            safe_name = f"{name}_{counter}{ext}"
            base_path = self.download_dir / safe_name
            counter += 1
        
        return safe_name

    def get_download_stats(self) -> Dict:
        """Get download statistics"""
        return self.download_stats.copy()

    async def download_all(self) -> List[str]:
        """📥 Download all discovered documents"""
        return await self.download_documents()

    async def download_pdfs_only(self) -> List[str]:
        """📄 Download only PDF documents"""
        return await self.download_documents({'extensions': ['.pdf']})

    async def download_by_language(self, language: str) -> List[str]:
        """🌍 Download documents by language"""
        return await self.download_documents({'language': language})

    async def download_by_type(self, doc_types: List[str]) -> List[str]:
        """📋 Download documents by type"""
        return await self.download_documents({'types': doc_types})

    def get_discovered_documents(self) -> List[DocumentInfo]:
        """Get list of discovered documents"""
        return self.discovered_documents.copy()