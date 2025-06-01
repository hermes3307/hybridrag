#!/usr/bin/env python3
"""
🌐 Web Document Downloader - Improved Version
Smart document discovery and downloading from web URLs
Enhanced with better GitHub support and robust error handling
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
import json
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
            '.odt': 'OpenDocument Text',
            '.ppt': 'PowerPoint',
            '.pptx': 'PowerPoint',
            '.xls': 'Excel',
            '.xlsx': 'Excel'
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

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'url': self.url,
            'filename': self.filename,
            'size': self.size,
            'doc_type': self.doc_type,
            'language': self.language
        }

class WebDocumentDownloader:
    """🌐 Smart web document downloader with conversational feedback"""
    
    def __init__(self, download_dir: str = "downloaded_docs", github_token: Optional[str] = None):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # GitHub API token
        self.github_token = github_token or os.environ.get('GITHUB_API_TOKEN') or os.environ.get('GITHUB_TOKEN')
        
        # Supported document extensions
        self.doc_extensions = {
            '.pdf', '.doc', '.docx', '.txt', '.md', '.html', '.htm', 
            '.rtf', '.odt', '.tex', '.rst', '.ppt', '.pptx', '.xls', '.xlsx'
        }
        
        # GitHub API patterns (improved from the original script)
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

    async def scan_documents(self, url: str, pattern: Optional[str] = None, max_depth: int = 5) -> List[DocumentInfo]:
        """🔍 Scan URL for downloadable documents"""
        print(f"🔍 Scanning {url} for documents...")
        
        self.discovered_documents = []
        
        if 'github.com' in url:
            await self._scan_github_enhanced(url, pattern, max_depth)
        else:
            await self._scan_generic_website(url, pattern)
        
        self.download_stats['total_found'] = len(self.discovered_documents)
        
        print(f"✅ Found {len(self.discovered_documents)} documents!")
        return self.discovered_documents

    async def _scan_github_enhanced(self, url: str, pattern: Optional[str] = None, max_depth: int = 5) -> None:
        """Enhanced GitHub scanning with better API usage"""
        print("📁 Detected GitHub repository, using enhanced API scanning...")
        
        try:
            # Check if it's already a GitHub API URL
            if 'api.github.com/repos' in url:
                await self._scan_direct_github_api(url, pattern)
                return
                
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.strip('/').split('/')

            if len(path_parts) < 2:
                logger.error(f"Invalid GitHub repository URL: {url}")
                return

            owner = path_parts[0]
            repo_name = path_parts[1]
            
            # Determine the path within the repo
            api_path_parts = []
            if 'tree' in path_parts or 'blob' in path_parts:
                idx = path_parts.index('tree') if 'tree' in path_parts else path_parts.index('blob')
                if len(path_parts) > idx + 2:
                    api_path_parts = path_parts[idx+2:]
            
            api_path = '/'.join(api_path_parts)
            
            await self._scan_github_api_recursive(owner, repo_name, api_path, pattern, max_depth, 0)
            
        except Exception as e:
            logger.error(f"Error scanning GitHub repository: {e}")
            logger.error(traceback.format_exc())
                
    async def _scan_github_api_recursive(self, owner: str, repo_name: str, path: str = '', 
                                    pattern: Optional[str] = None, max_depth: int = 5, 
                                    current_depth: int = 0) -> None:
        """Recursively scan GitHub using API (enhanced from original script)"""
        if current_depth >= max_depth:
            logger.warning(f"Max depth ({max_depth}) reached for {owner}/{repo_name}/{path}")
            return

        try:
            # Make sure to preserve case for GitHub API - GitHub is case sensitive!
            api_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
            logger.info(f"Accessing GitHub API: {api_url} (Depth: {current_depth})")

            headers = {
                'User-Agent': 'Python-GitHub-Downloader/1.0',
                'Accept': 'application/vnd.github.v3+json'
            }
            if self.github_token:
                headers['Authorization'] = f"token {self.github_token}"

            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers, timeout=30) as response:
                    if response.status == 403:
                        logger.error("GitHub API rate limit exceeded. Consider using a GitHub token.")
                        return
                    
                    response.raise_for_status()
                    items = await response.json()

                    # Handle single file response
                    if isinstance(items, dict) and items.get('type') == 'file':
                        if self._matches_pattern(items.get('name', ''), pattern):
                            doc_info = self._create_document_info_from_github(items)
                            if doc_info:
                                self.discovered_documents.append(doc_info)
                        return

                    if not isinstance(items, list):
                        logger.warning(f"Unexpected API response format from {api_url}")
                        return

                    for item in items:
                        item_type = item.get('type')
                        item_name = item.get('name', '')

                        if item_type == 'file':
                            if self._matches_pattern(item_name, pattern):
                                doc_info = self._create_document_info_from_github(item)
                                if doc_info:
                                    self.discovered_documents.append(doc_info)
                        elif item_type == 'dir':
                            # Recursive call for subdirectories
                            await asyncio.sleep(0.1)  # Be polite to the API
                            new_path = f"{path}/{item_name}" if path else item_name
                            await self._scan_github_api_recursive(
                                owner, repo_name, new_path, pattern, max_depth, current_depth + 1
                            )

        except aiohttp.ClientError as e:
            logger.error(f"GitHub API request error for {owner}/{repo_name}/{path}: {e}")
        except Exception as e:
            logger.error(f"Error scanning GitHub path {owner}/{repo_name}/{path}: {e}")
            logger.error(traceback.format_exc())

    async def _scan_direct_github_api(self, api_url: str, pattern: Optional[str] = None) -> None:
        """Handle direct GitHub API URLs"""
        try:
            headers = {
                'User-Agent': 'Python-GitHub-Downloader/1.0',
                'Accept': 'application/vnd.github.v3+json'
            }
            if self.github_token:
                headers['Authorization'] = f"token {self.github_token}"

            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers, timeout=30) as response:
                    if response.status == 403:
                        logger.error("GitHub API rate limit exceeded. Consider using a GitHub token.")
                        return
                    
                    response.raise_for_status()
                    items = await response.json()

                    # Handle single file response
                    if isinstance(items, dict) and items.get('type') == 'file':
                        if self._matches_pattern(items.get('name', ''), pattern):
                            doc_info = self._create_document_info_from_github(items)
                            if doc_info:
                                self.discovered_documents.append(doc_info)
                        return

                    if not isinstance(items, list):
                        logger.warning(f"Unexpected API response format from {api_url}")
                        return

                    for item in items:
                        item_type = item.get('type')
                        item_name = item.get('name', '')

                        if item_type == 'file':
                            if self._matches_pattern(item_name, pattern):
                                doc_info = self._create_document_info_from_github(item)
                                if doc_info:
                                    self.discovered_documents.append(doc_info)

        except aiohttp.ClientError as e:
            logger.error(f"GitHub API request error: {e}")
        except Exception as e:
            logger.error(f"Error scanning GitHub API: {e}")

    def _create_document_info_from_github(self, item: Dict) -> Optional[DocumentInfo]:
        """Create DocumentInfo from GitHub API item"""
        try:
            filename = item.get('name', '')
            if not filename:
                return None
            
            # Check if it's a document we're interested in
            if not any(filename.lower().endswith(ext) for ext in self.doc_extensions):
                return None
            
            # Convert HTML URL to raw URL for downloading
            html_url = item.get('html_url', '')
            if not html_url:
                return None
                
            raw_url = self._convert_github_html_to_raw_url(html_url)
            
            return DocumentInfo(
                url=raw_url,
                filename=filename,
                size=item.get('size')
            )
        except Exception as e:
            logger.error(f"Error creating DocumentInfo from GitHub item: {e}")
            return None

    def _convert_github_html_to_raw_url(self, html_url: str) -> str:
        """Convert GitHub HTML file URL to raw content URL (from original script)"""
        parsed_url = urlparse(html_url)
        if 'blob' in parsed_url.path:
            raw_path = parsed_url.path.replace('/blob/', '/', 1)
            return f"https://raw.githubusercontent.com{raw_path}"
        else:
            logger.warning(f"URL {html_url} does not contain '/blob/'. Using as-is.")
            return html_url

    def _matches_pattern(self, filename: str, pattern: Optional[str]) -> bool:
        """Check if filename matches the given pattern"""
        if not pattern:
            return any(filename.lower().endswith(ext) for ext in self.doc_extensions)
        
        try:
            return bool(re.search(pattern, filename))
        except re.error:
            logger.warning(f"Invalid regex pattern: {pattern}")
            return any(filename.lower().endswith(ext) for ext in self.doc_extensions)

    async def _scan_generic_website(self, url: str, pattern: Optional[str] = None) -> None:
        """Enhanced generic website scanning"""
        print("🌐 Scanning website for document links...")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        await self._extract_document_links_enhanced(html, url, pattern)
                    else:
                        logger.error(f"Failed to access {url}: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"Error scanning website {url}: {e}")

    async def _extract_document_links_enhanced(self, html: str, base_url: str, pattern: Optional[str] = None) -> None:
        """Enhanced document link extraction"""
        # Multiple patterns for different link formats
        link_patterns = [
            re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE),
            re.compile(r'src=["\']([^"\']+)["\']', re.IGNORECASE),
            re.compile(r'data-url=["\']([^"\']+)["\']', re.IGNORECASE)
        ]
        
        all_links = set()
        for link_pattern in link_patterns:
            links = link_pattern.findall(html)
            all_links.update(links)
        
        for link in all_links:
            try:
                # Skip javascript, mailto, and other non-http links
                if link.startswith(('javascript:', 'mailto:', '#', 'tel:')):
                    continue
                
                # Resolve relative URLs
                absolute_url = urljoin(base_url, link)
                
                # Parse URL
                parsed = urlparse(absolute_url)
                if not parsed.scheme or not parsed.netloc:
                    continue
                
                path = parsed.path.lower()
                filename = os.path.basename(unquote(parsed.path))
                
                # Check if it's a document
                if filename and self._matches_pattern(filename, pattern):
                    doc_info = DocumentInfo(url=absolute_url, filename=filename)
                    self.discovered_documents.append(doc_info)
                    
            except Exception as e:
                logger.debug(f"Error processing link {link}: {e}")

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

    async def download_documents(self, filter_criteria: Optional[Dict] = None, max_files: int = 100) -> List[str]:
        """📥 Download documents with optional filtering"""
        if not self.discovered_documents:
            print("❌ No documents discovered yet. Run scan_documents() first.")
            return []
        
        # Apply filters
        docs_to_download = self._apply_filters(self.discovered_documents, filter_criteria or {})
        
        if not docs_to_download:
            print("❌ No documents match your criteria.")
            return []
        
        # Limit number of files
        if len(docs_to_download) > max_files:
            print(f"📄 Found {len(docs_to_download)} documents. Downloading first {max_files}.")
            docs_to_download = docs_to_download[:max_files]
        
        print(f"📥 Downloading {len(docs_to_download)} documents...")
        
        downloaded_files = []
        semaphore = asyncio.Semaphore(3)  # Limit concurrent downloads
        
        tasks = [
            self._download_single_document(doc, semaphore) 
            for doc in docs_to_download
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, str):  # Success - file path returned
                downloaded_files.append(result)
                self.download_stats['downloaded'] += 1
            else:  # Exception occurred
                self.download_stats['failed'] += 1
                doc_name = docs_to_download[i].filename if i < len(docs_to_download) else "unknown"
                logger.error(f"Download failed for {doc_name}: {result}")
        
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
        """Download a single document with enhanced error handling"""
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
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(doc.url, headers=headers, timeout=60) as response:
                        if response.status == 200:
                            # Ensure directory exists
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            async with aiofiles.open(file_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    await f.write(chunk)
                            
                            # Update stats
                            file_size = file_path.stat().st_size
                            self.download_stats['bytes_downloaded'] += file_size
                            
                            print(f"✅ Downloaded {safe_filename} ({file_size} bytes)")
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
        safe_name = re.sub(r'\s+', '_', safe_name)
        
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

    async def download_all(self, max_files: int = 100) -> List[str]:
        """📥 Download all discovered documents"""
        return await self.download_documents(max_files=max_files)

    async def download_pdfs_only(self, max_files: int = 100) -> List[str]:
        """📄 Download only PDF documents"""
        return await self.download_documents({'extensions': ['.pdf']}, max_files)

    async def download_by_language(self, language: str, max_files: int = 100) -> List[str]:
        """🌍 Download documents by language"""
        return await self.download_documents({'language': language}, max_files)

    async def download_by_type(self, doc_types: List[str], max_files: int = 100) -> List[str]:
        """📋 Download documents by type"""
        return await self.download_documents({'types': doc_types}, max_files)

    def get_discovered_documents(self) -> List[DocumentInfo]:
        """Get list of discovered documents"""
        return self.discovered_documents.copy()

    def save_document_list(self, filepath: str) -> None:
        """Save discovered documents to JSON file"""
        try:
            doc_data = [doc.to_dict() for doc in self.discovered_documents]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=2)
            print(f"📄 Saved document list to {filepath}")
        except Exception as e:
            logger.error(f"Error saving document list: {e}")

    def load_document_list(self, filepath: str) -> None:
        """Load documents from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            self.discovered_documents = []
            for data in doc_data:
                doc = DocumentInfo(
                    url=data['url'],
                    filename=data['filename'],
                    size=data.get('size'),
                    doc_type=data.get('doc_type'),
                    language=data.get('language')
                )
                self.discovered_documents.append(doc)
            
            print(f"📄 Loaded {len(self.discovered_documents)} documents from {filepath}")
        except Exception as e:
            logger.error(f"Error loading document list: {e}")


# Example usage and interactive functions
async def interactive_mode():
    """Interactive mode for document downloading"""
    print("=" * 60)
    print("   🌐 Web Document Downloader - Interactive Mode")
    print("=" * 60)

    # Get GitHub token if available
    github_token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GITHUB_API_TOKEN')
    if not github_token:
        token_input = input("Enter GitHub Personal Access Token (optional, press Enter to skip): ").strip()
        if token_input:
            github_token = token_input

    # Get URL with default
    default_url = "https://github.com/ALTIBASE/Documents/tree/master/Manuals/Altibase_7.3/eng/PDF"
    url = input(f"Enter URL to scan for documents [{default_url}]: ").strip()
    if not url:
        url = default_url
        print(f"Using default URL: {url}")

    # Get pattern
    pattern = input("Enter file pattern (regex, optional): ").strip() or None

    # Get download directory
    download_dir = input("Enter download directory [default: downloaded_docs]: ").strip() or "downloaded_docs"

    # Create downloader
    downloader = WebDocumentDownloader(download_dir, github_token)

    try:
        # Scan for documents
        documents = await downloader.scan_documents(url, pattern)
        
        if not documents:
            print("❌ No documents found")
            return

        # Show summary
        print("\n" + downloader.summarize_findings(documents))

        # Ask what to download
        print("\nDownload options:")
        print("1. Download all documents")
        print("2. Download PDFs only")
        print("3. Download by language")
        print("4. Custom filter")
        print("5. Save document list and exit")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            max_files = int(input("Max files to download [100]: ").strip() or "100")
            await downloader.download_all(max_files)
        elif choice == "2":
            max_files = int(input("Max PDF files to download [100]: ").strip() or "100")
            await downloader.download_pdfs_only(max_files)
        elif choice == "3":
            language = input("Enter language (English, Korean, Japanese, Chinese, Unknown): ").strip()
            max_files = int(input("Max files to download [100]: ").strip() or "100")
            await downloader.download_by_language(language, max_files)
        elif choice == "4":
            # Custom filter implementation
            filter_criteria = {}
            
            extensions = input("File extensions (comma-separated, e.g., .pdf,.docx): ").strip()
            if extensions:
                filter_criteria['extensions'] = [ext.strip() for ext in extensions.split(',')]
            
            filename_pattern = input("Filename pattern (regex): ").strip()
            if filename_pattern:
                filter_criteria['filename_pattern'] = filename_pattern
            
            max_files = int(input("Max files to download [100]: ").strip() or "100")
            await downloader.download_documents(filter_criteria, max_files)
        elif choice == "5":
            filename = input("Enter filename to save document list [documents.json]: ").strip() or "documents.json"
            downloader.save_document_list(filename)
        else:
            print("Invalid choice")

        # Show final stats
        stats = downloader.get_download_stats()
        print(f"\n📊 Final Statistics:")
        print(f"   Found: {stats['total_found']} documents")
        print(f"   Downloaded: {stats['downloaded']} files")
        print(f"   Failed: {stats['failed']} downloads")
        if stats['bytes_downloaded'] > 0:
            mb_downloaded = stats['bytes_downloaded'] / (1024 * 1024)
            print(f"   Total size: {mb_downloaded:.1f} MB")

    except Exception as e:
        logger.error(f"Error in interactive mode: {e}")
        logger.error(traceback.format_exc())


def main():
    """Main function for command line usage"""
    import argparse
    
    default_url = "https://github.com/ALTIBASE/Documents/tree/master/Manuals/Altibase_7.3/eng/PDF"
    
    parser = argparse.ArgumentParser(description='Web Document Downloader')
    parser.add_argument('url', nargs='?', help=f'URL to scan for documents (default: {default_url})')
    parser.add_argument('--pattern', help='File pattern (regex)')
    parser.add_argument('--download-dir', default='downloaded_docs', help='Download directory')
    parser.add_argument('--token', help='GitHub API token')
    parser.add_argument('--max-files', type=int, default=100, help='Maximum files to download')
    parser.add_argument('--max-depth', type=int, default=5, help='Maximum recursion depth')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--scan-only', action='store_true', help='Only scan, do not download')
    
    args = parser.parse_args()
    
    # Set default URL if not provided and not in interactive mode
    if not args.url and not args.interactive:
        args.url = default_url
    
    async def run():
        if args.interactive:
            await interactive_mode()
        else:
            print(f"🔍 Scanning URL: {args.url}")
            downloader = WebDocumentDownloader(args.download_dir, args.token)
            
            # Scan for documents
            documents = await downloader.scan_documents(args.url, args.pattern, args.max_depth)
            
            if documents:
                print(downloader.summarize_findings(documents))
                
                if not args.scan_only:
                    await downloader.download_all(args.max_files)
                    
                    stats = downloader.get_download_stats()
                    print(f"\n📊 Download Statistics:")
                    print(f"   Downloaded: {stats['downloaded']} files")
                    print(f"   Failed: {stats['failed']} downloads")
            else:
                print("❌ No documents found")
    
    asyncio.run(run())


if __name__ == "__main__":
    main()