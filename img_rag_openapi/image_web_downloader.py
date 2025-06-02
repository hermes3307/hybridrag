#!/usr/bin/env python3
"""
🖼️ Image Web Downloader
Smart image discovery and downloading from web URLs with AI-powered content analysis
"""

import os
import re
import asyncio
import aiohttp
import aiofiles
import base64
from urllib.parse import urlparse, urljoin, unquote
from typing import List, Dict, Optional, Set, Tuple
import mimetypes
from pathlib import Path
import time
import logging
import json
import traceback
from PIL import Image
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageInfo:
    """Information about a discovered image"""
    def __init__(self, url: str, filename: str, size: Optional[int] = None, 
                 image_type: Optional[str] = None, width: Optional[int] = None,
                 height: Optional[int] = None, alt_text: Optional[str] = None):
        self.url = url
        self.filename = filename
        self.size = size
        self.image_type = image_type or self._detect_image_type()
        self.width = width
        self.height = height
        self.alt_text = alt_text
        self.ai_description = None  # Will be filled by AI analysis
        self.ai_metadata = {}  # Additional AI-extracted metadata
    
    def _detect_image_type(self) -> str:
        """Detect image type from filename"""
        ext = Path(self.filename).suffix.lower()
        type_mapping = {
            '.jpg': 'JPEG Image',
            '.jpeg': 'JPEG Image',
            '.png': 'PNG Image',
            '.gif': 'GIF Image',
            '.bmp': 'BMP Image',
            '.webp': 'WebP Image',
            '.svg': 'SVG Image',
            '.tiff': 'TIFF Image',
            '.tif': 'TIFF Image',
            '.ico': 'Icon Image',
            '.heic': 'HEIC Image',
            '.heif': 'HEIF Image'
        }
        return type_mapping.get(ext, 'Unknown Image')
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'url': self.url,
            'filename': self.filename,
            'size': self.size,
            'image_type': self.image_type,
            'width': self.width,
            'height': self.height,
            'alt_text': self.alt_text,
            'ai_description': self.ai_description,
            'ai_metadata': self.ai_metadata
        }

class ImageWebDownloader:
    """🖼️ Smart web image downloader with conversational feedback"""
    
    def __init__(self, download_dir: str = "downloaded_images"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Supported image extensions
        self.image_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', 
            '.svg', '.tiff', '.tif', '.ico', '.heic', '.heif'
        }
        
        # 일반 파일 확장자도 포함 (이미지가 아닌 파일들)
        self.file_extensions = {
            '.pdf', '.doc', '.docx', '.txt', '.zip', '.rar', 
            '.xlsx', '.xls', '.ppt', '.pptx', '.mp4', '.avi', 
            '.mp3', '.wav', '.css', '.js', '.html', '.xml', '.json'
        }
        
        self.discovered_images = []
        self.download_stats = {
            'total_found': 0,
            'downloaded': 0,
            'failed': 0,
            'bytes_downloaded': 0,
            'ai_analyzed': 0
        }

    async def scan_images(self, url: str, pattern: Optional[str] = None, max_depth: int = 5, 
                         include_all_files: bool = False) -> List[ImageInfo]:
        """🔍 Scan URL for downloadable files (images and other files)"""
        print(f"🔍 Scanning {url} for {'all files' if include_all_files else 'images'}...")
        
        self.discovered_images = []
        await self._scan_generic_website_enhanced(url, pattern, include_all_files)
        
        self.download_stats['total_found'] = len(self.discovered_images)
        
        print(f"✅ Found {len(self.discovered_images)} {'files' if include_all_files else 'images'}!")
        return self.discovered_images

    async def _scan_generic_website_enhanced(self, url: str, pattern: Optional[str] = None, 
                                           include_all_files: bool = False) -> None:
        """Enhanced generic website scanning for all types of files"""
        print("🌐 Scanning website for file links...")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            timeout = aiohttp.ClientTimeout(total=60, connect=30)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.get(url, headers=headers, allow_redirects=True) as response:
                        if response.status == 200:
                            html = await response.text()
                            await self._extract_file_links_enhanced(html, url, pattern, include_all_files)
                        else:
                            logger.error(f"Failed to access {url}: HTTP {response.status}")
                            # 에러가 있어도 디렉토리 리스팅을 시도해보자
                            await self._try_directory_listing(url, pattern, include_all_files, session)
                except asyncio.TimeoutError:
                    logger.error(f"Timeout accessing {url}")
                except aiohttp.ClientError as e:
                    logger.error(f"Client error accessing {url}: {e}")
                    # 직접 디렉토리 리스팅 시도
                    await self._try_directory_listing(url, pattern, include_all_files, session)
                        
        except Exception as e:
            logger.error(f"Error scanning website {url}: {e}")

    async def _try_directory_listing(self, base_url: str, pattern: Optional[str], 
                                   include_all_files: bool, session: aiohttp.ClientSession) -> None:
        """Try to access directory listing or common file patterns"""
        print("🔍 Trying directory listing approach...")
        
        # 일반적인 디렉토리 리스팅 패턴들
        common_patterns = [
            '',  # 기본 디렉토리
            'index.html',
            'index.php',
            'files/',
            'images/',
            'img/',
            'assets/',
            'media/',
            'downloads/',
            'docs/',
            'documents/'
        ]
        
        for pattern_path in common_patterns:
            try:
                test_url = urljoin(base_url, pattern_path)
                async with session.get(test_url, timeout=30) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        if 'text/html' in content_type:
                            html = await response.text()
                            if self._looks_like_directory_listing(html):
                                print(f"📁 Found directory listing at: {test_url}")
                                await self._extract_file_links_enhanced(html, test_url, pattern, include_all_files)
                                break
            except Exception as e:
                logger.debug(f"Failed to check {test_url}: {e}")

    def _looks_like_directory_listing(self, html: str) -> bool:
        """Check if HTML looks like a directory listing"""
        indicators = [
            'Index of /',
            'Directory listing',
            'Parent Directory',
            '[DIR]',
            'folder.gif',
            'text.gif',
            'image.gif',
            '<pre>',  # Apache style listings often use <pre>
            'Last modified',
            'Size</th>',
            'Name</th>'
        ]
        
        html_lower = html.lower()
        return any(indicator.lower() in html_lower for indicator in indicators)

    async def _extract_file_links_enhanced(self, html: str, base_url: str, pattern: Optional[str] = None, 
                                         include_all_files: bool = False) -> None:
        """Enhanced file link extraction for all file types"""
        found_files = set()
        
        # 1. 표준 <a> 태그에서 파일 링크 추출
        link_patterns = [
            # 기본 링크 패턴
            re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>', re.IGNORECASE),
            # Apache 스타일 디렉토리 리스팅
            re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>', re.IGNORECASE),
            # 직접 파일 URL 패턴
            re.compile(r'https?://[^\s<>"\']+\.(?:' + '|'.join(ext[1:] for ext in self.image_extensions | self.file_extensions) + r')', re.IGNORECASE)
        ]
        
        # 2. 이미지 태그에서도 추출
        if not include_all_files:
            img_patterns = [
                re.compile(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*(?:alt=["\']([^"\']*)["\'])?[^>]*>', re.IGNORECASE),
                re.compile(r'data-src=["\']([^"\']+\.(?:jpg|jpeg|png|gif|bmp|webp|svg|tiff|tif|ico|heic|heif))["\']', re.IGNORECASE),
                re.compile(r'background-image:\s*url\(["\']?([^"\'()]+\.(?:jpg|jpeg|png|gif|bmp|webp|svg|tiff|tif|ico|heic|heif))["\']?\)', re.IGNORECASE)
            ]
            link_patterns.extend(img_patterns)
        
        # 모든 패턴으로 링크 추출
        for pattern_obj in link_patterns:
            matches = pattern_obj.findall(html)
            for match in matches:
                if isinstance(match, tuple):
                    file_url = match[0] if match[0] else (match[1] if len(match) > 1 else '')
                    alt_text = match[1] if len(match) > 1 and match[1] else ''
                else:
                    file_url = match
                    alt_text = ''
                
                if file_url:
                    found_files.add((file_url, alt_text))
        
        # 3. 파일 처리
        for file_url, alt_text in found_files:
            try:
                # 잘못된 URL 스킵
                if file_url.startswith(('data:', 'javascript:', 'mailto:', '#', '?')):
                    continue
                
                # 상대 URL을 절대 URL로 변환
                absolute_url = urljoin(base_url, file_url)
                
                # URL 파싱
                parsed = urlparse(absolute_url)
                if not parsed.scheme or not parsed.netloc:
                    continue
                
                # 파일명 추출
                filename = os.path.basename(unquote(parsed.path))
                if not filename:
                    # 파일명이 없으면 URL에서 추출 시도
                    path_parts = parsed.path.strip('/').split('/')
                    filename = path_parts[-1] if path_parts else 'unknown'
                
                # 파일 확장자 확인
                file_ext = Path(filename).suffix.lower()
                
                # 파일 타입 확인
                is_image = file_ext in self.image_extensions
                is_other_file = file_ext in self.file_extensions
                
                # 확장자가 없는 경우 Content-Type으로 확인 (선택적)
                if not file_ext:
                    # URL 끝에 파라미터가 있을 수 있으므로 ? 이전까지만 확인
                    clean_path = parsed.path.split('?')[0]
                    if any(clean_path.lower().endswith(ext) for ext in (self.image_extensions | self.file_extensions)):
                        filename = os.path.basename(clean_path)
                        file_ext = Path(filename).suffix.lower()
                        is_image = file_ext in self.image_extensions
                        is_other_file = file_ext in self.file_extensions
                
                # 필터링 조건
                should_include = False
                
                if include_all_files:
                    # 모든 파일 포함
                    should_include = is_image or is_other_file or not file_ext
                else:
                    # 이미지만 포함
                    should_include = is_image
                
                # 패턴 매칭
                if should_include and pattern:
                    try:
                        should_include = bool(re.search(pattern, filename, re.IGNORECASE))
                    except re.error:
                        logger.warning(f"Invalid regex pattern: {pattern}")
                
                if should_include:
                    # 중복 제거 (URL 기준)
                    if not any(img.url == absolute_url for img in self.discovered_images):
                        image_info = ImageInfo(
                            url=absolute_url,
                            filename=filename,
                            alt_text=alt_text if alt_text else None
                        )
                        
                        # 파일 크기 예측 시도 (HEAD 요청은 무거우므로 생략)
                        self.discovered_images.append(image_info)
                        
            except Exception as e:
                logger.debug(f"Error processing file link {file_url}: {e}")

    def summarize_findings(self, images: List[ImageInfo]) -> str:
        """📊 Create a human-friendly summary of found files"""
        if not images:
            return "❌ No files found"
        
        # Group by type
        by_type = {}
        total_size = 0
        
        for img in images:
            # By type
            img_type = img.image_type
            if img_type not in by_type:
                by_type[img_type] = []
            by_type[img_type].append(img)
            
            # Size
            if img.size:
                total_size += img.size
        
        summary = f"📁 **File Discovery Summary**\n\n"
        summary += f"📸 Total files: {len(images)}\n"
        
        if total_size > 0:
            size_mb = total_size / (1024 * 1024)
            summary += f"💾 Total size: {size_mb:.1f} MB\n"
        
        summary += "\n📋 **By File Type:**\n"
        for img_type, imgs in by_type.items():
            summary += f"   • {img_type}: {len(imgs)} files\n"
        
        # Show some examples
        summary += "\n📝 **Sample Files:**\n"
        for img in images[:10]:
            alt_info = f" (Alt: {img.alt_text})" if img.alt_text else ""
            summary += f"   • {img.filename} ({img.image_type}){alt_info}\n"
        
        if len(images) > 10:
            summary += f"   ... and {len(images) - 10} more files\n"
        
        return summary

    async def download_images(self, filter_criteria: Optional[Dict] = None, max_files: int = 100,
                            chunk_size: int = 8192, max_concurrent: int = 5) -> List[str]:
        """📥 Download files with enhanced settings"""
        if not self.discovered_images:
            print("❌ No files discovered yet. Run scan_images() first.")
            return []
        
        # Apply filters
        images_to_download = self._apply_filters(self.discovered_images, filter_criteria or {})
        
        if not images_to_download:
            print("❌ No files match your criteria.")
            return []
        
        # Limit number of files
        if len(images_to_download) > max_files:
            print(f"📸 Found {len(images_to_download)} files. Downloading first {max_files}.")
            images_to_download = images_to_download[:max_files]
        
        print(f"📥 Downloading {len(images_to_download)} files...")
        
        downloaded_files = []
        semaphore = asyncio.Semaphore(max_concurrent)  # Configurable concurrent downloads
        
        tasks = [
            self._download_single_image_enhanced(img, semaphore, chunk_size) 
            for img in images_to_download
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, str):  # Success - file path returned
                downloaded_files.append(result)
                self.download_stats['downloaded'] += 1
            else:  # Exception occurred
                self.download_stats['failed'] += 1
                img_name = images_to_download[i].filename if i < len(images_to_download) else "unknown"
                logger.error(f"Download failed for {img_name}: {result}")
        
        print(f"✅ Downloaded {len(downloaded_files)} files successfully!")
        if self.download_stats['failed'] > 0:
            print(f"⚠️  {self.download_stats['failed']} downloads failed")
        
        return downloaded_files

    async def _download_single_image_enhanced(self, img: ImageInfo, semaphore: asyncio.Semaphore, 
                                            chunk_size: int = 8192) -> str:
        """Enhanced single file download with better error handling and progress"""
        async with semaphore:
            try:
                # Create safe filename
                safe_filename = self._create_safe_filename(img.filename)
                file_path = self.download_dir / safe_filename
                
                # Skip if already exists
                if file_path.exists():
                    print(f"⏭️  Skipping {safe_filename} (already exists)")
                    return str(file_path)
                
                print(f"📥 Downloading {safe_filename}...")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Sec-Fetch-Dest': 'image',
                    'Sec-Fetch-Mode': 'no-cors',
                    'Sec-Fetch-Site': 'cross-site'
                }
                
                timeout = aiohttp.ClientTimeout(total=300, connect=30)  # 5분 타임아웃
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(img.url, headers=headers, allow_redirects=True) as response:
                        if response.status == 200:
                            # Ensure directory exists
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # 파일 크기 확인
                            content_length = response.headers.get('Content-Length')
                            if content_length:
                                file_size = int(content_length)
                                print(f"📦 File size: {file_size / 1024 / 1024:.2f} MB")
                            
                            # 스트리밍 다운로드
                            downloaded_bytes = 0
                            async with aiofiles.open(file_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(chunk_size):
                                    await f.write(chunk)
                                    downloaded_bytes += len(chunk)
                            
                            # Update stats
                            actual_size = file_path.stat().st_size
                            self.download_stats['bytes_downloaded'] += actual_size
                            img.size = actual_size
                            
                            # Try to get image dimensions (only for actual images)
                            if any(img.filename.lower().endswith(ext) for ext in self.image_extensions):
                                try:
                                    with Image.open(file_path) as pil_img:
                                        img.width, img.height = pil_img.size
                                except Exception as e:
                                    logger.debug(f"Could not get dimensions for {safe_filename}: {e}")
                            
                            print(f"✅ Downloaded {safe_filename} ({actual_size / 1024 / 1024:.2f} MB)")
                            return str(file_path)
                        else:
                            raise aiohttp.ClientError(f"HTTP {response.status}: {response.reason}")
                            
            except asyncio.TimeoutError:
                print(f"⏰ Timeout downloading {img.filename}")
                raise
            except aiohttp.ClientError as e:
                print(f"🌐 Network error downloading {img.filename}: {e}")
                raise
            except Exception as e:
                print(f"❌ Failed to download {img.filename}: {e}")
                raise

    def _apply_filters(self, images: List[ImageInfo], criteria: Dict) -> List[ImageInfo]:
        """Apply filtering criteria to file list"""
        filtered = images
        
        # Filter by type
        if 'types' in criteria:
            allowed_types = criteria['types']
            filtered = [img for img in filtered if img.image_type in allowed_types]
        
        # Filter by extension
        if 'extensions' in criteria:
            allowed_exts = criteria['extensions']
            filtered = [img for img in filtered 
                       if any(img.filename.lower().endswith(ext) for ext in allowed_exts)]
        
        # Filter by size
        if 'max_size' in criteria:
            max_size = criteria['max_size']
            filtered = [img for img in filtered if not img.size or img.size <= max_size]
        
        if 'min_size' in criteria:
            min_size = criteria['min_size']
            filtered = [img for img in filtered if not img.size or img.size >= min_size]
        
        # Filter by filename pattern
        if 'filename_pattern' in criteria:
            pattern = re.compile(criteria['filename_pattern'], re.IGNORECASE)
            filtered = [img for img in filtered if pattern.search(img.filename)]
        
        return filtered

    def _create_safe_filename(self, filename: str) -> str:
        """Create a safe filename for the filesystem"""
        # Remove/replace unsafe characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        safe_name = re.sub(r'\s+', '_', safe_name)
        
        # 파일명이 너무 길면 자르기 (확장자는 보존)
        if len(safe_name) > 200:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:200-len(ext)] + ext
        
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
        stats = self.download_stats.copy()
        if stats['bytes_downloaded'] > 0:
            stats['size_mb'] = stats['bytes_downloaded'] / 1024 / 1024
        return stats

    async def download_all_files(self, max_files: int = 100, max_concurrent: int = 5) -> List[str]:
        """📥 Download all discovered files"""
        return await self.download_images(max_files=max_files, max_concurrent=max_concurrent)

    async def download_images_only(self, max_files: int = 100) -> List[str]:
        """🖼️ Download only image files"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff', '.tif', '.ico', '.heic', '.heif']
        return await self.download_images({'extensions': image_extensions}, max_files)

    async def download_by_extension(self, extensions: List[str], max_files: int = 100) -> List[str]:
        """📋 Download files by extension"""
        return await self.download_images({'extensions': extensions}, max_files)

    async def download_large_files_only(self, min_size_mb: float = 1.0, max_files: int = 100) -> List[str]:
        """📦 Download only large files"""
        min_size_bytes = int(min_size_mb * 1024 * 1024)
        return await self.download_images({'min_size': min_size_bytes}, max_files)

    def get_discovered_images(self) -> List[ImageInfo]:
        """Get list of discovered files"""
        return self.discovered_images.copy()

    def save_file_list(self, filepath: str) -> None:
        """Save discovered files to JSON file"""
        try:
            file_data = [img.to_dict() for img in self.discovered_images]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(file_data, f, ensure_ascii=False, indent=2)
            print(f"📄 Saved file list to {filepath}")
        except Exception as e:
            logger.error(f"Error saving file list: {e}")

    def load_file_list(self, filepath: str) -> None:
        """Load files from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            self.discovered_images = []
            for data in file_data:
                img = ImageInfo(
                    url=data['url'],
                    filename=data['filename'],
                    size=data.get('size'),
                    image_type=data.get('image_type'),
                    width=data.get('width'),
                    height=data.get('height'),
                    alt_text=data.get('alt_text')
                )
                img.ai_description = data.get('ai_description')
                img.ai_metadata = data.get('ai_metadata', {})
                self.discovered_images.append(img)
            
            print(f"📄 Loaded {len(self.discovered_images)} files from {filepath}")
        except Exception as e:
            logger.error(f"Error loading file list: {e}")


if __name__ == "__main__":
    # Example usage
    async def main():
        downloader = ImageWebDownloader()
        
        # Scan for all files (not just images)
        url = "http://211.177.94.209:8080/moment/images/"
        files = await downloader.scan_images(url, include_all_files=True)
        
        if files:
            print(downloader.summarize_findings(files))
            
            # Download all files with higher concurrency
            downloaded = await downloader.download_all_files(max_files=50, max_concurrent=8)
            print(f"Downloaded {len(downloaded)} files")
            
            # Print statistics
            stats = downloader.get_download_stats()
            print(f"📊 Download Stats: {stats}")
        
    asyncio.run(main())