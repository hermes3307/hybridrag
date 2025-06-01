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
    
    def __init__(self, download_dir: str = "downloaded_images", github_token: Optional[str] = None):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # GitHub API token
        self.github_token = github_token or os.environ.get('GITHUB_API_TOKEN') or os.environ.get('GITHUB_TOKEN')
        
        # Supported image extensions
        self.image_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', 
            '.svg', '.tiff', '.tif', '.ico', '.heic', '.heif'
        }
        
        # GitHub API patterns
        self.github_patterns = {
            'repo': re.compile(r'github\.com/([^/]+)/([^/]+)'),
            'tree': re.compile(r'github\.com/([^/]+)/([^/]+)/tree/([^/]+)(?:/(.+))?'),
            'blob': re.compile(r'github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)')
        }
        
        self.discovered_images = []
        self.download_stats = {
            'total_found': 0,
            'downloaded': 0,
            'failed': 0,
            'bytes_downloaded': 0,
            'ai_analyzed': 0
        }

    async def scan_images(self, url: str, pattern: Optional[str] = None, max_depth: int = 5) -> List[ImageInfo]:
        """🔍 Scan URL for downloadable images"""
        print(f"🔍 Scanning {url} for images...")
        
        self.discovered_images = []
        
        if 'github.com' in url:
            await self._scan_github_images(url, pattern, max_depth)
        else:
            await self._scan_generic_website_images(url, pattern)
        
        self.download_stats['total_found'] = len(self.discovered_images)
        
        print(f"✅ Found {len(self.discovered_images)} images!")
        return self.discovered_images

    async def _scan_github_images(self, url: str, pattern: Optional[str] = None, max_depth: int = 5) -> None:
        """Enhanced GitHub image scanning with better API usage"""
        print("📁 Detected GitHub repository, using enhanced API scanning for images...")
        
        try:
            # Check if it's already a GitHub API URL
            if 'api.github.com/repos' in url:
                await self._scan_direct_github_api_images(url, pattern)
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
            
            await self._scan_github_api_recursive_images(owner, repo_name, api_path, pattern, max_depth, 0)
            
        except Exception as e:
            logger.error(f"Error scanning GitHub repository for images: {e}")
            logger.error(traceback.format_exc())

    async def _scan_github_api_recursive_images(self, owner: str, repo_name: str, path: str = '', 
                                          pattern: Optional[str] = None, max_depth: int = 5, 
                                          current_depth: int = 0) -> None:
        """Recursively scan GitHub using API for images"""
        if current_depth >= max_depth:
            logger.warning(f"Max depth ({max_depth}) reached for {owner}/{repo_name}/{path}")
            return

        try:
            api_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
            logger.info(f"Accessing GitHub API for images: {api_url} (Depth: {current_depth})")

            headers = {
                'User-Agent': 'Python-GitHub-Image-Downloader/1.0',
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
                        if self._matches_image_pattern(items.get('name', ''), pattern):
                            image_info = self._create_image_info_from_github(items)
                            if image_info:
                                self.discovered_images.append(image_info)
                        return

                    if not isinstance(items, list):
                        logger.warning(f"Unexpected API response format from {api_url}")
                        return

                    for item in items:
                        item_type = item.get('type')
                        item_name = item.get('name', '')

                        if item_type == 'file':
                            if self._matches_image_pattern(item_name, pattern):
                                image_info = self._create_image_info_from_github(item)
                                if image_info:
                                    self.discovered_images.append(image_info)
                        elif item_type == 'dir':
                            # Recursive call for subdirectories
                            await asyncio.sleep(0.1)  # Be polite to the API
                            new_path = f"{path}/{item_name}" if path else item_name
                            await self._scan_github_api_recursive_images(
                                owner, repo_name, new_path, pattern, max_depth, current_depth + 1
                            )

        except aiohttp.ClientError as e:
            logger.error(f"GitHub API request error for images {owner}/{repo_name}/{path}: {e}")
        except Exception as e:
            logger.error(f"Error scanning GitHub path for images {owner}/{repo_name}/{path}: {e}")
            logger.error(traceback.format_exc())

    async def _scan_direct_github_api_images(self, api_url: str, pattern: Optional[str] = None) -> None:
        """Handle direct GitHub API URLs for images"""
        try:
            headers = {
                'User-Agent': 'Python-GitHub-Image-Downloader/1.0',
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
                        if self._matches_image_pattern(items.get('name', ''), pattern):
                            image_info = self._create_image_info_from_github(items)
                            if image_info:
                                self.discovered_images.append(image_info)
                        return

                    if not isinstance(items, list):
                        logger.warning(f"Unexpected API response format from {api_url}")
                        return

                    for item in items:
                        item_type = item.get('type')
                        item_name = item.get('name', '')

                        if item_type == 'file':
                            if self._matches_image_pattern(item_name, pattern):
                                image_info = self._create_image_info_from_github(item)
                                if image_info:
                                    self.discovered_images.append(image_info)

        except aiohttp.ClientError as e:
            logger.error(f"GitHub API request error for images: {e}")
        except Exception as e:
            logger.error(f"Error scanning GitHub API for images: {e}")

    def _create_image_info_from_github(self, item: Dict) -> Optional[ImageInfo]:
        """Create ImageInfo from GitHub API item"""
        try:
            filename = item.get('name', '')
            if not filename:
                return None
            
            # Check if it's an image we're interested in
            if not any(filename.lower().endswith(ext) for ext in self.image_extensions):
                return None
            
            # Convert HTML URL to raw URL for downloading
            html_url = item.get('html_url', '')
            if not html_url:
                return None
                
            raw_url = self._convert_github_html_to_raw_url(html_url)
            
            return ImageInfo(
                url=raw_url,
                filename=filename,
                size=item.get('size')
            )
        except Exception as e:
            logger.error(f"Error creating ImageInfo from GitHub item: {e}")
            return None

    def _convert_github_html_to_raw_url(self, html_url: str) -> str:
        """Convert GitHub HTML file URL to raw content URL"""
        parsed_url = urlparse(html_url)
        if 'blob' in parsed_url.path:
            raw_path = parsed_url.path.replace('/blob/', '/', 1)
            return f"https://raw.githubusercontent.com{raw_path}"
        else:
            logger.warning(f"URL {html_url} does not contain '/blob/'. Using as-is.")
            return html_url

    def _matches_image_pattern(self, filename: str, pattern: Optional[str]) -> bool:
        """Check if filename matches the given pattern for images"""
        if not pattern:
            return any(filename.lower().endswith(ext) for ext in self.image_extensions)
        
        try:
            return bool(re.search(pattern, filename))
        except re.error:
            logger.warning(f"Invalid regex pattern: {pattern}")
            return any(filename.lower().endswith(ext) for ext in self.image_extensions)

    async def _scan_generic_website_images(self, url: str, pattern: Optional[str] = None) -> None:
        """Enhanced generic website image scanning"""
        print("🌐 Scanning website for image links...")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        await self._extract_image_links_enhanced(html, url, pattern)
                    else:
                        logger.error(f"Failed to access {url}: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"Error scanning website {url} for images: {e}")

    async def _extract_image_links_enhanced(self, html: str, base_url: str, pattern: Optional[str] = None) -> None:
        """Enhanced image link extraction"""
        # Multiple patterns for different image link formats
        img_patterns = [
            re.compile(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*(?:alt=["\']([^"\']*)["\'])?[^>]*>', re.IGNORECASE),
            re.compile(r'<img[^>]+alt=["\']([^"\']*)["\'][^>]*src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE),
            re.compile(r'src=["\']([^"\']+\.(?:jpg|jpeg|png|gif|bmp|webp|svg|tiff|tif|ico|heic|heif))["\']', re.IGNORECASE),
            re.compile(r'data-src=["\']([^"\']+\.(?:jpg|jpeg|png|gif|bmp|webp|svg|tiff|tif|ico|heic|heif))["\']', re.IGNORECASE),
            re.compile(r'background-image:\s*url\(["\']?([^"\'()]+\.(?:jpg|jpeg|png|gif|bmp|webp|svg|tiff|tif|ico|heic|heif))["\']?\)', re.IGNORECASE)
        ]
        
        found_images = set()
        
        for pattern_obj in img_patterns:
            matches = pattern_obj.findall(html)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle diffeƒrent regex group patterns
                    if len(match) == 2:
                        # Either (src, alt) or (alt, src)
                        if any(match[0].lower().endswith(ext) for ext in self.image_extensions):
                            found_images.add((match[0], match[1] if match[1] else ''))
                        elif any(match[1].lower().endswith(ext) for ext in self.image_extensions):
                            found_images.add((match[1], match[0] if match[0] else ''))
                    else:
                        found_images.add((match[0], ''))
                else:
                    found_images.add((match, ''))
        
        for img_url, alt_text in found_images:
            try:
                # Skip data URLs and invalid URLs
                if img_url.startswith(('data:', 'javascript:', 'mailto:', '#')):
                    continue
                
                # Resolve relative URLs
                absolute_url = urljoin(base_url, img_url)
                
                # Parse URL
                parsed = urlparse(absolute_url)
                if not parsed.scheme or not parsed.netloc:
                    continue
                
                filename = os.path.basename(unquote(parsed.path))
                
                # Check if it's an image
                if filename and self._matches_image_pattern(filename, pattern):
                    image_info = ImageInfo(
                        url=absolute_url, 
                        filename=filename,
                        alt_text=alt_text if alt_text else None
                    )
                    self.discovered_images.append(image_info)
                    
            except Exception as e:
                logger.debug(f"Error processing image link {img_url}: {e}")

    def summarize_findings(self, images: List[ImageInfo]) -> str:
        """📊 Create a human-friendly summary of found images"""
        if not images:
            return "❌ No images found"
        
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
        
        summary = f"🖼️ **Image Discovery Summary**\n\n"
        summary += f"📸 Total images: {len(images)}\n"
        
        if total_size > 0:
            size_mb = total_size / (1024 * 1024)
            summary += f"💾 Total size: {size_mb:.1f} MB\n"
        
        summary += "\n📋 **By Image Type:**\n"
        for img_type, imgs in by_type.items():
            summary += f"   • {img_type}: {len(imgs)} files\n"
        
        # Show some examples
        summary += "\n📝 **Sample Images:**\n"
        for img in images[:5]:
            alt_info = f" (Alt: {img.alt_text})" if img.alt_text else ""
            summary += f"   • {img.filename} ({img.image_type}){alt_info}\n"
        
        if len(images) > 5:
            summary += f"   ... and {len(images) - 5} more images\n"
        
        return summary

    async def download_images(self, filter_criteria: Optional[Dict] = None, max_files: int = 100) -> List[str]:
        """📥 Download images with optional filtering"""
        if not self.discovered_images:
            print("❌ No images discovered yet. Run scan_images() first.")
            return []
        
        # Apply filters
        images_to_download = self._apply_filters(self.discovered_images, filter_criteria or {})
        
        if not images_to_download:
            print("❌ No images match your criteria.")
            return []
        
        # Limit number of files
        if len(images_to_download) > max_files:
            print(f"📸 Found {len(images_to_download)} images. Downloading first {max_files}.")
            images_to_download = images_to_download[:max_files]
        
        print(f"📥 Downloading {len(images_to_download)} images...")
        
        downloaded_files = []
        semaphore = asyncio.Semaphore(3)  # Limit concurrent downloads
        
        tasks = [
            self._download_single_image(img, semaphore) 
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
        
        print(f"✅ Downloaded {len(downloaded_files)} images successfully!")
        if self.download_stats['failed'] > 0:
            print(f"⚠️  {self.download_stats['failed']} downloads failed")
        
        return downloaded_files

    def _apply_filters(self, images: List[ImageInfo], criteria: Dict) -> List[ImageInfo]:
        """Apply filtering criteria to image list"""
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
        
        # Filter by filename pattern
        if 'filename_pattern' in criteria:
            pattern = re.compile(criteria['filename_pattern'], re.IGNORECASE)
            filtered = [img for img in filtered if pattern.search(img.filename)]
        
        return filtered

    async def _download_single_image(self, img: ImageInfo, semaphore: asyncio.Semaphore) -> str:
        """Download a single image with enhanced error handling and basic analysis"""
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
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(img.url, headers=headers, timeout=60) as response:
                        if response.status == 200:
                            # Ensure directory exists
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            async with aiofiles.open(file_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    await f.write(chunk)
                            
                            # Update stats
                            file_size = file_path.stat().st_size
                            self.download_stats['bytes_downloaded'] += file_size
                            
                            # Try to get image dimensions
                            try:
                                with Image.open(file_path) as pil_img:
                                    img.width, img.height = pil_img.size
                            except Exception as e:
                                logger.debug(f"Could not get dimensions for {safe_filename}: {e}")
                            
                            print(f"✅ Downloaded {safe_filename} ({file_size} bytes)")
                            return str(file_path)
                        else:
                            raise aiohttp.ClientError(f"HTTP {response.status}")
                            
            except Exception as e:
                print(f"❌ Failed to download {img.filename}: {e}")
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

    async def download_all_images(self, max_files: int = 100) -> List[str]:
        """📥 Download all discovered images"""
        return await self.download_images(max_files=max_files)

    async def download_jpegs_only(self, max_files: int = 100) -> List[str]:
        """📄 Download only JPEG images"""
        return await self.download_images({'extensions': ['.jpg', '.jpeg']}, max_files)

    async def download_pngs_only(self, max_files: int = 100) -> List[str]:
        """🖼️ Download only PNG images"""
        return await self.download_images({'extensions': ['.png']}, max_files)

    async def download_by_type(self, img_types: List[str], max_files: int = 100) -> List[str]:
        """📋 Download images by type"""
        return await self.download_images({'types': img_types}, max_files)

    def get_discovered_images(self) -> List[ImageInfo]:
        """Get list of discovered images"""
        return self.discovered_images.copy()

    def save_image_list(self, filepath: str) -> None:
        """Save discovered images to JSON file"""
        try:
            img_data = [img.to_dict() for img in self.discovered_images]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(img_data, f, ensure_ascii=False, indent=2)
            print(f"📄 Saved image list to {filepath}")
        except Exception as e:
            logger.error(f"Error saving image list: {e}")

    def load_image_list(self, filepath: str) -> None:
        """Load images from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                img_data = json.load(f)
            
            self.discovered_images = []
            for data in img_data:
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
            
            print(f"📄 Loaded {len(self.discovered_images)} images from {filepath}")
        except Exception as e:
            logger.error(f"Error loading image list: {e}")


if __name__ == "__main__":
    # Example usage
    async def main():
        downloader = ImageWebDownloader()
        
        # Scan for images
        #url = "https://github.com/ALTIBASE/Documents/tree/master/Manuals/Altibase_7.3/eng"
        url = "http://211.177.94.209:8080/moment/images/"
        images = await downloader.scan_images(url)
        
        if images:
            print(downloader.summarize_findings(images))
            
            # Download first 10 images
            downloaded = await downloader.download_all_images(max_files=10)
            print(f"Downloaded {len(downloaded)} images")
        
    asyncio.run(main())