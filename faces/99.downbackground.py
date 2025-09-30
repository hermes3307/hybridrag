#!/usr/bin/env python3
"""
Background Face Download System
Downloads faces in background with duplicate checking, configuration support, and terminal display
"""

import requests
import os
import time
import hashlib
import signal
import sys
import json
import threading
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import argparse
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DownloadConfig:
    """Configuration for download system"""
    faces_dir: str = "./faces"
    delay: float = 1.0
    max_workers: int = 3
    check_duplicates: bool = True
    unlimited_download: bool = True
    download_limit: Optional[int] = None

    @classmethod
    def from_file(cls, config_path: str = "download_config.json"):
        """Load configuration from file"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}")
        return cls()

    def save_to_file(self, config_path: str = "download_config.json"):
        """Save configuration to file"""
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

class DownloadStats:
    """Statistics tracking for downloads"""
    def __init__(self):
        self.total_attempts = 0
        self.successful_downloads = 0
        self.duplicates_skipped = 0
        self.errors = 0
        self.start_time = datetime.now()
        self.lock = threading.Lock()

    def increment_attempts(self):
        with self.lock:
            self.total_attempts += 1

    def increment_success(self):
        with self.lock:
            self.successful_downloads += 1

    def increment_duplicates(self):
        with self.lock:
            self.duplicates_skipped += 1

    def increment_errors(self):
        with self.lock:
            self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            return {
                'total_attempts': self.total_attempts,
                'successful_downloads': self.successful_downloads,
                'duplicates_skipped': self.duplicates_skipped,
                'errors': self.errors,
                'elapsed_time': elapsed,
                'download_rate': self.successful_downloads / elapsed if elapsed > 0 else 0
            }

class BackgroundDownloader:
    """Background face downloader with duplicate checking and abort capability"""

    def __init__(self, config: DownloadConfig):
        self.config = config
        self.stats = DownloadStats()
        self.running = True
        self.downloaded_hashes: Set[str] = set()
        self.lock = threading.Lock()

        # Create faces directory
        os.makedirs(config.faces_dir, exist_ok=True)

        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Load existing hashes if duplicate checking is enabled
        if config.check_duplicates:
            self._load_existing_hashes()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    def _load_existing_hashes(self):
        """Load hashes of existing face images to avoid duplicates"""
        logger.info("Loading existing face hashes for duplicate checking...")

        if not os.path.exists(self.config.faces_dir):
            return

        for filename in os.listdir(self.config.faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(self.config.faces_dir, filename)
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        self.downloaded_hashes.add(file_hash)
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")

        logger.info(f"Loaded {len(self.downloaded_hashes)} existing face hashes")

    def _is_duplicate(self, image_data: bytes) -> bool:
        """Check if image is a duplicate"""
        if not self.config.check_duplicates:
            return False

        image_hash = hashlib.md5(image_data).hexdigest()

        with self.lock:
            if image_hash in self.downloaded_hashes:
                return True
            self.downloaded_hashes.add(image_hash)
            return False

    def download_single_face(self, face_id: str) -> Optional[str]:
        """Download a single face image"""
        # Check running flag at start
        if not self.running:
            return None

        try:
            self.stats.increment_attempts()

            # Check running flag before download
            if not self.running:
                return None

            # Download from ThisPersonDoesNotExist
            url = "https://thispersondoesnotexist.com/"
            response = self.session.get(url, timeout=10)  # Reduced timeout
            response.raise_for_status()

            # Check running flag after download
            if not self.running:
                return None

            # Check for duplicates
            if self._is_duplicate(response.content):
                self.stats.increment_duplicates()
                logger.info(f"Face {face_id}: Duplicate detected, skipping")
                return None

            # Check running flag before saving
            if not self.running:
                return None

            # Save image
            download_time = datetime.now()
            image_hash = hashlib.md5(response.content).hexdigest()
            filename = f"face_{face_id}_{image_hash[:8]}.jpg"
            file_path = os.path.join(self.config.faces_dir, filename)

            # Prepare metadata filename
            json_filename = f"face_{face_id}_{image_hash[:8]}.json"
            json_path = os.path.join(self.config.faces_dir, json_filename)

            # Save image file
            with open(file_path, 'wb') as f:
                f.write(response.content)

            # Extract image properties and save metadata
            try:
                with Image.open(file_path) as img:
                    image_width, image_height = img.size
                    image_format = img.format
                    image_mode = img.mode

                # Collect metadata
                metadata = {
                    'filename': filename,
                    'file_path': file_path,
                    'face_id': face_id,
                    'file_size_bytes': len(response.content),
                    'file_size_kb': round(len(response.content) / 1024, 2),
                    'md5_hash': image_hash,
                    'download_timestamp': download_time.isoformat(),
                    'download_date': download_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'source_url': url,
                    'http_status_code': response.status_code,
                    'http_headers': dict(response.headers),
                    'image_properties': {
                        'width': image_width,
                        'height': image_height,
                        'format': image_format,
                        'mode': image_mode,
                        'dimensions': f"{image_width}x{image_height}"
                    },
                    'downloader_config': {
                        'faces_dir': self.config.faces_dir,
                        'delay': self.config.delay,
                        'max_workers': self.config.max_workers,
                        'check_duplicates': self.config.check_duplicates
                    }
                }

                # Save metadata as JSON
                with open(json_path, 'w') as json_file:
                    json.dump(metadata, json_file, indent=2)

                logger.debug(f"Face {face_id}: Saved metadata to {json_filename}")

            except Exception as meta_error:
                logger.warning(f"Face {face_id}: Failed to save metadata - {meta_error}")

            self.stats.increment_success()
            logger.info(f"Face {face_id}: Downloaded successfully -> {filename}")

            # Check running flag before delay
            if not self.running:
                return file_path

            # Respect delay with frequent checks
            if self.config.delay > 0:
                delay_steps = max(1, int(self.config.delay * 10))  # Check 10 times per second
                step_delay = self.config.delay / delay_steps
                for _ in range(delay_steps):
                    if not self.running:
                        break
                    time.sleep(step_delay)

            return file_path

        except Exception as e:
            self.stats.increment_errors()
            logger.error(f"Face {face_id}: Download failed - {e}")
            return None

    def start_background_download(self):
        """Start unlimited background downloading"""
        logger.info("Starting background face download...")
        logger.info(f"Configuration: {self.config.__dict__}")

        # Start stats display thread
        stats_thread = threading.Thread(target=self._display_stats_loop, daemon=True)
        stats_thread.start()

        face_counter = 1

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}

            while self.running:
                # Check if we've reached download limit
                if (not self.config.unlimited_download and
                    self.config.download_limit and
                    self.stats.successful_downloads >= self.config.download_limit):
                    logger.info(f"Download limit of {self.config.download_limit} reached")
                    break

                # Submit new download if we have capacity
                if len(futures) < self.config.max_workers:
                    face_id = f"{face_counter:06d}"
                    future = executor.submit(self.download_single_face, face_id)
                    futures[future] = face_id
                    face_counter += 1

                # Check completed downloads
                completed = []
                for future in list(futures.keys()):
                    if future.done():
                        completed.append(future)

                for future in completed:
                    try:
                        result = future.result()
                        del futures[future]
                    except Exception as e:
                        logger.error(f"Future execution error: {e}")
                        del futures[future]

                # Small delay to prevent busy waiting
                time.sleep(0.1)

            # Cancel remaining downloads immediately when stopped
            if not self.running:
                logger.info("Cancelling remaining downloads...")
                for future in futures:
                    try:
                        future.cancel()  # Try to cancel if not started
                        if not future.cancelled():
                            # If can't cancel, wait a short time
                            future.result(timeout=3)
                    except Exception as e:
                        logger.warning(f"Error cancelling download: {e}")
            else:
                # Wait for remaining downloads to complete normally
                logger.info("Waiting for remaining downloads to complete...")
                for future in futures:
                    try:
                        future.result(timeout=30)
                    except Exception as e:
                        logger.error(f"Error completing download: {e}")

        # Final stats display
        self._display_final_stats()

    def _display_stats_loop(self):
        """Display download statistics in a loop"""
        while self.running:
            time.sleep(5)  # Update every 5 seconds
            self._display_current_stats()

    def _display_current_stats(self):
        """Display current download statistics"""
        stats = self.stats.get_stats()

        print("\n" + "="*60)
        print("📊 DOWNLOAD STATISTICS")
        print("="*60)
        print(f"🎯 Total Attempts:      {stats['total_attempts']}")
        print(f"✅ Successful Downloads: {stats['successful_downloads']}")
        print(f"🔄 Duplicates Skipped:   {stats['duplicates_skipped']}")
        print(f"❌ Errors:              {stats['errors']}")
        print(f"⏱️  Elapsed Time:        {stats['elapsed_time']:.1f} seconds")
        print(f"📈 Download Rate:       {stats['download_rate']:.2f} faces/second")
        print(f"📁 Storage Directory:   {self.config.faces_dir}")
        print("="*60)
        print("Press Ctrl+C to stop downloading")
        print("="*60)

    def _display_final_stats(self):
        """Display final download statistics"""
        stats = self.stats.get_stats()

        print("\n" + "="*60)
        print("🏁 FINAL DOWNLOAD SUMMARY")
        print("="*60)
        print(f"🎯 Total Attempts:      {stats['total_attempts']}")
        print(f"✅ Successful Downloads: {stats['successful_downloads']}")
        print(f"🔄 Duplicates Skipped:   {stats['duplicates_skipped']}")
        print(f"❌ Errors:              {stats['errors']}")
        print(f"⏱️  Total Time:          {stats['elapsed_time']:.1f} seconds")
        print(f"📈 Average Rate:        {stats['download_rate']:.2f} faces/second")
        print(f"📁 Storage Directory:   {self.config.faces_dir}")
        print("="*60)

        if stats['successful_downloads'] > 0:
            print(f"🎉 Successfully downloaded {stats['successful_downloads']} unique faces!")
        else:
            print("⚠️  No faces were downloaded successfully.")

def create_default_config():
    """Create default configuration file"""
    config = DownloadConfig()
    config.save_to_file()
    print(f"Created default configuration file: download_config.json")
    print(f"Configuration: {config.__dict__}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Background Face Download System')
    parser.add_argument('--config', type=str, default='download_config.json',
                       help='Configuration file path (default: download_config.json)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file and exit')
    parser.add_argument('--faces-dir', type=str,
                       help='Override faces directory from config')
    parser.add_argument('--delay', type=float,
                       help='Override delay between downloads (seconds)')
    parser.add_argument('--workers', type=int,
                       help='Override number of worker threads')
    parser.add_argument('--no-duplicates', action='store_true',
                       help='Disable duplicate checking')
    parser.add_argument('--limit', type=int,
                       help='Set download limit (overrides unlimited mode)')

    args = parser.parse_args()

    if args.create_config:
        create_default_config()
        return

    # Load configuration
    config = DownloadConfig.from_file(args.config)

    # Apply command line overrides
    if args.faces_dir:
        config.faces_dir = args.faces_dir
    if args.delay is not None:
        config.delay = args.delay
    if args.workers:
        config.max_workers = args.workers
    if args.no_duplicates:
        config.check_duplicates = False
    if args.limit:
        config.unlimited_download = False
        config.download_limit = args.limit

    print("🎭 Background Face Download System")
    print("="*50)

    # Create and start downloader
    downloader = BackgroundDownloader(config)

    try:
        downloader.start_background_download()
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Download system error: {e}")

    print("\n👋 Download system stopped")

if __name__ == "__main__":
    main()